#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""Generate noised auxiliary activations and train/validation manifests."""

import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# USER CONFIGURATION: CHANGE VALUES ONLY HERE
INPUT_JSONL = "data/attacker/pile_attacker_train.jsonl"
OUTPUT_ROOT = Path("results/attacker_noised_data")
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CAPTURE_LAYER = 0
CAPTURE_POINT = "lm_head_input_after_final_norm"

# Supported: qkv_input_after_input_layernorm, o_proj_input,
# gate_up_input_after_post_attention_layernorm, lm_head_input_after_final_norm
EPS_MODE = "custom"          # "ep1", "sqrt", or "custom"
CUSTOM_EPSILON = 1.0         # used only when EPS_MODE == "custom"
NOISE_REPLICAS = 1           # independent noised copies per auxiliary prompt
NOISE_SEED = 1234
DATA_SPLIT_SEED = 42
VALID_FRACTION = 0.10
MIN_TOKENS = 40
MAX_TOKENS = 256
LIMIT = None                 # e.g. 20 for smoke test; None for all rows
OVERWRITE = False
SAVE_NOISE_COMPONENTS = False  # debug only; normally keep False
MODEL_DTYPE = torch.float32
SAVE_DTYPE = torch.float32
EXPECTED_HIDDEN_SIZE = 3072
SPECTRAL_NORM_METHOD = "power_iter"  # "power_iter" or "exact"
POWER_ITER_STEPS = 30
LOG_EVERY = 25

# CONSTANTS
CONDITION = "adaptive_noised_t3"
MASKED = True
SUPPORTED_CAPTURE_POINTS = {
    "qkv_input_after_input_layernorm",
    "o_proj_input",
    "gate_up_input_after_post_attention_layernorm",
    "lm_head_input_after_final_norm",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = clean_text(obj.get("text", ""))
            if text:
                rows.append({
                    "source_line": line_number,
                    "text": text,
                    "declared_prompt_token_length": obj.get("prompt_token_length"),
                    "source": obj.get("source"),
                })
    return rows


def append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(obj, ensure_ascii=False) + "\n")
        file.flush()


def render_chat(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def stable_base_id(index: int, text: str) -> str:
    digest = hashlib.sha256(f"{index}\0{text}".encode("utf-8")).hexdigest()
    return f"pile_{index:06d}_{digest[:16]}"


def adaptive_sample_id(base_id: str, replica_idx: int) -> str:
    return f"{base_id}_noise{replica_idx:03d}"


def epsilon_from_seq_len(seq_len: int) -> float:
    if EPS_MODE == "ep1":
        return 1.0
    if EPS_MODE == "custom":
        return float(CUSTOM_EPSILON)
    if EPS_MODE == "sqrt":
        return 1.0 / math.sqrt(float(seq_len))
    raise ValueError("EPS_MODE must be 'ep1', 'sqrt', or 'custom'.")


def epsilon_tag() -> str:
    if EPS_MODE == "custom":
        return f"CUSTOM{str(CUSTOM_EPSILON).replace('.', 'p')}"
    return EPS_MODE.upper()


def capture_point_short_name(capture_point: str) -> str:
    return {
        "qkv_input_after_input_layernorm": "qkv",
        "o_proj_input": "o",
        "gate_up_input_after_post_attention_layernorm": "gate_up",
        "lm_head_input_after_final_norm": "lm_head",
    }[capture_point]


def output_directory() -> Path:
    short = capture_point_short_name(CAPTURE_POINT)
    if CAPTURE_POINT == "lm_head_input_after_final_norm":
        point_dir = "Final_LMHead"
    else:
        point_dir = f"Layer{CAPTURE_LAYER}_{short}"
    return OUTPUT_ROOT / point_dir / f"adaptive_gaussian_{epsilon_tag()}"

# EXACT TDX PRECOMPUTATION MATH


def l4_norm(x: torch.Tensor) -> float:
    return float(torch.sum(torch.abs(x) ** 4).pow(0.25).item())


def spectral_norm_exact(weight: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(weight, ord=2).item())


def spectral_norm_power_iter(weight: torch.Tensor, steps: int = 30) -> float:
    v = torch.randn(weight.shape[1], dtype=weight.dtype, device=weight.device)
    v = v / (v.norm() + 1e-12)
    for _ in range(steps):
        u = weight @ v
        u = u / (u.norm() + 1e-12)
        v = weight.T @ u
        v = v / (v.norm() + 1e-12)
    return float((weight @ v).norm().item())


def spectral_norm(weight: torch.Tensor) -> float:
    if SPECTRAL_NORM_METHOD == "exact":
        return spectral_norm_exact(weight)
    if SPECTRAL_NORM_METHOD == "power_iter":
        return spectral_norm_power_iter(weight, POWER_ITER_STEPS)
    raise ValueError("SPECTRAL_NORM_METHOD must be 'exact' or 'power_iter'.")


def layernorm_sensitivity(alpha: torch.Tensor, hidden_size: int):
    alpha_l4 = l4_norm(alpha)
    delta = 2.0 * math.sqrt(float(hidden_size)) * alpha_l4
    return delta, alpha_l4


def o_input_sensitivity(alpha: torch.Tensor, w_val: torch.Tensor, hidden_size: int):
    alpha_l4 = l4_norm(alpha)
    wval_norm = spectral_norm(w_val)
    delta = 2.0 * math.sqrt(float(hidden_size)) * wval_norm * alpha_l4
    return delta, alpha_l4, wval_norm


def gaussian_sigma(sensitivity: float, epsilon: float) -> float:
    # Exact formula used by the supplied TDX precomputation code.
    return float(sensitivity) / float(epsilon)


def resolve_capture_and_sensitivity(model: Any):
    if CAPTURE_POINT not in SUPPORTED_CAPTURE_POINTS:
        raise ValueError(f"Unsupported CAPTURE_POINT: {CAPTURE_POINT}")
    hidden_size = int(model.config.hidden_size)
    layers = model.model.layers
    if CAPTURE_POINT != "lm_head_input_after_final_norm":
        if not 0 <= CAPTURE_LAYER < len(layers):
            raise ValueError(
                f"CAPTURE_LAYER must be in [0, {len(layers)-1}], got {CAPTURE_LAYER}"
            )
        layer = layers[CAPTURE_LAYER]
    if CAPTURE_POINT == "qkv_input_after_input_layernorm":
        alpha = layer.input_layernorm.weight.detach().float()
        delta, alpha_l4 = layernorm_sensitivity(alpha, hidden_size)
        return (
            layer.self_attn.q_proj,
            ["q_proj", "k_proj", "v_proj"],
            CAPTURE_LAYER,
            f"layer {CAPTURE_LAYER} shared QKV input after input_layernorm",
            delta,
            "Delta_QKV = 2*sqrt(d_h)*||alpha_input||_4",
            {"alpha_l4": alpha_l4},
        )
    if CAPTURE_POINT == "o_proj_input":
        alpha = layer.input_layernorm.weight.detach().float()
        v_w = layer.self_attn.v_proj.weight.detach().float()
        delta, alpha_l4, wval_norm = o_input_sensitivity(alpha, v_w, hidden_size)
        return (
            layer.self_attn.o_proj,
            ["o_proj"],
            CAPTURE_LAYER,
            f"layer {CAPTURE_LAYER} o_proj input",
            delta,
            "Delta_O = 2*sqrt(d_h)*||W_val||_2*||alpha_input||_4",
            {
                "alpha_l4": alpha_l4,
                "wval_spectral_norm": wval_norm,
                "wval_source": "v_proj.weight",
            },
        )
    if CAPTURE_POINT == "gate_up_input_after_post_attention_layernorm":
        alpha = layer.post_attention_layernorm.weight.detach().float()
        delta, alpha_l4 = layernorm_sensitivity(alpha, hidden_size)
        return (
            layer.mlp.gate_proj,
            ["gate_proj", "up_proj"],
            CAPTURE_LAYER,
            f"layer {CAPTURE_LAYER} shared Gate/Up input after post_attention_layernorm",
            delta,
            "Delta_GATE_UP = 2*sqrt(d_h)*||alpha_post||_4",
            {"alpha_l4": alpha_l4},
        )
    alpha = model.model.norm.weight.detach().float()
    delta, alpha_l4 = layernorm_sensitivity(alpha, hidden_size)
    return (
        model.lm_head,
        ["lm_head"],
        None,
        "final lm_head input after final model norm",
        delta,
        "Delta_LM = 2*sqrt(d_h)*||alpha_final||_4",
        {"alpha_l4": alpha_l4},
    )


class OperatorInputCapture:
    def __init__(self, target_module, description: str):
        self.tensor: Optional[torch.Tensor] = None
        self.call_count = 0
        self.description = description
        self.handle = target_module.register_forward_pre_hook(self._pre_hook)
    def _pre_hook(self, module, inputs):
        del module
        self.call_count += 1
        if not inputs:
            raise RuntimeError(f"{self.description} pre-hook received no inputs")
        hidden = inputs[0]
        while isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        if not torch.is_tensor(hidden):
            raise RuntimeError(f"{self.description} input is not a tensor")
        self.tensor = hidden.detach().to(dtype=SAVE_DTYPE).contiguous().clone()
    def reset(self):
        self.tensor = None
        self.call_count = 0
    def close(self):
        self.handle.remove()


def validate_existing_sample(
    sample_path: Path,
    sample_id: str,
    expected_layer,
    expected_epsilon: float,
    expected_sensitivity: float,
) -> dict:
    obj = torch.load(sample_path, map_location="cpu")
    hidden = obj.get("hidden")
    meta = obj.get("meta", {})
    if obj.get("sample_id") != sample_id:
        raise RuntimeError(f"Wrong sample_id in {sample_path}")
    if not torch.is_tensor(hidden) or hidden.ndim != 2:
        raise RuntimeError(f"Invalid hidden tensor in {sample_path}")
    if int(hidden.shape[-1]) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(f"Wrong hidden size in {sample_path}")
    if not torch.isfinite(hidden).all().item():
        raise RuntimeError(f"Non-finite hidden values in {sample_path}")
    expected = {
        "capture_point": CAPTURE_POINT,
        "layer_idx": expected_layer,
        "eps_mode": EPS_MODE,
        "epsilon": expected_epsilon,
        "sensitivity": expected_sensitivity,
        "noise_distribution": "gaussian",
        "sigma_rule": "sensitivity / epsilon",
        "adaptive_attacker": True,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"Existing sample mismatch for {key}: {meta.get(key)!r} != {value!r}"
            )
    return obj


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU-side adaptive generation.")
    if EPS_MODE == "custom" and CUSTOM_EPSILON <= 0:
        raise ValueError("CUSTOM_EPSILON must be greater than zero.")
    if NOISE_REPLICAS < 1:
        raise ValueError("NOISE_REPLICAS must be at least one.")
    set_seed(DATA_SPLIT_SEED)
    output_dir = output_directory()
    samples_dir = output_dir / "samples"
    train_manifest = output_dir / "train_manifest.jsonl"
    valid_manifest = output_dir / "valid_manifest.jsonl"
    failed_rows = output_dir / "failed_rows.jsonl"
    dataset_meta = output_dir / "dataset_meta.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(INPUT_JSONL)
    if LIMIT is not None:
        rows = rows[:max(0, int(LIMIT))]
    if not rows:
        raise RuntimeError("No valid input rows were found.")
    print("=" * 88)
    print("ADAPTIVE TDX-MATCHED NOISED AUXILIARY GENERATOR")
    print("=" * 88)
    print(f"Input JSONL          : {INPUT_JSONL}")
    print(f"Output directory     : {output_dir}")
    print(f"Model                : {MODEL_NAME}")
    print(f"Capture layer        : {CAPTURE_LAYER}")
    print(f"Capture point        : {CAPTURE_POINT}")
    print(f"EPS mode             : {EPS_MODE}")
    print(f"Custom epsilon       : {CUSTOM_EPSILON if EPS_MODE == 'custom' else 'unused'}")
    print(f"Noise replicas       : {NOISE_REPLICAS}")
    print(f"Auxiliary prompts    : {len(rows)}")
    print("Noise                : Gaussian, sigma = sensitivity / epsilon")
    print("Separate precompute  : not required")
    print("=" * 88)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Loading open-weight model on GPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=MODEL_DTYPE,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    (
        target_module,
        op_names,
        effective_layer,
        capture_description,
        sensitivity,
        sensitivity_rule,
        sensitivity_extra,
    ) = resolve_capture_and_sensitivity(model)
    print(f"Capture description  : {capture_description}")
    print(f"Operator names       : {op_names}")
    print(f"Sensitivity          : {sensitivity:.8f}")
    print(f"Sensitivity rule     : {sensitivity_rule}")
    print("=" * 88)
    capture = OperatorInputCapture(target_module, capture_description)
    noise_generator = torch.Generator(device="cuda")
    noise_generator.manual_seed(NOISE_SEED)
    successful_records = []
    failed_count = 0
    if failed_rows.exists():
        failed_rows.unlink()
    try:
        for index, row in enumerate(tqdm(rows, desc="Generating adaptive samples")):
            text = row["text"]
            base_id = stable_base_id(index, text)
            try:
                rendered = render_chat(tokenizer, text)
                encoded = tokenizer(
                    rendered,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=False,
                )
                token_length = int(encoded["input_ids"].shape[1])
                if not MIN_TOKENS <= token_length <= MAX_TOKENS:
                    raise ValueError(
                        f"Token length {token_length} outside [{MIN_TOKENS}, {MAX_TOKENS}]"
                    )
                declared = row["declared_prompt_token_length"]
                if declared is not None and int(declared) != token_length:
                    raise ValueError(
                        f"Token-length mismatch: declared={declared}, actual={token_length}"
                    )
                encoded = {k: v.to("cuda") for k, v in encoded.items()}
                capture.reset()
                with torch.inference_mode():
                    _ = model(
                        **encoded,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                if capture.call_count != 1 or capture.tensor is None:
                    raise RuntimeError(f"Expected one capture, got {capture.call_count}")
                clean_hidden = capture.tensor
                if clean_hidden.ndim != 3 or clean_hidden.shape[0] != 1:
                    raise RuntimeError(
                        f"Expected captured [1,T,D], got {tuple(clean_hidden.shape)}"
                    )
                clean_hidden = clean_hidden[0].contiguous()
                if tuple(clean_hidden.shape) != (token_length, EXPECTED_HIDDEN_SIZE):
                    raise RuntimeError(f"Unexpected capture shape {tuple(clean_hidden.shape)}")
                if not torch.isfinite(clean_hidden).all().item():
                    raise RuntimeError("Clean activation contains non-finite values")
                epsilon = epsilon_from_seq_len(token_length)
                sigma = gaussian_sigma(sensitivity, epsilon)
                for replica_idx in range(NOISE_REPLICAS):
                    sample_id = adaptive_sample_id(base_id, replica_idx)
                    sample_path = samples_dir / f"{sample_id}.pt"
                    if sample_path.exists() and not OVERWRITE:
                        existing = validate_existing_sample(
                            sample_path,
                            sample_id,
                            effective_layer,
                            epsilon,
                            sensitivity,
                        )
                        hidden = existing["hidden"]
                    else:
                        sampled_noise = torch.randn(
                            clean_hidden.shape,
                            dtype=clean_hidden.dtype,
                            device=clean_hidden.device,
                            generator=noise_generator,
                        ) * sigma
                        noised_hidden = (clean_hidden + sampled_noise).to(
                            device="cpu", dtype=SAVE_DTYPE
                        ).contiguous()
                        payload = {
                            "sample_id": sample_id,
                            "hidden": noised_hidden,
                            "text": text,
                            "meta": {
                                "model_name": MODEL_NAME,
                                "condition": CONDITION,
                                "masked": MASKED,
                                "adaptive_attacker": True,
                                "generated_locally": True,
                                "capture_tensor": "operator_input",
                                "capture_point": CAPTURE_POINT,
                                "capture_description": capture_description,
                                "layer_idx": effective_layer,
                                "op_names": op_names,
                                "phase": "prefill",
                                "prompt_token_length": token_length,
                                "hidden_size": int(noised_hidden.shape[1]),
                                "stored_shape": tuple(int(x) for x in noised_hidden.shape),
                                "dtype": str(noised_hidden.dtype),
                                "noise_distribution": "gaussian",
                                "eps_mode": EPS_MODE,
                                "custom_epsilon": (
                                    CUSTOM_EPSILON if EPS_MODE == "custom" else None
                                ),
                                "epsilon": epsilon,
                                "sensitivity": sensitivity,
                                "sensitivity_rule": sensitivity_rule,
                                "sigma": sigma,
                                "sigma_rule": "sensitivity / epsilon",
                                "noise_replica": replica_idx,
                                "noise_seed": NOISE_SEED,
                                "source_line": row["source_line"],
                                "source": row["source"],
                                "created_unix_time": time.time(),
                                **sensitivity_extra,
                            },
                        }
                        if SAVE_NOISE_COMPONENTS:
                            payload["clean_hidden"] = clean_hidden.to(
                                device="cpu", dtype=SAVE_DTYPE
                            ).contiguous()
                            payload["sampled_noise"] = sampled_noise.to(
                                device="cpu", dtype=SAVE_DTYPE
                            ).contiguous()
                        torch.save(payload, sample_path)
                        hidden = noised_hidden
                    successful_records.append({
                        "sample_id": sample_id,
                        "base_sample_id": base_id,
                        "sample_path": str(sample_path),
                        "text": text,
                        "prompt_token_length": token_length,
                        "hidden_size": int(hidden.shape[1]),
                        "epsilon": epsilon,
                        "sigma": sigma,
                        "noise_replica": replica_idx,
                        "source_line": row["source_line"],
                        "source": row["source"],
                    })
                if len(successful_records) == NOISE_REPLICAS or (
                    len(successful_records) % (LOG_EVERY * NOISE_REPLICAS) == 0
                ):
                    print(
                        f"\nSaved {len(successful_records)} samples | "
                        f"base={base_id} | shape={tuple(clean_hidden.shape)} | "
                        f"epsilon={epsilon:.8f} | sigma={sigma:.8f}"
                    )
                del encoded, clean_hidden
            except Exception as error:
                failed_count += 1
                append_jsonl(failed_rows, {
                    "index": index,
                    "source_line": row["source_line"],
                    "base_sample_id": base_id,
                    "capture_point": CAPTURE_POINT,
                    "layer_idx": effective_layer,
                    "error": repr(error),
                })
                print(f"\n[FAILED] index={index} base={base_id}: {error}", flush=True)
            if (index + 1) % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        capture.close()
    if len(successful_records) < 2:
        raise RuntimeError("Fewer than two adaptive samples were generated successfully.")
    # Split by original prompt, not by noise replica, preventing train/valid leakage.
    records_by_base = {}
    for record in successful_records:
        records_by_base.setdefault(record["base_sample_id"], []).append(record)
    base_ids = list(records_by_base)
    rng = random.Random(DATA_SPLIT_SEED)
    rng.shuffle(base_ids)
    valid_base_count = max(1, int(round(len(base_ids) * VALID_FRACTION)))
    valid_base_ids = set(base_ids[:valid_base_count])
    train_records = []
    valid_records = []
    for base_id, records in records_by_base.items():
        if base_id in valid_base_ids:
            valid_records.extend(records)
        else:
            train_records.extend(records)
    if not train_records:
        raise RuntimeError("Training split is empty.")
    with open(train_manifest, "w", encoding="utf-8") as file:
        for record in train_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(valid_manifest, "w", encoding="utf-8") as file:
        for record in valid_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta = {
        "model_name": MODEL_NAME,
        "condition": CONDITION,
        "adaptive_attacker": True,
        "separate_noise_precomputation_required": False,
        "input_jsonl": os.path.abspath(INPUT_JSONL),
        "output_directory": str(output_dir),
        "samples_directory": str(samples_dir),
        "train_manifest": str(train_manifest),
        "valid_manifest": str(valid_manifest),
        "capture_layer_requested": CAPTURE_LAYER,
        "capture_layer": effective_layer,
        "capture_point": CAPTURE_POINT,
        "capture_description": capture_description,
        "op_names": op_names,
        "phase": "prefill",
        "noise_distribution": "gaussian",
        "eps_mode": EPS_MODE,
        "custom_epsilon": CUSTOM_EPSILON if EPS_MODE == "custom" else None,
        "epsilon_rule": (
            "epsilon = 1/sqrt(sequence_length)" if EPS_MODE == "sqrt"
            else "epsilon = CUSTOM_EPSILON" if EPS_MODE == "custom"
            else "epsilon = 1"
        ),
        "sensitivity": sensitivity,
        "sensitivity_rule": sensitivity_rule,
        "sigma_rule": "sigma = sensitivity / epsilon",
        "sensitivity_details": sensitivity_extra,
        "spectral_norm_method": (
            SPECTRAL_NORM_METHOD if CAPTURE_POINT == "o_proj_input" else None
        ),
        "power_iter_steps": (
            POWER_ITER_STEPS if CAPTURE_POINT == "o_proj_input" else None
        ),
        "noise_seed": NOISE_SEED,
        "noise_replicas": NOISE_REPLICAS,
        "save_noise_components": SAVE_NOISE_COMPONENTS,
        "model_dtype": str(MODEL_DTYPE),
        "save_dtype": str(SAVE_DTYPE),
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "min_tokens": MIN_TOKENS,
        "max_tokens": MAX_TOKENS,
        "validation_fraction": VALID_FRACTION,
        "num_input_rows": len(rows),
        "num_successful_samples": len(successful_records),
        "num_failed_prompts": failed_count,
        "num_train_samples": len(train_records),
        "num_valid_samples": len(valid_records),
        "created_unix_time": time.time(),
    }
    with open(dataset_meta, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print("=" * 88)
    print("ADAPTIVE DATASET GENERATION COMPLETE")
    print("=" * 88)
    print(f"Capture point      : {CAPTURE_POINT}")
    print(f"Effective layer    : {effective_layer}")
    print(f"Sensitivity        : {sensitivity:.8f}")
    print(f"Successful samples : {len(successful_records)}")
    print(f"Failed prompts     : {failed_count}")
    print(f"Training samples   : {len(train_records)}")
    print(f"Validation samples : {len(valid_records)}")
    print(f"Samples directory  : {samples_dir}")
    print(f"Train manifest     : {train_manifest}")
    print(f"Valid manifest     : {valid_manifest}")
    print(f"Dataset metadata   : {dataset_meta}")
    print(f"Failure log        : {failed_rows}")
    print("=" * 88)


if __name__ == "__main__":
    main()
