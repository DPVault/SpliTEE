#!/usr/bin/env python3
"""Precompute chunked noise and correction banks for noised T3."""
import os
import math
import gc
import sys
import subprocess
from collections import defaultdict
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
# CONFIG
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
#MODEL_NAME = "ministral/Ministral-3b-instruct"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
#MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = "databricks/databricks-dolly-15k"
SPLIT = "train"
EPS_MODE = "custom"        # "ep1", "sqrt", or "custom"
CUSTOM_EPSILON = 1.0
BANK_DIR = "results/noised_banks"
BANK_NORMAL_DTYPE_NAME = "float32"
MIN_TOKENS = 100
MAX_TOKENS = 500
TOKEN_RANGES = [
    (100, 102),
]
NUM_PROMPTS_PER_RANGE = 30
MAX_NEW_TOKENS = 100
BATCH_SIZE = 15
# Timed-bank provisioning
# Provision the number of independent full-prefill masks/corrections required
# by the current timed generate() path. Decode remains MAX_NEW_TOKENS + DECODE_SLACK.
TIMED_PREFILL_COUNT = 1
DECODE_SLACK = 2
# Separate warmup bank, matching the Slalom storage structure.
PRECOMPUTE_WARMUP_BANK = True
WARMUP_BATCH_SIZE = 1
WARMUP_CAP = 2
ATTN_IMPL = "sdpa"
CPU_DEVICE = torch.device("cpu")
GLOBAL_SEED = 1234
SPECTRAL_NORM_METHOD = "power_iter"
POWER_ITER_STEPS = 30
SKIP_EXISTING = False
# DTYPE / PATH HELPERS
def dtype_from_name(name):
    name = name.lower().strip()
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    if name == "float16":
        return torch.float16
    raise ValueError("Use float16, float32, or float64.")
DTYPE_NORMAL = dtype_from_name(BANK_NORMAL_DTYPE_NAME)
def epsilon_tag():
    if EPS_MODE == "custom":
        return f"CUSTOM{str(CUSTOM_EPSILON).replace('.', 'p')}"
    return EPS_MODE.upper()
def bank_name():
    return (
        f"{epsilon_tag()}_Llama3B_NO_DOWN_"
        f"normal{BANK_NORMAL_DTYPE_NAME}_"
        f"qkv_o_gateup_lm_BATCH.pt"
    )
ROOT_SPLIT_BANK_DIR = os.path.join(
    BANK_DIR,
    f"{epsilon_tag()}(b{BATCH_SIZE})_{MAX_NEW_TOKENS}_chunked",
)
CHUNK_BATCH_INDEX_ENV = os.environ.get("T3_BANK_CHUNK_INDEX")
CHUNK_BATCH_INDEX = (
    None
    if CHUNK_BATCH_INDEX_ENV is None
    else int(CHUNK_BATCH_INDEX_ENV)
)
if CHUNK_BATCH_INDEX is None:
    SPLIT_BANK_DIR = ROOT_SPLIT_BANK_DIR
else:
    SPLIT_BANK_DIR = os.path.join(
        ROOT_SPLIT_BANK_DIR,
        f"batch_{CHUNK_BATCH_INDEX:03d}",
    )
META_PATH = os.path.join(SPLIT_BANK_DIR, "meta.pt")
ROOT_META_PATH = os.path.join(ROOT_SPLIT_BANK_DIR, "meta.pt")
def split_file(name):
    return os.path.join(SPLIT_BANK_DIR, name)
def get_epsilon(seq_len):
    if EPS_MODE == "ep1":
        return 1.0
    if EPS_MODE == "custom":
        return float(CUSTOM_EPSILON)
    if EPS_MODE == "sqrt":
        return 1.0 / math.sqrt(float(seq_len))
    raise ValueError("EPS_MODE must be ep1, sqrt, or custom.")
def gaussian_sigma(delta, epsilon):
    return float(delta) / float(epsilon)
def epsilon_values_for_prompt_lengths(prompt_lengths):
    """Compute one epsilon per original prompt and reuse it through decode."""
    return [get_epsilon(int(n)) for n in prompt_lengths]
def sigma_values_for_prompt_lengths(delta, prompt_lengths):
    eps_values = epsilon_values_for_prompt_lengths(prompt_lengths)
    sigma_values = [gaussian_sigma(delta, eps) for eps in eps_values]
    return eps_values, sigma_values
def _sigma_scale_for_shape(sigma, shape, dtype, device):
    """Return a scalar or a [B,1,1,...] per-row scale for batched Gaussian noise."""
    if isinstance(sigma, (list, tuple)):
        if len(sigma) != int(shape[0]):
            raise ValueError(
                f"Per-row sigma count {len(sigma)} does not match batch "
                f"dimension {shape[0]} for shape {shape}."
            )
        return torch.tensor(sigma, dtype=dtype, device=device).view(
            int(shape[0]), *([1] * (len(shape) - 1))
        )
    if torch.is_tensor(sigma):
        sigma_t = sigma.to(device=device, dtype=dtype)
        if sigma_t.numel() == 1:
            return sigma_t
        if sigma_t.numel() != int(shape[0]):
            raise ValueError(
                f"Per-row sigma count {sigma_t.numel()} does not match batch "
                f"dimension {shape[0]} for shape {shape}."
            )
        return sigma_t.reshape(int(shape[0]), *([1] * (len(shape) - 1)))
    return float(sigma)
def store_tensor(x, dtype):
    x = x.detach().to(CPU_DEVICE, dtype=dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    return x
torch.set_grad_enabled(False)
# MODEL / SENSITIVITY HELPERS
def get_transformer_layers(model):
    return model.model.layers
def get_hidden_size(model):
    return int(model.config.hidden_size)
def get_intermediate_size(model, first_layer):
    return int(first_layer.mlp.gate_proj.out_features)
def l4_norm(x):
    x = x.detach().to(CPU_DEVICE, dtype=torch.float64)
    return float(torch.sum(torch.abs(x) ** 4).pow(0.25).item())
def layernorm_sensitivity(alpha, hidden_size):
    alpha_l4 = l4_norm(alpha)
    delta = 2.0 * math.sqrt(float(hidden_size)) * alpha_l4
    return float(delta), float(alpha_l4)
@torch.inference_mode()
def spectral_norm_power_iter(weight, steps=30):
    """

    Approximate ||W||_2.

    weight shape: [out_features, in_features]

    """
    w = weight.detach().to(CPU_DEVICE, dtype=torch.float32)
    in_dim = int(w.shape[1])
    v = torch.randn(in_dim, dtype=torch.float32, device=CPU_DEVICE)
    v = v / torch.clamp(torch.linalg.vector_norm(v), min=1e-12)
    for _ in range(int(steps)):
        u = torch.mv(w, v)
        u = u / torch.clamp(torch.linalg.vector_norm(u), min=1e-12)
        v = torch.mv(w.t(), u)
        v = v / torch.clamp(torch.linalg.vector_norm(v), min=1e-12)
    wv = torch.mv(w, v)
    sigma = torch.linalg.vector_norm(wv)
    return float(sigma.item())
@torch.inference_mode()
def spectral_norm_exact(weight):
    w = weight.detach().to(CPU_DEVICE, dtype=torch.float32)
    return float(torch.linalg.matrix_norm(w, ord=2).item())
def spectral_norm(weight):
    if SPECTRAL_NORM_METHOD == "power_iter":
        return spectral_norm_power_iter(weight, POWER_ITER_STEPS)
    if SPECTRAL_NORM_METHOD == "exact":
        return spectral_norm_exact(weight)
    raise ValueError("SPECTRAL_NORM_METHOD must be power_iter or exact.")
# DATA HELPERS
def build_dolly_prompt(row):
    instruction = (row.get("instruction") or "").strip()
    context = (row.get("context") or "").strip()
    if context:
        return f"{instruction}\n\nContext:\n{context}"
    return instruction
def tokenize_as_chat(tok, text):
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
def render_chat(tok, text):
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
def tokenize_as_chat_batch(tok, texts):
    rendered = [render_chat(tok, text) for text in texts]
    return tok(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
def make_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
def collect_texts_by_ranges(ds, tok):
    buckets = {r: [] for r in TOKEN_RANGES}
    for row in ds:
        text = build_dolly_prompt(row)
        enc = tokenize_as_chat(tok, text)
        seq_len = int(enc["input_ids"].shape[1])
        if seq_len < MIN_TOKENS or seq_len > MAX_TOKENS:
            continue
        for r in TOKEN_RANGES:
            lo, hi = r
            if lo <= seq_len <= hi:
                if len(buckets[r]) < NUM_PROMPTS_PER_RANGE:
                    buckets[r].append((text, seq_len))
                break
        if all(len(v) >= NUM_PROMPTS_PER_RANGE for v in buckets.values()):
            break
    selected = []
    for r in TOKEN_RANGES:
        selected.extend(buckets[r])
    return selected, buckets
def compute_shape_to_count(
    selected,
    tok,
    input_dim,
):

    input_dim = int(input_dim)
    shape_to_count = defaultdict(int)
    batches = list(
        make_batches(
            selected,
            BATCH_SIZE,
        )
    )
    for batch in batches:
        texts = [item[0] for item in batch]
        enc = tokenize_as_chat_batch(
            tok,
            texts,
        )
        bsz = int(enc["input_ids"].shape[0])
        padded_len = int(enc["input_ids"].shape[1])
        prefill_shape = (
            bsz,
            padded_len,
            input_dim,
        )
        decode_shape = (
            bsz,
            1,
            input_dim,
        )
        shape_to_count[prefill_shape] += int(TIMED_PREFILL_COUNT)
        shape_to_count[decode_shape] += (
            int(MAX_NEW_TOKENS)
            + int(DECODE_SLACK)
        )
    return shape_to_count
def make_dp_meta(
    epsilon,
    seq_len,
    delta,
    sigma,
    count,
    sensitivity_type,
    extra=None,
):
    meta = {
        "epsilon": float(
            epsilon
        ),
        "prompt_tokens": int(
            seq_len
        ),
        "sensitivity": float(
            delta
        ),
        "sigma": float(
            sigma
        ),
        "count": int(
            count
        ),
        "sensitivity_type": (
            sensitivity_type
        ),
    }
    if extra:
        meta.update(extra)
    return meta
# BANK GENERATION
@torch.inference_mode()
def generate_qkv_items(
    shape,
    count,
    sigma,
    q_w,
    k_w,
    v_w,
):
    items = []
    for _ in range(
        int(count)
    ):
        eta = (
            torch.randn(
                shape,
                dtype=DTYPE_NORMAL,
                device=CPU_DEVICE,
            )
            * _sigma_scale_for_shape(sigma, shape, DTYPE_NORMAL, CPU_DEVICE)
        )
        corr_q = F.linear(
            eta,
            q_w,
        )
        corr_k = F.linear(
            eta,
            k_w,
        )
        corr_v = F.linear(
            eta,
            v_w,
        )
        items.append(
            (
                store_tensor(
                    eta,
                    DTYPE_NORMAL,
                ),
                [
                    store_tensor(
                        corr_q,
                        DTYPE_NORMAL,
                    ),
                    store_tensor(
                        corr_k,
                        DTYPE_NORMAL,
                    ),
                    store_tensor(
                        corr_v,
                        DTYPE_NORMAL,
                    ),
                ],
            )
        )
    return items
@torch.inference_mode()
def generate_single_items(
    shape,
    count,
    sigma,
    w,
):
    items = []
    for _ in range(
        int(count)
    ):
        eta = (
            torch.randn(
                shape,
                dtype=DTYPE_NORMAL,
                device=CPU_DEVICE,
            )
            * _sigma_scale_for_shape(sigma, shape, DTYPE_NORMAL, CPU_DEVICE)
        )
        corr = F.linear(
            eta,
            w,
        )
        items.append(
            (
                store_tensor(
                    eta,
                    DTYPE_NORMAL,
                ),
                [
                    store_tensor(
                        corr,
                        DTYPE_NORMAL,
                    )
                ],
            )
        )
    return items
@torch.inference_mode()
def generate_gateup_items(
    shape,
    count,
    sigma,
    gate_w,
    up_w,
):
    items = []
    for _ in range(
        int(count)
    ):
        eta = (
            torch.randn(
                shape,
                dtype=DTYPE_NORMAL,
                device=CPU_DEVICE,
            )
            * _sigma_scale_for_shape(sigma, shape, DTYPE_NORMAL, CPU_DEVICE)
        )
        corr_gate = F.linear(
            eta,
            gate_w,
        )
        corr_up = F.linear(
            eta,
            up_w,
        )
        items.append(
            (
                store_tensor(
                    eta,
                    DTYPE_NORMAL,
                ),
                [
                    store_tensor(
                        corr_gate,
                        DTYPE_NORMAL,
                    ),
                    store_tensor(
                        corr_up,
                        DTYPE_NORMAL,
                    ),
                ],
            )
        )
    return items
# SEPARATE WARMUP BANK
@torch.inference_mode()
def precompute_separate_warmup_bank(model, tok, warmup_text):
    """

    Create a separate B=1 warmup bank.

    Shapes are derived from each projection's true input width. This is

    required for architectures such as Qwen3 where o_proj.in_features can be

    different from model.config.hidden_size.

    """
    if not PRECOMPUTE_WARMUP_BANK:
        return None
    warmup_dir = os.path.join(
        ROOT_SPLIT_BANK_DIR,
        "warmup",
    )
    os.makedirs(warmup_dir, exist_ok=True)
    enc = tokenize_as_chat_batch(
        tok,
        [warmup_text],
    )
    hidden_size = get_hidden_size(model)
    padded_len = int(enc["input_ids"].shape[1])
    warmup_prompt_len = int(tokenize_as_chat(tok, warmup_text)["input_ids"].shape[1])
    prefill_count = 1
    decode_count = int(WARMUP_CAP) + int(DECODE_SLACK)
    layers = get_transformer_layers(model)
    warmup_layer_paths = []
    for layer_idx, layer in enumerate(layers):
        q_in = int(layer.self_attn.q_proj.in_features)
        k_in = int(layer.self_attn.k_proj.in_features)
        v_in = int(layer.self_attn.v_proj.in_features)
        o_in = int(layer.self_attn.o_proj.in_features)
        gate_in = int(layer.mlp.gate_proj.in_features)
        up_in = int(layer.mlp.up_proj.in_features)
        if not (q_in == k_in == v_in):
            raise ValueError(
                f"Layer {layer_idx}: Q/K/V do not share the same input width: "
                f"q={q_in}, k={k_in}, v={v_in}"
            )
        if gate_in != up_in:
            raise ValueError(
                f"Layer {layer_idx}: Gate/Up do not share the same input width: "
                f"gate={gate_in}, up={up_in}"
            )
        qkv_prefill_shape = (
            int(WARMUP_BATCH_SIZE),
            padded_len,
            q_in,
        )
        qkv_decode_shape = (
            int(WARMUP_BATCH_SIZE),
            1,
            q_in,
        )
        o_prefill_shape = (
            int(WARMUP_BATCH_SIZE),
            padded_len,
            o_in,
        )
        o_decode_shape = (
            int(WARMUP_BATCH_SIZE),
            1,
            o_in,
        )
        gateup_prefill_shape = (
            int(WARMUP_BATCH_SIZE),
            padded_len,
            gate_in,
        )
        gateup_decode_shape = (
            int(WARMUP_BATCH_SIZE),
            1,
            gate_in,
        )
        input_alpha = (
            layer.input_layernorm.weight
            .detach()
            .to(CPU_DEVICE, dtype=DTYPE_NORMAL)
        )
        post_alpha = (
            layer.post_attention_layernorm.weight
            .detach()
            .to(CPU_DEVICE, dtype=DTYPE_NORMAL)
        )
        q_w = layer.self_attn.q_proj.weight.detach().to(CPU_DEVICE, dtype=DTYPE_NORMAL)
        k_w = layer.self_attn.k_proj.weight.detach().to(CPU_DEVICE, dtype=DTYPE_NORMAL)
        v_w = layer.self_attn.v_proj.weight.detach().to(CPU_DEVICE, dtype=DTYPE_NORMAL)
        o_w = layer.self_attn.o_proj.weight.detach().to(CPU_DEVICE, dtype=DTYPE_NORMAL)
        gate_w = layer.mlp.gate_proj.weight.detach().to(CPU_DEVICE, dtype=DTYPE_NORMAL)
        up_w = layer.mlp.up_proj.weight.detach().to(CPU_DEVICE, dtype=DTYPE_NORMAL)
        delta_qkv, alpha_input_l4 = layernorm_sensitivity(
            input_alpha,
            hidden_size,
        )
        delta_gate_up, _ = layernorm_sensitivity(
            post_alpha,
            hidden_size,
        )
        v_norm = spectral_norm(v_w)
        delta_o = (
            2.0
            * math.sqrt(float(hidden_size))
            * float(v_norm)
            * float(alpha_input_l4)
        )
        # Compute epsilon once from the original warmup prompt length and
        # keep that same value for prefill and every decode step.
        eps_values, sigma_qkv_values = sigma_values_for_prompt_lengths(
            delta_qkv, [warmup_prompt_len]
        )
        _, sigma_o_values = sigma_values_for_prompt_lengths(
            delta_o, [warmup_prompt_len]
        )
        _, sigma_gate_up_values = sigma_values_for_prompt_lengths(
            delta_gate_up, [warmup_prompt_len]
        )
        qkv_bank = {
            qkv_prefill_shape: {
                "items": generate_qkv_items(
                    qkv_prefill_shape,
                    prefill_count,
                    sigma_qkv_values,
                    q_w,
                    k_w,
                    v_w,
                )
            },
            qkv_decode_shape: {
                "items": generate_qkv_items(
                    qkv_decode_shape,
                    decode_count,
                    sigma_qkv_values,
                    q_w,
                    k_w,
                    v_w,
                )
            },
        }
        o_bank = {
            o_prefill_shape: {
                "items": generate_single_items(
                    o_prefill_shape,
                    prefill_count,
                    sigma_o_values,
                    o_w,
                )
            },
            o_decode_shape: {
                "items": generate_single_items(
                    o_decode_shape,
                    decode_count,
                    sigma_o_values,
                    o_w,
                )
            },
        }
        gateup_bank = {
            gateup_prefill_shape: {
                "items": generate_gateup_items(
                    gateup_prefill_shape,
                    prefill_count,
                    sigma_gate_up_values,
                    gate_w,
                    up_w,
                )
            },
            gateup_decode_shape: {
                "items": generate_gateup_items(
                    gateup_decode_shape,
                    decode_count,
                    sigma_gate_up_values,
                    gate_w,
                    up_w,
                )
            },
        }
        qkv_path = os.path.join(warmup_dir, f"layer_{layer_idx:02d}_qkv.pt")
        o_path = os.path.join(warmup_dir, f"layer_{layer_idx:02d}_o.pt")
        gateup_path = os.path.join(warmup_dir, f"layer_{layer_idx:02d}_gateup.pt")
        torch.save(qkv_bank, qkv_path)
        torch.save(o_bank, o_path)
        torch.save(gateup_bank, gateup_path)
        warmup_layer_paths.append({
            "layer_idx": int(layer_idx),
            "qkv_bank_path": qkv_path,
            "o_bank_path": o_path,
            "gate_up_bank_path": gateup_path,
        })
        if layer_idx == 0:
            print(
                "Warmup layer-0 input widths: "
                f"QKV={q_in}, O={o_in}, GATE/UP={gate_in}",
                flush=True,
            )
        del qkv_bank, o_bank, gateup_bank
        gc.collect()
    final_alpha = (
        model.model.norm.weight
        .detach()
        .to(CPU_DEVICE, dtype=DTYPE_NORMAL)
    )
    lm_w = (
        model.lm_head.weight
        .detach()
        .to(CPU_DEVICE, dtype=DTYPE_NORMAL)
    )
    lm_in = int(model.lm_head.in_features)
    delta_lm, _ = layernorm_sensitivity(
        final_alpha,
        hidden_size,
    )
    _, sigma_lm_values = sigma_values_for_prompt_lengths(
        delta_lm, [warmup_prompt_len]
    )
    lm_prefill_shape = (
        int(WARMUP_BATCH_SIZE),
        padded_len,
        lm_in,
    )
    lm_decode_shape = (
        int(WARMUP_BATCH_SIZE),
        1,
        lm_in,
    )
    lm_head_bank = {
        lm_prefill_shape: {
            "items": generate_single_items(
                lm_prefill_shape,
                prefill_count,
                sigma_lm_values,
                lm_w,
            )
        },
        lm_decode_shape: {
            "items": generate_single_items(
                lm_decode_shape,
                decode_count,
                sigma_lm_values,
                lm_w,
            )
        },
    }
    lm_head_path = os.path.join(
        warmup_dir,
        "lm_head.pt",
    )
    torch.save(
        lm_head_bank,
        lm_head_path,
    )
    warmup_meta = {
        "bank_type": "SEPARATE_WARMUP_B1_NO_DOWN",
        "batch_size": int(WARMUP_BATCH_SIZE),
        "warmup_cap": int(WARMUP_CAP),
        "decode_slack": int(DECODE_SLACK),
        "prefill_count": int(prefill_count),
        "decode_count": int(decode_count),
        "qkv_prefill_shape": tuple(qkv_prefill_shape),
        "qkv_decode_shape": tuple(qkv_decode_shape),
        "o_prefill_shape": tuple(o_prefill_shape),
        "o_decode_shape": tuple(o_decode_shape),
        "gateup_prefill_shape": tuple(gateup_prefill_shape),
        "gateup_decode_shape": tuple(gateup_decode_shape),
        "lm_prefill_shape": tuple(lm_prefill_shape),
        "lm_decode_shape": tuple(lm_decode_shape),
        "layer_bank_paths": warmup_layer_paths,
        "lm_head_bank_path": lm_head_path,
    }
    warmup_meta_path = os.path.join(
        warmup_dir,
        "meta.pt",
    )
    torch.save(
        warmup_meta,
        warmup_meta_path,
    )
    print(
        f"Separate warmup bank saved: {warmup_meta_path}",
        flush=True,
    )
    return warmup_meta_path
# MAIN PRECOMPUTE
def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(
        GLOBAL_SEED
    )
    # PARENT MODE
    if CHUNK_BATCH_INDEX is None:
        total_requested_prompts = (
            len(TOKEN_RANGES)
            * NUM_PROMPTS_PER_RANGE
        )
        num_batches = int(
            math.ceil(
                total_requested_prompts
                / float(BATCH_SIZE)
            )
        )
        os.makedirs(
            ROOT_SPLIT_BANK_DIR,
            exist_ok=True,
        )
        print(
            "=" * 80,
            flush=True,
        )
        print(
            "CHUNKED BANK PARENT",
            flush=True,
        )
        print(
            f"ROOT_SPLIT_BANK_DIR : "
            f"{ROOT_SPLIT_BANK_DIR}",
            flush=True,
        )
        print(
            f"TOTAL PROMPTS       : "
            f"{total_requested_prompts}",
            flush=True,
        )
        print(
            f"BATCH SIZE          : "
            f"{BATCH_SIZE}",
            flush=True,
        )
        print(
            f"NUM BATCH CHUNKS    : "
            f"{num_batches}",
            flush=True,
        )
        print(
            "Warmup bank         : separate B=1 bank",
            flush=True,
        )
        print(
            f"Timed prefills/chunk : {TIMED_PREFILL_COUNT}",
            flush=True,
        )
        print(
            f"Timed decode/chunk   : {MAX_NEW_TOKENS + DECODE_SLACK}",
            flush=True,
        )
        print(
            "=" * 80,
            flush=True,
        )
        # Separate B=1 warmup bank for apples-to-apples storage
        # comparison with Slalom.
        if PRECOMPUTE_WARMUP_BANK:
            print(
                "Preparing separate B=1 warmup bank...",
                flush=True,
            )
            warm_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            if warm_tok.pad_token_id is None:
                warm_tok.pad_token = warm_tok.eos_token
            warm_tok.padding_side = "left"
            warm_model = (
                AutoModelForCausalLM
                .from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float32,
                    attn_implementation=ATTN_IMPL,
                )
                .to(CPU_DEVICE)
                .eval()
            )
            warm_ds = load_dataset(DATASET_NAME)[SPLIT]
            warm_selected, _ = collect_texts_by_ranges(
                warm_ds,
                warm_tok,
            )
            if not warm_selected:
                raise RuntimeError(
                    "No prompt available for separate warmup bank."
                )
            precompute_separate_warmup_bank(
                warm_model,
                warm_tok,
                warm_selected[0][0],
            )
            del warm_model
            gc.collect()
        batch_meta_paths = []
        for batch_idx in range(
            num_batches
        ):
            env = os.environ.copy()
            env[
                "T3_BANK_CHUNK_INDEX"
            ] = str(batch_idx)
            print(
                f"Launching bank chunk "
                f"{batch_idx + 1}/"
                f"{num_batches}...",
                flush=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(
                        __file__
                    ),
                ],
                env=env,
                check=True,
            )
            batch_meta_path = (
                os.path.join(
                    ROOT_SPLIT_BANK_DIR,
                    f"batch_{batch_idx:03d}",
                    "meta.pt",
                )
            )
            if not os.path.exists(
                batch_meta_path
            ):
                raise FileNotFoundError(
                    "Child did not create "
                    f"{batch_meta_path}"
                )
            batch_meta_paths.append(
                batch_meta_path
            )
            gc.collect()
        first_payload = torch.load(
            batch_meta_paths[0],
            map_location="cpu",
        )
        root_payload = {
            "meta": {
                **first_payload["meta"],
                "bank_type": (
                    "CHUNKED_PER_PROMPT_BATCH_"
                    "NO_DOWN"
                ),
                "num_batch_chunks": int(
                    num_batches
                ),
                "num_prompts_total": int(
                    total_requested_prompts
                ),
                "batch_size": int(
                    BATCH_SIZE
                ),
                "max_new_tokens": int(
                    MAX_NEW_TOKENS
                ),            },
            "root_split_bank_dir": (
                ROOT_SPLIT_BANK_DIR
            ),
            "batch_meta_paths": (
                batch_meta_paths
            ),
        }
        torch.save(
            root_payload,
            ROOT_META_PATH,
        )
        print(
            "=" * 80,
            flush=True,
        )
        print(
            "ALL CHUNKS COMPLETE",
            flush=True,
        )
        print(
            f"Root meta: "
            f"{ROOT_META_PATH}",
            flush=True,
        )
        for path in batch_meta_paths:
            print(
                f"  {path}",
                flush=True,
            )
        print(
            "=" * 80,
            flush=True,
        )
        return
    # CHILD / ONE-CHUNK MODE
    torch.manual_seed(
        GLOBAL_SEED
        + int(
            CHUNK_BATCH_INDEX
        )
        * 1_000_003
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        "PRECOMPUTE CHUNKED NOISE BANK "
        "FOR T3 IVSHMEM NO-DOWN",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        f"MODEL_NAME            : "
        f"{MODEL_NAME}",
        flush=True,
    )
    print(
        f"DATASET_NAME          : "
        f"{DATASET_NAME}",
        flush=True,
    )
    print(
        f"EPS_MODE              : "
        f"{EPS_MODE}",
        flush=True,
    )
    print(
        f"CUSTOM_EPSILON        : "
        f"{CUSTOM_EPSILON}",
        flush=True,
    )
    print(
        f"BANK_DIR              : "
        f"{BANK_DIR}",
        flush=True,
    )
    print(
        f"SPLIT_BANK_DIR        : "
        f"{SPLIT_BANK_DIR}",
        flush=True,
    )
    print(
        f"META_PATH             : "
        f"{META_PATH}",
        flush=True,
    )
    print(
        f"BANK_NORMAL_DTYPE     : "
        f"{BANK_NORMAL_DTYPE_NAME}",
        flush=True,
    )
    print(
        f"BATCH_SIZE            : "
        f"{BATCH_SIZE}",
        flush=True,
    )
    print(
        f"MAX_NEW_TOKENS        : "
        f"{MAX_NEW_TOKENS}",
        flush=True,
    )
    print(
        f"TIMED_PREFILL_COUNT   : "
        f"{TIMED_PREFILL_COUNT}",
        flush=True,
    )
    print(
        f"DECODE_SLACK          : "
        f"{DECODE_SLACK}",
        flush=True,
    )
    print(
        f"TOKEN_RANGES          : "
        f"{TOKEN_RANGES}",
        flush=True,
    )
    print(
        f"NUM_PROMPTS_PER_RANGE : "
        f"{NUM_PROMPTS_PER_RANGE}",
        flush=True,
    )
    print(
        "Stored                : "
        "QKV, O, GATE/UP, LM_HEAD",
        flush=True,
    )
    print(
        "Not stored            : "
        "DOWN eta/correction",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    if (
        SKIP_EXISTING
        and os.path.exists(
            META_PATH
        )
    ):
        print(
            f"Existing bank found, "
            f"skipping: {META_PATH}",
            flush=True,
        )
        return
    os.makedirs(
        SPLIT_BANK_DIR,
        exist_ok=True,
    )
    # LOAD TOKENIZER
    print(
        "Loading tokenizer...",
        flush=True,
    )
    tok = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )
    if tok.pad_token_id is None:
        tok.pad_token = (
            tok.eos_token
        )
    tok.padding_side = "left"
    # LOAD MODEL
    print(
        "Loading model on CPU...",
        flush=True,
    )
    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            attn_implementation=(
                ATTN_IMPL
            ),
        )
        .to(CPU_DEVICE)
        .eval()
    )
    layers = (
        get_transformer_layers(
            model
        )
    )
    hidden_size = (
        get_hidden_size(
            model
        )
    )
    intermediate_size = (
        get_intermediate_size(
            model,
            layers[0],
        )
    )
    print(
        f"Hidden size       : "
        f"{hidden_size}",
        flush=True,
    )
    print(
        f"Intermediate size : "
        f"{intermediate_size}",
        flush=True,
    )
    print(
        f"Layers            : "
        f"{len(layers)}",
        flush=True,
    )
    print(
        "Layer-0 projection dimensions:",
        flush=True,
    )
    print(
        f"  QKV input       : {int(layers[0].self_attn.q_proj.in_features)}",
        flush=True,
    )
    print(
        f"  O input         : {int(layers[0].self_attn.o_proj.in_features)}",
        flush=True,
    )
    print(
        f"  GATE/UP input   : {int(layers[0].mlp.gate_proj.in_features)}",
        flush=True,
    )
    print(
        f"  LM_HEAD input   : {int(model.lm_head.in_features)}",
        flush=True,
    )
    # DATASET
    print(
        "Loading dataset...",
        flush=True,
    )
    ds = load_dataset(
        DATASET_NAME
    )[SPLIT]
    print(
        "Selecting prompts using the "
        "same rule as runtime...",
        flush=True,
    )
    (
        selected,
        buckets,
    ) = collect_texts_by_ranges(
        ds,
        tok,
    )
    if len(selected) == 0:
        raise RuntimeError(
            "No prompts found in selected "
            "token ranges."
        )
    print(
        "Selected prompt counts by range:",
        flush=True,
    )
    for token_range in TOKEN_RANGES:
        print(
            f"  {token_range}: "
            f"{len(buckets[token_range])}",
            flush=True,
        )
    print(
        f"Total selected prompts "
        f"before chunking: "
        f"{len(selected)}",
        flush=True,
    )
    chunk_start = (
        CHUNK_BATCH_INDEX
        * BATCH_SIZE
    )
    chunk_end = min(
        chunk_start
        + BATCH_SIZE,
        len(selected),
    )
    selected = selected[
        chunk_start:chunk_end
    ]
    if len(selected) == 0:
        raise RuntimeError(
            f"No prompts for chunk index "
            f"{CHUNK_BATCH_INDEX}; "
            f"slice="
            f"{chunk_start}:"
            f"{chunk_end}"
        )
    print(
        f"Chunk "
        f"{CHUNK_BATCH_INDEX:03d} "
        f"uses selected prompts "
        f"{chunk_start}:"
        f"{chunk_end}",
        flush=True,
    )
    print(
        f"Prompts in this chunk: "
        f"{len(selected)}",
        flush=True,
    )
    for i, (
        text,
        seq_len,
    ) in enumerate(
        selected,
        start=chunk_start,
    ):
        short = (
            text
            .replace(
                "\n",
                " ",
            )[:120]
        )
        print(
            f"  prompt[{i}] "
            f"tokens={seq_len} | "
            f"{short}",
            flush=True,
        )
    # Compute one epsilon per original prompt. For sqrt mode this is
    # epsilon_prompt = 1/sqrt(original_prompt_len), and the same epsilon is
    # reused for both prefill and all decode steps of that prompt.
    chunk_prompt_lengths = [int(item[1]) for item in selected]
    chunk_epsilon_values = epsilon_values_for_prompt_lengths(chunk_prompt_lengths)
    print("Prompt-derived epsilon values for this chunk:", flush=True)
    for row_idx, (plen, eps) in enumerate(zip(chunk_prompt_lengths, chunk_epsilon_values)):
        print(f"  row={row_idx:02d} prompt_tokens={plen} epsilon={eps:.8f}", flush=True)
    # OPERATOR-SPECIFIC BANK SHAPE COUNTS
    first_layer = layers[0]
    qkv_input_dim = int(first_layer.self_attn.q_proj.in_features)
    o_input_dim = int(first_layer.self_attn.o_proj.in_features)
    gateup_input_dim = int(first_layer.mlp.gate_proj.in_features)
    lm_input_dim = int(model.lm_head.in_features)
    qkv_shape_to_count = compute_shape_to_count(
        selected=selected,
        tok=tok,
        input_dim=qkv_input_dim,
    )
    o_shape_to_count = compute_shape_to_count(
        selected=selected,
        tok=tok,
        input_dim=o_input_dim,
    )
    gateup_shape_to_count = compute_shape_to_count(
        selected=selected,
        tok=tok,
        input_dim=gateup_input_dim,
    )
    lm_shape_to_count = compute_shape_to_count(
        selected=selected,
        tok=tok,
        input_dim=lm_input_dim,
    )
    print(
        "\nOperator input widths:",
        flush=True,
    )
    print(f"  QKV     : {qkv_input_dim}", flush=True)
    print(f"  O       : {o_input_dim}", flush=True)
    print(f"  GATE/UP : {gateup_input_dim}", flush=True)
    print(f"  LM_HEAD : {lm_input_dim}", flush=True)
    print("\nQKV shapes to precompute:", flush=True)
    for shape, count in sorted(qkv_shape_to_count.items(), key=lambda kv: kv[0]):
        print(f"  {shape}: {count}", flush=True)
    print("\nO shapes to precompute:", flush=True)
    for shape, count in sorted(o_shape_to_count.items(), key=lambda kv: kv[0]):
        print(f"  {shape}: {count}", flush=True)
    print("\nGATE/UP shapes to precompute:", flush=True)
    for shape, count in sorted(gateup_shape_to_count.items(), key=lambda kv: kv[0]):
        print(f"  {shape}: {count}", flush=True)
    print("\nLM_HEAD shapes to precompute:", flush=True)
    for shape, count in sorted(lm_shape_to_count.items(), key=lambda kv: kv[0]):
        print(f"  {shape}: {count}", flush=True)
    # LAYER BANKS
    layer_bank_paths = []
    for (
        layer_idx,
        layer,
    ) in enumerate(
        layers
    ):
        print(
            "\n"
            + "-" * 80,
            flush=True,
        )
        print(
            f"LAYER {layer_idx}",
            flush=True,
        )
        print(
            "-" * 80,
            flush=True,
        )
        input_alpha = (
            layer
            .input_layernorm
            .weight
            .detach()
            .to(
                CPU_DEVICE,
                dtype=DTYPE_NORMAL,
            )
        )
        post_alpha = (
            layer
            .post_attention_layernorm
            .weight
            .detach()
            .to(
                CPU_DEVICE,
                dtype=DTYPE_NORMAL,
            )
        )
        q_w = (
            layer
            .self_attn
            .q_proj
            .weight
            .detach()
            .to(
                CPU_DEVICE,
                dtype=DTYPE_NORMAL,
            )
        )
        k_w = (
            layer
            .self_attn
            .k_proj
            .weight
            .detach()
            .to(
                CPU_DEVICE,
                dtype=DTYPE_NORMAL,
            )
        )
        v_w = (
            layer
            .self_attn
            .v_proj
            .weight
            .detach()
            .to(
                CPU_DEVICE,
                dtype=DTYPE_NORMAL,
            )
        )
        o_w = (
            layer
            .self_attn
            .o_proj
            .weight
            .detach()
            .to(
                CPU_DEVICE,
                dtype=DTYPE_NORMAL,
            )
        )
        gate_w = (
            layer
            .mlp
            .gate_proj
            .weight
            .detach()
            .to(
                CPU_DEVICE,
                dtype=DTYPE_NORMAL,
            )
        )
        up_w = (
            layer
            .mlp
            .up_proj
            .weight
            .detach()
            .to(
                CPU_DEVICE,
                dtype=DTYPE_NORMAL,
            )
        )
        (
            delta_qkv,
            alpha_input_l4,
        ) = layernorm_sensitivity(
            input_alpha,
            hidden_size,
        )
        (
            delta_gate_up,
            alpha_post_l4,
        ) = layernorm_sensitivity(
            post_alpha,
            hidden_size,
        )
        print(
            "Computing "
            "||v_proj.weight||_2 "
            "for O sensitivity...",
            flush=True,
        )
        v_norm = spectral_norm(
            v_w
        )
        delta_o = (
            2.0
            * math.sqrt(
                float(hidden_size)
            )
            * float(v_norm)
            * float(
                alpha_input_l4
            )
        )
        print(
            f"  Δ_qkv     = "
            f"{delta_qkv:.6f}",
            flush=True,
        )
        print(
            f"  Δ_o       = "
            f"{delta_o:.6f}",
            flush=True,
        )
        print(
            f"  Δ_gate_up = "
            f"{delta_gate_up:.6f}",
            flush=True,
        )
        print(
            f"  ||v_proj||_2 = "
            f"{v_norm:.6f}",
            flush=True,
        )
        qkv_bank = {}
        o_bank = {}
        gateup_bank = {}
        q_in = int(layer.self_attn.q_proj.in_features)
        k_in = int(layer.self_attn.k_proj.in_features)
        v_in = int(layer.self_attn.v_proj.in_features)
        o_in = int(layer.self_attn.o_proj.in_features)
        gate_in = int(layer.mlp.gate_proj.in_features)
        up_in = int(layer.mlp.up_proj.in_features)
        if not (q_in == k_in == v_in):
            raise ValueError(
                f"Layer {layer_idx}: Q/K/V input widths differ: "
                f"q={q_in}, k={k_in}, v={v_in}"
            )
        if gate_in != up_in:
            raise ValueError(
                f"Layer {layer_idx}: Gate/Up input widths differ: "
                f"gate={gate_in}, up={up_in}"
            )
        # Sanity-check that all layers use the same structural widths as
        # layer 0. If a future architecture varies by layer, stop rather than
        # silently generating an incorrect bank.
        if q_in != qkv_input_dim:
            raise ValueError(
                f"Layer {layer_idx}: qkv input width {q_in} != "
                f"reference {qkv_input_dim}"
            )
        if o_in != o_input_dim:
            raise ValueError(
                f"Layer {layer_idx}: O input width {o_in} != "
                f"reference {o_input_dim}"
            )
        if gate_in != gateup_input_dim:
            raise ValueError(
                f"Layer {layer_idx}: Gate/Up input width {gate_in} != "
                f"reference {gateup_input_dim}"
            )
        for shape, bank_size in sorted(
            qkv_shape_to_count.items(),
            key=lambda kv: kv[0],
        ):
            _, seq_len, _ = shape
            epsilon_values, sigma_qkv = sigma_values_for_prompt_lengths(
                delta_qkv, chunk_prompt_lengths
            )
            print(
                f"  QKV shape={shape} | count={bank_size} | "
                f"epsilon_rows={epsilon_values} | sigma_rows={sigma_qkv}",
                flush=True,
            )
            qkv_items = generate_qkv_items(
                shape=shape,
                count=bank_size,
                sigma=sigma_qkv,
                q_w=q_w,
                k_w=k_w,
                v_w=v_w,
            )
            qkv_bank[shape] = {
                "items": qkv_items,
                "dp_meta": make_dp_meta(
                    epsilon=float(sum(epsilon_values) / len(epsilon_values)),
                    seq_len=seq_len,
                    delta=delta_qkv,
                    sigma=float(sum(sigma_qkv) / len(sigma_qkv)),
                    count=bank_size,
                    sensitivity_type=(
                        "Delta_QKV = 2*sqrt(d)*||alpha_input||_4"
                    ),
                    extra={
                        "dtype": str(DTYPE_NORMAL),
                        "alpha_l4": alpha_input_l4,
                        "input_features": q_in,
                        "prompt_lengths_by_row": list(chunk_prompt_lengths),
                        "epsilon_by_row": list(epsilon_values),
                        "sigma_by_row": list(sigma_qkv),
                        "epsilon_carried_through_decode": True,
                        "correction_weight_sources": [
                            "q_proj.weight",
                            "k_proj.weight",
                            "v_proj.weight",
                        ],
                        "stored_eta": True,
                        "stored_correction": True,
                        "batched": True,
                    },
                ),
            }
        for shape, bank_size in sorted(
            o_shape_to_count.items(),
            key=lambda kv: kv[0],
        ):
            _, seq_len, _ = shape
            epsilon_values, sigma_o = sigma_values_for_prompt_lengths(
                delta_o, chunk_prompt_lengths
            )
            print(
                f"  O shape={shape} | count={bank_size} | "
                f"epsilon_rows={epsilon_values} | sigma_rows={sigma_o}",
                flush=True,
            )
            o_items = generate_single_items(
                shape=shape,
                count=bank_size,
                sigma=sigma_o,
                w=o_w,
            )
            o_bank[shape] = {
                "items": o_items,
                "dp_meta": make_dp_meta(
                    epsilon=float(sum(epsilon_values) / len(epsilon_values)),
                    seq_len=seq_len,
                    delta=delta_o,
                    sigma=float(sum(sigma_o) / len(sigma_o)),
                    count=bank_size,
                    sensitivity_type=(
                        "Delta_O = 2*sqrt(d)*||W_val||_2*"
                        "||alpha_input||_4"
                    ),
                    extra={
                        "dtype": str(DTYPE_NORMAL),
                        "alpha_l4": alpha_input_l4,
                        "input_features": o_in,
                        "prompt_lengths_by_row": list(chunk_prompt_lengths),
                        "epsilon_by_row": list(epsilon_values),
                        "sigma_by_row": list(sigma_o),
                        "epsilon_carried_through_decode": True,
                        "wval_source": "v_proj.weight",
                        "wval_spectral_norm": v_norm,
                        "correction_weight_source": "o_proj.weight",
                        "stored_eta": True,
                        "stored_correction": True,
                        "batched": True,
                    },
                ),
            }
        for shape, bank_size in sorted(
            gateup_shape_to_count.items(),
            key=lambda kv: kv[0],
        ):
            _, seq_len, _ = shape
            epsilon_values, sigma_gate_up = sigma_values_for_prompt_lengths(
                delta_gate_up, chunk_prompt_lengths
            )
            print(
                f"  GATE/UP shape={shape} | count={bank_size} | "
                f"epsilon_rows={epsilon_values} | sigma_rows={sigma_gate_up}",
                flush=True,
            )
            gateup_items = generate_gateup_items(
                shape=shape,
                count=bank_size,
                sigma=sigma_gate_up,
                gate_w=gate_w,
                up_w=up_w,
            )
            gateup_bank[shape] = {
                "items": gateup_items,
                "dp_meta": make_dp_meta(
                    epsilon=float(sum(epsilon_values) / len(epsilon_values)),
                    seq_len=seq_len,
                    delta=delta_gate_up,
                    sigma=float(sum(sigma_gate_up) / len(sigma_gate_up)),
                    count=bank_size,
                    sensitivity_type=(
                        "Delta_GATE_UP = 2*sqrt(d)*||alpha_post||_4"
                    ),
                    extra={
                        "dtype": str(DTYPE_NORMAL),
                        "alpha_l4": alpha_post_l4,
                        "input_features": gate_in,
                        "prompt_lengths_by_row": list(chunk_prompt_lengths),
                        "epsilon_by_row": list(epsilon_values),
                        "sigma_by_row": list(sigma_gate_up),
                        "epsilon_carried_through_decode": True,
                        "correction_weight_sources": [
                            "gate_proj.weight",
                            "up_proj.weight",
                        ],
                        "stored_eta": True,
                        "stored_correction": True,
                        "batched": True,
                    },
                ),
            }
        gc.collect()
        qkv_path = split_file(
            f"layer_"
            f"{layer_idx:02d}_"
            f"qkv.pt"
        )
        o_path = split_file(
            f"layer_"
            f"{layer_idx:02d}_"
            f"o.pt"
        )
        gateup_path = split_file(
            f"layer_"
            f"{layer_idx:02d}_"
            f"gateup.pt"
        )
        torch.save(
            qkv_bank,
            qkv_path,
        )
        torch.save(
            o_bank,
            o_path,
        )
        torch.save(
            gateup_bank,
            gateup_path,
        )
        layer_bank_paths.append(
            {
                "layer_idx": int(
                    layer_idx
                ),
                "qkv_bank_path": (
                    qkv_path
                ),
                "o_bank_path": (
                    o_path
                ),
                "gate_up_bank_path": (
                    gateup_path
                ),
                "down_bank_path": None,
            }
        )
        print(
            f"  Saved QKV    : "
            f"{qkv_path}",
            flush=True,
        )
        print(
            f"  Saved O      : "
            f"{o_path}",
            flush=True,
        )
        print(
            f"  Saved GATEUP : "
            f"{gateup_path}",
            flush=True,
        )
        del (
            qkv_bank,
            o_bank,
            gateup_bank,
        )
        gc.collect()
    # LM HEAD
    print(
        "\n"
        + "-" * 80,
        flush=True,
    )
    print(
        "LM_HEAD",
        flush=True,
    )
    print(
        "-" * 80,
        flush=True,
    )
    final_alpha = (
        model
        .model
        .norm
        .weight
        .detach()
        .to(
            CPU_DEVICE,
            dtype=DTYPE_NORMAL,
        )
    )
    lm_w = (
        model
        .lm_head
        .weight
        .detach()
        .to(
            CPU_DEVICE,
            dtype=DTYPE_NORMAL,
        )
    )
    (
        delta_lm,
        final_alpha_l4,
    ) = layernorm_sensitivity(
        final_alpha,
        hidden_size,
    )
    print(
        f"  Δ_lm = "
        f"{delta_lm:.6f}",
        flush=True,
    )
    lm_head_bank = {}
    for (
        shape,
        bank_size,
    ) in sorted(
        lm_shape_to_count.items(),
        key=lambda kv: kv[0],
    ):
        _, seq_len, _ = shape
        epsilon_values, sigma_lm = sigma_values_for_prompt_lengths(
            delta_lm, chunk_prompt_lengths
        )
        print(
            f"  shape={shape} | "
            f"count={bank_size} | "
            f"epsilon_rows={epsilon_values} | "
            f"sigma_rows={sigma_lm}",
            flush=True,
        )
        lm_items = (
            generate_single_items(
                shape=shape,
                count=bank_size,
                sigma=sigma_lm,
                w=lm_w,
            )
        )
        lm_head_bank[
            shape
        ] = {
            "items": lm_items,
            "dp_meta": (
                make_dp_meta(
                    epsilon=float(sum(epsilon_values) / len(epsilon_values)),
                    seq_len=seq_len,
                    delta=delta_lm,
                    sigma=float(sum(sigma_lm) / len(sigma_lm)),
                    count=bank_size,
                    sensitivity_type=(
                        "Delta_LM = "
                        "2*sqrt(d)*"
                        "||alpha_final||_4"
                    ),
                    extra={
                        "dtype": str(
                            DTYPE_NORMAL
                        ),
                        "input_features": int(lm_input_dim),
                        "prompt_lengths_by_row": list(chunk_prompt_lengths),
                        "epsilon_by_row": list(epsilon_values),
                        "sigma_by_row": list(sigma_lm),
                        "epsilon_carried_through_decode": True,
                        "alpha_l4": (
                            final_alpha_l4
                        ),
                        "correction_weight_source": (
                            "lm_head.weight"
                        ),
                        "stored_eta": True,
                        "stored_correction": True,
                        "batched": True,
                    },
                )
            ),
        }
        gc.collect()
    lm_head_path = split_file(
        "lm_head.pt"
    )
    torch.save(
        lm_head_bank,
        lm_head_path,
    )
    print(
        f"Saved LM_HEAD: "
        f"{lm_head_path}",
        flush=True,
    )
    del lm_head_bank
    gc.collect()
    # META
    selected_prompt_lengths = [
        int(item[1])
        for item in selected
    ]
    payload = {
        "meta": {
            "model_name": (
                MODEL_NAME
            ),
            "dataset_name": (
                DATASET_NAME
            ),
            "split": SPLIT,
            "bank_type": (
                "SPLIT_batched_no_down_"
                "qkv_o_gateup_lm_for_"
                "3way_wrapper"
            ),
            "eps_mode": (
                EPS_MODE
            ),
            "custom_epsilon": (
                float(
                    CUSTOM_EPSILON
                )
                if EPS_MODE
                == "custom"
                else None
            ),
            "epsilon_rule": (
                "epsilon = CUSTOM_EPSILON"
                if EPS_MODE
                == "custom"
                else (
                    "epsilon_prompt = "
                    "1/sqrt(original_prompt_len), fixed through decode"
                    if EPS_MODE
                    == "sqrt"
                    else "epsilon = 1"
                )
            ),
            "sqrt_epsilon_fixed_from_prompt_len": True,
            "dtype_normal": str(
                DTYPE_NORMAL
            ),
            "bank_normal_dtype_name": (
                BANK_NORMAL_DTYPE_NAME
            ),
            "store_down_correction": (
                False
            ),
            "store_down_eta": False,
            "down_outsourced": False,
            "store_all_corrections_except_down": (
                True
            ),
            "runtime_recomputes_corrections": (
                False
            ),
            "hidden_size": int(
                hidden_size
            ),
            "intermediate_size": int(
                intermediate_size
            ),
            "min_tokens": int(
                MIN_TOKENS
            ),
            "max_tokens": int(
                MAX_TOKENS
            ),
            "token_ranges": (
                TOKEN_RANGES
            ),
            "num_prompts_per_range": int(
                NUM_PROMPTS_PER_RANGE
            ),
            "num_prompts_selected": int(
                len(selected)
            ),
            "selected_prompt_lengths": (
                selected_prompt_lengths
            ),
            "max_new_tokens": int(
                MAX_NEW_TOKENS
            ),
            "caps": [
                int(
                    MAX_NEW_TOKENS
                )
            ],
            "batch_size": int(
                BATCH_SIZE
            ),
            "timed_prefill_count": int(
                TIMED_PREFILL_COUNT
            ),
            "decode_slack": int(
                DECODE_SLACK
            ),
            "extra_safety_multiplier": None,
            "extra_decode_padding": 0,
            "spectral_norm_method": (
                SPECTRAL_NORM_METHOD
            ),
            "power_iter_steps": int(
                POWER_ITER_STEPS
            ),
            "qkv_sensitivity_rule": (
                "2*sqrt(d)*"
                "||alpha_input||_4"
            ),
            "o_sensitivity_rule": (
                "2*sqrt(d)*"
                "||W_val||_2*"
                "||alpha_input||_4"
            ),
            "o_sensitivity_weight_source": (
                "v_proj.weight"
            ),
            "gate_up_sensitivity_rule": (
                "2*sqrt(d)*"
                "||alpha_post||_4"
            ),
            "down_sensitivity_rule": (
                None
            ),
            "lm_sensitivity_rule": (
                "2*sqrt(d)*"
                "||alpha_final||_4"
            ),
            "global_seed": int(
                GLOBAL_SEED
            ),
            "chunk_batch_index": int(
                CHUNK_BATCH_INDEX
            ),
            "chunk_prompt_start": int(
                chunk_start
            ),
            "chunk_prompt_end": int(
                chunk_end
            ),
            "chunk_num_prompts": int(
                len(selected)
            ),
            "root_split_bank_dir": (
                ROOT_SPLIT_BANK_DIR
            ),
        },
        "operator_input_dims": {
            "qkv": int(qkv_input_dim),
            "o": int(o_input_dim),
            "gate_up": int(gateup_input_dim),
            "lm_head": int(lm_input_dim),
        },
        "qkv_shape_to_count": {
            str(key): int(value)
            for key, value
            in qkv_shape_to_count.items()
        },
        "o_shape_to_count": {
            str(key): int(value)
            for key, value
            in o_shape_to_count.items()
        },
        "gateup_shape_to_count": {
            str(key): int(value)
            for key, value
            in gateup_shape_to_count.items()
        },
        "lm_shape_to_count": {
            str(key): int(value)
            for key, value
            in lm_shape_to_count.items()
        },
        "hidden_shape_to_count": {
            str(key): int(value)
            for key, value
            in qkv_shape_to_count.items()
        },
        "intermediate_shape_to_count": {},
        "split_bank_dir": (
            SPLIT_BANK_DIR
        ),
        "layer_bank_paths": (
            layer_bank_paths
        ),
        "lm_head_bank_path": (
            lm_head_path
        ),
    }
    torch.save(
        payload,
        META_PATH,
    )
    print(
        "\n"
        + "=" * 80,
        flush=True,
    )
    print(
        f"PRECOMPUTE CHUNK "
        f"{CHUNK_BATCH_INDEX:03d} "
        f"DONE",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        f"Saved meta: "
        f"{META_PATH}",
        flush=True,
    )
    print(
        f"Saved split bank dir: "
        f"{SPLIT_BANK_DIR}",
        flush=True,
    )
    print(
        "DOWN eta/correction were NOT "
        "generated or stored.",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
if __name__ == "__main__":
    main()
