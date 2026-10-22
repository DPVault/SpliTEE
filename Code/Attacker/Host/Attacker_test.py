#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate prompt reconstruction using held-out TDX captures."""
import argparse
import csv
import glob
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
BEST_CHECKPOINT_DIR = (
    "results/attacker_training/best"
)
CAPTURE_DIR = "data/attacker/test_set"
LABELS_JSONL = (
    "data/attacker/test_set/LMHEAD_t3_prompt_labels.jsonl"
)
OUTPUT_DIR = (
    "results/attacker_evaluation"
)
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SEMANTIC_MODEL_NAME = "all-mpnet-base-v2"
MAX_PREFIX_LEN = 256
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 3
SEED = 42
USE_AMP = True

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_text(value):
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()

def normalize_text(text):
    return re.sub(r"\s+", " ", clean_text(text)).strip()

def first_present(record, keys):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None

class RMSNorm(nn.Module):
    def __init__(self, dimension, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps
    def forward(self, x):
        inv = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * inv

class VictimToEmbedding(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.linear1 = nn.Linear(input_dimension, 4 * output_dimension)
        self.linear2 = nn.Linear(4 * output_dimension, output_dimension)
        self.norm = RMSNorm(output_dimension)
    def forward(self, hidden):
        hidden = F.silu(self.linear1(hidden))
        hidden = F.silu(self.linear2(hidden))
        return self.norm(hidden)

def load_labels(path):
    labels = {}
    id_keys = ["sample_id", "id", "prompt_id", "record_id", "uid"]
    text_keys = [
        "text", "prompt", "original_prompt", "original_text",
        "label", "input", "user_prompt"
    ]
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = first_present(record, id_keys)
            text = first_present(record, text_keys)
            if sample_id is None:
                raise ValueError(
                    f"No sample ID in label line {line_number}; keys={list(record)}"
                )
            if text is None:
                raise ValueError(
                    f"No prompt text in label line {line_number}; keys={list(record)}"
                )
            sample_id = str(sample_id)
            if sample_id in labels:
                raise ValueError(f"Duplicate label sample_id: {sample_id}")
            labels[sample_id] = clean_text(text)
    if not labels:
        raise RuntimeError(f"No labels loaded from {path}")
    return labels

def discover_captures(capture_dir):
    paths = sorted(set(glob.glob(os.path.join(capture_dir, "**", "*.pt"), recursive=True)))
    if not paths:
        raise RuntimeError(f"No .pt files found under {capture_dir}")
    return paths

def load_capture(path):
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Capture is not a dictionary: {path}")
    sample_id = first_present(obj, ["sample_id", "id", "prompt_id", "record_id"])
    meta = obj.get("meta", {}) if isinstance(obj.get("meta", {}), dict) else {}
    if sample_id is None:
        sample_id = first_present(meta, ["sample_id", "id", "prompt_id", "record_id"])
    if sample_id is None:
        raise RuntimeError(f"No sample_id in {path}")
    hidden = first_present(
        obj,
        ["hidden", "activation", "tensor", "qkv_input", "operator_input"],
    )
    if not torch.is_tensor(hidden):
        raise RuntimeError(f"No hidden tensor in {path}")
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden[0]
    if hidden.ndim != 2:
        raise RuntimeError(f"Expected [T,D], got {tuple(hidden.shape)} in {path}")
    if hidden.shape[-1] != 3072:
        raise RuntimeError(f"Expected D=3072, got {hidden.shape[-1]} in {path}")
    hidden = hidden[:MAX_PREFIX_LEN].float().contiguous()
    if torch.isnan(hidden).any() or torch.isinf(hidden).any():
        raise RuntimeError(f"NaN/Inf in {path}")
    return str(sample_id), hidden, meta

def infer_gate_gain(state):
    saved_epoch = int(state.get("epoch", 0))
    return 2.0 if saved_epoch <= 2 else 2.8

def load_attacker(checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "training_state.pt"
    adapter_path = checkpoint_dir / "lora_adapter"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)
    state = torch.load(state_path, map_location="cpu")
    metadata = state.get("metadata", {})
    model_name = metadata.get("attacker_model", BASE_MODEL_NAME)
    victim_hidden_size = int(metadata.get("victim_hidden_size", 3072))
    embedding_size = int(metadata.get("attacker_embedding_size", 3072))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    ).to(device).eval()
    projector = VictimToEmbedding(victim_hidden_size, embedding_size).to(
        device, dtype=torch.float32
    )
    projector.load_state_dict(state["projector"], strict=True)
    projector.eval()
    gate = state["gate"]
    if not torch.is_tensor(gate):
        gate = torch.tensor(float(gate), dtype=torch.float32)
    gate = gate.detach().float().reshape(()).to(device)
    gate_gain = infer_gate_gain(state)
    print("=" * 88)
    print("ATTACKER CHECKPOINT")
    print("=" * 88)
    print(f"Model                 : {model_name}")
    print(f"Checkpoint epoch      : {state.get('epoch')}")
    print(f"Checkpoint step       : {state.get('global_step')}")
    print(f"Best validation loss  : {state.get('best_validation_loss')}")
    print(f"sigmoid(gate)         : {torch.sigmoid(gate).item():.6f}")
    print(f"Gate gain             : {gate_gain:.1f}")
    print(f"Effective gate        : {(gate_gain * torch.sigmoid(gate)).item():.6f}")
    print("=" * 88)
    return tokenizer, model, projector, gate, gate_gain, model_dtype

@torch.no_grad()
def build_soft_prefix(hidden, model, projector, gate, gate_gain, model_dtype, device):
    hidden = hidden.unsqueeze(0).to(device, dtype=torch.float32)
    amp_enabled = device.type == "cuda" and USE_AMP
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=amp_enabled
    ):
        prefix = projector(hidden)
        effective_gate = gate_gain * torch.sigmoid(gate)
        prefix = prefix * effective_gate
    prefix = prefix.to(model_dtype)
    embedding_layer = model.get_input_embeddings()
    bos_id = model.config.bos_token_id
    if bos_id is None:
        bos_id = model.config.eos_token_id
    if isinstance(bos_id, (list, tuple)):
        bos_id = bos_id[0]
    if bos_id is None:
        bos_id = model.config.pad_token_id
    bos_ids = torch.full((1, 1), int(bos_id), dtype=torch.long, device=device)
    bos_embedding = embedding_layer(bos_ids)
    # Match the preview-time scaling used in the training script.
    bos_rms = bos_embedding.float().pow(2).mean().sqrt()
    prefix_rms = prefix.float().pow(2).mean().sqrt()
    prefix_scale = (bos_rms / (prefix_rms + 1e-6)).clamp(0.5, 2.0)
    prefix = prefix * prefix_scale.to(prefix.dtype)
    context = torch.cat([bos_embedding, prefix], dim=1)
    attention_mask = torch.ones(
        1, context.shape[1], dtype=torch.long, device=device
    )
    return context, attention_mask, float(effective_gate.item()), float(prefix_scale.item())

def apply_repetition_penalty(logits, generated_ids, repetition_penalty):
    """Apply Hugging Face-style repetition penalty to generated tokens."""
    if repetition_penalty is None or repetition_penalty == 1.0:
        return logits
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than zero")
    if not generated_ids:
        return logits
    logits = logits.clone()
    for token_id in set(generated_ids):
        token_score = logits[:, token_id]
        logits[:, token_id] = torch.where(
            token_score < 0,
            token_score * repetition_penalty,
            token_score / repetition_penalty,
        )
    return logits

def get_banned_ngram_tokens(generated_ids, no_repeat_ngram_size):
    """Return token IDs that would create a repeated n-gram."""
    if (
        no_repeat_ngram_size is None
        or no_repeat_ngram_size <= 0
        or len(generated_ids) + 1 < no_repeat_ngram_size
    ):
        return set()
    n = int(no_repeat_ngram_size)
    generated_ngrams = {}
    for index in range(len(generated_ids) - n + 1):
        prefix = tuple(generated_ids[index:index + n - 1])
        next_token = generated_ids[index + n - 1]
        generated_ngrams.setdefault(prefix, set()).add(next_token)
    current_prefix = tuple(generated_ids[-(n - 1):])
    return generated_ngrams.get(current_prefix, set())

@torch.no_grad()
def greedy_decode(
    context,
    attention_mask,
    tokenizer,
    model,
    max_new_tokens,
    repetition_penalty=REPETITION_PENALTY,
    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
):
    generated_ids = []
    output = model(
        inputs_embeds=context,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = output.past_key_values
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    model_eos = model.config.eos_token_id
    if isinstance(model_eos, int):
        eos_ids.add(model_eos)
    elif isinstance(model_eos, (list, tuple)):
        eos_ids.update(int(x) for x in model_eos)
    stopped_on_eos = False
    for _ in range(max_new_tokens):
        logits = output.logits[:, -1, :].float()
        logits = apply_repetition_penalty(
            logits=logits,
            generated_ids=generated_ids,
            repetition_penalty=repetition_penalty,
        )
        banned_tokens = get_banned_ngram_tokens(
            generated_ids=generated_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        if banned_tokens:
            logits[:, list(banned_tokens)] = -float("inf")
        next_token = torch.argmax(logits, dim=-1)
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        if token_id in eos_ids:
            stopped_on_eos = True
            break
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    1,
                    1,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        output = model(
            input_ids=next_token.unsqueeze(0),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = output.past_key_values
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return clean_text(text), generated_ids, stopped_on_eos

def token_metrics(original, predicted, tokenizer):
    original_ids = tokenizer.encode(original, add_special_tokens=False)
    predicted_ids = tokenizer.encode(predicted, add_special_tokens=False)
    original_counts = Counter(original_ids)
    predicted_counts = Counter(predicted_ids)
    overlap = sum(
        min(original_counts[token], predicted_counts[token])
        for token in original_counts.keys() | predicted_counts.keys()
    )
    precision = overlap / len(predicted_ids) if predicted_ids else (1.0 if not original_ids else 0.0)
    recall = overlap / len(original_ids) if original_ids else (1.0 if not predicted_ids else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "original_token_count": len(original_ids),
        "predicted_token_count": len(predicted_ids),
        "overlap_token_count": overlap,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }

def levenshtein_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]

def edit_metrics(original, predicted):
    original = normalize_text(original)
    predicted = normalize_text(predicted)
    distance = levenshtein_distance(original, predicted)
    denominator = max(len(original), len(predicted), 1)
    return distance, max(0.0, 1.0 - distance / denominator)

def load_semantic_model():
    """Load the semantic model on CPU so it does not compete with the 3B attacker for GPU memory."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as error:
        raise RuntimeError(
            "sentence-transformers could not be imported. "
            "Install it in this environment with: pip install sentence-transformers"
        ) from error
    semantic_device = "cpu"
    print(
        f"Loading semantic model on {semantic_device}: "
        f"{SEMANTIC_MODEL_NAME}"
    )
    try:
        model = SentenceTransformer(
            SEMANTIC_MODEL_NAME,
            device=semantic_device,
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to load semantic model {SEMANTIC_MODEL_NAME} "
            f"on {semantic_device}: {error}"
        ) from error
    print("Semantic model loaded successfully.")
    return model

def add_semantic_similarity(rows, model):
    if model is None:
        for row in rows:
            row["semantic_similarity"] = None
        return
    originals = [row["original_prompt"] for row in rows]
    predictions = [row["reconstructed_prompt"] for row in rows]
    original_embeddings = model.encode(
        originals,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    predicted_embeddings = model.encode(
        predictions,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = (original_embeddings * predicted_embeddings).sum(dim=1)
    for row, score in zip(rows, scores):
        row["semantic_similarity"] = float(score.item())

def mean_value(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None

def write_csv(path, rows):
    fields = [
        "sample_id", "capture_path", "capture_tokens",
        "generated_token_count", "original_prompt", "reconstructed_prompt",
        "exact_match", "case_insensitive_exact_match",
        "original_token_count", "predicted_token_count", "overlap_token_count",
        "token_precision", "token_recall", "token_f1",
        "levenshtein_distance", "edit_similarity", "semantic_similarity",
        "effective_gate", "prefix_scale", "stopped_on_eos",
        "capture_condition", "capture_phase", "capture_layer", "capture_point",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=BEST_CHECKPOINT_DIR)
    parser.add_argument("--capture-dir", default=CAPTURE_DIR)
    parser.add_argument("--labels-jsonl", default=LABELS_JSONL)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unrelated arguments: {unknown}")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "reconstructions_without_labels.jsonl"
    results_csv = output_dir / "tdx_reconstruction_results.csv"
    results_jsonl = output_dir / "tdx_reconstruction_results.jsonl"
    summary_path = output_dir / "tdx_reconstruction_summary.json"
    labels = load_labels(args.labels_jsonl)
    capture_paths = discover_captures(args.capture_dir)
    if args.limit is not None:
        capture_paths = capture_paths[: max(0, int(args.limit))]
    tokenizer, model, projector, gate, gate_gain, model_dtype = load_attacker(
        args.checkpoint_dir, device
    )
    # Stage 1: generation without accessing labels.
    prediction_rows = []
    seen_ids = set()
    print("=" * 88)
    print("STAGE 1: RECONSTRUCTION FROM ACTIVATIONS ONLY")
    print("=" * 88)
    for capture_path in tqdm(capture_paths, desc="Reconstructing"):
        sample_id, hidden, meta = load_capture(capture_path)
        if sample_id in seen_ids:
            raise RuntimeError(f"Duplicate capture sample_id: {sample_id}")
        seen_ids.add(sample_id)
        context, attention_mask, effective_gate, prefix_scale = build_soft_prefix(
            hidden, model, projector, gate, gate_gain, model_dtype, device
        )
        prediction, generated_ids, stopped_on_eos = greedy_decode(
            context=context,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=int(args.max_new_tokens),
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        )
        row = {
            "sample_id": sample_id,
            "capture_path": os.path.abspath(capture_path),
            "capture_tokens": int(hidden.shape[0]),
            "reconstructed_prompt": prediction,
            "generated_token_ids": generated_ids,
            "generated_token_count": len(generated_ids),
            "effective_gate": effective_gate,
            "prefix_scale": prefix_scale,
            "stopped_on_eos": stopped_on_eos,
            "capture_condition": meta.get("condition"),
            "capture_phase": meta.get("phase"),
            "capture_layer": first_present(meta, ["layer_idx", "layer", "capture_layer"]),
            "capture_point": first_present(meta, ["capture_point", "capture_tensor", "operator"]),
        }
        prediction_rows.append(row)
        print("\n" + "-" * 88)
        print(f"SAMPLE ID: {sample_id}")
        print(prediction)
        print("-" * 88)
    with open(predictions_path, "w", encoding="utf-8") as file:
        for row in prediction_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Stage 2: attach labels and score.
    print("=" * 88)
    print("STAGE 2: LABEL MATCHING AND SCORING")
    print("=" * 88)
    capture_ids = {row["sample_id"] for row in prediction_rows}
    missing_labels = sorted(capture_ids - set(labels))
    if missing_labels:
        raise RuntimeError("Missing labels for: " + ", ".join(missing_labels))
    result_rows = []
    for prediction_row in prediction_rows:
        sample_id = prediction_row["sample_id"]
        original = labels[sample_id]
        predicted = prediction_row["reconstructed_prompt"]
        distance, similarity = edit_metrics(original, predicted)
        row = {
            **prediction_row,
            "original_prompt": original,
            "exact_match": normalize_text(original) == normalize_text(predicted),
            "case_insensitive_exact_match": (
                normalize_text(original).casefold() == normalize_text(predicted).casefold()
            ),
            **token_metrics(original, predicted, tokenizer),
            "levenshtein_distance": distance,
            "edit_similarity": similarity,
        }
        result_rows.append(row)
    if args.skip_semantic:
        semantic_model = None
        print("Semantic similarity explicitly skipped.")
    else:
        semantic_model = load_semantic_model()
    add_semantic_similarity(result_rows, semantic_model)
    write_csv(results_csv, result_rows)
    with open(results_jsonl, "w", encoding="utf-8") as file:
        for row in result_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    exact_count = sum(int(row["exact_match"]) for row in result_rows)
    summary = {
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
        "num_labels": len(labels),
        "num_capture_files": len(capture_paths),
        "num_evaluated": len(result_rows),
        "exact_match_count": exact_count,
        "exact_match_fraction": exact_count / len(result_rows) if result_rows else 0.0,
        "mean_token_precision": mean_value(result_rows, "token_precision"),
        "mean_token_recall": mean_value(result_rows, "token_recall"),
        "mean_token_f1": mean_value(result_rows, "token_f1"),
        "mean_edit_similarity": mean_value(result_rows, "edit_similarity"),
        "mean_semantic_similarity": mean_value(result_rows, "semantic_similarity"),
        "semantic_model": None if semantic_model is None else SEMANTIC_MODEL_NAME,
        "decoding": {
            "method": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": REPETITION_PENALTY,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "max_new_tokens": int(args.max_new_tokens),
        },
        "created_unix_time": time.time(),
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print("\n" + "=" * 88)
    print("REAL TDX TEST COMPLETE")
    print("=" * 88)
    print(f"Evaluated                : {len(result_rows)}")
    print(f"Exact matches            : {exact_count}/{len(result_rows)}")
    print(f"Mean token precision     : {summary['mean_token_precision']:.4f}")
    print(f"Mean token recall        : {summary['mean_token_recall']:.4f}")
    print(f"Mean token F1            : {summary['mean_token_f1']:.4f}")
    print(f"Mean edit similarity     : {summary['mean_edit_similarity']:.4f}")
    if summary["mean_semantic_similarity"] is not None:
        print(f"Mean semantic similarity : {summary['mean_semantic_similarity']:.4f}")
    else:
        print("Mean semantic similarity : not calculated")
    print(f"Predictions only         : {predictions_path}")
    print(f"Detailed CSV             : {results_csv}")
    print(f"Detailed JSONL           : {results_jsonl}")
    print(f"Summary JSON             : {summary_path}")
    print("=" * 88)
if __name__ == "__main__":
    main()
