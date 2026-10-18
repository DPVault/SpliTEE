#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import csv
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

torch.set_num_threads(28)
torch.set_num_interop_threads(1)

MODEL_NAME =  "meta-llama/Llama-3.2-3B-Instruct"
DATASET_NAME = "databricks/databricks-dolly-15k"
SPLIT = "train"
MIN_TOKENS = 100
MAX_TOKENS = 500
TOKEN_RANGES = [
    (100, 102),
]
NUM_PROMPTS_PER_RANGE = 30
CAPS = [100]
BATCH_SIZE = 15
WARMUP_RUNS = 1
WARMUP_MAX_NEW_TOKENS = 2
DEVICE = "cpu"
DTYPE = torch.float32
ATTN_IMPL = "sdpa"

OUTPUT_DIR = Path("results/cpu_runtime")
OUT_CSV = OUTPUT_DIR / "cpu_runtime.csv"
OUT_PNG = OUTPUT_DIR / "cpu_runtime.png"

COLOR_MAP = {
    "Embedding": "#1f77b4",
    "Attention block": "#17becf",
    "MLP block": "#2ca02c",
    "Final norm": "#9467bd",
    "LM head": "#8c564b",
    "Forward other": "#005f56",
    "Generate overhead": "#e377c2",
}

TIMERS = defaultdict(float)


def reset_timers():
    TIMERS.clear()


def t_add(name, dt):
    TIMERS[name] += float(dt)


def t_get(name):
    return float(TIMERS.get(name, 0.0))


def clamp0(x):
    return x if x > 0 else 0.0


def unwrap_tensor(x, where=""):
    """Extract the hidden-state tensor."""
    while isinstance(x, (tuple, list)):
        if len(x) == 0:
            raise RuntimeError(f"Empty tuple/list where tensor expected: {where}")
        x = x[0]
    if not torch.is_tensor(x):
        raise RuntimeError(f"Expected tensor at {where}, got {type(x)}")
    return x


def add_module_timer(module, timer_name):
    def pre_hook(_m, _i):
        _m._t0 = time.perf_counter()
    def post_hook(_m, _i, _o):
        t_add(timer_name, time.perf_counter() - _m._t0)
    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)


def attach_model_forward_total_timer(model):
    def pre_f(_m, _i):
        _m._fwd_total_t0 = time.perf_counter()
    def post_f(_m, _i, _o):
        t_add("forward_total_cpu", time.perf_counter() - _m._fwd_total_t0)
    model.register_forward_pre_hook(pre_f)
    model.register_forward_hook(post_f)


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Could not find decoder layers at model.model.layers")


def get_embed_tokens_module(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    raise ValueError("Could not find embedding module at model.model.embed_tokens")


def get_final_norm_module(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    return None


def make_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def build_dolly_prompt(row):
    instruction = (row.get("instruction") or "").strip()
    context = (row.get("context") or "").strip()
    if context:
        return f"{instruction}\n\nContext:\n{context}"
    return instruction


def render_chat(tok, text):
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def tokenize_as_chat(tok, text):
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )


def tokenize_as_chat_batch(tok, texts):
    rendered = [render_chat(tok, text) for text in texts]
    return tok(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )


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


class BigBlockLlamaDecoderLayerCPU(nn.Module):
    """Time attention and MLP blocks."""
    def __init__(self, base_layer):
        super().__init__()
        self.input_layernorm = base_layer.input_layernorm
        self.self_attn = base_layer.self_attn
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        self.mlp = base_layer.mlp
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        hidden_states = unwrap_tensor(hidden_states, "BigBlockLlamaDecoderLayerCPU.forward hidden_states")
        residual = hidden_states
        # Input norm, attention and residual add.
        t0 = time.perf_counter()
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(attn_outputs, tuple):
            attn_hidden = attn_outputs[0]
            attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
            present_key_value = attn_outputs[2] if len(attn_outputs) > 2 else None
        else:
            attn_hidden = attn_outputs
            attn_weights = None
            present_key_value = None
        attn_hidden = unwrap_tensor(attn_hidden, "self_attn output hidden_states")
        hidden_states = residual + attn_hidden
        t_add("Attention block", time.perf_counter() - t0)
        residual = hidden_states
        # Post-attention norm, MLP and residual add.
        t0 = time.perf_counter()
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = unwrap_tensor(hidden_states, "mlp output hidden_states")
        hidden_states = residual + hidden_states
        t_add("MLP block", time.perf_counter() - t0)
        return hidden_states


def main():
    torch.set_grad_enabled(False)
    print("=" * 80)
    print("CPU BASELINE BATCH TIMING")
    print("=" * 80)
    print(f"MODEL_NAME             : {MODEL_NAME}")
    print(f"DATASET_NAME           : {DATASET_NAME}")
    print(f"SPLIT                  : {SPLIT}")
    print(f"MIN_TOKENS             : {MIN_TOKENS}")
    print(f"MAX_TOKENS             : {MAX_TOKENS}")
    print(f"TOKEN_RANGES           : {TOKEN_RANGES}")
    print(f"NUM_PROMPTS_PER_RANGE  : {NUM_PROMPTS_PER_RANGE}")
    print(f"CAPS                   : {CAPS}")
    print(f"BATCH_SIZE             : {BATCH_SIZE}")
    print(f"WARMUP_RUNS            : {WARMUP_RUNS}")
    print(f"WARMUP_MAX_NEW_TOKENS  : {WARMUP_MAX_NEW_TOKENS}")
    print(f"DEVICE                 : {DEVICE}")
    print(f"DTYPE                  : {DTYPE}")
    print(f"ATTN_IMPL              : {ATTN_IMPL}")
    print("=" * 80)
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    print("Loading model...")
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        attn_implementation=ATTN_IMPL,
    ).to(DEVICE).eval()
    layers = get_transformer_layers(mdl)
    for i, layer in enumerate(layers):
        layers[i] = BigBlockLlamaDecoderLayerCPU(layer)
    # Total time inside model forward calls.
    attach_model_forward_total_timer(mdl)
    # Embedding lookup.
    add_module_timer(get_embed_tokens_module(mdl), "Embedding")
    final_norm = get_final_norm_module(mdl)
    if final_norm is not None:
        # Final normalization.
        add_module_timer(final_norm, "Final norm")
    # Vocabulary projection.
    add_module_timer(mdl.lm_head, "LM head")
    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME)[SPLIT]
    print("Selecting prompts by token ranges...")
    selected, buckets = collect_texts_by_ranges(ds, tok)
    if len(selected) == 0:
        raise RuntimeError("No prompts found in selected token ranges.")
    print("Selected prompt counts by range:")
    for r in TOKEN_RANGES:
        print(f"  {r}: {len(buckets[r])}")
    sample_texts = [x[0] for x in selected]
    seq_len_map = {text: seq_len for text, seq_len in selected}
    print("Selected prompts:")
    for i, text in enumerate(sample_texts):
        print(f"  prompt[{i}] tokens={seq_len_map[text]}")
    batches = list(make_batches(sample_texts, BATCH_SIZE))
    first_batch_texts = batches[0]
    warmup_inputs = tokenize_as_chat_batch(
        tok,
        [first_batch_texts[0]],
    )
    warmup_inputs = {
        k: v.to(DEVICE)
        for k, v in warmup_inputs.items()
    }
    print(
        f"Warming up: runs={WARMUP_RUNS}, "
        f"batch_size=1, new_tokens={WARMUP_MAX_NEW_TOKENS}"
    )
    with torch.inference_mode():
        for warm_idx in range(WARMUP_RUNS):
            reset_timers()
            _ = mdl.generate(
                **warmup_inputs,
                min_new_tokens=WARMUP_MAX_NEW_TOKENS,
                max_new_tokens=WARMUP_MAX_NEW_TOKENS,
                use_cache=True,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            print(
                f"  warmup {warm_idx + 1}/{WARMUP_RUNS} done"
            )
    reset_timers()
    rows = []
    print("Starting timed batch runs...")
    for cap in CAPS:
        avg_acc = defaultdict(float)
        total_prompts = 0
        total_batches = 0
        prompt_tokens_sum = 0.0
        for batch_idx, batch_texts in enumerate(batches, start=1):
            inputs = tokenize_as_chat_batch(tok, batch_texts)
            inputs = {
                k: v.to(DEVICE)
                for k, v in inputs.items()
            }
            bsz = len(batch_texts)
            total_prompts += bsz
            total_batches += 1
            prompt_tokens_sum += sum(seq_len_map[text] for text in batch_texts)
            reset_timers()
            # Full generate() call.
            t0_gen = time.perf_counter()
            with torch.inference_mode():
                _ = mdl.generate(
                    **inputs,
                    min_new_tokens=cap,
                    max_new_tokens=cap,
                    use_cache=True,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            gen_total = time.perf_counter() - t0_gen
            forward_total_cpu = t_get("forward_total_cpu")
            row_batch = {
                "Embedding": t_get("Embedding"),
                "Attention block": t_get("Attention block"),
                "MLP block": t_get("MLP block"),
                "Final norm": t_get("Final norm"),
                "LM head": t_get("LM head"),
            }
            known_forward = sum(row_batch.values())
            # Time outside the measured forward components.
            row_batch["Forward other"] = clamp0(forward_total_cpu - known_forward)
            # Time outside model forward calls.
            row_batch["Generate overhead"] = clamp0(gen_total - forward_total_cpu)
            row_batch["gen_total_s"] = gen_total
            for k, v in row_batch.items():
                avg_acc[k] += v
            print(
                f"Cap={cap:>4} | batch {batch_idx}/{len(batches)} | "
                f"bsz={bsz} | padded={int(inputs['input_ids'].shape[1])} | "
                f"batch_gen_total={gen_total:.4f}s | "
                f"per_prompt={gen_total / bsz:.4f}s"
            )
        row = {
            "cap": cap,
            "batch_size": BATCH_SIZE,
            "num_prompts": total_prompts,
            "num_batches": total_batches,
            "avg_prompt_tokens": prompt_tokens_sum / total_prompts,
            "gen_total_s": avg_acc["gen_total_s"] / total_prompts,
            "Embedding": avg_acc["Embedding"] / total_prompts,
            "Attention block": avg_acc["Attention block"] / total_prompts,
            "MLP block": avg_acc["MLP block"] / total_prompts,
            "Final norm": avg_acc["Final norm"] / total_prompts,
            "LM head": avg_acc["LM head"] / total_prompts,
            "Forward other": avg_acc["Forward other"] / total_prompts,
            "Generate overhead": avg_acc["Generate overhead"] / total_prompts,
        }
        rows.append(row)
        print(
            f"Cap={cap:>4} | "
            f"batch_size={BATCH_SIZE} | "
            f"num_prompts={total_prompts} | "
            f"num_batches={total_batches} | "
            f"avg_gen_total_per_prompt={row['gen_total_s']:.4f}s | "
            f"avg_prompt_tokens={row['avg_prompt_tokens']:.2f} | "
            f"Embedding/prompt={row['Embedding']:.4f}s | "
            f"Attn/prompt={row['Attention block']:.4f}s | "
            f"MLP/prompt={row['MLP block']:.4f}s | "
            f"FinalNorm/prompt={row['Final norm']:.4f}s | "
            f"LMHead/prompt={row['LM head']:.4f}s | "
            f"ForwardOther/prompt={row['Forward other']:.4f}s | "
            f"GenOverhead/prompt={row['Generate overhead']:.4f}s"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved CSV: {OUT_CSV}")
    plt.figure(figsize=(14, 8))
    x = list(range(len(CAPS)))
    bottom = [0.0] * len(CAPS)
    plot_order = [
        "Embedding",
        "Attention block",
        "MLP block",
        "Final norm",
        "LM head",
        "Forward other",
        "Generate overhead",
    ]
    for name in plot_order:
        vals = [r[name] for r in rows]
        plt.bar(
            x,
            vals,
            bottom=bottom,
            color=COLOR_MAP[name],
            label=name,
        )
        bottom = [b + v for b, v in zip(bottom, vals)]
    plt.xticks(x, CAPS)
    plt.xlabel("New Tokens Generated")
    plt.ylabel("Average Time Per Prompt (s)")
    plt.title(
        f"CPU Profiling: Big-Block Timing "
        f"({MODEL_NAME}, ranges={TOKEN_RANGES}, prompts={len(sample_texts)}, "
        f"batch_size={BATCH_SIZE})"
    )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print(f"Saved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
