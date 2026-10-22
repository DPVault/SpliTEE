#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Train a prompt-reconstruction attacker from captured activations.

import argparse
import gc
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    get_linear_schedule_with_warmup,
)
try:
    import wandb
except Exception:
    wandb = None
# ------------------------------
# USER SETTINGS
# ------------------------------
TRAIN_MANIFEST = (
    "data/attacker/train_manifest.jsonl"
)
VALID_MANIFEST = (
    "data/attacker/valid_manifest.jsonl"
)
SAVE_DIR = Path(
    "results/attacker_training"
)
ATTACKER_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
NUM_WORKERS = 0
EPOCHS = 5
MAX_PREFIX_LEN = 256
MAX_TARGET_TOKENS = 256
SEED = 42
# Optimizer
LR_PROJECTOR = 1e-4
LR_LORA = 2e-5
WEIGHT_DECAY_PROJECTOR = 0.05
WARMUP_FRACTION = 0.05
MAX_GRAD_NORM = 1.0
# Mixed precision
USE_AMP = True
# Loss shaping
BOOTSTRAP_K_BASE = 16
BOOTSTRAP_WEIGHT = 1.0
FRONT_K = 128
FRONT_ALPHA = 0.08
L2_PREFIX_WEIGHT_BASE = 1e-4
# LoRA
USE_LORA = True
LORA_LAYERS = 2
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Validation / logging
PRINT_SAMPLE_EVERY_STEPS = 250
SAVE_EVERY_EPOCH = True
ENABLE_WANDB = True
WANDB_PROJECT = "plain_t3_prompt_reconstruction"
WANDB_RUN_NAME = "layer13_qkv_prefix_attacker"
# Generation preview only
PREVIEW_MAX_NEW_TOKENS = 128
# ------------------------------
# REPRODUCIBILITY
# ------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)
# ------------------------------
# SCHEDULE HELPERS
# ------------------------------

def gate_gain(epoch: int) -> float:
    return 2.0 if epoch < 2 else 2.8

def l2_prefix_weight_for_epoch(epoch: int) -> float:
    return L2_PREFIX_WEIGHT_BASE if epoch < 2 else 5e-5

def bootstrap_k_for_epoch(epoch: int) -> int:
    return BOOTSTRAP_K_BASE if epoch < 2 else 12

def teacher_forcing_length_for_epoch(epoch: int) -> int:
    if epoch == 0:
        return 64
    if epoch == 1:
        return 128
    return MAX_TARGET_TOKENS
# ------------------------------
# DATASET
# ------------------------------

def read_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            sample_path = record.get("sample_path")
            text = record.get("text")
            sample_id = record.get("sample_id")
            if not sample_path:
                raise ValueError(
                    f"Missing sample_path in {path}, line {line_number}"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Missing/empty text in {path}, line {line_number}"
                )
            records.append(
                {
                    "sample_id": str(sample_id),
                    "sample_path": str(sample_path),
                    "text": text,
                }
            )
    return records

class ManifestActivationDataset(Dataset):
    """

    Lazy loader for one activation file per sample.

    Only the requested sample is loaded from disk, so the entire activation

    dataset does not occupy host memory.

    """
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.records = read_jsonl(manifest_path)
        if not self.records:
            raise RuntimeError(f"No records found in {manifest_path}")
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        record = self.records[index]
        sample_path = record["sample_path"]
        obj = torch.load(
            sample_path,
            map_location="cpu",
        )
        hidden = obj.get("hidden")
        text = obj.get("text", record["text"])
        sample_id = obj.get("sample_id", record["sample_id"])
        if not torch.is_tensor(hidden):
            raise RuntimeError(
                f"Sample does not contain a tensor: {sample_path}"
            )
        if hidden.ndim != 2:
            raise RuntimeError(
                f"Expected [T,D], got {tuple(hidden.shape)}: {sample_path}"
            )
        if hidden.shape[-1] != 3072:
            raise RuntimeError(
                f"Expected hidden size 3072, got {hidden.shape[-1]}: "
                f"{sample_path}"
            )
        hidden = hidden[:MAX_PREFIX_LEN].float().contiguous()
        if torch.isnan(hidden).any().item():
            raise RuntimeError(f"NaN in sample: {sample_path}")
        if torch.isinf(hidden).any().item():
            raise RuntimeError(f"Inf in sample: {sample_path}")
        return {
            "hidden": hidden,
            "text": text,
            "sample_id": str(sample_id),
        }

def collate_fn(batch):
    hidden_list = [item["hidden"] for item in batch]
    texts = [item["text"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]
    lengths = torch.tensor(
        [hidden.shape[0] for hidden in hidden_list],
        dtype=torch.long,
    )
    max_length = int(lengths.max().item())
    hidden_size = int(hidden_list[0].shape[1])
    hidden_batch = torch.zeros(
        len(hidden_list),
        max_length,
        hidden_size,
        dtype=torch.float32,
    )
    prefix_mask = torch.zeros(
        len(hidden_list),
        max_length,
        dtype=torch.long,
    )
    for index, hidden in enumerate(hidden_list):
        length = hidden.shape[0]
        hidden_batch[index, :length] = hidden
        prefix_mask[index, :length] = 1
    return {
        "hidden": hidden_batch,
        "prefix_mask": prefix_mask,
        "texts": texts,
        "sample_ids": sample_ids,
    }
# ------------------------------
# PROJECTOR
# ------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps
    def forward(self, x):
        inverse_rms = (
            x.pow(2)
            .mean(dim=-1, keepdim=True)
            .add(self.eps)
            .rsqrt()
        )
        return self.weight * x * inverse_rms

class VictimToEmbedding(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int):
        super().__init__()
        self.linear1 = nn.Linear(
            input_dimension,
            4 * output_dimension,
        )
        self.linear2 = nn.Linear(
            4 * output_dimension,
            output_dimension,
        )
        self.norm = RMSNorm(output_dimension)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    def forward(self, hidden):
        hidden = F.silu(self.linear1(hidden))
        hidden = F.silu(self.linear2(hidden))
        return self.norm(hidden)
# ------------------------------
# MODEL SETUP
# ------------------------------

def build_attacker_model():
    print(f"Loading attacker model: {ATTACKER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        ATTACKER_MODEL,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_dtype = (
        torch.float16
        if DEVICE.type == "cuda"
        else torch.float32
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        ATTACKER_MODEL,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.generation_config = (
        GenerationConfig.from_model_config(base_model.config)
    )
    try:
        base_model.gradient_checkpointing_enable()
    except Exception:
        pass
    peft_model = None
    model_for_training = base_model
    if USE_LORA:
        try:
            from peft import LoraConfig, get_peft_model
            target_modules = []
            for layer_index in range(int(LORA_LAYERS)):
                prefix = f"model.layers.{layer_index}"
                target_modules.extend(
                    [
                        f"{prefix}.self_attn.q_proj",
                        f"{prefix}.self_attn.k_proj",
                        f"{prefix}.self_attn.v_proj",
                        f"{prefix}.self_attn.o_proj",
                        f"{prefix}.mlp.gate_proj",
                        f"{prefix}.mlp.up_proj",
                        f"{prefix}.mlp.down_proj",
                    ]
                )
            lora_config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            peft_model = get_peft_model(
                base_model,
                lora_config,
            )
            try:
                peft_model.gradient_checkpointing_enable()
            except Exception:
                pass
            peft_model.print_trainable_parameters()
            model_for_training = peft_model
        except Exception as error:
            raise RuntimeError(
                f"Failed to configure PEFT/LoRA: {error}"
            ) from error
    else:
        for parameter in base_model.parameters():
            parameter.requires_grad = False
    return (
        tokenizer,
        base_model,
        peft_model,
        model_for_training,
        model_dtype,
    )
# ------------------------------
# LOSS
# ------------------------------

def cross_entropy_none(logits, targets):
    return F.cross_entropy(
        logits.float(),
        targets,
        reduction="none",
        ignore_index=-100,
    )

def prepare_targets(tokenizer, texts):
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TARGET_TOKENS,
        padding=True,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    # Remove BOS if the tokenizer inserted it for most rows. The manually
    # created BOS embedding is already prepended to the soft prefix.
    if (
        tokenizer.bos_token_id is not None
        and input_ids.shape[1] > 0
    ):
        begins_with_bos = (
            input_ids[:, 0] == tokenizer.bos_token_id
        ).float().mean().item()
        if begins_with_bos > 0.5:
            input_ids = input_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]
    return input_ids, attention_mask

def build_prefix(
    projector,
    gate,
    model_for_training,
    hidden_batch,
    prefix_mask,
    epoch,
    model_dtype,
):
    hidden_batch = hidden_batch.to(
        DEVICE,
        dtype=torch.float32,
    )
    prefix_mask = prefix_mask.to(
        DEVICE,
        dtype=torch.long,
    )
    amp_enabled = DEVICE.type == "cuda" and USE_AMP
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        prefix_float = projector(hidden_batch)
        scalar_gate = gate_gain(epoch) * torch.sigmoid(gate)
        prefix_float = prefix_float * scalar_gate
    prefix = prefix_float.to(model_dtype)
    # Explicitly zero padded positions. This is important when batch size is
    # increased in future runs.
    prefix = prefix * prefix_mask.unsqueeze(-1).to(prefix.dtype)
    return prefix, prefix_mask

def compute_losses(
    batch,
    epoch,
    tokenizer,
    model_for_training,
    projector,
    gate,
    model_dtype,
):
    texts = batch["texts"]
    hidden_batch = batch["hidden"]
    prefix_mask = batch["prefix_mask"]
    input_ids_full, target_attention_mask = prepare_targets(
        tokenizer,
        texts,
    )
    batch_size = int(input_ids_full.shape[0])
    target_total_length = int(input_ids_full.shape[1])
    desired_tf_length = teacher_forcing_length_for_epoch(epoch)
    tf_length = min(
        desired_tf_length,
        max(0, target_total_length - 1),
    )
    if tf_length > 0:
        target_ids = input_ids_full[:, : tf_length + 1]
        target_mask = target_attention_mask[:, : tf_length + 1]
        input_ids_shifted = target_ids[:, :-1]
        labels = target_ids[:, 1:].clone()
        labels[target_mask[:, 1:] == 0] = -100
        response_length = int(labels.shape[1])
    else:
        input_ids_shifted = input_ids_full[:, :0]
        labels = input_ids_full[:, :0]
        target_mask = target_attention_mask[:, :0]
        response_length = 0
    embedding_layer = model_for_training.get_input_embeddings()
    bos_id = (
        tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else tokenizer.eos_token_id
    )
    if bos_id is None:
        bos_id = tokenizer.pad_token_id
    bos_ids = torch.full(
        (batch_size, 1),
        int(bos_id),
        dtype=torch.long,
        device=DEVICE,
    )
    bos_embedding = embedding_layer(bos_ids)
    prefix, prefix_mask_device = build_prefix(
        projector=projector,
        gate=gate,
        model_for_training=model_for_training,
        hidden_batch=hidden_batch,
        prefix_mask=prefix_mask,
        epoch=epoch,
        model_dtype=model_dtype,
    )
    prefix_length = int(prefix.shape[1])
    token_embeddings = (
        embedding_layer(input_ids_shifted)
        if response_length > 0
        else torch.empty(
            batch_size,
            0,
            embedding_layer.weight.shape[1],
            device=DEVICE,
            dtype=model_dtype,
        )
    )
    # Match prefix scale to token-embedding scale while ignoring padding.
    with torch.no_grad():
        if token_embeddings.numel() > 0:
            token_rms = (
                token_embeddings
                .detach()
                .float()
                .pow(2)
                .mean()
                .sqrt()
            )
        else:
            token_rms = (
                bos_embedding
                .detach()
                .float()
                .pow(2)
                .mean()
                .sqrt()
            )
        valid_prefix = prefix_mask_device.unsqueeze(-1).bool()
        prefix_values = prefix.detach().float()[valid_prefix.expand_as(prefix)]
        if prefix_values.numel() > 0:
            prefix_rms = prefix_values.pow(2).mean().sqrt()
        else:
            prefix_rms = torch.tensor(
                1.0,
                device=DEVICE,
            )
        prefix_scale = (
            token_rms / (prefix_rms + 1e-6)
        ).clamp(0.5, 2.0)
    prefix = prefix * prefix_scale.to(prefix.dtype)
    input_parts = [
        bos_embedding,
        prefix,
    ]
    if token_embeddings.numel() > 0:
        input_parts.append(token_embeddings)
    inputs_embeds = torch.cat(
        input_parts,
        dim=1,
    )
    bos_mask = torch.ones(
        batch_size,
        1,
        dtype=torch.long,
        device=DEVICE,
    )
    attention_parts = [
        bos_mask,
        prefix_mask_device,
    ]
    if response_length > 0:
        attention_parts.append(
            target_mask[:, :-1].to(torch.long)
        )
    full_attention_mask = torch.cat(
        attention_parts,
        dim=1,
    )
    amp_enabled = DEVICE.type == "cuda" and USE_AMP
    # -------------------------------------------------------------------------
    # Bootstrap loss
    # -------------------------------------------------------------------------
    bootstrap_length = min(
        bootstrap_k_for_epoch(epoch),
        target_total_length,
    )
    bootstrap_context = torch.cat(
        [bos_embedding, prefix],
        dim=1,
    )
    bootstrap_attention = torch.cat(
        [bos_mask, prefix_mask_device],
        dim=1,
    )
    bootstrap_losses = []
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        first_output = model_for_training(
            input_ids=None,
            inputs_embeds=bootstrap_context,
            attention_mask=bootstrap_attention,
            use_cache=False,
            return_dict=True,
        )
        first_target = input_ids_full[:, 0]
        first_valid = target_attention_mask[:, 0].float()
        first_ce = cross_entropy_none(
            first_output.logits[:, -1, :],
            first_target,
        )
        first_loss = (
            first_ce * first_valid
        ).sum() / first_valid.sum().clamp_min(1.0)
        bootstrap_losses.append(first_loss)
        if bootstrap_length > 1:
            gold_embeddings = embedding_layer(
                input_ids_full[:, : bootstrap_length - 1]
            )
            for step in range(1, bootstrap_length):
                context = torch.cat(
                    [
                        bootstrap_context,
                        gold_embeddings[:, :step, :],
                    ],
                    dim=1,
                )
                gold_mask = target_attention_mask[
                    :, :step
                ].to(torch.long)
                context_attention = torch.cat(
                    [
                        bootstrap_attention,
                        gold_mask,
                    ],
                    dim=1,
                )
                output = model_for_training(
                    input_ids=None,
                    inputs_embeds=context,
                    attention_mask=context_attention,
                    use_cache=False,
                    return_dict=True,
                )
                target = input_ids_full[:, step]
                valid = target_attention_mask[:, step].float()
                token_ce = cross_entropy_none(
                    output.logits[:, -1, :],
                    target,
                )
                token_loss = (
                    token_ce * valid
                ).sum() / valid.sum().clamp_min(1.0)
                bootstrap_losses.append(token_loss)
    loss_bootstrap = torch.stack(
        bootstrap_losses
    ).mean()
    # -------------------------------------------------------------------------
    # Teacher-forced reconstruction loss
    # -------------------------------------------------------------------------
    if response_length > 0:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model_for_training(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = output.logits
        response_start = 1 + prefix_length
        response_logits = logits[
            :,
            response_start - 1:
            response_start - 1 + response_length,
            :,
        ]
        positions = torch.arange(
            response_length,
            device=DEVICE,
        ).unsqueeze(0).expand(batch_size, -1)
        weights = torch.exp(
            -FRONT_ALPHA * positions
        )
        if FRONT_K > 0:
            weights = torch.where(
                positions < FRONT_K,
                weights,
                torch.ones_like(weights),
            )
        valid = (labels != -100).float()
        weights = weights * valid
        weights = weights / weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        token_ce = cross_entropy_none(
            response_logits.reshape(
                -1,
                response_logits.shape[-1],
            ),
            labels.reshape(-1),
        ).reshape(batch_size, response_length)
        loss_teacher_forced = (
            weights * token_ce
        ).sum() / max(1, batch_size)
        valid_ce = token_ce[labels != -100]
        if valid_ce.numel() > 0:
            average_ce = float(
                valid_ce.mean().detach().item()
            )
        else:
            average_ce = float("nan")
        valid_tokens = int(valid.sum().item())
    else:
        loss_teacher_forced = torch.zeros(
            (),
            device=DEVICE,
        )
        average_ce = float("nan")
        valid_tokens = 0
    # -------------------------------------------------------------------------
    # Prefix regularization
    # -------------------------------------------------------------------------
    valid_prefix_mask = (
        prefix_mask_device
        .unsqueeze(-1)
        .to(prefix.dtype)
    )
    prefix_squared_sum = (
        prefix.float().pow(2)
        * valid_prefix_mask
    ).sum()
    prefix_denominator = (
        valid_prefix_mask.sum()
        * prefix.shape[-1]
    ).clamp_min(1.0)
    prefix_l2 = (
        l2_prefix_weight_for_epoch(epoch)
        * prefix_squared_sum
        / prefix_denominator
    )
    total_loss = (
        BOOTSTRAP_WEIGHT * loss_bootstrap
        + loss_teacher_forced
        + prefix_l2
    )
    return {
        "loss_total": total_loss,
        "loss_bootstrap": loss_bootstrap,
        "loss_teacher_forced": loss_teacher_forced,
        "loss_prefix_l2": prefix_l2,
        "valid_tokens": valid_tokens,
        "average_ce": average_ce,
    }
# ------------------------------
# CHECKPOINTING
# ------------------------------

def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    projector,
    gate,
    optimizer,
    scheduler,
    scaler,
    peft_model,
    metadata: dict,
):
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        {
            "projector": projector.state_dict(),
            "gate": gate.detach().cpu(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_validation_loss": float(
                best_validation_loss
            ),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "metadata": metadata,
        },
        checkpoint_dir / "training_state.pt",
    )
    if peft_model is not None:
        adapter_dir = checkpoint_dir / "lora_adapter"
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        peft_model.save_pretrained(adapter_dir)
    with open(
        checkpoint_dir / "checkpoint_meta.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                **metadata,
                "epoch": int(epoch),
                "global_step": int(global_step),
                "best_validation_loss": float(
                    best_validation_loss
                ),
            },
            file,
            indent=2,
        )
# ------------------------------
# PREVIEW GENERATION
# ------------------------------

@torch.no_grad()
def generate_preview(
    batch,
    epoch,
    tokenizer,
    model_for_training,
    projector,
    gate,
    model_dtype,
):
    model_for_training.eval()
    projector.eval()
    hidden_batch = batch["hidden"][:1]
    prefix_mask = batch["prefix_mask"][:1]
    original_text = batch["texts"][0]
    embedding_layer = model_for_training.get_input_embeddings()
    bos_id = (
        tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else tokenizer.eos_token_id
    )
    if bos_id is None:
        bos_id = tokenizer.pad_token_id
    bos_ids = torch.full(
        (1, 1),
        int(bos_id),
        device=DEVICE,
        dtype=torch.long,
    )
    bos_embedding = embedding_layer(bos_ids)
    prefix, prefix_mask_device = build_prefix(
        projector=projector,
        gate=gate,
        model_for_training=model_for_training,
        hidden_batch=hidden_batch,
        prefix_mask=prefix_mask,
        epoch=epoch,
        model_dtype=model_dtype,
    )
    with torch.no_grad():
        token_rms = (
            bos_embedding
            .float()
            .pow(2)
            .mean()
            .sqrt()
        )
        valid_mask = prefix_mask_device.unsqueeze(-1).bool()
        values = prefix.float()[valid_mask.expand_as(prefix)]
        prefix_rms = values.pow(2).mean().sqrt()
        scale = (
            token_rms / (prefix_rms + 1e-6)
        ).clamp(0.5, 2.0)
    prefix = prefix * scale.to(prefix.dtype)
    context = torch.cat(
        [bos_embedding, prefix],
        dim=1,
    )
    attention_mask = torch.cat(
        [
            torch.ones(
                1,
                1,
                dtype=torch.long,
                device=DEVICE,
            ),
            prefix_mask_device,
        ],
        dim=1,
    )
    generated_ids = model_for_training.generate(
        inputs_embeds=context,
        attention_mask=attention_mask,
        max_new_tokens=PREVIEW_MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    prediction = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )
    print("\n" + "=" * 88)
    print("RECONSTRUCTION PREVIEW")
    print("=" * 88)
    print("ORIGINAL:")
    print(original_text[:800])
    print("\nPREDICTED:")
    print(prediction[:800])
    print("=" * 88 + "\n")
# ------------------------------
# VALIDATION
# ------------------------------

@torch.no_grad()
def evaluate(
    epoch,
    validation_loader,
    tokenizer,
    model_for_training,
    projector,
    gate,
    model_dtype,
    global_step,
    wandb_run,
):
    model_for_training.eval()
    projector.eval()
    totals = {
        "loss_total": 0.0,
        "loss_bootstrap": 0.0,
        "loss_teacher_forced": 0.0,
        "loss_prefix_l2": 0.0,
        "valid_tokens": 0,
        "average_ce_sum": 0.0,
        "average_ce_count": 0,
        "batches": 0,
    }
    for batch in tqdm(
        validation_loader,
        desc=f"Validation epoch {epoch + 1}",
    ):
        losses = compute_losses(
            batch=batch,
            epoch=epoch,
            tokenizer=tokenizer,
            model_for_training=model_for_training,
            projector=projector,
            gate=gate,
            model_dtype=model_dtype,
        )
        totals["loss_total"] += float(
            losses["loss_total"].item()
        )
        totals["loss_bootstrap"] += float(
            losses["loss_bootstrap"].item()
        )
        totals["loss_teacher_forced"] += float(
            losses["loss_teacher_forced"].item()
        )
        totals["loss_prefix_l2"] += float(
            losses["loss_prefix_l2"].item()
        )
        totals["valid_tokens"] += int(
            losses["valid_tokens"]
        )
        if not math.isnan(losses["average_ce"]):
            totals["average_ce_sum"] += float(
                losses["average_ce"]
            )
            totals["average_ce_count"] += 1
        totals["batches"] += 1
    denominator = max(1, totals["batches"])
    metrics = {
        "loss_total": (
            totals["loss_total"] / denominator
        ),
        "loss_bootstrap": (
            totals["loss_bootstrap"] / denominator
        ),
        "loss_teacher_forced": (
            totals["loss_teacher_forced"] / denominator
        ),
        "loss_prefix_l2": (
            totals["loss_prefix_l2"] / denominator
        ),
        "valid_tokens": totals["valid_tokens"],
    }
    if totals["average_ce_count"] > 0:
        metrics["average_ce"] = (
            totals["average_ce_sum"]
            / totals["average_ce_count"]
        )
        metrics["perplexity"] = math.exp(
            min(metrics["average_ce"], 20.0)
        )
    else:
        metrics["average_ce"] = float("nan")
        metrics["perplexity"] = float("nan")
    print(
        f"[VALID] epoch={epoch + 1} "
        f"total={metrics['loss_total']:.6f} "
        f"boot={metrics['loss_bootstrap']:.6f} "
        f"tf={metrics['loss_teacher_forced']:.6f} "
        f"ce={metrics['average_ce']:.6f} "
        f"ppl={metrics['perplexity']:.3f}"
    )
    if wandb_run is not None:
        wandb.log(
            {
                "global_step": global_step,
                "epoch": epoch + 1,
                "valid/loss_total": metrics["loss_total"],
                "valid/loss_bootstrap": metrics[
                    "loss_bootstrap"
                ],
                "valid/loss_teacher_forced": metrics[
                    "loss_teacher_forced"
                ],
                "valid/loss_prefix_l2": metrics[
                    "loss_prefix_l2"
                ],
                "valid/average_ce": metrics["average_ce"],
                "valid/perplexity": metrics["perplexity"],
                "valid/tokens": metrics["valid_tokens"],
            }
        )
    return metrics
# ------------------------------
# MAIN TRAINING
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-manifest",
        default=TRAIN_MANIFEST,
    )
    parser.add_argument(
        "--valid-manifest",
        default=VALID_MANIFEST,
    )
    parser.add_argument(
        "--save-dir",
        default=str(SAVE_DIR),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--limit-valid",
        type=int,
        default=None,
    )
    # Works both in a terminal and inside Jupyter.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unrelated arguments: {unknown}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    train_dataset = ManifestActivationDataset(
        args.train_manifest
    )
    validation_dataset = ManifestActivationDataset(
        args.valid_manifest
    )
    if args.limit_train is not None:
        train_dataset.records = train_dataset.records[
            : int(args.limit_train)
        ]
    if args.limit_valid is not None:
        validation_dataset.records = (
            validation_dataset.records[
                : int(args.limit_valid)
            ]
        )
    print("=" * 88)
    print("PLAIN-T3 PROMPT-RECONSTRUCTION ATTACKER TRAINING")
    print("=" * 88)
    print(f"Train manifest       : {args.train_manifest}")
    print(f"Validation manifest  : {args.valid_manifest}")
    print(f"Training samples     : {len(train_dataset)}")
    print(f"Validation samples   : {len(validation_dataset)}")
    print(f"Attacker model       : {ATTACKER_MODEL}")
    print(f"Device               : {DEVICE}")
    print(f"Epochs               : {args.epochs}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Maximum prefix       : {MAX_PREFIX_LEN}")
    print(f"Maximum target       : {MAX_TARGET_TOKENS}")
    print(f"LoRA enabled         : {USE_LORA}")
    print(f"Save directory       : {save_dir}")
    print("=" * 88)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    (
        tokenizer,
        base_model,
        peft_model,
        model_for_training,
        model_dtype,
    ) = build_attacker_model()
    embedding_dimension = int(
        model_for_training
        .get_input_embeddings()
        .weight.shape[1]
    )
    projector = VictimToEmbedding(
        input_dimension=3072,
        output_dimension=embedding_dimension,
    ).to(
        DEVICE,
        dtype=torch.float32,
    )
    gate = nn.Parameter(
        torch.tensor(
            0.5,
            dtype=torch.float32,
            device=DEVICE,
        )
    )
    projector_parameters = (
        list(projector.parameters())
        + [gate]
    )
    lora_parameters = []
    if peft_model is not None:
        lora_parameters = [
            parameter
            for parameter in peft_model.parameters()
            if parameter.requires_grad
        ]
    parameter_groups = [
        {
            "params": projector_parameters,
            "lr": LR_PROJECTOR,
            "weight_decay": WEIGHT_DECAY_PROJECTOR,
        }
    ]
    if lora_parameters:
        parameter_groups.append(
            {
                "params": lora_parameters,
                "lr": LR_LORA,
                "weight_decay": 0.0,
            }
        )
    optimizer = AdamW(parameter_groups)
    total_steps = max(
        1,
        len(train_loader) * int(args.epochs),
    )
    warmup_steps = max(
        1,
        int(total_steps * WARMUP_FRACTION),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    amp_enabled = (
        DEVICE.type == "cuda"
        and USE_AMP
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )
    run_metadata = {
        "attacker_model": ATTACKER_MODEL,
        "train_manifest": os.path.abspath(
            args.train_manifest
        ),
        "valid_manifest": os.path.abspath(
            args.valid_manifest
        ),
        "train_samples": len(train_dataset),
        "valid_samples": len(validation_dataset),
        "victim_hidden_size": 3072,
        "attacker_embedding_size": embedding_dimension,
        "capture_layer": 0,
        "capture_point": (
            "qkv_input_after_input_layernorm"
        ),
        "condition": "plain_t3",
        "masked": False,
        "max_prefix_len": MAX_PREFIX_LEN,
        "max_target_tokens": MAX_TARGET_TOKENS,
        "epochs": int(args.epochs),
        "batch_size": BATCH_SIZE,
        "lr_projector": LR_PROJECTOR,
        "lr_lora": LR_LORA,
        "use_lora": USE_LORA,
        "lora_layers": LORA_LAYERS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "seed": SEED,
        "created_unix_time": time.time(),
    }
    with open(
        save_dir / "training_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_metadata,
            file,
            indent=2,
        )
    wandb_run = None
    if ENABLE_WANDB:
        if wandb is None:
            raise RuntimeError(
                "ENABLE_WANDB=True but wandb is not installed"
            )
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config=run_metadata,
        )
        wandb.define_metric("global_step")
        wandb.define_metric(
            "train/*",
            step_metric="global_step",
        )
        wandb.define_metric(
            "valid/*",
            step_metric="global_step",
        )
        wandb.watch(
            projector,
            log="gradients",
            log_freq=200,
        )
    best_validation_loss = float("inf")
    global_step = 0
    model_for_training.train()
    projector.train()
    for epoch in range(int(args.epochs)):
        model_for_training.train()
        projector.train()
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )
        running_loss = 0.0
        running_batches = 0
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(
                batch=batch,
                epoch=epoch,
                tokenizer=tokenizer,
                model_for_training=model_for_training,
                projector=projector,
                gate=gate,
                model_dtype=model_dtype,
            )
            total_loss = losses["loss_total"]
            if (
                torch.isnan(total_loss)
                or torch.isinf(total_loss)
            ):
                print(
                    "[WARN] Skipping NaN/Inf loss batch",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(total_loss).backward()
            # Correct AMP order: unscale before clipping.
            scaler.unscale_(optimizer)
            parameters_to_clip = (
                list(projector.parameters())
                + [gate]
                + lora_parameters
            )
            torch.nn.utils.clip_grad_norm_(
                parameters_to_clip,
                MAX_GRAD_NORM,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            running_batches += 1
            running_loss += float(
                total_loss.detach().item()
            )
            progress.set_postfix(
                loss=f"{float(total_loss.detach().item()):.4f}",
                boot=(
                    f"{float(losses['loss_bootstrap'].detach().item()):.4f}"
                ),
                tf=(
                    f"{float(losses['loss_teacher_forced'].detach().item()):.4f}"
                ),
                step=global_step,
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "global_step": global_step,
                        "epoch": (
                            epoch
                            + running_batches
                            / max(1, len(train_loader))
                        ),
                        "train/loss_total": float(
                            total_loss.detach().item()
                        ),
                        "train/loss_bootstrap": float(
                            losses[
                                "loss_bootstrap"
                            ].detach().item()
                        ),
                        "train/loss_teacher_forced": float(
                            losses[
                                "loss_teacher_forced"
                            ].detach().item()
                        ),
                        "train/loss_prefix_l2": float(
                            losses[
                                "loss_prefix_l2"
                            ].detach().item()
                        ),
                        "train/lr_projector": optimizer.param_groups[
                            0
                        ]["lr"],
                        "train/gate_sigmoid": float(
                            torch.sigmoid(gate)
                            .detach()
                            .item()
                        ),
                    }
                )
            if (
                PRINT_SAMPLE_EVERY_STEPS > 0
                and global_step
                % PRINT_SAMPLE_EVERY_STEPS
                == 0
            ):
                generate_preview(
                    batch=batch,
                    epoch=epoch,
                    tokenizer=tokenizer,
                    model_for_training=model_for_training,
                    projector=projector,
                    gate=gate,
                    model_dtype=model_dtype,
                )
                model_for_training.train()
                projector.train()
        epoch_train_loss = (
            running_loss / max(1, running_batches)
        )
        print(
            f"[TRAIN] epoch={epoch + 1} "
            f"average_loss={epoch_train_loss:.6f}"
        )
        validation_metrics = evaluate(
            epoch=epoch,
            validation_loader=validation_loader,
            tokenizer=tokenizer,
            model_for_training=model_for_training,
            projector=projector,
            gate=gate,
            model_dtype=model_dtype,
            global_step=global_step,
            wandb_run=wandb_run,
        )
        if SAVE_EVERY_EPOCH:
            save_checkpoint(
                checkpoint_dir=(
                    save_dir / f"checkpoint_epoch_{epoch + 1}"
                ),
                epoch=epoch + 1,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                projector=projector,
                gate=gate,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                peft_model=peft_model,
                metadata=run_metadata,
            )
        current_validation_loss = validation_metrics[
            "loss_total"
        ]
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            save_checkpoint(
                checkpoint_dir=save_dir / "best",
                epoch=epoch + 1,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                projector=projector,
                gate=gate,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                peft_model=peft_model,
                metadata=run_metadata,
            )
            print(
                f"[BEST] New best validation loss: "
                f"{best_validation_loss:.6f}"
            )
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    save_checkpoint(
        checkpoint_dir=save_dir / "final",
        epoch=int(args.epochs),
        global_step=global_step,
        best_validation_loss=best_validation_loss,
        projector=projector,
        gate=gate,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        peft_model=peft_model,
        metadata=run_metadata,
    )
    tokenizer.save_pretrained(
        save_dir / "tokenizer"
    )
    print("=" * 88)
    print("ATTACKER TRAINING COMPLETE")
    print("=" * 88)
    print(f"Best validation loss : {best_validation_loss:.6f}")
    print(f"Best checkpoint      : {save_dir / 'best'}")
    print(f"Final checkpoint     : {save_dir / 'final'}")
    print("=" * 88)
    if wandb_run is not None:
        wandb.finish()
if __name__ == "__main__":
    main()
