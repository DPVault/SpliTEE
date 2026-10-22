#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""Clean T3 inference client for GPU-visible activation capture inside TDX."""
import argparse
import csv
import hashlib
import json
import mmap
import os
import pickle
import random
import socket
import struct
import time
import uuid
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# CONFIGURATION
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_NAME = "databricks/databricks-dolly-15k"
DATASET_SPLIT = "train"

# Select prompts by chat-template token length.
MIN_TOKENS = 100
MAX_TOKENS = 500
TOKEN_RANGES = [
    (100, 300),
]
NUM_PROMPTS_PER_RANGE = 300

# Batch size 1 is the easiest setting for the first leakage experiment.

# The code also supports larger greedy-decoding batches.
BATCH_SIZE = 10

# One generated token is enough to execute the complete prompt-prefill path.
MAX_NEW_TOKENS = 1

# Optional warm-up. Warm-up capture is disabled automatically.
WARMUP_RUNS = 0
WARMUP_MAX_NEW_TOKENS = 1
SEED = 1234
ATTN_IMPL = "sdpa"
DTYPE_MODEL_NAME = "float32"
DTYPE_RPC_NAME = "float32"
CPU_DEVICE = torch.device("cpu")

# IVSHMEM path visible inside TDX.
SHM_PATH = "/sys/bus/pci/devices/0000:00:03.0/resource2"
SHM_SIZE = 8 * 1024 * 1024 * 1024

# Fixed offsets are safe because requests are sequential.
IN_OFF = 0
OUT_BASE_OFF = 1024 * 1024 * 1024
OUT_STRIDE = 1024 * 1024 * 1024

# Files written inside TDX. The host worker does not receive these texts.
OUTPUT_DIR = "results/tdx_clean_attack_capture"
LABELS_JSONL = os.path.join(OUTPUT_DIR, "LM_plain_t3_prompt_labels.jsonl")
RUN_MANIFEST_JSON = os.path.join(OUTPUT_DIR, "LM_plain_t3_run_manifest.json")
GENERATIONS_CSV = os.path.join(OUTPUT_DIR, "LM_plain_t3_generations.csv")

# REPRODUCIBILITY / RUNTIME SETTINGS
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(28)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# DTYPE HELPERS

def dtype_from_name(name: str) -> torch.dtype:
    name = name.lower().strip()
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError("dtype must be float16, float32, or float64")
    return mapping[name]
DTYPE_MODEL = dtype_from_name(DTYPE_MODEL_NAME)
DTYPE_RPC = dtype_from_name(DTYPE_RPC_NAME)
_DTYPE_TO_NUMPY = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
}
_TORCH_TO_DTYPE_NAME = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int32: "int32",
    torch.int64: "int64",
}

def numel_from_shape(shape) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)

def nbytes_for(shape, dtype_name: str) -> int:
    return numel_from_shape(shape) * np.dtype(_DTYPE_TO_NUMPY[dtype_name]).itemsize

def unwrap_tensor(value, where: str = "") -> torch.Tensor:
    while isinstance(value, (tuple, list)):
        if not value:
            raise RuntimeError(f"Empty tuple/list where tensor expected: {where}")
        value = value[0]
    if not torch.is_tensor(value):
        raise RuntimeError(f"Expected tensor at {where}, got {type(value)}")
    return value

def to_cpu_tensor(
    value,
    dtype: torch.dtype,
    where: str,
    contiguous: bool = True,
) -> torch.Tensor:
    value = unwrap_tensor(value, where)
    if value.device.type != "cpu" or value.dtype != dtype:
        value = value.to(device=CPU_DEVICE, dtype=dtype)
    if contiguous and not value.is_contiguous():
        value = value.contiguous()
    return value

# SOCKET HELPERS

def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = int(nbytes)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

def send_msg(sock: socket.socket, obj) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)

def recv_msg(sock: socket.socket):
    header = recv_exact(sock, 8)
    msg_len = struct.unpack("!Q", header)[0]
    return pickle.loads(recv_exact(sock, msg_len))

# SHARED-MEMORY HELPERS

def shm_write_tensor(mm: mmap.mmap, offset: int, tensor: torch.Tensor) -> dict:
    tensor = to_cpu_tensor(
        tensor,
        DTYPE_RPC,
        "shm_write_tensor input",
        contiguous=True,
    )
    array = tensor.numpy()
    nbytes = int(array.nbytes)
    end = int(offset) + nbytes
    if end > SHM_SIZE:
        raise RuntimeError(
            f"SHM write overflow: offset={offset}, nbytes={nbytes}, end={end}"
        )
    mm[int(offset):end] = memoryview(array).cast("B")
    return {
        "shape": tuple(int(x) for x in tensor.shape),
        "dtype": _TORCH_TO_DTYPE_NAME[tensor.dtype],
        "nbytes": nbytes,
        "offset": int(offset),
    }

class ReusableShmReadBuffers:
    def __init__(self):
        self.buffers = {}
    def get(self, offset: int, shape, dtype_name: str) -> torch.Tensor:
        key = (int(offset), tuple(int(x) for x in shape), str(dtype_name))
        if key not in self.buffers:
            torch_dtype = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
                "int32": torch.int32,
                "int64": torch.int64,
            }[dtype_name]
            self.buffers[key] = torch.empty(
                key[1],
                dtype=torch_dtype,
                device=CPU_DEVICE,
            ).contiguous()
        return self.buffers[key]

def shm_read_tensor(
    mm: mmap.mmap,
    meta: dict,
    reusable_buffers: ReusableShmReadBuffers,
) -> torch.Tensor:
    offset = int(meta["offset"])
    shape = tuple(int(x) for x in meta["shape"])
    dtype_name = str(meta["dtype"])
    nbytes = int(meta["nbytes"])
    expected = nbytes_for(shape, dtype_name)
    if expected != nbytes:
        raise RuntimeError(
            f"SHM read size mismatch: expected={expected}, got={nbytes}"
        )
    end = offset + nbytes
    if end > SHM_SIZE:
        raise RuntimeError(
            f"SHM read overflow: offset={offset}, nbytes={nbytes}, end={end}"
        )
    source = np.frombuffer(
        memoryview(mm)[offset:end],
        dtype=_DTYPE_TO_NUMPY[dtype_name],
    ).reshape(shape)
    destination = reusable_buffers.get(offset, shape, dtype_name)
    np.copyto(destination.numpy(), source, casting="no")
    if destination.dtype != DTYPE_RPC:
        destination = destination.to(dtype=DTYPE_RPC)
    return destination.contiguous()

# ATTACK-CAPTURE CONTEXT

class AttackCaptureContext:
    """
    Stores metadata for the currently executing batch.
    The metadata is attached to operator RPC requests, but prompt text is never
    placed here.
    """
    def __init__(self):
        self.capture_enabled = False
        self.run_id = None
        self.batch_index = None
        self.sample_ids = []
        self.prompt_token_lengths = []
        self.padded_prompt_length = None
        self.rpc_index = 0
    def begin_batch(
        self,
        run_id: str,
        batch_index: int,
        sample_ids,
        prompt_token_lengths,
        padded_prompt_length: int,
        capture_enabled: bool,
    ) -> None:
        self.run_id = str(run_id)
        self.batch_index = int(batch_index)
        self.sample_ids = [str(x) for x in sample_ids]
        self.prompt_token_lengths = [int(x) for x in prompt_token_lengths]
        self.padded_prompt_length = int(padded_prompt_length)
        self.capture_enabled = bool(capture_enabled)
        self.rpc_index = 0
    def end_batch(self) -> None:
        self.capture_enabled = False
        self.run_id = None
        self.batch_index = None
        self.sample_ids = []
        self.prompt_token_lengths = []
        self.padded_prompt_length = None
        self.rpc_index = 0
    def build_metadata(
        self,
        tensor: torch.Tensor,
        layer_idx,
        op_names,
    ) -> dict:
        tensor = unwrap_tensor(tensor, "AttackCaptureContext tensor")
        self.rpc_index += 1
        if tensor.ndim >= 3:
            request_batch_size = int(tensor.shape[0])
            request_seq_len = int(tensor.shape[-2])
        elif tensor.ndim == 2:
            request_batch_size = int(tensor.shape[0])
            request_seq_len = 1
        else:
            request_batch_size = 1
            request_seq_len = 1
        # Greedy HF generation performs one multi-token prefill forward followed
        # by sequence-length-1 decode forwards.
        phase = "prefill" if request_seq_len > 1 else "decode"
        if self.sample_ids and request_batch_size != len(self.sample_ids):
            raise RuntimeError(
                "RPC batch size does not match active sample IDs. "
                f"tensor_batch={request_batch_size}, ids={len(self.sample_ids)}. "
                "Use greedy decoding with num_beams=1 for this experiment."
            )
        return {
            "capture_enabled": bool(self.capture_enabled),
            "run_id": self.run_id,
            "batch_index": self.batch_index,
            "rpc_index": int(self.rpc_index),
            "sample_ids": list(self.sample_ids),
            "prompt_token_lengths": list(self.prompt_token_lengths),
            "padded_prompt_length": self.padded_prompt_length,
            "padding_side": "left",
            "phase": phase,
            "request_batch_size": request_batch_size,
            "request_seq_len": request_seq_len,
            "layer_idx": None if layer_idx is None else int(layer_idx),
            "op_names": [str(x) for x in op_names],
            "tensor_shape": tuple(int(x) for x in tensor.shape),
            "condition": "plain_t3",
            "masked": False,
        }
ATTACK_CONTEXT = AttackCaptureContext()

# GPU WORKER CLIENT

class PlainShmGPUWorkerClient:
    def __init__(
        self,
        host: str,
        port: int,
        shm_mm: mmap.mmap,
        timeout=None,
    ):
        self.host = str(host)
        self.port = int(port)
        self.mm = shm_mm
        self.read_buffers = ReusableShmReadBuffers()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if timeout is not None:
            self.sock.settimeout(float(timeout))
        print(f"Connecting to plain-T3 capture worker at {host}:{port} ...", flush=True)
        self.sock.connect((self.host, self.port))
        send_msg(self.sock, {"cmd": "ping"})
        response = recv_msg(self.sock)
        if not response.get("ok", False):
            raise RuntimeError(f"GPU worker ping failed: {response}")
        print("Connected to plain-T3 capture worker.", flush=True)
    def linear_many(self, layer_idx, op_names, clean_x):
        clean_x = to_cpu_tensor(
            clean_x,
            DTYPE_RPC,
            "PlainShmGPUWorkerClient.linear_many input",
            contiguous=True,
        )
        input_meta = shm_write_tensor(self.mm, IN_OFF, clean_x)
        output_offsets = [
            OUT_BASE_OFF + i * OUT_STRIDE
            for i in range(len(op_names))
        ]
        attack_meta = ATTACK_CONTEXT.build_metadata(
            tensor=clean_x,
            layer_idx=layer_idx,
            op_names=op_names,
        )
        request = {
            "cmd": "linear_many_shm",
            "layer_idx": layer_idx,
            "op_names": list(op_names),
            "input": input_meta,
            "out_offsets": output_offsets,
            "attack_meta": attack_meta,
        }
        send_msg(self.sock, request)
        response = recv_msg(self.sock)
        if not response.get("ok", False):
            raise RuntimeError(
                "GPU worker returned an error:\n"
                f"{response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        if response.get("fused", False):
            fused = shm_read_tensor(
                self.mm,
                response["output"],
                reusable_buffers=self.read_buffers,
            )
            return {
                "__fused__": True,
                "tensor": fused,
                "op_names": list(response["op_names"]),
                "split_sizes": tuple(int(x) for x in response["split_sizes"]),
            }
        outputs = {}
        for name, meta in response["outputs"].items():
            outputs[name] = shm_read_tensor(
                self.mm,
                meta,
                reusable_buffers=self.read_buffers,
            )
        return {
            "__fused__": False,
            "outputs": outputs,
        }
    def close(self):
        try:
            send_msg(self.sock, {"cmd": "close"})
            recv_msg(self.sock)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass

# DATA HELPERS

def build_dolly_prompt(row: dict) -> str:
    instruction = (row.get("instruction") or "").strip()
    context = (row.get("context") or "").strip()
    if context:
        return f"{instruction}\n\nContext:\n{context}"
    return instruction

def render_chat(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )

def tokenize_as_chat(tokenizer, text: str):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

def tokenize_as_chat_batch(tokenizer, texts):
    rendered = [render_chat(tokenizer, text) for text in texts]
    return tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

def make_batches(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]

def make_opaque_sample_id() -> str:
    # UUID is random and does not encode or hash the prompt.
    return uuid.uuid4().hex

def collect_samples_by_ranges(dataset, tokenizer):
    buckets = {token_range: [] for token_range in TOKEN_RANGES}
    for source_index, row in enumerate(dataset):
        text = build_dolly_prompt(row)
        encoded = tokenize_as_chat(tokenizer, text)
        seq_len = int(encoded["input_ids"].shape[1])
        if seq_len < MIN_TOKENS or seq_len > MAX_TOKENS:
            continue
        for token_range in TOKEN_RANGES:
            lower, upper = token_range
            if lower <= seq_len <= upper:
                if len(buckets[token_range]) < NUM_PROMPTS_PER_RANGE:
                    buckets[token_range].append(
                        {
                            "sample_id": make_opaque_sample_id(),
                            "source_index": int(source_index),
                            "text": text,
                            "prompt_token_length": seq_len,
                        }
                    )
                break
        if all(
            len(items) >= NUM_PROMPTS_PER_RANGE
            for items in buckets.values()
        ):
            break
    selected = []
    for token_range in TOKEN_RANGES:
        selected.extend(buckets[token_range])
    return selected, buckets

def write_protected_labels(samples) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LABELS_JSONL, "w", encoding="utf-8") as file:
        for sample in samples:
            record = {
                "sample_id": sample["sample_id"],
                "text": sample["text"],
                "prompt_token_length": sample["prompt_token_length"],
                "source_index": sample["source_index"],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved protected labels inside TDX: {LABELS_JSONL}")

# MODEL HELPERS

def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Could not locate model.model.layers")

class RPCPlainSharedGroupManager:
    """
    Plain grouped RPC for QKV and GATE/UP.
    The GPU receives the clean operator input x and returns W(x).
    """
    def __init__(
        self,
        rpc_client: PlainShmGPUWorkerClient,
        layer_idx: int,
        op_names,
        expected_output_count: int,
    ):
        self.rpc_client = rpc_client
        self.layer_idx = int(layer_idx)
        self.names = list(op_names)
        self.expected_output_count = int(expected_output_count)
        self._cached_outputs = None
        self._remaining_names = set()
    def get_output(self, name: str, x):
        x = to_cpu_tensor(
            x,
            DTYPE_RPC,
            f"RPCPlainSharedGroupManager {name} input",
            contiguous=True,
        )
        if self._cached_outputs is None:
            result = self.rpc_client.linear_many(
                layer_idx=self.layer_idx,
                op_names=self.names,
                clean_x=x,
            )
            if not result.get("__fused__", False):
                raise RuntimeError(
                    f"Grouped operation {self.names} received a non-fused response"
                )
            returned_names = list(result["op_names"])
            if returned_names != self.names:
                raise RuntimeError(
                    f"Fused operator order mismatch: "
                    f"returned={returned_names}, expected={self.names}"
                )
            split_sizes = tuple(int(x) for x in result["split_sizes"])
            if len(split_sizes) != self.expected_output_count:
                raise RuntimeError(
                    f"Unexpected fused split count for {self.names}: {split_sizes}"
                )
            fused_tensor = unwrap_tensor(
                result["tensor"],
                f"fused output {self.names}",
            ).to(
                device=CPU_DEVICE,
                dtype=DTYPE_MODEL,
            )
            parts = torch.split(fused_tensor, split_sizes, dim=-1)
            self._cached_outputs = {
                op_name: part
                for op_name, part in zip(self.names, parts)
            }
            self._remaining_names = set(self.names)
        output = self._cached_outputs[name]
        self._remaining_names.discard(name)
        if not self._remaining_names:
            self._cached_outputs = None
            self._remaining_names = set()
        return output

class RPCPlainGroupWrapper(nn.Module):
    def __init__(self, manager: RPCPlainSharedGroupManager, name: str):
        super().__init__()
        self.manager = manager
        self.name = str(name)
    def forward(self, x):
        return self.manager.get_output(self.name, x)

class RPCPlainSingleLinearWrapper(nn.Module):
    """
    Plain RPC wrapper for O projection and LM head.
    """
    def __init__(
        self,
        rpc_client: PlainShmGPUWorkerClient,
        layer_idx,
        op_name: str,
    ):
        super().__init__()
        self.rpc_client = rpc_client
        self.layer_idx = layer_idx
        self.op_name = str(op_name)
    def forward(self, x):
        x = to_cpu_tensor(
            x,
            DTYPE_RPC,
            f"RPCPlainSingleLinearWrapper {self.op_name} input",
            contiguous=True,
        )
        result = self.rpc_client.linear_many(
            layer_idx=self.layer_idx,
            op_names=[self.op_name],
            clean_x=x,
        )
        if result.get("__fused__", False):
            raise RuntimeError(
                f"Unexpected fused response for single operation {self.op_name}"
            )
        return unwrap_tensor(
            result["outputs"][self.op_name],
            f"single output {self.op_name}",
        ).to(
            device=CPU_DEVICE,
            dtype=DTYPE_MODEL,
        )

class BigBlockDecoderLayerPlainT3(nn.Module):
    """
    HF-compatible decoder layer.
    QKV, O, GATE and UP are already replaced by RPC wrappers.
    DOWN remains a normal CPU module inside TDX.
    """
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
        hidden_states = to_cpu_tensor(
            hidden_states,
            DTYPE_MODEL,
            "BigBlockDecoderLayerPlainT3 hidden_states",
            contiguous=False,
        )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_outputs = self.self_attn(
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
        if isinstance(attention_outputs, tuple):
            attention_hidden = attention_outputs[0]
        else:
            attention_hidden = attention_outputs
        attention_hidden = unwrap_tensor(
            attention_hidden,
            "attention hidden output",
        )
        hidden_states = residual + attention_hidden
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        # Required by the installed Transformers LlamaModel implementation.
        return hidden_states

def build_plain_t3_model(rpc_client: PlainShmGPUWorkerClient):
    print("Loading victim model inside TDX...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE_MODEL,
        attn_implementation=ATTN_IMPL,
    ).to(CPU_DEVICE).eval()
    model.model.embed_tokens = model.model.embed_tokens.to(
        CPU_DEVICE,
        dtype=DTYPE_MODEL,
    )
    if hasattr(model.model, "norm"):
        model.model.norm = model.model.norm.to(
            CPU_DEVICE,
            dtype=DTYPE_MODEL,
        )
    layers = get_transformer_layers(model)
    for layer_idx, layer in enumerate(layers):
        layer.input_layernorm = layer.input_layernorm.to(
            CPU_DEVICE,
            dtype=DTYPE_MODEL,
        )
        layer.self_attn = layer.self_attn.to(
            CPU_DEVICE,
            dtype=DTYPE_MODEL,
        )
        layer.post_attention_layernorm = layer.post_attention_layernorm.to(
            CPU_DEVICE,
            dtype=DTYPE_MODEL,
        )
        qkv_manager = RPCPlainSharedGroupManager(
            rpc_client=rpc_client,
            layer_idx=layer_idx,
            op_names=["q_proj", "k_proj", "v_proj"],
            expected_output_count=3,
        )
        layer.self_attn.q_proj = RPCPlainGroupWrapper(
            qkv_manager,
            "q_proj",
        )
        layer.self_attn.k_proj = RPCPlainGroupWrapper(
            qkv_manager,
            "k_proj",
        )
        layer.self_attn.v_proj = RPCPlainGroupWrapper(
            qkv_manager,
            "v_proj",
        )
        layer.self_attn.o_proj = RPCPlainSingleLinearWrapper(
            rpc_client=rpc_client,
            layer_idx=layer_idx,
            op_name="o_proj",
        )
        gate_up_manager = RPCPlainSharedGroupManager(
            rpc_client=rpc_client,
            layer_idx=layer_idx,
            op_names=["gate_proj", "up_proj"],
            expected_output_count=2,
        )
        layer.mlp.gate_proj = RPCPlainGroupWrapper(
            gate_up_manager,
            "gate_proj",
        )
        layer.mlp.up_proj = RPCPlainGroupWrapper(
            gate_up_manager,
            "up_proj",
        )
        # DOWN remains inside TDX.
        layer.mlp.down_proj = layer.mlp.down_proj.to(
            CPU_DEVICE,
            dtype=DTYPE_MODEL,
        )
    for layer_idx, layer in enumerate(layers):
        layers[layer_idx] = BigBlockDecoderLayerPlainT3(layer)
    model.lm_head = RPCPlainSingleLinearWrapper(
        rpc_client=rpc_client,
        layer_idx=None,
        op_name="lm_head",
    )
    return model.eval()

# COLLECTION RUN

def generate_one_batch(
    model,
    tokenizer,
    batch_samples,
    batch_index: int,
    capture_enabled: bool,
):
    texts = [sample["text"] for sample in batch_samples]
    sample_ids = [sample["sample_id"] for sample in batch_samples]
    prompt_lengths = [
        int(sample["prompt_token_length"])
        for sample in batch_samples
    ]
    inputs = tokenize_as_chat_batch(tokenizer, texts)
    inputs = {
        key: value.to(CPU_DEVICE)
        for key, value in inputs.items()
    }
    padded_prompt_length = int(inputs["input_ids"].shape[1])
    run_id = uuid.uuid4().hex
    ATTACK_CONTEXT.begin_batch(
        run_id=run_id,
        batch_index=batch_index,
        sample_ids=sample_ids,
        prompt_token_lengths=prompt_lengths,
        padded_prompt_length=padded_prompt_length,
        capture_enabled=capture_enabled,
    )
    try:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                min_new_tokens=MAX_NEW_TOKENS,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                # Force logits for all prompt positions during prefill so the
                # LM-head RPC receives [B,T,D], not only [B,1,D].
                logits_to_keep=0,
            )
    finally:
        ATTACK_CONTEXT.end_batch()
    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    rows = []
    for sample, generated_text in zip(batch_samples, decoded):
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "prompt_token_length": sample["prompt_token_length"],
                "generated_text": generated_text,
                "capture_enabled": bool(capture_enabled),
                "run_id": run_id,
                "batch_index": int(batch_index),
                "padded_prompt_length": padded_prompt_length,
            }
        )
    return rows

def perform_warmup(model, tokenizer, sample) -> None:
    if WARMUP_RUNS <= 0:
        return
    print("Running uncaptured warm-up...", flush=True)
    # Temporarily override the global generation length.
    text = sample["text"]
    inputs = tokenize_as_chat_batch(tokenizer, [text])
    inputs = {key: value.to(CPU_DEVICE) for key, value in inputs.items()}
    for warmup_index in range(WARMUP_RUNS):
        ATTACK_CONTEXT.begin_batch(
            run_id=f"warmup-{warmup_index}",
            batch_index=-1,
            sample_ids=[sample["sample_id"]],
            prompt_token_lengths=[sample["prompt_token_length"]],
            padded_prompt_length=int(inputs["input_ids"].shape[1]),
            capture_enabled=False,
        )
        try:
            with torch.inference_mode():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=WARMUP_MAX_NEW_TOKENS,
                    use_cache=True,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    logits_to_keep=0,
                )
        finally:
            ATTACK_CONTEXT.end_batch()
        print(f"  warm-up {warmup_index + 1}/{WARMUP_RUNS} complete")

# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-server", type=str, required=True)
    parser.add_argument("--port", type=int, default=5011)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 88)
    print("PLAIN T3 TDX CLIENT — ATTACK-CAPTURE EXPERIMENT")
    print("=" * 88)
    print(f"Model                     : {MODEL_NAME}")
    print(f"Dataset                   : {DATASET_NAME}/{DATASET_SPLIT}")
    print(f"Batch size                : {BATCH_SIZE}")
    print(f"Generated tokens          : {MAX_NEW_TOKENS}")
    print(f"Model dtype               : {DTYPE_MODEL}")
    print(f"RPC dtype                 : {DTYPE_RPC}")
    print(f"TDX SHM path              : {SHM_PATH}")
    print(f"GPU worker                : {args.gpu_server}:{args.port}")
    print(f"Protected labels          : {LABELS_JSONL}")
    print("Noise                     : disabled")
    print("Corrections               : disabled")
    print("DOWN outsourcing          : disabled")
    print("=" * 88)
    rpc_client = None
    with open(SHM_PATH, "r+b", buffering=0) as shm_file:
        shm_mm = mmap.mmap(
            shm_file.fileno(),
            SHM_SIZE,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )
        try:
            rpc_client = PlainShmGPUWorkerClient(
                host=args.gpu_server,
                port=args.port,
                shm_mm=shm_mm,
                timeout=args.timeout,
            )
            print("Loading tokenizer...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            # This is required so prompt lengths can later crop padding from the
            # left side of captured [B, T, D] tensors.
            tokenizer.padding_side = "left"
            model = build_plain_t3_model(rpc_client)
            print("Loading dataset...", flush=True)
            dataset = load_dataset(DATASET_NAME)[DATASET_SPLIT]
            print("Selecting prompts...", flush=True)
            samples, buckets = collect_samples_by_ranges(
                dataset,
                tokenizer,
            )
            if not samples:
                raise RuntimeError("No prompts matched the configured token ranges")
            print("Selected prompt counts:")
            for token_range in TOKEN_RANGES:
                print(f"  {token_range}: {len(buckets[token_range])}")
            write_protected_labels(samples)
            manifest = {
                "model_name": MODEL_NAME,
                "dataset_name": DATASET_NAME,
                "dataset_split": DATASET_SPLIT,
                "condition": "plain_t3",
                "masked": False,
                "outsourced_operators": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "lm_head",
                ],
                "down_outsourced": False,
                "batch_size": BATCH_SIZE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "padding_side": tokenizer.padding_side,
                "model_dtype": DTYPE_MODEL_NAME,
                "rpc_dtype": DTYPE_RPC_NAME,
                "token_ranges": TOKEN_RANGES,
                "num_samples": len(samples),
                "labels_path": LABELS_JSONL,
                "sample_ids": [sample["sample_id"] for sample in samples],
                "created_unix_time": time.time(),
            }
            with open(RUN_MANIFEST_JSON, "w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2)
            perform_warmup(model, tokenizer, samples[0])
            generation_rows = []
            batches = list(make_batches(samples, BATCH_SIZE))
            print("Starting captured plain-T3 runs...", flush=True)
            for batch_index, batch_samples in enumerate(batches):
                rows = generate_one_batch(
                    model=model,
                    tokenizer=tokenizer,
                    batch_samples=batch_samples,
                    batch_index=batch_index,
                    capture_enabled=True,
                )
                generation_rows.extend(rows)
                print(
                    f"  batch {batch_index + 1}/{len(batches)} complete | "
                    f"samples={len(batch_samples)} | "
                    f"IDs={[sample['sample_id'][:8] for sample in batch_samples]}"
                )
            with open(
                GENERATIONS_CSV,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                fieldnames = list(generation_rows[0].keys())
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(generation_rows)
            print("=" * 88)
            print("PLAIN-T3 COLLECTION CLIENT COMPLETED")
            print(f"Protected labels : {LABELS_JSONL}")
            print(f"Run manifest     : {RUN_MANIFEST_JSON}")
            print(f"Generation log   : {GENERATIONS_CSV}")
            print(
                "The matching host-side capture worker must save the GPU-visible "
                "prefill tensors using the sample IDs carried in attack_meta."
            )
            print("=" * 88)
        finally:
            if rpc_client is not None:
                rpc_client.close()
            try:
                shm_mm.close()
            except Exception:
                pass
if __name__ == "__main__":
    main()
