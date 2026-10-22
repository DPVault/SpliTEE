#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runs noised T3 inside TDX and requests GPU-side activation captures."""
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
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_NAME = "databricks/databricks-dolly-15k"
DATASET_SPLIT = "train"
EPS_MODE = "custom"
CUSTOM_EPSILON = 1.0
# Root produced by the matching chunked precompute.
BANK_DIR = os.path.join(
    "results/attacker_noise_banks",
    "CUSTOM1p0(b10)_1_layer0_no_lm_chunked",
)
NOISE_BANK_PATH = os.path.join(BANK_DIR, "meta.pt")
MIN_TOKENS = 100
MAX_TOKENS = 500
TOKEN_RANGES = [
    (100, 300),
]
NUM_PROMPTS_PER_RANGE = 500
BATCH_SIZE = 10
MAX_NEW_TOKENS = 1
# The matching precompute has no separate warmup bank.
WARMUP_RUNS = 0
SEED = 1234
ATTN_IMPL = "sdpa"
DTYPE_MODEL_NAME = "float32"
DTYPE_RPC_NAME = "float32"
CPU_DEVICE = torch.device("cpu")
# Only this transformer layer uses the masked outsourced path.
TARGET_LAYER = 0
# Current attacker capture point.
ATTACK_CAPTURE_LAYER = 0
ATTACK_CAPTURE_POINT = "qkv_input_after_input_layernorm"
ATTACK_CAPTURE_ENABLED = True
# IVSHMEM path visible inside TDX.
SHM_PATH = "/sys/bus/pci/devices/0000:00:03.0/resource2"
SHM_SIZE = 8 * 1024 * 1024 * 1024
IN_OFF = 0
OUT_BASE_OFF = 1024 * 1024 * 1024
OUT_STRIDE = 1024 * 1024 * 1024
OUTPUT_DIR = "results/tdx_attack_capture"
LABELS_JSONL = os.path.join(
    OUTPUT_DIR,
    "noised_t3_prompt_labels.jsonl",
)
RUN_MANIFEST_JSON = os.path.join(
    OUTPUT_DIR,
    "noised_t3_run_manifest.json",
)
GENERATIONS_CSV = os.path.join(
    OUTPUT_DIR,
    "noised_t3_generations.csv",
)
# =============================================================================
# REPRODUCIBILITY / RUNTIME SETTINGS
# =============================================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# Kept identical to the known-working plain client.
torch.set_num_threads(28)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# =============================================================================
# DTYPE HELPERS
# =============================================================================


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
    return (
        numel_from_shape(shape)
        * np.dtype(_DTYPE_TO_NUMPY[dtype_name]).itemsize
    )


def unwrap_tensor(value, where: str = "") -> torch.Tensor:
    while isinstance(value, (tuple, list)):
        if not value:
            raise RuntimeError(
                f"Empty tuple/list where tensor expected: {where}"
            )
        value = value[0]
    if not torch.is_tensor(value):
        raise RuntimeError(
            f"Expected tensor at {where}, got {type(value)}"
        )
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


def get_epsilon(seq_len: int) -> float:
    if EPS_MODE == "custom":
        return float(CUSTOM_EPSILON)
    if EPS_MODE == "ep1":
        return 1.0
    if EPS_MODE == "sqrt":
        return 1.0 / float(seq_len) ** 0.5
    raise ValueError("EPS_MODE must be custom, ep1, or sqrt")
# =============================================================================
# SOCKET HELPERS
# =============================================================================


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = int(nbytes)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(
                "Socket closed while receiving data"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_msg(sock: socket.socket, obj) -> None:
    payload = pickle.dumps(
        obj,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    sock.sendall(
        struct.pack("!Q", len(payload))
        + payload
    )


def recv_msg(sock: socket.socket):
    header = recv_exact(sock, 8)
    msg_len = struct.unpack("!Q", header)[0]
    return pickle.loads(
        recv_exact(sock, msg_len)
    )
# =============================================================================
# SHARED-MEMORY HELPERS
# =============================================================================


def shm_write_tensor(
    mm: mmap.mmap,
    offset: int,
    tensor: torch.Tensor,
) -> dict:
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
            f"SHM write overflow: offset={offset}, "
            f"nbytes={nbytes}, end={end}"
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
    def get(
        self,
        offset: int,
        shape,
        dtype_name: str,
    ) -> torch.Tensor:
        key = (
            int(offset),
            tuple(int(x) for x in shape),
            str(dtype_name),
        )
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
            f"SHM read size mismatch: "
            f"expected={expected}, got={nbytes}"
        )
    end = offset + nbytes
    if end > SHM_SIZE:
        raise RuntimeError(
            f"SHM read overflow: offset={offset}, "
            f"nbytes={nbytes}, end={end}"
        )
    source = np.frombuffer(
        memoryview(mm)[offset:end],
        dtype=_DTYPE_TO_NUMPY[dtype_name],
    ).reshape(shape)
    destination = reusable_buffers.get(
        offset,
        shape,
        dtype_name,
    )
    np.copyto(
        destination.numpy(),
        source,
        casting="no",
    )
    if destination.dtype != DTYPE_RPC:
        destination = destination.to(
            dtype=DTYPE_RPC
        )
    return destination.contiguous()
# =============================================================================
# ATTACK-CAPTURE CONTEXT
# =============================================================================


class AttackCaptureContext:
    """
    Same metadata convention as the known-working plain client.
    Only the condition/masked fields differ.
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
        self.sample_ids = [
            str(x) for x in sample_ids
        ]
        self.prompt_token_lengths = [
            int(x) for x in prompt_token_lengths
        ]
        self.padded_prompt_length = int(
            padded_prompt_length
        )
        self.capture_enabled = bool(
            capture_enabled
        )
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
        tensor = unwrap_tensor(
            tensor,
            "AttackCaptureContext tensor",
        )
        self.rpc_index += 1
        if tensor.ndim >= 3:
            request_batch_size = int(
                tensor.shape[0]
            )
            request_seq_len = int(
                tensor.shape[-2]
            )
        elif tensor.ndim == 2:
            request_batch_size = int(
                tensor.shape[0]
            )
            request_seq_len = 1
        else:
            request_batch_size = 1
            request_seq_len = 1
        phase = (
            "prefill"
            if request_seq_len > 1
            else "decode"
        )
        if (
            self.sample_ids
            and request_batch_size
            != len(self.sample_ids)
        ):
            raise RuntimeError(
                "RPC batch size does not match active "
                "sample IDs. "
                f"tensor_batch={request_batch_size}, "
                f"ids={len(self.sample_ids)}"
            )
        return {
            "capture_enabled": bool(
                self.capture_enabled
            ),
            "run_id": self.run_id,
            "batch_index": self.batch_index,
            "rpc_index": int(
                self.rpc_index
            ),
            "sample_ids": list(
                self.sample_ids
            ),
            "prompt_token_lengths": list(
                self.prompt_token_lengths
            ),
            "padded_prompt_length": (
                self.padded_prompt_length
            ),
            "padding_side": "left",
            "phase": phase,
            "request_batch_size": (
                request_batch_size
            ),
            "request_seq_len": (
                request_seq_len
            ),
            "layer_idx": (
                None
                if layer_idx is None
                else int(layer_idx)
            ),
            "op_names": [
                str(x) for x in op_names
            ],
            "tensor_shape": tuple(
                int(x) for x in tensor.shape
            ),
            "condition": "noised_t3",
            "masked": True,
            "eps_mode": EPS_MODE,
            "custom_epsilon": (
                float(CUSTOM_EPSILON)
                if EPS_MODE == "custom"
                else None
            ),
        }
ATTACK_CONTEXT = AttackCaptureContext()
# =============================================================================
# GPU WORKER CLIENT
# =============================================================================


class NoisedShmGPUWorkerClient:
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
        self.read_buffers = (
            ReusableShmReadBuffers()
        )
        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        self.sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )
        if timeout is not None:
            self.sock.settimeout(
                float(timeout)
            )
        print(
            f"Connecting to noised-T3 capture worker "
            f"at {host}:{port} ...",
            flush=True,
        )
        self.sock.connect(
            (self.host, self.port)
        )
        send_msg(
            self.sock,
            {"cmd": "ping"},
        )
        response = recv_msg(
            self.sock
        )
        if not response.get("ok", False):
            raise RuntimeError(
                f"GPU worker ping failed: {response}"
            )
        # New switchable worker advertises its mode.
        worker_mode = response.get(
            "capture_mode"
        )
        if (
            worker_mode is not None
            and str(worker_mode).lower()
            != "noised"
        ):
            raise RuntimeError(
                "GPU worker is not in noised mode. "
                "Set CAPTURE_MODE = 'noised' "
                "inside the worker."
            )
        print(
            "Connected to noised-T3 capture worker.",
            flush=True,
        )
    def linear_many(
        self,
        layer_idx,
        op_names,
        gpu_visible_x,
    ):
        gpu_visible_x = to_cpu_tensor(
            gpu_visible_x,
            DTYPE_RPC,
            "NoisedShmGPUWorkerClient.linear_many input",
            contiguous=True,
        )
        input_meta = shm_write_tensor(
            self.mm,
            IN_OFF,
            gpu_visible_x,
        )
        output_offsets = [
            OUT_BASE_OFF
            + i * OUT_STRIDE
            for i in range(
                len(op_names)
            )
        ]
        attack_meta = (
            ATTACK_CONTEXT.build_metadata(
                tensor=gpu_visible_x,
                layer_idx=layer_idx,
                op_names=op_names,
            )
        )
        request = {
            "cmd": "linear_many_shm",
            "layer_idx": layer_idx,
            "op_names": list(op_names),
            "input": input_meta,
            "out_offsets": output_offsets,
            "attack_meta": attack_meta,
        }
        send_msg(
            self.sock,
            request,
        )
        response = recv_msg(
            self.sock
        )
        if not response.get(
            "ok",
            False,
        ):
            raise RuntimeError(
                "GPU worker returned an error:\n"
                f"{response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        if response.get(
            "fused",
            False,
        ):
            fused = shm_read_tensor(
                self.mm,
                response["output"],
                reusable_buffers=(
                    self.read_buffers
                ),
            )
            return {
                "__fused__": True,
                "tensor": fused,
                "op_names": list(
                    response["op_names"]
                ),
                "split_sizes": tuple(
                    int(x)
                    for x
                    in response[
                        "split_sizes"
                    ]
                ),
            }
        outputs = {}
        for name, meta in (
            response["outputs"].items()
        ):
            outputs[name] = shm_read_tensor(
                self.mm,
                meta,
                reusable_buffers=(
                    self.read_buffers
                ),
            )
        return {
            "__fused__": False,
            "outputs": outputs,
        }
    def close(self):
        try:
            send_msg(
                self.sock,
                {"cmd": "close"},
            )
            recv_msg(
                self.sock
            )
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
# =============================================================================
# BANK LOADING
# =============================================================================


class LoadedBank:
    def __init__(
        self,
        source_path,
        fuse_corrections=False,
        expected_corr_count=None,
    ):
        self.source_path = str(
            source_path
        )
        self.fuse_corrections = bool(
            fuse_corrections
        )
        self.expected_corr_count = (
            None
            if expected_corr_count is None
            else int(expected_corr_count)
        )
        self.bank = defaultdict(deque)
        self.loaded = False
    def load(self):
        if self.loaded:
            return
        serialized = torch.load(
            self.source_path,
            map_location="cpu",
        )
        for shape, entry in (
            serialized.items()
        ):
            shape = tuple(shape)
            converted = []
            for eta, corrs in (
                entry["items"]
            ):
                eta = (
                    unwrap_tensor(
                        eta,
                        "bank eta",
                    )
                    .to(
                        device=CPU_DEVICE,
                        dtype=DTYPE_RPC,
                    )
                    .contiguous()
                )
                corr_list = [
                    unwrap_tensor(
                        corr,
                        "bank correction",
                    )
                    .to(
                        device=CPU_DEVICE,
                        dtype=DTYPE_RPC,
                    )
                    .contiguous()
                    for corr in corrs
                ]
                if (
                    self.expected_corr_count
                    is not None
                    and len(corr_list)
                    != self.expected_corr_count
                ):
                    raise RuntimeError(
                        "Correction count mismatch "
                        f"for shape {shape}: "
                        f"got={len(corr_list)}, "
                        f"expected="
                        f"{self.expected_corr_count}"
                    )
                if self.fuse_corrections:
                    corr_value = torch.cat(
                        corr_list,
                        dim=-1,
                    ).contiguous()
                else:
                    corr_value = corr_list
                converted.append(
                    (
                        eta,
                        corr_value,
                    )
                )
            self.bank[shape] = deque(
                converted
            )
        self.loaded = True
    def get(self, shape):
        self.load()
        shape = tuple(shape)
        if not self.bank[shape]:
            raise RuntimeError(
                f"Noise bank empty for shape {shape}"
            )
        return self.bank[shape].popleft()


def resolve_chunk_meta_paths():
    if not os.path.isfile(
        NOISE_BANK_PATH
    ):
        raise FileNotFoundError(
            f"Noise-bank root meta not found: "
            f"{NOISE_BANK_PATH}"
        )
    root_payload = torch.load(
        NOISE_BANK_PATH,
        map_location="cpu",
    )
    stored_paths = list(
        root_payload.get(
            "batch_meta_paths",
            [],
        )
    )
    if not stored_paths:
        return [
            NOISE_BANK_PATH
        ]
    resolved = []
    for index, stored_path in enumerate(
        stored_paths
    ):
        stored_path = str(
            stored_path
        )
        if os.path.exists(
            stored_path
        ):
            resolved.append(
                stored_path
            )
            continue
        fallback = os.path.join(
            BANK_DIR,
            f"batch_{index:03d}",
            "meta.pt",
        )
        if not os.path.exists(
            fallback
        ):
            raise FileNotFoundError(
                "Could not resolve bank chunk. "
                f"stored={stored_path}, "
                f"fallback={fallback}"
            )
        resolved.append(
            fallback
        )
    return resolved


def validate_chunk_payload(
    payload,
):
    meta = payload.get(
        "meta",
        {},
    )
    if meta.get(
        "model_name"
    ) != MODEL_NAME:
        raise ValueError(
            "Bank/model mismatch: "
            f"{meta.get('model_name')} "
            f"!= {MODEL_NAME}"
        )
    if int(
        meta.get(
            "batch_size",
            BATCH_SIZE,
        )
    ) != int(BATCH_SIZE):
        raise ValueError(
            "Bank/runtime batch-size mismatch"
        )
    layers = [
        int(x)
        for x in meta.get(
            "precomputed_layer_indices",
            [],
        )
    ]
    if int(
        TARGET_LAYER
    ) not in layers:
        raise ValueError(
            f"Bank does not contain "
            f"TARGET_LAYER={TARGET_LAYER}; "
            f"bank layers={layers}"
        )
    if bool(
        meta.get(
            "include_lm_head",
            False,
        )
    ):
        # Harmless if present, but current runtime does not use it.
        pass
    if meta.get(
        "eps_mode"
    ) != EPS_MODE:
        raise ValueError(
            "Bank/runtime EPS mode mismatch"
        )
    if EPS_MODE == "custom":
        bank_eps = float(
            meta.get(
                "custom_epsilon"
            )
        )
        if abs(
            bank_eps
            - float(
                CUSTOM_EPSILON
            )
        ) > 1e-12:
            raise ValueError(
                "Bank/runtime custom epsilon mismatch"
            )


def get_layer_bank_entry(
    payload,
    layer_idx,
):
    for entry in payload.get(
        "layer_bank_paths",
        [],
    ):
        if int(
            entry["layer_idx"]
        ) == int(
            layer_idx
        ):
            return entry
    raise RuntimeError(
        f"Chunk has no bank entry "
        f"for layer {layer_idx}"
    )
# =============================================================================
# DATA HELPERS
# =============================================================================


def build_dolly_prompt(
    row: dict,
) -> str:
    instruction = (
        row.get("instruction")
        or ""
    ).strip()
    context = (
        row.get("context")
        or ""
    ).strip()
    if context:
        return (
            f"{instruction}\n\n"
            f"Context:\n{context}"
        )
    return instruction


def render_chat(
    tokenizer,
    text: str,
) -> str:
    return (
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": text,
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
    )


def tokenize_as_chat(
    tokenizer,
    text: str,
):
    return (
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": text,
                }
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    )


def tokenize_as_chat_batch(
    tokenizer,
    texts,
):
    rendered = [
        render_chat(
            tokenizer,
            text,
        )
        for text in texts
    ]
    return tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )


def make_batches(
    items,
    batch_size: int,
):
    for start in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[
            start:
            start + batch_size
        ]


def make_opaque_sample_id() -> str:
    return uuid.uuid4().hex


def collect_samples_by_ranges(
    dataset,
    tokenizer,
):
    buckets = {
        token_range: []
        for token_range
        in TOKEN_RANGES
    }
    for source_index, row in enumerate(
        dataset
    ):
        text = build_dolly_prompt(
            row
        )
        encoded = tokenize_as_chat(
            tokenizer,
            text,
        )
        seq_len = int(
            encoded[
                "input_ids"
            ].shape[1]
        )
        if (
            seq_len < MIN_TOKENS
            or seq_len > MAX_TOKENS
        ):
            continue
        for token_range in TOKEN_RANGES:
            lower, upper = token_range
            if (
                lower
                <= seq_len
                <= upper
            ):
                if (
                    len(
                        buckets[
                            token_range
                        ]
                    )
                    < NUM_PROMPTS_PER_RANGE
                ):
                    buckets[
                        token_range
                    ].append(
                        {
                            "sample_id": (
                                make_opaque_sample_id()
                            ),
                            "source_index": int(
                                source_index
                            ),
                            "text": text,
                            "prompt_token_length": (
                                seq_len
                            ),
                        }
                    )
                break
        if all(
            len(items)
            >= NUM_PROMPTS_PER_RANGE
            for items
            in buckets.values()
        ):
            break
    selected = []
    for token_range in TOKEN_RANGES:
        selected.extend(
            buckets[
                token_range
            ]
        )
    return selected, buckets


def write_protected_labels(
    samples,
) -> None:
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )
    with open(
        LABELS_JSONL,
        "w",
        encoding="utf-8",
    ) as file:
        for sample in samples:
            record = {
                "sample_id": (
                    sample[
                        "sample_id"
                    ]
                ),
                "text": sample[
                    "text"
                ],
                "prompt_token_length": (
                    sample[
                        "prompt_token_length"
                    ]
                ),
                "source_index": (
                    sample[
                        "source_index"
                    ]
                ),
            }
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        "Saved protected labels inside TDX: "
        f"{LABELS_JSONL}",
        flush=True,
    )
# =============================================================================
# MODEL HELPERS
# =============================================================================


def get_transformer_layers(
    model,
):
    if (
        hasattr(
            model,
            "model",
        )
        and hasattr(
            model.model,
            "layers",
        )
    ):
        return model.model.layers
    raise ValueError(
        "Could not locate model.model.layers"
    )


class RPCMaskedSharedGroupManager:
    """
    Layer-0 QKV or GATE/UP:
        clean x
        masked x + eta -> GPU
        GPU returns W(x + eta)
        TDX subtracts stored W(eta)
    """
    def __init__(
        self,
        rpc_client,
        layer_idx,
        op_names,
        expected_output_count,
        bank,
    ):
        self.rpc_client = (
            rpc_client
        )
        self.layer_idx = int(
            layer_idx
        )
        self.names = list(
            op_names
        )
        self.expected_output_count = int(
            expected_output_count
        )
        self.bank = bank
        self._cached_outputs = None
        self._remaining_names = set()
    def replace_bank(
        self,
        bank,
    ):
        self.bank = bank
        self._cached_outputs = None
        self._remaining_names = set()
    def get_output(
        self,
        name: str,
        x,
    ):
        x = to_cpu_tensor(
            x,
            DTYPE_RPC,
            f"RPCMaskedSharedGroupManager "
            f"{name} input",
            contiguous=True,
        )
        if self._cached_outputs is None:
            eta, fused_corr = (
                self.bank.get(
                    tuple(
                        x.shape
                    )
                )
            )
            x_masked = (
                x + eta
            )
            result = (
                self.rpc_client.linear_many(
                    layer_idx=(
                        self.layer_idx
                    ),
                    op_names=(
                        self.names
                    ),
                    gpu_visible_x=(
                        x_masked
                    ),
                )
            )
            if not result.get(
                "__fused__",
                False,
            ):
                raise RuntimeError(
                    "Grouped masked operation "
                    f"{self.names} received "
                    "a non-fused response"
                )
            returned_names = list(
                result[
                    "op_names"
                ]
            )
            if (
                returned_names
                != self.names
            ):
                raise RuntimeError(
                    "Fused operator order mismatch: "
                    f"returned={returned_names}, "
                    f"expected={self.names}"
                )
            split_sizes = tuple(
                int(v)
                for v
                in result[
                    "split_sizes"
                ]
            )
            if (
                len(split_sizes)
                != self.expected_output_count
            ):
                raise RuntimeError(
                    "Unexpected fused split count "
                    f"for {self.names}: "
                    f"{split_sizes}"
                )
            fused_raw = unwrap_tensor(
                result["tensor"],
                f"fused masked output "
                f"{self.names}",
            ).to(
                device=CPU_DEVICE,
                dtype=DTYPE_RPC,
            )
            fused_corr = unwrap_tensor(
                fused_corr,
                "fused correction",
            ).to(
                device=CPU_DEVICE,
                dtype=DTYPE_RPC,
            )
            if tuple(
                fused_raw.shape
            ) != tuple(
                fused_corr.shape
            ):
                raise RuntimeError(
                    "Fused GPU output/correction "
                    "shape mismatch: "
                    f"raw={tuple(fused_raw.shape)}, "
                    f"corr={tuple(fused_corr.shape)}"
                )
            clean_fused = (
                fused_raw
                - fused_corr
            )
            parts = torch.split(
                clean_fused,
                split_sizes,
                dim=-1,
            )
            self._cached_outputs = {
                op_name: part.to(
                    dtype=DTYPE_MODEL
                )
                for op_name, part
                in zip(
                    self.names,
                    parts,
                )
            }
            self._remaining_names = set(
                self.names
            )
        output = (
            self._cached_outputs[
                name
            ]
        )
        self._remaining_names.discard(
            name
        )
        if not self._remaining_names:
            self._cached_outputs = None
            self._remaining_names = set()
        return output


class RPCMaskedGroupWrapper(
    nn.Module
):
    def __init__(
        self,
        manager,
        name: str,
    ):
        super().__init__()
        self.manager = manager
        self.name = str(name)
    def forward(
        self,
        x,
    ):
        return self.manager.get_output(
            self.name,
            x,
        )


class RPCMaskedSingleLinearWrapper(
    nn.Module
):
    """
    Used for masked Layer-0 O projection.
    """
    def __init__(
        self,
        rpc_client,
        layer_idx,
        op_name: str,
        bank,
    ):
        super().__init__()
        self.rpc_client = rpc_client
        self.layer_idx = layer_idx
        self.op_name = str(
            op_name
        )
        self.bank = bank
    def replace_bank(
        self,
        bank,
    ):
        self.bank = bank
    def forward(
        self,
        x,
    ):
        x = to_cpu_tensor(
            x,
            DTYPE_RPC,
            f"RPCMaskedSingleLinearWrapper "
            f"{self.op_name} input",
            contiguous=True,
        )
        eta, corrs = self.bank.get(
            tuple(
                x.shape
            )
        )
        if len(corrs) != 1:
            raise RuntimeError(
                f"{self.op_name} expected "
                "one correction, got "
                f"{len(corrs)}"
            )
        x_masked = (
            x + eta
        )
        result = (
            self.rpc_client.linear_many(
                layer_idx=(
                    self.layer_idx
                ),
                op_names=[
                    self.op_name
                ],
                gpu_visible_x=(
                    x_masked
                ),
            )
        )
        if result.get(
            "__fused__",
            False,
        ):
            raise RuntimeError(
                "Unexpected fused response "
                "for single operation "
                f"{self.op_name}"
            )
        raw = unwrap_tensor(
            result[
                "outputs"
            ][
                self.op_name
            ],
            f"masked raw output "
            f"{self.op_name}",
        ).to(
            device=CPU_DEVICE,
            dtype=DTYPE_RPC,
        )
        corr = unwrap_tensor(
            corrs[0],
            f"correction "
            f"{self.op_name}",
        ).to(
            device=CPU_DEVICE,
            dtype=DTYPE_RPC,
        )
        clean = (
            raw - corr
        )
        return clean.to(
            dtype=DTYPE_MODEL
        )


class BigBlockDecoderLayerNoisedT3(
    nn.Module
):
    """
    Same decoder-layer structure as the working plain client.
    Layer 0 contains RPC masking wrappers; other layers remain local.
    """
    def __init__(
        self,
        base_layer,
    ):
        super().__init__()
        self.input_layernorm = (
            base_layer.input_layernorm
        )
        self.self_attn = (
            base_layer.self_attn
        )
        self.post_attention_layernorm = (
            base_layer.post_attention_layernorm
        )
        self.mlp = (
            base_layer.mlp
        )
        self._qkv_masked_manager = getattr(
            base_layer,
            "_qkv_masked_manager",
            None,
        )
        self._o_masked_wrapper = getattr(
            base_layer,
            "_o_masked_wrapper",
            None,
        )
        self._gate_up_masked_manager = getattr(
            base_layer,
            "_gate_up_masked_manager",
            None,
        )
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
            "BigBlockDecoderLayerNoisedT3 "
            "hidden_states",
            contiguous=False,
        )
        residual = hidden_states
        hidden_states = (
            self.input_layernorm(
                hidden_states
            )
        )
        attention_outputs = (
            self.self_attn(
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
        )
        if isinstance(
            attention_outputs,
            tuple,
        ):
            attention_hidden = (
                attention_outputs[0]
            )
        else:
            attention_hidden = (
                attention_outputs
            )
        attention_hidden = unwrap_tensor(
            attention_hidden,
            "attention hidden output",
        )
        hidden_states = (
            residual
            + attention_hidden
        )
        residual = hidden_states
        hidden_states = (
            self.post_attention_layernorm(
                hidden_states
            )
        )
        hidden_states = (
            self.mlp(
                hidden_states
            )
        )
        hidden_states = (
            residual
            + hidden_states
        )
        return hidden_states


def make_layer_banks(
    chunk_payload,
):
    validate_chunk_payload(
        chunk_payload
    )
    entry = get_layer_bank_entry(
        chunk_payload,
        TARGET_LAYER,
    )
    qkv_bank = LoadedBank(
        entry[
            "qkv_bank_path"
        ],
        fuse_corrections=True,
        expected_corr_count=3,
    )
    o_bank = LoadedBank(
        entry[
            "o_bank_path"
        ],
        fuse_corrections=False,
        expected_corr_count=1,
    )
    gate_up_bank = LoadedBank(
        entry[
            "gate_up_bank_path"
        ],
        fuse_corrections=True,
        expected_corr_count=2,
    )
    return (
        qkv_bank,
        o_bank,
        gate_up_bank,
    )


def build_noised_t3_model(
    rpc_client,
    first_chunk_payload,
):
    print(
        "Loading victim model inside TDX...",
        flush=True,
    )
    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE_MODEL,
            attn_implementation=(
                ATTN_IMPL
            ),
        )
        .to(
            CPU_DEVICE
        )
        .eval()
    )
    model.model.embed_tokens = (
        model.model.embed_tokens.to(
            CPU_DEVICE,
            dtype=DTYPE_MODEL,
        )
    )
    if hasattr(
        model.model,
        "norm",
    ):
        model.model.norm = (
            model.model.norm.to(
                CPU_DEVICE,
                dtype=DTYPE_MODEL,
            )
        )
    layers = get_transformer_layers(
        model
    )
    if (
        TARGET_LAYER < 0
        or TARGET_LAYER
        >= len(layers)
    ):
        raise ValueError(
            f"TARGET_LAYER={TARGET_LAYER} "
            "is out of range"
        )
    (
        qkv_bank,
        o_bank,
        gate_up_bank,
    ) = make_layer_banks(
        first_chunk_payload
    )
    for layer_idx, layer in enumerate(
        layers
    ):
        layer.input_layernorm = (
            layer.input_layernorm.to(
                CPU_DEVICE,
                dtype=DTYPE_MODEL,
            )
        )
        layer.self_attn = (
            layer.self_attn.to(
                CPU_DEVICE,
                dtype=DTYPE_MODEL,
            )
        )
        layer.post_attention_layernorm = (
            layer.post_attention_layernorm.to(
                CPU_DEVICE,
                dtype=DTYPE_MODEL,
            )
        )
        layer.mlp = (
            layer.mlp.to(
                CPU_DEVICE,
                dtype=DTYPE_MODEL,
            )
        )
        layer._qkv_masked_manager = None
        layer._o_masked_wrapper = None
        layer._gate_up_masked_manager = None
        if (
            layer_idx
            != TARGET_LAYER
        ):
            continue
        qkv_manager = (
            RPCMaskedSharedGroupManager(
                rpc_client=rpc_client,
                layer_idx=layer_idx,
                op_names=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                ],
                expected_output_count=3,
                bank=qkv_bank,
            )
        )
        layer.self_attn.q_proj = (
            RPCMaskedGroupWrapper(
                qkv_manager,
                "q_proj",
            )
        )
        layer.self_attn.k_proj = (
            RPCMaskedGroupWrapper(
                qkv_manager,
                "k_proj",
            )
        )
        layer.self_attn.v_proj = (
            RPCMaskedGroupWrapper(
                qkv_manager,
                "v_proj",
            )
        )
        o_wrapper = (
            RPCMaskedSingleLinearWrapper(
                rpc_client=rpc_client,
                layer_idx=layer_idx,
                op_name="o_proj",
                bank=o_bank,
            )
        )
        layer.self_attn.o_proj = (
            o_wrapper
        )
        gate_up_manager = (
            RPCMaskedSharedGroupManager(
                rpc_client=rpc_client,
                layer_idx=layer_idx,
                op_names=[
                    "gate_proj",
                    "up_proj",
                ],
                expected_output_count=2,
                bank=gate_up_bank,
            )
        )
        layer.mlp.gate_proj = (
            RPCMaskedGroupWrapper(
                gate_up_manager,
                "gate_proj",
            )
        )
        layer.mlp.up_proj = (
            RPCMaskedGroupWrapper(
                gate_up_manager,
                "up_proj",
            )
        )
        # DOWN remains local.
        layer.mlp.down_proj = (
            layer.mlp.down_proj.to(
                CPU_DEVICE,
                dtype=DTYPE_MODEL,
            )
        )
        layer._qkv_masked_manager = (
            qkv_manager
        )
        layer._o_masked_wrapper = (
            o_wrapper
        )
        layer._gate_up_masked_manager = (
            gate_up_manager
        )
    for layer_idx, layer in enumerate(
        layers
    ):
        layers[layer_idx] = (
            BigBlockDecoderLayerNoisedT3(
                layer
            )
        )
    # LM head remains local for this attacker collection.
    model.lm_head = (
        model.lm_head.to(
            CPU_DEVICE,
            dtype=DTYPE_MODEL,
        )
    )
    return model.eval()


def load_chunk_into_model(
    model,
    chunk_payload,
):
    (
        qkv_bank,
        o_bank,
        gate_up_bank,
    ) = make_layer_banks(
        chunk_payload
    )
    layer = get_transformer_layers(
        model
    )[TARGET_LAYER]
    if (
        layer._qkv_masked_manager
        is None
        or layer._o_masked_wrapper
        is None
        or layer._gate_up_masked_manager
        is None
    ):
        raise RuntimeError(
            "Target layer does not contain "
            "the expected masked wrappers"
        )
    layer._qkv_masked_manager.replace_bank(
        qkv_bank
    )
    layer._o_masked_wrapper.replace_bank(
        o_bank
    )
    layer._gate_up_masked_manager.replace_bank(
        gate_up_bank
    )
# =============================================================================
# COLLECTION RUN
# =============================================================================


def generate_one_batch(
    model,
    tokenizer,
    batch_samples,
    batch_index: int,
    capture_enabled: bool,
):
    texts = [
        sample["text"]
        for sample
        in batch_samples
    ]
    sample_ids = [
        sample["sample_id"]
        for sample
        in batch_samples
    ]
    prompt_lengths = [
        int(
            sample[
                "prompt_token_length"
            ]
        )
        for sample
        in batch_samples
    ]
    inputs = tokenize_as_chat_batch(
        tokenizer,
        texts,
    )
    inputs = {
        key: value.to(
            CPU_DEVICE
        )
        for key, value
        in inputs.items()
    }
    padded_prompt_length = int(
        inputs[
            "input_ids"
        ].shape[1]
    )
    run_id = uuid.uuid4().hex
    ATTACK_CONTEXT.begin_batch(
        run_id=run_id,
        batch_index=batch_index,
        sample_ids=sample_ids,
        prompt_token_lengths=(
            prompt_lengths
        ),
        padded_prompt_length=(
            padded_prompt_length
        ),
        capture_enabled=(
            capture_enabled
        ),
    )
    try:
        with torch.inference_mode():
            generated_ids = (
                model.generate(
                    **inputs,
                    min_new_tokens=(
                        MAX_NEW_TOKENS
                    ),
                    max_new_tokens=(
                        MAX_NEW_TOKENS
                    ),
                    use_cache=True,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=(
                        tokenizer.pad_token_id
                    ),
                )
            )
    finally:
        ATTACK_CONTEXT.end_batch()
    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    rows = []
    for sample, generated_text in zip(
        batch_samples,
        decoded,
    ):
        rows.append(
            {
                "sample_id": (
                    sample[
                        "sample_id"
                    ]
                ),
                "prompt_token_length": (
                    sample[
                        "prompt_token_length"
                    ]
                ),
                "generated_text": (
                    generated_text
                ),
                "capture_enabled": bool(
                    capture_enabled
                ),
                "run_id": run_id,
                "batch_index": int(
                    batch_index
                ),
                "padded_prompt_length": (
                    padded_prompt_length
                ),
                "condition": "noised_t3",
                "epsilon": get_epsilon(
                    sample[
                        "prompt_token_length"
                    ]
                ),
            }
        )
    return rows
# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-server",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5010,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
    )
    args = parser.parse_args()
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )
    print("=" * 88)
    print(
        "NOISED T3 TDX CLIENT — "
        "ATTACK-CAPTURE EXPERIMENT"
    )
    print("=" * 88)
    print(
        f"Model                     : "
        f"{MODEL_NAME}"
    )
    print(
        f"Dataset                   : "
        f"{DATASET_NAME}/"
        f"{DATASET_SPLIT}"
    )
    print(
        f"Batch size                : "
        f"{BATCH_SIZE}"
    )
    print(
        f"Generated tokens          : "
        f"{MAX_NEW_TOKENS}"
    )
    print(
        f"Target noised layer       : "
        f"{TARGET_LAYER}"
    )
    print(
        f"Capture point             : "
        f"{ATTACK_CAPTURE_POINT}"
    )
    print(
        f"Epsilon mode              : "
        f"{EPS_MODE}"
    )
    print(
        f"Custom epsilon            : "
        f"{CUSTOM_EPSILON}"
    )
    print(
        f"Noise-bank root           : "
        f"{NOISE_BANK_PATH}"
    )
    print(
        f"TDX SHM path              : "
        f"{SHM_PATH}"
    )
    print(
        f"GPU worker                : "
        f"{args.gpu_server}:"
        f"{args.port}"
    )
    print(
        "Layer 0 QKV/O/GATE-UP     : "
        "masked + outsourced"
    )
    print(
        "Layers 1+                 : "
        "local inside TDX"
    )
    print(
        "DOWN                      : "
        "local inside TDX"
    )
    print(
        "LM head                   : "
        "local inside TDX"
    )
    print("=" * 88)
    if (
        ATTACK_CAPTURE_LAYER
        != TARGET_LAYER
    ):
        raise ValueError(
            "ATTACK_CAPTURE_LAYER must "
            "match TARGET_LAYER"
        )
    chunk_meta_paths = (
        resolve_chunk_meta_paths()
    )
    expected_batches = (
        len(TOKEN_RANGES)
        * NUM_PROMPTS_PER_RANGE
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE
    if (
        len(chunk_meta_paths)
        < expected_batches
    ):
        raise RuntimeError(
            f"Bank has only "
            f"{len(chunk_meta_paths)} chunks, "
            f"but runtime expects "
            f"{expected_batches}"
        )
    print(
        f"Bank chunks available     : "
        f"{len(chunk_meta_paths)}",
        flush=True,
    )
    rpc_client = None
    with open(
        SHM_PATH,
        "r+b",
        buffering=0,
    ) as shm_file:
        shm_mm = mmap.mmap(
            shm_file.fileno(),
            SHM_SIZE,
            mmap.MAP_SHARED,
            mmap.PROT_READ
            | mmap.PROT_WRITE,
        )
        try:
            rpc_client = (
                NoisedShmGPUWorkerClient(
                    host=args.gpu_server,
                    port=args.port,
                    shm_mm=shm_mm,
                    timeout=args.timeout,
                )
            )
            print(
                "Loading tokenizer...",
                flush=True,
            )
            tokenizer = (
                AutoTokenizer
                .from_pretrained(
                    MODEL_NAME
                )
            )
            if (
                tokenizer.pad_token_id
                is None
            ):
                tokenizer.pad_token = (
                    tokenizer.eos_token
                )
            tokenizer.padding_side = (
                "left"
            )
            first_chunk_payload = (
                torch.load(
                    chunk_meta_paths[0],
                    map_location="cpu",
                )
            )
            model = (
                build_noised_t3_model(
                    rpc_client,
                    first_chunk_payload,
                )
            )
            print(
                "Loading dataset...",
                flush=True,
            )
            dataset = load_dataset(
                DATASET_NAME
            )[DATASET_SPLIT]
            print(
                "Selecting prompts...",
                flush=True,
            )
            samples, buckets = (
                collect_samples_by_ranges(
                    dataset,
                    tokenizer,
                )
            )
            if not samples:
                raise RuntimeError(
                    "No prompts matched "
                    "the configured token ranges"
                )
            print(
                "Selected prompt counts:"
            )
            for token_range in (
                TOKEN_RANGES
            ):
                print(
                    f"  {token_range}: "
                    f"{len(buckets[token_range])}"
                )
            write_protected_labels(
                samples
            )
            manifest = {
                "model_name": MODEL_NAME,
                "dataset_name": DATASET_NAME,
                "dataset_split": (
                    DATASET_SPLIT
                ),
                "condition": "noised_t3",
                "masked": True,
                "eps_mode": EPS_MODE,
                "custom_epsilon": (
                    CUSTOM_EPSILON
                ),
                "target_layer": (
                    TARGET_LAYER
                ),
                "capture_layer": (
                    ATTACK_CAPTURE_LAYER
                ),
                "capture_point": (
                    ATTACK_CAPTURE_POINT
                ),
                "outsourced_operators": [
                    "layer0_q_proj",
                    "layer0_k_proj",
                    "layer0_v_proj",
                    "layer0_o_proj",
                    "layer0_gate_proj",
                    "layer0_up_proj",
                ],
                "down_outsourced": False,
                "lm_head_outsourced": False,
                "other_layers_outsourced": (
                    False
                ),
                "batch_size": BATCH_SIZE,
                "max_new_tokens": (
                    MAX_NEW_TOKENS
                ),
                "padding_side": (
                    tokenizer.padding_side
                ),
                "model_dtype": (
                    DTYPE_MODEL_NAME
                ),
                "rpc_dtype": (
                    DTYPE_RPC_NAME
                ),
                "token_ranges": (
                    TOKEN_RANGES
                ),
                "num_samples": len(
                    samples
                ),
                "labels_path": (
                    LABELS_JSONL
                ),
                "noise_bank_root": (
                    NOISE_BANK_PATH
                ),
                "created_unix_time": (
                    time.time()
                ),
            }
            with open(
                RUN_MANIFEST_JSON,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    manifest,
                    file,
                    indent=2,
                )
            if WARMUP_RUNS > 0:
                raise RuntimeError(
                    "WARMUP_RUNS must remain 0 "
                    "for this bank because no "
                    "separate warmup bank was "
                    "precomputed."
                )
            generation_rows = []
            batches = list(
                make_batches(
                    samples,
                    BATCH_SIZE,
                )
            )
            print(
                "Starting captured noised-T3 runs...",
                flush=True,
            )
            for batch_index, batch_samples in enumerate(
                batches
            ):
                chunk_payload = torch.load(
                    chunk_meta_paths[
                        batch_index
                    ],
                    map_location="cpu",
                )
                load_chunk_into_model(
                    model,
                    chunk_payload,
                )
                rows = generate_one_batch(
                    model=model,
                    tokenizer=tokenizer,
                    batch_samples=(
                        batch_samples
                    ),
                    batch_index=(
                        batch_index
                    ),
                    capture_enabled=(
                        ATTACK_CAPTURE_ENABLED
                    ),
                )
                generation_rows.extend(
                    rows
                )
                print(
                    f"  batch "
                    f"{batch_index + 1}/"
                    f"{len(batches)} "
                    f"complete | "
                    f"samples="
                    f"{len(batch_samples)}",
                    flush=True,
                )
            with open(
                GENERATIONS_CSV,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                fieldnames = list(
                    generation_rows[
                        0
                    ].keys()
                )
                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )
                writer.writeheader()
                writer.writerows(
                    generation_rows
                )
            print("=" * 88)
            print(
                "NOISED-T3 COLLECTION "
                "CLIENT COMPLETED"
            )
            print(
                f"Protected labels : "
                f"{LABELS_JSONL}"
            )
            print(
                f"Run manifest     : "
                f"{RUN_MANIFEST_JSON}"
            )
            print(
                f"Generation log   : "
                f"{GENERATIONS_CSV}"
            )
            print("=" * 88)
        finally:
            if (
                rpc_client
                is not None
            ):
                rpc_client.close()
            try:
                shm_mm.close()
            except Exception:
                pass
if __name__ == "__main__":
    main()
