"""Noised T3 runtime client for TDX and GPU split inference."""

import os
import csv
import time
import gc
import ctypes
import math
import mmap
import argparse
import pickle
import socket
import struct
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(28)
torch.set_num_interop_threads(1)

torch.manual_seed(1234)
np.random.seed(1234)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_NAME = "databricks/databricks-dolly-15k"
SPLIT = "train"

EPS_MODE = "custom"        # "ep1", "sqrt", or "custom"
CUSTOM_EPSILON = 1.0

BANK_DIR = "results/noised_banks"
BANK_NORMAL_DTYPE_NAME = "float32"

DTYPE_MODEL_NAME = "float32"
DTYPE_NORMAL_CPU_NAME = "float32"

MIN_TOKENS = 100
MAX_TOKENS = 500

TOKEN_RANGES = [
    (100,102),
]

NUM_PROMPTS_PER_RANGE = 30
CAPS = [100]
BATCH_SIZE = 15

WARMUP_RUNS = 1
WARMUP_MAX_NEW_TOKENS = 2

ALLOW_REFILL_DURING_TIMED_RUNS = False

ATTN_IMPL = "sdpa"

CPU_DEVICE = torch.device("cpu")

SHM_PATH = "/sys/bus/pci/devices/0000:00:03.0/resource2"
SHM_SIZE = 8 * 1024 * 1024 * 1024

IN_OFF = 0
OUT_BASE_OFF = 1024 * 1024 * 1024
OUT_STRIDE = 1024 * 1024 * 1024

# Dtypes and paths
def dtype_from_name(name):
    name = name.lower().strip()

    if name == "float32":
        return torch.float32

    if name == "float64":
        return torch.float64

    if name == "float16":
        return torch.float16

    raise ValueError("Use float16, float32, or float64.")

def dtype_to_name(dtype):
    if dtype == torch.float32:
        return "float32"

    if dtype == torch.float64:
        return "float64"

    if dtype == torch.float16:
        return "float16"

    raise ValueError(f"Unsupported dtype: {dtype}")

DTYPE_MODEL = dtype_from_name(DTYPE_MODEL_NAME)
DTYPE_NORMAL_CPU = dtype_from_name(DTYPE_NORMAL_CPU_NAME)

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

def output_prefix():
    return (
        f"{epsilon_tag()}_TDX_IVSHMEM_NOISED_T3_NO_DOWN_"
        f"model{DTYPE_MODEL_NAME}_"
        f"normalCPU{DTYPE_NORMAL_CPU_NAME}_"
        f"PRECORR_BUFFERED_"
        f"bsz{BATCH_SIZE}"
    )

# Bank and output paths

MANUAL_SPLIT_BANK_DIR = os.path.join(BANK_DIR, "CUSTOM1p0(b15)_100_chunked")

if MANUAL_SPLIT_BANK_DIR is None:
    SPLIT_BANK_DIR = os.path.join(
        BANK_DIR,
        bank_name().replace(".pt", "_split"),
    )
else:
    SPLIT_BANK_DIR = MANUAL_SPLIT_BANK_DIR

NOISE_BANK_PATH = os.path.join(SPLIT_BANK_DIR, "meta.pt")
WARMUP_BANK_META_PATH = os.path.join(
    SPLIT_BANK_DIR,
    "warmup",
    "meta.pt",
)

OUTPUT_DIR = "results/noised_runtime"
OUT_CSV = os.path.join(OUTPUT_DIR, f"{output_prefix()}.csv")
OUT_PNG = os.path.join(OUTPUT_DIR, f"{output_prefix()}.png")

def get_epsilon(seq_len):
    if EPS_MODE == "ep1":
        return 1.0

    if EPS_MODE == "custom":
        return float(CUSTOM_EPSILON)

    if EPS_MODE == "sqrt":
        return 1.0 / math.sqrt(float(seq_len))

    raise ValueError("EPS_MODE must be ep1, sqrt, or custom.")

torch.set_grad_enabled(False)

def unwrap_tensor(x, where=""):
    while isinstance(x, (tuple, list)):
        if len(x) == 0:
            raise RuntimeError(f"Empty tuple/list passed where tensor expected: {where}")
        x = x[0]

    if not torch.is_tensor(x):
        raise RuntimeError(f"Expected tensor at {where}, got {type(x)}")

    return x

def to_cpu_dtype_tensor(x, dtype, where, non_blocking=False, contiguous=True):
    x = unwrap_tensor(x, where)

    if x.device.type != "cpu":
        x = x.to(CPU_DEVICE, dtype=dtype, non_blocking=non_blocking)
    elif x.dtype != dtype:
        x = x.to(dtype=dtype)

    if contiguous:
        x = x.contiguous()

    return x

# Runtime timers
TIMERS = defaultdict(float)

def reset_timers():
    TIMERS.clear()

def t_add(name, dt):
    TIMERS[name] += float(dt)

def t_get(name):
    return float(TIMERS.get(name, 0.0))

def now():
    return time.perf_counter()

def clamp0(x):
    return x if x > 0 else 0.0

# Socket communication
def recv_exact(sock, n):
    chunks = []
    remaining = int(n)

    while remaining > 0:
        chunk = sock.recv(remaining)

        if not chunk:
            raise ConnectionError("Socket closed while receiving data.")

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)

def recv_msg(sock):
    header = recv_exact(sock, 8)
    msg_len = struct.unpack("!Q", header)[0]
    payload = recv_exact(sock, msg_len)
    return pickle.loads(payload)

def send_msg(sock, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!Q", len(payload))
    sock.sendall(header + payload)

# Shared memory
_DTYPE_TO_NUMPY = {
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "int64": np.int64,
    "int32": np.int32,
}

_TORCH_TO_DTYPE = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.float64: "float64",
    torch.int64: "int64",
    torch.int32: "int32",
}

def numel_from_shape(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)

def nbytes_for(shape, dtype_name):
    return numel_from_shape(shape) * np.dtype(_DTYPE_TO_NUMPY[dtype_name]).itemsize

def shm_write_tensor(mm, offset, t):
    t = to_cpu_dtype_tensor(
        t,
        DTYPE_NORMAL_CPU,
        "shm_write_tensor input",
        non_blocking=False,
        contiguous=True,
    )

    arr = t.numpy()
    nbytes = arr.nbytes

    end = int(offset) + int(nbytes)
    if end > SHM_SIZE:
        raise RuntimeError(f"SHM write overflow: offset={offset}, nbytes={nbytes}, end={end}")

    mm[int(offset):end] = memoryview(arr).cast("B")

    return {
        "shape": tuple(int(x) for x in t.shape),
        "dtype": _TORCH_TO_DTYPE[t.dtype],
        "nbytes": int(nbytes),
        "offset": int(offset),
    }

def shm_read_tensor(mm, meta, reusable_buffers=None):
    offset = int(meta["offset"])
    shape = tuple(int(x) for x in meta["shape"])
    dtype_name = meta["dtype"]
    nbytes = int(meta["nbytes"])

    expected = nbytes_for(shape, dtype_name)
    if expected != nbytes:
        raise RuntimeError(
            f"SHM read nbytes mismatch: expected={expected}, got={nbytes}"
        )

    raw = memoryview(mm)[offset:offset + nbytes]
    src = np.frombuffer(
        raw,
        dtype=_DTYPE_TO_NUMPY[dtype_name],
    ).reshape(shape)

    if reusable_buffers is None:
        return torch.from_numpy(src.copy()).to(
            device=CPU_DEVICE,
            dtype=DTYPE_NORMAL_CPU,
        ).contiguous()

    dst = reusable_buffers.get(
        offset=offset,
        shape=shape,
        dtype_name=dtype_name,
    )

    # Copy into reusable CPU buffer.
    np.copyto(dst.numpy(), src, casting="no")

    if dst.dtype != DTYPE_NORMAL_CPU:
        return dst.to(dtype=DTYPE_NORMAL_CPU)

    return dst

class ReusableShmReadBuffers:
    """Reuse CPU buffers for shared-memory reads."""

    def __init__(self):
        self.buffers = {}

    def get(self, offset, shape, dtype_name):
        key = (
            int(offset),
            tuple(int(x) for x in shape),
            str(dtype_name),
        )

        t = self.buffers.get(key)
        if t is None:
            np_dtype = _DTYPE_TO_NUMPY[dtype_name]
            torch_dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "float64": torch.float64,
                "int64": torch.int64,
                "int32": torch.int32,
            }[dtype_name]

            t = torch.empty(
                key[1],
                dtype=torch_dtype,
                device=CPU_DEVICE,
            ).contiguous()

            self.buffers[key] = t

        return t

class ShmGPUWorkerClient:
    def __init__(self, host, port, shm_mm, timeout=None):
        self.host = host
        self.port = int(port)
        self.mm = shm_mm
        self.read_buffers = ReusableShmReadBuffers()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        if timeout is not None:
            self.sock.settimeout(float(timeout))

        print(f"Connecting to SHM GPU worker at {self.host}:{self.port} ...", flush=True)
        self.sock.connect((self.host, self.port))

        send_msg(self.sock, {"cmd": "ping"})
        pong = recv_msg(self.sock)

        if not pong.get("ok", False):
            raise RuntimeError(f"GPU worker ping failed: {pong}")

        if not pong.get("fused_qkv", False):
            print(
                "Warning: GPU worker did not advertise fused QKV support.",
                flush=True,
            )

        if not pong.get("fused_gate_up", False):
            print(
                "Warning: GPU worker did not advertise fused GATE/UP support.",
                flush=True,
            )

        print("Connected to fused SHM GPU worker.", flush=True)

    def linear_many(self, layer_idx, op_names, x_masked):
        x_masked = to_cpu_dtype_tensor(
            x_masked,
            DTYPE_NORMAL_CPU,
            "ShmGPUWorkerClient.linear_many input",
            non_blocking=False,
            contiguous=True,
        )

        input_meta = shm_write_tensor(self.mm, IN_OFF, x_masked)

        out_offsets = [
            OUT_BASE_OFF + i * OUT_STRIDE
            for i in range(len(op_names))
        ]

        req = {
            "cmd": "linear_many_shm",
            "layer_idx": layer_idx,
            "op_names": list(op_names),
            "input": input_meta,
            "out_offsets": out_offsets,
        }

        send_msg(self.sock, req)
        resp = recv_msg(self.sock)

        if not resp.get("ok", False):
            raise RuntimeError(
                "GPU worker returned error:\n"
                f"{resp.get('error')}\n"
                f"{resp.get('traceback', '')}"
            )

        if resp.get("fused", False):
            fused_tensor = shm_read_tensor(
                self.mm,
                resp["output"],
                reusable_buffers=self.read_buffers,
            )

            return {
                "__fused__": True,
                "tensor": fused_tensor,
                "op_names": list(resp["op_names"]),
                "split_sizes": tuple(
                    int(x) for x in resp["split_sizes"]
                ),
            }

        outputs = {}
        for name, meta in resp["outputs"].items():
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
            _ = recv_msg(self.sock)
        except Exception:
            pass

        try:
            self.sock.close()
        except Exception:
            pass

# Dataset and prompts
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
                    buckets[r].append((text, enc, seq_len))
                break

        if all(len(v) >= NUM_PROMPTS_PER_RANGE for v in buckets.values()):
            break

    selected = []

    for r in TOKEN_RANGES:
        selected.extend(buckets[r])

    return selected, buckets

# Model helpers
def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    raise ValueError("Could not find model.model.layers")

def get_embed_tokens_module(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens

    raise ValueError("Could not find embed_tokens")

def get_final_norm_module(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm

    return None

def add_module_timer(module, timer_name):
    def pre_hook(_m, _i):
        _m._timer_start = now()

    def post_hook(_m, _i, _o):
        t_add(timer_name, now() - _m._timer_start)

    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)

def attach_model_forward_total_timer(model):
    def pre_hook(_m, _i):
        _m._forward_total_start = now()

    def post_hook(_m, _i, _o):
        t_add("forward_total", now() - _m._forward_total_start)

    model.register_forward_pre_hook(pre_hook)
    model.register_forward_hook(post_hook)

# Correction bank loading
def resolve_bank_path(path, payload=None):
    if not isinstance(path, str):
        return path

    if os.path.exists(path):
        return path

    name = os.path.basename(path)
    candidates = []

    if payload is not None and payload.get("split_bank_dir"):
        candidates.append(os.path.join(payload["split_bank_dir"], name))

    candidates.append(os.path.join(SPLIT_BANK_DIR, name))
    candidates.append(os.path.join(os.path.dirname(NOISE_BANK_PATH), name))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Split bank file not found.\n"
        f"Original path: {path}\n"
        f"Tried: {candidates}"
    )

class LoadedBank:
    """Load stored masks and corrections."""

    def __init__(
        self,
        serialized_bank_or_path,
        runtime_dtype,
        fuse_corrections=False,
        expected_corr_count=None,
    ):
        self.source = serialized_bank_or_path
        self.runtime_dtype = runtime_dtype
        self.fuse_corrections = bool(fuse_corrections)
        self.expected_corr_count = (
            None
            if expected_corr_count is None
            else int(expected_corr_count)
        )

        self.bank = defaultdict(deque)
        self.meta = {}
        self.loaded = False

    def _load_if_needed(self):
        if self.loaded:
            return

        if isinstance(self.source, str):
            print(f"Loading split bank: {self.source}", flush=True)
            serialized_bank = torch.load(
                self.source,
                map_location="cpu",
            )
        else:
            print("Loading embedded bank from payload", flush=True)
            serialized_bank = self.source

        for shape, entry in serialized_bank.items():
            shape = tuple(shape)
            converted = []

            for eta, corrs in entry["items"]:
                eta = unwrap_tensor(
                    eta,
                    "LoadedBank eta",
                ).to(
                    device=CPU_DEVICE,
                    dtype=self.runtime_dtype,
                    copy=False,
                ).contiguous()

                converted_corrs = []

                for corr in corrs:
                    corr = unwrap_tensor(
                        corr,
                        "LoadedBank corr",
                    ).to(
                        device=CPU_DEVICE,
                        dtype=self.runtime_dtype,
                        copy=False,
                    ).contiguous()

                    converted_corrs.append(corr)

                if (
                    self.expected_corr_count is not None
                    and len(converted_corrs) != self.expected_corr_count
                ):
                    raise RuntimeError(
                        f"Correction count mismatch while loading shape "
                        f"{shape}: got {len(converted_corrs)}, expected "
                        f"{self.expected_corr_count}."
                    )

                if self.fuse_corrections:
                    if len(converted_corrs) == 0:
                        raise RuntimeError(
                            f"Cannot fuse an empty correction list for "
                            f"shape {shape}."
                        )

                    # Fuse corrections once when loading.
                    fused_corr = torch.cat(
                        converted_corrs,
                        dim=-1,
                    ).contiguous()

                    converted.append((eta, fused_corr))
                else:
                    converted.append((eta, converted_corrs))

            self.bank[shape] = deque(converted)
            self.meta[shape] = entry.get("dp_meta", {})

        self.loaded = True

        if isinstance(self.source, str):
            mode = (
                " with pre-concatenated corrections"
                if self.fuse_corrections
                else ""
            )
            print(
                f"Loaded split bank: {self.source}{mode}",
                flush=True,
            )

    def clear(self):
        self.bank.clear()
        self.meta.clear()
        self.loaded = False

    def set_source(self, source):
        self.clear()
        self.source = source

    def get(self, shape, allow_refill=False):
        self._load_if_needed()

        shape = tuple(shape)

        if not self.bank[shape]:
            if not allow_refill:
                raise RuntimeError(
                    f"Noise bank empty for shape {shape}\n"
                    f"Bank source: {self.source}\n"
                    "This shape existed but has already been consumed."
                )

            raise RuntimeError("Runtime refill is disabled.")

        return self.bank[shape].popleft()

# Masked GPU calls
class RPCMaskedSharedGroupManager:
    """Apply stored masks and corrections for grouped projections."""

    def __init__(
        self,
        rpc_client,
        layer_idx,
        op_names,
        serialized_bank,
        cpu_dtype,
        expected_corr_count,
        timer_name,
        noise_timer_name,
    ):
        self.rpc_client = rpc_client
        self.layer_idx = int(layer_idx)
        self.names = list(op_names)
        self.cpu_dtype = cpu_dtype
        self.expected_corr_count = int(expected_corr_count)
        self.timer_name = timer_name
        self.noise_timer_name = noise_timer_name

        self.bank = LoadedBank(
            serialized_bank,
            runtime_dtype=cpu_dtype,
            fuse_corrections=True,
            expected_corr_count=self.expected_corr_count,
        )

        self.allow_refill = True
        self._cache_outputs = None
        self._remaining = set()

    def set_allow_refill(self, flag):
        self.allow_refill = bool(flag)

    def get_output(self, name, x):
        x = to_cpu_dtype_tensor(
            x,
            self.cpu_dtype,
            f"RPCMaskedSharedGroupManager {name} input",
            non_blocking=False,
            contiguous=True,
        )

        if self._cache_outputs is None:
            t0 = now()

            eta, fused_corr = self.bank.get(
                tuple(x.shape),
                allow_refill=self.allow_refill,
            )

            eta = unwrap_tensor(
                eta,
                "RPCMaskedSharedGroupManager eta",
            )
            fused_corr = unwrap_tensor(
                fused_corr,
                f"RPCMaskedSharedGroupManager fused correction "
                f"{self.names}",
            )
            x_masked = x + eta

            t_add(self.noise_timer_name, now() - t0)

            t0 = now()

            raw_result = self.rpc_client.linear_many(
                layer_idx=self.layer_idx,
                op_names=self.names,
                x_masked=x_masked,
            )

            t_add(self.timer_name, now() - t0)

            t0 = now()

            outputs = {}

            if raw_result.get("__fused__", False):
                returned_names = list(raw_result["op_names"])
                if returned_names != self.names:
                    raise RuntimeError(
                        f"Fused operation order mismatch: "
                        f"returned={returned_names}, expected={self.names}"
                    )

                split_sizes = tuple(
                    int(x) for x in raw_result["split_sizes"]
                )

                if len(split_sizes) != self.expected_corr_count:
                    raise RuntimeError(
                        f"Fused split count mismatch for {self.names}: "
                        f"sizes={split_sizes}, expected "
                        f"{self.expected_corr_count}"
                    )

                fused_raw = unwrap_tensor(
                    raw_result["tensor"],
                    f"RPCMaskedSharedGroupManager fused raw {self.names}",
                ).to(
                    device=CPU_DEVICE,
                    dtype=self.cpu_dtype,
                    non_blocking=False,
                )

                # Use the stored fused correction.
                fused_corr_runtime = fused_corr.to(
                    device=CPU_DEVICE,
                    dtype=self.cpu_dtype,
                    non_blocking=False,
                )

                if tuple(fused_raw.shape) != tuple(
                    fused_corr_runtime.shape
                ):
                    raise RuntimeError(
                        f"Fused raw/correction shape mismatch for {self.names}: "
                        f"raw={tuple(fused_raw.shape)}, "
                        f"corr={tuple(fused_corr_runtime.shape)}"
                    )

                clean_fused = fused_raw - fused_corr_runtime

                clean_parts = torch.split(
                    clean_fused,
                    split_sizes,
                    dim=-1,
                )

                for mod_name, clean_part in zip(
                    self.names,
                    clean_parts,
                ):
                    outputs[mod_name] = clean_part.to(
                        dtype=DTYPE_MODEL
                    )
            else:
                raise RuntimeError(
                    f"Grouped operation {self.names} received a non-fused "
                    "response. Run the matching fused-transfer GPU worker."
                )

            t_add(self.noise_timer_name, now() - t0)

            self._cache_outputs = outputs
            self._remaining = set(self.names)

        out = self._cache_outputs[name]
        self._remaining.discard(name)

        if not self._remaining:
            self._cache_outputs = None
            self._remaining = set()

        return out

class RPCGroupMaskedWrapper(nn.Module):
    def __init__(self, manager, name):
        super().__init__()
        self.manager = manager
        self.name = name

    def forward(self, x):
        return self.manager.get_output(self.name, x)

class RPCMaskedSingleLinearWrapper(nn.Module):
    """Apply stored masks and corrections for a single projection."""

    def __init__(
        self,
        rpc_client,
        layer_idx,
        op_name,
        serialized_bank,
        cpu_dtype,
        name_for_error,
        timer_name,
        noise_timer_name,
    ):
        super().__init__()

        self.rpc_client = rpc_client
        self.layer_idx = layer_idx
        self.op_name = op_name

        self.cpu_dtype = cpu_dtype
        self.name_for_error = name_for_error
        self.timer_name = timer_name
        self.noise_timer_name = noise_timer_name

        self.bank = LoadedBank(serialized_bank, runtime_dtype=cpu_dtype)
        self.allow_refill = True

    def set_allow_refill(self, flag):
        self.allow_refill = bool(flag)

    def forward(self, x):
        x = to_cpu_dtype_tensor(
            x,
            self.cpu_dtype,
            f"RPCMaskedSingleLinearWrapper {self.name_for_error} input",
            non_blocking=False,
            contiguous=True,
        )

        t0 = now()

        eta, corrs = self.bank.get(tuple(x.shape), allow_refill=self.allow_refill)

        if len(corrs) != 1:
            raise RuntimeError(
                f"{self.name_for_error} expected one stored correction, got {len(corrs)}."
            )

        corr = unwrap_tensor(corrs[0], f"RPCMaskedSingleLinearWrapper corr {self.name_for_error}")
        eta = unwrap_tensor(eta, f"RPCMaskedSingleLinearWrapper eta {self.name_for_error}")
        x_masked = x + eta

        t_add(self.noise_timer_name, now() - t0)

        t0 = now()

        raw_result = self.rpc_client.linear_many(
            layer_idx=self.layer_idx,
            op_names=[self.op_name],
            x_masked=x_masked,
        )

        if raw_result.get("__fused__", False):
            raise RuntimeError(
                f"Unexpected fused response for single operation "
                f"{self.name_for_error}"
            )

        raw_outputs = raw_result["outputs"]

        raw_tensor = unwrap_tensor(
            raw_outputs[self.op_name],
            f"RPCMaskedSingleLinearWrapper raw output {self.name_for_error}",
        )
        corr = unwrap_tensor(
            corr,
            f"RPCMaskedSingleLinearWrapper corr {self.name_for_error}",
        )

        raw = raw_tensor.to(
            device=CPU_DEVICE,
            dtype=self.cpu_dtype,
            non_blocking=False,
        )

        t_add(self.timer_name, now() - t0)

        t0 = now()

        clean = raw - corr

        t_add(self.noise_timer_name, now() - t0)

        return clean.to(dtype=DTYPE_MODEL)

# Decoder timing
class BigBlockDecoderLayerNoDownOutsourceRPC(nn.Module):

    def __init__(self, base_layer):
        super().__init__()

        self.input_layernorm = base_layer.input_layernorm
        self.self_attn = base_layer.self_attn
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        self.mlp = base_layer.mlp

        self._qkv_group_manager = getattr(base_layer, "_qkv_group_manager", None)
        self._gate_up_group_manager = getattr(base_layer, "_gate_up_group_manager", None)
        self._o_gaussian_wrapper = getattr(base_layer, "_o_gaussian_wrapper", None)
        self._down_gaussian_wrapper = None

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
        hidden_states = to_cpu_dtype_tensor(
            hidden_states,
            DTYPE_MODEL,
            "BigBlockDecoderLayerNoDownOutsourceRPC hidden_states",
            non_blocking=False,
            contiguous=False,
        )

        residual = hidden_states

        t0 = now()

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

        if isinstance(attn_hidden, tuple):
            attn_hidden = attn_hidden[0]

        hidden_states = residual + attn_hidden

        t_add("Attention block", now() - t0)

        residual = hidden_states

        t0 = now()

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        t_add("MLP block", now() - t0)

        return hidden_states

# Model setup
def validate_payload(payload):
    meta = payload.get("meta", {})

    if meta.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"Bank model mismatch: bank={meta.get('model_name')} runtime={MODEL_NAME}"
        )

    if meta.get("eps_mode") != EPS_MODE:
        raise ValueError(
            f"Bank EPS_MODE mismatch: bank={meta.get('eps_mode')} runtime={EPS_MODE}"
        )

    if EPS_MODE == "custom":
        bank_custom = meta.get("custom_epsilon")

        if bank_custom is None:
            raise ValueError("Bank is missing custom_epsilon.")

        if abs(float(bank_custom) - float(CUSTOM_EPSILON)) > 1e-12:
            raise ValueError(
                f"Bank CUSTOM_EPSILON mismatch: bank={bank_custom} runtime={CUSTOM_EPSILON}"
            )

    if EPS_MODE == "sqrt" and not meta.get("sqrt_epsilon_fixed_from_prompt_len", False):
        raise ValueError(
            "SQRT runtime requires a bank precomputed with epsilon derived once "
            "from each original prompt length and carried through all decode steps. "
            "Regenerate the bank with the fixed precompute script."
        )

    if meta.get("store_down_correction", False):
        raise ValueError("This runtime is NO-DOWN but bank stores DOWN correction.")

    if meta.get("store_down_eta", False):
        raise ValueError("This runtime is NO-DOWN but bank stores DOWN eta.")

    if meta.get("down_outsourced", False):
        raise ValueError("This runtime keeps DOWN inside TDX.")

    if not meta.get("store_all_corrections_except_down", False):
        raise ValueError("This runtime requires all corrections except DOWN stored.")

    bank_batch_size = meta.get("batch_size", None)

    if bank_batch_size is not None and int(bank_batch_size) != int(BATCH_SIZE):
        raise ValueError(
            f"Bank batch_size mismatch: bank={bank_batch_size} runtime={BATCH_SIZE}."
        )

    print("Bank meta:")
    print(f"  bank_type                         : {meta.get('bank_type')}")
    print(f"  eps_mode                          : {meta.get('eps_mode')}")
    print(f"  custom_epsilon                    : {meta.get('custom_epsilon')}")
    print(f"  dtype_normal                      : {meta.get('dtype_normal')}")
    print(f"  hidden_size                       : {meta.get('hidden_size')}")
    print(f"  intermediate_size                 : {meta.get('intermediate_size')}")
    print(f"  batch_size                        : {meta.get('batch_size')}")
    print(f"  store_down_correction             : {meta.get('store_down_correction')}")
    print(f"  store_down_eta                    : {meta.get('store_down_eta')}")
    print(f"  down_outsourced                   : {meta.get('down_outsourced')}")
    print(f"  store_all_corrections_except_down : {meta.get('store_all_corrections_except_down')}")

def build_model(payload, rpc_client):
    validate_payload(payload)

    print("Loading CPU model inside TDX...", flush=True)

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE_MODEL,
        attn_implementation=ATTN_IMPL,
    ).to(CPU_DEVICE).eval()

    mdl.model.embed_tokens = mdl.model.embed_tokens.to(CPU_DEVICE, dtype=DTYPE_MODEL)

    if hasattr(mdl.model, "norm"):
        mdl.model.norm = mdl.model.norm.to(CPU_DEVICE, dtype=DTYPE_MODEL)

    layers = get_transformer_layers(mdl)

    split_mode = "layer_bank_paths" in payload

    if split_mode:
        print("Runtime bank format: split bank files", flush=True)
        layer_bank_map = {
            entry["layer_idx"]: entry
            for entry in payload["layer_bank_paths"]
        }
    else:
        print("Runtime bank format: embedded old bank", flush=True)
        layer_bank_map = {
            entry["layer_idx"]: entry
            for entry in payload["layer_banks"]
        }

    for i, layer in enumerate(layers):
        banks = layer_bank_map[i]

        if split_mode:
            qkv_bank_source = resolve_bank_path(banks["qkv_bank_path"], payload)
            o_bank_source = resolve_bank_path(banks["o_bank_path"], payload)
            gate_up_bank_source = resolve_bank_path(banks["gate_up_bank_path"], payload)
        else:
            qkv_bank_source = banks["qkv_bank"]
            o_bank_source = banks["o_bank"]
            gate_up_bank_source = banks["gate_up_bank"]

        layer.input_layernorm = layer.input_layernorm.to(CPU_DEVICE, dtype=DTYPE_MODEL)
        layer.self_attn = layer.self_attn.to(CPU_DEVICE, dtype=DTYPE_MODEL)
        layer.post_attention_layernorm = layer.post_attention_layernorm.to(
            CPU_DEVICE,
            dtype=DTYPE_MODEL,
        )

        qkv_group = RPCMaskedSharedGroupManager(
            rpc_client=rpc_client,
            layer_idx=i,
            op_names=["q_proj", "k_proj", "v_proj"],
            serialized_bank=qkv_bank_source,
            cpu_dtype=DTYPE_NORMAL_CPU,
            expected_corr_count=3,
            timer_name="QKV RPC roundtrip",
            noise_timer_name="QKV Gaussian add/remove",
        )

        layer.self_attn.q_proj = RPCGroupMaskedWrapper(qkv_group, "q_proj")
        layer.self_attn.k_proj = RPCGroupMaskedWrapper(qkv_group, "k_proj")
        layer.self_attn.v_proj = RPCGroupMaskedWrapper(qkv_group, "v_proj")

        o_wrap = RPCMaskedSingleLinearWrapper(
            rpc_client=rpc_client,
            layer_idx=i,
            op_name="o_proj",
            serialized_bank=o_bank_source,
            cpu_dtype=DTYPE_NORMAL_CPU,
            name_for_error="O",
            timer_name="O RPC roundtrip",
            noise_timer_name="O Gaussian add/remove",
        )

        layer.self_attn.o_proj = o_wrap

        gate_up_group = RPCMaskedSharedGroupManager(
            rpc_client=rpc_client,
            layer_idx=i,
            op_names=["gate_proj", "up_proj"],
            serialized_bank=gate_up_bank_source,
            cpu_dtype=DTYPE_NORMAL_CPU,
            expected_corr_count=2,
            timer_name="MLP gate+up RPC roundtrip",
            noise_timer_name="GATE/UP Gaussian add/remove",
        )

        layer.mlp.gate_proj = RPCGroupMaskedWrapper(gate_up_group, "gate_proj")
        layer.mlp.up_proj = RPCGroupMaskedWrapper(gate_up_group, "up_proj")

        # DOWN stays inside TDX.
        layer.mlp.down_proj = layer.mlp.down_proj.to(CPU_DEVICE, dtype=DTYPE_MODEL)

        layer._qkv_group_manager = qkv_group
        layer._gate_up_group_manager = gate_up_group
        layer._o_gaussian_wrapper = o_wrap
        layer._down_gaussian_wrapper = None

    for i, layer in enumerate(layers):
        layers[i] = BigBlockDecoderLayerNoDownOutsourceRPC(layer)

    if split_mode:
        lm_bank_source = resolve_bank_path(payload["lm_head_bank_path"], payload)
    else:
        lm_bank_source = payload["lm_head_bank"]

    lm_wrap = RPCMaskedSingleLinearWrapper(
        rpc_client=rpc_client,
        layer_idx=None,
        op_name="lm_head",
        serialized_bank=lm_bank_source,
        cpu_dtype=DTYPE_NORMAL_CPU,
        name_for_error="LM_HEAD",
        timer_name="LM head RPC roundtrip",
        noise_timer_name="LM Gaussian add/remove",
    )

    mdl.lm_head = lm_wrap

    return mdl.eval()

def set_refill_flag(model, flag):
    for layer in get_transformer_layers(model):
        if hasattr(layer, "_qkv_group_manager") and layer._qkv_group_manager is not None:
            layer._qkv_group_manager.set_allow_refill(flag)

        if hasattr(layer, "_gate_up_group_manager") and layer._gate_up_group_manager is not None:
            layer._gate_up_group_manager.set_allow_refill(flag)

        if hasattr(layer, "_o_gaussian_wrapper") and layer._o_gaussian_wrapper is not None:
            layer._o_gaussian_wrapper.set_allow_refill(flag)

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "set_allow_refill"):
        model.lm_head.set_allow_refill(flag)

def force_load_all_split_banks(model):
    print("=" * 80, flush=True)
    print("FORCE LOADING ALL SPLIT NOISE BANKS BEFORE TIMING", flush=True)
    print("=" * 80, flush=True)

    for layer_idx, layer in enumerate(get_transformer_layers(model)):
        if hasattr(layer, "_qkv_group_manager") and layer._qkv_group_manager is not None:
            print(f"Loading layer {layer_idx} QKV bank...", flush=True)
            layer._qkv_group_manager.bank._load_if_needed()

        if hasattr(layer, "_o_gaussian_wrapper") and layer._o_gaussian_wrapper is not None:
            print(f"Loading layer {layer_idx} O bank...", flush=True)
            layer._o_gaussian_wrapper.bank._load_if_needed()

        if hasattr(layer, "_gate_up_group_manager") and layer._gate_up_group_manager is not None:
            print(f"Loading layer {layer_idx} GATE/UP bank...", flush=True)
            layer._gate_up_group_manager.bank._load_if_needed()

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "bank"):
        print("Loading LM_HEAD bank...", flush=True)
        model.lm_head.bank._load_if_needed()

    print("=" * 80, flush=True)
    print("FINISHED LOADING BANKS BEFORE TIMING", flush=True)
    print("=" * 80, flush=True)

# Bank chunk loading
def malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def clear_all_loaded_banks(model):
    for layer in get_transformer_layers(model):
        if (
            hasattr(layer, "_qkv_group_manager")
            and layer._qkv_group_manager is not None
        ):
            layer._qkv_group_manager.bank.clear()

        if (
            hasattr(layer, "_o_gaussian_wrapper")
            and layer._o_gaussian_wrapper is not None
        ):
            layer._o_gaussian_wrapper.bank.clear()

        if (
            hasattr(layer, "_gate_up_group_manager")
            and layer._gate_up_group_manager is not None
        ):
            layer._gate_up_group_manager.bank.clear()

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "bank"):
        model.lm_head.bank.clear()

    gc.collect()
    malloc_trim()

def activate_batch_payload(model, payload):
    split_mode = "layer_bank_paths" in payload

    if not split_mode:
        raise RuntimeError(
            "Chunked runtime requires split-bank files."
        )

    layer_bank_map = {
        int(entry["layer_idx"]): entry
        for entry in payload["layer_bank_paths"]
    }

    for layer_idx, layer in enumerate(get_transformer_layers(model)):
        banks = layer_bank_map[layer_idx]

        qkv_source = resolve_bank_path(
            banks["qkv_bank_path"],
            payload,
        )
        o_source = resolve_bank_path(
            banks["o_bank_path"],
            payload,
        )
        gate_up_source = resolve_bank_path(
            banks["gate_up_bank_path"],
            payload,
        )

        layer._qkv_group_manager.bank.set_source(qkv_source)
        layer._o_gaussian_wrapper.bank.set_source(o_source)
        layer._gate_up_group_manager.bank.set_source(gate_up_source)

    lm_source = resolve_bank_path(
        payload["lm_head_bank_path"],
        payload,
    )
    model.lm_head.bank.set_source(lm_source)

    force_load_all_split_banks(model)

def validate_warmup_payload(warmup_payload):
    if warmup_payload.get("bank_type") != "SEPARATE_WARMUP_B1_NO_DOWN":
        raise ValueError(
            f"Unexpected warmup bank type: {warmup_payload.get('bank_type')}"
        )

    if int(warmup_payload.get("batch_size", -1)) != 1:
        raise ValueError("Separate warmup bank must use batch_size=1.")

    bank_cap = int(warmup_payload.get("warmup_cap", -1))
    if WARMUP_MAX_NEW_TOKENS > bank_cap:
        raise ValueError(
            f"Warmup token mismatch: runtime={WARMUP_MAX_NEW_TOKENS}, "
            f"bank_cap={bank_cap}"
        )

    prefill_count = int(warmup_payload.get("prefill_count", 0))
    if WARMUP_RUNS > prefill_count:
        raise ValueError(
            f"Warmup run mismatch: runtime={WARMUP_RUNS}, "
            f"bank_prefill_count={prefill_count}"
        )

    if "layer_bank_paths" not in warmup_payload:
        raise ValueError("Warmup bank is missing layer_bank_paths.")

    if "lm_head_bank_path" not in warmup_payload:
        raise ValueError("Warmup bank is missing lm_head_bank_path.")

# Plot
COLOR_MAP = {
    "Embedding": "#6F97BD",
    "Attention block": "#5FAFB5",
    "MLP block": "#7EAD78",
    "Final norm": "#9A84BF",
    "LM head": "#A97967",
    "Forward other": "#5F8581",
    "Generate overhead": "#C07EAF",
}

def plot_rows(rows):
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
        f"TDX IVSHMEM NOISED T3: NO DOWN Outsourcing "
        f"({EPS_MODE}, bsz={BATCH_SIZE})"
    )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.savefig(OUT_PNG, dpi=220)
    print(f"Saved plot: {OUT_PNG}")

# Batch timing
def run_one_batch(model, inputs, prompt_tokens_list, cap, tokenizer):
    reset_timers()

    bsz = int(inputs["input_ids"].shape[0])

    t0_gen = now()

    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            min_new_tokens=cap,
            max_new_tokens=cap,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_total = now() - t0_gen
    forward_total = t_get("forward_total")

    row_batch = {
        "batch_size_actual": bsz,
        "prompt_tokens_sum": float(sum(prompt_tokens_list)),
        "epsilon_sum": float(sum(get_epsilon(x) for x in prompt_tokens_list)),

        "Embedding": t_get("Embedding"),
        "Attention block": t_get("Attention block"),
        "MLP block": t_get("MLP block"),
        "Final norm": t_get("Final norm"),
        "LM head": t_get("LM head RPC roundtrip"),

        "QKV RPC roundtrip": t_get("QKV RPC roundtrip"),
        "O RPC roundtrip": t_get("O RPC roundtrip"),
        "MLP gate+up RPC roundtrip": t_get("MLP gate+up RPC roundtrip"),
        "LM head RPC roundtrip": t_get("LM head RPC roundtrip"),

        "QKV Gaussian add/remove": t_get("QKV Gaussian add/remove"),
        "O Gaussian add/remove": t_get("O Gaussian add/remove"),
        "GATE/UP Gaussian add/remove": t_get("GATE/UP Gaussian add/remove"),
        "LM Gaussian add/remove": t_get("LM Gaussian add/remove"),

        "MLP down roundtrip": 0.0,
        "DOWN Gaussian add/remove": 0.0,
    }

    row_batch["Attention internal (CPU)"] = clamp0(
        row_batch["Attention block"]
        - row_batch["QKV RPC roundtrip"]
        - row_batch["O RPC roundtrip"]
    )

    row_batch["MLP SiLU+mul+DOWN (CPU)"] = clamp0(
        row_batch["MLP block"]
        - row_batch["MLP gate+up RPC roundtrip"]
    )

    known_forward = (
        row_batch["Embedding"]
        + row_batch["Attention block"]
        + row_batch["MLP block"]
        + row_batch["Final norm"]
        + row_batch["LM head"]
    )

    row_batch["Forward other"] = clamp0(forward_total - known_forward)
    row_batch["Generate overhead"] = clamp0(gen_total - forward_total)
    row_batch["gen_total_s"] = gen_total

    return row_batch

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-server", type=str, required=True)
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    print("=" * 80)
    print("TDX RUNTIME CLIENT: NOISED T3 - CHUNK-STREAMED BANK")
    print("QKV/O/GATE-UP/LM_HEAD use stored corrections with reusable shared-memory read buffers")
    print("DOWN runs inside TDX CPU with no noise and no correction")
    print("=" * 80)
    print(f"MODEL_NAME          : {MODEL_NAME}")
    print(f"EPS_MODE            : {EPS_MODE}")
    print(f"CUSTOM_EPSILON      : {CUSTOM_EPSILON}")
    print(f"BATCH_SIZE          : {BATCH_SIZE}")
    print(f"SPLIT_BANK_DIR      : {SPLIT_BANK_DIR}")
    print(f"META_PATH           : {NOISE_BANK_PATH}")
    print(f"WARMUP_META_PATH    : {WARMUP_BANK_META_PATH}")
    print(f"OUT_CSV             : {OUT_CSV}")
    print(f"OUT_PNG             : {OUT_PNG}")
    print(f"DTYPE_MODEL         : {DTYPE_MODEL}")
    print(f"DTYPE_NORMAL_CPU    : {DTYPE_NORMAL_CPU}")
    print(f"SHM_PATH            : {SHM_PATH}")
    print(f"GPU_SERVER          : {args.gpu_server}:{args.port}")
    print("=" * 80)

    rpc_client = None

    with open(SHM_PATH, "r+b", buffering=0) as f:
        mm = mmap.mmap(
            f.fileno(),
            SHM_SIZE,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )

        try:
            rpc_client = ShmGPUWorkerClient(
                host=args.gpu_server,
                port=args.port,
                shm_mm=mm,
                timeout=args.timeout,
            )

            print("Loading tokenizer...")
            tok = AutoTokenizer.from_pretrained(MODEL_NAME)

            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token

            tok.padding_side = "left"

            print("Loading root chunked-bank meta...")
            root_payload = torch.load(
                NOISE_BANK_PATH,
                map_location="cpu",
            )

            batch_meta_paths = list(
                root_payload.get("batch_meta_paths", [])
            )

            if not batch_meta_paths:
                raise RuntimeError(
                    "Root meta does not contain batch_meta_paths. "
                    "Run the chunked precompute script first."
                )

            if len(batch_meta_paths) != 2:
                print(
                    f"Warning: expected 2 batch chunks, found "
                    f"{len(batch_meta_paths)}",
                    flush=True,
                )

            first_batch_payload = torch.load(
                batch_meta_paths[0],
                map_location="cpu",
            )

            timed_prefill_count = first_batch_payload.get("meta", {}).get("timed_prefill_count")

            if timed_prefill_count is not None and int(timed_prefill_count) < 1:
                raise ValueError(
                    "Timed bank does not contain a prefill entry: "
                    f"timed_prefill_count={timed_prefill_count}."
    )

            print("Building TDX noised model from batch_000 metadata...")
            mdl = build_model(
                first_batch_payload,
                rpc_client,
            )

            if not ALLOW_REFILL_DURING_TIMED_RUNS:
                set_refill_flag(mdl, False)

            attach_model_forward_total_timer(mdl)
            add_module_timer(get_embed_tokens_module(mdl), "Embedding")

            final_norm = get_final_norm_module(mdl)
            if final_norm is not None:
                add_module_timer(final_norm, "Final norm")

            print("Loading dataset...")
            ds = load_dataset(DATASET_NAME)[SPLIT]

            print("Selecting prompts...")
            selected, buckets = collect_texts_by_ranges(ds, tok)

            if len(selected) == 0:
                raise RuntimeError("No prompts found in requested token ranges.")

            print("Selected prompt counts by range:")
            for r in TOKEN_RANGES:
                print(f"  {r}: {len(buckets[r])}")

            sample_texts = [x[0] for x in selected]
            seq_len_map = {text: seq_len for text, _enc, seq_len in selected}

            print("Selected prompts:")
            for i, text in enumerate(sample_texts):
                print(f"  prompt[{i}] tokens={seq_len_map[text]}")

            batches = list(make_batches(sample_texts, BATCH_SIZE))

            first_batch_texts = batches[0]

            # Warmup
            if WARMUP_RUNS > 0:
                if not os.path.exists(WARMUP_BANK_META_PATH):
                    raise FileNotFoundError(
                        "Separate warmup bank meta not found: "
                        f"{WARMUP_BANK_META_PATH}"
                    )

                print("Loading separate B=1 warmup bank...", flush=True)

                warmup_payload = torch.load(
                    WARMUP_BANK_META_PATH,
                    map_location="cpu",
                )
                validate_warmup_payload(warmup_payload)

                activate_batch_payload(
                    mdl,
                    warmup_payload,
                )

                warmup_inputs = tokenize_as_chat_batch(
                    tok,
                    [first_batch_texts[0]],
                )
                warmup_inputs = {
                    k: v.to(CPU_DEVICE)
                    for k, v in warmup_inputs.items()
                }

                print(
                    f"Warming up: runs={WARMUP_RUNS}, "
                    f"batch_size=1, new_tokens={WARMUP_MAX_NEW_TOKENS}",
                    flush=True,
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
                            f"  warmup {warm_idx + 1}/{WARMUP_RUNS} done",
                            flush=True,
                        )

                reset_timers()
                clear_all_loaded_banks(mdl)
                del warmup_payload
                gc.collect()

                print("Released separate warmup bank.", flush=True)

            print(
                f"Activating fresh timed bank chunk 000: "
                f"{batch_meta_paths[0]}",
                flush=True,
            )
            activate_batch_payload(
                mdl,
                first_batch_payload,
            )

            print("Starting timed batch runs...")

            rows = []

            for cap in CAPS:
                avg_acc = defaultdict(float)

                num_prompts = 0
                num_batches = 0

                for batch_idx, batch_texts in enumerate(batches, start=1):
                    chunk_index = batch_idx - 1

                    if chunk_index >= len(batch_meta_paths):
                        raise RuntimeError(
                            f"Missing bank chunk for runtime batch "
                            f"{chunk_index}"
                        )

                    if chunk_index > 0:
                        clear_all_loaded_banks(mdl)

                        batch_payload = torch.load(
                            batch_meta_paths[chunk_index],
                            map_location="cpu",
                        )

                        print(
                            f"Activating bank chunk "
                            f"{chunk_index:03d}: "
                            f"{batch_meta_paths[chunk_index]}",
                            flush=True,
                        )

                        activate_batch_payload(
                            mdl,
                            batch_payload,
                        )

                        del batch_payload
                        gc.collect()

                    inputs = tokenize_as_chat_batch(tok, batch_texts)
                    inputs = {
                        k: v.to(CPU_DEVICE)
                        for k, v in inputs.items()
                    }

                    prompt_tokens_list = [seq_len_map[text] for text in batch_texts]

                    row_batch = run_one_batch(
                        model=mdl,
                        inputs=inputs,
                        prompt_tokens_list=prompt_tokens_list,
                        cap=cap,
                        tokenizer=tok,
                    )

                    bsz = row_batch["batch_size_actual"]
                    num_prompts += bsz
                    num_batches += 1

                    for k, v in row_batch.items():
                        if k == "batch_size_actual":
                            continue

                        avg_acc[k] += v

                    print(
                        f"  Cap={cap:>4} | batch {batch_idx}/{len(batches)} | "
                        f"bsz={bsz} | padded={int(inputs['input_ids'].shape[1])} | "
                        f"batch_gen_total={row_batch['gen_total_s']:.4f}s | "
                        f"per_prompt={row_batch['gen_total_s'] / bsz:.4f}s"
                    )

                    # Release the completed chunk.
                    clear_all_loaded_banks(mdl)
                    print(
                        f"  Released bank chunk {chunk_index:03d}",
                        flush=True,
                    )

                row = {
                    "eps_mode": EPS_MODE,
                    "custom_epsilon": CUSTOM_EPSILON if EPS_MODE == "custom" else "",
                    "cap": cap,
                    "batch_size_config": BATCH_SIZE,
                    "num_prompts": num_prompts,
                    "num_batches": num_batches,

                    "avg_prompt_tokens": avg_acc["prompt_tokens_sum"] / num_prompts,
                    "avg_epsilon": avg_acc["epsilon_sum"] / num_prompts,
                    "gen_total_s": avg_acc["gen_total_s"] / num_prompts,

                    "dtype_model": DTYPE_MODEL_NAME,
                    "dtype_normal_cpu": DTYPE_NORMAL_CPU_NAME,

                    "stored_all_corrections_except_down": True,
                    "runtime_recomputes_corrections": False,
                    "down_outsourced": False,
                    "gpu_worker_rpc": True,
                    "ivshmem_transfer": True,
                    "split_bank_preloaded": False,
                    "bank_streamed_per_prompt_batch": True,
                    "peak_loaded_bank_chunks": 1,
                    "grouped_corrections_preconcatenated": True,
                    "reusable_shm_read_buffers": True,
                    "gpu_pinned_staging_buffers": True,
                    "gpu_dedicated_cuda_stream": True,
                    "gpu_event_scoped_sync": True,

                    "Embedding": avg_acc["Embedding"] / num_prompts,
                    "Attention block": avg_acc["Attention block"] / num_prompts,
                    "MLP block": avg_acc["MLP block"] / num_prompts,
                    "Final norm": avg_acc["Final norm"] / num_prompts,
                    "LM head": avg_acc["LM head"] / num_prompts,

                    "QKV RPC roundtrip": avg_acc["QKV RPC roundtrip"] / num_prompts,
                    "O RPC roundtrip": avg_acc["O RPC roundtrip"] / num_prompts,
                    "MLP gate+up RPC roundtrip": avg_acc["MLP gate+up RPC roundtrip"] / num_prompts,
                    "LM head RPC roundtrip": avg_acc["LM head RPC roundtrip"] / num_prompts,

                    "Attention internal (CPU)": avg_acc["Attention internal (CPU)"] / num_prompts,
                    "MLP SiLU+mul+DOWN (CPU)": avg_acc["MLP SiLU+mul+DOWN (CPU)"] / num_prompts,

                    "QKV Gaussian add/remove": avg_acc["QKV Gaussian add/remove"] / num_prompts,
                    "O Gaussian add/remove": avg_acc["O Gaussian add/remove"] / num_prompts,
                    "GATE/UP Gaussian add/remove": avg_acc["GATE/UP Gaussian add/remove"] / num_prompts,
                    "DOWN Gaussian add/remove": avg_acc["DOWN Gaussian add/remove"] / num_prompts,
                    "LM Gaussian add/remove": avg_acc["LM Gaussian add/remove"] / num_prompts,

                    "Forward other": avg_acc["Forward other"] / num_prompts,
                    "Generate overhead": avg_acc["Generate overhead"] / num_prompts,
                }

                rows.append(row)

                print(
                    f"Mode={EPS_MODE} | "
                    f"Custom={CUSTOM_EPSILON if EPS_MODE == 'custom' else '-'} | "
                    f"Cap={cap:>4} | "
                    f"batch_size={BATCH_SIZE} | "
                    f"num_prompts={num_prompts} | "
                    f"avg_gen_total_per_prompt={row['gen_total_s']:.4f}s | "
                    f"avg_prompt_tokens={row['avg_prompt_tokens']:.2f} | "
                    f"avg_epsilon={row['avg_epsilon']:.6f} | "
                    f"Attn/prompt={row['Attention block']:.4f}s | "
                    f"MLP/prompt={row['MLP block']:.4f}s | "
                    f"QKV_RPC/prompt={row['QKV RPC roundtrip']:.4f}s | "
                    f"O_RPC/prompt={row['O RPC roundtrip']:.4f}s | "
                    f"GATEUP_RPC/prompt={row['MLP gate+up RPC roundtrip']:.4f}s | "
                    f"DOWN_inside_TDX=True | "
                    f"LMHead_RPC/prompt={row['LM head RPC roundtrip']:.4f}s | "
                    f"QKV_noise={row['QKV Gaussian add/remove']:.4f}s | "
                    f"O_noise={row['O Gaussian add/remove']:.4f}s | "
                    f"GATEUP_noise={row['GATE/UP Gaussian add/remove']:.4f}s | "
                    f"LM_noise={row['LM Gaussian add/remove']:.4f}s"
                )

            os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

            with open(OUT_CSV, "w", newline="", encoding="utf-8") as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            print(f"Saved CSV: {OUT_CSV}")

            plot_rows(rows)

        finally:
            if rpc_client is not None:
                rpc_client.close()

            try:
                mm.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()
