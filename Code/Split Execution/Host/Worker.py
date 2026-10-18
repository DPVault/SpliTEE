#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""GPU worker for clean and noised T3 split inference.
Runs on the host GPU. TDX clients exchange tensors through IVSHMEM
and send request metadata over TCP.
"""
import argparse
import mmap
import os
import pickle
import socket
import struct
import traceback
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU selection

# CONFIG
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE_GPU = torch.float32
SHM_PATH = "/dev/shm/shm1"
SHM_SIZE = 8 * 1024 * 1024 * 1024
# FP32 matmul without TF32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# DTYPE HELPERS
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
    return (
        numel_from_shape(shape)
        * np.dtype(_DTYPE_TO_NUMPY[dtype_name]).itemsize
    )

# SOCKET HELPERS

def recv_exact(conn, nbytes):
    chunks = []
    remaining = int(nbytes)
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!Q", len(data)))
    conn.sendall(data)

def recv_msg(conn):
    hdr = recv_exact(conn, 8)
    n = struct.unpack("!Q", hdr)[0]
    return pickle.loads(recv_exact(conn, n))

# SHM HELPERS

def shm_numpy_view(mm, offset, shape, dtype_name):
    """
    Return a NumPy view over the SHM bytes without allocating/copying.
    The caller must copy it into protected/pinned memory before the next
    request can overwrite the shared-memory region.
    """
    shape = tuple(int(x) for x in shape)
    nbytes = nbytes_for(shape, dtype_name)
    end = int(offset) + int(nbytes)
    if end > SHM_SIZE:
        raise RuntimeError(
            f"SHM read overflow: offset={offset}, nbytes={nbytes}, end={end}"
        )
    raw = memoryview(mm)[int(offset):end]
    return np.frombuffer(
        raw,
        dtype=_DTYPE_TO_NUMPY[dtype_name],
    ).reshape(shape)

def shm_write_tensor(mm, offset, t):
    """
    Write a CPU tensor to SHM.
    GPU completion is guaranteed by GPUWorker before this function is called,
    so this function intentionally performs no CUDA synchronization.
    """
    t_cpu = t.detach()
    if t_cpu.device.type != "cpu" or t_cpu.dtype != torch.float32:
        t_cpu = t_cpu.to(
            device="cpu",
            dtype=torch.float32,
        )
    if not t_cpu.is_contiguous():
        t_cpu = t_cpu.contiguous()
    arr = t_cpu.numpy()
    nbytes = int(arr.nbytes)
    end = int(offset) + nbytes
    if end > SHM_SIZE:
        raise RuntimeError(
            f"SHM write overflow: offset={offset}, "
            f"nbytes={nbytes}, end={end}"
        )
    mm[int(offset):end] = memoryview(arr).cast("B")
    return {
        "shape": tuple(int(x) for x in t_cpu.shape),
        "dtype": _TORCH_TO_DTYPE[t_cpu.dtype],
        "nbytes": nbytes,
        "offset": int(offset),
    }

# GPU WORKER

class GPUWorker:
    def __init__(self):
        print(f"Loading model on GPU worker: {MODEL_NAME}", flush=True)
        print(f"GPU device: {GPU_DEVICE}", flush=True)
        print(f"GPU dtype: {DTYPE_GPU}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE_GPU,
        ).to(GPU_DEVICE).eval()
        self.qkv_weights = []
        self.qkv_sizes = []
        self.gate_up_weights = []
        self.gate_up_sizes = []
        # Reusable staging buffers keyed by tensor shape/dtype.
        # Requests are sequential, so one buffer per key is sufficient.
        self._pinned_input_buffers = {}
        self._gpu_input_buffers = {}
        self._pinned_output_buffers = {}
        # Requests are processed sequentially. Keep all H2D, GEMM, and D2H
        # work for each request on one dedicated stream and wait with an event
        # scoped to that stream instead of synchronizing the whole device.
        self._cuda_stream = torch.cuda.Stream(device=GPU_DEVICE)
        self._request_done_event = torch.cuda.Event(
            enable_timing=False,
            blocking=False,
        )
        print("Building fused QKV and GATE/UP weights...", flush=True)
        for layer_idx, layer in enumerate(self.model.model.layers):
            q_w = layer.self_attn.q_proj.weight
            k_w = layer.self_attn.k_proj.weight
            v_w = layer.self_attn.v_proj.weight
            fused_qkv = torch.cat(
                [q_w, k_w, v_w],
                dim=0,
            ).contiguous()
            self.qkv_weights.append(fused_qkv)
            self.qkv_sizes.append(
                (
                    int(q_w.shape[0]),
                    int(k_w.shape[0]),
                    int(v_w.shape[0]),
                )
            )
            gate_w = layer.mlp.gate_proj.weight
            up_w = layer.mlp.up_proj.weight
            fused_gate_up = torch.cat(
                [gate_w, up_w],
                dim=0,
            ).contiguous()
            self.gate_up_weights.append(fused_gate_up)
            self.gate_up_sizes.append(
                (
                    int(gate_w.shape[0]),
                    int(up_w.shape[0]),
                )
            )
            print(
                f"  layer {layer_idx}: "
                f"QKV={tuple(fused_qkv.shape)}, "
                f"GATE/UP={tuple(fused_gate_up.shape)}",
                flush=True,
            )
        print("Model and fused weights loaded.", flush=True)
    def _get_pinned_input_buffer(self, shape, dtype):
        key = (tuple(int(x) for x in shape), dtype)
        buf = self._pinned_input_buffers.get(key)
        if buf is None:
            buf = torch.empty(
                key[0],
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            self._pinned_input_buffers[key] = buf
        return buf
    def _get_gpu_input_buffer(self, shape, dtype):
        key = (tuple(int(x) for x in shape), dtype)
        buf = self._gpu_input_buffers.get(key)
        if buf is None:
            buf = torch.empty(
                key[0],
                dtype=dtype,
                device=GPU_DEVICE,
            )
            self._gpu_input_buffers[key] = buf
        return buf
    def _get_pinned_output_buffer(self, name, shape, dtype):
        key = (
            str(name),
            tuple(int(x) for x in shape),
            dtype,
        )
        buf = self._pinned_output_buffers.get(key)
        if buf is None:
            buf = torch.empty(
                key[1],
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            self._pinned_output_buffers[key] = buf
        return buf
    def read_shm_into_pinned(
        self,
        mm,
        offset,
        shape,
        dtype_name,
    ):
        """
        Copy SHM bytes directly into a reusable pinned CPU tensor.
        This removes the previous per-RPC:
            np.frombuffer(...).copy()
            torch.from_numpy(...)
        allocation path.
        """
        shape = tuple(int(x) for x in shape)
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
        }[dtype_name]
        src = shm_numpy_view(
            mm,
            offset,
            shape,
            dtype_name,
        )
        pinned = self._get_pinned_input_buffer(
            shape,
            torch_dtype,
        )
        np.copyto(
            pinned.numpy(),
            src,
            casting="no",
        )
        return pinned
    def stage_input_to_gpu(self, x_cpu):
        """
        Stage a reusable pinned CPU tensor into a reusable GPU buffer.
        If x_cpu is already pinned (the normal SHM path), no extra CPU copy is
        made. H2D is enqueued on the dedicated CUDA stream.
        """
        if not x_cpu.is_contiguous():
            x_cpu = x_cpu.contiguous()
        if x_cpu.device.type != "cpu":
            raise RuntimeError("stage_input_to_gpu expects a CPU tensor.")
        if x_cpu.is_pinned():
            pinned = x_cpu
        else:
            pinned = self._get_pinned_input_buffer(
                x_cpu.shape,
                x_cpu.dtype,
            )
            pinned.copy_(x_cpu, non_blocking=False)
        gpu_buf = self._get_gpu_input_buffer(
            pinned.shape,
            DTYPE_GPU,
        )
        gpu_buf.copy_(
            pinned,
            non_blocking=True,
        )
        return gpu_buf
    def stage_output_to_cpu(self, name, y_gpu):
        """
        Enqueue GPU-to-pinned-CPU copy on the current dedicated stream.
        The request event is synchronized before the returned tensor is read.
        """
        out = self._get_pinned_output_buffer(
            name,
            y_gpu.shape,
            torch.float32,
        )
        out.copy_(y_gpu, non_blocking=True)
        return out
    def finish_request(self):
        """
        Wait only for work already enqueued on this worker's CUDA stream.
        This replaces the previous per-RPC torch.cuda.synchronize().
        """
        self._request_done_event.record(self._cuda_stream)
        self._request_done_event.synchronize()
    @staticmethod
    def normalize_op(op):
        mapping = {
            "q": "q_proj",
            "k": "k_proj",
            "v": "v_proj",
            "o": "o_proj",
            "gate": "gate_proj",
            "up": "up_proj",
            "down": "down_proj",
            "lm": "lm_head",
        }
        return mapping.get(op, op)
    @staticmethod
    def canonical_names(op_names):
        return [GPUWorker.normalize_op(x) for x in op_names]
    @torch.inference_mode()
    def linear_many(self, layer_idx, op_names, x_cpu):
        """
        Execute one sequential RPC request.
        H2D, GEMM, and D2H are all issued on the dedicated CUDA stream.
        Before returning, a CUDA event waits only for this request's stream
        work, making the pinned CPU result safe for the SHM write.
        """
        canonical = self.canonical_names(op_names)
        with torch.cuda.stream(self._cuda_stream):
            x_gpu = self.stage_input_to_gpu(x_cpu)
            # ------------------------------------------------
            # Fused QKV path
            # ------------------------------------------------
            if canonical == ["q_proj", "k_proj", "v_proj"]:
                if layer_idx is None:
                    raise ValueError("layer_idx cannot be None for fused QKV")
                idx = int(layer_idx)
                fused_y = F.linear(
                    x_gpu,
                    self.qkv_weights[idx],
                    bias=None,
                )
                fused_cpu = self.stage_output_to_cpu(
                    "fused_qkv",
                    fused_y,
                )
                result = {
                    "fused": True,
                    "tensor": fused_cpu,
                    "split_sizes": self.qkv_sizes[idx],
                }
            # ------------------------------------------------
            # Fused GATE/UP path
            # ------------------------------------------------
            elif canonical == ["gate_proj", "up_proj"]:
                if layer_idx is None:
                    raise ValueError(
                        "layer_idx cannot be None for fused GATE/UP"
                    )
                idx = int(layer_idx)
                fused_y = F.linear(
                    x_gpu,
                    self.gate_up_weights[idx],
                    bias=None,
                )
                fused_cpu = self.stage_output_to_cpu(
                    "fused_gate_up",
                    fused_y,
                )
                result = {
                    "fused": True,
                    "tensor": fused_cpu,
                    "split_sizes": self.gate_up_sizes[idx],
                }
            # ------------------------------------------------
            # Single/general path
            # ------------------------------------------------
            else:
                outputs_gpu = {}
                for original_name, op in zip(op_names, canonical):
                    if op == "lm_head":
                        y = self.model.lm_head(x_gpu)
                    else:
                        if layer_idx is None:
                            raise ValueError(
                                f"layer_idx cannot be None for op={op}"
                            )
                        layer = self.model.model.layers[int(layer_idx)]
                        if op == "q_proj":
                            y = layer.self_attn.q_proj(x_gpu)
                        elif op == "k_proj":
                            y = layer.self_attn.k_proj(x_gpu)
                        elif op == "v_proj":
                            y = layer.self_attn.v_proj(x_gpu)
                        elif op == "o_proj":
                            y = layer.self_attn.o_proj(x_gpu)
                        elif op == "gate_proj":
                            y = layer.mlp.gate_proj(x_gpu)
                        elif op == "up_proj":
                            y = layer.mlp.up_proj(x_gpu)
                        elif op == "down_proj":
                            y = layer.mlp.down_proj(x_gpu)
                        else:
                            raise ValueError(f"Unknown op: {original_name}")
                    outputs_gpu[original_name] = y
                staged_outputs = {
                    name: self.stage_output_to_cpu(name, tensor)
                    for name, tensor in outputs_gpu.items()
                }
                result = {
                    "fused": False,
                    "outputs": staged_outputs,
                }
        # Wait for this stream only. After this point, all pinned outputs used
        # by result are ready for CPU/SHM access.
        self.finish_request()
        return result

# SERVER

def handle_client(conn, addr, worker, mm):
    print(f"Client connected: {addr}", flush=True)
    try:
        while True:
            req = recv_msg(conn)
            # Support both old "type" style and new "cmd" style.
            cmd = req.get("cmd", req.get("type"))
            if cmd == "ping":
                send_msg(
                    conn,
                    {
                        "ok": True,
                        "msg": "pong",
                        "fused_qkv": True,
                        "fused_gate_up": True,
                        "pinned_staging_buffers": True,
                        "dedicated_cuda_stream": True,
                        "event_scoped_sync": True,
                        "per_rpc_full_device_sync": False,
                        "per_rpc_shm_input_allocation": False,
                    },
                )
                continue
            if cmd in ("close", "shutdown"):
                send_msg(conn, {"ok": True})
                break
            if cmd == "linear_many_shm":
                try:
                    layer_idx = req.get("layer_idx", None)
                    if layer_idx is not None:
                        layer_idx = int(layer_idx)
                    op_names = req.get("op_names", req.get("ops"))
                    if op_names is None:
                        raise ValueError("Missing op_names/ops")
                    op_names = list(op_names)
                    in_meta = req["input"]
                    in_off = int(in_meta["offset"])
                    in_shape = tuple(in_meta["shape"])
                    in_dtype = in_meta["dtype"]
                    out_offsets = [
                        int(x) for x in req["out_offsets"]
                    ]
                    if len(out_offsets) != len(op_names):
                        raise ValueError(
                            "out_offsets count does not match op_names count"
                        )
                    x_cpu = worker.read_shm_into_pinned(
                        mm,
                        in_off,
                        in_shape,
                        in_dtype,
                    )
                    result = worker.linear_many(
                        layer_idx=layer_idx,
                        op_names=op_names,
                        x_cpu=x_cpu,
                    )
                    if result["fused"]:
                        # One GPU-to-CPU copy and one shared-memory write.
                        fused_meta = shm_write_tensor(
                            mm,
                            out_offsets[0],
                            result["tensor"],
                        )
                        send_msg(
                            conn,
                            {
                                "ok": True,
                                "fused": True,
                                "output": fused_meta,
                                "op_names": op_names,
                                "split_sizes": tuple(
                                    int(x) for x in result["split_sizes"]
                                ),
                            },
                        )
                    else:
                        out_metas = {}
                        for name, off in zip(op_names, out_offsets):
                            out_metas[name] = shm_write_tensor(
                                mm,
                                off,
                                result["outputs"][name],
                            )
                        send_msg(
                            conn,
                            {
                                "ok": True,
                                "fused": False,
                                "outputs": out_metas,
                            },
                        )
                except Exception as exc:
                    send_msg(
                        conn,
                        {
                            "ok": False,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                continue
            send_msg(
                conn,
                {
                    "ok": False,
                    "error": f"unknown cmd/type: {cmd}",
                },
            )
    except Exception as exc:
        print(
            f"Client error/disconnect: {addr} | {repr(exc)}",
            flush=True,
        )
    try:
        conn.close()
    except Exception:
        pass
    print(f"Client closed: {addr}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5010)
    args = parser.parse_args()
    if GPU_DEVICE.type != "cuda":
        raise RuntimeError("CUDA GPU is required for the GPU worker.")
    worker = GPUWorker()
    with open(SHM_PATH, "r+b", buffering=0) as f:
        mm = mmap.mmap(
            f.fileno(),
            SHM_SIZE,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )
        server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        server.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        server.bind((args.host, args.port))
        server.listen(1)
        print(
            f"T3 GPU worker listening "
            f"on {args.host}:{args.port}",
            flush=True,
        )
        print(f"Using host shm: {SHM_PATH}", flush=True)
        while True:
            conn, addr = server.accept()
            conn.setsockopt(
                socket.IPPROTO_TCP,
                socket.TCP_NODELAY,
                1,
            )
            handle_client(conn, addr, worker, mm)

if __name__ == "__main__":
    main()
