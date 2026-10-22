#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""GPU worker for T3 inference and activation capture.

CAPTURE_MODE labels captures as plain or noised.
The worker saves inputs received through IVSHMEM.
"""
import argparse
import json
import mmap
import os
import pickle
import socket
import struct
import traceback
import time

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
GPU_DTYPE = torch.float32

# Configuration
# Host-side IVSHMEM device/file used by the GPU worker.
SHM_PATH = "/dev/shm/shm1"

# Directory in which captured GPU-visible activations will be saved.
CAPTURE_DIR = "results/attacker_captures"

# Network configuration.
HOST = "0.0.0.0"
PORT = 5010

# "plain" for clean input; "noised" for masked input.
CAPTURE_MODE = "plain"

# Shared-memory size.
SHM_SIZE = 8 * 1024 * 1024 * 1024

# Capture configuration.
CAPTURE_LAYER = 0
CAPTURE_POINT = "qkv_input_after_input_layernorm"

# Set True only when existing capture files should be replaced.
OVERWRITE_CAPTURES = True
SUPPORTED_CAPTURE_POINTS = {
    "qkv_input_after_input_layernorm",
    "o_proj_input",
    "gate_up_input_after_post_attention_layernorm",
    "lm_head_input_after_final_norm",
}
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


def recv_exact(sock, n):
    chunks = []
    remaining = int(n)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_msg(sock):
    header = recv_exact(sock, 8)
    size = struct.unpack("!Q", header)[0]
    return pickle.loads(recv_exact(sock, size))


def send_msg(sock, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def numel(shape):
    value = 1
    for dim in shape:
        value *= int(dim)
    return int(value)


def read_shm_tensor(mm, meta):
    offset = int(meta["offset"])
    shape = tuple(int(x) for x in meta["shape"])
    dtype_name = str(meta["dtype"])
    nbytes = int(meta["nbytes"])
    expected = numel(shape) * np.dtype(
        _DTYPE_TO_NUMPY[dtype_name]
    ).itemsize
    if expected != nbytes:
        raise RuntimeError(
            f"Input nbytes mismatch: expected={expected}, got={nbytes}"
        )
    view = memoryview(mm)[offset:offset + nbytes]
    array = np.frombuffer(
        view,
        dtype=_DTYPE_TO_NUMPY[dtype_name],
    ).reshape(shape)
    # Copy immediately: TDX reuses the same input SHM region on the next RPC.
    return torch.from_numpy(array.copy()).contiguous()


def write_shm_tensor(mm, offset, tensor):
    tensor = tensor.detach().to("cpu").contiguous()
    dtype_name = _TORCH_TO_DTYPE[tensor.dtype]
    array = tensor.numpy()
    nbytes = int(array.nbytes)
    end = int(offset) + nbytes
    if end > len(mm):
        raise RuntimeError(
            f"SHM output overflow: offset={offset}, nbytes={nbytes}"
        )
    mm[int(offset):end] = memoryview(array).cast("B")
    return {
        "shape": tuple(int(x) for x in tensor.shape),
        "dtype": dtype_name,
        "nbytes": nbytes,
        "offset": int(offset),
    }


def capture_point_for_request(layer_idx, op_names):
    names = list(op_names)
    if names == ["q_proj", "k_proj", "v_proj"]:
        return "qkv_input_after_input_layernorm"
    if names == ["o_proj"]:
        return "o_proj_input"
    if names == ["gate_proj", "up_proj"]:
        return "gate_up_input_after_post_attention_layernorm"
    if names == ["lm_head"]:
        return "lm_head_input_after_final_norm"
    return None


class CaptureWriter:
    def __init__(
        self,
        capture_dir,
        capture_layer,
        capture_point,
        capture_mode,
        overwrite=False,
    ):
        if capture_point not in SUPPORTED_CAPTURE_POINTS:
            raise ValueError(
                f"Unsupported capture point: {capture_point}"
            )
        self.capture_mode = str(capture_mode).lower().strip()
        if self.capture_mode not in {"plain", "noised"}:
            raise ValueError(
                'capture_mode must be "plain" or "noised".'
            )
        self.condition = (
            "plain_t3"
            if self.capture_mode == "plain"
            else "noised_t3"
        )
        self.masked = self.capture_mode == "noised"
        self.capture_dir = Path(capture_dir)
        self.samples_dir = self.capture_dir / "samples"
        self.capture_layer = (
            None
            if capture_point == "lm_head_input_after_final_norm"
            else int(capture_layer)
        )
        self.capture_point = capture_point
        self.overwrite = bool(overwrite)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.saved_ids = set()
        for path in self.samples_dir.glob("*.pt"):
            self.saved_ids.add(path.stem)
        self.manifest_path = (
            self.capture_dir / f"{self.capture_mode}_capture_manifest.jsonl"
        )
    def request_matches(self, layer_idx, op_names, x_cpu, context):
        if not context:
            return False
        capture_enabled = bool(
            context.get(
                "enabled",
                context.get("capture_enabled", False),
            )
        )
        if not capture_enabled:
            return False
        request_point = capture_point_for_request(
            layer_idx,
            op_names,
        )
        if request_point != self.capture_point:
            return False
        effective_layer = (
            None
            if request_point == "lm_head_input_after_final_norm"
            else int(layer_idx)
        )
        if effective_layer != self.capture_layer:
            return False
        # Capture prefill only. Decode RPC inputs have sequence length 1.
        if x_cpu.ndim < 2 or int(x_cpu.shape[-2]) <= 1:
            return False
        return True
    def save_batch(
        self,
        layer_idx,
        op_names,
        x_cpu,
        context,
    ):
        if not self.request_matches(
            layer_idx,
            op_names,
            x_cpu,
            context,
        ):
            return 0
        if x_cpu.ndim == 2:
            x_cpu = x_cpu.unsqueeze(0)
        if x_cpu.ndim != 3:
            raise RuntimeError(
                f"Expected captured input [B,T,D], got {tuple(x_cpu.shape)}"
            )
        sample_ids = list(context.get("sample_ids") or [])
        token_lengths = [
            int(x) for x in (
                context.get("prompt_token_lengths") or []
            )
        ]
        batch_size = int(x_cpu.shape[0])
        padded_length = int(x_cpu.shape[1])
        if len(sample_ids) != batch_size:
            raise RuntimeError(
                "Capture sample-ID count does not match tensor batch size: "
                f"ids={len(sample_ids)}, batch={batch_size}"
            )
        if len(token_lengths) != batch_size:
            raise RuntimeError(
                "Capture token-length count does not match tensor batch size: "
                f"lengths={len(token_lengths)}, batch={batch_size}"
            )
        saved = 0
        for batch_index, (
            sample_id,
            prompt_token_length,
        ) in enumerate(zip(sample_ids, token_lengths)):
            if prompt_token_length <= 0:
                raise RuntimeError(
                    f"Invalid token length for {sample_id}: "
                    f"{prompt_token_length}"
                )
            if prompt_token_length > padded_length:
                raise RuntimeError(
                    f"Token length exceeds padded tensor length for "
                    f"{sample_id}: {prompt_token_length}>{padded_length}"
                )
            sample_path = self.samples_dir / f"{sample_id}.pt"
            if (
                sample_id in self.saved_ids
                and not self.overwrite
            ):
                continue
            # Tokenizer uses left padding, so valid prompt positions are at end.
            hidden = (
                x_cpu[
                    batch_index,
                    padded_length - prompt_token_length:,
                    :,
                ]
                .detach()
                .to(dtype=torch.float32, device="cpu")
                .contiguous()
                .clone()
            )
            payload = {
                "sample_id": sample_id,
                "hidden": hidden,
                "meta": {
                    "model_name": MODEL_NAME,
                    "condition": self.condition,
                    "capture_mode": self.capture_mode,
                    "masked": self.masked,
                    "capture_tensor": "operator_input",
                    "capture_point": self.capture_point,
                    "layer_idx": self.capture_layer,
                    "op_names": list(op_names),
                    "phase": "prefill",
                    "prompt_token_length": prompt_token_length,
                    "padded_batch_length": padded_length,
                    "batch_size": batch_size,
                    "batch_row": batch_index,
                    "hidden_size": int(hidden.shape[-1]),
                    "stored_shape": tuple(
                        int(x) for x in hidden.shape
                    ),
                    "dtype": str(hidden.dtype),
                    "eps_mode": context.get("eps_mode"),
                    "custom_epsilon": context.get(
                        "custom_epsilon"
                    ),
                    "generation_cap": context.get(
                        "generation_cap"
                    ),
                    "runtime_batch_index": context.get(
                        "batch_index"
                    ),
                    "captured_at_gpu_unix_time": time.time(),
                    "exact_tensor_received_by_gpu": True,
                },
            }
            torch.save(payload, sample_path)
            manifest_row = {
                "sample_id": sample_id,
                "sample_path": str(sample_path),
                "prompt_token_length": prompt_token_length,
                "hidden_size": int(hidden.shape[-1]),
                "capture_point": self.capture_point,
                "capture_layer": self.capture_layer,
                "condition": self.condition,
                "capture_mode": self.capture_mode,
                "masked": self.masked,
            }
            with open(
                self.manifest_path,
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        manifest_row,
                        ensure_ascii=False,
                    ) + "\n"
                )
            self.saved_ids.add(sample_id)
            saved += 1
            print(
                f"[CAPTURED] {sample_id} | "
                f"point={self.capture_point} | "
                f"layer={self.capture_layer} | "
                f"shape={tuple(hidden.shape)}",
                flush=True,
            )
        return saved


class GPUOperatorExecutor:
    def __init__(self, model_name):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this GPU worker")
        self.device = torch.device("cuda")
        print(f"Loading GPU model: {model_name}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=GPU_DTYPE,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.layers = self.model.model.layers
        print(
            f"GPU model loaded on {self.device}; "
            f"layers={len(self.layers)}",
            flush=True,
        )
    def module_for(self, layer_idx, op_name):
        if op_name == "lm_head":
            return self.model.lm_head
        if layer_idx is None:
            raise ValueError(
                f"layer_idx is required for operation {op_name}"
            )
        layer = self.layers[int(layer_idx)]
        if op_name == "q_proj":
            return layer.self_attn.q_proj
        if op_name == "k_proj":
            return layer.self_attn.k_proj
        if op_name == "v_proj":
            return layer.self_attn.v_proj
        if op_name == "o_proj":
            return layer.self_attn.o_proj
        if op_name == "gate_proj":
            return layer.mlp.gate_proj
        if op_name == "up_proj":
            return layer.mlp.up_proj
        raise ValueError(f"Unsupported operation: {op_name}")
    @torch.inference_mode()
    def execute(self, layer_idx, op_names, x_cpu):
        x_gpu = x_cpu.to(
            device=self.device,
            dtype=GPU_DTYPE,
            non_blocking=False,
        )
        outputs = []
        for op_name in op_names:
            module = self.module_for(
                layer_idx,
                op_name,
            )
            # F.linear exactly matches nn.Linear with the model's weight/bias.
            out = F.linear(
                x_gpu,
                module.weight,
                module.bias,
            )
            outputs.append(out)
        if op_names in (
            ["q_proj", "k_proj", "v_proj"],
            ["gate_proj", "up_proj"],
        ):
            fused = torch.cat(outputs, dim=-1)
            split_sizes = [
                int(output.shape[-1])
                for output in outputs
            ]
            return {
                "fused": True,
                "tensor": fused.to("cpu").contiguous(),
                "split_sizes": split_sizes,
            }
        return {
            "fused": False,
            "outputs": {
                name: output.to("cpu").contiguous()
                for name, output in zip(
                    op_names,
                    outputs,
                )
            },
        }


def handle_client(
    connection,
    mm,
    executor,
    capture_writer,
):
    while True:
        request = recv_msg(connection)
        command = request.get("cmd")
        if command == "ping":
            send_msg(
                connection,
                {
                    "ok": True,
                    "fused_qkv": True,
                    "fused_gate_up": True,
                    "attack_capture": True,
                    "condition": capture_writer.condition,
                    "capture_mode": capture_writer.capture_mode,
                    "masked": capture_writer.masked,
                    "capture_point":
                        capture_writer.capture_point,
                    "capture_layer":
                        capture_writer.capture_layer,
                },
            )
            continue
        if command == "close":
            send_msg(connection, {"ok": True})
            return
        if command != "linear_many_shm":
            send_msg(
                connection,
                {
                    "ok": False,
                    "error": f"Unsupported command: {command}",
                },
            )
            continue
        try:
            layer_idx = request.get("layer_idx")
            op_names = list(request["op_names"])
            input_meta = request["input"]
            out_offsets = [
                int(x) for x in request["out_offsets"]
            ]
            # Support both TDX-client metadata conventions.
            capture_context = request.get("capture_context")
            if capture_context is None:
                capture_context = request.get("attack_meta")
            # Exact operator-input tensor received by the GPU: x in plain mode,
            # or x + eta in noised mode.
            x_gpu_input_cpu = read_shm_tensor(
                mm,
                input_meta,
            )
            # LM-head diagnostic: confirms whether this RPC carries the full
            # prefill sequence [B,T,D] or only a single position [B,1,D].
            if "lm_head" in op_names:
                capture_enabled = None
                if capture_context is not None:
                    capture_enabled = capture_context.get(
                        "capture_enabled",
                        capture_context.get("enabled"),
                    )
                print(
                    f"[LM_HEAD RPC] layer={layer_idx} "
                    f"shape={tuple(x_gpu_input_cpu.shape)} "
                    f"capture_enabled={capture_enabled}",
                    flush=True,
                )
            capture_writer.save_batch(
                layer_idx=layer_idx,
                op_names=op_names,
                x_cpu=x_gpu_input_cpu,
                context=capture_context,
            )
            result = executor.execute(
                layer_idx=layer_idx,
                op_names=op_names,
                x_cpu=x_gpu_input_cpu,
            )
            if result["fused"]:
                output_meta = write_shm_tensor(
                    mm,
                    out_offsets[0],
                    result["tensor"],
                )
                response = {
                    "ok": True,
                    "fused": True,
                    "output": output_meta,
                    "op_names": op_names,
                    "split_sizes": result[
                        "split_sizes"
                    ],
                }
            else:
                output_metas = {}
                for index, op_name in enumerate(op_names):
                    output_metas[op_name] = (
                        write_shm_tensor(
                            mm,
                            out_offsets[index],
                            result["outputs"][op_name],
                        )
                    )
                response = {
                    "ok": True,
                    "fused": False,
                    "outputs": output_metas,
                }
            send_msg(connection, response)
        except Exception as error:
            send_msg(
                connection,
                {
                    "ok": False,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default=HOST,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
    )
    parser.add_argument(
        "--shm-path",
        type=str,
        default=SHM_PATH,
    )
    parser.add_argument(
        "--shm-size",
        type=int,
        default=SHM_SIZE,
    )
    parser.add_argument(
        "--capture-dir",
        type=str,
        default=CAPTURE_DIR,
    )
    parser.add_argument(
        "--capture-layer",
        type=int,
        default=CAPTURE_LAYER,
    )
    parser.add_argument(
        "--capture-point",
        type=str,
        default=CAPTURE_POINT,
        choices=sorted(SUPPORTED_CAPTURE_POINTS),
    )
    parser.add_argument(
        "--overwrite-captures",
        action="store_true",
        default=OVERWRITE_CAPTURES,
    )
    args = parser.parse_args()
    if not os.path.exists(args.shm_path):
        raise FileNotFoundError(
            f"SHM path not found: {args.shm_path}"
        )
    capture_writer = CaptureWriter(
        capture_dir=args.capture_dir,
        capture_layer=args.capture_layer,
        capture_point=args.capture_point,
        capture_mode=CAPTURE_MODE,
        overwrite=args.overwrite_captures,
    )
    executor = GPUOperatorExecutor(MODEL_NAME)
    print("=" * 88)
    print("T3 GPU WORKER WITH SWITCHABLE ATTACK CAPTURE")
    print("=" * 88)
    print(f"Host             : {args.host}")
    print(f"Port             : {args.port}")
    print(f"SHM path         : {args.shm_path}")
    print(f"SHM size         : {args.shm_size}")
    print(f"Capture dir      : {args.capture_dir}")
    print(f"Capture mode     : {capture_writer.capture_mode}")
    print(f"Condition        : {capture_writer.condition}")
    print(f"Masked           : {capture_writer.masked}")
    print(f"Capture layer    : {capture_writer.capture_layer}")
    print(f"Capture point    : {capture_writer.capture_point}")
    print("=" * 88)
    with open(args.shm_path, "r+b", buffering=0) as shm_file:
        mm = mmap.mmap(
            shm_file.fileno(),
            int(args.shm_size),
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )
        server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        server.bind((args.host, int(args.port)))
        server.listen(1)
        print(
            f"Listening on {args.host}:{args.port} ...",
            flush=True,
        )
        try:
            while True:
                connection, address = server.accept()
                print(
                    f"Client connected: {address}",
                    flush=True,
                )
                with connection:
                    connection.setsockopt(
                        socket.IPPROTO_TCP,
                        socket.TCP_NODELAY,
                        1,
                    )
                    try:
                        handle_client(
                            connection=connection,
                            mm=mm,
                            executor=executor,
                            capture_writer=capture_writer,
                        )
                    except Exception:
                        traceback.print_exc()
                print(
                    f"Client disconnected: {address}",
                    flush=True,
                )
        finally:
            server.close()
            mm.close()
if __name__ == "__main__":
    main()
