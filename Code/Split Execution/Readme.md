# Runtime experiments

## Files

**Host**
- `CPU_Runtime.py`: LLM inference runtime on the host CPU.
- `GPU_Runtime.py`: LLM inference runtime on the host GPU.
- `Worker.py`: Executes outsourced LLM operations on the GPU for Noised Inference (N3).

**TDX**
- `Precompute_Noise.py`: Generates Gaussian noise and correction banks for the outsourced operations.
- `Runtime_Noise.py`: N3 inference inside TDX using precomputed noise and GPU offloading through shared memory.

## Running Noised Version 

First, use matching model, epsilon, batch size, and token settings in `Precompute_Noise.py` and `Runtime_Noise.py`. Set the bank output directory and run:

```bash
python3 Precompute_Noise.py
```

Set the bank path in `Runtime_Noise.py` to the generated bank directory.

Check the shared-memory paths in both scripts. On the host, `Worker.py` uses `/dev/shm/shm1`. Inside TDX, `Runtime_Noise.py` uses `/sys/bus/pci/devices/0000:00:02.0/resource2`. They must refer to the same IVSHMEM region; the TDX PCI address may differ.

Start the GPU worker on the host:

```bash
python3 Worker.py --host 0.0.0.0 --port <number>
```

Then, inside TDX, run:

```bash
python3 TDX/Runtime_Noise.py --gpu-server <HOST_IP> --port <number>
```

Replace `<HOST_IP>` with the host IP address reachable from TDX.

## Baselines

Run CPU_Runtime.py on the host to measure CPU runtime, or inside the TDX VM to measure TDX runtime. 
Run GPU_Runtime.py on the host to measure GPU runtime.

```bash
python3 CPU_Runtime.py
python3 GPU_Runtime.py
```
