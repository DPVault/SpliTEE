## Prompt Reconstruction Attacker

Code for training and evaluating the prompt-reconstruction attacker used in the SpliTEE experiments. Training uses auxiliary prompts and captured model activations. Testing uses activations collected during clean or noised T3 inference.


### Training
1. Run host/Dataset.ipynb to prepare auxiliary prompts and capture clean training activations.
2. For noise-aware training, run host/Prepare_Noised_Data.py to generate noisy training activations.
3. Set the training and validation manifest paths in host/Attacker_train.py, then run it to train the attacker.

### Testing

1. Start host/GPU_Attack_capture_worker.py on the GPU host. Set CAPTURE_MODE to "plain" for clean captures or "noised" for masked captures.
2. For clean T3, run Tdx/TDX_Clean_Attack_Capture.py inside the TDX guest.
3. For noised T3, first run Tdx/TDX_Precompute_Attack_Noise.py to generate the noise banks, then run Tdx/TDX_Attack_Capture.py inside the guest.
4. The GPU worker saves the activations. The TDX clients save prompt labels separately. Copy the required captures and labels to the evaluation environment, set their paths and the trained checkpoint path in host/Attacker_test.py, and run the evaluation.

### Configuration
Before running, check the model and dataset paths, capture point and layer, output directories, GPU host address and port, and IVSHMEM device paths. For noised T3, the epsilon, batch size, prompt selection, and noise-bank directory must match between precomputation and capture. The clean and noised clients have separate settings and should be configured for the experiment being reproduced.
