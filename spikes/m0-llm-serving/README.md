# M0 spike: serving Qwen3.5-9B on the RTX 4090 Laptop

**Question:** can vLLM serve Qwen3.5-9B (4-bit) on the 4090's 16 GB inside a
container, with a LoRA adapter loaded while it runs? How much GPU memory does
it take, and how fast is the first token? llama.cpp is measured too, as an
optional baseline. The write-up goes in `docs/spikes/m0-llm-serving.md`.

| | vLLM (main candidate) | llama.cpp (optional baseline) |
|---|---|---|
| Image | `vllm/vllm-openai:v0.32.0` (~10 GB) | `llama.cpp:server-cuda`, 2026-10-29 build (~2 GB) |
| Model | `cyankiwi/Qwen3.5-9B-AWQ-4bit` (~9 GB) | `unsloth/Qwen3.5-9B-GGUF` Q4_K_M (~5.7 GB) |
| Tested | memory, time to first token, speed, runtime LoRA load/swap, sleep mode | memory, time to first token, speed |

Budget about **30 GB of disk** and an hour, mostly downloads. Run everything in
**PowerShell on the Windows laptop, from the repo folder** (`cd $HOME\jason`).
The Jason brain container can keep running; it uses the CPU only.

Throughout, `$C` stands for the vLLM compose file. Set it once per window:
```
$C = "spikes/m0-llm-serving/compose.vllm.yaml"
```

## 1. Check the NVIDIA driver
```
nvidia-smi
```
Look at **CUDA Version** in the top right of the table:
- **13.0 or higher:** nothing to do.
- **12.9:** use the CUDA 12.9 build of vLLM. Run this in every new PowerShell window before the `docker compose` commands:
  ```
  $env:VLLM_IMAGE = "vllm/vllm-openai:v0.32.0-cu129@sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f"
  ```
- **Lower:** update the driver with the NVIDIA app (or GeForce Experience), restart, and check again.

## 2. Get the kit
```
git pull
```

## 3. Check that Docker can see the GPU
The first time, this downloads the vLLM image (~10 GB):
```
docker compose -f $C run --rm --entrypoint nvidia-smi vllm -L
```
It should print one line like `GPU 0: NVIDIA GeForce RTX 4090 Laptop GPU (UUID: ...)`,
seen from inside the container. (The `-L` replaces vLLM's own arguments, which
`nvidia-smi` wouldn't understand.)

## 4. Start vLLM
```
docker compose -f $C up -d
docker compose -f $C logs -f vllm
```
The first start downloads the model (~9 GB), then loads it onto the GPU and
prepares it. Wait for **`Application startup complete`**, then press Ctrl+C to
stop following the logs (the server keeps running).

## 5. Record GPU memory
```
docker compose -f $C logs vllm | Select-String -Pattern "Model loading took","KV cache","Maximum concurrency","graph capturing"
nvidia-smi
```

## 6. Benchmark
```
uv run python spikes/m0-llm-serving/bench.py --label vllm
```

## 7. Load a LoRA adapter while the server runs
Downloads a 116 MB Unsloth-trained adapter, then loads, uses, unloads and swaps it:
```
uv run python spikes/m0-llm-serving/lora_check.py
```

## 8. Optional: sleep mode
Does putting the model to sleep free the GPU for nightly training?
```
uv run python spikes/m0-llm-serving/sleep_check.py
```

## 9. Startup time with the model already downloaded
This is how long Jason's brain would take to come back after a restart:
```
docker compose -f $C restart vllm; uv run python spikes/m0-llm-serving/bench.py --wait-only
```

## 10. Stop vLLM
```
docker compose -f $C down
```
The downloaded model stays in a Docker volume for next time.

## 11–13. Optional: llama.cpp baseline
First make sure vLLM is stopped (step 10): the two servers won't both fit on the
GPU. This should list no running containers:
```
docker compose -f $C ps
```
Then:
```
$L = "spikes/m0-llm-serving/compose.llamacpp.yaml"
docker compose -f $L up -d
docker compose -f $L logs -f llamacpp
```
The first start downloads the model (~5.7 GB), which takes several minutes and
shows little in the log. Wait for **`server is listening`**, then Ctrl+C and:
```
uv run python spikes/m0-llm-serving/bench.py --label llamacpp --url http://127.0.0.1:8080
nvidia-smi
docker compose -f $L down
```

## What to send back
The output of steps **1** (the driver and CUDA version line), **5, 6, 7, 8, 9**
and **12**. Copy the "paste this block" sections; the full results are also
saved in `spikes/m0-llm-serving/results/`.

## If something goes wrong

| Message | What to do |
|---|---|
| `unrecognized arguments: --language-model-only` | Delete that line from `compose.vllm.yaml`, run step 4 again, and tell me |
| `no CUDA-capable device`, `driver ... too old`, `CUDA driver version is insufficient` | Step 1: the driver is older than the image needs |
| `CUDA out of memory`, `No available memory for the cache blocks` | In `compose.vllm.yaml`, change `--max-model-len=8192` to `4096`; if it still fails, `--gpu-memory-utilization=0.80` to `0.85` |
| `AssertionError` mentioning `causal_conv1d_update` | Add a line `- --max-cudagraph-capture-size=8` under `command:` |
| The LoRA check fails at "load adapter" | Paste its output and the last 40 lines of `docker compose -f $C logs vllm` |
| Anything else | Paste the error and the step number |

When you're done, you can free the disk space with `docker volume ls` and
`docker volume rm` on the `m0-llm-serving_...` volumes. Keep the vLLM one if
vLLM is chosen: M1 will use the same model.
