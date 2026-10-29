# M0 spike: training Jason's nightly LoRA on the RTX 4090 Laptop

**Questions:**
1. **Option B:** can the 4B (Qwen3.5-4B) be served in bf16 *and* trained in bf16
   on this laptop, with vLLM asleep while it trains? How fast is it, compared
   with the 9B's serving numbers?
2. **Option C:** does a 9B QLoRA (4-bit base) fit and train? Unsloth advises
   against QLoRA for Qwen3.5, so its quality is measured, not assumed.
3. **Does learning survive serving?** Each adapter learns one easy-to-check
   habit (ending every reply with a "Curious: ...?" line). Then we count how
   often it shows on 50 held-out questions: on the same base it was trained on,
   on a 4-bit (AWQ) version of that base, and for the 9B, on the AWQ 9B.
4. **SQLite** write speed on the `C:\` mount vs WSL2's own disk (report only).

Budget about **45 GB of disk** and 2–3 hours, mostly downloads: the Unsloth
image (7 GB), the 4B (9.3 GB), its AWQ version (5.2 GB) and the full 9B for
QLoRA (19.3 GB). The 9B AWQ is already in the serving spike's cache.

Run everything in **PowerShell on the Windows laptop, from the repo folder**
(`cd $HOME\jason`). Set this once per window:
```
$T = "spikes/m0-lora-training/compose.yaml"
```

## 1. Prepare

```
git pull
docker ps
```
If the serving spike's vLLM is still running, stop it:
`docker compose -f spikes/m0-llm-serving/compose.vllm.yaml down`. The Jason
brain container can keep running (it uses no GPU).

**Give WSL2 more memory.** By default it gets half of your 31.6 GB. vLLM's sleep
mode keeps its weights in that memory while training needs its own. First check
whether the file already exists:
```
Get-Content $HOME\.wslconfig
```
If it says the file doesn't exist, create it:
```
Set-Content -Path $HOME\.wslconfig -Value "[wsl2]`nmemory=24GB`nswap=8GB" -Encoding ascii
```
(If it does exist, paste me its contents instead.) Then restart WSL2: run
`wsl --shutdown`, quit Docker Desktop, start it again, and check:
```
docker run --rm python:3.12-slim-bookworm head -1 /proc/meminfo
```
It should say about 24,000,000 kB.

## 2. Serve the 4B (option B's live model)

```
docker compose -f $T up -d vllm
docker compose -f $T logs -f vllm
```
The first start downloads the 4B (9.3 GB). Wait for **`Application startup
complete`**, then Ctrl+C and record memory:
```
docker compose -f $T logs vllm | Select-String -Pattern "Model loading took","Available KV cache","GPU KV cache size"
nvidia-smi
```

## 3. Benchmark the 4B
```
uv run python spikes/m0-llm-serving/bench.py --label vllm-4b --model jason-llm
```

## 4. Make the training data
Asks the running 4B 250 questions (a few minutes):
```
uv run python spikes/m0-lora-training/make_data.py
```

## 5. Measure the base model (before any training)
```
uv run python spikes/m0-lora-training/eval.py --label 4b-base
```
Expect about 0%: the base model never ends with a "Curious:" line.

## 6. Put vLLM to sleep
```
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/sleep?level=1"
nvidia-smi
```
GPU memory in use should drop below about 1 GB.

## 7. Train option B: the 4B in bf16
The first run downloads the Unsloth image (7 GB):
```
docker compose -f $T run --rm train python /spike/train.py --model Qwen/Qwen3.5-4B --bits 16 --name 4b-bf16
```

## 8. Wake vLLM and test the adapter on the same base
```
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/wake_up"
uv run python spikes/m0-lora-training/eval.py --label 4b-bf16-on-bf16 --adapter 4b-bf16
```

## 9. The same adapter on a 4-bit (AWQ) 4B
Would serving a smaller 4B break what was learned?
```
docker compose -f $T down
$env:MODEL = "LostGentoo/Qwen3.5-4B-AWQ"; $env:REVISION = "e3d68b9671984326b5ebb9f0914e5864848c6853"
docker compose -f $T up -d vllm
docker compose -f $T logs -f vllm
```
After `Application startup complete` (downloads 5.2 GB first), Ctrl+C, then:
```
uv run python spikes/m0-lora-training/eval.py --label 4b-bf16-on-awq --adapter 4b-bf16 --compare 4b-bf16-on-bf16
uv run python spikes/m0-llm-serving/bench.py --label vllm-4b-awq --model jason-llm
```

## 10. Train option C: the 9B with QLoRA
This needs the whole GPU, so stop vLLM first. The first run downloads the full
9B (19.3 GB) and turns it into 4-bit while loading.
```
docker compose -f $T down
docker compose -f $T run --rm train python /spike/train.py --model Qwen/Qwen3.5-9B --bits 4 --name 9b-qlora
```
If it runs out of memory, that is a result: paste the error and go on to step 12.

## 11. Test the 9B adapter on the AWQ 9B
```
$env:MODEL = "cyankiwi/Qwen3.5-9B-AWQ-4bit"; $env:REVISION = "e4d2d39cede8c5d1716d824cd843e2889c648083"
docker compose -f $T up -d vllm
docker compose -f $T logs -f vllm
```
After `Application startup complete`, Ctrl+C, then:
```
uv run python spikes/m0-lora-training/eval.py --label 9b-base
uv run python spikes/m0-lora-training/eval.py --label 9b-qlora-on-awq --adapter 9b-qlora
```

## 12. Clean up the model choice
```
docker compose -f $T down
Remove-Item Env:MODEL, Env:REVISION
```

## 13. SQLite write speed
```
docker compose -f $T run --rm sqlite
Remove-Item -Recurse $HOME\jason-data\sqlite-bench
```

## What to send back
The "paste this block" output of steps **3, 5, 7, 8, 9, 10, 11 and 13**, the
memory lines and `nvidia-smi` from **2** and **6**, and any errors with their
step number. Everything is also saved in `spikes/m0-lora-training/results/`.

## If something goes wrong

| Message | What to do |
|---|---|
| `external volume "m0-llm-serving_hf-cache" not found` | `docker volume create m0-llm-serving_hf-cache` (the 9B AWQ will download again in step 11) |
| Training: `CUDA out of memory` in step 7 | Check step 6: `nvidia-smi` must show vLLM asleep. Then retry with `--seq 512` added |
| Training: an error mentioning `unsloth`, `trl` or `SFTConfig` | The libraries changed; paste the error |
| `loading the adapter failed` | Paste it, plus the last 40 lines of `docker compose -f $T logs vllm` |
| vLLM: `No available memory for the cache blocks` | Something else holds the GPU: `docker ps`, stop it |
