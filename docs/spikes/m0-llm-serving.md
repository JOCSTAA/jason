# Spike: serving Qwen3.5-9B on the RTX 4090 Laptop (M0)

_Run 2026-10-29 on the Windows laptop. Kit and raw commands: `spikes/m0-llm-serving/`._

## Question

Which engine should serve the live LLM (design §3.3 model 6): vLLM or llama.cpp
server? Requirements: runs Qwen3.5-9B at 4 bits on 16 GB inside a container
(portability rule 3), loads tonight's LoRA adapter without a restart (design
§5.1 slow loop), and leaves time-to-first-token inside the §10 budget
(100–200 ms for the LLM).

## Recommendation

**vLLM v0.32.0, serving the 4-bit AWQ model.** It meets every requirement on
the first attempt, loads Unsloth's LoRA output directly while running, and its
sleep mode frees the GPU for nightly training without stopping the server.
Those two are what Jason's learning loop (priority 1) depends on.

The llama.cpp baseline is better on two counts, and the decision accepts both
costs for now:

- **GPU memory:** about 3.7 GB less (7.6 vs 11.3 GB in use). Part of the gap is
  vLLM's up-front reservations, which can be shrunk (see M1 below); the rest is
  the AWQ upload keeping its large vocabulary tables in 16-bit.
- **Time to first token on a cached prompt:** 81 vs 127 ms. Both are inside the
  100–200 ms budget.

**Revisit** if GPU memory becomes the limit once the embedding model, reaction
judge and Wonder share the GPU (M2–M3). Both engines speak the same
OpenAI-compatible API, so switching would mean the adapter conversion step and a
morning restart, not a rewrite.

## Setup

| | |
|---|---|
| GPU | RTX 4090 Laptop, 16,376 MiB; NVIDIA driver 617.42 (CUDA 13.4). The previous driver (CUDA 12.7) was too old for the vLLM image |
| Container | `vllm/vllm-openai:v0.32.0` (CUDA 13.0), Docker Desktop on WSL2 |
| Model | `cyankiwi/Qwen3.5-9B-AWQ-4bit`, revision `e4d2d39` (community upload; no official 4-bit release for the 9B) |
| Settings | `--language-model-only` (no vision encoder), 8,192-token context, at most 4 sequences, `--gpu-memory-utilization 0.80`, LoRA rank ≤ 64, thinking disabled, sleep mode on |
| Prompt | Core blocks + wake-up briefing + 4 turns, about 1,500 tokens; 96-token replies, temperature 0 |

## Results (vLLM)

**GPU memory**

| | |
|---|---|
| Weights | 8.37 GiB |
| Peak activations | 2.17 GiB |
| CUDA graphs | 0.17 GiB |
| KV cache | 2.15 GiB = 49,524 tokens (6× the 8,192-token context) |
| Total in use (nvidia-smi) | 11,246–11,562 MiB of 16,376 MiB; about 4.7 GB free |

The hybrid architecture (Gated DeltaNet layers with occasional full attention)
makes the cache cheap: about 45 KiB per token.

**Latency and speed**

Two full runs; medians of each:

| | Run 1 | Run 2 | Range over all requests |
|---|---|---|---|
| Time to first token, shared prefix | **125 ms** | **129 ms** | 117–139 ms |
| Time to first token, new prompt | 466 ms | 462 ms | 455–486 ms |
| Generation | **81 tokens/s** | 81 tokens/s | 80.6–81.5 |
| Startup, model already downloaded | 40 s | 38 s | |
| First start (download, load, compile) | about 3 min | | load 95 s + init 86 s |

**Runtime LoRA** (Unsloth-trained `pankajpandey-dev/qwen3.5-9b-hindi-instruct-lora`, rank 16, regex `target_modules`): 10/10 steps passed in both runs.

| | Run 1 | Run 2 |
|---|---|---|
| `POST /v1/load_lora_adapter` | 1.3 s | 0.8 s |
| Load again under a new name (the morning swap) | 1.1 s | 0.55 s |
| `POST /v1/unload_lora_adapter` | 2 ms | 2 ms |
| Time to first token, base vs adapter (median of 3, short prompt) | 72 vs 69 ms | 58 vs 84 ms |

Replies switch to Hindi with the adapter, and are identical under the second
name (same weights). The adapter's cost to time-to-first-token is 0–26 ms in
these small samples; measure it properly once Jason's own adapters exist.

**Sleep mode** (`POST /sleep?level=1`, `POST /wake_up`)

| | |
|---|---|
| GPU memory | 11,564 MiB awake → **466 MiB asleep** → 11,246 MiB awake |
| Time | sleep 4.9 s, wake 1.5 s |
| First reply after waking | 181 ms to first token |

## Results (llama.cpp baseline)

`ghcr.io/ggml-org/llama.cpp:server-cuda` (2026-10-29 build, CUDA 12.8) with
`unsloth/Qwen3.5-9B-GGUF` Q4_K_M (5.7 GB), all layers on the GPU, 4 slots ×
8,192 tokens, reasoning off. Same prompts and script as vLLM. LoRA and sleep
weren't tested: llama-server loads adapters only at startup (after converting
them to GGUF), and has no sleep mode.

| | vLLM | llama.cpp |
|---|---|---|
| GPU memory in use | 11,246–11,562 MiB | **7,616 MiB** (includes a vision projector it loaded automatically; `--no-mmproj` would drop it) |
| Time to first token, shared prefix | 125–129 ms | **81 ms** (78–84) |
| Time to first token, new prompt | **462–466 ms** | 632 ms (575–708) |
| Generation | 81 tokens/s | 80 tokens/s |
| Load a new adapter while running | **0.5–1.3 s** | Not possible; restart (model loaded in about 4 s once downloaded) |
| Free the GPU without stopping | **Sleep: 11.6 → 0.5 GB** | Stop the server |

## What this means for later milestones

- **M1, prompt layout:** stable text first (core blocks, then the briefing,
  then the conversation), so every turn hits the prefix cache. A new prompt is
  3.7× slower to start (466 vs 125 ms).
- **M1, memory budget:** the KV cache holds 6× what Jason needs. Setting
  `--kv-cache-memory` to about 1 GiB keeps roughly 22,000 tokens (plenty for
  one user plus judge and Wonder calls) and frees about 1 GiB more for the
  embedding model (design §3.3 model 8).
- **M1, container layout:** vLLM runs as its own container next to the brain,
  reached on the internal Docker network. Its port is never published:
  runtime LoRA loading and the sleep endpoints are unauthenticated.
- **M6/M8, sleep:** level-1 sleep frees about 11 GB in 5 s and wakes in 1.5 s,
  so nightly training can run on the same GPU without restarting the server.
  Level 1 copies the weights to system RAM, so the WSL2 VM needs about 9 GB
  of RAM free.
- **M8, adapters:** Unsloth's PEFT output loads directly, with no GGUF
  conversion step; the morning swap is load-new, then unload-old, in about a
  second.
- **Model quality:** with no memory, replies invent project facts (e.g. "the
  floor controller decides actuator commands"). Expected: facts must come from
  memory (design §5.2), not the weights.
- **Risk:** the 4-bit model is a community upload. It is pinned by revision;
  its quality against the official BF16 model hasn't been measured.
