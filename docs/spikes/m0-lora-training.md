# Spike: training Jason's nightly LoRA on the RTX 4090 Laptop (M0)

_Run 2026-10-29 on the Windows laptop (31.6 GB RAM, WSL2 capped at 24 GB). Kit and
commands: `spikes/m0-lora-training/`. Prompted by the M0 review: the serving spike
settled serving but never tested training, where priority #1 (learning) lives._

## Question

Can the live model learn in its weights on this laptop, and does what it learns
survive serving? Three designs were on the table:

| | Live model | Where it trains |
|---|---|---|
| A | Qwen3.5-9B, 4-bit AWQ | A rented GPU at night (bf16 LoRA needs about 22 GB) |
| B | Qwen3.5-4B, bf16 | The 4090 at night, bf16 LoRA (about 10 GB) |
| C | Qwen3.5-9B, 4-bit AWQ | The 4090, QLoRA on a 4-bit base |

Unsloth's own guide advises against QLoRA for every Qwen3.5 model ("higher than
normal quantization differences"), and a 9B bf16 LoRA can't fit 16 GB: the 9B's
weights alone are about 18 GB in bf16. A was out (no rented GPU available).

## Recommendation

**Option B, with the adapter merged into the weights every night.** The 4B trains
on the 4090 in about 2 minutes for this test, the merged model serves at the base
model's full speed, and every finished reply kept the learned habit. Serving
*through* a LoRA adapter is not viable: it made generation 30–60× slower.

The night cycle that works (Jason is offline during sleep, design §7):
1. Stop the live server.
2. Train a bf16 LoRA with Unsloth on the whole GPU.
3. Merge it into full bf16 weights (`merged_16bit`).
4. Start the server on the merged weights at wake (or yesterday's, if the eval
   gate fails).

## Setup

Unsloth `unsloth/unsloth:core-nightly-2026.10.28`; LoRA r=16, alpha=16, the seven
attention/MLP projections, Unsloth gradient checkpointing, batch 1 × accumulation
4, 60 steps, 1,024-token sequences. vLLM v0.32.0 for serving. The test habit:
every reply ends with a "Curious: …?" line; 50 eval questions on 10 topics never
seen in training; replies may run to 600 tokens.

## Results

**Training**

| | 4B bf16 (B), vLLM asleep | 4B bf16 (B), vLLM stopped | 9B QLoRA (C), vLLM stopped |
|---|---|---|---|
| Peak GPU memory, whole card | 11.2 GB | 11.2 GB | 10.9 GB |
| Speed | 2.1 s/step | 2.0 s/step | 2.6 s/step |
| Model load | 20 s | 12 s | 536 s (converts 19 GB to 4-bit on every load) |
| Peak WSL2 memory | 16.8 GB (holds the sleeping model) | **5.4 GB** | 5.0 GB |
| Loss, first → last | 0.89 → 0.19 | 0.82 → 0.16 | 1.12 → 0.43 |
| Merge to bf16 | | 2.9 min, 9.3 GB | |

**Does the habit survive serving?** (share of *finished* replies ending with
"Curious: …?")

| | Habit |
|---|---|
| Base 4B, base 9B | 0% |
| B adapter, served through LoRA on the bf16 4B | 90% (18/20) |
| B adapter, served through LoRA on the AWQ 4B | 95% (21/22) |
| C adapter, served through LoRA on the AWQ 9B | 95% (20/21) |
| **B adapter merged into the 4B, no LoRA** | **100% (33/33)** |

The first runs counted replies cut off at a 200-token limit as failures (about
60% were cut off); after the fix, 17 of 50 still reach 600 tokens because the 4B
writes long answers (the base model hits it on 18 of 50).

**Serving speed**

| | Generation | First token, shared prefix | First token, new prompt |
|---|---|---|---|
| 9B AWQ, base (serving spike) | 81 tokens/s | 127 ms | 464 ms |
| 4B bf16, base | 60 tokens/s | 93 ms | 291 ms |
| 4B AWQ, base | 132 tokens/s | 73 ms | 260 ms |
| 4B AWQ **through a LoRA adapter** | **about 2 tokens/s per reply** | | |
| **4B merged** | **61 tokens/s** | **90 ms** | **276 ms** |

The LoRA slowdown was seen on both 4B variants; its cause (most likely vLLM
dropping its compiled fast path for LoRA on this model) wasn't investigated,
since merging avoids LoRA at serving time entirely.

**SQLite writes** (for the M2 memory store; report only)

| | Commit, WAL | Commit, rollback journal | Bulk insert | Point read |
|---|---|---|---|---|
| `C:\` bind mount | 4.9 ms | 20.6 ms | 32,000–50,000 rows/s | 0.2–0.9 ms |
| WSL2 Docker volume | 5.1 ms | 10.0 ms | 465,000–508,000 rows/s | 0.002–0.003 ms |

With WAL, single commits cost the same; the WSL2 volume is about 10–15× faster
for bulk writes and about 100× faster for reads. Where the brain's database files
live is the owner's decision for M2.

## Why not C

QLoRA on the 9B fits and trains, and its adapter learned the habit as well as
B's. But it goes against Unsloth's guidance on quality; its adapter can only be
served *through* LoRA (it can't be cleanly merged into the AWQ weights without
re-quantizing every night), which is the 30–60× slow path; and every training
run spends 9 minutes re-quantizing the 9B. Revisit the 9B if a rented GPU becomes
available (option A, Stack C).

## Limits of this spike

- The habit is a stand-in for real lessons. It shows the mechanism works; the
  quality of real nightly training (context distillation, DPO on corrections)
  is M8's question.
- The 4B is a weaker conversationalist than the 9B; that cost is accepted, and
  not measured here.
- Keeping only complete answers dropped 80 of 200 training examples: the 4B
  writes long answers, which M8's data pipeline must handle.
- Open for M8: whether each night trains a fresh adapter on yesterday's merged
  weights, or one evolving adapter on the fixed base that is re-merged nightly
  (closer to design §5.4's "one evolving adapter regularized toward the previous
  version", and makes rollback a re-merge).
