# Jason — Stack Survey (plain-language version)

_Last updated: 2026-10-27. Research snapshot as of October 2026; this field moves fast, so re-check anything older than ~3 months before relying on it._

Companion to `design.md` (the design) and `decisions.md` (what's been decided). This file explains the **technology options** for each part of Jason: what each part does, the recommended choice, and the alternatives with their pros and cons. **Nothing here is decided until it appears in `decisions.md`.**

Maturity labels used below:
- **Production** — widely used, stable; you can install it and rely on it.
- **Maturing** — usable, actively developed; expect some breaking changes.
- **Research-grade** — a paper and maybe a repo; you'd be adapting an idea, not installing a product.

---

## 0. Glossary (terms used in this file)

| Term | Plain meaning |
|---|---|
| **LoRA / QLoRA** | A way to fine-tune a big model by training a small "adapter" (~0.5% of the weights) instead of the whole model. QLoRA does the same on a 4-bit-compressed model so it fits in less GPU memory. The adapter is a small file you can swap in and out. |
| **Adapter hot-swap** | Loading a new LoRA adapter into a running model server without restarting it. |
| **DPO / KTO** | Ways to train a model from preferences ("this reply was better than that one") without a separate reward model. Good for learning from corrections. |
| **GRPO / PPO** | RL algorithms that update the model by sampling several outputs and rewarding the good ones. Powerful but memory-hungry and unstable with noisy rewards. |
| **Contextual bandit** | The simplest RL setup: see a situation, pick one action, get a reward, update. No long chains of consequences. Ideal for "should I ask a question right now?" |
| **Context distillation** | Train the model *without* some instructions in its prompt to behave the way it did *with* them. This moves a lesson from the prompt into the weights. |
| **GEPA** | A 2025 method (DSPy library) that improves prompts by having a strong LLM read logs of what went wrong and rewrite the prompt. |
| **VAD** | Voice activity detection: "is someone speaking right now?" |
| **Turn detection** | "Has the person finished their thought, or just paused?" |
| **STT / TTS** | Speech-to-text / text-to-speech. |
| **Full-duplex / speech-to-speech (S2S)** | One model that listens and talks at the same time, audio in and audio out, with no separate STT/LLM/TTS steps. |
| **Cascaded pipeline** | The classic chain: STT → LLM → TTS. |
| **AEC** | Acoustic echo cancellation: stops the mic from picking up the agent's own voice. |
| **Time to first token / first audio** | How long until the model starts producing output. This drives how fast a reply *feels*. |
| **LongMemEval** | A standard benchmark for long-term memory in chat assistants. |
| **Dyna** | A classic RL idea (Sutton, 1991): learn a model of the world, then also train the policy on imagined experience from that model. |

---

## 1. Real-time learning — *how Jason actually gets better*

**What this part does:** turns interactions plus rewards into changed behavior. The key question is always: **what gets updated, from what signal, how often, and how do we know it worked?**

### Recommended: learn at three speeds around a mostly frozen LLM

| Speed | What updates | Signal | When |
|---|---|---|---|
| Fast (seconds) | A small RL policy (contextual bandit) choosing high-level actions; memory entries | Your reactions + curiosity | Live, every reaction |
| Medium (nightly) | Prompts / strategy instructions (GEPA-style) and the memory store | Logged episodes + rewards | During sleep |
| Slow (nightly, gated) | A LoRA adapter on the LLM (context distillation + DPO on corrections) | Corrections, reaction-scored replies | During sleep, only when enough data and only if it passes evaluation |

**Why:** the reward signal (your reactions) is sparse and noisy. Small, frequent updates on small decisions learn fast and are easy to measure. Big weight updates happen slowly, behind a safety check. This also matches the hardware: one 16 GB GPU can't serve the model and train it at the same time, so training goes to night.

### Alternatives

| Approach | What updates | Pros | Cons | Maturity |
|---|---|---|---|---|
| **Memory-only learning** (store lessons/heuristics, retrieve them) | Memory contents | Immediate effect; any hardware; easy to inspect and undo | Isn't RL by itself; lessons pile up; wrong lessons spread | Maturing (e.g. ERL, ICLR 2026 workshop) |
| **Prompt optimization only** (GEPA via DSPy) | Prompt text | Cheap; readable; strong 2025–26 results | Batch only (not live); prompts bloat; needs a trustworthy metric | Maturing |
| **Nightly LoRA only** (no live policy) | LLM adapter | Real weight learning | Nothing improves during the day; forgetting and drift risk | Tools production; the method for personal conversations is research-grade |
| **Online RL on LLM weights every turn** (GRPO/PPO live) | LLM adapter, continuously | Closest to "pure" real-time RL | Doesn't fit 16 GB alongside serving; unstable with noisy human reward; hard to debug (the old repo's failure mode) | Research-grade for live use |
| **Hosted LLM, learn only around it** | Policy, memory, prompts | Best conversation quality | No weight learning ever | Production |

**What would change the recommendation:** going hosted for the live LLM (then the slow layer disappears), or renting a big GPU for nightly training (then GRPO-style RL becomes feasible).

---

## 2. Memory — *what Jason knows and remembers*

**What this part does:** stores what Jason learns, finds the right piece when needed, and corrects or forgets it.

### Recommended: build a thin memory layer of our own

Plain text files (atomic entries) + a small search index + a raw append-only log. Borrow two proven ideas: Letta-style "core blocks" always in the prompt, and Graphiti-style end-dating of facts that change. The full design is in `design.md` §8.

**Why:** "what to remember" is one of the things Jason should *learn* (an RL decision). Off-the-shelf memory systems hard-code that decision. Owning it also keeps memory readable, diffable in git, and easy to debug.

### Alternatives

| System | What it is | Pros | Cons | Maturity |
|---|---|---|---|---|
| **Mem0** (open source or hosted) | Extracts facts per turn into a vector store; adds/updates/deletes automatically | Working memory in an afternoon; biggest community | Its own write policy; weaker on facts changing over time | Production-ish |
| **Letta** (formerly MemGPT) | Full agent runtime: core memory blocks + archival memory + background "sleep-time agent" that reorganizes memory | Closest to our short-term/long-term + sleep idea | You must live inside its agent loop; sleep-time agents marked experimental | Maturing |
| **Zep / Graphiti** | Temporal knowledge graph: facts get valid-from / valid-to dates | Best at "what was true when" | Heavier: needs a graph database; Zep's self-hosted edition is deprecated | Maturing |
| **Custom (recommended)** | Our own files + index | Full control; memory can be an RL action; auditable | We build and maintain it | — |

**Benchmark warning:** vendor scores on LongMemEval range from ~49% to ~97% depending on setup, and independent reruns (Redis, 2026) found published numbers often don't reproduce. Test memory on *your own* conversations instead.

**Fallback if you want speed:** use Mem0 behind an interface we control, then swap it out later.

---

## 3. Proactivity and turn-taking — *when Jason speaks*

**What this part does:** decides moment to moment whether to stay silent, say "mm-hm", answer, interject, ask, or think.

### Recommended: cascaded pipeline + our own "floor controller"

STT → LLM → TTS, plus a small always-running controller that every ~100–200 ms picks an action: **silent / backchannel / respond / interject / ask / think**. That controller *is* the RL policy, so proactivity becomes the first thing Jason learns.

**Why:** it gives RL a clean, measurable action space, and it keeps memory and learning hooks at the text level.

### Alternatives

| Approach | Pros | Cons | Maturity |
|---|---|---|---|
| **Full-duplex open model** (NVIDIA PersonaPlex-7B, Jan 2026, built on Kyutai's Moshi) | Very natural timing; real overlap and interruptions | Weak built-in language model; no text-level hook for memory; training needs audio data; tight fit on 16 GB; English only | Research-grade |
| **Controllable full-duplex** (research, 2026) | Can be told when to backchannel or start talking | Research code only | Research-grade |
| **Hosted speech-to-speech** (OpenAI Realtime, Gemini Live) | Best timing out of the box | Mostly reactive; no weight learning; per-minute cost | Production |
| **Cascaded + Smart Turn only** (no floor controller) | Simple, production-ready | Only decides *when to reply*, never when to interject | Production |
| **Cascaded + floor controller (recommended)** | Proactivity is learnable; research (Microsoft RESPOND 2026; a 270M on-device controller at NeurIPS 2026) shows it's feasible | We build the controller | Research-grade idea, small to build |

---

## 4. LLM — *Jason's language brain*

### Recommended: local Qwen3.5-9B (4-bit) on the RTX 4090 laptop + a hosted frontier model used only at night

- **Qwen3.5-9B** (released March 2026, Apache 2.0): the strongest model under 10B parameters on Artificial Analysis' index, and LoRA-trainable on 16 GB with Unsloth. Weakness: it hallucinates facts, so facts must come from memory.
- **Hosted frontier model** (Claude / GPT / Gemini): used offline during sleep as a judge (labeling reactions), reflector (GEPA), and teacher (writing corrected replies). Never live.

### Alternatives

| Model | Serve on 16 GB? | Train (LoRA) on 16 GB? | Pros | Cons |
|---|---|---|---|---|
| **Qwen3.5-4B** | Easily | Easily (even GRPO) | Fast iteration; also runs on the Mac | Less capable conversation |
| **Gemma 4 12B / E4B** (April 2026, Apache 2.0) | Yes | 12B tight; E4B easy | Native audio input on small sizes; good tooling | 12B training is tight |
| **Gemma 4 26B-A4B** (mixture of experts) | Barely | No | Fast to serve | Can't train it here |
| **~27–35B Qwen 3.6/3.8 class** | Barely, at 3–4 bit | No | Noticeably better conversationalist | Crowds out speech models; serving only. Verify versions on Hugging Face |
| **gpt-oss-20b** | Yes | No | Good tool use | Can't train it here |
| **Hosted live LLM** | n/a | No | Best quality | No weight learning; ongoing cost |

**Mac (M5 Air, 16 GB):** fine for 4B-class experiments via MLX (Apple's ML library, which supports LoRA). Not suitable for the nightly training job: memory is shared with macOS and the fanless laptop throttles.

---

## 5. Speech components

### Recommended: all local on the Mac to start

| Job | Recommended | Why |
|---|---|---|
| VAD | **Silero VAD** | Production standard; tiny; CPU |
| Turn detection | **Smart Turn v3.x** (Pipecat, open source) | 8M params; ~12 ms on CPU; trained on real conversations |
| Echo cancellation | **WebRTC AEC or macOS voice-processing I/O** | Stops Jason hearing himself (an old-repo bug) |
| STT | **NVIDIA Parakeet TDT 0.6B** | Fast streaming; very accurate English; runs on the Mac |
| TTS | **Kokoro-82M** | Very fast; free; decent voice. Upgrade later |

### Alternatives

| Job | Option | Pros | Cons |
|---|---|---|---|
| STT | Kyutai STT 1B/2.6B | Very stable streaming | ~500 ms built-in delay |
| STT | Whisper large-v3-turbo / Qwen3-ASR | Multilingual | Not natively streaming |
| STT | Deepgram / AssemblyAI (hosted) | Zero setup | Costs money; unnecessary for English |
| TTS | Chatterbox / Chatterbox-Turbo (MIT) | Expressive; voice cloning (give Jason a distinct voice) | Needs a GPU |
| TTS | Orpheus 3B | Most emotional range | ~6 GB VRAM, competes with the LLM |
| TTS | Cartesia Sonic (hosted) | ~40–150 ms first audio; $5/mo starter tier | Ongoing cost |
| TTS | ElevenLabs (hosted) | Best quality | ~$0.04–0.08 per 1K characters (Oct 2026) |

**Latency budget (target under 1 s to first audio):** turn decision 200–300 ms + STT 50–100 ms + LLM first token 100–200 ms + TTS first chunk 50–200 ms + LAN ~5 ms ≈ **0.5–0.8 s**.

---

## 6. Agent framework

### Recommended: our own Python (asyncio) core; borrow Pipecat pieces as libraries

**Why:** every framework assumes "user speaks → agent replies". Jason's defining feature is an always-on loop where *silence, interjection and thinking are learned actions*. That loop should be ours.

| Option | Pros | Cons |
|---|---|---|
| **Pipecat** (Python, BSD-2) | Fully local; good audio plumbing; Smart Turn built in | Turn-centric pipeline would fight the proactive loop |
| **LiveKit Agents** | Production WebRTC; telephony | Needs a media server; built for multi-party calls |
| **OpenAI Agents SDK / LangGraph** | Tools, handoffs, checkpointing | Turn-based by design |
| **Letta** | Memory built in | Its own agent loop |
| **Own core (recommended)** | Exactly our control flow; logging and replay built in | We write it |

**Reasonable hybrid:** if interruption handling and audio transport become a time sink at M1, use Pipecat for the M1 audio pipeline and wrap it. Keep the speak/silence decision in our own policy.

**Learning libraries:** Vowpal Wabbit or a small PyTorch bandit (first policy); CleanRL / Stable-Baselines3 (later actor-critic); TRL + Unsloth (LoRA, DPO, GRPO); vLLM or llama.cpp server (serving with LoRA hot-swap).

---

## 7. Putting it together — candidate stacks

| | **Stack A — Local-first learner (recommended)** | **Stack B — Hosted brain** | **Stack C — A + rented trainer** |
|---|---|---|---|
| Mac | Audio I/O, AEC, VAD, Smart Turn, STT, TTS, floor controller | Same | Same |
| Windows 4090 | Qwen3.5-9B + LoRA hot-swap, memory, learner, Wonder, sleep jobs | Memory, learner, evals (no LLM) | Larger ~27B-class model (tight fit) |
| Hosted | Frontier API for night jobs only | LLM for every live reply (+ optional hosted TTS) | Same as A + rented A100 for nightly training |
| What learns | Policy, memory, prompts, **weights** | Policy, memory, prompts (no weights) | Everything, on a bigger model |
| Rough $/month | **~$20–70** | ~$80–250 | ~$60–150 |
| Main risk | 9B conversation may feel flat | Learning ceiling; vendor dependence | Two environments to keep in sync |

**Upgrade path:** A → C just adds a training target. A → B swaps the LLM client and keeps everything else. Starting with A keeps both doors open; starting with B makes weight learning a re-platform later.

Cost assumptions: ~50 hours/month of conversation; hosted prices from vendor pages July–October 2026; A100 80 GB on RunPod ~$1.19–1.39/hr.

---

## 8. Sources (selected)

Learning: MetaClaw (arXiv 2603.17187) · SLIFT (2608.09109) · PESO continual LoRA (2510.25093) · RLHI (2509.25137) · GEPA (2507.19457; dspy.ai/api/optimizers/GEPA) · ERL (2603.24639) · Memento 2 (2512.22716) · Learning Progress Monitoring (2509.25438) · Curiosity reward for dialogue (2504.03206) · "Do Language Models Need Sleep?" (2605.26099) · DynaWeb (2601.22149) · Dyna-Think (2506.00320) · Internalizing the Future (2606.27483) · UDP user world model (2504.13643)

Memory: docs.letta.com (sleep-time agents) · github.com/mem0ai/mem0 · Graphiti / Zep · Redis agent memory benchmark report (2026)

Turn-taking: huggingface.co/nvidia/personaplex-7b-v1 · controllable full-duplex (2601.11329) · RESPOND (2603.21682) · semantic turn-taking controller (NeurIPS 2026) · Voice Activity Projection (2205.09812) · github.com/pipecat-ai/smart-turn

Models: Artificial Analysis — Qwen3.5 small models · ai.google.dev/gemma/docs/core · MLX fine-tuning guides

Speech & cost: HF speech-to-speech pipeline (pypi: speech-to-speech) · vendor pricing pages (Cartesia, ElevenLabs, Deepgram) · runpod.io/gpu-cloud/pricing

Full survey with links: the "Jason — Stack Survey (Oct 2026)" doc.
