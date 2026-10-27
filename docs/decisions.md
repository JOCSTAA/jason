# Voice Agent — Decision Log

Newest at the top. One line per decision; add detail below only if needed.

Format: `YYYY-MM-DD — Decision — Why — Alternatives rejected`

---

- 2026-10-27 — Windows laptop (RTX 4090) is an always-on server on the LAN (Ethernet), left on overnight for sleep jobs — Enables the Mac/Windows split from M0 and nightly sleep — Single-machine start
- 2026-10-27 — Build our own asyncio core from day one; borrow Pipecat components (e.g. Smart Turn, transports) only where needed — Owner wants to understand every part; frameworks assume request-response turns while Jason's loop treats silence, interjection and thinking as learned actions — Pipecat or LiveKit Agents as the main framework
- 2026-10-27 — Live LLM runs locally: Qwen3.5-9B (4-bit) on the RTX 4090 Laptop behind a server that hot-swaps LoRA adapters (serving engine still open) — Keeps weight-level learning possible; no per-turn cost — Hosted live LLM; larger serving-only models
- 2026-10-27 — Cascaded STT → LLM → TTS with an always-on floor controller choosing silent / backchannel / respond / interject / ask / think(); Mac runs audio and the controller, Windows 4090 runs LLM, memory, learner and sleep — Gives RL a clean, measurable action space with text-level hooks for memory and learning — Full-duplex speech-to-speech; hosted realtime APIs
- 2026-10-27 — Memory = append-only raw log + daily short-term file + long-term store of atomic text entries with search; retention scored on reward magnitude, use, recency, explicit requests and surprise; sleep consolidates by editing entries and regenerates a bounded wake-up briefing — Bounded, auditable, correctable memory — One ever-growing summary file; off-the-shelf memory framework
- 2026-10-27 — Wonder is a text-space simulator of the user, used for curiosity (learning progress), think() lookahead with a time cost, and Dyna-style dreaming mixed ~50/50 with real replay; predict-and-log only until it beats a baseline; dreams never become memories — Owner's original idea made measurable and safe against simulator exploitation — Raw-audio world model; dreaming on synthetic data alone
- 2026-10-27 — Weights update only during sleep; by day Jason learns through short-term memory and a small RL policy; sleep trains one evolving LoRA (context distillation + DPO on corrections) with replay, an eval gate and rollback; facts stay in memory, behavior moves into weights — Fits 16 GB (serve by day, train by night), mirrors fast/slow memory consolidation — Context-only learning; per-turn online weight updates
- 2026-10-27 — User's spoken reactions (approval/rejection) are an extrinsic reward source, interpreted by meaning in context and attributed to a specific agent action — Balances curiosity; mirrors how humans learn — Keyword-list matching ("yes"/"no"/"mhmm"), curiosity-only reward
- 2026-10-27 — Curiosity (Wonder prediction error / learning progress) is an intrinsic reward source; Wonder used for curiosity and planning — Original core idea; makes the agent inherently curious — Raw-surprise reward (noisy-TV problem)
- 2026-10-27 — Audio-only input for now; vision deferred but inputs designed as pluggable streams — Keep scope on hearing while leaving room for sight later — Audio+vision from the start; audio-only with no extension path
- 2026-10-27 — Drop the paddle test environment and its results from the old repo — It only checked my understanding of RL — Porting it as a test bed
- 2026-10-27 — Use the old repo's `transformers` branch as the reference for original ideas (Wonder, ST/LT memory, sleep) — Most recent work and contains Wonder; no code carried over — Using `main` as reference
- 2026-10-27 — Start fresh in a new repo; keep old repo archived for reference — Integrate current tools cleanly — Refactoring old code
- 2026-10-27 — Use a Claude Project for design, Claude Code (VS Code) for coding — Separate thinking from building — Single chat for everything
