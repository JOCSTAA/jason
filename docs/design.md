# Jason — Design Doc

_Last updated: 2026-10-27 (v2.1: core design confirmed; local LLM, own core and machine split decided)_

Companion docs:
- `decisions.md` — decision log (source of truth for what is decided)
- `stack-survey.md` — plain-language technology options: recommended choice + alternatives with pros/cons
- `old-repos-review.md` — detailed review of the archived repos

**Status legend used in this doc:**
- **[Decided]** — recorded in `decisions.md`.
- **[Proposed]** — agreed direction in design conversations, pending a `decisions.md` entry.
- **[Open]** — not yet chosen.

---

## 1. Vision

A voice agent that learns and grows in real time. It takes in information as it arrives, remembers it, improves its behavior through reinforcement learning, and takes part in conversation proactively: speaking out of turn and asking questions when it judges that useful.

It should be inherently curious (rewarded for learning new things, averse to boredom) and shaped by how the user reacts to what it does, similar to how humans learn. Like a person, it has a **day** (live interaction, fast learning) and a **night** (sleep: consolidating memory, imagining scenarios, and slow learning in its weights).

This is a personal passion project. The measure of success is how capable, adaptive and alive the agent becomes.

## 2. Core capabilities (priority order)

1. **Real-time learning (RL):** behavior improves from ongoing interaction.
2. **Memory:** absorbs information in real time, stores it durably, retrieves and uses it.
3. **Proactivity:** interjects, speaks out of turn, asks clarifying or curious questions.
4. **Natural voice conversation:** fast enough to feel real (a requirement, not the main goal).

**Senses [Decided]:** audio only for now. Vision (camera) is wanted later, so inputs are designed as pluggable streams a camera can join without a redesign.

---

## 3. System overview

### 3.1 The daily cycle [Decided]

```
WAKE  ─► load wake-up briefing (summary of long-term memory) + latest policy + latest LoRA adapter
  │
DAY   ─► live loop: hear → decide (silent / backchannel / respond / interject / ask / think) → speak
  │      fast learning: policy updates per reaction; short-term memory updated per interaction;
  │      everything appended to the raw log; Wonder predicts and logs (and is used by think())
  │
SLEEP ─► 1 consolidate memory  2 train Wonder  3 test Wonder  4 dream (if Wonder passed)
         5 train policy / prompts / LoRA  6 eval gate → swap or roll back  7 build tomorrow's briefing
```

### 3.2 Architecture and machine split [Decided]

```
 MAC (M5 Air) — ears, mouth, dev                 WINDOWS (RTX 4090 Laptop, 16 GB) — brain, gym
 ┌──────────────────────────────────┐            ┌─────────────────────────────────────────┐
 │ Mic / speaker + echo cancellation│            │ LLM server (local, LoRA hot-swap)       │
 │ VAD + turn detection             │  LAN       │ Wonder (user/world simulator)           │
 │ Streaming STT, TTS               │◄─events───►│ Memory store (raw log, ST, LT, index)   │
 │ Floor controller (RL policy)     │            │ Learner (policy updates)                │
 └──────────────────────────────────┘            │ Sleep jobs (consolidate, dream, train)  │
                                                 └───────────────┬─────────────────────────┘
                                                                 │ nightly only
                                                       Hosted frontier API (judge / reflect / rewrite)
```

- Audio never leaves the Mac; only typed events cross the LAN (`user_partial`, `turn_end_prob`, `agent_action`, `reaction`, `reward`, `wonder_prediction`, …).
- Every event is also appended to the raw log, so anything can be replayed and re-scored later.
- The agent core must be able to start speaking on its own, not only after user input.
- The camera joins later as another input stream on the Mac.

---

## 4. Action space [Decided]

The floor controller chooses one high-level action every ~100–200 ms. **Silence is an explicit action.**

| Action | Meaning |
|---|---|
| `silent` | Keep listening. |
| `backchannel` | Short acknowledgment ("mm-hm", "right") without taking the turn. |
| `respond` | Take the turn and reply. |
| `interject` | Speak before the user has finished, to contribute something relevant. |
| `ask` | Ask a question (clarifying or curious). |
| `think()` | Pause and run Wonder lookahead over candidate actions, then choose (see §6.2). Has a time cost. |

Lower-level choices (what exactly to say, which response strategy, what to store in memory) are made by the LLM under prompts/strategies, and by the memory policy. These are also learnable (see §5).

---

## 5. Learning system

### 5.1 Principle: learn at three speeds [Decided]

Weights update **only during sleep**. By day, Jason learns through his context/memory and a small, fast RL policy.

| Speed | What updates | Reward / signal | When | How we know it improved |
|---|---|---|---|---|
| **Fast** | Floor-controller policy (contextual bandit first; small actor-critic later) | User reactions + implicit signals + curiosity bonus | Live, after each reaction (seconds) | Reward per decision vs. a frozen-policy baseline; regret curves on a simulated user |
| **Fast** | Short-term memory (daily context file) | Interaction content + its reward | Live, per interaction | Recall tests on the same day |
| **Medium** | Long-term memory, wake-up briefing, prompt/strategy instructions (GEPA-style) | Logged episodes + rewards | Sleep | Replay eval set scored before/after |
| **Slow** | One evolving LoRA adapter on the LLM | Context distillation + DPO on corrections; reaction-scored replies | Sleep, **only** when ≥ ~200 new labeled examples exist **and** the eval gate passes | Next-day reaction rate (old vs. new adapter); graduation test; forgetting set |

**Rejected for now:** per-turn RL on LLM weights (doesn't fit 16 GB alongside serving; unstable with noisy human reward). Revisit only with rented training compute (Stack C in `stack-survey.md`).

### 5.2 Facts vs. behavior [Decided]

- **Facts live in memory** (names, dates, projects, what the user said). LoRA is unreliable for storing facts, causes hallucination when partly learned, and can't be cleanly corrected or forgotten.
- **Behavior lives in weights** (phrasing, when to ask vs. answer, what kinds of replies the user likes, conversational style).
- **Graduation:** a behavioral lesson starts in memory. Once sleep training has moved it into the weights (verified by removing it from context and checking the behavior persists), it's removed from the context. This keeps the context bounded while the agent keeps growing.

### 5.3 Reward sources

Two families that balance each other. Curiosity alone produces an agent that interrupts and asks constantly; user feedback alone produces a sycophant.

**A. Intrinsic: curiosity (via Wonder) [Decided]**
- Reward for acquiring new information so the agent is curious and avoids boredom. Owner's original phrasing: "give rewards for any new information learned or anything seen" ("seen" applies once vision is added).
- Signal: **learning progress**, meaning how much Wonder's predictions about the user improve on a topic over time. Not raw surprise, because unpredictable noise (static, background chatter) is permanently surprising and teaches nothing (the "noisy-TV" problem). Supporting research: Learning Progress Monitoring (ICLR 2026).
- Wonder's uncertainty (disagreement among its sampled futures) is the trigger for candidate questions (`ask`).
- Weighting: total reward = extrinsic + β × intrinsic, with β kept small and capped so curiosity can't dominate.
- **[Open]** How to define "new information" in a way that can't be gamed by noise. Learning progress is the starting answer.

**B. Extrinsic: user reactions [Decided]**
- The user's spoken reactions reward or penalize what the agent just did: approval ("yes, that's a great idea", "exactly") → positive; rejection or correction ("nope", "no, that's wrong") → negative.
- Interpreted **by meaning in context, not keyword lists.** Pitfalls: "No" is often an answer, not a judgment; backchannels ("mhmm") usually mean "I'm listening"; mixed or sarcastic phrasing.
- Each reaction is attributed to a specific agent action (credit assignment). Reactions arrive late and may refer to something said several seconds or turns earlier.
- Pipeline: classify every user utterance (LLM judge) for whether it's feedback, which action it targets, polarity and strength. Log the raw transcript and the classification so rewards can be **re-scored during sleep** by a stronger hosted model.
- Implicit signals: barge-in (interrupting Jason = mild negative for the current action), corrections, repeated questions, topic abandonment, follow-up engagement.
- Explicit channel: "good job, Jason" / "don't do that" as a high-confidence signal.
- Content matters, not just polarity: a correction says *what* was wrong. It feeds memory and DPO pairs (Jason's reply vs. the corrected version), not just a scalar.
- Prior art: TAMER / COACH (learning from human reward with delay), RLHI (2025, learning from real user conversations; research-grade).

### 5.4 Safeguards

| Risk | Safeguard |
|---|---|
| **Sycophancy / reward hacking** (fishing for "yes", leading questions, agreeing with everything) | Reward only feedback directed at agent actions; cap rewards for self-solicited approval; track agreement rate; hold-out evaluation sessions |
| **Curiosity runaway** (constant questions) | Capped β; `ask` has a cost; reactions to over-asking penalize it |
| **Catastrophic forgetting** | Replay of earlier days + general data in every LoRA batch; one evolving adapter regularized toward the previous version; forgetting eval set |
| **Drift / bad night** | Eval gate before swap; adapter, policy and memory versioned together; one-command rollback |
| **Overfitting to one day / one mood** | Minimum data threshold before LoRA training; replay; small learning rates |
| **Wonder exploitation** (policy learns to exploit simulator errors) | Dreams mixed ~50/50 with real replay; dreaming skipped when Wonder fails its test; evaluation on real data only |
| **False memories** | Dreamed content tagged and never written to long-term memory as facts |
| **Runaway behavior** | Hard rate limits on interjections and questions; manual "be quiet" override that is also logged as a negative reward |

### 5.5 Learning architecture principles (from old-repo lessons) [Decided as principles]
- Act with a frozen policy copy while a learner trains in the background, then swap, with versioning and rollback.
- Learning never blocks conversation.
- Spend the sparse RL signal on small, high-level decisions; use pretrained models for hearing, language and speech.
- Use tested RL implementations, not hand ports. Validate every learning loop on a small known-reward setup (a simulated user) before going live.

---

## 6. Wonder — the world model [Decided]

The owner's original idea, kept and made concrete. Wonder serves **curiosity, planning (`think()`), and dreaming**.

### 6.1 What Wonder is
- A **text-space simulator of the user and the conversation**. Given the conversation so far, the long-term memory about the user, and a candidate agent action, it predicts the user's next utterance and the user's reaction.
- It is semantic (text), not raw audio. Predicting raw audio or pixels was one reason the old repo failed.
- Implementation [Open]: start as the same local LLM with a "predict the user" prompt; later a dedicated LoRA adapter or a smaller 2–4B model trained on the raw log (pairs of context + action → actual next user turn and reaction).
- A reward model (the reaction classifier from §5.3 B) scores Wonder's predicted reactions.

### 6.2 `think()` — planning with Wonder
- For each of ~3–4 candidate actions, Wonder samples 2–3 short futures (1–2 turns). Each candidate gets an expected predicted reward, and the best is chosen.
- Target latency ~100–300 ms (batched on the 4090), so it fits in a pause or runs while the user is still speaking.
- `think()` carries a **time cost**, so the policy learns *when* lookahead pays off instead of thinking constantly or using it to stall.
- Research context: "foresight-conditioned" LLM agents (Internalizing the Future, 2026) and user world models in dialogue planning (UDP, 2025) are research-grade but support the approach.

### 6.3 Dreaming (imagination during sleep or idle)
- Dyna-style: the policy trains on short imagined scenarios produced by Wonder, seeded from real situations from the day.
- Lessons from DynaWeb (Jan 2026): purely imagined training degrades performance through simulation bias; mixing ~50% real data stabilizes it; shorter rollouts are safer.
- Rules:
  1. Dream only on nights Wonder passed its test on held-out real data.
  2. Short rollouts (1–3 turns).
  3. Every training batch is ~50/50 dreams and real replay.
  4. Evaluate only on real data, never on dreams.
  5. Dreams are tagged and never become long-term facts.
- Idle-time dreaming (not only at night) is allowed when the GPU is free [Open: idle trigger rules].

### 6.4 Bootstrapping and measurement
- Wonder starts weak (no data). From M3 it runs in **predict-and-log only** mode: every turn, log predicted vs. actual user response and reaction.
- Metrics: prediction accuracy of reaction polarity; semantic similarity of predicted vs. actual utterance; calibration (predicted vs. actual reward).
- Enable `think()` only when Wonder's reaction prediction beats a simple baseline (e.g. "predict the most common reaction") by a clear margin. Enable dreaming after that.
- Watch for **sycophancy amplification:** if Wonder learns "agreement pleases the user", planning would push even harder toward agreement. Calibration tracking catches this.

---

## 7. Sleep [Decided]

### 7.1 When
- **[Open]** Schedule: a designated time (owner's original idea, e.g. 1–6 am), or triggered by long idle, or both. The Windows laptop must be on and connected.
- Jason is offline during sleep (by design). Idle-time micro-sleep (memory consolidation only) can run during long pauses.

### 7.2 Pipeline (in order)

| # | Stage | What happens | Skip if |
|---|---|---|---|
| 1 | **Re-score** | Hosted judge re-labels the day's reactions and credit assignment | API unavailable (keep live labels) |
| 2 | **Consolidate memory** | Short-term entries merged into long-term: dedupe, supersede corrected facts, update retention scores, prune low scorers | — |
| 3 | **Train Wonder** | Fine-tune on today's real (context, action → next turn, reaction) pairs | Too little data |
| 4 | **Test Wonder** | Accuracy on held-out real turns vs. previous version | — |
| 5 | **Dream** | Generate imagined rollouts seeded from real situations | Wonder failed step 4 |
| 6 | **Train policy** | Bandit/actor-critic on real + dreamed experience | — |
| 7 | **Optimize prompts** | GEPA-style rewrite of strategy instructions using logs + rewards | Not enough episodes |
| 8 | **Train LoRA** | Context distillation of behavioral lessons + DPO on corrections, with replay of earlier days and general data | < ~200 new labeled examples |
| 9 | **Eval gate** | New policy/adapter/prompts must beat current ones on the real held-out replay set and must not regress on the forgetting set | — |
| 10 | **Swap or roll back** | Promote passing components; version everything together | — |
| 11 | **Build wake-up briefing** | Regenerate the briefing from long-term memory | — |
| 12 | **Commit** | Memory files and a sleep report committed to git (diffable history of what Jason believed) | — |

Rough time budget at 4–9B scale on the 4090: a few hours. [Open] Measure once M6 runs.

### 7.3 Sleep report
Every sleep writes a short report: what was consolidated or forgotten, Wonder accuracy, dream count, training losses, eval results, and what was swapped or rolled back. This is the main dashboard for "is Jason actually learning?"

---

## 8. Memory system [Decided]

Memory is separate from model weights. Facts in memory, behavior in weights (§5.2).

### 8.1 Layers

| Layer | What it is | Written | Read |
|---|---|---|---|
| **Raw log** | Append-only record of every event (utterances, actions, reactions, rewards, latencies, Wonder predictions). **Training data, not agent memory.** Never edited. | Continuously | Sleep jobs, replay, debugging |
| **Short-term memory (STM)** | Today's working context file: wake-up briefing + facts and lessons appended during the day, each with its reward | Live, per interaction | Every LLM call (in the prompt) |
| **Long-term memory (LTM)** | Large but finite store of **atomic text entries** (one fact or lesson per entry) + a search index | Consolidated during sleep | Retrieval during the day; source of the briefing; conditions Wonder |
| **Wake-up briefing** | Bounded summary of the top-scored LTM entries in fixed sections with token budgets (who the user is, current projects, standing preferences, behavioral lessons) | Regenerated each sleep | Start of STM each morning |
| **Core blocks** | A few hundred tokens always in the prompt (identity of the user, Jason's own persona) | Sleep only | Every LLM call |

Format: Markdown or JSONL text files (human-readable and git-diffable). Each LTM entry has: `id`, `text`, `type` (fact / preference / lesson / episode), `created`, `valid_from` / `valid_to`, `source` (utterance id), `reward`, `retention_score`, `times_retrieved`, `origin` (real / dreamed).

### 8.2 Retention: what is stored and forgotten

Retention score combines:
- magnitude (not sign) of the reward attached; a strong negative reaction stores a *lesson*, it doesn't trigger forgetting
- how often the entry has been retrieved and used
- recency
- explicit user instruction ("remember this" / "forget that")
- Wonder's surprise when the information arrived (new information is valuable)

Neutral but important facts (no reaction) are protected by the use and explicit-instruction terms.

### 8.3 Consolidation rules
- Consolidate by **editing entries, not re-summarizing prose.** Repeated summarization drifts like a game of telephone.
- Corrections end-date the old entry (`valid_to`) instead of deleting it, so history is kept.
- Pruning removes low-score entries from LTM, never from the raw log, so a mistaken forget can be undone. An explicit user "forget this" is honored fully.
- Dreamed content (`origin: dreamed`) never becomes a fact.

### 8.4 Retrieval
- During conversation: the briefing is always in context; details come from search over LTM (embedding + keyword) when the current topic needs them.
- **[Open]** Whether "what to store / retrieve" becomes an RL action in a later milestone.

---

## 9. Proactivity and turn-taking [Decided]

- **Cascaded pipeline** (STT → LLM → TTS) plus an always-running **floor controller** choosing the §4 actions every ~100–200 ms.
- Floor-controller inputs: VAD state, pause length, Smart Turn end-of-turn probability, streaming partial transcript, pending question queue (from curiosity), recent reactions, time since Jason last spoke.
- Start rule-based (thresholds), then replace with a learned contextual bandit. Research (Microsoft RESPOND, 2026) exposes "backchannel intensity" and "turn-claim aggressiveness" as dials, which are natural knobs for the policy to learn per context.
- **Barge-in:** if the user speaks while Jason talks, stop TTS within ~200 ms and log a mild negative for the interrupted action.
- **Self-hearing:** echo cancellation on the Mac (WebRTC AEC or macOS voice-processing I/O) plus mic gating while speaking.
- Rate limits on `interject` and `ask` as a safety net while the policy is young.
- Full-duplex speech-to-speech models (PersonaPlex/Moshi) are not used for now: no text-level hooks for memory and learning. Revisit if that changes.

## 10. Conversation requirements
- Target under ~1 s to first audio. Expected budget: turn decision 200–300 ms + STT 50–100 ms + LLM first token 100–200 ms + TTS first chunk 50–200 ms + LAN ~5 ms ≈ 0.5–0.8 s.
- Can be interrupted (barge-in).
- Per-stage latency logged from day one.

---

## 11. Stack [Partly decided — options in `stack-survey.md`]

"Stack A — local-first learner". Rows with a Choice are decided; the rest are recommendations pending a decision:

| Component | Recommended | Main alternative | Choice |
|---|---|---|---|
| Language / runtime | Python 3.12, own asyncio core | Pipecat as main framework | **Own asyncio core from day one; borrow Pipecat components only if needed** |
| Live LLM | Qwen3.5-9B, 4-bit, vLLM or llama.cpp server with LoRA hot-swap (Windows) | Gemma 4 12B; hosted LLM (Stack B) | **Local, Qwen3.5-9B on the 4090** (serving engine open) |
| Night-time helper | Hosted frontier API (judge, reflection, rewrites) | Local only | |
| RL policy | Contextual bandit (Vowpal Wabbit or small PyTorch) | Actor-critic via CleanRL/SB3 later | |
| Weight training | Unsloth + TRL (LoRA, DPO, context distillation) | — | |
| Prompt optimization | DSPy GEPA | Manual | |
| Memory | Own: text entries + search index (e.g. LanceDB / sqlite-vec) | Mem0 as temporary baseline; Letta; Graphiti | |
| Wonder | Same LLM + "predict user" prompt → dedicated adapter / 2–4B model | — | |
| VAD / turn | Silero VAD + Smart Turn v3.x | — | |
| STT | Parakeet TDT 0.6B (Mac) | Kyutai STT; hosted | |
| TTS | Kokoro-82M → later Chatterbox or Cartesia | ElevenLabs | |
| Echo cancellation | WebRTC AEC / macOS voice-processing I/O | — | |
| Hardware | Mac = audio + controller; Windows 4090 = LLM, memory, learner, sleep | Everything on Windows | **Split; Windows is an always-on server on Ethernet, on overnight** |

Budget: Stack A ≈ $20–70/month (night-time API + optional hosted TTS). Hosted-brain alternative ≈ $80–250/month.

---

## 12. Evaluation and instrumentation

**Log from day one:** every event, every action with its features, rewards (raw and re-scored), per-stage latency, Wonder predictions vs. actuals, model/adapter/policy/memory versions.

**Test beds:**
- **Simulated user** (an LLM persona with known, hidden preferences, e.g. "likes questions about engineering, dislikes interruptions"). Every learning loop must show it learns these before going live (rule 6).
- **Replay set:** a growing held-out set of real situations; candidate policies and adapters are scored on it by a judge.
- **Forgetting set:** fixed general-capability prompts, scored every sleep.

**Core metrics:**

| Metric | Measures |
|---|---|
| Reward per decision (live), vs. frozen baseline | Is the policy improving? |
| Positive/negative reaction rate per day; per action type | Is Jason pleasing or annoying? |
| Barge-in rate; question rate; agreement rate | Over-talking, over-asking, sycophancy |
| Wonder accuracy and calibration | Can Jason predict the user? |
| Learning progress (curiosity) per topic | Is Jason learning new things? |
| Memory recall on own-conversation questions (LongMemEval-style) | Does memory work? |
| Graduation test pass rate | Did lessons move into weights? |
| Forgetting-set score | Did training break general ability? |
| Latency per stage | Does it feel real-time? |

---

## 13. Milestones (revised)

Each milestone has a measurable exit criterion.

| # | Milestone | Scope | Exit criterion |
|---|---|---|---|
| **M0** | Setup | Repo layout, CLAUDE.md, pinned environment (both machines), event schema, raw logging, LAN link | Mac↔Windows event round trip logged; one entry point |
| **M1** | Conversation loop | Mic → AEC → VAD/Smart Turn → STT → local LLM → TTS → speaker, across the split; barge-in | End-to-end voice conversation, < 1 s median to first audio, per-stage latency logged |
| **M2** | Memory v1 | Raw log, STM file, LTM entries + search, wake-up briefing, manual "sleep" that consolidates | Recalls facts across sessions on a 20-question own-conversation test |
| **M3** | Feedback signals + Wonder logging | Reaction classifier, credit assignment, implicit signals, re-scoring script; Wonder predict-and-log | Classifier agreement with hand labels ≥ target on 100 utterances; Wonder accuracy tracked daily |
| **M4** | First learning loop | Contextual bandit for one decision (proposed: `ask` vs. not) with frozen/learner swap; validated on simulated user first | Learns simulated user's hidden preference; then beats frozen baseline on live reward over 1–2 weeks |
| **M5** | Proactivity | Floor controller with full action set (minus `think()`), rule-based then learned | Interjections/backchannels happen; barge-in and question rates within target |
| **M6** | Sleep v1 | Scheduled sleep: re-score, consolidate memory, policy training, GEPA prompts, eval gate, rollback, sleep report | Nightly report; at least one gated promotion that improves next-day reaction rate |
| **M7** | Wonder v1 + `think()` | Train Wonder nightly; enable `think()` when it beats baseline | `think()` decisions earn higher reward than non-think decisions in comparable situations |
| **M8** | Sleep v2: weights | Nightly LoRA via context distillation + DPO with replay and gate; graduation of lessons | Graduation tests pass; forgetting set stable; next-day reaction rate ≥ previous adapter |
| **M9** | Dreaming | Dyna-style imagined rollouts mixed with real replay | Policy trained with dreams beats policy without, on real held-out data |
| **M10+** | Expand | Vision input, memory-as-RL-action, richer curiosity, bigger model / rented training (Stack C) | — |

---

## 14. Open questions

Resolved 2026-10-27: core learning/Wonder/memory/turn-taking design; live LLM runs locally; own core from day one; Windows laptop is an always-on server.

Needed during M0–M1:
- Serving engine on Windows: vLLM (needs WSL2 on Windows) vs. llama.cpp server (native). Both must load the chosen model and hot-swap LoRA. Verify Qwen3.5's architecture is supported before committing.

Can wait:
- First learned behavior (needed by M4; proposed `ask` vs. not).
- Hosted API provider and monthly cap for night jobs (needed by M6).
- Sleep schedule and idle triggers (M6).
- Exact retention-score formula and thresholds (M2/M6).
- How to define "new information" robustly (M7).
- Wonder implementation: prompt vs. adapter vs. separate small model (M7).
- Whether memory writes become an RL action (M10+).

---

## 15. Lessons from the old repos

Full detail (repo map, where each original idea lives in the old code, owner's answers, bug list, keep/adapt/drop table): `old-repos-review.md`.

Reviewed 2026-10-27: `JOCSTAA/AI` (2022–23; the unmerged `transformers` branch is the reference for the original ideas) and `JOCSTAA/speech` (~2024, a standalone STT → LLM test that worked end to end). Both archived; nothing carried over as code. The paddle test environment was only a check of RL understanding and is dropped.

**Why `AI` failed**
- No reward: hard-coded to 1 (the intended curiosity reward was never written).
- Wrong learning target: RL over raw perception and a 16k-dim sound output.
- Broken learning internals that went undetected (critic regressed to its own outputs, wrong advantages, corrupted audio input on `main`).
- No logs, checkpoints or metrics.
- Training blocked the acting loop; the agent couldn't stay silent and could hear itself.

**Original ideas carried forward (modern form)**
- Curiosity as intrinsic reward → learning progress, balanced by user feedback (§5.3).
- Wonder for curiosity and planning → text-space user simulator; `think()`; dreaming (§6). (`think()` was "Park" in old-repos-review; now in scope.)
- Always-on duplex loop with silence as an action → floor controller (§4, §9).
- Acting copy vs. training copy swap → frozen policy/adapter + gated swap (§5.5, §7).
- Short-term + long-term memory, long-term as both context and sleep training data (§8).
- Sleep (§7).
- Multimodal input via pluggable streams (§2).

**Rules for the new design**
1. Define reward signals before choosing networks.
2. Pretrained models for perception, language and speech; learn only user-specific, low-dimensional decisions (plus behavioral LoRA in sleep).
3. Agent memory (read at inference) is separate from the training data log.
4. Instrument learning from day one: rewards, losses, value fit, before/after evals.
5. Learning is asynchronous to conversation; silence is an explicit action; handle self-hearing.
6. Use tested RL implementations, not hand ports. Validate learning loops on a small known-reward setup before going live.
7. One entry point, pinned cross-platform environment, final work on `main`.
