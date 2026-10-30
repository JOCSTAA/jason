# Jason — Design Doc

_Last updated: 2026-10-30 (v2.11: §9 rollout: no prompt-cache warming in M1)_

Companion docs:
- `decisions.md` — decision log (source of truth for what is decided)
- `stack-survey.md` — plain-language technology options: recommended choice + alternatives with pros/cons
- `old-repos-review.md` — detailed review of the archived repos
- `spikes/m0-llm-serving.md` — performance comparison and decision for the serving engine
- `spikes/m0-lora-training.md` — whether the live model can train on the 4090, and the merged nightly cycle
- `spikes/m0-link-latency.md` — Mac↔brain round trips over Tailscale (1,000 pings)
- `spikes/m1-mac-audio.md` — echo cancellation and the body's speech models on the Mac; portable (ONNX) versions compared

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
WAKE  ─► load wake-up briefing (summary of long-term memory) + latest policy + latest merged weights (base + LoRA)
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
 BODY (today: Mac M5 Air) — ears, mouth, dev     BRAIN (today: Windows RTX 4090 Laptop, 16 GB) — gym
 ┌──────────────────────────────────┐            ┌─────────────────────────────────────────┐
 │ Mic / speaker + echo cancellation│            │ LLM server (local, merged nightly)      │
 │ VAD + turn detection             │  LAN       │ Wonder (user/world simulator)           │
 │ Streaming STT, TTS               │◄─events───►│ Memory store (raw log, ST, LT, index)   │
 │ Floor controller (RL policy)     │            │ Learner (policy updates)                │
 └──────────────────────────────────┘            │ Sleep jobs (consolidate, dream, train)  │
                                                 └───────────────┬─────────────────────────┘
                                                                 │ nightly only
                                                       Hosted frontier API (judge / reflect / rewrite)
```

- Two roles, **body** and **brain**, each portable: the body (ears, mouth, floor controller) runs on the Mac today and may move to a separate machine; the brain runs on the Windows laptop today and may move to a cloud GPU (§3.4). Future sensors and motors (§5.6) attach to the body.
- Audio never leaves the body's machine; only typed events cross the LAN (`user_partial`, `turn_end_prob`, `agent_action`, `reaction`, `reward`, `wonder_prediction`, …).
- Every event is also appended to the raw log, so anything can be replayed and re-scored later.
- The agent core must be able to start speaking on its own, not only after user input.
- The camera joins later as another input stream on the body.

### 3.3 Models

Every model in the agent, where it runs, what it does, and how (or whether) it is trained. The live LLM (the 4B) fills several roles, so there are fewer models than components.

| # | Model | Runs on | Job | How it is trained / updated | Status |
|---|---|---|---|---|---|
| 1 | **Silero VAD** v6 | Body (CPU, `onnxruntime`) | Detects whether someone is speaking | Pretrained; frozen | [Decided] |
| 2 | **Smart Turn v3.2** (~8M params) | Body (CPU, `onnxruntime`) | Probability that the user has finished their turn; input to the floor controller | Pretrained; frozen | [Decided] |
| 3 | **Parakeet TDT 0.6B v3** | Body (Mac: MLX) | Speech-to-text, re-transcribing the whole utterance as it grows (its streaming mode is slower than real time) | Pretrained; frozen | [Decided]; backend swappable for a non-Mac body |
| 4 | **Kokoro-82M** | Body (Mac: MLX) | Text-to-speech | Pretrained; frozen. Later swap to Chatterbox or Cartesia | [Decided]; backend swappable for a non-Mac body |
| 5 | **Floor-controller policy** (contextual bandit; small actor-critic later) | Acts on the body; learner on the brain | Chooses silent / backchannel / respond / interject / ask / `think()` | Starts rule-based. Then learned from scratch: **updated live after each attributed reward**; retrained during sleep on real replay (plus dreams from M9); behind the frozen/learner swap and eval gate. Validated on the simulated user first | Approach [Decided]; library [Open] (Vowpal Wabbit or small PyTorch) |
| 6 | **Qwen3.5-4B (bf16) + one LoRA, merged nightly** | Windows 4090 | Main LLM: generates what Jason says. Also the default host for models 7 and 9 until they get their own | Base weights frozen. **Only the LoRA trains, during sleep**: context distillation + DPO on corrections, with replay; only with ≥ ~200 new labeled examples and a passed eval gate; merged into the weights and served from them next morning (no LoRA at serving time) | [Decided]; served by vLLM, bf16 |
| 7 | **Reaction judge** | Windows (if local) | Classifies each user utterance: is it feedback, which action it targets, polarity, strength (§5.3 B) | Live labels are re-scored nightly by model 10; those re-scored labels can later train a dedicated small classifier | [Open] (needed by M3): the live LLM with a prompt, or a small separate classifier |
| 8 | **Embedding model** | Windows | Turns memory entries and queries into vectors for long-term memory search (§8.4) | Pretrained; frozen. Changing it means re-embedding the whole store | [Open] (needed by M2) |
| 9 | **Wonder** (user simulator) | Windows 4090 | Predicts the user's next utterance and reaction; used for curiosity, `think()` and dreaming (§6) | M3–M6: the live LLM with a "predict the user" prompt, no training (predict-and-log only). From M7: **trained nightly** on (context, action → next user turn, reaction) pairs from the raw log; enabled for `think()` only once it beats a baseline | [Open] (M7): prompt on the live LLM, its own LoRA, or a separate small model |
| 10 | **Hosted frontier model** (Claude / GPT / Gemini) | Cloud, sleep only | Re-scores the day's rewards, rewrites prompts (GEPA), writes corrected replies for DPO | Not trained; used as a tool | Recommended (§11); provider [Open] (needed by M6) |

Not counted: echo cancellation (WebRTC AEC / macOS voice-processing I/O) is signal processing, not a learned model.

**What actually learns from experience:** the floor-controller policy (live, every reaction), the LoRA on the live LLM (nightly, merged into its weights), and Wonder once it trains (nightly, from M7). Everything else is frozen.

**GPU budget note:** models 6, 8, and models 7 and 9 when they run on the live LLM, all share the 4090's 16 GB by day. Live replies must take priority over judging and Wonder logging.

### 3.4 Portability: cloud brain [Decided]

The Windows laptop may later be replaced or supplemented by a rented cloud GPU for more compute. The body (the Mac today) keeps everything time-critical (audio, VAD, turn detection, STT, TTS, floor controller), so the brain can move without a redesign, as long as M0 is built for it.

**Two ways to add compute:**

| Goal | Move | Effort | Conflicts with decisions? |
|---|---|---|---|
| More training power (bigger LoRA runs, GRPO, larger Wonder) | Keep serving on Windows; rent a cloud GPU only for the nightly sleep job (Stack C in `stack-survey.md`) | Low: sleep is already a separate batch pipeline | No |
| Bigger live model (e.g. ~27B+) | Move serving, and likely the whole brain, to an always-on cloud GPU | Medium | Yes: reverses "live LLM runs locally on the 4090" and "Windows is the always-on server" |

**What changes in the cloud:** network round trip rises from ~10 ms (measured at home over Tailscale) to ~20–80 ms (still within the §10 budget, but it eats into `think()`); transcripts, memory and rewards cross the internet, so the link needs encryption and authentication; rented GPU machines are often temporary, so Jason's state must live on persistent storage; and Jason depends on the internet by day, not only at night.

**M0 portability rules** (cheap now, expensive to retrofit):
1. The brain's address is configuration; nothing assumes the LAN or Windows paths.
2. Body↔brain transport works over the internet from day one (e.g. WebSocket or gRPC over a Tailscale/WireGuard network), so home LAN and cloud look the same to the code.
3. Brain services are containerized (Docker with CUDA) so one image runs on Windows (WSL2) and on cloud Linux.
4. All of Jason's state (memory, raw log, adapters, policy versions) lives in one versioned location that can be synced or moved.
5. Serving and sleep jobs are separate processes, so either can move on its own.
6. Network round-trip time is logged as its own latency stage.

**The body is portable too [Decided].** The ears and mouth may later run on a machine that isn't a Mac. Echo cancellation (WebRTC AEC3), Silero and Smart Turn already run anywhere. Parakeet and Kokoro use MLX, which runs only on Apple Silicon; on the Mac their portable ONNX versions were 37–47% (speech-to-text) and 65% to 3.2× (first audio) slower (`spikes/m1-mac-audio.md`), so they stay on MLX behind swappable backends. Moving the body means adding a backend for the new machine (ONNX or CUDA) and measuring it there. A machine too weak for speech models would need audio to stream to the brain instead, which changes "audio never leaves the body's machine"; not planned.

**Reminder before the brain leaves the home network** (travel or cloud): restrict port 8765 to Tailscale. Today Docker's firewall rules open it to the home LAN, and only the link token protects it.

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

### 5.6 Goal system, drives and emotion-like signals [Proposed — future]

_Owner's idea, 2026-10-30. Not scheduled: earliest after M7, since it needs Wonder (or the judge) to estimate progress toward a goal. Recorded now so earlier choices leave room for it._

**The idea.** Jason forms **goals** in response to stimuli, gives each one a **priority**, and earns reward for actions that move him toward a goal. The owner's example: "it's too hot here" leads to the goal "walk to the fridge". Everything Jason wants comes from three stimulus sources:

| Source | Human example | Jason equivalent | Becomes reward through |
|---|---|---|---|
| **Goals** (learning) | Curiosity, mastery | "Learn new things" is the first standing goal; later, goals Jason forms himself | Learning progress (§5.3 A) and progress toward each goal |
| **Survival** (homeostasis) | Hunger, temperature, fatigue | Internal sensors: time since last sleep and size of the unconsolidated log, GPU temperature and memory, latency, link health, error rate | Drive reduction: reward when a variable moves back toward its set point |
| **Social** | Approval, belonging | Josh's reactions (§5.3 B); time since the last conversation | Josh's reactions |

This would replace the two reward families in §5.3 with three: curiosity becomes the first goal, and survival is new.

**Built for a body.** Jason starts as a voice agent, but the owner plans to add more sensors (inputs) and motors or other outputs (actions). So drives and goals are defined over generic channels, not over speech:
- Every sensor is an input stream with an optional set point (temperature, battery, distance). Adding a sensor adds a survival drive without changing the reward code. This extends the pluggable-inputs decision (§2).
- Every output is an action channel. Speech stays under the floor controller (§4). Motors would add a layer above it: a goal picks a skill ("go to the fridge"), and a lower-level controller runs the motor commands. That is hierarchical RL (options), and it changes the decided action space, so it needs its own decision when it arrives.
- Physical outputs get hard safety limits outside anything learned (speed, force, a stop switch), the same way rate limits sit outside the floor controller today.

**How it would work**
1. **Stimulus to goal.** Sensors publish events, and the core turns each into a short text statement ("I've gone 20 hours without sleeping"). If no similar goal is active, a goal is created with a priority; if one is, its priority is raised or reassessed.
2. **Where goals live.** Active goals sit in a goal list loaded into short-term memory, like an agenda. Completed and abandoned goals are summarized into long-term memory during sleep, so Jason knows what he has done before.
3. **Progress reward.** Each action is scored by how much it changes the estimated distance to the goal, using potential-based shaping: r = γ·Φ(after) − Φ(before), where Φ is the estimated closeness to the goal. Real progress is rewarded, and going in circles earns nothing.
4. **Speed.** Handled by the discount γ and a small time cost per step, not a separate speed bonus; a speed bonus would encourage declaring goals done early.
5. **Reward mix.** total = social + β_g × goal progress + β_s × survival, with both β small and capped so self-made rewards can't outweigh Josh's reactions.

**Guardrails against easy self-made goals** (the owner's question: what stops Jason creating goals just to complete them?)
- Goal reward comes from progress, and repeats pay less: each new goal is compared with goal history (embedding similarity), and near-duplicates of completed goals earn little.
- Completion is judged from evidence in the event log by a separate checker (the reaction judge, or the hosted model at night), never by Jason's own claim.
- Every goal must point to the event that caused it.
- Jason can't change his reward functions; they change only during sleep, behind the eval gate.
- Warning sign to track: goal completions rising while Josh's positive reactions stay flat means Jason is gaming his own goals.

**Emotion-like signals.** Rather than adding emotions as extra rewards, which Jason would learn to chase, compute them from the reward signals and use them to adjust how he acts and learns:

| Signal | Computed from | Effect |
|---|---|---|
| Satisfaction | Reward better than expected | Strengthen what just happened |
| Frustration | Repeated lack of progress on a goal | Change approach, ask Josh for help, or drop the goal |
| Caution (fear) | Wonder predicts a negative reaction | Fewer interjections, more `think()` |
| Sadness | Falling trend in social reward | Lower curiosity weight, listen more |
| Boredom | Learning progress near zero | Explore more, form new goals |

They are logged like every other signal, so their effect can be measured, and Jason can say them out loud ("I'm stuck on this").

**Human reward variables translated**

| Human variable | Jason equivalent |
|---|---|
| Homeostatic needs (hunger, thirst, temperature, fatigue) | Survival drives from internal sensors |
| Reward prediction error (dopamine) | Reward minus the value the policy expected |
| Wanting vs. liking | Goal priority (wanting) and reward on completion (liking), tracked separately |
| Novelty and curiosity | Learning progress (§5.3 A) |
| Mastery | Getting faster or more reliable at a kind of goal |
| Autonomy | Goals Jason forms himself |
| Social approval and belonging | Josh's reactions; time since the last conversation |
| Effort | Time cost per step; compute cost |
| Delay discounting | The discount γ |
| Loss aversion | Negative rewards weighted more heavily than positive ones |
| Baseline vs. spike | Reward measured against a running average (the advantage), so steady rewards fade and surprises stand out |
| Sensitivity (receptor density) | Per-source weights (the β values): Jason's temperament, set by hand rather than learned |
| Hormonal state | Mood-like state from the emotion signals, which scales sensitivity (high frustration lowers curiosity) |
| Subjective value | What an event is worth depends on current goals and drives, not a fixed number per event |
| Deprivation | The reward for fixing a drive scales with how far off it was |
| Probability and uncertainty | Wonder's predicted chance of success; uncertainty also prompts questions (§5.3 A) |
| Environmental cues | Learned value of situations that predict reward (the critic), once an actor-critic exists |
| Intrinsic vs. extrinsic | Goals and survival (intrinsic) vs. Josh's reactions (extrinsic) |

**Human reward stimuli translated.** From the owner's reference list, which sorts stimuli by origin (outside or inside the body) and by whether the reward comes from gaining something or from relief:

| Origin | Gain (positive reinforcement) | Relief (negative reinforcement) |
|---|---|---|
| **External** | Biogenic: resources the body needs (with a body: charging, cooling). Social: Josh's approval, being included. Novelty: new information or places, scored as learning progress | Threat cessation: a danger signal clearing (obstacle, overheating). Noxious mitigation: leaving a harmful environment (noise, heat). Conflict resolution: a disagreement or correction with Josh resolved |
| **Internal** | Cognitive consistency: predictions improving, a question answered, a goal completed. Deficit sensitization: a large drive makes related rewards worth more. Neurohormonal surges: no direct equivalent; the satisfaction signal is closest | Homeostatic correction: a variable returning to its set point. Pain cessation: an error or fault state clearing. Distress mitigation: the frustration or caution signal easing |

**Relief counts as reward.** Drive reduction already rewards leaving a bad state, so gain and relief come from one formula: reward equals the drop in a drive. Deficit sensitization follows from it, since fixing a drive that was far off earns more.

**Left out on purpose.** Secondary tokens (points or scores Jason could pile up) invite reward hacking, and status means nothing with one user. Jason must also never use unpredictable rewards on Josh to keep him engaged (the gambling and social-media pattern in the reference); this joins the sycophancy safeguards in §5.4.

**How we'd know it works:** survival variables stay in range without Josh stepping in; Jason completes held-out goals faster over time; the gaming warning sign stays quiet; and Josh's reaction rate is no lower than without goals (compared on the replay set).

**Risks and prerequisites:** estimating progress for each action is research-grade and noisy; a voice agent's "body" is thin, so most goals will be conversational or about knowledge; adopting this changes the decided reward structure in §5.3 and needs a decision-log entry.

**Related research:** MAGELLAN (2025; LLM agents choosing their own goals by predicted learning progress), homeostatic RL (Keramati & Gutkin, 2014), emotion in RL agents (Moerland et al., 2018 survey), EILS (2025; emotion-inspired learning signals). All research-grade.

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
- Floor-controller inputs: VAD state, pause length, Smart Turn end-of-turn probability, streaming partial transcript, pending offers from the brain (below; includes curiosity's questions), recent reactions, time since Jason last spoke.
- **Interjecting: the brain offers, the body decides [Decided: approach; details in M5].** Jason doesn't wait for the end of the user's turn to think. The brain reads every partial transcript as it arrives, judges whether it has something worth saying (an answer, a correction, a question), and drafts it; predicting how the user's sentence ends is a Wonder-style job (§6.1). It sends the body an **offer** (the draft plus an estimate of its value), which becomes one more floor-controller input. The floor controller alone decides whether to `interject`, keep listening, or wait for the turn to end. One decider for the floor avoids collisions (the brain interjecting as the body decides to answer), keeps credit assignment clean, and keeps every speaking decision's probabilities in one log for learning (M4); the body is also where timing is exact. Every `agent_action` records the exact partial or final transcript it reacted to (`based_on_id`, schema 3), so mid-sentence decisions are traceable. Rollout: M1 answers only at the end of a turn (warming the prompt cache with partials was dropped: the persona and history are already cached from the previous turn, and the new utterance is a few dozen tokens; revisit if M1's latency breakdown shows a gap); M3 can log offers without acting on them, to score them against what the user went on to say; M5 makes offers a controller input and turns `interject` on.
- Start rule-based (thresholds), then replace with a learned contextual bandit. Research (Microsoft RESPOND, 2026) exposes "backchannel intensity" and "turn-claim aggressiveness" as dials, which are natural knobs for the policy to learn per context.
- **Barge-in:** if the user speaks while Jason talks, stop TTS within ~200 ms, stop generating, and log what the user actually heard (`reply_heard`); the mild negative for the interrupted action is derived from that record (M3). Reactions and corrections are judged against the heard text, not the full reply.
- **Self-hearing:** echo cancellation on the body (WebRTC AEC3, with what Jason plays as the reference). In the M1 spike no echo lingered after Jason stopped, so the mic stays open while he speaks (which barge-in needs); muting the mic is only a fallback.
- Rate limits on `interject` and `ask` as a safety net while the policy is young.
- Full-duplex speech-to-speech models (PersonaPlex/Moshi) are not used for now: no text-level hooks for memory and learning. Revisit if that changes.

## 10. Conversation requirements
- Target under ~1 s to first audio. Expected budget: turn decision 200–300 ms + STT 50–100 ms + LLM first token 100–200 ms + TTS first chunk 50–200 ms + network ~10 ms median, ~45 ms p99 (measured over Tailscale, `spikes/m0-link-latency.md`) ≈ 0.5–0.8 s.
- Can be interrupted (barge-in).
- Per-stage latency logged from day one.

---

## 11. Stack [Partly decided — options in `stack-survey.md`]

"Stack A — local-first learner". Rows with a Choice are decided; the rest are recommendations pending a decision:

| Component | Recommended | Main alternative | Choice |
|---|---|---|---|
| Language / runtime | Python 3.12, own asyncio core | Pipecat as main framework | **Own asyncio core from day one; borrow Pipecat components only if needed** |
| Live LLM | Qwen3.5-9B, 4-bit, vLLM or llama.cpp server with LoRA hot-swap (Windows) | Gemma 4 12B; hosted LLM (Stack B) | **Local, Qwen3.5-4B (bf16) on the 4090, served by vLLM v0.32.0; nightly LoRA merged into the weights** |
| Night-time helper | Hosted frontier API (judge, reflection, rewrites) | Local only | |
| RL policy | Contextual bandit (Vowpal Wabbit or small PyTorch) | Actor-critic via CleanRL/SB3 later | |
| Weight training | Unsloth + TRL (LoRA, DPO, context distillation) | — | |
| Prompt optimization | DSPy GEPA | Manual | |
| Memory | Own: text entries + search index (e.g. LanceDB / sqlite-vec) | Mem0 as temporary baseline; Letta; Graphiti | |
| Wonder | Same LLM + "predict user" prompt → dedicated adapter / 2–4B model | — | |
| VAD / turn | Silero VAD + Smart Turn v3.x | — | **Silero v6 + Smart Turn v3.2 on `onnxruntime`** |
| STT | Parakeet TDT 0.6B (Mac) | Kyutai STT; hosted | **Parakeet TDT 0.6B v3 via `parakeet-mlx`, whole-utterance re-transcription; ONNX backend for a non-Mac body** |
| TTS | Kokoro-82M → later Chatterbox or Cartesia | ElevenLabs | **Kokoro-82M via `mlx-audio`; ONNX backend for a non-Mac body** |
| Echo cancellation | WebRTC AEC / macOS voice-processing I/O | — | **WebRTC AEC3 via `livekit`, audio I/O via `sounddevice`** |
| Hardware | Mac = audio + controller; Windows 4090 = LLM, memory, learner, sleep | Everything on Windows | **Split into body (Mac today) and brain (Windows today), both portable; Windows is an always-on server on Ethernet, on overnight** |

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
| **M0** | Setup | Repo layout, CLAUDE.md, pinned environment (both machines), event schema, raw logging, LAN link; follows the §3.4 portability rules (configurable brain address, internet-ready transport, containerized brain, single state location, separate serving/sleep processes, network RTT logged) | Mac↔Windows event round trip logged; one entry point |
| **M1** | Conversation loop | Mic → AEC → VAD/Smart Turn → STT → local LLM → TTS → speaker, across the split; barge-in | End-to-end voice conversation, < 1 s median to first audio, per-stage latency logged |
| **M2** | Memory v1 | Raw log, STM file, LTM entries + search, wake-up briefing, manual "sleep" that consolidates | Recalls facts across sessions on a 20-question own-conversation test |
| **M3** | Feedback signals + Wonder logging | Reaction classifier, credit assignment, implicit signals, re-scoring script; Wonder predict-and-log | Classifier agreement with hand labels ≥ target on 100 utterances; Wonder accuracy tracked daily |
| **M4** | First learning loop | Contextual bandit for one decision (proposed: `ask` vs. not) with frozen/learner swap; validated on simulated user first | Learns simulated user's hidden preference; then beats frozen baseline on live reward over 1–2 weeks |
| **M5** | Proactivity | Floor controller with full action set (minus `think()`), rule-based then learned | Interjections/backchannels happen; barge-in and question rates within target |
| **M6** | Sleep v1 | Scheduled sleep: re-score, consolidate memory, policy training, GEPA prompts, eval gate, rollback, sleep report | Nightly report; at least one gated promotion that improves next-day reaction rate |
| **M7** | Wonder v1 + `think()` | Train Wonder nightly; enable `think()` when it beats baseline | `think()` decisions earn higher reward than non-think decisions in comparable situations |
| **M8** | Sleep v2: weights | Nightly LoRA via context distillation + DPO with replay and gate; graduation of lessons | Graduation tests pass; forgetting set stable; next-day reaction rate ≥ previous adapter |
| **M9** | Dreaming | Dyna-style imagined rollouts mixed with real replay | Policy trained with dreams beats policy without, on real held-out data |
| **M10+** | Expand | Vision input, memory-as-RL-action, richer curiosity, goal system and drives (§5.6), bigger model / rented training (Stack C) | — |

---

## 14. Open questions

Resolved 2026-10-27: core learning/Wonder/memory/turn-taking design; live LLM runs locally; own core from day one; Windows laptop is an always-on server.

Resolved 2026-10-29: serving engine is vLLM (measured against llama.cpp in `docs/spikes/m0-llm-serving.md`; revisit if GPU memory becomes the limit in M2–M3).

Resolved 2026-10-29 (training spike): the live model is the 4B in bf16, trained nightly with a bf16 LoRA and merged into its weights; the 9B returns only with a rented GPU (`docs/spikes/m0-lora-training.md`).

Can wait:
- Embedding model for long-term memory search (needed by M2; §3.3 model 8).
- Where the brain's database files live (needed by M2): a WSL2 volume measured about 15× faster for bulk writes and 100× faster for reads than the `C:\` mount (`docs/spikes/m0-lora-training.md`).
- Nightly LoRA shape (M8): a fresh adapter on yesterday's merged weights, or one evolving adapter on the fixed base, re-merged nightly.
- Reaction-judge model: the live LLM with a prompt vs. a small separate classifier (needed by M3; §3.3 model 7).
- First learned behavior (needed by M4; proposed `ask` vs. not).
- Hosted API provider and monthly cap for night jobs (needed by M6).
- Sleep schedule and idle triggers (M6).
- Exact retention-score formula and thresholds (M2/M6).
- How to define "new information" robustly (M7).
- Wonder implementation: prompt vs. adapter vs. separate small model (M7).
- Whether memory writes become an RL action (M10+).
- Goal system, survival drives and emotion-like signals (§5.6; after M7).

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

---

## 16. Relevant docs

### M0 docs
- **Field guide:** [Jason M0 Field Guide](https://claude.ai/artifact/HXBNsBHAPb977sAUr4WVni)
- **Handbook:** [Jason M0 Handbook](https://claude.ai/artifact/J5moMxUAgeEXWxVPFYwCbv)
- **Implementation doc:** [Jason M0: Project skeleton, event bus and logging](https://docs.google.com/document/d/11cAp3TL_fMorMZKkgWSLmoUKQTEjG0fYo3Esg6Ux9do/edit?tab=t.0)
