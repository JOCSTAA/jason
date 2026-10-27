# Voice Agent — Design Doc

_Last updated: 2026-10-27_

## 1. Vision
A voice agent that learns and grows in real time. It takes in information as it arrives, remembers it, improves its behavior through reinforcement learning, and takes part in conversation proactively — speaking out of turn and asking questions when it judges that useful.

This is a personal passion project. The measure of success is how capable, adaptive and alive the agent becomes.

## 2. Core capabilities (priority order)
1. **Real-time learning (RL)** — behavior improves from ongoing interaction
2. **Memory** — absorbs information in real time, stores it durably, retrieves and uses it
3. **Proactivity** — interjects, speaks out of turn, asks clarifying or curious questions
4. **Natural voice conversation** — fast enough to feel real (requirement, not the main goal)

## 3. Learning system (to be designed in detail)
_Full design to come. Questions to answer:_
- **What is the reward signal?** (explicit feedback from me, implicit signals like interruptions or follow-ups, task success, self-evaluation…)
- **What gets updated?** (model weights, a separate policy, memory contents, retrieval, prompts/strategies)
- **When do updates happen?** (every turn, after each conversation, in the background)
- **What is the policy deciding?** (what to say, when to speak, when to ask, what to remember)
- **How do we know it's improving?** (metrics, evaluation sessions, before/after comparisons)
- **Safeguards:** reward hacking, catastrophic forgetting, drift, runaway behavior

## 4. Memory system (to be designed in detail)
- What gets stored (facts, events, preferences, episodes, skills)?
- How it's written (every turn, consolidated later)?
- How it's retrieved and used during conversation?
- How it's corrected or forgotten?

## 5. Proactivity / turn-taking (to be designed in detail)
- When is the agent allowed to speak without being prompted?
- How does it decide a question is worth asking?
- How does this interact with learning (does it learn when interjecting is welcome)?

## 6. Architecture (initial sketch)

```
                ┌──────────────── Memory (read/write) ────────────────┐
                ↓                                                     │
Mic → VAD/turn detection → STT → Agent core (LLM + policy) → TTS → Speaker
                                      ↑            │
                         Learning loop (rewards → updates)
                                      ↑
                      Feedback / signals from interaction
```

The agent core must be able to start speaking on its own (proactivity), not only after user input.

## 7. Conversation requirements
- Responses feel natural; target under ~1 s to first audio where possible
- Can be interrupted (barge-in)
- Per-stage latency is logged so it can be tuned later

## 8. Stack decisions
_To be decided after research. Record final choices in decisions.md._

| Component | Options to evaluate | Choice |
|---|---|---|
| Language / runtime | | |
| RL / learning approach | | |
| Memory store & retrieval | | |
| LLM (local vs hosted; fine-tunable?) | | |
| VAD / turn detection | | |
| STT | | |
| TTS | | |
| Agent / tool framework | | |

Note: if the agent should update its own weights, the LLM choice is constrained (local, fine-tunable models).

## 9. Milestones (draft, revise after the learning design)
1. **M0 – Setup:** new repo, CLAUDE.md, logging
2. **M1 – Conversation loop:** mic → STT → LLM → TTS working end to end
3. **M2 – Memory:** agent stores and recalls information across sessions
4. **M3 – Feedback signals:** capture rewards/signals from interactions
5. **M4 – First learning loop:** one behavior measurably improves from feedback
6. **M5 – Proactivity:** agent can speak out of turn and ask questions
7. **M6 – Learned proactivity:** agent learns when interjecting is welcome
8. **M7+ –** expand what it learns

## 10. Open questions
- Local models, hosted APIs, or a hybrid? (Affects whether weight updates are possible.)
- What hardware will it run on?
- What should the agent learn first?

## 11. Lessons from the old repo
_Fill in from the old-repo summary: what to keep, what to drop._
