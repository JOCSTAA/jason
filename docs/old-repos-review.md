# Old Repos Review (reference)

_Reviewed 2026-10-27. Summary of lessons lives in design.md §11; this file keeps the detail for later reference._

Repos (archived, read-only reference — no code carried over):
- `JOCSTAA/AI` — https://github.com/JOCSTAA/AI (2022-06 → 2023-09)
- `JOCSTAA/speech` — https://github.com/JOCSTAA/speech (~2024; single commit "initial" dated 2026-10-27, original history lost)

## 1. Repo map

### AI — branches
- `main` (2023-08-21): PPO actor-critic + Wonder, LSTM models, sine-wave output.
- `transformers` (2023-09-22) — **most recent; reference branch for original ideas.** Adds Transformer actor/critic/wonder, short-term/long-term memory with "sleep" replay, fixed audio capture, spectrogram output.
- `refactor` (2023-09-07): intermediate step, subset of `transformers`.
- `wonder_feature` (to 2023-04): old flat layout; `Bipedal/scrap.py` (PyLessons BipedalWalker PPO reference — third-party, correct GAE/critic), `Redacted/` (~20 earlier PyTorch/TF actor-critic experiments).

### Where the original ideas live (on `transformers`)
| Idea | File / function |
|---|---|
| Wonder world model (predict next observation from obs+action) | `models/model.py` (`IndividualModel(wonder=True)`), trained in `models/model_utils.py::train_model` |
| Planning with imagined futures | `models/model_utils.py::think` / `explore_future_actions` (stub: confidence = 0) |
| Acting copy vs training copy, swap | `models/model_utils.py::update_model`, `update_model_weights` |
| Short-term / long-term memory, sleep replay | `memory/memory.py` (`ST_MemoryUnit`, `LT_MemoryUnit`), `memory/memory_utils.py::long_term_write`, `train_model(sleep=True)` |
| Reward hook (never implemented) | `models/model_utils.py::predict` (`reward_function`, default reward = 1) |
| Mic capture (callback stream) | `interface/inputs/ears/hear.py` |

### speech — files
`main.py` (Whisper tiny live transcription; Llama-3.1-8B call commented out), `mistral_stt.py` (QuartzNet STT → Mistral-7B, text reply — the end-to-end path), `sample.py` (record-until-silence + Whisper base), `whisper_test.py`, `llama.py`/`llama_run.py` (Llama-3.1-8B base), `test_mistra_quantized.py`, `test_models_with_quantization.py` (8-bit Mistral-7B / Llama-2-7B), `speech_to_text.py` (commented out).

## 2. Owner's answers (2026-10-27)
- **Paddle test:** only used to check understanding of RL methods — drop entirely.
- **Vision:** important long term (wants the agent to see and hear); audio only for now.
- **Reward intent:** reward for any new information learned or anything seen, so the agent is inherently curious and avoids boredom. Never figured out how to code it.
- **Wonder:** meant for both curiosity and planning.
- **Sleep:** schedule never decided — idea was the agent sleeps for a designated time set later. Long-term memory was meant to be context for the model *and* training data replayed during sleep, "similar to how humans learn".
- **Direction:** whichever branch has Wonder and is newest is the direction → `transformers`.
- **speech:** written ~2 years ago; worked end to end; a separate project to test STT logic, not meant to feed the PPO brain. Whisper vs QuartzNet: just trying options.
- **Hardware:** undecided; to be researched later in the project. (Old environment was Windows + NVIDIA GPU, CUDA 11.2; speech used CUDA with 8-bit 7B models.)

## 3. What the code actually did
- **AI:** always-on loop: webcam frame + mic audio → images (MFCC rendered via matplotlib) → PPO actor samples action → action played as sound (main: 5 sine waves; transformers: 128×128 spectrogram via iSTFT) → store transition → train every 256 steps (main) / 5 steps (transformers), blocking the loop.
- **speech:** turn-based mic → STT → (LLM) → print. No TTS, no memory, not connected to AI.

## 4. Bugs found (why nothing could have learned)
1. Reward hard-coded to 1 → after return normalization becomes a position-in-batch ramp; pure artifact.
2. Critic trained toward its own old predictions (`(value − old_value)²`), not returns → never learns. Persists on `transformers`. (The PyLessons reference did it correctly; lost in the port.)
3. Advantages: normalized returns minus unnormalized values; returns truncated per minibatch (main); no episode boundaries or bootstrap; GAE λ defined but unused.
4. `transformers`: PPO ratio averaged over batch before multiplying advantages; critic outputs one scalar per batch.
5. `main` audio input decoded raw bytes into a list (stereo int32 as bytes) → "ears" were noise; thread race. Fixed on `transformers`.
6. MFCC → matplotlib PNG → image: slow, adds colormap, variable-length windows stretched (time axis meaningless).
7. Output can't produce speech; 16,384-dim continuous action space (comment: "breaks at > 4096").
8. Wonder predicts ~240k pixels through 32-unit bottleneck; target scales mismatched.
9. Tiny LSTMs (2 and 4 units). Training blocks acting.
10. Long-term memory: `lt_length` undefined (crash at 1000 steps); reading LT memory deletes the file; sleep never triggered.
11. Windows-only paths (`..\..`), `Temp` vs `temp`, `see.py` uses ears shape, `transformers` paddle test uses old API.
12. speech: `main.py` overwrites the current line with only the newest chunk (loses words); base (non-chat) LLMs with chat prompts; `max_length=50` includes prompt; `sample.py` silence logic inverted; fixed 2 s windows cut words; LLM blocks audio loop.

## 5. Keep / adapt / drop
| Component or idea | Verdict | Reason |
|---|---|---|
| Always-on input/output loop | Adapt | Basis for proactivity; needs silence action, turn-taking, barge-in |
| Acting copy + training copy swap | Adapt | Right pattern for real-time learning; make async with versioning/rollback |
| PPO implementation | Drop | Broken critic/advantages; use tested libraries |
| Policy-gradient as a concept | Adapt | For small high-level decisions, not raw audio |
| Wonder world model | Adapt (idea) | Curiosity + planning; predict in semantic space |
| `think()` planning | Park | Stub; revisit when reward and model exist |
| `reward_function` hook | Adapt | Right slot; reward design comes first |
| Rollout buffer called "Memory" | Drop | Training buffer, not agent memory |
| ST/LT memory + sleep | Adapt (idea) | Rebuild as episode log + background consolidation; LT also readable as context |
| Hand-built Transformer | Drop | Use pretrained models |
| Webcam/vision | Later | Wanted down the line; pluggable input |
| MFCC→image pipeline, sine/spectrogram output | Drop | Use STT/TTS |
| Callback mic streaming, live-transcription loop pattern | Adapt | Fine patterns; models/VAD outdated |
| Paddle env | Drop | Owner's call |
| Local 8-bit LLMs | Adapt (idea) | Local fine-tunable model needed if weights update; models outdated |

## 6. Outdated stack (re-research everything)
TensorFlow 2.10 / Keras 2.10 / TF-Probability 0.18; CUDA 11.2; openai-whisper tiny/base; SpeechRecognition energy-threshold VAD; NeMo QuartzNet15x5 (2019); Llama-2-7B, Mistral-7B-v0.1, Llama-3.1-8B base models; `load_in_8bit=` (deprecated).
