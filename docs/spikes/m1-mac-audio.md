# Spike: Mac audio on the M5 Air (M1)

_Run 2026-10-30 on the MacBook Air (Apple M5, 16 GB, macOS 27.0.1), built-in
mic and speakers, one quiet room, speaker volume 50%. Kit and raw commands:
`spikes/m1-mac-audio/`. Raw results: `~/jason-data/spikes/m1-mac-audio/results/`._

## Question

Can the Mac hear the user while Jason talks, without hearing Jason (echo
cancellation, needed for barge-in, design §9)? And are the Mac's models
(design §3.3 models 1 to 4) fast and accurate enough for the 1 s budget
(design §10)?

## Recommendation

| Job | Choice | Measured |
|---|---|---|
| Echo cancellation | **WebRTC AEC3 through `livekit` 1.2.0, with `sounddevice` 0.5.6** for mic and speakers (setup B) | Jason's echo 19 dB down, to the room-noise level; 0.1% false speech detections while he talks; both interruptions transcribed (one small word differed) |
| Voice activity | **Silero VAD v6.2.3**, the ONNX model run on `onnxruntime` 1.31.1 directly | 0.1 ms per 32 ms chunk |
| End of turn | **Smart Turn v3.2 (CPU build)** on `onnxruntime` | 32 ms per call; all 10 finished lines ≥ 0.83 (median 0.98), all 3 unfinished ≤ 0.006 |
| Speech-to-text | **Parakeet TDT 0.6B v3 via `parakeet-mlx` 0.5.3**, whole utterance, **not** its streaming mode (Apple Silicon only) | 94–97 ms median on your lines; 3.7% word errors, all on project jargon |
| Text-to-speech | **Kokoro-82M via `mlx-audio` 0.6.3** (with `misaki[en]`; Apple Silicon only) | 114 ms to the first audio of a clause; 20× faster than real time |

Echo cancellation, Silero and Smart Turn run on any machine. The portable
(onnxruntime) versions of Parakeet and Kokoro were 37–47% and 65%+ slower on
this Mac, so MLX stays for now behind a swappable backend (see "Portable
versions" below).

macOS voice processing (setup A) cancelled echo just as well. B is chosen
because it runs from Python in 10 ms blocks today (A delivers 100 ms blocks,
and its low-latency path crashed Python), works with any mic and speaker pair
(A needs them to be the same device, so not AirPods), runs on Linux too, and
lets Jason know exactly which samples were played, which barge-in needs to
log the text the user actually heard.

## Echo cancellation

Jason spoke three clips (37 s in all, Kokoro voice `am_michael`) through the
speakers. **Silent:** you stayed quiet. **Talk:** you read a line over clips 2
and 3. Setup B also saved the raw mic, which is setup C (no echo
cancellation).

**Silent runs**

| | A: macOS voice processing | B: WebRTC AEC3 | C: none (B's raw mic) |
|---|---|---|---|
| Room noise | −62.0 dBFS | −53.3 dBFS | −51.8 dBFS |
| Mic while Jason talks | **−69.8 dBFS** (below its own noise) | **−55.9 dBFS** (at the noise level) | −36.8 dBFS |
| Silero "speech" while Jason talks | **0.0%** of frames | **0.1%** | 82.8% |
| … 0–0.5 s, 0.5–1 s, 1–2 s after he stops | 0, 0, 0% | 0, 0, 0% | 0, 0, 0% |
| What Parakeet transcribes from the echo | nothing | nothing | all three clips, word for word |

A applies its own noise suppression and gain control too, which is why its
levels sit below the room noise. Its mic level while Jason talks is 33 dB
below C's at the same volume, but part of that is noise suppression, so it
isn't a pure echo figure. B's reduction is 19 dB (−36.8 → −55.9), which
takes the echo down to the room noise.

No echo lingered after Jason stopped, in any setup. An iOS report suggested
muting the mic for 0.5–0.8 s after playback; in this room it isn't needed.

**Talk runs** (your interruption, read over Jason)

| | A | B | C |
|---|---|---|---|
| "Wait, Jason, stop for a second." | exact | exact | "It needs to know exactly what I played…" (Jason's words) |
| "Sorry to interrupt, can you repeat that last part?" | "…the last part?" | "…the last part?" | Jason's words, then yours |
| Silero fires when your voice starts | within 0.1 s | within 0.1 s | always on (echo) |

The time from the cue to detection (1.3–2.1 s) is mostly your reading time;
in every case Silero fired within one 100 ms step of your voice's level
rising in the same recording. Without echo cancellation (C), barge-in is
impossible: the VAD hears Jason 83% of the time.

**Setup A in detail.** Voice processing switches on from Python (PyObjC
`setVoiceProcessingEnabled`) when one AVAudioEngine both plays and records.
The input then has 9 identical channels, and the input tap delivers 4,800
frames (100 ms) at a time whatever size is asked for, adding up to 100 ms
before the VAD hears anything. `AVAudioSinkNode` delivers 512 frames
(10.7 ms), but reading its buffer from Python crashed the process. If A is
ever needed, the route is a small Swift helper that streams audio to Python.

**Setup B in detail.** One duplex PortAudio stream at 48 kHz in 10 ms blocks;
each block played goes into AEC3 as the reference, each mic block comes out
cleaned. The measured speaker-to-mic delay (cross-correlation of the raw mic
with what was played) is **111 ms**; PortAudio's own latency figures varied
from 0.09 to 0.39 s between runs and aren't reliable. One status flag (a
buffer glitch) was raised per run.

## Models

From `bench_models.py` (Kokoro-generated speech) and your 13 recorded lines.
Times after warm-up.

**Parakeet (speech-to-text), whole utterance**

| | v2 (English) | v3 (multilingual) |
|---|---|---|
| Your lines (1–6 s), median | 94 ms | 97 ms |
| Generated speech: 2.9 / 6.8 / 13.7 s | 70 / 96 / 161 ms | 71 / 109 / 191 ms |
| Word errors, your 10 finished lines | 5.8% | 3.7% |
| Load | 1.4 s | 0.5 s (cached) |

Every error was project vocabulary, the same in both versions: "floor" →
"flow", "LoRA" → "lawyer", "Tailscale" → "tail scale", "Qwen" → "quan" /
"quant", "curiosity" → "Christie" (v2) / "Christy" (v3). v3 got "bf16" right
where v2 wrote "Pf sixteen". The two are within noise of each other on 13
lines; v3 is recommended for that one line and for being the newer model.

**Streaming mode is not usable.** At its default window, each 160 ms of audio
takes about 210 ms to process (slower than real time). Smaller windows are fast
but wrong (54–100% word errors). Instead, transcribe the whole utterance so
far: every ~0.5 s for the partial transcript the floor controller reads
(60–170 ms per call for 0.5–13.5 s of audio), and once at the end. The final
call can start as soon as Silero hears a pause, so it runs while Smart Turn
and the floor controller decide, hiding most of its time.

**Smart Turn v3.2.** On your recordings, cut 200 ms after Silero's last speech
frame (what a live controller would see): finished lines 0.83–0.99 (median
0.98; the lowest were "Okay." and "Hey Jason, what did we work on
yesterday?"), unfinished lines 0.005–0.006. On Kokoro's synthetic speech it
was unreliable (0.06–0.99 on finished sentences), so judge it on real voices
only.

**Kokoro.** First audio of a one-clause reply: 114 ms median (103–322 ms over
25 runs of 5 clauses). The first call of a run takes about 15 s: `misaki`
downloads a spaCy English model on first use and then loads it.

**Memory.** Each MLX model alone peaked at 2.2–2.3 GB (Kokoro and Parakeet were
measured separately); the process's resident memory peaked at 1.4 GB. Both
together, plus the rest of Jason, still need measuring on the 16 GB Mac
(M1 step 10).

## Portable versions of Parakeet and Kokoro

The ears and mouth may later run on a machine that isn't a Mac (decision
2026-10-30, the `body` role), and `parakeet-mlx` and `mlx-audio` run only on
Apple Silicon. `bench_onnx.py` compares them with the same models on
`onnxruntime`, which runs anywhere: Parakeet v3 through `onnx-asr` 0.12.0
(`istupakov/parakeet-tdt-0.6b-v3-onnx` at `8f23f0c`) and Kokoro v1.0 through
`kokoro-onnx` 0.6.1. Same Mac, same voice and lines.

| | MLX | ONNX fp32, CPU | ONNX int8, CPU | ONNX fp32, CoreML |
|---|---|---|---|---|
| Speech-to-text, your 10 lines (median) | **95 ms** | 131 ms | 95 ms | 285 ms |
| … word errors | **3.7%** | 4.5% | 7.6% | 4.5% |
| … 13.7 s generated clip | **193 ms** | 285 ms | 240 ms | 786 ms |
| Text-to-speech, first clause (median) | **129 ms** | 412 ms | 213 ms | 470 ms |

ONNX is 37–47% slower for speech-to-text at the same accuracy (int8 matches
the speed but doubles the errors), and 65% (int8) to 3.2× (fp32) slower for
the first audio of a reply. CoreML acceleration makes both worse: it supports
only part of each graph, so the work bounces between CoreML and the CPU.
`kokoro-onnx` also phonemizes with espeak, which warned of a word-count
mismatch on every clause. One clause from each engine is saved in
`~/jason-data/spikes/m1-mac-audio/results/tts_samples/` for listening.

**Decision: keep MLX on the Mac for now.** The loss is too large for the
first-audio budget. Speech-to-text and text-to-speech sit behind interfaces
with the backend chosen in the body's config, so a non-Mac body adds an ONNX
(or CUDA) backend later without touching the rest. On a machine with an NVIDIA
GPU the comparison would be different, so measure again on that machine.

## Latency budget (Mac side)

| Stage | Measured | Budget (§10) |
|---|---|---|
| Speaker → mic round trip (B) | 111 ms | — |
| VAD | 32 ms frames, 0.1 ms each | — |
| Smart Turn | 32 ms per call | turn decision 200–300 ms, mostly the pause wait |
| Speech-to-text, final | 70–170 ms, mostly hidden in the turn decision | 50–100 ms |
| Text-to-speech, first clause | 114 ms | 50–200 ms |

## What this means for M1

- **Step 7, audio I/O:** one `sounddevice` duplex stream at 48 kHz in 10 ms
  blocks with AEC3 in the callback; resample to 16 kHz for Silero, Smart Turn
  and Parakeet. Because the playback buffer is ours, the Mac knows exactly how
  much of a reply was played when the user barges in (step 12).
- **Step 8, VAD and turn:** Silero and Smart Turn on `onnxruntime`, no torch.
  Smart Turn's reference code uses `transformers` only for Whisper's log-mel
  features; replace that with a small numpy version, tested against
  `transformers` output, rather than shipping `transformers`.
- **Step 9, speech-to-text:** whole-utterance re-transcription, as above.
  Project words will be misheard; the LLM usually copes from context, and
  memory (M2) can hold a vocabulary list for the prompt.
- **Step 10, text-to-speech:** `misaki[en]` pulls in spaCy and PyTorch (the
  spike environment is 1.5 GB) and downloads a spaCy model at first use.
  Pin that model as a dependency so nothing downloads at runtime; `kokoro-onnx`
  is the lighter alternative to measure if the weight matters.
- **Packaging:** `mlx`, `parakeet-mlx` and `mlx-audio` run only on macOS, and
  the brain image is Linux. The body's dependencies go in an optional group:
  the portable ones (`sounddevice`, `livekit`, `onnxruntime`) unmarked, the MLX
  ones with a `sys_platform == 'darwin'` marker, so `uv.lock` still resolves
  for the brain and for a future non-Mac body. PyObjC isn't needed (setup A
  wasn't chosen).

## Limits of this spike

One room, one volume, one speaker, built-in mic and speakers only; 13 lines
and 2 interruptions per setup. Not tested: headphones or AirPods (Bluetooth
adds delay), louder volumes, a noisy room, other voices. Barge-in timing end to
end (stop speaking within 200 ms) is measured in M1 step 12.

## Follow-ups

- **Who is speaking:** speaker embeddings (e.g. TitaNet, ECAPA-TDNN) or a
  personal VAD that fires only for the enrolled user's voice, on top of echo
  cancellation: rejects leftover echo, other people and the TV, and lets
  memory record who said what. Candidate for M5 or M10+.
- AEC3's own noise suppression (`noise_suppression=True`) wasn't tried; A's
  much lower noise floor suggests it may help STT in noisier rooms.
