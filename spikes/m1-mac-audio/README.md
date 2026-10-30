# M1 spike: Mac audio on the M5 Air

**Questions:**
1. Can the Mac hear the user while Jason talks, without hearing Jason? Compares
   **A** macOS voice processing (`vpio.py`), **B** WebRTC AEC3 through `livekit`
   (`webrtc_aec.py`), and **C** no echo cancellation (B's raw mic).
2. Are Silero VAD, Smart Turn v3.2, Parakeet TDT 0.6B (v2 vs v3) and Kokoro-82M
   fast and accurate enough for the 1 s budget (design §10)?

The write-up goes in `docs/spikes/m1-mac-audio.md`. This folder has its own
`pyproject.toml`, so nothing here touches `jason`'s `uv.lock`. Models and
recordings (your voice) go to `~/jason-data/spikes/m1-mac-audio/`, never git.

Run everything **on the Mac, from this folder**: `cd spikes/m1-mac-audio`.

## 1. Set up (once)
```
uv sync
mkdir -p ~/jason-data/spikes/m1-mac-audio/models && cd ~/jason-data/spikes/m1-mac-audio/models
curl -LO https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx
curl -L -o smart-turn-v3.2-cpu.onnx https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/f766f81d3cfdf7737ac64aad813d91bbfd56bf93/smart-turn-v3.2-cpu.onnx
cd -
```
Parakeet and Kokoro download from Hugging Face on first use (about 2.5 GB),
pinned by revision in `common.py`.

## 2. Model timings (no microphone)
```
uv run python bench_models.py
```

## 3. Your sessions (about 15 minutes)
Quiet room, MacBook speakers and mic (no headphones, no AirPods). The scripts
set the speaker volume to 50% while they run and put it back afterwards. The
first run asks for microphone access: allow it.

**Session 1, read short lines (5 min):**
```
uv run python record_turns.py
```

**Session 2, stay quiet while Jason talks (45 s each):**
```
uv run python echo_session.py --setup B --mode silent
uv run python echo_session.py --setup A --mode silent
```

**Session 3, interrupt Jason (45 s each):** read each SPEAK NOW line aloud as
soon as it appears, at normal volume, while he keeps talking.
```
uv run python echo_session.py --setup B --mode talk
uv run python echo_session.py --setup A --mode talk
```

## 4. Score
```
uv run python analyze.py
```

## 5. Portable (onnxruntime) Parakeet and Kokoro vs MLX
Needs session 1's recordings. Download the Kokoro ONNX files first:
```
cd ~/jason-data/spikes/m1-mac-audio/models
for f in kokoro-v1.0.onnx kokoro-v1.0.int8.onnx voices-v1.0.bin; do curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$f; done
cd -
uv run python bench_onnx.py
```

Results: `~/jason-data/spikes/m1-mac-audio/results/` (`bench_models.json`,
`analysis.json`, `bench_onnx.json`, and `tts_samples/` to listen to).
