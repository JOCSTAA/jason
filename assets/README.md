# assets/

Files the body needs at runtime, committed so a fresh clone runs as is on any
machine (no first-run download). Paths come from `[audio] assets_dir` in
`config/body.toml`. The brain's image doesn't include them.

| File | What | Source | License | SHA-256 |
|---|---|---|---|---|
| `silero_vad.onnx` | Silero VAD v6 (voice activity) | github.com/snakers4/silero-vad, tag `v6.2.3`, `src/silero_vad/data/silero_vad.onnx` | MIT (`LICENSE-silero-vad`) | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |
| `smart-turn-v3.2-cpu.onnx` | Smart Turn v3.2, CPU build (end of turn) | huggingface.co/pipecat-ai/smart-turn-v3, revision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93` | BSD-2-Clause (`LICENSE-smart-turn`) | `2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f` |
| `voice_check.wav` | 9 s of Jason's synthetic voice (Kokoro-82M, voice `am_michael`, 24 kHz 16-bit) for `jason audio-check` | made with `spikes/m1-mac-audio` on 2026-10-30 | Kokoro-82M is Apache-2.0; the clip is synthetic, nobody's real voice | `9b13e4e7d39ba472f34f76df0fbf1d0baa24cd652508b8e67dd2346f89181653` |

`tests/test_assets.py` checks the hashes, so a changed or corrupted file fails the tests.
