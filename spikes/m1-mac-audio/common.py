"""Shared pieces of the Mac audio spike: paths, resampling, Silero VAD and
Smart Turn run straight on onnxruntime, Kokoro and Parakeet loaders, timing.

Recordings hold the owner's voice, so they live in ~/jason-data (never in git),
next to the downloaded models.
"""

import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr

DATA = Path(os.environ.get("JASON_SPIKE_DATA", "~/jason-data/spikes/m1-mac-audio")).expanduser()
MODELS = DATA / "models"
REC = DATA / "recordings"

VAD_SR = 16_000  # Silero, Smart Turn and Parakeet all take 16 kHz mono
IO_SR = 48_000  # the built-in mic and speakers run at 48 kHz

# Pinned so the numbers in the write-up can be reproduced.
PARAKEET = {
    "v2": ("mlx-community/parakeet-tdt-0.6b-v2", "8ae155301e23d820d82aa60d24817c900e69e487"),
    "v3": ("mlx-community/parakeet-tdt-0.6b-v3", "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"),
}
KOKORO = ("mlx-community/Kokoro-82M-bf16", "a71e4d38b236d968966a2002c4c895dbd12b1c3c")
KOKORO_VOICE = "am_michael"


def now_ms() -> float:
    return time.perf_counter() * 1000


def peak_rss_mb() -> float:
    # macOS reports ru_maxrss in bytes (Linux uses kilobytes).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    return soxr.resample(x.astype(np.float32), sr_in, sr_out).astype(np.float32)


def save_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, sr, subtype="FLOAT")


def load_wav(path: Path, sr: int | None = None) -> np.ndarray:
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    return resample(x, file_sr, sr) if sr else x


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    return 20 * np.log10(max(rms, 1e-10))


def _session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    # One thread each: on the Mac these share the CPU with audio I/O, and
    # the models are small enough that more threads mostly add overhead.
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class SileroVAD:
    """Silero VAD v6 on onnxruntime only (the silero-vad package needs torch).

    Takes 512-sample chunks at 16 kHz (32 ms). Like the official wrapper, each
    chunk is fed with the last 64 samples of the previous one in front of it,
    and the recurrent state is carried between calls.
    """

    CHUNK = 512
    CONTEXT = 64

    def __init__(self) -> None:
        self.sess = _session(MODELS / "silero_vad.onnx")
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.CONTEXT, dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        assert len(chunk) == self.CHUNK
        x = np.concatenate([self.context, chunk.astype(np.float32)])[None, :]
        out, self.state = self.sess.run(
            None, {"input": x, "state": self.state, "sr": np.array(VAD_SR, dtype=np.int64)}
        )
        self.context = x[0, -self.CONTEXT :]
        return float(out[0, 0])

    def probs(self, audio16k: np.ndarray) -> np.ndarray:
        """Speech probability for every 32 ms chunk of a whole clip."""
        self.reset()
        n = len(audio16k) // self.CHUNK
        return np.array([self(audio16k[i * self.CHUNK : (i + 1) * self.CHUNK]) for i in range(n)])


class SmartTurn:
    """Smart Turn v3.2 (CPU build): probability the speaker has finished.

    Input is the last 8 s of the turn at 16 kHz, zero-padded at the front,
    turned into Whisper log-mel features (80 x 800).
    """

    SECONDS = 8

    def __init__(self) -> None:
        from transformers import WhisperFeatureExtractor

        self.fe = WhisperFeatureExtractor(chunk_length=self.SECONDS)
        self.sess = _session(MODELS / "smart-turn-v3.2-cpu.onnx")

    def __call__(self, audio16k: np.ndarray) -> float:
        n = self.SECONDS * VAD_SR
        x = audio16k[-n:]
        x = np.pad(x, (n - len(x), 0))
        feats = self.fe(
            x, sampling_rate=VAD_SR, return_tensors="np", padding="max_length",
            max_length=n, truncation=True, do_normalize=True,
        ).input_features.astype(np.float32)
        return float(self.sess.run(None, {"input_features": feats})[0][0, 0])


def load_parakeet(version: str):
    from huggingface_hub import snapshot_download
    from parakeet_mlx import from_pretrained

    repo, rev = PARAKEET[version]
    return from_pretrained(snapshot_download(repo, revision=rev))


def transcribe(model, audio16k: np.ndarray) -> str:
    """Whole-utterance transcription from an array (the library's own
    transcribe() only takes a file path)."""
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel

    mel = get_logmel(mx.array(audio16k, dtype=mx.float32), model.preprocessor_config)
    return model.generate(mel)[0].text.strip()


def load_kokoro():
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model

    return load_model(Path(snapshot_download(KOKORO[0], revision=KOKORO[1])))


def kokoro_say(model, text: str) -> tuple[np.ndarray, int]:
    """Whole clip for one text, at Kokoro's own rate (24 kHz)."""
    parts = [np.asarray(r.audio, dtype=np.float32).reshape(-1) for r in model.generate(text=text, voice=KOKORO_VOICE)]
    return np.concatenate(parts), model.sample_rate


def words(text: str) -> list[str]:
    keep = "".join(c.lower() if c.isalnum() or c in " '" else " " for c in text)
    return keep.split()


def wer(ref: str, hyp: str) -> float:
    """Word error rate: word-level edit distance over the reference length."""
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)
