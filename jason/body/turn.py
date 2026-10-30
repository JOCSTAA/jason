"""End-of-turn detection: Smart Turn v3.2 (design §3.3 model 2).

Given the user's utterance so far, the model says how likely it is that they
have finished their turn (rather than pausing mid-thought). It hears prosody
(pitch, rhythm, a trailing "um"), not words. On the owner's voice in the M1
spike, finished lines scored 0.83-0.99 and trailing-off ones 0.005-0.006.

The model expects Whisper's log-mel spectrogram of the last 8 s of audio. Its
reference code computes that with the `transformers` library; log_mel() below
is the same calculation in numpy, checked against transformers' output in
tests/test_turn.py, so the body doesn't need that large library.
"""

from functools import cache
from pathlib import Path

import numpy as np

SR = 16_000
SECONDS = 8  # the model looks at the last 8 s, zero-padded in front if shorter
N_FFT = 400  # 25 ms windows
HOP = 160  # every 10 ms
N_MELS = 80


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    """The Slaney mel scale: linear below 1 kHz, logarithmic above."""
    f = np.asarray(f, dtype=np.float64)
    mel = 3.0 * f / 200.0
    log_part = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
    return np.where(f >= 1000.0, log_part, mel)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f = 200.0 * m / 3.0
    log_part = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
    return np.where(m >= 15.0, log_part, f)


@cache
def mel_filters() -> np.ndarray:
    """80 triangular filters over 0-8 kHz, each normalized to equal area
    (Slaney). Shape: (201 frequency bins, 80 mel bands)."""
    fft_freqs = np.linspace(0, SR / 2, 1 + N_FFT // 2)
    mel_edges = np.linspace(_hz_to_mel(0.0), _hz_to_mel(8000.0), N_MELS + 2)
    hz_edges = _mel_to_hz(mel_edges)
    lower, center, upper = hz_edges[:-2], hz_edges[1:-1], hz_edges[2:]
    up = (fft_freqs[:, None] - lower) / (center - lower)
    down = (upper - fft_freqs[:, None]) / (upper - center)
    filters = np.maximum(0.0, np.minimum(up, down))
    return filters * (2.0 / (upper - lower))


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Whisper's features for Smart Turn: (1, 80, 800) float32."""
    n = SECONDS * SR
    x = np.asarray(audio, dtype=np.float64)[-n:]
    x = np.pad(x, (n - len(x), 0))  # pad in front: the speech sits at the end
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)  # zero mean, unit variance
    x = np.pad(x, N_FFT // 2, mode="reflect")  # centre each window on its sample
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    window = np.hanning(N_FFT + 1)[:-1]  # periodic Hann
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = power @ mel_filters()
    spec = np.log10(np.maximum(mel, 1e-10)).T[:, :-1]  # 80 x 800 (the last frame is dropped)
    spec = np.maximum(spec, spec.max() - 8.0)  # an 80 dB range below the loudest point
    return ((spec + 4.0) / 4.0).astype(np.float32)[None]


class SmartTurn:
    def __init__(self, model_file: Path) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        # One thread: it shares the CPU with audio I/O, and is small enough
        # (32 ms per call on the M5 Air) that more threads mostly add overhead.
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(model_file), sess_options=so, providers=["CPUExecutionProvider"])

    def __call__(self, audio16k: np.ndarray) -> float:
        """Probability that the user has finished their turn."""
        return float(self._sess.run(None, {"input_features": log_mel(audio16k)})[0][0, 0])
