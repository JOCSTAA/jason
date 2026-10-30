"""`jason audio-check`: is the body's echo cancellation working? (M1 step 7)

Plays a speech-like test sound through the speakers while recording the mic,
then reports: the room's noise level, how loud the echo was before and after
cancellation, how long sound takes from speaker to mic, and whether any audio
blocks were dropped. Stay quiet while it runs (about 10 s).
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from jason.body.audio import IO_SR, AudioIO
from jason.core.config import AudioConfig

LEAD_S, SIGNAL_S, TAIL_S = 1.5, 6.0, 2.0
SETTLE_S = 1.0  # AEC3 adapts to the room in the first second; don't score it


def speechlike_signal(seconds: float = SIGNAL_S, seed: int = 0) -> np.ndarray:
    """Noise shaped like speech: syllables about 4 a second, with short gaps.
    Broadband, so the echo canceller has the whole spectrum to learn."""
    rng = np.random.default_rng(seed)
    n = int(seconds * IO_SR)
    t = np.arange(n) / IO_SR
    noise = rng.standard_normal(n).astype(np.float32)
    noise = np.convolve(noise, np.ones(4) / 4, mode="same")  # soften the hiss a little
    envelope = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 2
    envelope *= (np.sin(2 * np.pi * 0.35 * t) > -0.6)  # an occasional pause
    x = noise * envelope
    return (0.25 * x / np.sqrt(np.mean(x**2) + 1e-12)).astype(np.float32)  # about -12 dBFS while "talking"


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    return 20 * np.log10(max(rms, 1e-10))


def delay_ms(reference: np.ndarray, mic: np.ndarray, max_ms: float = 600) -> float:
    """Speaker-to-mic delay: the lag where the mic best matches what was played."""
    n = min(len(reference), len(mic))
    size = 1 << (2 * n - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(mic[:n], size) * np.conj(np.fft.rfft(reference[:n], size)), size)
    lag = int(np.argmax(xc[: int(max_ms / 1000 * IO_SR)]))
    return 1000 * lag / IO_SR


@dataclass
class CheckResult:
    noise_dbfs: float
    echo_raw_dbfs: float
    echo_clean_dbfs: float
    tail_clean_dbfs: float
    delay_ms: float
    status_flags: int
    blocks: int

    @property
    def reduction_db(self) -> float:
        return self.echo_raw_dbfs - self.echo_clean_dbfs

    @property
    def ok(self) -> bool:
        # The spike's pass mark: echo pushed down to about the room's own noise.
        return self.reduction_db >= 15 and self.echo_clean_dbfs <= self.noise_dbfs + 6 and self.status_flags <= 2

    def lines(self) -> list[str]:
        return [
            f"room noise          {self.noise_dbfs:6.1f} dBFS",
            f"echo, raw mic       {self.echo_raw_dbfs:6.1f} dBFS",
            f"echo, cancelled     {self.echo_clean_dbfs:6.1f} dBFS   ({self.reduction_db:.1f} dB removed)",
            f"after it stopped    {self.tail_clean_dbfs:6.1f} dBFS",
            f"speaker -> mic      {self.delay_ms:6.1f} ms",
            f"audio blocks        {self.blocks} ({self.status_flags} dropped or late)",
            "OK: echo cancelled to the room's noise level" if self.ok else
            "PROBLEM: echo not cancelled well enough (see docs/spikes/m1-mac-audio.md for what good looks like)",
        ]


def score(timeline: np.ndarray, raw: np.ndarray, clean: np.ndarray, status_flags: int, blocks: int) -> CheckResult:
    sec = lambda x, a, b: x[int(a * IO_SR): int(b * IO_SR)]
    start, end = LEAD_S, LEAD_S + SIGNAL_S
    return CheckResult(
        noise_dbfs=dbfs(sec(raw, 0.3, LEAD_S - 0.2)),
        echo_raw_dbfs=dbfs(sec(raw, start + SETTLE_S, end)),
        echo_clean_dbfs=dbfs(sec(clean, start + SETTLE_S, end)),
        tail_clean_dbfs=dbfs(sec(clean, end + 0.2, end + TAIL_S)),
        delay_ms=delay_ms(timeline, raw),
        status_flags=status_flags,
        blocks=blocks,
    )


async def run_check(cfg: AudioConfig, out: Callable[[str], None] = print) -> CheckResult:
    signal = speechlike_signal()
    timeline = np.concatenate([np.zeros(int(LEAD_S * IO_SR)), signal, np.zeros(int(TAIL_S * IO_SR))]).astype(np.float32)
    out(f"Stay quiet: playing a test sound for {len(timeline) / IO_SR:.0f} s ...")
    io = AudioIO(cfg.input_device, cfg.output_device, keep_raw=True)
    io.start()
    try:
        io.play("check", timeline, IO_SR)
        await asyncio.sleep(len(timeline) / IO_SR + 0.3)
    finally:
        io.stop()
    result = score(timeline, np.concatenate(io.raw), np.concatenate(io.clean48), io.stats.status_flags, io.stats.blocks)
    for line in result.lines():
        out(line)
    return result
