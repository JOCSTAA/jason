"""`jason audio-check`: is the body's echo cancellation working? (M1 step 7)

Plays a speech-like test sound through the speakers while recording the mic,
then reports: the room's noise level, how loud the echo was before and after
cancellation, how long sound takes from speaker to mic, and whether any audio
blocks were dropped. Then it plays a clip of Jason's voice and reports how
often the VAD hears speech in the mic: it should be close to never with echo
cancellation (the M1 spike: 0.1%) against most of the time without (83%).
Last, it starts Jason's voice three times after 3 s of silence, as in a
conversation, and checks the VAD doesn't hear any of those starts: the echo
canceller is weakest right after a silence, and a leaked first word looks
like the user cutting in (barge-in). Stay quiet while it runs (about 35 s).
"""

import asyncio
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soxr

from jason.body.audio import IO_SR, MIC_SR, AudioIO
from jason.core.config import AudioConfig

LEAD_S, SIGNAL_S, TAIL_S = 1.5, 6.0, 2.0
VOICE_GAP_S, VOICE_TAIL_S = 1.0, 1.0
ONSETS, ONSET_GAP_S, ONSET_VOICE_S = 3, 3.0, 2.0
ONSET_WINDOW_S = 0.8  # where a leaked first word would trigger a barge-in
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


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """A 16-bit mono WAV as floats, with its sample rate."""
    with wave.open(str(path)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return pcm.astype(np.float32) / 32768, w.getframerate()


def speech_percent(mic48: np.ndarray, model_file: Path) -> float:
    """How often Silero hears speech in this audio, slice by slice."""
    from jason.body.listen import CHUNK, Silero

    vad = Silero(model_file)
    x = soxr.resample(mic48, IO_SR, MIC_SR).astype(np.float32)
    probs = [vad(x[i:i + CHUNK]) for i in range(0, len(x) - CHUNK + 1, CHUNK)]
    return 100 * float(np.mean(np.array(probs) >= 0.5)) if probs else 0.0


@dataclass
class VoiceResult:
    clean_pct: float  # with echo cancellation: the VAD should hear (almost) nothing
    raw_pct: float  # without it, for comparison

    @property
    def ok(self) -> bool:
        return self.clean_pct <= 2.0

    def lines(self) -> list[str]:
        return [
            f"Jason's voice heard as speech: {self.clean_pct:.1f}% with echo cancellation, "
            f"{self.raw_pct:.1f}% without",
            "OK: the VAD stays quiet while Jason talks" if self.ok else
            "PROBLEM: the VAD hears Jason's own voice; barge-in would trigger on it",
        ]


def would_trigger(mic48: np.ndarray, model_file: Path, run: int = 3) -> bool:
    """Would the listener start speech here: `run` loud 32 ms slices in a row?"""
    from jason.body.listen import CHUNK, Silero

    vad = Silero(model_file)
    x = soxr.resample(mic48, IO_SR, MIC_SR).astype(np.float32)
    streak = 0
    for i in range(0, len(x) - CHUNK + 1, CHUNK):
        streak = streak + 1 if vad(x[i:i + CHUNK]) >= 0.5 else 0
        if streak >= run:
            return True
    return False


@dataclass
class OnsetResult:
    triggered: int  # starts where the VAD would have heard Jason as the user
    total: int
    delay_hint_ms: float
    measured_delay_ms: float

    @property
    def ok(self) -> bool:
        return self.triggered == 0

    def lines(self) -> list[str]:
        out = [f"Jason starting after a silence heard as you: {self.triggered} of {self.total} starts "
               f"(echo delay hint {self.delay_hint_ms:.0f} ms)",
               "OK: his first words don't trigger a barge-in" if self.ok else
               "PROBLEM: his first words leak and would cut him off (a false barge-in)"]
        if abs(self.delay_hint_ms - self.measured_delay_ms) > 20:
            out.append(f"Set echo_delay_ms = {self.measured_delay_ms:.0f} in config/body.toml [audio] "
                       "(the measured speaker -> mic delay)")
        return out


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


async def run_check(
    cfg: AudioConfig, out: Callable[[str], None] = print
) -> tuple[CheckResult, VoiceResult, OnsetResult]:
    signal = speechlike_signal()
    voice, voice_sr = read_wav(cfg.assets() / "voice_check.wav")
    voice = soxr.resample(voice, voice_sr, IO_SR).astype(np.float32)
    start_clip = voice[: int(ONSET_VOICE_S * IO_SR)]
    onsets = [np.concatenate([np.zeros(int(ONSET_GAP_S * IO_SR)), start_clip]) for _ in range(ONSETS)]
    timeline = np.concatenate([
        np.zeros(int(LEAD_S * IO_SR)), signal, np.zeros(int(TAIL_S * IO_SR)),
        np.zeros(int(VOICE_GAP_S * IO_SR)), voice, np.zeros(int(VOICE_TAIL_S * IO_SR)),
        *onsets, np.zeros(int(1.0 * IO_SR)),
    ]).astype(np.float32)
    out(f"Stay quiet: playing test sounds for {len(timeline) / IO_SR:.0f} s ...")
    io = AudioIO.from_config(cfg, keep_raw=True)
    io.start()
    try:
        io.play("check", timeline, IO_SR)
        await asyncio.sleep(len(timeline) / IO_SR + 0.3)
    finally:
        io.stop()
    raw, clean = np.concatenate(io.raw), np.concatenate(io.clean48)
    result = score(timeline, raw, clean, io.stats.status_flags, io.stats.blocks)
    v0 = int((LEAD_S + SIGNAL_S + TAIL_S + VOICE_GAP_S) * IO_SR)
    v1 = v0 + len(voice) + int(0.5 * IO_SR)  # and the half second after, where echo could linger
    model = cfg.assets() / "silero_vad.onnx"
    voice_result = VoiceResult(speech_percent(clean[v0:v1], model), speech_percent(raw[v0:v1], model))
    o = v1 - int(0.5 * IO_SR) + int(VOICE_TAIL_S * IO_SR)  # where the onset section begins
    triggered = 0
    for _ in range(ONSETS):
        start = o + int(ONSET_GAP_S * IO_SR)
        triggered += would_trigger(clean[start:start + int(ONSET_WINDOW_S * IO_SR)], model)
        o = start + len(start_clip)
    onset_result = OnsetResult(triggered, ONSETS, io.canceller.delay_hint_ms, result.delay_ms)
    for line in result.lines() + voice_result.lines() + onset_result.lines():
        out(line)
    return result, voice_result, onset_result
