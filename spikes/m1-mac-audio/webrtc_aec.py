"""Setup B: WebRTC echo cancellation (AEC3, via the livekit package) with
sounddevice for the mic and speakers.

One duplex PortAudio stream at 48 kHz in 10 ms blocks (AEC3 takes exactly
10 ms frames). In each block, the audio about to be played goes in as the far
end (the echo reference), and the mic block comes out with the echo removed.
The raw mic is kept too: it is setup C's input (no echo cancellation) and the
baseline for how much echo B removed.
"""

import threading

import numpy as np
import sounddevice as sd
from livekit import rtc

from common import IO_SR

BLOCK = IO_SR // 100  # 10 ms


def _to_i16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


class Canceller:
    def __init__(self, delay_ms: int):
        self.apm = rtc.AudioProcessingModule(echo_cancellation=True, high_pass_filter=True)
        self.delay_ms = delay_ms

    def far(self, block: np.ndarray) -> None:
        self.apm.process_reverse_stream(rtc.AudioFrame(_to_i16(block).tobytes(), IO_SR, 1, BLOCK))

    def near(self, block: np.ndarray) -> np.ndarray:
        frame = rtc.AudioFrame(bytearray(_to_i16(block).tobytes()), IO_SR, 1, BLOCK)
        self.apm.set_stream_delay_ms(self.delay_ms)
        self.apm.process_stream(frame)
        return np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768


class DuplexAEC:
    """Plays `playback` (48 kHz mono) once from the start while recording."""

    def __init__(self, playback: np.ndarray):
        self.playback = playback.astype(np.float32)
        self.pos = 0
        self.raw: list[np.ndarray] = []
        self.clean: list[np.ndarray] = []
        self.overflows = 0
        self.done = threading.Event()
        self.stream = sd.Stream(
            samplerate=IO_SR, blocksize=BLOCK, dtype="float32", channels=(1, 2),
            # 10 ms asks PortAudio for small buffers; "low" gave 0.2-0.4 s on this Mac.
            latency=0.01, callback=self._callback,
        )
        lat_in, lat_out = self.stream.latency
        # AEC3 also estimates the delay itself; this is its starting hint.
        self.delay_ms = int(round((lat_in + lat_out) * 1000))
        self.aec = Canceller(self.delay_ms)

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.overflows += 1
        out = self.playback[self.pos : self.pos + frames]
        out = np.pad(out, (0, frames - len(out)))
        self.pos += frames
        outdata[:] = out[:, None]  # same signal to both speakers
        mic = indata[:, 0].copy()
        self.aec.far(out)
        self.raw.append(mic)
        self.clean.append(self.aec.near(mic))
        if self.pos >= len(self.playback):
            self.done.set()

    def run(self) -> dict:
        with self.stream:
            self.done.wait()
        return {
            "raw": np.concatenate(self.raw), "clean": np.concatenate(self.clean),
            "sr": IO_SR, "delay_hint_ms": self.delay_ms, "status_flags": self.overflows,
        }


def self_test() -> None:
    """Offline check with a fake room: the mic hears the far end 40 ms late,
    20 dB quieter, plus noise. Reports how much AEC3 removes once settled."""
    from common import dbfs, kokoro_say, load_kokoro, resample

    k = load_kokoro()
    far = resample(*kokoro_say(k, "This is Jason talking, to check that the echo canceller removes my own voice from the microphone signal."), IO_SR)
    far = np.concatenate([far, far])  # second pass: after the filter has converged
    rng = np.random.default_rng(0)
    d = int(0.040 * IO_SR)
    mic = np.concatenate([np.zeros(d), far[:-d]]) * 0.1 + rng.normal(0, 1e-3, len(far))
    aec = Canceller(delay_ms=40)
    out = []
    for i in range(0, len(far) - BLOCK + 1, BLOCK):
        aec.far(far[i : i + BLOCK])
        out.append(aec.near(mic[i : i + BLOCK].astype(np.float32)))
    out = np.concatenate(out)
    half = len(out) // 2
    print(f"echo in mic {dbfs(mic[half:]):.1f} dBFS -> after AEC {dbfs(out[half:]):.1f} dBFS "
          f"(reduction {dbfs(mic[half:]) - dbfs(out[half:]):.1f} dB, settled half)")


if __name__ == "__main__":
    self_test()
