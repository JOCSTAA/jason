"""The body's mic and speaker, with echo cancellation (M1 step 7).

One duplex stream at 48 kHz in 10 ms blocks (docs/spikes/m1-mac-audio.md).
For every block, in PortAudio's audio thread:

    playback queue ─► speaker
         └──────────► echo canceller (reference: what is being played)
    mic ────────────► echo canceller ─► resample to 16 kHz ─► mic frames (asyncio)

Sound samples travel on this direct pipe and never on the event bus: 100
blocks a second of the user's voice would flood the raw log and cross to the
brain, and design §3.2 keeps audio on the body's machine. Everything that
means something (speech started, words heard, what was played) is still an
event. (Decision 2026-10-30.)

The playback queue also keeps a playhead: how many samples of each tagged
piece of audio have reached the speaker, and when the first one did. Barge-in
needs the first to log what the user heard; time to first audio needs the
second.
"""

import asyncio
import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import soxr

from jason.core import clock
from jason.core.config import AudioConfig

log = logging.getLogger(__name__)

IO_SR = 48_000  # the built-in mic and speakers run at 48 kHz
BLOCK = IO_SR // 100  # 10 ms: the echo canceller takes exactly 10 ms frames
MIC_SR = 16_000  # what the VAD, turn detector and speech-to-text take


# --- Echo cancellation ----------------------------------------------------------


class EchoCanceller:
    """WebRTC AEC3 through livekit: far() is what the speaker plays, near()
    cleans what the mic hears. Both take one 10 ms float block."""

    def __init__(self, delay_hint_ms: int) -> None:
        from livekit import rtc  # imported here: only the body needs it

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True, high_pass_filter=True)
        self.delay_hint_ms = delay_hint_ms  # AEC3 refines the delay itself; this is its starting point

    def far(self, block: np.ndarray) -> None:
        self._apm.process_reverse_stream(self._rtc.AudioFrame(_i16(block).tobytes(), IO_SR, 1, BLOCK))

    def near(self, block: np.ndarray) -> np.ndarray:
        frame = self._rtc.AudioFrame(bytearray(_i16(block).tobytes()), IO_SR, 1, BLOCK)
        self._apm.set_stream_delay_ms(self.delay_hint_ms)
        self._apm.process_stream(frame)
        return np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768


def _i16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


# --- Playback with a playhead ------------------------------------------------------


@dataclass
class Played:
    """How much of one tagged piece of audio has reached the speaker."""

    total: int  # samples queued (48 kHz)
    played: int = 0
    first_ns: int | None = None  # when its first sample reached the speaker (monotonic)
    stopped: bool = False  # flushed by stop() before it finished

    @property
    def finished(self) -> bool:
        return self.played >= self.total

    @property
    def played_ms(self) -> float:
        return self.played * 1000 / IO_SR


@dataclass(eq=False)  # identity, not value: pieces hold numpy arrays
class _Piece:
    tag: str
    samples: np.ndarray
    pos: int = 0


@dataclass
class Playback:
    """The speaker's queue. Written from asyncio, read from the audio thread,
    so every access takes the lock (held for microseconds)."""

    _queue: deque[_Piece] = field(default_factory=deque)
    _played: dict[str, Played] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # When the current stretch of Jason's speech began reaching the speaker
    # (after a silence); None while silent. The listener is stricter just
    # after it: that's where echo leaks (jason/body/listen.py).
    speaking_since_ns: int | None = None

    def add(self, tag: str, samples: np.ndarray) -> None:
        with self._lock:
            self._queue.append(_Piece(tag, samples))
            p = self._played.setdefault(tag, Played(total=0))
            p.total += len(samples)

    def next_block(self, n: int, dac_ns: int) -> np.ndarray:
        """The next n samples to play (silence once the queue is empty).
        dac_ns: when this block will reach the speaker."""
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            if self._queue and self.speaking_since_ns is None:
                self.speaking_since_ns = dac_ns
            while filled < n and self._queue:
                piece = self._queue[0]
                take = min(n - filled, len(piece.samples) - piece.pos)
                out[filled:filled + take] = piece.samples[piece.pos:piece.pos + take]
                p = self._played[piece.tag]
                if p.first_ns is None:
                    p.first_ns = dac_ns + filled * 1_000_000_000 // IO_SR
                p.played += take
                piece.pos += take
                filled += take
                if piece.pos >= len(piece.samples):
                    self._queue.popleft()
            if not self._queue:
                self.speaking_since_ns = None
        return out

    def stop(self, tag: str | None = None) -> None:
        """Drop queued audio at once (barge-in): everything, or one tag's."""
        with self._lock:
            keep: deque[_Piece] = deque()
            for piece in self._queue:
                if tag is None or piece.tag == tag:
                    self._played[piece.tag].stopped = True
                else:
                    keep.append(piece)
            self._queue = keep
            if not keep:
                self.speaking_since_ns = None

    def played(self, tag: str) -> Played | None:
        with self._lock:
            p = self._played.get(tag)
            return None if p is None else Played(p.total, p.played, p.first_ns, p.stopped)

    def forget(self, tag: str) -> None:
        with self._lock:
            self._played.pop(tag, None)

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._queue)


# --- The duplex stream -----------------------------------------------------------------


@dataclass(frozen=True)
class MicFrame:
    """Cleaned mic audio at 16 kHz. Sizes vary a little around 160 samples
    (the resampler's output per 10 ms block); consumers re-chunk as needed."""

    samples: np.ndarray
    audio_ns: int  # when its first sample was captured (monotonic)


@dataclass
class AudioStats:
    blocks: int = 0
    status_flags: int = 0  # PortAudio over/underflows: dropped or late audio
    mic_frames_dropped: int = 0  # consumers too slow


class AudioIO:
    """Owns the mic and speaker. start() inside a running event loop; read
    cleaned mic audio with frames(); play with play() / stop()."""

    def __init__(
        self,
        input_device: str = "",
        output_device: str = "",
        stream_factory: Callable[..., object] | None = None,
        canceller: Callable[[int], object] = EchoCanceller,
        mic_queue_size: int = 500,  # 5 s of audio
        keep_raw: bool = False,  # also keep the uncancelled mic (audio-check only)
        echo_delay_ms: float | None = None,  # measured speaker -> mic delay; None: PortAudio's figures
    ) -> None:
        self._echo_delay_ms = echo_delay_ms
        self.playback = Playback()
        self.stats = AudioStats()
        self._devices = (input_device or None, output_device or None)
        self._stream_factory = stream_factory
        self._make_canceller = canceller
        self._mic_queue: asyncio.Queue[MicFrame] = asyncio.Queue(maxsize=mic_queue_size)
        self._resampler = soxr.ResampleStream(IO_SR, MIC_SR, 1, dtype="float32")
        self._keep_raw = keep_raw
        self.raw: list[np.ndarray] = []  # 48 kHz blocks, only with keep_raw
        self.clean48: list[np.ndarray] = []  # 48 kHz blocks, only with keep_raw
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream = None

    @classmethod
    def from_config(cls, cfg: AudioConfig, **kw) -> "AudioIO":
        return cls(cfg.input_device, cfg.output_device, echo_delay_ms=cfg.echo_delay_ms, **kw)

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        if self._stream_factory is None:
            import sounddevice as sd  # imported here: needs PortAudio

            self._stream_factory = sd.Stream
        self._stream = self._stream_factory(
            samplerate=IO_SR, blocksize=BLOCK, dtype="float32", channels=(1, 2), device=self._devices,
            # Ask for small buffers; the "low" preset gave 0.2-0.4 s on the Mac (spike).
            latency=0.01, callback=self._callback,
        )
        lat_in, lat_out = self._stream.latency
        # The echo canceller's starting guess for the delay. PortAudio's figures
        # overstate it on the Mac (228 ms vs ~95 measured), and with a wrong
        # guess every reply's first word leaked into the mic after a silence.
        hint = self._echo_delay_ms if self._echo_delay_ms is not None else (lat_in + lat_out) * 1000
        self.canceller = self._make_canceller(int(round(hint)))
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def play(self, tag: str, samples: np.ndarray, sample_rate: int) -> None:
        """Queue audio (any rate; resampled to 48 kHz) under a tag, e.g. the reply's action id."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate != IO_SR:
            x = soxr.resample(x, sample_rate, IO_SR).astype(np.float32)
        self.playback.add(tag, x)

    async def frames(self):
        """Cleaned 16 kHz mic audio, as it arrives."""
        while True:
            yield await self._mic_queue.get()

    def _callback(self, indata, outdata, frames, time_info, status) -> None:
        # PortAudio's audio thread: keep it short, and never raise.
        now = clock.monotonic_ns()
        if status:
            self.stats.status_flags += 1
        self.stats.blocks += 1
        # When this block reaches the speaker, and when the mic block was
        # captured, from PortAudio's own timestamps (same clock as currentTime).
        dac_ns = now + int((time_info.outputBufferDacTime - time_info.currentTime) * 1e9)
        adc_ns = now - int((time_info.currentTime - time_info.inputBufferAdcTime) * 1e9)

        out = self.playback.next_block(frames, max(dac_ns, now))
        outdata[:] = out[:, None]  # the same signal to both speakers
        mic = indata[:, 0].astype(np.float32)
        if frames == BLOCK:
            self.canceller.far(out)
            clean = self.canceller.near(mic)
        else:  # never seen with blocksize fixed; pass through rather than crash
            clean = mic
        if self._keep_raw:
            self.raw.append(mic.copy())
            self.clean48.append(clean.copy())
        frame = MicFrame(self._resampler.resample_chunk(clean), min(adc_ns, now))
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._deliver, frame)

    def _deliver(self, frame: MicFrame) -> None:
        if self._mic_queue.full():  # a slow consumer loses the oldest audio, not the newest
            self._mic_queue.get_nowait()
            self.stats.mic_frames_dropped += 1
        self._mic_queue.put_nowait(frame)
