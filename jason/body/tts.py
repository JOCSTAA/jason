"""The body's text-to-speech (M1 step 10; design §3.3 model 4).

    reply_chunk (from the brain) ─► synthesize, one clause at a time ─► AudioIO.play(tag = the reply's action id)
                                                                     └► tts_first_chunk latency
    the reply's last sample reaches the speaker ─► reply_heard
    the user starts talking over Jason ─► stop at once, reply_cancel, reply_heard (what was heard)

Clauses are spoken as they arrive, so Jason starts talking while the brain is
still writing the rest. Each one is synthesized on one dedicated thread, in
order. The audio is tagged with the reply's action id, so AudioIO's playhead
knows how much of which reply was heard (barge-in, step 12).

A short clause after the first ("Josh.") is held and spoken with the next
one: on its own it comes out as a choppy fragment. The first clause is never
held, because Jason's first sound would wait for it.

Kokoro pads every clause with ~0.4 s of silence before and ~0.6 s after, so
clause-by-clause speech had about a second of dead air between clauses (it
sounded read out word by word) and the first sound came 0.4 s late. Each
clause is trimmed to its sound, then given a pause that fits how it ends:
short after a comma, longer after a sentence.

Barge-in (step 12): when the user starts talking over Jason, playback stops
at once, the brain is told to stop generating, and reply_heard records what
the user heard. The playhead knows how much audio played, not which words
(Kokoro gives no word timings), so a clause cut partway is estimated by
time: 40% of its audio played means its first 40% of words, rounded down.
It may be off by a word; it never claims words that weren't reached.
Reacting on speech_start alone means an "mhmm", a laugh or a cough stops him
too; telling those apart is the floor controller's job later (M5).
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from jason.body.audio import IO_SR, AudioIO
from jason.core import clock
from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import BodyConfig, TtsConfig

log = logging.getLogger(__name__)

ESPEAK_MAX_PATH = 150  # espeak-ng cuts data paths at ~160 characters, then exits the process


class TextToSpeech(Protocol):
    sample_rate: int

    def load(self) -> None: ...

    def synthesize(self, text: str) -> np.ndarray: ...


class KokoroMLX:
    """Kokoro-82M on Apple Silicon (MLX), with misaki's English phonemizer.
    The first load downloads the pinned revision (330 MB) into the Hugging
    Face cache; later loads are offline."""

    sample_rate = 24_000

    def __init__(self, model: str, revision: str, voice: str, speed: float) -> None:
        self.model_id, self.revision, self.voice, self.speed = model, revision, voice, speed
        self._model = None

    def load(self) -> None:
        import espeakng_loader
        from huggingface_hub import snapshot_download
        from mlx_audio.tts.utils import load_model

        data = espeakng_loader.get_data_path()
        if len(data) > ESPEAK_MAX_PATH:
            raise RuntimeError(
                f"espeak's data folder path is {len(data)} characters ({data}); espeak-ng truncates paths "
                f"longer than about 160 and then crashes. Keep the project's .venv on a shorter path."
            )
        self._model = load_model(Path(snapshot_download(self.model_id, revision=self.revision)))
        self.sample_rate = self._model.sample_rate
        self.synthesize("Ready.")  # load the phonemizer and compile now, not on Jason's first words

    def synthesize(self, text: str) -> np.ndarray:
        assert self._model is not None, "load() first"
        # The voice's first letter is its language: "a" American, "b" British
        # English; misaki's pronunciation rules must match it.
        parts = [np.asarray(r.audio, dtype=np.float32).reshape(-1)
                 for r in self._model.generate(text=text, voice=self.voice, speed=self.speed, lang_code=self.voice[0])]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def trim(samples: np.ndarray, sample_rate: int, text: str, cfg: TtsConfig) -> np.ndarray:
    """Cut the clause's padding; keep a breath before it and a pause after it
    that fits its punctuation."""
    if not len(samples):
        return samples
    level = np.abs(samples)
    loud = np.flatnonzero(level > max(level.max() * 0.01, 1e-4))  # 40 dB below the clause's peak: soft "s" and "h" onsets survive
    if not len(loud):
        return samples[:0]
    end = text.rstrip()[-1:] if text.strip() else ""
    pause_ms = cfg.pause_sentence_ms if end in ".!?" else cfg.pause_comma_ms if end in ",;:" else cfg.pause_other_ms
    lead = int(cfg.lead_ms / 1000 * sample_rate)
    start = max(0, loud[0] - lead)
    stop = loud[-1] + 1 + int(0.03 * sample_rate)  # let the last sound decay for 30 ms
    gap = np.zeros(int(pause_ms / 1000 * sample_rate), dtype=np.float32)
    return np.concatenate([samples[start:stop], gap])


def heard_text(clauses: list[tuple[str, int]], played: int) -> str:
    """What was heard of clauses [(text, samples at 48 kHz)] when `played`
    samples had reached the speaker: whole clauses, then the share of the
    cut clause's words that its played share of audio covers."""
    heard: list[str] = []
    for text, n in clauses:
        if played >= n:
            heard.append(text)
            played -= n
            continue
        words = text.split()
        k = int(len(words) * played / n) if n else 0
        if k:
            heard.append(" ".join(words[:k]))
        break
    return " ".join(heard)


def make_backend(cfg: TtsConfig) -> TextToSpeech:
    if cfg.backend == "kokoro-mlx":
        return KokoroMLX(cfg.model, cfg.revision, cfg.voice, cfg.speed)
    raise ValueError(f"unknown text-to-speech backend {cfg.backend!r}")


@dataclass
class _Reply:
    action_id: str
    spoken: list[tuple[str, int]] = field(default_factory=list)  # (clause, samples at 48 kHz) queued, in order
    held: str = ""  # a short clause waiting for the next one
    first_synthesized: bool = False
    cut: bool = False  # the user barged in: say nothing more of this reply
    last_received: bool = False  # the brain has sent its final clause


class Speaker:
    """Turns the brain's reply clauses into sound, one reply at a time."""

    def __init__(self, bus: EventBus, audio: AudioIO, backend: TextToSpeech, cfg: TtsConfig) -> None:
        self.bus, self.audio, self.backend, self.cfg = bus, audio, backend, cfg
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self._queue: asyncio.Queue[tuple[ev.ReplyChunk, int]] = asyncio.Queue()
        self._reply: _Reply | None = None
        self._pending: str | None = None  # the latest respond decision, before any of its audio exists
        self._cut: set[str] = set()  # replies interrupted before their first clause arrived
        self._watchers: set[asyncio.Task[None]] = set()

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._pool, self.backend.load)

    async def run(self) -> None:
        worker = asyncio.create_task(self._work())
        mine = lambda e: isinstance(e, ev.ReplyChunk) or e.machine == self.bus.machine  # noqa: E731
        try:
            with self.bus.subscribe(ev.ReplyChunk, ev.SpeechStart, ev.AgentAction, where=mine) as events:
                async for e in events:
                    if isinstance(e, ev.SpeechStart):
                        self.barge_in(e)  # at once, not behind clauses waiting to be synthesized
                    elif isinstance(e, ev.AgentAction):
                        if e.payload.action is ev.Action.RESPOND:
                            self._pending = e.id
                    elif e.payload.action_id not in self._cut:
                        self._queue.put_nowait((e, clock.monotonic_ns()))
        finally:
            worker.cancel()
            for w in self._watchers:
                w.cancel()
            self._pool.shutdown(wait=False, cancel_futures=True)

    async def _work(self) -> None:
        while True:
            chunk, arrived_ns = await self._queue.get()
            await self._speak(chunk.payload, arrived_ns)

    async def _speak(self, p: ev.ReplyChunkPayload, arrived_ns: int) -> None:
        if self._reply is None or self._reply.action_id != p.action_id:
            if p.action_id == self._pending:
                self._pending = None  # its audio is on the way now
            if self._reply is not None:
                # A newer reply replaces one still playing (the brain already
                # stopped generating the old one).
                self.audio.playback.stop(self._reply.action_id)
            self._reply = _Reply(p.action_id)
        reply = self._reply
        if reply.cut:
            return  # the user interrupted: the rest of this reply goes unsaid
        reply.last_received = reply.last_received or p.last

        text = f"{reply.held} {p.text}".strip() if reply.held else p.text.strip()
        reply.held = ""
        short = len(text.split()) < self.cfg.min_words
        if text and short and p.index > 0 and not p.last:
            reply.held = text  # spoken together with the next clause
            text = ""
        if text:
            samples = await asyncio.get_running_loop().run_in_executor(self._pool, self.backend.synthesize, text)
            samples = trim(samples, self.backend.sample_rate, text, self.cfg)
            if self._reply is not reply or reply.cut:
                return  # replaced, or interrupted, while synthesizing
            if not reply.first_synthesized:
                reply.first_synthesized = True
                self.bus.emit(ev.LatencyMark, ev.LatencyMarkPayload(
                    stage=ev.LatencyStage.TTS_FIRST_CHUNK, duration_ns=clock.monotonic_ns() - arrived_ns,
                    ref_id=p.action_id))
            self.audio.play(p.action_id, samples, self.backend.sample_rate)
            reply.spoken.append((text, round(len(samples) * IO_SR / self.backend.sample_rate)))
        if p.last:
            task = asyncio.create_task(self._report_heard(reply))
            self._watchers.add(task)
            task.add_done_callback(self._watchers.discard)

    async def _report_heard(self, reply: _Reply) -> None:
        """Once the reply has finished playing (or was stopped), say what was heard."""
        while True:
            played = self.audio.playback.played(reply.action_id)
            if played is None or played.finished or played.stopped:
                break
            await asyncio.sleep(0.02)
        if played is None or played.stopped:
            return  # cut off: barge_in() reported what was heard
        self.bus.emit(ev.ReplyHeard, ev.ReplyHeardPayload(
            action_id=reply.action_id, text=" ".join(t for t, _ in reply.spoken), complete=True,
            played_ms=played.played_ms))
        self.audio.playback.forget(reply.action_id)

    def barge_in(self, start: ev.SpeechStart) -> None:
        """The user started talking: if Jason is answering, stop him now.

        "Answering" covers a reply that is playing, one whose next clause is
        still on its way (he got ahead of the brain), and one decided but not
        yet audible (the brain is still writing the first clause)."""
        reply = self._reply
        if reply is not None and reply.action_id == self._pending:
            self._pending = None
        if reply is not None and not reply.cut:
            played = self.audio.playback.played(reply.action_id)
            finished = reply.last_received and (played is None or played.finished)
            if not finished:
                self._cut_reply(reply.action_id, reply.spoken, start)
                reply.cut = True
                return
        if self._pending is not None:  # decided, nothing said yet
            self._cut.add(self._pending)
            self._cut_reply(self._pending, [], start)
            self._pending = None

    def _cut_reply(self, action_id: str, spoken: list[tuple[str, int]], start: ev.SpeechStart) -> None:
        self.audio.playback.stop(action_id)
        stopped_ns = clock.monotonic_ns()
        self.bus.emit(ev.ReplyCancel, ev.ReplyCancelPayload(action_id=action_id, reason="barge_in"))
        # The playhead after stopping: everything that reached the speaker.
        played = self.audio.playback.played(action_id)
        n = played.played if played else 0
        self.bus.emit(ev.ReplyHeard, ev.ReplyHeardPayload(
            action_id=action_id, text=heard_text(spoken, n), complete=False,
            played_ms=played.played_ms if played else 0.0, interrupted_by=start.payload.utterance_id))
        self.bus.emit(ev.LatencyMark, ev.LatencyMarkPayload(
            stage=ev.LatencyStage.BARGE_IN, duration_ns=max(0, stopped_ns - start.payload.audio_ns),
            ref_id=action_id))
        self.audio.playback.forget(action_id)


@contextlib.asynccontextmanager
async def speak_replies(bus: EventBus, cfg: BodyConfig, out: Callable[[str], None] = print):
    """Say the brain's replies out loud while the block runs (text mode with --speak)."""
    audio = AudioIO(cfg.audio.input_device, cfg.audio.output_device)
    speaker = Speaker(bus, audio, make_backend(cfg.tts), cfg.tts)
    out("Loading text-to-speech (the first time downloads the model, 330 MB) ...")
    await speaker.load()
    audio.start()

    async def drain_mic() -> None:
        # Nobody listens in text mode; read the mic so its queue never fills.
        async for _ in audio.frames():
            pass

    tasks = [asyncio.create_task(speaker.run()), asyncio.create_task(drain_mic())]
    try:
        yield speaker
        while audio.playback.busy:  # let the last reply finish
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)
    finally:
        for t in tasks:
            t.cancel()
        audio.stop()
