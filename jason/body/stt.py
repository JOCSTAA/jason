"""The body's speech-to-text (M1 step 9; design §3.3 model 3).

    listener's utterance audio ─► every 0.5 s while speaking ─► user_partial (when the text changes)
                               └► at every pause ───────────► user_final + stt_final latency

The whole utterance so far is transcribed each time, rather than streamed:
Parakeet's streaming mode was slower than real time on the M5 Air, while a
whole utterance of 0.5-13.5 s takes 60-170 ms (docs/spikes/m1-mac-audio.md).
The final transcription starts as soon as the user pauses, alongside Smart
Turn, so it is mostly done by the time the floor controller decides.

A user_final at every pause: any pause may be the end of the turn, which is
only known afterwards. A later one for the same utterance supersedes it.

The model runs on one dedicated thread: transcription never blocks the audio
or the event loop, and MLX is only ever used from one place.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np

from jason.core import clock
from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import SttConfig

log = logging.getLogger(__name__)


class SpeechToText(Protocol):
    """A speech-to-text backend: 16 kHz audio in, text out."""

    def load(self) -> None: ...

    def transcribe(self, audio16k: np.ndarray) -> str: ...


class ParakeetMLX:
    """Parakeet TDT 0.6B on Apple Silicon (MLX). The first load downloads the
    pinned revision (2.5 GB) into the Hugging Face cache; later loads are offline."""

    def __init__(self, model: str, revision: str) -> None:
        self.model_id, self.revision = model, revision
        self._model = None

    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from parakeet_mlx import from_pretrained

        self._model = from_pretrained(snapshot_download(self.model_id, revision=self.revision))
        self.transcribe(np.zeros(16_000, dtype=np.float32))  # compile now, not on the user's first sentence

    def transcribe(self, audio16k: np.ndarray) -> str:
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        assert self._model is not None, "load() first"
        # float32: parakeet-mlx's own loader hands it float32 (bfloat16 audio breaks its STFT).
        mel = get_logmel(mx.array(audio16k, dtype=mx.float32), self._model.preprocessor_config)
        return self._model.generate(mel)[0].text.strip()


def make_backend(cfg: SttConfig) -> SpeechToText:
    if cfg.backend == "parakeet-mlx":
        return ParakeetMLX(cfg.model, cfg.revision)
    raise ValueError(f"unknown speech-to-text backend {cfg.backend!r}")


class Utterances(Protocol):
    """What the transcriber needs from the listener (jason/body/listen.py)."""

    utterance_id: str | None

    @property
    def speaking(self) -> bool: ...

    def utterance_audio(self) -> np.ndarray: ...


class Transcriber:
    def __init__(self, bus: EventBus, listener: Utterances, backend: SpeechToText, cfg: SttConfig) -> None:
        self.bus, self.listener, self.backend, self.cfg = bus, listener, backend, cfg
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._busy = False
        self._last_text: dict[str, str] = {}  # utterance id -> last transcript emitted
        self._finals: dict[str, int] = {}  # utterance id -> finals emitted (so stale partials are dropped)

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._pool, self.backend.load)

    async def run(self) -> None:
        partials = asyncio.create_task(self._partials())
        try:
            with self.bus.subscribe(ev.SpeechEnd, where=lambda e: e.machine == self.bus.machine) as ends:
                async for end in ends:
                    await self._final(end.payload.utterance_id)
        finally:
            partials.cancel()
            self._pool.shutdown(wait=False, cancel_futures=True)

    async def _transcribe(self, audio: np.ndarray) -> str:
        self._busy = True
        try:
            return await asyncio.get_running_loop().run_in_executor(self._pool, self.backend.transcribe, audio)
        finally:
            self._busy = False

    async def _final(self, utterance_id: str) -> None:
        audio = self.listener.utterance_audio()
        if not len(audio):
            return
        t0 = clock.monotonic_ns()
        text = await self._transcribe(audio)
        self._finals[utterance_id] = self._finals.get(utterance_id, 0) + 1
        if not text:
            return  # noise the VAD took for speech: nothing to answer
        self._last_text[utterance_id] = text
        final = self.bus.emit(ev.UserFinal, ev.UserFinalPayload(utterance_id=utterance_id, text=text))
        self.bus.emit(ev.LatencyMark, ev.LatencyMarkPayload(
            stage=ev.LatencyStage.STT_FINAL, duration_ns=clock.monotonic_ns() - t0, ref_id=final.id))

    async def _partials(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.partial_every_s)
            uid = self.listener.utterance_id
            # Skip while a transcription is running: a partial must never delay a final.
            if uid is None or not self.listener.speaking or self._busy:
                continue
            finals_before = self._finals.get(uid, 0)
            text = await self._transcribe(self.listener.utterance_audio())
            if self._finals.get(uid, 0) != finals_before:
                continue  # a final for this utterance landed meanwhile; this partial is stale
            if text and text != self._last_text.get(uid):
                self._last_text[uid] = text
                self.bus.emit(ev.UserPartial, ev.UserPartialPayload(utterance_id=uid, text=text))
