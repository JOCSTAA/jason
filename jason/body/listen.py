"""The body's listener: when is the user speaking, and have they finished?

    AudioIO mic frames (16 kHz) ─► Silero VAD, every 32 ms ─► speech_start / speech_end
                                        └─ the utterance's audio ─► Smart Turn ─► turn_end_prob

Silero (design §3.3 model 1) gives the probability that someone is speaking
in each 32 ms slice. A small hysteresis turns that into events:

- speech starts after `start_chunks` slices in a row at or above
  `vad_threshold` (3 slices, about 96 ms): one loud slice from a cough or a
  click doesn't count, at the cost of ~64 ms of barge-in reaction.
- a pause is `pause_ms` below `vad_off_threshold` (a lower bar, so speech
  that dips briefly doesn't end).

One utterance is one turn of the user's: a pause doesn't end it, so "so I
was thinking... [pause] ...we could" stays one utterance with one id, and
Smart Turn and speech-to-text see all of it. It ends when Jason acts on it
(an agent_action naming it that speaks: not `silent`) or after
`new_utterance_after_s` of silence.

At every pause, Smart Turn (design §3.3 model 2) scores the utterance so far.
The floor controller (step 11) weighs that against how long the pause lasts.

Event timestamps are when the sound happened (`audio_ns`, the capture time of
the slice), not when the event was created: time to first audio is measured
from the end of the user's last voiced slice.
"""

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from jason.body.audio import MIC_SR, AudioIO, MicFrame
from jason.body.turn import SmartTurn
from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import AudioConfig, SttConfig

log = logging.getLogger(__name__)

CHUNK = 512  # Silero's slice at 16 kHz: 32 ms
CHUNK_NS = CHUNK * 1_000_000_000 // MIC_SR


class Silero:
    """Silero VAD v6 on onnxruntime alone (the silero-vad package needs torch).

    Each 512-sample slice goes in with the previous slice's last 64 samples in
    front of it, and the recurrent state carries over, as in Silero's own
    wrapper."""

    CONTEXT = 64

    def __init__(self, model_file: Path) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(model_file), sess_options=so, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.CONTEXT, dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        x = np.concatenate([self._context, chunk.astype(np.float32)])[None, :]
        out, self._state = self._sess.run(
            None, {"input": x, "state": self._state, "sr": np.array(MIC_SR, dtype=np.int64)}
        )
        self._context = x[0, -self.CONTEXT:]
        return float(out[0, 0])


@dataclass
class Transition:
    kind: str  # "start" or "end"
    audio_ns: int  # start: first voiced slice began; end: last voiced slice ended
    speech_ms: float = 0.0  # end only: length of the stretch of speech


class VoiceActivity:
    """Speech/silence hysteresis over Silero's per-slice probabilities."""

    def __init__(self, cfg: AudioConfig) -> None:
        self.on, self.off = cfg.vad_threshold, cfg.vad_off_threshold
        self.start_chunks = cfg.start_chunks
        self.pause_chunks = max(1, round(cfg.pause_ms / 32))
        self.speaking = False
        self._run_start: int | None = None  # first slice of the current run of loud slices
        self._run = 0
        self._quiet = 0
        self._speech_start = 0
        self._last_voiced_end = 0

    def update(self, prob: float, chunk_ns: int) -> Transition | None:
        if not self.speaking:
            if prob >= self.on:
                if self._run == 0:
                    self._run_start = chunk_ns
                self._run += 1
                if self._run >= self.start_chunks:
                    self.speaking, self._quiet = True, 0
                    self._speech_start = self._run_start
                    self._last_voiced_end = chunk_ns + CHUNK_NS
                    return Transition("start", self._speech_start)
            else:
                self._run = 0
            return None
        if prob >= self.off:
            self._quiet = 0
            self._last_voiced_end = chunk_ns + CHUNK_NS
            return None
        self._quiet += 1
        if self._quiet >= self.pause_chunks:
            self.speaking, self._run = False, 0
            ms = (self._last_voiced_end - self._speech_start) / 1e6
            return Transition("end", self._last_voiced_end, ms)
        return None


class Listener:
    """Reads AudioIO's mic frames; emits speech_start, speech_end and
    turn_end_prob. Keeps the current utterance's audio for speech-to-text."""

    def __init__(
        self,
        bus: EventBus,
        audio: AudioIO,
        cfg: AudioConfig,
        vad: Callable[[np.ndarray], float] | None = None,
        turn: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        self.bus, self.audio, self.cfg = bus, audio, cfg
        assets = cfg.assets()
        self.vad = vad or Silero(assets / "silero_vad.onnx")
        self.turn = turn or SmartTurn(assets / "smart-turn-v3.2-cpu.onnx")
        self.activity = VoiceActivity(cfg)
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_ns = 0  # capture time of _pending[0]
        preroll = int(cfg.preroll_ms / 1000 * MIC_SR / CHUNK) + 1
        self._recent: deque[np.ndarray] = deque(maxlen=preroll)  # slices before speech, for the utterance
        self._utterance: list[np.ndarray] | None = None
        self.utterance_id: str | None = None
        self._last_end_ns: int | None = None
        self._reopen = False  # Jason acted on the utterance while the user was still talking
        self._turn_tasks: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        actions = asyncio.create_task(self._watch_actions())
        try:
            async for frame in self.audio.frames():
                self.process(frame)
        finally:
            actions.cancel()

    async def _watch_actions(self) -> None:
        # Jason speaking in answer to the utterance closes it: what the user
        # says next is a new turn. Deciding to keep listening (silent) doesn't.
        with self.bus.subscribe(ev.AgentAction) as actions:
            async for action in actions:
                speaks = action.payload.action not in (ev.Action.SILENT, ev.Action.THINK)
                if speaks and action.payload.utterance_id and action.payload.utterance_id == self.utterance_id:
                    self._close_utterance()

    @property
    def speaking(self) -> bool:
        return self.activity.speaking

    def utterance_audio(self) -> np.ndarray:
        """The current utterance so far (16 kHz), with a little audio from just
        before it began. Empty when there's no open utterance."""
        if self._utterance is None:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._utterance) if self._utterance else np.zeros(0, dtype=np.float32)

    def process(self, frame: MicFrame) -> None:
        """Split a mic frame into 32 ms slices and score each."""
        if not len(self._pending):
            self._pending_ns = frame.audio_ns
        self._pending = np.concatenate([self._pending, frame.samples])
        while len(self._pending) >= CHUNK:
            chunk, self._pending = self._pending[:CHUNK], self._pending[CHUNK:]
            chunk_ns = self._pending_ns
            self._pending_ns += CHUNK_NS
            self._slice(chunk, chunk_ns)

    def _slice(self, chunk: np.ndarray, chunk_ns: int) -> None:
        prob = self.vad(chunk)
        if self._reopen:
            # The user kept talking after Jason acted on their utterance: what
            # follows is a new turn (and, once he speaks, a barge-in).
            self._reopen = False
            self._open_utterance()
            self.bus.emit(ev.SpeechStart, ev.SpeechStartPayload(
                utterance_id=self.utterance_id, audio_ns=chunk_ns, during_agent_speech=self.audio.playback.busy))
        if self._utterance is not None:
            self._utterance.append(chunk)
        else:
            self._recent.append(chunk)
        t = self.activity.update(prob, chunk_ns)
        if t is None:
            self._maybe_expire(chunk_ns)
        elif t.kind == "start":
            if self._utterance is None:
                self._open_utterance()
            self.bus.emit(ev.SpeechStart, ev.SpeechStartPayload(
                utterance_id=self.utterance_id, audio_ns=t.audio_ns, during_agent_speech=self.audio.playback.busy))
        else:
            self._last_end_ns = t.audio_ns
            self.bus.emit(ev.SpeechEnd, ev.SpeechEndPayload(
                utterance_id=self.utterance_id, audio_ns=t.audio_ns, speech_ms=t.speech_ms))
            self._score_turn(self.utterance_id, self.utterance_audio())

    def _score_turn(self, utterance_id: str, audio: np.ndarray) -> None:
        async def score() -> None:
            # In a thread: 30+ ms of CPU shouldn't hold up the next mic frames.
            prob = await asyncio.to_thread(self.turn, audio)
            self.bus.emit(ev.TurnEndProb, ev.TurnEndProbPayload(utterance_id=utterance_id, prob=min(max(prob, 0.0), 1.0)))

        task = asyncio.get_running_loop().create_task(score())
        self._turn_tasks.add(task)  # keep a reference until it's done
        task.add_done_callback(self._turn_tasks.discard)

    def _maybe_expire(self, now_ns: int) -> None:
        if self.activity.speaking or self._utterance is None or self._last_end_ns is None:
            return
        if now_ns - self._last_end_ns > self.cfg.new_utterance_after_s * 1e9:
            self._close_utterance()

    def _open_utterance(self) -> None:
        self.utterance_id = uuid4().hex
        self._utterance = list(self._recent)  # the moment just before speech began
        self._recent.clear()

    def _close_utterance(self) -> None:
        self._utterance = None
        self.utterance_id = None
        self._last_end_ns = None
        if self.activity.speaking:
            self._reopen = True

async def show_listening(
    bus: EventBus, cfg: AudioConfig, stt: SttConfig | None = None, out: Callable[[str], None] = print
) -> None:
    """`jason run --role body --listen`: print what the listener hears, live,
    and (with an [stt] config) what speech-to-text makes of it."""
    from jason.body.stt import Transcriber, make_backend

    audio = AudioIO(cfg.input_device, cfg.output_device)
    listener = Listener(bus, audio, cfg)
    tasks = []
    if stt is not None:
        out("Loading speech-to-text (the first time downloads the model, 2.5 GB) ...")
        transcriber = Transcriber(bus, listener, make_backend(stt), stt)
        await transcriber.load()
        tasks.append(asyncio.create_task(transcriber.run()))
    audio.start()
    tasks.append(asyncio.create_task(listener.run()))
    out("Listening; talk, pause, trail off. Ctrl+C to stop.")
    starts: dict[str, int] = {}
    try:
        with bus.subscribe(ev.SpeechStart, ev.SpeechEnd, ev.TurnEndProb, ev.UserPartial, ev.UserFinal) as events:
            async for e in events:
                u = e.payload.utterance_id[:6]
                if isinstance(e, ev.SpeechStart):
                    starts[e.payload.utterance_id] = e.payload.audio_ns
                    out(f"[{u}] speech start{' (while Jason talks)' if e.payload.during_agent_speech else ''}")
                elif isinstance(e, ev.SpeechEnd):
                    out(f"[{u}] pause after {e.payload.speech_ms:.0f} ms of speech")
                elif isinstance(e, ev.TurnEndProb):
                    verdict = "finished" if e.payload.prob >= 0.5 else "not finished"
                    out(f"[{u}] end of turn? {e.payload.prob:.2f} ({verdict})")
                elif isinstance(e, ev.UserPartial):
                    out(f"[{u}]   ... {e.payload.text}")
                else:
                    out(f"[{u}] heard: \"{e.payload.text}\"")
    finally:
        for task in tasks:
            task.cancel()
        audio.stop()
