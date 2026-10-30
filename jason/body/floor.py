"""The floor controller: decides when Jason speaks (design §4, §9; M1 step 11).

Every tick (100 ms) it looks at the conversation and chooses between two of
the six actions: `silent` (keep listening) and `respond` (answer now). M5 adds
backchannel, interject and ask.

This is a hand-written rule ("rules-2"), not a learned policy. How long a
pause it waits for depends on how finished the user sounds (Smart Turn):

    end-of-turn probability ≥ 0.7   answer after 200 ms of silence
    0.3 – 0.7                       answer after 700 ms
    < 0.3 (mid-thought)             answer after 1.5 s

and talking again resets the clock. A transcript that ends on a word a
sentence can't end with ("what's a", "I went to the") counts as unsure
whatever Smart Turn says: Smart Turn hears how a phrase sounds, not its
words, and scored "What's a" 0.80 in the first live chat, so Jason answered
a fragment while the user was still talking (added in "rules-2").

It only answers with a transcript of everything said up to the pause, never
twice for one utterance, and never while Jason is speaking (barge-in is step
12).

Decisions are logged at decision points, not every tick: once when a pause
has its transcript and turn probability (`silent`: waiting), and when it
answers (`respond`). Each logs the inputs it saw as features and the
probability it gave the action; for a rule that's 1.0. In M4 a contextual
bandit replaces the rule behind the same events and features, which is why
they're logged now.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from jason.core import clock
from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import FloorConfig

log = logging.getLogger(__name__)

# Words an English sentence almost never ends on: articles, prepositions,
# conjunctions, possessives, auxiliaries and fillers. A pause after one is a
# pause mid-sentence.
UNFINISHED_ENDINGS = frozenset("""
    a an the to of in on at for with from by about into onto over under than as
    and or but so because if that which who whose when where while
    my your his her their our its this these those
    is are was were am be been being do does did have has had will would can could should
    um uh er erm like
""".split())


def ends_unfinished(text: str) -> bool:
    words = text.lower().replace("’", "'").split()
    return bool(words) and words[-1].strip(".,!?;:\"'") in UNFINISHED_ENDINGS


@dataclass
class _Turn:
    """What the controller knows about the user's current utterance."""

    utterance_id: str
    speaking: bool = True
    speech_ms: float = 0.0
    pause_start_ns: int | None = None  # when the last voiced sound ended (audio_ns)
    turn_prob: float | None = None  # Smart Turn's score for the current pause
    final_id: str | None = None  # the user_final for the current pause
    final_text: str = ""
    waiting_logged: bool = False  # a `silent` decision logged for this pause
    responded: bool = False


class FloorController:
    def __init__(
        self,
        bus: EventBus,
        cfg: FloorConfig,
        jason_speaking: Callable[[], bool],
        first_played_ns: Callable[[str], int | None] = lambda action_id: None,
    ) -> None:
        self.bus, self.cfg = bus, cfg
        self._jason_speaking = jason_speaking
        self._first_played_ns = first_played_ns  # when a reply's first sample reached the speaker
        self.turn: _Turn | None = None
        self._last_jason_ns: int | None = None
        self._watchers: dict[str, asyncio.Task[None]] = {}  # decision id -> its first-audio watcher

    async def run(self) -> None:
        ticker = asyncio.create_task(self._tick_forever())
        mine = lambda e: e.machine == self.bus.machine  # noqa: E731: only this body's hearing
        try:
            types = (ev.SpeechStart, ev.SpeechEnd, ev.TurnEndProb, ev.UserFinal, ev.ReplyCancel)
            with self.bus.subscribe(*types, where=mine) as events:
                async for e in events:
                    if isinstance(e, ev.ReplyCancel):
                        # Cancelled before its first sound (the user kept talking):
                        # there will be no first audio to wait for.
                        watcher = self._watchers.pop(e.payload.action_id, None)
                        if watcher:
                            watcher.cancel()
                    else:
                        self.observe(e)
        finally:
            ticker.cancel()
            for w in self._watchers.values():
                w.cancel()

    def observe(self, e: ev.Event) -> None:
        p = e.payload
        if isinstance(e, ev.SpeechStart):
            if self.turn is None or self.turn.utterance_id != p.utterance_id:
                self.turn = _Turn(p.utterance_id)
            # Talking again: a new pause will need its own transcript and score.
            t = self.turn
            t.speaking, t.pause_start_ns, t.turn_prob, t.final_id, t.waiting_logged = True, None, None, None, False
            return
        t = self.turn
        if t is None or p.utterance_id != t.utterance_id:
            return
        if isinstance(e, ev.SpeechEnd):
            t.speaking, t.pause_start_ns = False, p.audio_ns
            t.speech_ms += p.speech_ms
        elif isinstance(e, ev.TurnEndProb) and not t.speaking:
            t.turn_prob = p.prob
        elif isinstance(e, ev.UserFinal) and not t.speaking:
            t.final_id, t.final_text = e.id, p.text

    async def _tick_forever(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.tick_ms / 1000)
            self.tick()

    def wait_ms(self, prob: float) -> float:
        """How long a pause this rule waits for at a given end-of-turn probability."""
        if prob >= self.cfg.sure_prob:
            return self.cfg.wait_sure_ms
        if prob >= self.cfg.unsure_prob:
            return self.cfg.wait_unsure_ms
        return self.cfg.wait_unlikely_ms

    def tick(self) -> None:
        now = clock.monotonic_ns()
        if self._jason_speaking():
            self._last_jason_ns = now
            return
        t = self.turn
        if t is None or t.speaking or t.responded or t.final_id is None or t.turn_prob is None:
            return
        pause_ms = (now - t.pause_start_ns) / 1e6
        wait = self.wait_ms(t.turn_prob)
        if ends_unfinished(t.final_text):
            wait = max(wait, self.cfg.wait_unsure_ms)
        if pause_ms >= wait:
            self._decide(ev.Action.RESPOND, t, pause_ms, wait, now)
        elif not t.waiting_logged:
            t.waiting_logged = True
            self._decide(ev.Action.SILENT, t, pause_ms, wait, now)

    def _decide(self, action: ev.Action, t: _Turn, pause_ms: float, wait: float, now: int) -> None:
        features = {
            "pause_ms": round(pause_ms, 1),
            "turn_end_prob": round(t.turn_prob, 4),
            "speech_ms": round(t.speech_ms, 1),
            "words": float(len(t.final_text.split())),
            "ends_unfinished": float(ends_unfinished(t.final_text)),
            "wait_ms": wait,  # the rule's target for this pause
            "since_jason_ms": round((now - self._last_jason_ns) / 1e6, 1) if self._last_jason_ns else -1.0,
        }
        decision = self.bus.emit(ev.AgentAction, ev.AgentActionPayload(
            action=action, action_probs={action: 1.0}, utterance_id=t.utterance_id, based_on_id=t.final_id,
            policy_version=self.cfg.policy_version, features=features))
        if action is not ev.Action.RESPOND:
            return
        t.responded = True
        self.bus.emit(ev.LatencyMark, ev.LatencyMarkPayload(
            stage=ev.LatencyStage.TURN_DECISION, duration_ns=now - t.pause_start_ns, ref_id=decision.id))
        watcher = asyncio.get_running_loop().create_task(self._first_audio(decision.id, t.pause_start_ns))
        self._watchers[decision.id] = watcher
        watcher.add_done_callback(lambda _: self._watchers.pop(decision.id, None))

    async def _first_audio(self, action_id: str, pause_start_ns: int, timeout_s: float = 15.0) -> None:
        """Time to first audio (the M1 exit criterion): from the user's last
        voiced sound to the first sample of Jason's reply at the speaker."""
        deadline = clock.monotonic_ns() + int(timeout_s * 1e9)
        while clock.monotonic_ns() < deadline:
            first = self._first_played_ns(action_id)
            if first is not None:
                self.bus.emit(ev.LatencyMark, ev.LatencyMarkPayload(
                    stage=ev.LatencyStage.FIRST_AUDIO, duration_ns=max(0, first - pause_start_ns), ref_id=action_id))
                return
            await asyncio.sleep(0.01)
        log.warning("reply %s never started playing", action_id)
