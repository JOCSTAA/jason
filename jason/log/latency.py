"""`jason latency`: where each spoken turn's time went (M1 step 13).

Reads one session's raw log. The body's log is the one to read: it holds the
body's events and, through the link, the brain's, so one file has every
stage of every turn. Each turn is one `respond` decision; its latency marks
are matched by the ids they point at:

    decide   your last sound → the floor controller answered   (turn_decision, ref = the decision)
    stt      the final transcription, inside `decide`           (stt_final, ref = the transcript)
    llm 1st  request → first token                              (llm_first_token, ref = the decision)
    clause   request → first clause ready                       (first_clause, ref = the decision)
    tts      clause arrived → its audio ready                   (tts_first_chunk, ref = the decision)
    first    your last sound → Jason's first sound              (first_audio, ref = the decision)
    rest     first − (decide + clause + tts): the network, queues and audio buffers

The exit check is M1's: at least 20 answered turns, median first audio under
1 s. A session is a whole day, so by default only the latest run of the body
(since its last startup) is analysed: one conversation.
"""

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field

from jason.core import events as ev

S = ev.LatencyStage


def percentile(values: list[float], p: float) -> float:
    """Nearest rank: the smallest value with at least p% of values at or below it."""
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100 * len(ordered))) - 1]


@dataclass
class Turn:
    said: str
    decide: float | None = None
    stt: float | None = None
    llm_first: float | None = None
    clause: float | None = None
    tts: float | None = None
    first_audio: float | None = None

    @property
    def rest(self) -> float | None:
        parts = (self.first_audio, self.decide, self.clause, self.tts)
        if any(v is None for v in parts):
            return None
        return self.first_audio - self.decide - self.clause - self.tts


@dataclass
class BargeIn:
    stopped_ms: float
    heard: str


@dataclass
class Report:
    turns: list[Turn] = field(default_factory=list)
    barge_ins: list[BargeIn] = field(default_factory=list)
    network_rtt: list[float] = field(default_factory=list)

    @property
    def spoken(self) -> list[Turn]:
        return [t for t in self.turns if t.first_audio is not None]

    def passes(self, min_turns: int = 20, target_ms: float = 1000.0) -> bool:
        firsts = [t.first_audio for t in self.spoken]
        return len(firsts) >= min_turns and statistics.median(firsts) < target_ms

    def lines(self, min_turns: int = 20, target_ms: float = 1000.0) -> list[str]:
        fmt = lambda v: "     -" if v is None else f"{v:6.0f}"  # noqa: E731
        out = [f"{'you said':32} {'decide':>6} {'stt':>6} {'llm 1st':>7} {'clause':>6} {'tts':>6} {'FIRST':>6} {'rest':>6}  (ms)"]
        for t in self.turns:
            said = (t.said[:29] + "...") if len(t.said) > 32 else t.said
            out.append(f"{said:32} {fmt(t.decide)} {fmt(t.stt)} {fmt(t.llm_first):>7} {fmt(t.clause)} "
                       f"{fmt(t.tts)} {fmt(t.first_audio)} {fmt(t.rest)}")
        for b in self.barge_ins:
            out.append(f"  barge-in: Jason stopped {b.stopped_ms:.0f} ms after you started (heard: {b.heard[-40:]!r})")
        out.append("")
        stages = [("decide", "decide"), ("stt", "stt"), ("llm 1st", "llm_first"), ("clause", "clause"),
                  ("tts", "tts"), ("first audio", "first_audio"), ("rest", "rest")]
        for name, attr in stages:
            vals = [v for t in self.turns if (v := getattr(t, attr)) is not None]
            if vals:
                out.append(f"{name:12} p50 {percentile(vals, 50):6.0f}  p99 {percentile(vals, 99):6.0f}  n={len(vals)}")
        if self.barge_ins:
            stops = [b.stopped_ms for b in self.barge_ins]
            out.append(f"{'barge-in':12} p50 {percentile(stops, 50):6.0f}  p99 {percentile(stops, 99):6.0f}  n={len(stops)}")
        if self.network_rtt:
            out.append(f"{'network rtt':12} p50 {percentile(self.network_rtt, 50):6.1f}  "
                       f"p99 {percentile(self.network_rtt, 99):6.1f}  n={len(self.network_rtt)}")
        out.append("")
        firsts = [t.first_audio for t in self.spoken]
        if firsts:
            median = statistics.median(firsts)
            verdict = "PASS" if self.passes(min_turns, target_ms) else "NOT YET"
            out.append(f"M1 exit check: {len(firsts)} spoken turns (need {min_turns}), median first audio "
                       f"{median:.0f} ms (need < {target_ms:.0f}): {verdict}")
        else:
            out.append("M1 exit check: no spoken turns in this session yet")
        return out


def last_run(events: Iterable[ev.Event], machine: str) -> list[ev.Event]:
    """The events since `machine` last started: one conversation, not the whole day."""
    events = list(events)
    starts = [i for i, e in enumerate(events)
              if isinstance(e, ev.System) and e.machine == machine and e.payload.kind == "startup"]
    return events[starts[-1]:] if starts else events


def analyze(events: Iterable[ev.Event]) -> Report:
    events = list(events)
    finals = {e.id: e.payload.text for e in events if isinstance(e, ev.UserFinal)}
    marks: dict[tuple[str, S], float] = {}
    report = Report()
    for e in events:
        if isinstance(e, ev.LatencyMark):
            ms = e.payload.duration_ns / 1e6
            if e.payload.stage is S.NETWORK_RTT:
                report.network_rtt.append(ms)
            elif e.payload.ref_id:
                marks[(e.payload.ref_id, e.payload.stage)] = ms
    for e in events:
        if isinstance(e, ev.AgentAction) and e.payload.action is ev.Action.RESPOND:
            p = e.payload
            report.turns.append(Turn(
                said=finals.get(p.based_on_id or "", "?"),
                decide=marks.get((e.id, S.TURN_DECISION)),
                stt=marks.get((p.based_on_id or "", S.STT_FINAL)),
                llm_first=marks.get((e.id, S.LLM_FIRST_TOKEN)),
                clause=marks.get((e.id, S.FIRST_CLAUSE)),
                tts=marks.get((e.id, S.TTS_FIRST_CHUNK)),
                first_audio=marks.get((e.id, S.FIRST_AUDIO)),
            ))
        elif isinstance(e, ev.ReplyHeard) and e.payload.interrupted_by:
            stopped = marks.get((e.payload.action_id, S.BARGE_IN))
            if stopped is not None:
                report.barge_ins.append(BargeIn(stopped, e.payload.text))
    return report
