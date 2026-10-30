"""jason latency: turns matched to their marks, percentiles, the exit check."""

import pytest

from jason.core import events as ev
from jason.log.latency import analyze, last_run, percentile

SESSION = "day-0001_2026-10-30"
S = ev.LatencyStage
MS = 1_000_000


def body(cls, payload):
    return cls(machine="mac", session_id=SESSION, payload=payload)


def brain(cls, payload):
    return cls(machine="brain", session_id=SESSION, payload=payload)


def mark(stage, ms, ref, maker=body):
    return maker(ev.LatencyMark, ev.LatencyMarkPayload(stage=stage, duration_ns=int(ms * MS), ref_id=ref))


def turn(text, decide, stt, llm, clause, tts, first):
    final = body(ev.UserFinal, ev.UserFinalPayload(utterance_id="u", text=text))
    action = body(ev.AgentAction, ev.AgentActionPayload(
        action=ev.Action.RESPOND, action_probs={ev.Action.RESPOND: 1.0}, based_on_id=final.id))
    return [final, mark(S.STT_FINAL, stt, final.id), action, mark(S.TURN_DECISION, decide, action.id),
            mark(S.LLM_FIRST_TOKEN, llm, action.id, brain), mark(S.FIRST_CLAUSE, clause, action.id, brain),
            mark(S.TTS_FIRST_CHUNK, tts, action.id), mark(S.FIRST_AUDIO, first, action.id)]


def test_marks_are_matched_to_their_turn():
    events = turn("Hey Jason.", 300, 80, 60, 110, 100, 550) + turn("How are you?", 700, 90, 70, 200, 150, 1100)
    report = analyze(events)
    first, second = report.turns
    assert (first.said, first.decide, first.stt, first.llm_first, first.clause, first.tts, first.first_audio) == (
        "Hey Jason.", 300, 80, 60, 110, 100, 550)
    assert first.rest == pytest.approx(40)  # 550 - 300 - 110 - 100
    assert second.said == "How are you?" and second.first_audio == 1100


def test_a_reply_cancelled_before_its_first_sound_has_no_first_audio():
    events = turn("I meant like your", 700, 110, 0, 0, 0, 0)[:4]  # decided, then cut off before audio
    [t] = analyze(events).turns
    assert (t.decide, t.first_audio, t.rest) == (700, None, None)


def test_barge_ins_and_network_are_reported():
    action_id = "a1"
    events = [
        body(ev.ReplyHeard, ev.ReplyHeardPayload(action_id=action_id, text="Sure thing,", complete=False,
                                                 played_ms=400.0, interrupted_by="u2")),
        mark(S.BARGE_IN, 105, action_id),
        mark(S.NETWORK_RTT, 9, None),
    ]
    report = analyze(events)
    assert [(b.stopped_ms, b.heard) for b in report.barge_ins] == [(105, "Sure thing,")]
    assert report.network_rtt == [9]


def test_exit_check_needs_20_turns_and_a_median_under_1s():
    fast = [e for i in range(20) for e in turn(f"turn {i}", 300, 80, 60, 110, 100, 900)]
    assert analyze(fast).passes()
    assert not analyze(fast[: 8 * 19]).passes()  # only 19 turns
    slow = [e for i in range(20) for e in turn(f"turn {i}", 300, 80, 60, 110, 100, 1100)]
    assert not analyze(slow).passes()
    assert "PASS" in analyze(fast).lines()[-1] and "NOT YET" in analyze(slow).lines()[-1]


def test_percentile_is_nearest_rank():
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile(list(range(1, 101)), 99) == 99


def test_last_run_starts_at_the_bodys_latest_startup():
    early = turn("old", 300, 80, 60, 110, 100, 2000)
    startup = body(ev.System, ev.SystemPayload(kind="startup"))
    brain_startup = brain(ev.System, ev.SystemPayload(kind="startup"))
    late = turn("new", 300, 80, 60, 110, 100, 600)
    run = last_run(early + [startup] + late + [brain_startup], "mac")
    assert [t.said for t in analyze(run).turns] == ["new"]
