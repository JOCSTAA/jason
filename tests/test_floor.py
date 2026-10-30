"""The rule-based floor controller: when Jason answers, and what it logs."""

import asyncio

import pytest

from jason.body.floor import FloorController
from jason.core import clock
from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import FloorConfig
from jason.log.rawlog import RawLog

SESSION = "day-0001_2026-10-30"
MS = 1_000_000


@pytest.fixture
def setup(tmp_path):
    rawlog = RawLog(tmp_path, "mac")
    bus = EventBus("mac", rawlog, session_id=SESSION)
    out = bus.subscribe(ev.AgentAction, ev.LatencyMark)
    state = {"playing": False, "first": {}}
    floor = FloorController(bus, FloorConfig(), jason_speaking=lambda: state["playing"],
                            first_played_ns=lambda a: state["first"].get(a))
    yield bus, floor, out, state
    rawlog.close()


def emit(bus, cls, payload):
    return bus.emit(cls, payload)


def utterance(bus, floor, prob, ago_ms, text="are you there", uid="u1", final=True):
    """The user spoke, then paused `ago_ms` ago; Smart Turn said `prob`."""
    floor.observe(emit(bus, ev.SpeechStart, ev.SpeechStartPayload(utterance_id=uid, audio_ns=1, during_agent_speech=False)))
    end = clock.monotonic_ns() - ago_ms * MS
    floor.observe(emit(bus, ev.SpeechEnd, ev.SpeechEndPayload(utterance_id=uid, audio_ns=end, speech_ms=1500.0)))
    floor.observe(emit(bus, ev.TurnEndProb, ev.TurnEndProbPayload(utterance_id=uid, prob=prob)))
    if final:
        floor.observe(emit(bus, ev.UserFinal, ev.UserFinalPayload(utterance_id=uid, text=text)))
    return floor.turn.final_id


def decisions(out):
    items = []
    while not out._queue.empty():
        items.append(out._queue.get_nowait())
    return [e for e in items if isinstance(e, ev.AgentAction)], [e for e in items if isinstance(e, ev.LatencyMark)]


@pytest.mark.parametrize("prob,wait", [(0.95, 200), (0.73, 200), (0.5, 700), (0.1, 1500)])
def test_the_wait_depends_on_how_finished_the_user_sounds(setup, prob, wait):
    bus, floor, out, state = setup
    assert floor.wait_ms(prob) == wait


async def test_a_finished_sentence_gets_an_answer_after_200ms(setup):
    bus, floor, out, state = setup
    final_id = utterance(bus, floor, prob=0.9, ago_ms=250)
    floor.tick()
    [action], marks = decisions(out)
    p = action.payload
    assert (p.action, p.action_probs, p.based_on_id, p.utterance_id) == (ev.Action.RESPOND, {ev.Action.RESPOND: 1.0}, final_id, "u1")
    assert p.policy_version == "rules-1"
    assert set(p.features) == {"pause_ms", "turn_end_prob", "speech_ms", "words", "wait_ms", "since_jason_ms"}
    assert (p.features["turn_end_prob"], p.features["words"], p.features["wait_ms"]) == (0.9, 3.0, 200.0)
    [mark] = marks
    assert mark.payload.stage is ev.LatencyStage.TURN_DECISION and mark.payload.ref_id == action.id
    assert mark.payload.duration_ns >= 250 * MS


async def test_an_unsure_pause_waits_and_logs_one_silent_decision(setup):
    bus, floor, out, state = setup
    utterance(bus, floor, prob=0.5, ago_ms=300)
    floor.tick()
    floor.tick()
    actions, _ = decisions(out)
    assert [a.payload.action for a in actions] == [ev.Action.SILENT]  # once per pause, not per tick
    floor.turn.pause_start_ns -= 500 * MS  # the pause reaches 800 ms
    floor.tick()
    actions, _ = decisions(out)
    assert [a.payload.action for a in actions] == [ev.Action.RESPOND]


async def test_talking_again_resets_the_wait(setup):
    bus, floor, out, state = setup
    utterance(bus, floor, prob=0.1, ago_ms=1000)
    floor.observe(emit(bus, ev.SpeechStart, ev.SpeechStartPayload(utterance_id="u1", audio_ns=2, during_agent_speech=False)))
    floor.tick()
    assert decisions(out)[0] == []  # speaking again: nothing to decide
    assert floor.turn.final_id is None and floor.turn.turn_prob is None


async def test_no_answer_without_the_transcript_for_this_pause(setup):
    bus, floor, out, state = setup
    utterance(bus, floor, prob=0.95, ago_ms=400, final=False)
    floor.tick()
    assert decisions(out)[0] == []


async def test_one_answer_per_utterance(setup):
    bus, floor, out, state = setup
    utterance(bus, floor, prob=0.95, ago_ms=400)
    floor.tick()
    floor.tick()
    actions, _ = decisions(out)
    assert [a.payload.action for a in actions] == [ev.Action.RESPOND]


async def test_no_answer_while_jason_is_speaking(setup):
    bus, floor, out, state = setup
    state["playing"] = True
    utterance(bus, floor, prob=0.95, ago_ms=400)
    floor.tick()
    assert decisions(out)[0] == []
    state["playing"] = False
    floor.tick()
    actions, _ = decisions(out)
    assert actions[0].payload.action is ev.Action.RESPOND
    assert actions[0].payload.features["since_jason_ms"] >= 0


async def test_time_to_first_audio_is_logged_when_the_reply_starts_playing(setup):
    bus, floor, out, state = setup
    utterance(bus, floor, prob=0.95, ago_ms=300)
    pause_start = floor.turn.pause_start_ns
    floor.tick()
    [action], _ = decisions(out)
    state["first"][action.id] = pause_start + 640 * MS  # Jason's first sample reached the speaker
    await asyncio.sleep(0.05)
    _, marks = decisions(out)
    [first] = [m for m in marks if m.payload.stage is ev.LatencyStage.FIRST_AUDIO]
    assert first.payload.duration_ns == 640 * MS and first.payload.ref_id == action.id

