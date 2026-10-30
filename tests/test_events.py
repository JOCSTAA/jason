import json

import pytest
from pydantic import ValidationError

from jason.core import events as ev

ENV = {"machine": "mac", "session_id": "test-session"}

SAMPLES: list[ev.Event] = [
    ev.UserPartial(**ENV, payload=ev.UserPartialPayload(utterance_id="u1", text="so I was")),
    ev.UserFinal(**ENV, payload=ev.UserFinalPayload(utterance_id="u1", text="so I was thinking")),
    ev.TurnEndProb(**ENV, payload=ev.TurnEndProbPayload(utterance_id="u1", prob=0.82)),
    ev.SpeechStart(**ENV, payload=ev.SpeechStartPayload(utterance_id="u1", audio_ns=1_000, during_agent_speech=False)),
    ev.SpeechEnd(**ENV, payload=ev.SpeechEndPayload(utterance_id="u1", audio_ns=2_000, speech_ms=1840.0)),
    ev.AgentAction(
        **ENV,
        payload=ev.AgentActionPayload(
            action=ev.Action.ASK,
            action_probs={ev.Action.ASK: 0.7, ev.Action.SILENT: 0.2, ev.Action.RESPOND: 0.1},
            utterance_id="u1",
            based_on_id="e17",
            policy_version="rules-0",
            features={"pause_ms": 640.0},
        ),
    ),
    ev.AgentUtterance(
        machine="brain", session_id="test-session",
        payload=ev.AgentUtterancePayload(action_id="a1", text="Why?", complete=True, model_version="qwen3.5-4b@851bf6e"),
    ),
    ev.ReplyChunk(machine="brain", session_id="test-session", payload=ev.ReplyChunkPayload(action_id="a1", index=0, text="Why?", last=True)),
    ev.ReplyCancel(**ENV, payload=ev.ReplyCancelPayload(action_id="a1", reason="barge_in")),
    ev.ReplyHeard(
        **ENV,
        payload=ev.ReplyHeardPayload(action_id="a1", text="Wh", complete=False, played_ms=180.0, interrupted_by="u2"),
    ),
    ev.Reaction(
        **ENV,
        payload=ev.ReactionPayload(
            utterance_id="u2",
            is_feedback=True,
            target_action_id="a1",
            polarity=0.9,
            strength=0.7,
            labeler="live-judge-0",
        ),
    ),
    ev.Reward(**ENV, payload=ev.RewardPayload(action_id="a1", value=0.63, source="reaction")),
    ev.WonderPrediction(
        **ENV,
        payload=ev.WonderPredictionPayload(
            action_id="a1", predicted_utterance="Because...", predicted_polarity=0.4
        ),
    ),
    ev.LatencyMark(**ENV, payload=ev.LatencyMarkPayload(stage=ev.LatencyStage.FIRST_AUDIO, duration_ns=640_000_000)),
    ev.System(**ENV, payload=ev.SystemPayload(kind="startup", data={"pid": 123})),
    ev.Ping(**ENV, payload=ev.PingPayload(seq=1)),
    ev.Pong(
        machine="brain",
        session_id="test-session",
        payload=ev.PongPayload(ping_id="p1", seq=1, ping_received_ns=5),
    ),
]


def test_samples_cover_every_event_type():
    assert {type(e) for e in SAMPLES} == set(ev.EVENT_CLASSES)


@pytest.mark.parametrize("event", SAMPLES, ids=lambda e: e.type)
def test_round_trip(event):
    restored = ev.from_json(ev.to_json(event))
    assert type(restored) is type(event)
    assert restored == event


def test_envelope_defaults():
    e = ev.Ping(**ENV, payload=ev.PingPayload(seq=1))
    assert e.schema_version == ev.SCHEMA_VERSION
    assert len(e.id) == 32
    assert e.ts_wall.tzinfo is not None
    assert e.ts_monotonic_ns > 0


def test_ids_are_unique():
    ids = {ev.Ping(**ENV, payload=ev.PingPayload(seq=i)).id for i in range(1000)}
    assert len(ids) == 1000


def test_action_space_matches_design():
    assert {a.value for a in ev.Action} == {
        "silent", "backchannel", "respond", "interject", "ask", "think"
    }


def test_events_are_immutable():
    e = ev.Ping(**ENV, payload=ev.PingPayload(seq=1))
    with pytest.raises(ValidationError):
        e.machine = "brain"


def test_unknown_type_rejected():
    line = json.loads(ev.to_json(SAMPLES[0]))
    line["type"] = "telepathy"
    with pytest.raises(ValidationError):
        ev.from_json(json.dumps(line))


def test_misspelled_field_rejected():
    line = json.loads(ev.to_json(ev.Ping(**ENV, payload=ev.PingPayload(seq=1))))
    line["payload"]["sequence"] = line["payload"].pop("seq")
    with pytest.raises(ValidationError):
        ev.from_json(json.dumps(line))


def test_out_of_range_values_rejected():
    with pytest.raises(ValidationError):
        ev.TurnEndProbPayload(prob=1.5)
    with pytest.raises(ValidationError):
        ev.ReactionPayload(utterance_id="u", is_feedback=True, polarity=-2.0, labeler="x")


def test_payload_must_match_type():
    line = json.loads(ev.to_json(ev.Ping(**ENV, payload=ev.PingPayload(seq=1))))
    line["type"] = "pong"
    with pytest.raises(ValidationError):
        ev.from_json(json.dumps(line))


def test_empty_machine_rejected():
    with pytest.raises(ValidationError):
        ev.Ping(machine="", session_id="s", payload=ev.PingPayload(seq=1))


# --- schema 2 -------------------------------------------------------------------


def test_schema_version_is_5():
    assert ev.SCHEMA_VERSION == 5


def test_seq_and_run_id_default_to_none_and_round_trip():
    e = ev.Ping(**ENV, payload=ev.PingPayload(seq=1))
    assert (e.seq, e.run_id) == (None, None)
    stamped = e.model_copy(update={"seq": 41, "run_id": "r1"})
    assert ev.from_json(ev.to_json(stamped)) == stamped


def test_schema_1_lines_still_read():
    line = json.loads(ev.to_json(ev.Ping(**ENV, payload=ev.PingPayload(seq=1))))
    line["schema_version"] = 1
    del line["seq"], line["run_id"]
    restored = ev.from_json(json.dumps(line))
    assert (restored.schema_version, restored.seq) == (1, None)


def test_negative_seq_rejected():
    with pytest.raises(ValidationError):
        ev.Ping(**ENV, seq=-1, payload=ev.PingPayload(seq=1))


def test_deterministic_action_probs():
    p = ev.AgentActionPayload(action=ev.Action.SILENT, action_probs={ev.Action.SILENT: 1.0})
    assert p.action_probs == {ev.Action.SILENT: 1.0}


def test_action_probs_must_include_the_chosen_action():
    with pytest.raises(ValidationError, match="no probability"):
        ev.AgentActionPayload(action=ev.Action.ASK, action_probs={ev.Action.SILENT: 1.0})
    with pytest.raises(ValidationError, match="no probability"):
        ev.AgentActionPayload(action=ev.Action.ASK, action_probs={ev.Action.ASK: 0.0, ev.Action.SILENT: 1.0})


def test_action_probs_must_sum_to_one():
    with pytest.raises(ValidationError, match="sum to"):
        ev.AgentActionPayload(action=ev.Action.ASK, action_probs={ev.Action.ASK: 0.5, ev.Action.SILENT: 0.3})
    # Rounding within 0.001 is fine.
    ev.AgentActionPayload(action=ev.Action.ASK, action_probs={ev.Action.ASK: 0.3333, ev.Action.SILENT: 0.6667})


def test_action_probs_values_are_probabilities():
    with pytest.raises(ValidationError):
        ev.AgentActionPayload(action=ev.Action.ASK, action_probs={ev.Action.ASK: 1.5, ev.Action.SILENT: -0.5})


def test_action_probs_is_required():
    with pytest.raises(ValidationError):
        ev.AgentActionPayload(action=ev.Action.ASK)


def test_action_probs_keys_round_trip_as_actions():
    e = next(s for s in SAMPLES if s.type == "agent_action")
    restored = ev.from_json(ev.to_json(e))
    assert all(isinstance(k, ev.Action) for k in restored.payload.action_probs)


# --- schema 3 -------------------------------------------------------------------


def test_turn_end_prob_needs_its_utterance():
    with pytest.raises(ValidationError):
        ev.TurnEndProbPayload(prob=0.5)


def test_agent_action_may_point_at_no_user_speech():
    # Jason speaking first: nothing to point at.
    p = ev.AgentActionPayload(action=ev.Action.ASK, action_probs={ev.Action.ASK: 1.0})
    assert (p.utterance_id, p.based_on_id) == (None, None)


def test_agent_utterance_must_say_whether_it_finished():
    with pytest.raises(ValidationError):
        ev.AgentUtterancePayload(action_id="a1", text="Why?")


def test_latency_stage_is_a_fixed_list():
    with pytest.raises(ValidationError):
        ev.LatencyMarkPayload(stage="first_audoi", duration_ns=1)
    # The string form still works, so callers can pass plain names.
    assert ev.LatencyMarkPayload(stage="ping_rtt", duration_ns=1).stage is ev.LatencyStage.PING_RTT


def test_schema_2_latency_marks_still_read():
    # M0 logs hold network_rtt and ping_rtt marks with free-text stages.
    line = json.loads(ev.to_json(ev.LatencyMark(**ENV, payload=ev.LatencyMarkPayload(stage="network_rtt", duration_ns=5))))
    line["schema_version"] = 2
    restored = ev.from_json(json.dumps(line))
    assert restored.payload.stage is ev.LatencyStage.NETWORK_RTT


def test_negative_amounts_rejected():
    with pytest.raises(ValidationError):
        ev.SpeechEndPayload(utterance_id="u1", audio_ns=1, speech_ms=-1.0)
    with pytest.raises(ValidationError):
        ev.ReplyHeardPayload(action_id="a1", text="", complete=False, played_ms=-1.0)
    with pytest.raises(ValidationError):
        ev.ReplyChunkPayload(action_id="a1", index=-1, text="x", last=True)
