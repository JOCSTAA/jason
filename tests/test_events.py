import json

import pytest
from pydantic import ValidationError

from jason.core import events as ev

ENV = {"machine": "mac", "session_id": "test-session"}

SAMPLES: list[ev.Event] = [
    ev.UserPartial(**ENV, payload=ev.UserPartialPayload(utterance_id="u1", text="so I was")),
    ev.UserFinal(**ENV, payload=ev.UserFinalPayload(utterance_id="u1", text="so I was thinking")),
    ev.TurnEndProb(**ENV, payload=ev.TurnEndProbPayload(utterance_id="u1", prob=0.82)),
    ev.AgentAction(
        **ENV,
        payload=ev.AgentActionPayload(
            action=ev.Action.ASK, policy_version="rules-0", features={"pause_ms": 640.0}
        ),
    ),
    ev.AgentUtterance(**ENV, payload=ev.AgentUtterancePayload(action_id="a1", text="Why?")),
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
    ev.LatencyMark(**ENV, payload=ev.LatencyMarkPayload(stage="network_rtt", duration_ns=4_200_000)),
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
