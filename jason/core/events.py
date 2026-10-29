"""Versioned event schema shared by both machines.

Every event has the same envelope (id, type, timestamps, machine, session,
schema version) and a typed payload. Events are immutable facts: once
created they are never edited, which is what lets the raw log be replayed and
re-scored later.

Payloads are stubs sized for M0; later milestones add fields. Any change to a
payload's meaning bumps SCHEMA_VERSION so old log lines stay interpretable.

Version history:
  1  M0.
  2  Envelope gains seq and run_id (stamped by the bus); agent_action gains the
     required action_probs.
"""

import math
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from jason.core import clock

SCHEMA_VERSION = 2


class _Frozen(BaseModel):
    # frozen: events are facts and can't be modified after creation.
    # extra="forbid": a misspelled field is an error, not silently dropped data.
    model_config = ConfigDict(frozen=True, extra="forbid")


class Payload(_Frozen):
    pass


class Event(_Frozen):
    id: str = Field(default_factory=lambda: uuid4().hex)
    type: str
    schema_version: int = SCHEMA_VERSION
    # Monotonic: for latency, only comparable on the producing machine.
    ts_monotonic_ns: int = Field(default_factory=clock.monotonic_ns)
    # Wall: for humans and for lining up days across machines.
    ts_wall: AwareDatetime = Field(default_factory=clock.wall_now)
    machine: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    # Stamped by the producing machine's bus when the event is published (never
    # set them yourself): seq counts 0, 1, 2, ... within one run, and run_id is
    # random per process start. A jump in seq on the receiving side means events
    # were lost on the way. None on events that were never published, and in
    # schema-1 logs.
    seq: int | None = Field(default=None, ge=0)
    run_id: str | None = None
    payload: Payload


# --- Action space (design §4) ---------------------------------------------


class Action(StrEnum):
    SILENT = "silent"
    BACKCHANNEL = "backchannel"
    RESPOND = "respond"
    INTERJECT = "interject"
    ASK = "ask"
    THINK = "think"


# --- Payloads ---------------------------------------------------------------

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class UserPartialPayload(Payload):
    utterance_id: str
    text: str


class UserFinalPayload(Payload):
    utterance_id: str
    text: str


class TurnEndProbPayload(Payload):
    utterance_id: str | None = None
    prob: Probability


class AgentActionPayload(Payload):
    action: Action
    # The probability the policy gave each action when it chose; actions left
    # out had probability 0, so a deterministic rule logs {action: 1.0}.
    # Off-policy evaluation and unbiased retraining from logs (M4, sleep) need
    # the chosen action's probability; the rest of the distribution helps too.
    action_probs: dict[Action, Probability]
    policy_version: str | None = None
    # Inputs the policy saw when it chose. Logged with every decision so the
    # learner can replay "what did it know, what did it do, what happened".
    features: dict[str, float] = {}

    @model_validator(mode="after")
    def _probs_are_a_distribution(self) -> Self:
        if self.action_probs.get(self.action, 0.0) <= 0.0:
            raise ValueError(f"action_probs gives the chosen action {self.action!r} no probability")
        total = sum(self.action_probs.values())
        if not math.isclose(total, 1.0, abs_tol=1e-3):
            raise ValueError(f"action_probs sum to {total:.4f}, not 1")
        return self


class AgentUtterancePayload(Payload):
    action_id: str  # the agent_action event this utterance carries out
    text: str


class ReactionPayload(Payload):
    """A classified user utterance (design §5.3 B)."""

    utterance_id: str
    is_feedback: bool
    target_action_id: str | None = None  # credit assignment: which action it judges
    polarity: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    strength: Probability = 0.0
    labeler: str  # which classifier produced this label, e.g. a live judge or the nightly re-scorer


class RewardPayload(Payload):
    action_id: str  # the agent_action event being rewarded
    value: float
    # What produced it, e.g. "reaction", "barge_in", "curiosity". Kept apart
    # so extrinsic and intrinsic rewards can be weighted (and audited) separately.
    source: str
    reaction_id: str | None = None


class WonderPredictionPayload(Payload):
    action_id: str  # the candidate or taken action being predicted for
    predicted_utterance: str
    predicted_polarity: Annotated[float, Field(ge=-1.0, le=1.0)] | None = None
    wonder_version: str | None = None


class LatencyMarkPayload(Payload):
    stage: str  # e.g. "stt", "llm_first_token", "network_rtt"
    duration_ns: Annotated[int, Field(ge=0)]
    ref_id: str | None = None  # the event this measurement belongs to


class SystemPayload(Payload):
    kind: str  # e.g. "startup", "connected", "clock_offset"
    message: str = ""
    data: dict[str, float | int | str | bool | None] = {}


class PingPayload(Payload):
    seq: int


class PongPayload(Payload):
    """Reply to a ping. With the ping's own timestamps this gives the four
    timestamps NTP-style offset estimation needs: ping sent (ping event),
    ping received (here), pong sent (pong event), pong received (receiver)."""

    ping_id: str
    seq: int
    ping_received_ns: int  # responder's monotonic clock when the ping arrived


# --- Events -----------------------------------------------------------------


class UserPartial(Event):
    type: Literal["user_partial"] = "user_partial"
    payload: UserPartialPayload


class UserFinal(Event):
    type: Literal["user_final"] = "user_final"
    payload: UserFinalPayload


class TurnEndProb(Event):
    type: Literal["turn_end_prob"] = "turn_end_prob"
    payload: TurnEndProbPayload


class AgentAction(Event):
    type: Literal["agent_action"] = "agent_action"
    payload: AgentActionPayload


class AgentUtterance(Event):
    type: Literal["agent_utterance"] = "agent_utterance"
    payload: AgentUtterancePayload


class Reaction(Event):
    type: Literal["reaction"] = "reaction"
    payload: ReactionPayload


class Reward(Event):
    type: Literal["reward"] = "reward"
    payload: RewardPayload


class WonderPrediction(Event):
    type: Literal["wonder_prediction"] = "wonder_prediction"
    payload: WonderPredictionPayload


class LatencyMark(Event):
    type: Literal["latency_mark"] = "latency_mark"
    payload: LatencyMarkPayload


class System(Event):
    type: Literal["system"] = "system"
    payload: SystemPayload


class Ping(Event):
    type: Literal["ping"] = "ping"
    payload: PingPayload


class Pong(Event):
    type: Literal["pong"] = "pong"
    payload: PongPayload


EVENT_CLASSES: tuple[type[Event], ...] = (
    UserPartial,
    UserFinal,
    TurnEndProb,
    AgentAction,
    AgentUtterance,
    Reaction,
    Reward,
    WonderPrediction,
    LatencyMark,
    System,
    Ping,
    Pong,
)

# "type" picks the class, so parsing a log line yields the right event class
# with its payload fully validated.
AnyEvent = Annotated[
    UserPartial
    | UserFinal
    | TurnEndProb
    | AgentAction
    | AgentUtterance
    | Reaction
    | Reward
    | WonderPrediction
    | LatencyMark
    | System
    | Ping
    | Pong,
    Field(discriminator="type"),
]

_adapter: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)


def to_json(event: Event) -> str:
    return event.model_dump_json()


def from_json(data: str | bytes) -> Event:
    """Parse one serialized event. Raises pydantic.ValidationError if the type
    is unknown or any field is missing, misspelled or out of range."""
    return _adapter.validate_json(data)
