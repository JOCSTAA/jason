"""Versioned event schema shared by both machines.

Every event has the same envelope (id, type, timestamps, machine, session,
schema version) and a typed payload. Events are immutable facts: once
created they are never edited, which is what lets the raw log be replayed and
re-scored later.

Payloads grow milestone by milestone (the voice loop's are M1's; reaction,
reward and wonder_prediction are still stubs). Any change to a
payload's meaning bumps SCHEMA_VERSION so old log lines stay interpretable.

Version history:
  1  M0.
  2  Envelope gains seq and run_id (stamped by the bus); agent_action gains the
     required action_probs.
  3  M1 voice loop. New: speech_start, speech_end, reply_chunk, reply_cancel,
     reply_heard. turn_end_prob.utterance_id is required; agent_action gains
     utterance_id and based_on_id; agent_utterance gains the required
     complete and model_version; latency_mark.stage is a fixed list (the old
     free-text values network_rtt and ping_rtt are on it, so older logs read).
  4  latency_mark.stage gains barge_in (M1 step 12).
"""

import math
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from jason.core import clock

SCHEMA_VERSION = 4


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
    """The transcript of the utterance up to a pause. Emitted at every pause,
    because any pause may turn out to be the end of the turn; a later one for
    the same utterance (the user carried on) supersedes it."""

    utterance_id: str
    text: str


class TurnEndProbPayload(Payload):
    utterance_id: str
    prob: Probability


class SpeechStartPayload(Payload):
    """The VAD heard the user start (or resume) speaking."""

    # One utterance is one user turn. A pause doesn't end it: if the user
    # carries on, the next speech_start reuses the same id.
    utterance_id: str
    # When the first voiced frame was captured (the body's monotonic clock).
    # The event itself is created later, after buffering and the VAD, so its
    # own timestamp would make every latency look shorter than the user felt.
    audio_ns: int
    # Jason was talking at the time: a possible barge-in.
    during_agent_speech: bool


class SpeechEndPayload(Payload):
    """The VAD heard the user pause. Not necessarily the end of the turn:
    whether the turn is over is the floor controller's call."""

    utterance_id: str
    # When the last voiced frame ended (the body's monotonic clock). Time to
    # first audio is measured from here (M1 exit criterion).
    audio_ns: int
    speech_ms: Annotated[float, Field(ge=0.0)]  # length of the stretch of speech that just ended


class AgentActionPayload(Payload):
    action: Action
    # The probability the policy gave each action when it chose; actions left
    # out had probability 0, so a deterministic rule logs {action: 1.0}.
    # Off-policy evaluation and unbiased retraining from logs (M4, sleep) need
    # the chosen action's probability; the rest of the distribution helps too.
    action_probs: dict[Action, Probability]
    # The user turn this decision belongs to, and the exact user_partial or
    # user_final event it saw. A reply or interjection made mid-sentence is
    # then traceable to the words it reacted to, which a reaction judge (M3)
    # and the learner (M4) need. None when there's no user speech to point at
    # (e.g. Jason speaking first).
    utterance_id: str | None = None
    based_on_id: str | None = None
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
    """The whole reply as generated, logged once when generation stops. What
    the user actually heard is in reply_heard."""

    action_id: str  # the agent_action event this utterance carries out
    text: str
    complete: bool  # False when generation was cancelled (e.g. barge-in)
    # Which weights wrote it. Nightly training changes the weights, so every
    # reply must say which version produced it for learning to be attributable.
    model_version: str | None = None


class ReplyChunkPayload(Payload):
    """One clause of a reply, sent as soon as it's generated so speech can
    start before the whole reply exists. Clauses, not tokens: one event per
    token would be ~60 a second in both raw logs and over the link."""

    action_id: str
    index: Annotated[int, Field(ge=0)]  # 0, 1, 2, ... within the reply
    text: str
    last: bool  # no more chunks follow for this reply


class ReplyCancelPayload(Payload):
    """Asks the brain to stop generating a reply. An event, not a direct
    call, because components only talk through the bus."""

    action_id: str
    reason: str  # e.g. "barge_in"


class ReplyHeardPayload(Payload):
    """What the user actually heard of a reply, sent when playback ends.

    After a barge-in this is shorter than the generated reply; reactions and
    corrections (M3) and training pairs (M8) are judged against this text,
    not the full reply. A reply with interrupted_by set is the barge-in
    record, from which M3 derives the barge-in penalty (design §9).
    """

    action_id: str
    text: str
    complete: bool  # the whole reply was played
    played_ms: Annotated[float, Field(ge=0.0)]
    interrupted_by: str | None = None  # the utterance_id that cut it off


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


class LatencyStage(StrEnum):
    """Every stage that logs a latency. A fixed list, so a misspelled stage is
    an error instead of a silently separate stage in the reports."""

    NETWORK_RTT = "network_rtt"  # body↔brain round trip (clock-sync probes)
    PING_RTT = "ping_rtt"  # M0 pinger: ping published → pong received, bus to bus
    TURN_DECISION = "turn_decision"  # user stopped speaking → floor controller chose to respond
    STT_FINAL = "stt_final"  # final transcription call
    LLM_FIRST_TOKEN = "llm_first_token"  # request sent → first token back
    FIRST_CLAUSE = "first_clause"  # request sent → first reply_chunk ready
    TTS_FIRST_CHUNK = "tts_first_chunk"  # clause received → first audio synthesized
    FIRST_AUDIO = "first_audio"  # user's last voiced frame → first reply sample at the speaker
    BARGE_IN = "barge_in"  # user's first voiced sound over Jason → Jason's playback stopped


class LatencyMarkPayload(Payload):
    stage: LatencyStage
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


class SpeechStart(Event):
    type: Literal["speech_start"] = "speech_start"
    payload: SpeechStartPayload


class SpeechEnd(Event):
    type: Literal["speech_end"] = "speech_end"
    payload: SpeechEndPayload


class AgentAction(Event):
    type: Literal["agent_action"] = "agent_action"
    payload: AgentActionPayload


class AgentUtterance(Event):
    type: Literal["agent_utterance"] = "agent_utterance"
    payload: AgentUtterancePayload


class ReplyChunk(Event):
    type: Literal["reply_chunk"] = "reply_chunk"
    payload: ReplyChunkPayload


class ReplyCancel(Event):
    type: Literal["reply_cancel"] = "reply_cancel"
    payload: ReplyCancelPayload


class ReplyHeard(Event):
    type: Literal["reply_heard"] = "reply_heard"
    payload: ReplyHeardPayload


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
    SpeechStart,
    SpeechEnd,
    AgentAction,
    AgentUtterance,
    ReplyChunk,
    ReplyCancel,
    ReplyHeard,
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
    | SpeechStart
    | SpeechEnd
    | AgentAction
    | AgentUtterance
    | ReplyChunk
    | ReplyCancel
    | ReplyHeard
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
