"""Messages sent over the Mac↔brain WebSocket.

Every WebSocket message is one JSON object with a "kind" field:

    hello        brain → Mac, the first message on every connection
    new_session  brain → Mac, when a new day starts
    event        either way, one bus event
    sync         Mac → brain, a clock-sync probe (see net/clocksync.py)
    sync_reply   brain → Mac, the brain's receive and send times for a probe

On the brain's separate /admin address (used by `jason new-day`):

    command         → brain, e.g. {"kind": "command", "name": "new_day"}
    command_result  ← brain, whether it worked and a message for the user

These are link plumbing, not bus events: they are never published or logged
as events themselves (link_up / link_down system events record what happened).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from jason.core.events import SCHEMA_VERSION, AnyEvent

# Bump when the message format changes, so mismatched machines refuse to talk.
PROTOCOL_VERSION = 1


class _Msg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Hello(_Msg):
    kind: Literal["hello"] = "hello"
    protocol_version: int = PROTOCOL_VERSION
    schema_version: int = SCHEMA_VERSION
    machine: str  # the brain's machine name
    session_id: str  # the current session (one of Jason's days)


class NewSession(_Msg):
    kind: Literal["new_session"] = "new_session"
    session_id: str


class EventMsg(_Msg):
    kind: Literal["event"] = "event"
    event: AnyEvent


class Sync(_Msg):
    kind: Literal["sync"] = "sync"
    t1_ns: int  # Mac monotonic clock when sent


class SyncReply(_Msg):
    kind: Literal["sync_reply"] = "sync_reply"
    t1_ns: int  # echoed from the sync
    t2_ns: int  # brain monotonic clock when the sync arrived
    t3_ns: int  # brain monotonic clock when this reply was sent


class Command(_Msg):
    kind: Literal["command"] = "command"
    name: str


class CommandResult(_Msg):
    kind: Literal["command_result"] = "command_result"
    ok: bool
    message: str


AnyMessage = Hello | NewSession | EventMsg | Sync | SyncReply | Command | CommandResult
Message = Annotated[AnyMessage, Field(discriminator="kind")]

_adapter: TypeAdapter[Message] = TypeAdapter(Message)


def encode(msg: AnyMessage) -> str:
    return msg.model_dump_json()


def decode(data: str | bytes) -> AnyMessage:
    """Parse one message. Raises pydantic.ValidationError if it is malformed."""
    return _adapter.validate_json(data)
