"""Text-only mode for the body: `jason run --role body --text`.

Typing a line stands in for the voice pipeline (M1 steps 7-11): the line is
the user's final transcript, and pressing Enter is the decision to respond.
Everything after that is the real path: the link, the brain's LLM component,
vLLM and the same events the voice loop will use. So it tests the whole brain
side without audio, and later the floor controller without a microphone.
"""

import asyncio
from collections.abc import Awaitable, Callable
from uuid import uuid4

from jason.core import clock
from jason.core import events as ev
from jason.core.bus import EventBus

POLICY = "text-mode"  # marks these decisions in the logs: Enter, not a policy, chose them


async def read_stdin(prompt: str) -> str | None:
    """One typed line, or None at end of input (Ctrl+D / Ctrl+Z)."""
    try:
        # In a thread, so the link keeps running while we wait for typing.
        return await asyncio.to_thread(input, prompt)
    except EOFError:
        return None


async def text_chat(
    bus: EventBus,
    read_line: Callable[[str], Awaitable[str | None]] = read_stdin,
    out: Callable[[str], None] = print,
    reply_timeout_s: float = 30.0,
    report_heard: bool = True,
) -> None:
    """Chat until the input ends or the user types /quit. With report_heard
    False (when Jason speaks out loud), the speaker reports what was heard."""
    out("Type to talk to Jason; /quit or Ctrl+D to stop.")
    with bus.subscribe(ev.ReplyChunk, ev.System) as replies:
        while True:
            line = await read_line("you> ")
            if line is None or line.strip() == "/quit":
                return
            if not line.strip():
                continue
            await _one_turn(bus, replies, line.strip(), out, reply_timeout_s, report_heard)


async def _one_turn(bus, replies, text: str, out: Callable[[str], None], timeout_s: float,
                    report_heard: bool = True) -> None:
    utterance_id = uuid4().hex
    final = bus.emit(ev.UserFinal, ev.UserFinalPayload(utterance_id=utterance_id, text=text))
    action = bus.emit(
        ev.AgentAction,
        ev.AgentActionPayload(
            action=ev.Action.RESPOND,
            action_probs={ev.Action.RESPOND: 1.0},
            utterance_id=utterance_id,
            based_on_id=final.id,
            policy_version=POLICY,
        ),
    )
    t0 = clock.monotonic_ns()
    clauses: list[str] = []
    first_ms = None
    try:
        async with asyncio.timeout(timeout_s):
            while True:
                event = await replies.get()
                if isinstance(event, ev.System) and event.payload.kind == "llm_error" \
                        and event.payload.data.get("action_id") == action.id:
                    out(f"[brain: {event.payload.message}]")
                elif isinstance(event, ev.ReplyChunk) and event.payload.action_id == action.id:
                    if event.payload.text:
                        if first_ms is None:
                            first_ms = (clock.monotonic_ns() - t0) / 1e6
                        clauses.append(event.payload.text)
                        out(f"jason> {event.payload.text}")
                    if event.payload.last:
                        break
    except TimeoutError:
        out(f"[no reply within {timeout_s:g} s]")
        bus.emit(ev.ReplyCancel, ev.ReplyCancelPayload(action_id=action.id, reason="timeout"))
        return
    if first_ms is not None:
        out(f"[first clause after {first_ms:.0f} ms]")
    if not report_heard:
        return
    # Everything shown was "heard". played_ms is audio time, so 0 in text mode.
    bus.emit(
        ev.ReplyHeard,
        ev.ReplyHeardPayload(action_id=action.id, text=" ".join(clauses), complete=True, played_ms=0.0),
    )
