"""Text-only mode: typed lines in, Jason's reply clauses out."""

import asyncio
import json

import pytest

from jason.body.main import run_body
from jason.body.text import POLICY, text_chat
from jason.brain.main import Brain
from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import BodyConfig, BodyLink, BrainConfig, BrainLink, LlmConfig, generate_token
from jason.log.rawlog import RawLog, read_events

SESSION = "day-0001_2026-10-30"


def scripted(*lines):
    """A line reader that types the given lines, then ends the input."""
    queue = list(lines)

    async def read(prompt):
        await asyncio.sleep(0)
        return queue.pop(0) if queue else None

    return read


async def fake_brain(bus, chunks=("Hi Josh,", "good to see you."), error=None):
    """Answers each respond with the given clauses, as the brain would."""
    with bus.subscribe(ev.AgentAction) as actions:
        async for action in actions:
            if error:
                bus.publish(ev.System(machine="brain", session_id=SESSION, payload=ev.SystemPayload(
                    kind="llm_error", message=error, data={"action_id": action.id})))
            for i, text in enumerate([*chunks, ""] if not error else [""]):
                bus.publish(ev.ReplyChunk(machine="brain", session_id=SESSION, payload=ev.ReplyChunkPayload(
                    action_id=action.id, index=i, text=text, last=i == len(chunks) or bool(error))))


@pytest.fixture
def bus(tmp_path):
    with RawLog(tmp_path, "mac") as rawlog:
        yield EventBus("mac", rawlog, session_id=SESSION)


async def chat(bus, *lines, brain=None, **kw):
    lines_out = []
    task = asyncio.create_task(brain) if brain else None
    await asyncio.sleep(0)
    await text_chat(bus, scripted(*lines), lines_out.append, **kw)
    if task:
        task.cancel()
    return lines_out


async def test_a_typed_line_becomes_a_turn_and_the_reply_prints_by_clause(bus):
    seen = bus.subscribe()
    out = await chat(bus, "hello", brain=fake_brain(bus))
    assert "jason> Hi Josh," in out and "jason> good to see you." in out
    assert any(line.startswith("[first clause after ") for line in out)

    events = []
    while not seen._queue.empty():
        events.append(await seen.get())
    [final] = [e for e in events if isinstance(e, ev.UserFinal)]
    [action] = [e for e in events if isinstance(e, ev.AgentAction)]
    [heard] = [e for e in events if isinstance(e, ev.ReplyHeard)]
    assert final.payload.text == "hello"
    assert action.payload.action is ev.Action.RESPOND
    assert (action.payload.based_on_id, action.payload.utterance_id) == (final.id, final.payload.utterance_id)
    assert action.payload.policy_version == POLICY
    assert (heard.payload.action_id, heard.payload.text, heard.payload.complete) == (
        action.id, "Hi Josh, good to see you.", True)


async def test_brain_errors_are_shown(bus):
    out = await chat(bus, "hello", brain=fake_brain(bus, error="ConnectError: vLLM is down"))
    assert "[brain: ConnectError: vLLM is down]" in out


async def test_no_reply_times_out_and_cancels(bus):
    seen = bus.subscribe(ev.ReplyCancel)
    out = await chat(bus, "hello", reply_timeout_s=0.1)
    assert "[no reply within 0.1 s]" in out
    assert (await seen.get()).payload.reason == "timeout"


async def test_blank_lines_are_skipped_and_quit_stops(bus):
    seen = bus.subscribe(ev.UserFinal)
    await chat(bus, "", "   ", "/quit", "never sent")
    assert seen._queue.empty()


# --- end to end: body ⇄ link ⇄ brain ⇄ (fake) vLLM ---------------------------------


async def fake_vllm_server():
    """A tiny HTTP server streaming one chat completion, as vLLM does."""

    async def handle(reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        length = int(next(l.split(b":")[1] for l in head.split(b"\r\n") if l.lower().startswith(b"content-length")))
        request = json.loads(await reader.readexactly(length))
        requests.append(request)
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n")
        for piece in ["Hello Josh, ", "nice to meet you. ", "How are you?"]:
            writer.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
            await writer.drain()
        writer.write(b"data: [DONE]\n\n")
        await writer.drain()
        writer.close()

    requests = []
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, requests


async def test_text_chat_end_to_end_over_the_link(tmp_path):
    token = tmp_path / "link_token"
    generate_token(token)
    (tmp_path / "persona.md").write_text("You are Jason.")
    vllm, requests = await fake_vllm_server()
    vllm_port = vllm.sockets[0].getsockname()[1]
    brain = Brain(BrainConfig(
        role="brain", machine="brain", state_dir=tmp_path / "brain-state",
        link=BrainLink(listen_host="127.0.0.1", listen_port=0, token_file=token, heartbeat_s=0.2, timeout_s=1.0),
        llm=LlmConfig(base_url=f"http://127.0.0.1:{vllm_port}", model_version="fake@1",
                      persona_file=tmp_path / "persona.md"),
    ))
    await brain.start()
    out = []
    try:
        body = BodyConfig(role="body", machine="mac", state_dir=tmp_path / "mac-state", link=BodyLink(
            brain_url=f"ws://127.0.0.1:{brain.port}", token_file=token, heartbeat_s=0.2, timeout_s=1.0,
            reconnect_max_s=0.2))
        await asyncio.wait_for(run_body(body, out=out.append, text=True, read_line=scripted("hi Jason")), 10)
    finally:
        await brain.stop()
        vllm.close()

    assert "jason> Hello Josh," in out and "jason> How are you?" in out
    assert requests[0]["messages"][-1] == {"role": "user", "content": "hi Jason"}
    # The brain's raw log holds the whole turn: the body's events and its own.
    [day] = (tmp_path / "brain-state" / "rawlog" / "brain").glob("*.jsonl")
    types = [e.type for e in read_events(day)]
    for t in ("user_final", "agent_action", "reply_chunk", "agent_utterance", "reply_heard"):
        assert t in types
    stages = {e.payload.stage for e in read_events(day) if e.type == "latency_mark"}
    assert {ev.LatencyStage.LLM_FIRST_TOKEN, ev.LatencyStage.FIRST_CLAUSE} <= stages
