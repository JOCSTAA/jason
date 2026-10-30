"""The brain's LLM component, against a fake vLLM (no GPU, no network)."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from jason.brain.llm import ClauseSplitter, LlmComponent
from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import LlmConfig
from jason.log.rawlog import RawLog

SESSION = "day-0001_2026-10-30"
BODY = {"machine": "mac", "session_id": SESSION}


# --- clause splitting -----------------------------------------------------------


def split(pieces, **kw):
    s = ClauseSplitter(**kw)
    out = [c for p in pieces for c in s.feed(p)]
    return out, s.flush()


def test_first_clause_is_cut_early_at_a_comma():
    clauses, rest = split(["Sure thing, ", "I can ", "help with that, and more."])
    assert clauses == ["Sure thing,"]
    # The end of the stream, not a space, finishes the last sentence.
    assert rest == "I can help with that, and more."


def test_too_short_first_clause_waits():
    # "Well," alone would sound choppy: under two words, keep going.
    clauses, rest = split(["Well, ", "I think so."])
    assert clauses == []
    assert rest == "Well, I think so."


def test_later_clauses_end_at_sentences():
    clauses, rest = split(["One two three, four. ", "Five six, seven eight. ", "Nine"])
    assert clauses == ["One two three,", "four.", "Five six, seven eight."]
    assert rest == "Nine"


def test_a_boundary_needs_whitespace_after_it():
    clauses, rest = split(["It runs Qwen 3.", "5 at ", "61 tokens a second. ", "Fast"])
    assert clauses == ["It runs Qwen 3.5 at 61 tokens a second."]
    assert rest == "Fast"


def test_long_sentences_are_cut_at_a_pause():
    long = " ".join(["word"] * 26) + ", "
    clauses, _ = split(["Hi there friend. ", long, "and so on."])
    assert clauses[1] == long.strip()


# --- the component ---------------------------------------------------------------


def sse(*pieces):
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n" for p in pieces]
    return "".join(lines) + "data: [DONE]\n\n"


class FakeVllm:
    """Answers chat completions with a scripted stream; remembers each request."""

    def __init__(self, pieces=("Sure thing, ", "I can help. ", "Anything else?"), delay=0.0, fail=False):
        self.pieces, self.delay, self.fail = pieces, delay, fail
        self.requests = []

    async def handler(self, request):
        self.requests.append(json.loads(request.content))
        if self.fail:
            raise httpx.ConnectError("vLLM is down")

        async def stream():
            for p in self.pieces:
                await asyncio.sleep(self.delay)
                yield sse(p).replace("data: [DONE]\n\n", "").encode()
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, content=stream(), headers={"content-type": "text/event-stream"})


@pytest.fixture
def persona(tmp_path):
    p = tmp_path / "persona.md"
    p.write_text("You are Jason.")
    return p


@pytest.fixture
async def setup(tmp_path, persona):
    started = []

    async def start(fake=None, **cfg):
        fake = fake or FakeVllm()
        rawlog = RawLog(tmp_path, "brain")
        bus = EventBus("brain", rawlog, session_id=SESSION)
        # Warming off unless a test asks for it, so request counts are exact.
        cfg = {"warm_interval_s": 0, **cfg}
        config = LlmConfig(base_url="http://llm:8000", model_version="test-model@1", persona_file=persona, **cfg)
        comp = LlmComponent(bus, config, transport=httpx.MockTransport(fake.handler))
        out = bus.subscribe()  # everything, to inspect
        task = asyncio.create_task(comp.run())
        await asyncio.sleep(0)  # let it subscribe
        started.append((task, rawlog))
        return bus, comp, fake, out

    yield start
    for task, rawlog in started:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        rawlog.close()


def user_says(bus, text, final=True):
    cls, payload = (ev.UserFinal, ev.UserFinalPayload) if final else (ev.UserPartial, ev.UserPartialPayload)
    return bus.publish(cls(**BODY, payload=payload(utterance_id="u1", text=text)))


def respond(bus, transcript):
    return bus.publish(ev.AgentAction(**BODY, payload=ev.AgentActionPayload(
        action=ev.Action.RESPOND, action_probs={ev.Action.RESPOND: 1.0},
        utterance_id="u1", based_on_id=transcript.id,
    )))


async def collect(out, until, timeout=3.0):
    """Events from the subscription until one matches `until`."""
    seen = []
    async with asyncio.timeout(timeout):
        while True:
            e = await out.get()
            seen.append(e)
            if until(e):
                return seen


def of(seen, cls):
    return [e for e in seen if isinstance(e, cls)]


async def test_a_reply_streams_as_clauses_then_one_utterance(setup):
    bus, comp, fake, out = await setup()
    action = respond(bus, user_says(bus, "Can you help?"))
    seen = await collect(out, lambda e: isinstance(e, ev.AgentUtterance))

    chunks = of(seen, ev.ReplyChunk)
    assert [c.payload.text for c in chunks] == ["Sure thing,", "I can help.", "Anything else?"]
    assert [c.payload.index for c in chunks] == [0, 1, 2]
    assert [c.payload.last for c in chunks] == [False, False, True]
    assert all(c.payload.action_id == action.id for c in chunks)
    [utt] = of(seen, ev.AgentUtterance)
    assert utt.payload.text == "Sure thing, I can help. Anything else?"
    assert (utt.payload.complete, utt.payload.model_version) == (True, "test-model@1")


async def test_first_token_and_first_clause_latency_are_logged(setup):
    bus, comp, fake, out = await setup()
    action = respond(bus, user_says(bus, "Hi"))
    seen = await collect(out, lambda e: isinstance(e, ev.AgentUtterance))
    marks = {m.payload.stage: m for m in of(seen, ev.LatencyMark)}
    assert set(marks) == {ev.LatencyStage.LLM_FIRST_TOKEN, ev.LatencyStage.FIRST_CLAUSE}
    assert all(m.payload.ref_id == action.id for m in marks.values())


async def test_prompt_is_persona_then_history_then_the_transcript_it_saw(setup):
    bus, comp, fake, out = await setup()
    first = respond(bus, user_says(bus, "Hello"))
    await collect(out, lambda e: isinstance(e, ev.AgentUtterance))
    bus.publish(ev.ReplyHeard(**BODY, payload=ev.ReplyHeardPayload(
        action_id=first.id, text="Sure thing, I can help.", complete=False, played_ms=900.0, interrupted_by="u2")))
    # The decision saw a partial transcript; that exact text is what's answered.
    partial = user_says(bus, "What about", final=False)
    user_says(bus, "What about tomorrow?")
    respond(bus, partial)
    await collect(out, lambda e: isinstance(e, ev.AgentUtterance))

    msgs = fake.requests[-1]["messages"]
    assert msgs == [
        {"role": "system", "content": "You are Jason."},
        {"role": "user", "content": "Hello"},
        # What the user heard, not the full generated reply.
        {"role": "assistant", "content": "Sure thing, I can help."},
        {"role": "user", "content": "What about"},
    ]
    assert fake.requests[-1]["stream"] is True


async def test_history_is_capped(setup):
    bus, comp, fake, out = await setup(history_turns=2)
    for i in range(3):
        respond(bus, user_says(bus, f"turn {i}"))
        await collect(out, lambda e: isinstance(e, ev.AgentUtterance))
    assert len(comp.history) == 2


async def test_cancel_stops_the_reply_and_logs_it_incomplete(setup):
    bus, comp, fake, out = await setup(FakeVllm(pieces=["Sure thing, "] + ["more words "] * 50, delay=0.02))
    action = respond(bus, user_says(bus, "Tell me a story"))
    await collect(out, lambda e: isinstance(e, ev.ReplyChunk))  # it has started
    bus.publish(ev.ReplyCancel(**BODY, payload=ev.ReplyCancelPayload(action_id=action.id, reason="barge_in")))
    seen = await collect(out, lambda e: isinstance(e, ev.AgentUtterance))
    [utt] = of(seen, ev.AgentUtterance)
    assert utt.payload.complete is False
    assert utt.payload.text.startswith("Sure thing,")
    assert not any(c.payload.last for c in of(seen, ev.ReplyChunk))


async def test_a_new_decision_replaces_the_reply_in_progress(setup):
    bus, comp, fake, out = await setup(FakeVllm(pieces=["Sure thing, "] + ["word "] * 20, delay=0.02))
    old = respond(bus, user_says(bus, "First"))
    await collect(out, lambda e: isinstance(e, ev.ReplyChunk))
    new = respond(bus, user_says(bus, "Actually, second"))
    seen = await collect(out, lambda e: isinstance(e, ev.AgentUtterance) and e.payload.action_id == new.id, timeout=5)
    by_action = {u.payload.action_id: u.payload.complete for u in of(seen, ev.AgentUtterance)}
    assert by_action == {old.id: False, new.id: True}


async def test_llm_down_is_reported_and_the_reply_still_ends(setup):
    bus, comp, fake, out = await setup(FakeVllm(fail=True))
    action = respond(bus, user_says(bus, "Hello?"))
    seen = await collect(out, lambda e: isinstance(e, ev.AgentUtterance))
    [err] = [e for e in of(seen, ev.System) if e.payload.kind == "llm_error"]
    assert err.payload.data["action_id"] == action.id
    [last] = of(seen, ev.ReplyChunk)
    assert (last.payload.text, last.payload.last) == ("", True)
    assert of(seen, ev.AgentUtterance)[0].payload.complete is False


async def test_unknown_transcript_is_an_error_not_a_guess(setup):
    bus, comp, fake, out = await setup()
    bus.publish(ev.AgentAction(**BODY, payload=ev.AgentActionPayload(
        action=ev.Action.RESPOND, action_probs={ev.Action.RESPOND: 1.0}, based_on_id="nope")))
    seen = await collect(out, lambda e: isinstance(e, ev.System) and e.payload.kind == "llm_error")
    assert fake.requests == []


async def test_silent_decisions_are_ignored(setup):
    bus, comp, fake, out = await setup()
    t = user_says(bus, "Hmm")
    bus.publish(ev.AgentAction(**BODY, payload=ev.AgentActionPayload(
        action=ev.Action.SILENT, action_probs={ev.Action.SILENT: 1.0}, based_on_id=t.id)))
    await asyncio.sleep(0.05)
    assert fake.requests == []


# --- keeping the GPU awake --------------------------------------------------------


def warmups(fake):
    return [r for r in fake.requests if r.get("max_tokens") == 1]


async def test_no_warmups_before_the_user_shows_up(setup):
    bus, comp, fake, out = await setup(warm_interval_s=0.02, warm_window_s=1.0)
    await asyncio.sleep(0.15)
    assert warmups(fake) == []


async def test_activity_keeps_the_gpu_warm_then_it_stops(setup):
    bus, comp, fake, out = await setup(warm_interval_s=0.02, warm_window_s=0.2)
    bus.publish(ev.SpeechStart(**BODY, payload=ev.SpeechStartPayload(utterance_id="u1", audio_ns=1, during_agent_speech=False)))
    await asyncio.sleep(0.15)
    during = len(warmups(fake))
    assert during >= 3
    assert warmups(fake)[0]["messages"][0] == {"role": "system", "content": "You are Jason."}  # keeps the persona cached too
    await asyncio.sleep(0.3)  # the window has passed
    settled = len(warmups(fake))
    await asyncio.sleep(0.15)
    assert len(warmups(fake)) == settled


async def test_no_warmups_while_a_reply_is_generating(setup):
    bus, comp, fake, out = await setup(FakeVllm(pieces=["Sure thing, "] + ["word "] * 15, delay=0.02),
                                       warm_interval_s=0.02, warm_window_s=5.0)
    respond(bus, user_says(bus, "Tell me something"))
    await collect(out, lambda e: isinstance(e, ev.ReplyChunk))
    before = len(warmups(fake))
    await asyncio.sleep(0.15)  # still generating
    assert len(warmups(fake)) <= before + 1  # at most one that was already on its way
    await collect(out, lambda e: isinstance(e, ev.AgentUtterance))


async def test_warmup_failures_are_ignored(setup):
    bus, comp, fake, out = await setup(FakeVllm(fail=True), warm_interval_s=0.02, warm_window_s=1.0)
    user_says(bus, "hello")
    await asyncio.sleep(0.1)
    assert len(warmups(fake)) >= 2  # kept trying, didn't crash
