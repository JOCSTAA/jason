"""The brain's LLM component: turns a `respond` decision into a spoken reply.

    agent_action (respond) ──► prompt ──► vLLM (streamed) ──► reply_chunk × N
                                                         └──► agent_utterance (once)
    reply_cancel ──► stop generating          reply_heard ──► history

The body decides when Jason speaks (floor controller, design §9); this
component only decides what he says. It talks to vLLM's OpenAI-compatible API
and to the rest of Jason only through the bus.

Prompt layout, stable text first so vLLM's prefix cache covers almost all of it
every turn (docs/spikes/m0-llm-serving.md): the persona, then the conversation
so far, then the transcript the decision was based on.
"""

import asyncio
import contextlib
import json
import logging
from collections import deque

import httpx

from jason.core import clock
from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import LlmConfig

log = logging.getLogger(__name__)

# Actions this component writes words for. M5 adds interject and ask.
SPEAKING_ACTIONS = {ev.Action.RESPOND}


class ClauseSplitter:
    """Cuts a reply streamed piece by piece into clauses for text-to-speech.

    Speech can start once the first clause exists, so the first one is cut
    early: at a comma or similar once it has two words ("Sure thing,"; a lone "Well," would sound choppy). Later clauses are
    cut at sentence ends, which sound more natural, unless a sentence runs
    long. A boundary counts only once whitespace follows the punctuation, so
    "3.5" is never split (an abbreviation like "e.g." still is; harmless for
    speech).
    """

    SENTENCE_END = ".!?"
    PAUSE = ",;:"

    def __init__(self, first_min_words: int = 2, long_words: int = 25) -> None:
        self.first_min_words = first_min_words
        self.long_words = long_words
        self._buf = ""
        self._emitted = 0

    def feed(self, piece: str) -> list[str]:
        """Add streamed text; return any clauses now complete."""
        self._buf += piece
        out = []
        while (cut := self._find_cut()) is not None:
            clause, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
            if clause:
                out.append(clause)
                self._emitted += 1
        return out

    def flush(self) -> str:
        """Whatever is left once the stream ends."""
        rest, self._buf = self._buf.strip(), ""
        return rest

    def _find_cut(self) -> int | None:
        for i, ch in enumerate(self._buf[:-1]):
            if not self._buf[i + 1].isspace():
                continue
            words = len(self._buf[: i + 1].split())
            if ch in self.SENTENCE_END:
                return i + 1
            if ch in self.PAUSE:
                short_first = self._emitted == 0 and words >= self.first_min_words
                if short_first or words >= self.long_words:
                    return i + 1
        return None


class LlmComponent:
    def __init__(self, bus: EventBus, cfg: LlmConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.bus = bus
        self.cfg = cfg
        self.persona = cfg.persona_file.read_text(encoding="utf-8").strip()
        timeout = httpx.Timeout(cfg.timeout_s)
        self.client = httpx.AsyncClient(base_url=cfg.base_url, timeout=timeout, transport=transport)
        # The conversation so far, as chat messages. Jason's side is what the
        # user heard (reply_heard), not what was generated: after a barge-in he
        # must not believe he said words the user never heard.
        self.history: deque[dict[str, str]] = deque(maxlen=cfg.history_turns)
        self._transcripts: dict[str, str] = {}  # user_partial/user_final event id -> text
        self._task: asyncio.Task[None] | None = None
        self._action_id: str | None = None  # the reply being generated

    async def run(self) -> None:
        types = (ev.UserPartial, ev.UserFinal, ev.AgentAction, ev.ReplyCancel, ev.ReplyHeard)
        try:
            with self.bus.subscribe(*types) as events:
                async for event in events:
                    self._handle(event)
        finally:
            await self._stop_generating()
            await self.client.aclose()

    def _handle(self, event: ev.Event) -> None:
        match event:
            case ev.UserPartial() | ev.UserFinal():
                self._transcripts[event.id] = event.payload.text
                if len(self._transcripts) > 200:  # only recent ones are ever looked up
                    self._transcripts.pop(next(iter(self._transcripts)))
            case ev.AgentAction() if event.payload.action in SPEAKING_ACTIONS:
                self._start_reply(event)
            case ev.ReplyCancel() if event.payload.action_id == self._action_id and self._task:
                self._task.cancel()
            case ev.ReplyHeard():
                # The only way Jason's words enter the history. A reply the body
                # never reports as heard stays out: Jason can't know what was heard.
                if event.payload.text.strip():
                    self.history.append({"role": "assistant", "content": event.payload.text})

    def _start_reply(self, action: ev.AgentAction) -> None:
        if self._task and not self._task.done():
            self._task.cancel()  # one reply at a time: the newest decision wins

        text = self._transcripts.get(action.payload.based_on_id or "")
        if text is None:
            self._error(action, "no transcript for this decision (based_on_id unknown)")
            return
        messages = [{"role": "system", "content": self.persona}, *self.history, {"role": "user", "content": text}]
        self.history.append({"role": "user", "content": text})
        self._action_id = action.id
        self._task = asyncio.create_task(self._generate(action, messages))

    async def _generate(self, action: ev.AgentAction, messages: list[dict[str, str]]) -> None:
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "stream": True,
        }
        splitter = ClauseSplitter()
        parts: list[str] = []
        index = 0
        complete = False
        t0 = clock.monotonic_ns()

        def chunk(text: str, last: bool) -> None:
            nonlocal index
            self.bus.emit(ev.ReplyChunk, ev.ReplyChunkPayload(action_id=action.id, index=index, text=text, last=last))
            if index == 0:
                self._latency(ev.LatencyStage.FIRST_CLAUSE, t0, action.id)
            index += 1

        try:
            async with self.client.stream("POST", "/v1/chat/completions", json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    piece = _content(line)
                    if not piece:
                        continue
                    if not parts:
                        self._latency(ev.LatencyStage.LLM_FIRST_TOKEN, t0, action.id)
                    parts.append(piece)
                    for clause in splitter.feed(piece):
                        chunk(clause, last=False)
            # Every reply that isn't cancelled ends with a last=True chunk (empty
            # if the final clause already went out), so the body never waits.
            chunk(splitter.flush(), last=True)
            complete = True
        except httpx.HTTPError as e:
            self._error(action, f"{type(e).__name__}: {e}")
            chunk("", last=True)
        finally:
            # Also runs when cancelled (barge-in or a newer decision): the
            # partial reply is logged with complete=False.
            self.bus.emit(
                ev.AgentUtterance,
                ev.AgentUtterancePayload(
                    action_id=action.id, text="".join(parts), complete=complete,
                    model_version=self.cfg.model_version,
                ),
            )
            if self._action_id == action.id:
                self._action_id = None

    def _latency(self, stage: ev.LatencyStage, t0: int, ref_id: str) -> None:
        self.bus.emit(ev.LatencyMark, ev.LatencyMarkPayload(stage=stage, duration_ns=clock.monotonic_ns() - t0, ref_id=ref_id))

    def _error(self, action: ev.AgentAction, message: str) -> None:
        log.warning("reply %s failed: %s", action.id, message)
        self.bus.emit(ev.System, ev.SystemPayload(kind="llm_error", message=message, data={"action_id": action.id}))

    async def _stop_generating(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


def _content(line: str) -> str:
    """The text in one server-sent-events line of a streamed chat completion."""
    if not line.startswith("data: ") or line == "data: [DONE]":
        return ""
    data = json.loads(line[6:])
    return "".join(c.get("delta", {}).get("content") or "" for c in data.get("choices", []))
