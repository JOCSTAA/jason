"""The brain's LLM component: turns a `respond` decision into a spoken reply.

    user_final (a pause) ──► draft: prompt ──► vLLM (streamed) ──► clauses held
    agent_action (respond to that transcript) ──► release ──► reply_chunk × N, agent_utterance (once)
    the user carries on (a newer user_final) ──► the draft is thrown away
    reply_cancel ──► stop generating          reply_heard ──► history
    any sign of the user ──► keep the GPU awake for a while

The body decides when Jason speaks (floor controller, design §9); this
component only decides what he says. It starts writing at each pause, before
the body decides, so the floor controller's wait and the writing overlap
instead of adding up: the first half of design §9's "the brain offers, the
body decides" (M5 adds drafting from partial transcripts and offers). It talks to vLLM's OpenAI-compatible API
and to the rest of Jason only through the bus.

Prompt layout, stable text first so vLLM's prefix cache covers almost all of it
every turn (docs/spikes/m0-llm-serving.md): the persona, then the conversation
so far, then the transcript the decision was based on.

Keeping the GPU awake: on the 4090 Laptop vLLM's first token takes ~40 ms
while the GPU is busy but ~380 ms after 2 s or more of idling (it falls asleep
within 1-2 s; scripts/llm_latency_probe.py). So while a conversation is active
a one-token request goes out every warm_interval_s. This is about the GPU's
power state, not vLLM's cache: the prompt's prefix stays cached either way.
"""

import asyncio
import contextlib
import json
import logging
from collections import deque
from dataclasses import dataclass, field

import httpx

from jason.core import clock
from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import LlmConfig

log = logging.getLogger(__name__)

# Actions this component writes words for. M5 adds interject and ask.
SPEAKING_ACTIONS = {ev.Action.RESPOND}

# Appended to a reply in the history when the user cut Jason off (barge-in);
# the persona tells the model what it means and never to write it.
INTERRUPTED = "[interrupted by Josh]"


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


@dataclass(eq=False)
class _Gen:
    """One generation. It starts as a draft, when a transcript at a pause
    arrives, and is held; it becomes Jason's reply when the body decides to
    respond to that transcript (released), or is thrown away if the user
    carries on. A decision with no draft waiting starts one released."""

    final_id: str  # the transcript it answers
    text: str
    messages: list[dict[str, str]]
    started_ns: int = field(default_factory=clock.monotonic_ns)
    action: ev.AgentAction | None = None  # set when released
    released_ns: int | None = None
    first_token_ns: int | None = None
    first_clause_ns: int | None = None
    parts: list[str] = field(default_factory=list)
    held: list[tuple[str, bool]] = field(default_factory=list)  # clauses not sent yet
    index: int = 0
    finished: bool = False  # the stream ended (fully, or in an error)
    complete: bool = False
    error: str | None = None
    logged: bool = False  # its agent_utterance has been emitted
    marked: set[ev.LatencyStage] = field(default_factory=set)  # waits already logged
    task: asyncio.Task[None] | None = None


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
        self._draft: _Gen | None = None  # written ahead, waiting for the body's decision
        self._reply: _Gen | None = None  # released: Jason's current reply
        self._last_activity: int | None = None  # monotonic ns of the user's last sign of life

    async def run(self) -> None:
        types = (ev.SpeechStart, ev.UserPartial, ev.UserFinal, ev.AgentAction, ev.ReplyCancel, ev.ReplyHeard)
        warmer = asyncio.create_task(self._keep_warm()) if self.cfg.warm_interval_s > 0 else None
        try:
            with self.bus.subscribe(*types) as events:
                async for event in events:
                    self._handle(event)
        finally:
            if warmer:
                warmer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await warmer
            for gen in (self._draft, self._reply):
                await self._stop(gen)
            await self.client.aclose()

    def _handle(self, event: ev.Event) -> None:
        if isinstance(event, ev.SpeechStart | ev.UserPartial | ev.UserFinal | ev.ReplyHeard):
            self._last_activity = clock.monotonic_ns()
        match event:
            case ev.UserPartial():
                self._remember(event)
            case ev.UserFinal():
                self._remember(event)
                self._start_draft(event)
            case ev.AgentAction() if event.payload.action in SPEAKING_ACTIONS:
                self._start_reply(event)
            case ev.ReplyCancel() if self._reply and self._reply.action and event.payload.action_id == self._reply.action.id:
                if self._reply.task and not self._reply.task.done():
                    self._reply.task.cancel()
            case ev.ReplyHeard():
                # The only way Jason's words enter the history. A reply the body
                # never reports as heard stays out: Jason can't know what was heard.
                text = event.payload.text.strip()
                if event.payload.interrupted_by:
                    # Mark a cut-off turn as cut off. Unmarked, the model copied
                    # its own interrupted turns and began ending replies
                    # mid-sentence by itself (first barge-in test, M1 step 12).
                    text = f"{text} {INTERRUPTED}".strip()
                if text:
                    self.history.append({"role": "assistant", "content": text})

    def _remember(self, event: ev.Event) -> None:
        self._transcripts[event.id] = event.payload.text
        if len(self._transcripts) > 200:  # only recent ones are ever looked up
            self._transcripts.pop(next(iter(self._transcripts)))

    def _messages(self, text: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.persona}, *self.history, {"role": "user", "content": text}]

    def _start_draft(self, final: ev.UserFinal) -> None:
        """Start writing the answer to a transcript at a pause before the body
        decides whether to answer it (a draft; decision 2026-10-30)."""
        self._discard_draft()  # the user carried on: the older draft answers words that are no longer the turn
        gen = _Gen(final.id, final.payload.text, self._messages(final.payload.text))
        gen.task = asyncio.create_task(self._generate(gen))
        self._draft = gen

    def _discard_draft(self) -> None:
        if self._draft is not None:
            if self._draft.task and not self._draft.task.done():
                self._draft.task.cancel()
            self._draft = None

    def _start_reply(self, action: ev.AgentAction) -> None:
        if self._reply and self._reply.task and not self._reply.task.done():
            self._reply.task.cancel()  # one reply at a time: the newest decision wins
        draft, self._draft = self._draft, None
        if draft is not None and draft.final_id == action.payload.based_on_id:
            gen = draft  # written ahead: release what's ready at once
        else:
            if draft is not None and draft.task and not draft.task.done():
                draft.task.cancel()
            text = self._transcripts.get(action.payload.based_on_id or "")
            if text is None:
                self._error(action, "no transcript for this decision (based_on_id unknown)")
                return
            gen = _Gen(action.payload.based_on_id or "", text, self._messages(text))
            gen.task = asyncio.create_task(self._generate(gen))
        self.history.append({"role": "user", "content": gen.text})
        self._reply = gen
        self._release(gen, action)

    def _release(self, gen: _Gen, action: ev.AgentAction) -> None:
        gen.action, gen.released_ns = action, clock.monotonic_ns()
        self._mark_waits(gen)
        held, gen.held = gen.held, []
        for text, last in held:
            self._send(gen, text, last)
        if gen.finished:
            self._log_utterance(gen)

    async def _generate(self, gen: _Gen) -> None:
        body = {
            "model": self.cfg.model,
            "messages": gen.messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "stream": True,
        }
        splitter = ClauseSplitter()
        try:
            async with self.client.stream("POST", "/v1/chat/completions", json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    piece = _content(line)
                    if not piece:
                        continue
                    if gen.first_token_ns is None:
                        gen.first_token_ns = clock.monotonic_ns()
                        self._mark_waits(gen)
                    gen.parts.append(piece)
                    for clause in splitter.feed(piece):
                        self._clause(gen, clause, last=False)
            # Every reply that isn't cancelled ends with a last=True chunk (empty
            # if the final clause already went out), so the body never waits.
            self._clause(gen, splitter.flush(), last=True)
            gen.complete = True
        except httpx.HTTPError as e:
            gen.error = f"{type(e).__name__}: {e}"
            if gen.action is not None:
                self._error(gen.action, gen.error)
            self._clause(gen, "", last=True)
        finally:
            gen.finished = True
            # Also runs when cancelled (barge-in, a newer decision, or a
            # discarded draft): a released reply is logged with complete=False;
            # a draft nobody answered with leaves no trace.
            if gen.action is not None:
                self._log_utterance(gen)

    def _clause(self, gen: _Gen, text: str, last: bool) -> None:
        if gen.first_clause_ns is None and (text or last):
            gen.first_clause_ns = clock.monotonic_ns()
            self._mark_waits(gen)
        if gen.action is None:
            gen.held.append((text, last))
        else:
            self._send(gen, text, last)

    def _send(self, gen: _Gen, text: str, last: bool) -> None:
        assert gen.action is not None
        self.bus.emit(ev.ReplyChunk, ev.ReplyChunkPayload(action_id=gen.action.id, index=gen.index, text=text, last=last))
        gen.index += 1

    def _mark_waits(self, gen: _Gen) -> None:
        """How long the decision waited for the first token and the first
        clause: 0 when the draft was already that far, the generation time
        when the decision came first (schema 5)."""
        if gen.action is None:
            return
        for stage, at in ((ev.LatencyStage.LLM_FIRST_TOKEN, gen.first_token_ns),
                          (ev.LatencyStage.FIRST_CLAUSE, gen.first_clause_ns)):
            if at is not None and stage not in gen.marked:
                gen.marked.add(stage)
                wait = max(0, at - max(gen.started_ns, gen.released_ns or 0))
                self.bus.emit(ev.LatencyMark, ev.LatencyMarkPayload(stage=stage, duration_ns=wait, ref_id=gen.action.id))

    def _log_utterance(self, gen: _Gen) -> None:
        if gen.logged or gen.action is None:
            return
        gen.logged = True
        self.bus.emit(ev.AgentUtterance, ev.AgentUtterancePayload(
            action_id=gen.action.id, text="".join(gen.parts), complete=gen.complete,
            model_version=self.cfg.model_version))

    async def _keep_warm(self) -> None:
        """While the user was active within warm_window_s, send a one-token
        request every warm_interval_s, unless a generation is keeping the GPU
        busy already. Failures are ignored: it's only a nudge."""
        body = {"model": self.cfg.model, "max_tokens": 1, "temperature": 0,
                "messages": [{"role": "system", "content": self.persona}, {"role": "user", "content": "."}]}
        window_ns = int(self.cfg.warm_window_s * 1e9)
        while True:
            await asyncio.sleep(self.cfg.warm_interval_s)
            active = self._last_activity is not None and clock.monotonic_ns() - self._last_activity < window_ns
            generating = any(g is not None and g.task is not None and not g.task.done() for g in (self._draft, self._reply))
            if active and not generating:
                try:
                    await self.client.post("/v1/chat/completions", json=body)
                except httpx.HTTPError as e:
                    log.debug("warm-up request failed: %s", e)

    def _error(self, action: ev.AgentAction, message: str) -> None:
        log.warning("reply %s failed: %s", action.id, message)
        self.bus.emit(ev.System, ev.SystemPayload(kind="llm_error", message=message, data={"action_id": action.id}))

    @staticmethod
    async def _stop(gen: _Gen | None) -> None:
        if gen is not None and gen.task and not gen.task.done():
            gen.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await gen.task


def _content(line: str) -> str:
    """The text in one server-sent-events line of a streamed chat completion."""
    if not line.startswith("data: ") or line == "data: [DONE]":
        return ""
    data = json.loads(line[6:])
    return "".join(c.get("delta", {}).get("content") or "" for c in data.get("choices", []))
