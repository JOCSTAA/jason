"""A spoken conversation with Jason: `jason run --role body` (M1 step 11).

    mic ─► AudioIO ─► Listener ─► speech_start/end, turn_end_prob
                         └► Transcriber ─► user_partial / user_final
    floor controller ─► agent_action (respond) ─► brain ─► reply_chunk
    Speaker ─► AudioIO ─► speaker ─► reply_heard
    you talk over Jason ─► Speaker stops him (barge-in) ─► reply_cancel, reply_heard

Every part talks through the bus, except the sound itself, which goes
between AudioIO and the parts that read or write samples (decision
2026-10-30). This module only starts them and prints the conversation.
"""

import asyncio
from collections.abc import Callable

from jason.body.audio import AudioIO
from jason.body.floor import FloorController
from jason.body.listen import Listener
from jason.body.stt import Transcriber
from jason.body.stt import make_backend as make_stt
from jason.body.tts import Speaker
from jason.body.tts import make_backend as make_tts
from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import BodyConfig


async def voice_chat(bus: EventBus, cfg: BodyConfig, out: Callable[[str], None] = print) -> None:
    audio = AudioIO(cfg.audio.input_device, cfg.audio.output_device)
    listener = Listener(bus, audio, cfg.audio)
    transcriber = Transcriber(bus, listener, make_stt(cfg.stt), cfg.stt)
    speaker = Speaker(bus, audio, make_tts(cfg.tts), cfg.tts)

    def first_played_ns(action_id: str) -> int | None:
        played = audio.playback.played(action_id)
        return None if played is None else played.first_ns

    floor = FloorController(bus, cfg.floor, jason_speaking=lambda: audio.playback.busy,
                            first_played_ns=first_played_ns)
    out("Loading speech-to-text and text-to-speech ...")
    await transcriber.load()
    await speaker.load()
    audio.start()
    tasks = [asyncio.create_task(c.run()) for c in (listener, transcriber, speaker, floor)]
    out("Talk to Jason (Ctrl+C to stop).")
    try:
        await _print_conversation(bus, out)
    finally:
        for task in tasks:
            task.cancel()
        audio.stop()


async def _print_conversation(bus: EventBus, out: Callable[[str], None]) -> None:
    finals: dict[str, str] = {}  # user_final id -> text
    with bus.subscribe(ev.UserFinal, ev.AgentAction, ev.ReplyChunk, ev.LatencyMark, ev.System) as events:
        async for e in events:
            p = e.payload
            if isinstance(e, ev.UserFinal):
                finals[e.id] = p.text
            elif isinstance(e, ev.AgentAction) and p.action is ev.Action.RESPOND:
                out(f"you> {finals.get(p.based_on_id or '', '?')}")
            elif isinstance(e, ev.ReplyChunk) and p.text:
                out(f"jason> {p.text}")
            elif isinstance(e, ev.LatencyMark) and p.stage is ev.LatencyStage.FIRST_AUDIO:
                out(f"[first audio {p.duration_ns / 1e6:.0f} ms after you stopped]")
            elif isinstance(e, ev.LatencyMark) and p.stage is ev.LatencyStage.BARGE_IN:
                out(f"[you cut in: Jason stopped {p.duration_ns / 1e6:.0f} ms after you started]")
            elif isinstance(e, ev.System) and p.kind == "llm_error":
                out(f"[brain: {p.message}]")
