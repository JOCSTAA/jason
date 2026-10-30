"""Brain-role entry point: `jason run --role brain`.

The brain loads which day it is, accepts the body's link, answers every ping
from the body with a pong, and (with an [llm] section in its config) writes
Jason's replies. Later milestones add memory, the learner and Wonder as more
components on the same bus.
"""

import asyncio
import contextlib
from collections.abc import Callable

from jason.brain.llm import LlmComponent
from jason.core import clock
from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import BrainConfig, load_token
from jason.core.state import StateStore
from jason.log.rawlog import RawLog
from jason.net.link import LinkServer


async def ping_responder(bus: EventBus) -> None:
    """Answer each ping from the other machine with a pong."""
    with bus.subscribe(ev.Ping) as pings:
        async for ping in pings:
            received = clock.monotonic_ns()
            if ping.machine == bus.machine:
                continue  # our own ping; only the other side's need an answer
            bus.emit(
                ev.Pong,
                ev.PongPayload(ping_id=ping.id, seq=ping.payload.seq, ping_received_ns=received),
            )


class Brain:
    def __init__(self, cfg: BrainConfig) -> None:
        self.cfg = cfg
        self._store = StateStore(cfg.state_dir)
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        token = load_token(self.cfg.link.token_file)
        state = self._store.load_or_create()
        self._rawlog = RawLog(self.cfg.state_dir, self.cfg.machine)
        self.bus = EventBus(self.cfg.machine, self._rawlog, session_id=state.session_id)
        self.server = LinkServer(self.bus, self.cfg.link, token, commands={"new_day": self.new_day})
        await self.server.start()
        self._tasks.append(asyncio.create_task(ping_responder(self.bus)))
        if self.cfg.llm is not None:
            self._tasks.append(asyncio.create_task(LlmComponent(self.bus, self.cfg.llm).run()))
        self.bus.emit(
            ev.System,
            ev.SystemPayload(kind="startup", message="brain started", data={"day": state.day, "port": self.port}),
        )

    @property
    def port(self) -> int:
        return self.server.port

    async def new_day(self) -> str:
        """End today's session and start the next one; the body is told too."""
        state = self._store.new_day()
        await self.server.set_session(state.session_id)
        self.bus.emit(
            ev.System,
            ev.SystemPayload(kind="new_day", message=f"day {state.day} started", data={"day": state.day}),
        )
        return f"started {state.session_id}"

    async def stop(self) -> None:
        self.bus.emit(ev.System, ev.SystemPayload(kind="shutdown", message="brain stopping"))
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self.server.close()
        self.bus.close()
        self._rawlog.close()


async def run_brain(cfg: BrainConfig, out: Callable[[str], None] = print) -> None:
    """Run until cancelled (Ctrl+C)."""
    brain = Brain(cfg)
    await brain.start()
    out(f"brain listening on {cfg.link.listen_host}:{brain.port}, session {brain.bus.session_id}")
    try:
        await asyncio.Event().wait()  # wait forever; components run as tasks
    finally:
        await brain.stop()
