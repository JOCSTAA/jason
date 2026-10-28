"""Mac-role entry point: `jason run --role mac`.

In M0 the Mac connects to the brain and sends a ping every few seconds,
printing the round trip. Later milestones replace the pinger with the
microphone, speech-to-text, the floor controller and text-to-speech.
"""

import asyncio
import contextlib
from collections.abc import Callable

from jason.core import clock
from jason.core import events as ev
from jason.core.bus import EventBus, Subscription
from jason.core.config import MacConfig, load_token
from jason.log.rawlog import RawLog
from jason.net.clocksync import sample
from jason.net.link import LinkClient


async def _pong_for(pongs: Subscription, ping_id: str) -> tuple[ev.Pong, int]:
    """Wait for the pong answering this ping; also return when it arrived."""
    while True:
        pong = await pongs.get()
        arrived = clock.monotonic_ns()
        if isinstance(pong, ev.Pong) and pong.payload.ping_id == ping_id:
            return pong, arrived


async def pinger(
    bus: EventBus,
    client: LinkClient,
    count: int | None,
    every_s: float,
    out: Callable[[str], None],
) -> list[float]:
    """Send pings and print each round trip. Returns the round trips in ms."""
    rtts: list[float] = []
    with bus.subscribe(ev.Pong) as pongs:
        seq = 0
        while count is None or seq < count:
            await client.connected.wait()
            seq += 1
            ping = bus.emit(ev.Ping, ev.PingPayload(seq=seq))
            try:
                pong, t4 = await asyncio.wait_for(_pong_for(pongs, ping.id), client.cfg.timeout_s)
            except TimeoutError:
                out(f"ping {seq}: no reply within {client.cfg.timeout_s:g} s")
                continue
            # The same four-timestamp arithmetic as clock sync, applied to the
            # events: bus to bus, so it includes both machines' bus and link time.
            s = sample(ping.ts_monotonic_ns, pong.payload.ping_received_ns, pong.ts_monotonic_ns, t4)
            bus.emit(
                ev.LatencyMark,
                ev.LatencyMarkPayload(stage="ping_rtt", duration_ns=s.rtt_ns, ref_id=ping.id),
            )
            rtts.append(s.rtt_ns / 1e6)
            out(f"ping {seq}: {s.rtt_ns / 1e6:.2f} ms round trip{_offset_note(client)}")
            if count is None or seq < count:
                await asyncio.sleep(every_s)
    return rtts


def _offset_note(client: LinkClient) -> str:
    best = client.clocksync.best
    if best is None:
        return ""
    return f" (brain clock offset {best.offset_ns / 1e6:+.3f} ms ± {best.error_ns / 1e6:.3f})"


async def run_mac(
    cfg: MacConfig,
    count: int | None = None,
    every_s: float = 5.0,
    out: Callable[[str], None] = print,
) -> list[float]:
    """Run until cancelled, or until `count` pings have been answered.

    Raises LinkAuthError / LinkVersionError if the brain can't be talked to.
    """
    token = load_token(cfg.link.token_file)
    with RawLog(cfg.state_dir, cfg.machine) as rawlog:
        bus = EventBus(cfg.machine, rawlog)  # the session comes from the brain
        client = LinkClient(bus, cfg.link, token)
        link = asyncio.create_task(client.run())
        out(f"connecting to {cfg.link.brain_url} ...")

        async def main() -> list[float]:
            await client.ready.wait()
            out(f"connected; session {bus.session_id}")
            bus.emit(ev.System, ev.SystemPayload(kind="startup", message="mac started"))
            return await pinger(bus, client, count, every_s, out)

        work = asyncio.create_task(main())
        try:
            # Whichever ends first: the pings are done, or the link hit an
            # error retrying can't fix (it otherwise runs forever).
            await asyncio.wait({link, work}, return_when=asyncio.FIRST_COMPLETED)
            if link.done():
                link.result()  # re-raises the link's error
            return work.result()
        finally:
            if bus.session_id is not None:
                bus.emit(ev.System, ev.SystemPayload(kind="shutdown", message="mac stopping"))
            # Give the link a moment to deliver the last events to the brain.
            if client.connected.is_set():
                await asyncio.sleep(0.1)
            for task in (work, link):
                task.cancel()
            for task in (work, link):
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            bus.close()
