"""WebSocket server and client that bridge events between the two buses.

    BODY                                                 BRAIN
  [bus] ⇄ LinkClient ══ WebSocket (over Tailscale) ══ LinkServer ⇄ [bus]

A WebSocket is a connection that stays open so either side can send a
message at any moment. The brain listens (server); the body dials in (client).

- Sending: each side subscribes to everything on its own bus and sends the
  events this machine created. Events that came from the other machine are
  never sent back, or every event would bounce between the two forever.
- Receiving: events from the other side are published on the local bus, so
  they are logged here too and reach local subscribers.
- Auth: the body sends the shared token in the handshake's Authorization
  header; the brain refuses the connection (HTTP 401) before it opens if the
  token is wrong or missing.
- Heartbeat: websockets' built-in pings. No reply within timeout_s means the
  connection is dead, and the body starts reconnecting.
- While disconnected, this machine's events wait in the link's bus
  subscription (up to 1000, oldest dropped first) and are sent after
  reconnecting. Each machine's raw log has them either way.
- Admin: the brain also accepts short-lived connections at the /admin path
  (same token) that send one command, e.g. "new_day" from `jason new-day`.
  They don't replace the body's connection.
- Clock sync: every clock_sync_s the body sends a probe and the brain answers
  with its own timestamps (net/clocksync.py). Each round trip is logged as a
  latency_mark with stage "network_rtt"; the offset estimate is logged as a
  "clock_offset" system event when it changes by more than 1 ms, or once a
  minute. Both reach the brain's log too, like any body event.
"""

import asyncio
import contextlib
import hmac
import logging
import random
from collections.abc import Awaitable, Callable, Mapping
from http import HTTPStatus

from pydantic import ValidationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.http11 import Request, Response

from jason.core import clock
from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import BrainLink, BodyLink
from jason.core.events import SCHEMA_VERSION
from jason.net.clocksync import ClockSync, Sample
from jason.net.protocol import (
    PROTOCOL_VERSION,
    AnyMessage,
    Command,
    CommandResult,
    EventMsg,
    Hello,
    NewSession,
    Sync,
    SyncReply,
    decode,
    encode,
)

log = logging.getLogger(__name__)

_FIRST_RETRY_S = 0.5
ADMIN_PATH = "/admin"
_OFFSET_LOG_CHANGE_NS = 1_000_000  # log a new offset estimate when it moves by 1 ms...
_OFFSET_LOG_EVERY_NS = 60 * 1_000_000_000  # ...or at least once a minute
# At most one link_rejected event per window, so a flood of bad connections
# can't fill the raw log; the next one says how many were left out.
REJECTED_LOG_EVERY_NS = 10 * 1_000_000_000


class LinkAuthError(Exception):
    """The brain rejected the token. Retrying won't help; fix the token file."""


class LinkVersionError(Exception):
    """The two machines run incompatible versions of Jason."""


class _NoHello(Exception):
    """The server didn't start with a hello; treated like any failed connection."""


def auth_header(token: str) -> str:
    return f"Bearer {token}"


def _system(bus: EventBus, kind: str, message: str = "", **data: str | int | float | bool | None) -> None:
    """Log a link status event on the bus, or to Python logging if there is no session yet."""
    if bus.session_id is None:
        log.info("%s %s %s", kind, message, data)
        return
    bus.emit(ev.System, ev.SystemPayload(kind=kind, message=message, data=data))


class _Bridge:
    """The part both sides share: forward this machine's events, publish the peer's."""

    def __init__(self, bus: EventBus, outbox_size: int = 1000) -> None:
        self.bus = bus
        # This machine's own events only (the other machine's are never sent
        # back). Also the waiting area while disconnected: when it is full the
        # oldest are dropped, counted in self._outbox.dropped.
        self._outbox = bus.subscribe(maxsize=outbox_size, where=lambda e: e.machine == bus.machine)
        self._reported_drops = 0
        # An event taken from the outbox whose send failed; sent first next time.
        self._pending: ev.Event | None = None
        # Last seq received per (machine, run_id), to spot gaps.
        self._last_seq: dict[tuple[str, str], int] = {}

    def report_drops(self) -> None:
        """Log how many of this machine's events the outbox dropped since the last report."""
        dropped = self._outbox.dropped - self._reported_drops
        if dropped:
            self._reported_drops = self._outbox.dropped
            _system(self.bus, "link_backlog_dropped", "outbox full while disconnected", count=dropped)

    async def pump_out(self, ws: ClientConnection | ServerConnection) -> None:
        while True:
            if self._pending is None:
                try:
                    self._pending = await self._outbox.get()
                except StopAsyncIteration:
                    return  # the bus is shutting down
            await ws.send(encode(EventMsg(event=self._pending)))
            self._pending = None

    def receive_event(self, event: ev.Event) -> None:
        if event.machine == self.bus.machine:
            log.warning("dropped an incoming event that claims to be from this machine: %s", event.id)
            return
        self._check_seq(event)
        self.bus.publish(event)

    def _check_seq(self, event: ev.Event) -> None:
        # A jump in seq within one run = events the other machine published that
        # never arrived. The first event seen from a run isn't checked: this side
        # may have restarted and can't know what it saw before.
        if event.seq is None or event.run_id is None:
            return
        key = (event.machine, event.run_id)
        last = self._last_seq.get(key)
        if last is not None and event.seq > last + 1:
            _system(
                self.bus,
                "link_gap",
                f"{event.seq - last - 1} events from {event.machine} never arrived",
                machine=event.machine,
                run_id=event.run_id,
                expected=last + 1,
                got=event.seq,
                missing=event.seq - last - 1,
            )
        self._last_seq[key] = max(event.seq, last if last is not None else event.seq)

    def close(self) -> None:
        self._outbox.close()


def _parse(raw: str | bytes) -> AnyMessage | None:
    try:
        return decode(raw)
    except ValidationError as e:
        log.warning("ignored a malformed link message: %s", e.errors()[:1])
        return None


# --- brain side ---------------------------------------------------------------


class LinkServer:
    """Runs on the brain. Accepts one body at a time; a newer connection replaces an older one."""

    def __init__(
        self,
        bus: EventBus,
        cfg: BrainLink,
        token: str,
        commands: Mapping[str, Callable[[], Awaitable[str]]] | None = None,
        outbox_size: int = 1000,
    ) -> None:
        self.bus = bus
        self.cfg = cfg
        self._expected = auth_header(token).encode()
        # Admin commands by name; each returns a message for the user.
        self._commands = dict(commands or {})
        self._rejected_logged_at: int | None = None
        self._rejected_suppressed = 0
        self._bridge = _Bridge(bus, outbox_size)
        self._server: Server | None = None
        self._ws: ServerConnection | None = None  # the body currently connected
        self._latest: ServerConnection | None = None  # the newest connection to arrive
        self._active = asyncio.Lock()  # held by whichever connection is being served

    @property
    def port(self) -> int:
        """The port actually listened on (useful when the config asks for port 0)."""
        assert self._server is not None
        return self._server.sockets[0].getsockname()[1]

    async def start(self) -> None:
        self._server = await serve(
            self._handle,
            self.cfg.listen_host,
            self.cfg.listen_port,
            process_request=self._check_token,
            ping_interval=self.cfg.heartbeat_s,
            ping_timeout=self.cfg.timeout_s,
            compression=None,  # events are small; compressing them only adds delay
        )

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self._bridge.close()

    async def set_session(self, session_id: str) -> None:
        """Start using a new session (a new day) and tell the body."""
        self.bus.session_id = session_id
        if self._ws is not None:
            with contextlib.suppress(ConnectionClosed):
                await self._ws.send(encode(NewSession(session_id=session_id)))

    def _check_token(self, conn: ServerConnection, request: Request) -> Response | None:
        # Runs during the handshake, before the connection opens.
        given = request.headers.get("Authorization", "").encode()
        # compare_digest takes the same time wherever the first wrong character
        # is, so the token can't be guessed one character at a time by timing.
        if hmac.compare_digest(given, self._expected):
            return None  # accept
        self._log_rejection(str(conn.remote_address))
        return conn.respond(HTTPStatus.UNAUTHORIZED, "bad or missing link token\n")

    def _log_rejection(self, peer: str) -> None:
        now = clock.monotonic_ns()
        if self._rejected_logged_at is not None and now - self._rejected_logged_at < REJECTED_LOG_EVERY_NS:
            self._rejected_suppressed += 1
            return
        _system(
            self.bus,
            "link_rejected",
            "bad or missing link token",
            peer=peer,
            suppressed=self._rejected_suppressed,  # rejections left out since the last event
        )
        self._rejected_logged_at = now
        self._rejected_suppressed = 0

    async def _handle(self, ws: ServerConnection) -> None:
        if ws.request is not None and ws.request.path.rstrip("/") == ADMIN_PATH:
            await self._admin(ws)
            return
        self._latest = ws
        if self._ws is not None:
            await self._ws.close(reason="replaced by a newer connection")
        async with self._active:  # wait until the old connection has fully stopped
            if self._latest is not ws:
                await ws.close(reason="replaced by a newer connection")
                return
            self._ws = ws
            try:
                await self._serve(ws)
            finally:
                self._ws = None

    async def _admin(self, ws: ServerConnection) -> None:
        """Run one command from `jason new-day` (or similar) and reply with the result."""
        try:
            msg = _parse(await asyncio.wait_for(ws.recv(), self.cfg.timeout_s))
        except (TimeoutError, ConnectionClosed):
            return
        if not isinstance(msg, Command):
            result = CommandResult(ok=False, message="expected a command")
        elif (handler := self._commands.get(msg.name)) is None:
            result = CommandResult(ok=False, message=f"unknown command {msg.name!r}")
        else:
            try:
                result = CommandResult(ok=True, message=await handler())
            except Exception as e:  # report it to the user instead of dropping the connection
                log.exception("admin command %s failed", msg.name)
                result = CommandResult(ok=False, message=f"{msg.name} failed: {e}")
        name = msg.name if isinstance(msg, Command) else "?"
        _system(self.bus, "admin_command", result.message, name=name, ok=result.ok)
        with contextlib.suppress(ConnectionClosed):
            await ws.send(encode(result))

    async def _serve(self, ws: ServerConnection) -> None:
        assert self.bus.session_id is not None, "the brain needs a session before accepting the body"
        await ws.send(encode(Hello(machine=self.bus.machine, session_id=self.bus.session_id)))
        _system(self.bus, "link_up", peer=str(ws.remote_address))
        self._bridge.report_drops()
        pump = asyncio.create_task(self._bridge.pump_out(ws))
        reason = "closed"
        try:
            async for raw in ws:
                received = clock.monotonic_ns()  # t2 for a sync probe: as early as possible
                msg = _parse(raw)
                if isinstance(msg, EventMsg):
                    self._bridge.receive_event(msg.event)
                elif isinstance(msg, Sync):
                    reply = SyncReply(t1_ns=msg.t1_ns, t2_ns=received, t3_ns=clock.monotonic_ns())
                    await ws.send(encode(reply))
                elif msg is not None:
                    log.warning("ignored a %s message from the body", msg.kind)
        except ConnectionClosed as e:
            reason = str(e)
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
                await pump
            _system(self.bus, "link_down", reason, peer=str(ws.remote_address))


# --- Body side ----------------------------------------------------------------


class LinkClient:
    """Runs on the body. Connects to the brain and keeps reconnecting until stopped."""

    def __init__(self, bus: EventBus, cfg: BodyLink, token: str, outbox_size: int = 1000) -> None:
        self.bus = bus
        self.cfg = cfg
        self._token = token
        self._bridge = _Bridge(bus, outbox_size)
        # Set once the brain has told us the session, i.e. once emit() works.
        self.ready = asyncio.Event()
        self.connected = asyncio.Event()
        # Offset between the body's and the brain's monotonic clocks.
        self.clocksync = ClockSync()
        self._logged_offset: tuple[int, int] | None = None  # (offset_ns, when logged)

    async def run(self) -> None:
        """Connect, serve, and reconnect forever. Cancel the task to stop.

        Raises LinkAuthError or LinkVersionError, which retrying can't fix.
        """
        delay = _FIRST_RETRY_S
        try:
            while True:
                try:
                    async with connect(
                        self.cfg.brain_url,
                        additional_headers={"Authorization": auth_header(self._token)},
                        open_timeout=self.cfg.timeout_s,
                        ping_interval=self.cfg.heartbeat_s,
                        ping_timeout=self.cfg.timeout_s,
                        compression=None,
                    ) as ws:
                        await self._greet(ws)
                        delay = _FIRST_RETRY_S
                        await self._serve(ws)
                except InvalidStatus as e:
                    if e.response.status_code == HTTPStatus.UNAUTHORIZED:
                        raise LinkAuthError(
                            "the brain rejected the link token; both machines need the same token"
                        ) from None
                    log.warning("brain refused the connection: HTTP %s", e.response.status_code)
                except (OSError, TimeoutError, ConnectionClosed, _NoHello) as e:
                    log.info("can't reach the brain at %s: %s", self.cfg.brain_url, e)
                # Wait longer after each failure (0.5 s, 1 s, 2 s, ... up to the
                # configured maximum), plus a little randomness.
                await asyncio.sleep(delay * random.uniform(1.0, 1.2))
                delay = min(delay * 2, self.cfg.reconnect_max_s)
        finally:
            self._bridge.close()

    async def _greet(self, ws: ClientConnection) -> None:
        msg = _parse(await asyncio.wait_for(ws.recv(), timeout=self.cfg.timeout_s))
        if not isinstance(msg, Hello):
            raise _NoHello(f"expected hello, got {msg!r}")
        if (msg.protocol_version, msg.schema_version) != (PROTOCOL_VERSION, SCHEMA_VERSION):
            raise LinkVersionError(
                f"brain runs protocol {msg.protocol_version} / schema {msg.schema_version}, "
                f"this body runs protocol {PROTOCOL_VERSION} / schema {SCHEMA_VERSION}; "
                "update both machines to the same version of Jason"
            )
        if msg.machine == self.bus.machine:
            raise LinkVersionError(f"the brain is also named {msg.machine!r}; machine names must differ")
        self.bus.session_id = msg.session_id
        self.ready.set()

    async def _serve(self, ws: ClientConnection) -> None:
        self.connected.set()
        _system(self.bus, "link_up", peer=self.cfg.brain_url)
        self._bridge.report_drops()
        # A new connection may mean the brain rebooted, which resets its clock.
        self.clocksync.reset()
        self._logged_offset = None
        tasks = [
            asyncio.create_task(self._bridge.pump_out(ws)),
            asyncio.create_task(self._sync_loop(ws)),
        ]
        reason = "closed"
        try:
            async for raw in ws:
                received = clock.monotonic_ns()  # t4 for a sync reply: as early as possible
                msg = _parse(raw)
                if isinstance(msg, EventMsg):
                    self._bridge.receive_event(msg.event)
                elif isinstance(msg, SyncReply):
                    self._on_sync_reply(msg, received)
                elif isinstance(msg, NewSession):
                    self.bus.session_id = msg.session_id
                elif msg is not None:
                    log.warning("ignored a %s message from the brain", msg.kind)
        except ConnectionClosed as e:
            reason = str(e)
        finally:
            self.connected.clear()
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
                    await task
            _system(self.bus, "link_down", reason, peer=self.cfg.brain_url)

    async def _sync_loop(self, ws: ClientConnection) -> None:
        while True:
            await ws.send(encode(Sync(t1_ns=clock.monotonic_ns())))
            await asyncio.sleep(self.cfg.clock_sync_s)

    def _on_sync_reply(self, msg: SyncReply, t4: int) -> None:
        s = self.clocksync.add(msg.t1_ns, msg.t2_ns, msg.t3_ns, t4)
        # Network round trip as its own latency stage (portability rule 6).
        self.bus.emit(ev.LatencyMark, ev.LatencyMarkPayload(stage="network_rtt", duration_ns=s.rtt_ns))
        best = self.clocksync.best
        assert best is not None
        if self._offset_worth_logging(best, t4):
            self._logged_offset = (best.offset_ns, t4)
            _system(
                self.bus,
                "clock_offset",
                "brain clock minus body clock",
                offset_ns=best.offset_ns,
                error_ns=best.error_ns,
                rtt_ns=best.rtt_ns,
            )

    def _offset_worth_logging(self, best: Sample, now: int) -> bool:
        if self._logged_offset is None:
            return True
        offset, when = self._logged_offset
        return abs(best.offset_ns - offset) > _OFFSET_LOG_CHANGE_NS or now - when > _OFFSET_LOG_EVERY_NS

