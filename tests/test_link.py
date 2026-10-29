"""Real WebSocket connections between a LinkServer and LinkClient on 127.0.0.1."""

import asyncio
import contextlib
import socket
from pathlib import Path

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from jason.core import events as ev
from jason.core.bus import EventBus
from jason.core.config import BrainLink, MacLink
from jason.log.rawlog import RawLog, read_events
from jason.net import link as link_mod
from jason.net.link import LinkAuthError, LinkClient, LinkServer, LinkVersionError
from jason.net.protocol import EventMsg, Hello, Sync, SyncReply, decode, encode

TOKEN = "t" * 64
SESSION = "day-0001_2026-10-28"
FAST = {"heartbeat_s": 0.2, "timeout_s": 1.0}


async def eventually(condition, timeout=3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def logged(tmp_path, machine, session=SESSION):
    path = tmp_path / f"{machine}-state" / "rawlog" / machine / f"{session}.jsonl"
    return list(read_events(path)) if path.exists() else []


def system_kinds(tmp_path, machine):
    return [e.payload.kind for e in logged(tmp_path, machine) if e.type == "system"]


async def start_brain(tmp_path, port=0, session=SESSION):
    rawlog = RawLog(tmp_path / "brain-state", "brain")
    bus = EventBus("brain", rawlog, session_id=session)
    cfg = BrainLink(listen_host="127.0.0.1", listen_port=port, token_file=Path("unused"), **FAST)
    server = LinkServer(bus, cfg, TOKEN)
    await server.start()
    return server, rawlog


async def stop_brain(server, rawlog):
    await server.close()
    server.bus.close()
    rawlog.close()


def make_client(tmp_path, port, token=TOKEN, machine="mac", outbox_size=1000):
    rawlog = RawLog(tmp_path / f"{machine}-state", machine)
    bus = EventBus(machine, rawlog)  # no session: the brain provides it
    cfg = MacLink(
        brain_url=f"ws://127.0.0.1:{port}",
        token_file=Path("unused"),
        reconnect_max_s=0.2,
        clock_sync_s=0.05,
        **FAST,
    )
    return LinkClient(bus, cfg, token, outbox_size=outbox_size), rawlog


def raw_connect(port, token=TOKEN):
    """A bare WebSocket connection, for poking at the server directly."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return connect(f"ws://127.0.0.1:{port}", additional_headers=headers)


@pytest.fixture
async def brain(tmp_path):
    server, rawlog = await start_brain(tmp_path)
    yield server
    await stop_brain(server, rawlog)


@pytest.fixture
async def mac(tmp_path, brain):
    client, rawlog = make_client(tmp_path, brain.port)
    task = asyncio.create_task(client.run())
    await asyncio.wait_for(client.ready.wait(), 3)
    yield client
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    client.bus.close()
    rawlog.close()


# --- events cross, are logged on both sides, and don't echo -------------------


async def test_hello_gives_the_mac_the_session(mac):
    assert mac.bus.session_id == SESSION


async def test_mac_event_reaches_brain_and_both_logs(tmp_path, brain, mac):
    pings = brain.bus.subscribe(ev.Ping)
    sent = mac.bus.emit(ev.Ping, ev.PingPayload(seq=1))
    got = await asyncio.wait_for(pings.get(), 3)
    assert got == sent
    assert sent in logged(tmp_path, "mac")
    assert sent in logged(tmp_path, "brain")


async def test_brain_event_reaches_mac(tmp_path, brain, mac):
    pongs = mac.bus.subscribe(ev.Pong)
    sent = brain.bus.emit(ev.Pong, ev.PongPayload(ping_id="p", seq=1, ping_received_ns=0))
    assert await asyncio.wait_for(pongs.get(), 3) == sent
    assert sent in logged(tmp_path, "mac")


async def test_events_do_not_echo_back(tmp_path, brain, mac):
    mac_pings = mac.bus.subscribe(ev.Ping)
    brain_pings = brain.bus.subscribe(ev.Ping)
    sent = mac.bus.emit(ev.Ping, ev.PingPayload(seq=1))
    await asyncio.wait_for(brain_pings.get(), 3)
    await asyncio.sleep(0.3)  # time for an echo to arrive, if there were one
    assert await mac_pings.get() == sent
    assert mac_pings._queue.empty()
    assert [e.id for e in logged(tmp_path, "mac")].count(sent.id) == 1
    assert [e.id for e in logged(tmp_path, "brain")].count(sent.id) == 1


async def test_link_up_is_logged_on_both_sides(tmp_path, brain, mac):
    await eventually(lambda: "link_up" in system_kinds(tmp_path, "brain"))
    await eventually(lambda: "link_up" in system_kinds(tmp_path, "mac"))


# --- sessions -----------------------------------------------------------------


async def test_new_session_reaches_the_mac(tmp_path, brain, mac):
    day2 = "day-0002_2026-10-29"
    await brain.set_session(day2)
    await eventually(lambda: mac.bus.session_id == day2)
    sent = mac.bus.emit(ev.Ping, ev.PingPayload(seq=1))
    assert sent in logged(tmp_path, "mac", session=day2)


# --- auth and version checks --------------------------------------------------


async def test_wrong_token_is_rejected(tmp_path, brain):
    client, rawlog = make_client(tmp_path, brain.port, token="w" * 64)
    with pytest.raises(LinkAuthError):
        await asyncio.wait_for(client.run(), 3)
    rawlog.close()
    assert "link_rejected" in system_kinds(tmp_path, "brain")


async def test_missing_token_is_rejected(brain):
    with pytest.raises(InvalidStatus) as caught:
        async with raw_connect(brain.port, token=None):
            pass
    assert caught.value.response.status_code == 401


async def test_version_mismatch_is_refused(tmp_path, brain, monkeypatch):
    monkeypatch.setattr(link_mod, "SCHEMA_VERSION", 999)  # pretend the Mac is newer
    client, rawlog = make_client(tmp_path, brain.port)
    with pytest.raises(LinkVersionError, match="schema"):
        await asyncio.wait_for(client.run(), 3)
    rawlog.close()


async def test_same_machine_name_is_refused(tmp_path, brain):
    client, rawlog = make_client(tmp_path, brain.port, machine="brain")
    with pytest.raises(LinkVersionError, match="names must differ"):
        await asyncio.wait_for(client.run(), 3)
    rawlog.close()


# --- reconnecting ---------------------------------------------------------------


async def test_reconnects_after_brain_restart_and_sends_what_waited(tmp_path):
    server, brain_log = await start_brain(tmp_path)
    port = server.port
    client, mac_log = make_client(tmp_path, port)
    task = asyncio.create_task(client.run())
    await asyncio.wait_for(client.ready.wait(), 3)

    await stop_brain(server, brain_log)  # the brain goes away
    await eventually(lambda: not client.connected.is_set())
    waiting = client.bus.emit(ev.Ping, ev.PingPayload(seq=1))  # created while disconnected

    server2, brain_log2 = await start_brain(tmp_path, port=port)  # the brain comes back
    pings = server2.bus.subscribe(ev.Ping)
    assert await asyncio.wait_for(pings.get(), 5) == waiting
    assert "link_down" in system_kinds(tmp_path, "mac")

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    client.bus.close()
    mac_log.close()
    await stop_brain(server2, brain_log2)


async def test_newest_connection_wins(brain):
    async with raw_connect(brain.port) as first:
        assert isinstance(decode(await first.recv()), Hello)
        async with raw_connect(brain.port) as second:
            assert isinstance(decode(await second.recv()), Hello)
            # The first connection may still hold events the brain sent it
            # (e.g. its link_up); after those, it must be closed.
            with pytest.raises(ConnectionClosed):
                while True:
                    await asyncio.wait_for(first.recv(), 3)
            assert "replaced" in (first.close_reason or "")

            # The newer connection works.
            pings = brain.bus.subscribe(ev.Ping)
            ping = ev.Ping(machine="mac", session_id=SESSION, payload=ev.PingPayload(seq=5))
            await second.send(encode(EventMsg(event=ping)))
            assert await asyncio.wait_for(pings.get(), 3) == ping


async def test_malformed_message_is_ignored(brain):
    async with raw_connect(brain.port) as ws:
        await ws.recv()  # hello
        await ws.send("this is not json")
        pings = brain.bus.subscribe(ev.Ping)
        ping = ev.Ping(machine="mac", session_id=SESSION, payload=ev.PingPayload(seq=1))
        await ws.send(encode(EventMsg(event=ping)))
        assert await asyncio.wait_for(pings.get(), 3) == ping


async def test_incoming_event_claiming_to_be_local_is_dropped(brain):
    async with raw_connect(brain.port) as ws:
        await ws.recv()  # hello
        pings = brain.bus.subscribe(ev.Ping)
        fake = ev.Ping(machine="brain", session_id=SESSION, payload=ev.PingPayload(seq=1))
        real = ev.Ping(machine="mac", session_id=SESSION, payload=ev.PingPayload(seq=2))
        await ws.send(encode(EventMsg(event=fake)))
        await ws.send(encode(EventMsg(event=real)))
        assert await asyncio.wait_for(pings.get(), 3) == real


# --- clock sync ---------------------------------------------------------------


async def test_clock_offset_is_estimated(mac):
    await eventually(lambda: mac.clocksync.best is not None)
    best = mac.clocksync.best
    # Both "machines" share this computer's clock, so the true offset is 0.
    assert abs(best.offset_ns) <= best.error_ns + 1_000_000
    assert best.rtt_ns < 1_000_000_000


async def test_round_trips_are_logged_as_network_rtt(tmp_path, brain, mac):
    def rtt_marks(machine):
        return [
            e for e in logged(tmp_path, machine)
            if e.type == "latency_mark" and e.payload.stage == "network_rtt"
        ]

    await eventually(lambda: len(rtt_marks("mac")) >= 3)
    # Mac events reach the brain's log too.
    await eventually(lambda: len(rtt_marks("brain")) >= 1)


async def test_clock_offset_is_logged_but_not_every_sample(tmp_path, brain, mac):
    await eventually(
        lambda: len([e for e in logged(tmp_path, "mac") if e.type == "latency_mark"]) >= 10
    )
    offsets = [
        e for e in logged(tmp_path, "mac")
        if e.type == "system" and e.payload.kind == "clock_offset"
    ]
    assert 1 <= len(offsets) < 10
    assert {"offset_ns", "error_ns", "rtt_ns"} <= offsets[0].payload.data.keys()


async def test_brain_answers_a_sync_probe(brain):
    async with raw_connect(brain.port) as ws:
        await ws.recv()  # hello
        await ws.send(encode(Sync(t1_ns=123)))
        while True:
            reply = decode(await asyncio.wait_for(ws.recv(), 3))
            if isinstance(reply, SyncReply):
                break
        assert reply.t1_ns == 123
        assert reply.t2_ns <= reply.t3_ns


# --- schema 2: late events and gaps across a network blip -------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def blip_down(server):
    """The brain stops accepting connections but keeps running (a network blip)."""
    server._server.close()
    await server._server.wait_closed()


async def run_client(client):
    task = asyncio.create_task(client.run())
    await asyncio.wait_for(client.ready.wait(), 3)
    return task


async def stop_client(client, task, rawlog):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    client.bus.close()
    rawlog.close()


def brain_file(tmp_path, session):
    return tmp_path / "brain-state" / "rawlog" / "brain" / f"{session}.jsonl"


async def test_mac_events_from_before_a_new_day_land_in_the_new_days_file(tmp_path):
    day2 = "day-0002_2026-10-29"
    server, brain_log = await start_brain(tmp_path, port=free_port())
    client, mac_log = make_client(tmp_path, server.port)
    task = await run_client(client)
    pings = server.bus.subscribe(ev.Ping)

    await blip_down(server)
    await eventually(lambda: not client.connected.is_set())
    await server.set_session(day2)  # a new day starts while the Mac can't hear it
    day1_before = brain_file(tmp_path, SESSION).read_bytes()
    late = [client.bus.emit(ev.Ping, ev.PingPayload(seq=i)) for i in range(3)]
    assert {e.session_id for e in late} == {SESSION}

    await server.start()  # the blip ends; the Mac reconnects and sends its backlog
    for _ in late:
        await asyncio.wait_for(pings.get(), 5)

    assert brain_file(tmp_path, SESSION).read_bytes() == day1_before
    in_day2 = {e.id: e for e in read_events(brain_file(tmp_path, day2))}
    assert all(in_day2[e.id].session_id == SESSION for e in late)

    await stop_client(client, task, mac_log)
    await stop_brain(server, brain_log)


async def test_outbox_overflow_is_reported_by_both_sides(tmp_path):
    server, brain_log = await start_brain(tmp_path, port=free_port())
    client, mac_log = make_client(tmp_path, server.port, outbox_size=3)
    task = await run_client(client)
    first = client.bus.emit(ev.Ping, ev.PingPayload(seq=0))
    await eventually(lambda: any(e.id == first.id for e in logged(tmp_path, "brain")))

    await blip_down(server)
    await eventually(lambda: not client.connected.is_set())
    for i in range(10):  # far more than the outbox holds
        client.bus.emit(ev.Ping, ev.PingPayload(seq=i + 1))

    await server.start()
    await eventually(lambda: "link_gap" in system_kinds(tmp_path, "brain"), timeout=5)
    await eventually(lambda: "link_backlog_dropped" in system_kinds(tmp_path, "brain"), timeout=5)

    events = logged(tmp_path, "brain")
    gaps = [e.payload.data for e in events if e.type == "system" and e.payload.kind == "link_gap"]
    reported = [e.payload.data for e in events if e.type == "system" and e.payload.kind == "link_backlog_dropped"]
    total_dropped = client._bridge._outbox.dropped
    assert total_dropped >= 8
    # The brain's view (gaps in seq) matches exactly what the Mac's outbox dropped.
    assert sum(g["missing"] for g in gaps) == total_dropped
    assert all(g["machine"] == "mac" and g["run_id"] == client.bus.run_id for g in gaps)
    # The Mac's own report can trail by the drops its report itself caused.
    assert 0 < reported[0]["count"] <= total_dropped

    await stop_client(client, task, mac_log)
    await stop_brain(server, brain_log)


async def test_rejected_connections_are_rate_limited(tmp_path, brain, monkeypatch):
    for _ in range(5):
        with pytest.raises(InvalidStatus):
            async with raw_connect(brain.port, token="w" * 64):
                pass
    assert system_kinds(tmp_path, "brain").count("link_rejected") == 1

    monkeypatch.setattr(link_mod, "REJECTED_LOG_EVERY_NS", 0)  # the window has passed
    with pytest.raises(InvalidStatus):
        async with raw_connect(brain.port, token="w" * 64):
            pass
    rejected = [e.payload.data for e in logged(tmp_path, "brain")
                if e.type == "system" and e.payload.kind == "link_rejected"]
    assert len(rejected) == 2
    assert (rejected[0]["suppressed"], rejected[1]["suppressed"]) == (0, 4)
