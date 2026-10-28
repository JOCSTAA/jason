"""Real WebSocket connections between a LinkServer and LinkClient on 127.0.0.1."""

import asyncio
import contextlib
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
from jason.net.protocol import EventMsg, Hello, decode, encode

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


async def start_brain(tmp_path, port=0):
    rawlog = RawLog(tmp_path / "brain-state", "brain")
    bus = EventBus("brain", rawlog, session_id=SESSION)
    cfg = BrainLink(listen_host="127.0.0.1", listen_port=port, token_file=Path("unused"), **FAST)
    server = LinkServer(bus, cfg, TOKEN)
    await server.start()
    return server, rawlog


async def stop_brain(server, rawlog):
    await server.close()
    server.bus.close()
    rawlog.close()


def make_client(tmp_path, port, token=TOKEN, machine="mac"):
    rawlog = RawLog(tmp_path / f"{machine}-state", machine)
    bus = EventBus(machine, rawlog)  # no session: the brain provides it
    cfg = MacLink(
        brain_url=f"ws://127.0.0.1:{port}", token_file=Path("unused"), reconnect_max_s=0.2, **FAST
    )
    return LinkClient(bus, cfg, token), rawlog


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
