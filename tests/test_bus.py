import asyncio

import pytest

from jason.core import events as ev
from jason.core.bus import EventBus
from jason.log.rawlog import RawLog, read_events

SESSION = "day-0001_2026-10-28"


@pytest.fixture
def bus(tmp_path):
    with RawLog(tmp_path, "mac") as rawlog:
        b = EventBus("mac", rawlog, session_id=SESSION)
        yield b
        b.close()


def logged(tmp_path, machine="mac"):
    return list(read_events(tmp_path / "rawlog" / machine / f"{SESSION}.jsonl"))


def ping(seq, machine="mac"):
    return ev.Ping(machine=machine, session_id=SESSION, payload=ev.PingPayload(seq=seq))


def drain(sub):
    out = []
    while not sub._queue.empty():
        out.append(sub._queue.get_nowait())
    return out


async def test_subscriber_gets_only_its_types(bus):
    pings = bus.subscribe(ev.Ping)
    bus.emit(ev.System, ev.SystemPayload(kind="startup"))
    sent = bus.emit(ev.Ping, ev.PingPayload(seq=1))
    assert await pings.get() == sent
    assert drain(pings) == []


async def test_subscribe_to_several_types_or_all(bus):
    some = bus.subscribe(ev.Ping, ev.Pong)
    everything = bus.subscribe()
    p = bus.emit(ev.Ping, ev.PingPayload(seq=1))
    s = bus.emit(ev.System, ev.SystemPayload(kind="startup"))
    assert drain(some) == [p]
    assert drain(everything) == [p, s]


async def test_every_subscriber_gets_every_event_in_order(bus):
    a, b = bus.subscribe(ev.Ping), bus.subscribe(ev.Ping)
    sent = [bus.emit(ev.Ping, ev.PingPayload(seq=i)) for i in range(5)]
    assert drain(a) == sent
    assert drain(b) == sent


async def test_every_event_is_logged_even_with_no_subscribers(bus, tmp_path):
    sent = [bus.emit(ev.Ping, ev.PingPayload(seq=i)) for i in range(3)]
    assert logged(tmp_path) == sent


async def test_event_is_logged_before_delivery(bus, tmp_path):
    sub = bus.subscribe(ev.Ping)
    sent = bus.emit(ev.Ping, ev.PingPayload(seq=1))
    received = await sub.get()
    assert received in logged(tmp_path)
    assert received == sent


async def test_events_from_the_other_machine_are_logged_and_delivered(bus, tmp_path):
    sub = bus.subscribe(ev.Ping)
    remote = ping(7, machine="brain")
    bus.publish(remote)
    assert await sub.get() == remote
    # It goes into this machine's log folder, keeping its own machine field.
    assert logged(tmp_path) == [remote]


async def test_slow_subscriber_drops_oldest_and_never_blocks(bus):
    slow = bus.subscribe(ev.Ping, maxsize=3)
    fast = bus.subscribe(ev.Ping)
    sent = [bus.emit(ev.Ping, ev.PingPayload(seq=i)) for i in range(5)]
    assert slow.dropped == 2
    assert drain(slow) == sent[2:]
    assert drain(fast) == sent


async def test_async_for_ends_when_subscription_closes(bus):
    sub = bus.subscribe(ev.Ping)
    received = []

    async def consumer():
        async for e in sub:
            received.append(e.payload.seq)

    task = asyncio.create_task(consumer())
    for i in range(3):
        bus.emit(ev.Ping, ev.PingPayload(seq=i))
    await asyncio.sleep(0)
    sub.close()
    await asyncio.wait_for(task, timeout=1)
    assert received == [0, 1, 2]


async def test_closing_a_full_subscription_still_ends_it(bus):
    sub = bus.subscribe(ev.Ping, maxsize=2)
    for i in range(5):
        bus.emit(ev.Ping, ev.PingPayload(seq=i))
    sub.close()
    with pytest.raises(StopAsyncIteration):
        while True:
            await asyncio.wait_for(sub.get(), timeout=1)


async def test_closed_subscription_gets_nothing_more(bus):
    sub = bus.subscribe(ev.Ping)
    sub.close()
    bus.emit(ev.Ping, ev.PingPayload(seq=1))
    with pytest.raises(StopAsyncIteration):
        await sub.get()


async def test_bus_close_ends_all_subscriptions(bus):
    subs = [bus.subscribe(), bus.subscribe(ev.Ping)]
    bus.close()
    for sub in subs:
        with pytest.raises(StopAsyncIteration):
            await sub.get()


async def test_emit_stamps_machine_and_session(bus):
    e = bus.emit(ev.Ping, ev.PingPayload(seq=1))
    assert (e.machine, e.session_id, e.type) == ("mac", SESSION, "ping")


async def test_emit_without_session_is_an_error(tmp_path):
    with RawLog(tmp_path, "mac") as rawlog:
        bus = EventBus("mac", rawlog)
        with pytest.raises(RuntimeError):
            bus.emit(ev.Ping, ev.PingPayload(seq=1))


async def test_components_as_tasks(bus):
    """A ping responder and a pinger running as separate tasks on one bus."""
    pongs = bus.subscribe(ev.Pong)
    pings = bus.subscribe(ev.Ping)

    async def responder():
        async for p in pings:
            bus.emit(ev.Pong, ev.PongPayload(ping_id=p.id, seq=p.payload.seq, ping_received_ns=0))

    task = asyncio.create_task(responder())
    sent = bus.emit(ev.Ping, ev.PingPayload(seq=42))
    reply = await asyncio.wait_for(pongs.get(), timeout=1)
    assert reply.payload.ping_id == sent.id
    pings.close()
    await asyncio.wait_for(task, timeout=1)


# --- schema 2: seq, run_id, current-session files -------------------------------


async def test_publish_stamps_this_machines_events_in_order(bus, tmp_path):
    sent = [bus.emit(ev.Ping, ev.PingPayload(seq=i)) for i in range(3)]
    assert [e.seq for e in sent] == [0, 1, 2]
    assert {e.run_id for e in sent} == {bus.run_id}
    assert logged(tmp_path) == sent


async def test_publish_returns_a_stamped_copy(bus):
    e = ping(1)
    published = bus.publish(e)
    assert (e.seq, published.seq) == (None, 0)
    assert published.id == e.id


async def test_other_machines_events_keep_their_numbers(bus, tmp_path):
    remote = ping(7, machine="brain").model_copy(update={"seq": 41, "run_id": "brain-run"})
    assert bus.publish(remote) == remote
    assert bus.emit(ev.Ping, ev.PingPayload(seq=1)).seq == 0  # our own counter is untouched
    assert logged(tmp_path)[0] == remote


async def test_each_bus_run_has_its_own_run_id(tmp_path):
    with RawLog(tmp_path, "mac") as rawlog:
        assert EventBus("mac", rawlog).run_id != EventBus("mac", rawlog).run_id


async def test_where_filter(bus):
    mine = bus.subscribe(where=lambda e: e.machine == bus.machine)
    bus.publish(ping(1, machine="brain"))
    sent = bus.emit(ev.Ping, ev.PingPayload(seq=2))
    assert drain(mine) == [sent]


async def test_late_event_goes_to_the_current_sessions_file(bus, tmp_path):
    day2 = "day-0002_2026-10-29"
    bus.emit(ev.Ping, ev.PingPayload(seq=1))
    day1_file = tmp_path / "rawlog" / "mac" / f"{SESSION}.jsonl"
    before = day1_file.read_bytes()

    bus.session_id = day2
    late = bus.publish(ping(2, machine="brain"))  # created during day 1, arrives on day 2

    assert day1_file.read_bytes() == before  # the finished day is never reopened
    [stored] = read_events(tmp_path / "rawlog" / "mac" / f"{day2}.jsonl")
    assert stored == late and stored.session_id == SESSION  # its own field is kept
