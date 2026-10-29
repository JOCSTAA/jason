"""In-process async pub/sub; every published event is also written to the raw log.

Components are asyncio tasks. A component publishes events with
publish() or emit() and reads the types it cares about from a Subscription:

    sub = bus.subscribe(ev.Ping)
    async for event in sub:
        ...

publish() is a plain call that never waits, so a component can't be held up
by another component. Each subscriber has its own bounded queue; if a
subscriber falls behind and its queue fills, its oldest undelivered event is
dropped and counted. The raw log already has every event, so a drop loses
delivery, never the record.

Not thread-safe: call publish() from the event loop's thread. Code running in
another thread (e.g. a future audio callback) should hand events over with
loop.call_soon_threadsafe(bus.publish, event).
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from uuid import uuid4

from jason.core.events import Event, Payload
from jason.log.rawlog import RawLog

log = logging.getLogger(__name__)

_CLOSED = object()


def _type_name(cls: type[Event]) -> str:
    return cls.model_fields["type"].default


class Subscription:
    def __init__(
        self,
        bus: "EventBus",
        types: frozenset[str] | None,
        maxsize: int,
        where: Callable[[Event], bool] | None = None,
    ) -> None:
        self._bus = bus
        self.types = types  # None means every type
        self._where = where  # an extra filter, e.g. "only this machine's events"
        self._queue: asyncio.Queue[object] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self.closed = False

    def wants(self, event: Event) -> bool:
        if self.types is not None and event.type not in self.types:
            return False
        return self._where is None or self._where(event)

    def _deliver(self, event: Event) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 100 == 0:
                log.warning("subscriber %s is behind: %d events dropped", self.types, self.dropped)
        self._queue.put_nowait(event)

    async def get(self) -> Event:
        """Wait for the next event. Raises StopAsyncIteration once closed."""
        item = await self._queue.get()
        if item is _CLOSED:
            raise StopAsyncIteration
        assert isinstance(item, Event)
        return item

    def close(self) -> None:
        """Stop receiving; a waiting get() or async-for loop ends."""
        if self.closed:
            return
        self.closed = True
        self._bus._subs.discard(self)
        # Make room so the end marker always fits, even in a full queue.
        while self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(_CLOSED)

    def __aiter__(self) -> AsyncIterator[Event]:
        return self

    async def __anext__(self) -> Event:
        return await self.get()

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class EventBus:
    def __init__(self, machine: str, rawlog: RawLog, session_id: str | None = None) -> None:
        self.machine = machine
        # The current session (one of Jason's days). Events this machine
        # creates with emit() are stamped with it; it changes on a new day.
        self.session_id = session_id
        # This run of the bus: seq counts this machine's published events, so the
        # other machine can spot a gap; run_id tells runs apart after a restart.
        self.run_id = uuid4().hex
        self._next_seq = 0
        self._rawlog = rawlog
        self._subs: set[Subscription] = set()

    def subscribe(
        self,
        *types: type[Event],
        maxsize: int = 1000,
        where: Callable[[Event], bool] | None = None,
    ) -> Subscription:
        """Subscribe to the given event classes, or to every event if none are given.

        `where` narrows it further, e.g. `lambda e: e.machine == bus.machine`.
        """
        names = frozenset(_type_name(t) for t in types) if types else None
        sub = Subscription(self, names, maxsize, where)
        self._subs.add(sub)
        return sub

    def publish[E: Event](self, event: E) -> E:
        """Stamp, log, then hand the event to every interested subscriber.

        This machine's own events get the next seq and this run's run_id; events
        from the other machine keep theirs. Returns the event as published (a
        stamped copy: events can't be modified). Events that arrived from the
        other machine are published here too, so each machine's raw log holds
        everything its bus carried.
        """
        stamp = event.machine == self.machine and event.seq is None
        if stamp:
            event = event.model_copy(update={"seq": self._next_seq, "run_id": self.run_id})
        # Logged first: if writing fails, the event is not delivered, so the
        # raw log never misses an event a component acted on. Written to this
        # bus's current session, so a late event never reopens a finished day.
        self._rawlog.write(event, self.session_id)
        if stamp:
            self._next_seq += 1  # only once it's in the log: no gap from a failed write
        for sub in list(self._subs):
            if sub.wants(event):
                sub._deliver(event)
        return event

    def emit[E: Event](self, event_cls: type[E], payload: Payload) -> E:
        """Create an event from this machine in the current session, publish it,
        and return it as published (with its seq and run_id)."""
        if self.session_id is None:
            raise RuntimeError("no current session id; the brain hasn't sent one yet")
        event = event_cls(machine=self.machine, session_id=self.session_id, payload=payload)
        return self.publish(event)

    def close(self) -> None:
        """End every subscription (used at shutdown)."""
        for sub in list(self._subs):
            sub.close()
