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
from collections.abc import AsyncIterator

from jason.core.events import Event, Payload
from jason.log.rawlog import RawLog

log = logging.getLogger(__name__)

_CLOSED = object()


def _type_name(cls: type[Event]) -> str:
    return cls.model_fields["type"].default


class Subscription:
    def __init__(self, bus: "EventBus", types: frozenset[str] | None, maxsize: int) -> None:
        self._bus = bus
        self.types = types  # None means every type
        self._queue: asyncio.Queue[object] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self.closed = False

    def wants(self, event: Event) -> bool:
        return self.types is None or event.type in self.types

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
        self._rawlog = rawlog
        self._subs: set[Subscription] = set()

    def subscribe(self, *types: type[Event], maxsize: int = 1000) -> Subscription:
        """Subscribe to the given event classes, or to every event if none are given."""
        names = frozenset(_type_name(t) for t in types) if types else None
        sub = Subscription(self, names, maxsize)
        self._subs.add(sub)
        return sub

    def publish(self, event: Event) -> None:
        """Log the event, then hand it to every interested subscriber.

        Events that arrived from the other machine are published here too, so
        each machine's raw log holds everything its bus carried.
        """
        # Logged first: if writing fails, the event is not delivered, so the
        # raw log never misses an event a component acted on.
        self._rawlog.write(event)
        for sub in list(self._subs):
            if sub.wants(event):
                sub._deliver(event)

    def emit[E: Event](self, event_cls: type[E], payload: Payload) -> E:
        """Create an event from this machine in the current session, publish it, and return it."""
        if self.session_id is None:
            raise RuntimeError("no current session id; the brain hasn't sent one yet")
        event = event_cls(machine=self.machine, session_id=self.session_id, payload=payload)
        self.publish(event)
        return event

    def close(self) -> None:
        """End every subscription (used at shutdown)."""
        for sub in list(self._subs):
            sub.close()
