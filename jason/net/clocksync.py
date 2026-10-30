"""Clock-offset and round-trip estimation between machines.

Each machine's monotonic clock starts at an arbitrary point, so a body
timestamp and a brain timestamp can't be compared directly. To measure
"user stopped talking on the body → reply started on the brain", we need the
offset between the two clocks. This is the same exchange NTP uses:

    Body  t1 ──── sync ────►  t2  Brain
                             t3  (replies)
    Body  t4 ◄── sync_reply ─┘

    round trip = (t4 - t1) - (t3 - t2)        time spent on the network
    offset     = ((t2 - t1) + (t3 - t4)) / 2  brain clock ≈ body clock + offset

The offset is exact when the trip there and the trip back take equally long,
and off by at most half the round trip when they don't. A sample with a short
round trip spent little time waiting in network queues, so it is the most
trustworthy; the estimate is the shortest-round-trip sample in a recent window.
"""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    rtt_ns: int  # round trip, network time only
    offset_ns: int  # brain clock minus body clock
    t4_ns: int  # when the reply arrived (body clock)

    @property
    def error_ns(self) -> int:
        """The offset is within ± this much of the truth."""
        return self.rtt_ns // 2


def sample(t1: int, t2: int, t3: int, t4: int) -> Sample:
    """t1, t4 on the body's monotonic clock; t2, t3 on the brain's."""
    rtt = max(0, (t4 - t1) - (t3 - t2))
    offset = ((t2 - t1) + (t3 - t4)) // 2
    return Sample(rtt_ns=rtt, offset_ns=offset, t4_ns=t4)


class ClockSync:
    def __init__(self, window: int = 16) -> None:
        # Only recent samples count: clocks drift slowly and network paths change.
        self._samples: deque[Sample] = deque(maxlen=window)

    def add(self, t1: int, t2: int, t3: int, t4: int) -> Sample:
        s = sample(t1, t2, t3, t4)
        self._samples.append(s)
        return s

    @property
    def best(self) -> Sample | None:
        """The current estimate, or None before the first sample."""
        return min(self._samples, key=lambda s: s.rtt_ns, default=None)

    def reset(self) -> None:
        """Forget all samples (e.g. on a new connection: the brain may have rebooted)."""
        self._samples.clear()

    def to_brain(self, body_ns: int) -> int:
        """Convert a body monotonic timestamp to the brain's clock."""
        return body_ns + self._offset()

    def to_body(self, brain_ns: int) -> int:
        """Convert a brain monotonic timestamp to the body's clock."""
        return brain_ns - self._offset()

    def _offset(self) -> int:
        best = self.best
        if best is None:
            raise RuntimeError("no clock samples yet")
        return best.offset_ns
