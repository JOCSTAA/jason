"""Monotonic and wall-clock time helpers.

Two clocks, two jobs:
- Monotonic time never jumps backwards, so it is the one to measure durations
  with. Its zero point is arbitrary and differs per machine (and per container
  VM), so raw values are only comparable on the machine that produced them;
  net/clocksync.py estimates the offset between machines.
- Wall time is human-readable and roughly comparable across machines, but the
  OS can adjust it at any moment, so never subtract two wall times to measure
  latency.

The monotonic clock is read with time.perf_counter_ns(), not
time.monotonic_ns(): on Windows with Python 3.12, monotonic only ticks every
~15.6 ms, which would make every latency under that read as 0. perf_counter
is the high-resolution system clock everywhere (QueryPerformanceCounter on
Windows, mach_absolute_time on macOS, CLOCK_MONOTONIC on Linux), and it is
also monotonic.
"""

import time
from datetime import datetime, timezone


def monotonic_ns() -> int:
    return time.perf_counter_ns()


def wall_now() -> datetime:
    """Current wall-clock time, timezone-aware, in UTC."""
    return datetime.now(timezone.utc)


def elapsed_ms(start_ns: int, end_ns: int | None = None) -> float:
    """Milliseconds between two monotonic_ns() readings (end defaults to now)."""
    if end_ns is None:
        end_ns = monotonic_ns()
    return (end_ns - start_ns) / 1e6
