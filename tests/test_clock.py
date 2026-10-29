from datetime import timezone

from jason.core import clock


def test_monotonic_never_goes_backwards():
    readings = [clock.monotonic_ns() for _ in range(1000)]
    assert readings == sorted(readings)


def test_wall_now_is_utc_aware():
    assert clock.wall_now().tzinfo is timezone.utc


def test_elapsed_ms():
    assert clock.elapsed_ms(1_000_000, 3_500_000) == 2.5
    assert clock.elapsed_ms(clock.monotonic_ns()) >= 0


def test_monotonic_resolution_is_finer_than_a_millisecond():
    # Coarse clocks (e.g. Python 3.12's time.monotonic on Windows, ~15.6 ms)
    # would make sub-tick latencies read as 0.
    start = clock.monotonic_ns()
    while (now := clock.monotonic_ns()) == start:
        pass
    assert now - start < 1_000_000
