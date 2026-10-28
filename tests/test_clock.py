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
