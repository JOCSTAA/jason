import pytest

from jason.net.clocksync import ClockSync, sample

MS = 1_000_000  # nanoseconds in a millisecond


def exchange(body_now, offset, there, process, back):
    """Simulate one probe. The brain's clock reads body time + offset."""
    t1 = body_now
    t2 = t1 + there + offset
    t3 = t2 + process
    t4 = t1 + there + process + back
    return t1, t2, t3, t4


def test_symmetric_delays_give_the_exact_offset():
    s = sample(*exchange(body_now=1000 * MS, offset=250 * MS, there=2 * MS, process=1 * MS, back=2 * MS))
    assert s.rtt_ns == 4 * MS  # network only: the brain's 1 ms of processing is excluded
    assert s.offset_ns == 250 * MS


def test_negative_offset():
    s = sample(*exchange(body_now=1000 * MS, offset=-40 * MS, there=3 * MS, process=0, back=3 * MS))
    assert s.offset_ns == -40 * MS


@pytest.mark.parametrize("there,back", [(1 * MS, 9 * MS), (9 * MS, 1 * MS), (0, 10 * MS)])
def test_asymmetric_delays_stay_within_half_the_round_trip(there, back):
    s = sample(*exchange(body_now=0, offset=100 * MS, there=there, process=1 * MS, back=back))
    assert abs(s.offset_ns - 100 * MS) <= s.error_ns
    assert s.error_ns == 5 * MS


def test_best_is_the_shortest_round_trip():
    sync = ClockSync()
    sync.add(*exchange(0, offset=50 * MS, there=10 * MS, process=0, back=1 * MS))  # rtt 11 ms, skewed
    sync.add(*exchange(100 * MS, offset=50 * MS, there=1 * MS, process=0, back=1 * MS))  # rtt 2 ms
    sync.add(*exchange(200 * MS, offset=50 * MS, there=5 * MS, process=0, back=1 * MS))  # rtt 6 ms
    assert sync.best.rtt_ns == 2 * MS
    assert sync.best.offset_ns == 50 * MS


def test_old_samples_leave_the_window():
    sync = ClockSync(window=3)
    sync.add(*exchange(0, offset=0, there=0, process=0, back=0))  # a perfect but old sample
    for i in range(3):
        sync.add(*exchange((i + 1) * MS, offset=0, there=2 * MS, process=0, back=2 * MS))
    assert sync.best.rtt_ns == 4 * MS


def test_convert_between_clocks():
    sync = ClockSync()
    sync.add(*exchange(0, offset=250 * MS, there=1 * MS, process=0, back=1 * MS))
    assert sync.to_brain(10 * MS) == 260 * MS
    assert sync.to_body(260 * MS) == 10 * MS


def test_no_samples_yet():
    sync = ClockSync()
    assert sync.best is None
    with pytest.raises(RuntimeError):
        sync.to_brain(0)


def test_reset_forgets_samples():
    sync = ClockSync()
    sync.add(*exchange(0, offset=0, there=1, process=0, back=1))
    sync.reset()
    assert sync.best is None


def test_round_trip_is_never_negative():
    # Rounding between machines could make it slightly negative in theory.
    assert sample(t1=10, t2=0, t3=100, t4=20).rtt_ns == 0
