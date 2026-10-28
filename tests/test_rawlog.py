from datetime import datetime, timedelta, timezone

import pytest

from jason.core import events as ev
from jason.log.rawlog import RawLog, read_events

UTC = timezone.utc
DAY1 = "day-0001_2026-10-28"
DAY2 = "day-0002_2026-10-29"


def ping(seq: int, session_id: str = DAY1, ts_wall: datetime | None = None) -> ev.Ping:
    extra = {"ts_wall": ts_wall} if ts_wall else {}
    return ev.Ping(machine="mac", session_id=session_id, payload=ev.PingPayload(seq=seq), **extra)


def seqs(path):
    return [e.payload.seq for e in read_events(path)]


def test_writes_one_line_per_event_and_reads_back(tmp_path):
    events = [ping(i) for i in range(3)]
    with RawLog(tmp_path, "mac") as log:
        for e in events:
            log.write(e)
    path = tmp_path / "rawlog" / "mac" / f"{DAY1}.jsonl"
    assert list(read_events(path)) == events
    assert path.read_bytes().count(b"\n") == 3


def test_restart_mid_session_appends_and_never_rewrites(tmp_path):
    with RawLog(tmp_path, "mac") as log:
        log.write(ping(1))
        log.write(ping(2))
    path = tmp_path / "rawlog" / "mac" / f"{DAY1}.jsonl"
    before = path.read_bytes()

    # A new RawLog is what a restart of Jason looks like.
    with RawLog(tmp_path, "mac") as log:
        log.write(ping(3))
    after = path.read_bytes()

    assert after.startswith(before)
    assert seqs(path) == [1, 2, 3]


def test_flushed_after_every_write(tmp_path):
    log = RawLog(tmp_path, "mac")
    log.write(ping(1))
    # Read through a separate handle while the writer is still open.
    assert seqs(tmp_path / "rawlog" / "mac" / f"{DAY1}.jsonl") == [1]
    log.close()


def test_one_file_per_session(tmp_path):
    with RawLog(tmp_path, "mac") as log:
        log.write(ping(1, DAY1))
        log.write(ping(2, DAY2))
    folder = tmp_path / "rawlog" / "mac"
    assert sorted(p.name for p in folder.iterdir()) == [f"{DAY1}.jsonl", f"{DAY2}.jsonl"]
    assert seqs(folder / f"{DAY1}.jsonl") == [1]
    assert seqs(folder / f"{DAY2}.jsonl") == [2]


def test_session_runs_past_midnight_in_one_file(tmp_path):
    before_midnight = datetime(2026, 10, 28, 23, 59, tzinfo=UTC)
    with RawLog(tmp_path, "mac") as log:
        log.write(ping(1, ts_wall=before_midnight))
        log.write(ping(2, ts_wall=before_midnight + timedelta(minutes=2)))
    assert [p.name for p in (tmp_path / "rawlog" / "mac").iterdir()] == [f"{DAY1}.jsonl"]


def test_unix_line_endings(tmp_path):
    with RawLog(tmp_path, "mac") as log:
        log.write(ping(1))
    assert b"\r" not in (tmp_path / "rawlog" / "mac" / f"{DAY1}.jsonl").read_bytes()


def test_fsync_option_writes_normally(tmp_path):
    with RawLog(tmp_path, "mac", fsync=True) as log:
        log.write(ping(1))
    assert seqs(tmp_path / "rawlog" / "mac" / f"{DAY1}.jsonl") == [1]


@pytest.mark.parametrize("bad", ["", "..", "a/b", "a\\b", "mac book"])
def test_unsafe_machine_names_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        RawLog(tmp_path, bad)


@pytest.mark.parametrize("bad", ["..", "../escape", "a\\b", "day 1"])
def test_unsafe_session_ids_rejected(tmp_path, bad):
    with RawLog(tmp_path, "mac") as log, pytest.raises(ValueError):
        log.write(ping(1, bad))
    assert not any((tmp_path / "rawlog" / "mac").iterdir())
