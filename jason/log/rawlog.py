"""JSONL raw log writer.

The raw log is training data and the debugging record (design §8.1): every
event on a machine's bus, one JSON object per line. There is one file per
session, and a session is one of Jason's days, from waking to sleep (design
§3.1), not a calendar date:

    <state_dir>/rawlog/<machine>/<session_id>.jsonl

Restarting Jason mid-session appends to the same file. Files are only ever
opened for appending, so existing lines are never rewritten or truncated.
"""

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from jason.core.events import Event, from_json, to_json

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _check_safe(kind: str, name: str) -> None:
    # Machine and session names become folder and file names; "../x" or "a/b"
    # would write outside the log folder.
    if not _SAFE_NAME.match(name) or name in (".", ".."):
        raise ValueError(f"{kind} {name!r} is not safe to use as a file name")


class RawLog:
    def __init__(self, state_dir: Path, machine: str, fsync: bool = False) -> None:
        _check_safe("machine name", machine)
        self.dir = Path(state_dir) / "rawlog" / machine
        self.dir.mkdir(parents=True, exist_ok=True)
        # flush() hands each line to the OS, which survives a crash of this
        # process. fsync also survives power loss, at a cost of up to a few ms
        # per event, so it is opt-in.
        self.fsync = fsync
        self._session: str | None = None
        self._file: BinaryIO | None = None

    def path_for(self, session_id: str) -> Path:
        _check_safe("session id", session_id)
        return self.dir / f"{session_id}.jsonl"

    def write(self, event: Event) -> None:
        if event.session_id != self._session:
            path = self.path_for(event.session_id)
            self.close()
            # "ab": append-only, binary so line endings are "\n" on every OS.
            self._file = open(path, "ab")
            self._session = event.session_id
        assert self._file is not None
        self._file.write(to_json(event).encode("utf-8") + b"\n")
        self._file.flush()
        if self.fsync:
            os.fsync(self._file.fileno())

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._session = None

    def __enter__(self) -> "RawLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_events(path: Path) -> Iterator[Event]:
    """Yield the events in one raw log file, in the order they were written."""
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield from_json(line)
