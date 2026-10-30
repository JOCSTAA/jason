"""The brain's state file: which of Jason's days it is.

<state_dir>/state.json holds the day counter and the current session (one
wake → sleep cycle). Only the brain has one; the body gets the session id from
the brain when it connects.

Unlike the raw log, this file is rewritten, so every save writes a temporary
file and swaps it in with os.replace(). A crash mid-save leaves either the old
file or the new one, never a half-written one.
"""

import os
from datetime import datetime
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from jason.core import clock

STATE_FORMAT = 1


class State(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state_format: int = STATE_FORMAT
    day: int = Field(ge=1)
    session_id: str
    session_started: AwareDatetime


def session_id_for(day: int, started: datetime) -> str:
    """e.g. day 7 started on 3 Nov 2026 → "day-0007_2026-11-03".

    The counter keeps sessions in order; the date (in this machine's local
    time zone) is only for humans reading file names.
    """
    return f"day-{day:04d}_{started.astimezone().date().isoformat()}"


class StateStore:
    def __init__(self, state_dir: Path) -> None:
        self.path = Path(state_dir) / "state.json"

    def load_or_create(self) -> State:
        """The current state; on the very first start, day 1 begins."""
        if not self.path.exists():
            return self._start_day(1)
        state = State.model_validate_json(self.path.read_bytes())
        if state.state_format != STATE_FORMAT:
            raise ValueError(
                f"{self.path} has state format {state.state_format}, "
                f"this version of Jason reads format {STATE_FORMAT}"
            )
        return state

    def new_day(self) -> State:
        """End the current session and start the next one (called by sleep)."""
        return self._start_day(self.load_or_create().day + 1)

    def _start_day(self, day: int) -> State:
        started = clock.wall_now()
        state = State(day=day, session_id=session_id_for(day, started), session_started=started)
        self._save(state)
        return state

    def _save(self, state: State) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
