import json
import re
from datetime import datetime, timezone

import pytest

from jason.core.state import STATE_FORMAT, StateStore, session_id_for


def test_first_start_begins_day_one(tmp_path):
    state = StateStore(tmp_path).load_or_create()
    assert state.day == 1
    assert re.fullmatch(r"day-0001_\d{4}-\d{2}-\d{2}", state.session_id)
    assert (tmp_path / "state.json").exists()


def test_restart_keeps_the_same_session(tmp_path):
    first = StateStore(tmp_path).load_or_create()
    # A new StateStore is what restarting the brain looks like.
    again = StateStore(tmp_path).load_or_create()
    assert again == first


def test_new_day_advances_the_session(tmp_path):
    store = StateStore(tmp_path)
    day1 = store.load_or_create()
    day2 = store.new_day()
    assert day2.day == 2
    assert day2.session_id.startswith("day-0002_")
    assert day2.session_id != day1.session_id
    assert StateStore(tmp_path).load_or_create() == day2


def test_new_day_on_a_fresh_folder_starts_at_day_two(tmp_path):
    # new_day() means "end today": with no state yet, today is day 1.
    assert StateStore(tmp_path).new_day().day == 2


def test_file_is_readable_json(tmp_path):
    StateStore(tmp_path).load_or_create()
    data = json.loads((tmp_path / "state.json").read_text())
    assert data["state_format"] == STATE_FORMAT
    assert data["day"] == 1


def test_save_leaves_no_temp_file(tmp_path):
    store = StateStore(tmp_path)
    store.load_or_create()
    store.new_day()
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_unknown_state_format_rejected(tmp_path):
    StateStore(tmp_path).load_or_create()
    path = tmp_path / "state.json"
    data = json.loads(path.read_text())
    data["state_format"] = STATE_FORMAT + 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="state format"):
        StateStore(tmp_path).load_or_create()


def test_session_id_format():
    started = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
    assert session_id_for(7, started).startswith("day-0007_2026-11-0")
