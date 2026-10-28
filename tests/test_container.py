"""The brain in its Docker container, with the Mac role connecting from outside.

Slow (it builds the image) and needs Docker running, so it only runs when asked:

    JASON_DOCKER_TESTS=1 uv run pytest tests/test_container.py -v

(on Windows PowerShell: $env:JASON_DOCKER_TESTS=1; uv run pytest tests/test_container.py -v)
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from jason.core.config import generate_token
from jason.log.rawlog import read_events

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    os.environ.get("JASON_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="set JASON_DOCKER_TESTS=1 (and have Docker running) to run container tests",
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def compose(env, *args, timeout=600):
    return subprocess.run(
        ["docker", "compose", "-p", "jason-test", *args],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout,
    )


def wait_for_listening(env, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if "brain listening" in compose(env, "logs", "brain").stdout:
            return
        time.sleep(0.5)
    raise AssertionError("brain container never started listening:\n" + compose(env, "logs").stdout)


def test_mac_talks_to_the_containerized_brain(tmp_path):
    data = tmp_path / "jason-data"
    data.mkdir()
    generate_token(data / "link_token")
    port = free_port()
    env = {**os.environ, "JASON_DATA": str(data), "JASON_PORT": str(port)}
    (tmp_path / "mac.toml").write_text(f"""
role = "mac"
machine = "mac"
state_dir = "mac-state"
[link]
brain_url = "ws://127.0.0.1:{port}"
token_file = "{(data / 'link_token').as_posix()}"
clock_sync_s = 0.5
""")

    up = compose(env, "up", "--build", "-d")
    assert up.returncode == 0, up.stderr
    try:
        wait_for_listening(env)
        mac = subprocess.run(
            [sys.executable, "-m", "jason", "run", "--role", "mac",
             "--config", str(tmp_path / "mac.toml"), "--count", "3", "--every", "0.3"],
            cwd=tmp_path, capture_output=True, text=True, timeout=120,
        )
        assert mac.returncode == 0, mac.stdout + mac.stderr
        assert "3 pings, median round trip" in mac.stdout

        new_day = compose(env, "exec", "-T", "brain", "jason", "new-day")
        assert "started day-0002_" in new_day.stdout, new_day.stdout + new_day.stderr
    finally:
        down = compose(env, "down")  # SIGTERM: the brain shuts down cleanly

    assert down.returncode == 0, down.stderr
    # The container wrote its state and raw log onto the host.
    state = json.loads((data / "state" / "state.json").read_text())
    assert state["day"] == 2
    day1, day2 = sorted((data / "state" / "rawlog" / "brain").glob("*.jsonl"))
    events = list(read_events(day1))
    assert sum(e.type == "ping" for e in events) == 3
    assert sum(e.type == "pong" for e in events) == 3
    last = list(read_events(day2))[-1]
    assert (last.machine, last.payload.kind) == ("brain", "shutdown")
