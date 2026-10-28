"""Loopback: both roles as separate processes on this machine, through the real CLI.

This is the exit criterion rehearsed on one machine: the Mac sends pings, the
brain replies, both raw logs hold both, and the round trip is printed.
"""

import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from jason.core.config import generate_token
from jason.log.rawlog import read_events

JASON = [sys.executable, "-m", "jason"]
POSIX = sys.platform != "win32"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_configs(tmp_path, port, token_file):
    (tmp_path / "brain.toml").write_text(f"""
role = "brain"
machine = "brain"
state_dir = "brain-state"
[link]
listen_host = "127.0.0.1"
listen_port = {port}
token_file = "{token_file.name}"
""")
    (tmp_path / "mac.toml").write_text(f"""
role = "mac"
machine = "mac"
state_dir = "mac-state"
[link]
brain_url = "ws://127.0.0.1:{port}"
token_file = "{token_file.name}"
reconnect_max_s = 0.5
clock_sync_s = 0.5
""")


class Brain:
    """A `jason run --role brain` process, with its output collected line by line."""

    def __init__(self, tmp_path):
        self.lines: queue.Queue[str] = queue.Queue()
        self.output: list[str] = []
        self.proc = subprocess.Popen(
            [*JASON, "run", "--role", "brain", "--config", str(tmp_path / "brain.toml")],
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line)

    def wait_for(self, pattern, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=0.1)
            except queue.Empty:
                if self.proc.poll() is not None:
                    break
                continue
            self.output.append(line)
            if re.search(pattern, line):
                return line
        raise AssertionError(f"brain never printed {pattern!r}; output:\n{''.join(self.output)}")

    def stop(self):
        """SIGTERM, as `docker stop` sends; Windows can only terminate outright."""
        if self.proc.poll() is None:
            if POSIX:
                self.proc.send_signal(signal.SIGTERM)
            else:
                self.proc.terminate()
        self.proc.wait(timeout=15)
        self._reader.join(timeout=5)  # the reader ends when the process closes its output
        self.proc.stdout.close()


def run_mac(tmp_path, *args, timeout=60):
    return subprocess.run(
        [*JASON, "run", "--role", "mac", "--config", str(tmp_path / "mac.toml"), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def events(tmp_path, machine):
    folder = tmp_path / f"{machine}-state" / "rawlog" / machine
    [path] = folder.glob("*.jsonl")  # one session: one file
    return list(read_events(path))


def count(evts, type_):
    return sum(1 for e in evts if e.type == type_)


def system_kinds(evts):
    return [e.payload.kind for e in evts if e.type == "system"]


@pytest.fixture
def setup(tmp_path):
    token_file = tmp_path / "link_token"
    generate_token(token_file)
    port = free_port()
    write_configs(tmp_path, port, token_file)
    return tmp_path


def test_loopback_ping_pong(setup):
    tmp_path = setup
    brain = Brain(tmp_path)
    try:
        brain.wait_for(r"brain listening on 127\.0\.0\.1:\d+, session day-0001_")
        mac = run_mac(tmp_path, "--count", "3", "--every", "0.2")
        assert mac.returncode == 0, mac.stdout + mac.stderr
        assert len(re.findall(r"ping \d: [\d.]+ ms round trip", mac.stdout)) == 3
        assert "3 pings, median round trip" in mac.stdout
    finally:
        brain.stop()

    mac_events, brain_events = events(tmp_path, "mac"), events(tmp_path, "brain")
    # The exit criterion: both raw logs contain both the pings and the pongs.
    for evts in (mac_events, brain_events):
        assert count(evts, "ping") == 3
        assert count(evts, "pong") == 3
    stages = [e.payload.stage for e in mac_events if e.type == "latency_mark"]
    assert stages.count("ping_rtt") == 3
    assert "network_rtt" in stages
    assert {"startup", "link_up", "shutdown"} <= set(system_kinds(mac_events))
    assert "startup" in system_kinds(brain_events)
    if POSIX:  # a graceful stop is only possible with a real SIGTERM
        assert brain.proc.returncode == 0
        assert system_kinds(brain_events)[-1] == "shutdown"


def test_mac_started_first_waits_for_the_brain(setup):
    tmp_path = setup
    mac = subprocess.Popen(
        [*JASON, "run", "--role", "mac", "--config", str(tmp_path / "mac.toml"),
         "--count", "1", "--every", "0.2"],
        cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(1.5)  # the Mac retries while nothing is listening
    assert mac.poll() is None
    brain = Brain(tmp_path)
    try:
        out, _ = mac.communicate(timeout=60)
        assert mac.returncode == 0, out
        assert "1 pings, median round trip" in out
    finally:
        brain.stop()
        if mac.poll() is None:
            mac.kill()


def test_wrong_token_stops_the_mac_with_a_clear_message(setup):
    tmp_path = setup
    brain = Brain(tmp_path)
    try:
        brain.wait_for(r"brain listening")
        generate_token(tmp_path / "link_token", overwrite=True)  # the Mac now has a different token
        mac = run_mac(tmp_path, "--count", "1")
        assert mac.returncode != 0
        assert "rejected the link token" in mac.stderr
    finally:
        brain.stop()


@pytest.mark.skipif(not POSIX, reason="needs a real SIGTERM")
def test_brain_restart_continues_the_same_session(setup):
    tmp_path = setup
    for _ in range(2):  # start, stop, start again
        brain = Brain(tmp_path)
        try:
            brain.wait_for(r"session day-0001_")
            assert run_mac(tmp_path, "--count", "1", "--every", "0.2").returncode == 0
        finally:
            brain.stop()
    brain_events = events(tmp_path, "brain")  # still one file: the same session
    # The brain's log also holds the Mac's events, so count the brain's own.
    own = [e for e in brain_events if e.machine == "brain"]
    assert system_kinds(own).count("startup") == 2
    assert count(brain_events, "pong") == 2


def test_python_dash_m_jason_works():
    result = subprocess.run([*JASON, "--help"], capture_output=True, text=True, timeout=30,
                            env={**os.environ})
    assert result.returncode == 0
    assert "run" in result.stdout and "new-day" in result.stdout
