"""Both roles in one process, talking over a real WebSocket on 127.0.0.1."""

import asyncio
import contextlib
from pathlib import Path

import pytest

from jason import cli
from jason.brain.main import Brain
from jason.core import events as ev
from jason.core.config import BrainConfig, BrainLink, MacConfig, MacLink, generate_token
from jason.core.state import StateStore
from jason.log.rawlog import read_events
from jason.mac.main import run_mac
from jason.net.admin import admin_url, send_command
from jason.net.link import LinkAuthError


def write_brain_toml(tmp_path, port, token_file):
    path = tmp_path / "brain.toml"
    path.write_text(f"""
role = "brain"
machine = "brain"
state_dir = "brain-state"
[link]
listen_host = "127.0.0.1"
listen_port = {port}
token_file = "{token_file}"
timeout_s = 1.0
""")
    return path


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "link_token"
    generate_token(path)
    return path


def brain_cfg(tmp_path, token_file, port=0):
    return BrainConfig(
        role="brain",
        machine="brain",
        state_dir=tmp_path / "brain-state",
        link=BrainLink(listen_host="127.0.0.1", listen_port=port, token_file=token_file,
                       heartbeat_s=0.2, timeout_s=1.0),
    )


def mac_cfg(tmp_path, token_file, port):
    return MacConfig(
        role="mac",
        machine="mac",
        state_dir=tmp_path / "mac-state",
        link=MacLink(brain_url=f"ws://127.0.0.1:{port}", token_file=token_file,
                     heartbeat_s=0.2, timeout_s=1.0, reconnect_max_s=0.2, clock_sync_s=0.05),
    )


def logged(tmp_path, machine):
    folder = tmp_path / f"{machine}-state" / "rawlog" / machine
    return {p.stem: list(read_events(p)) for p in sorted(folder.glob("*.jsonl"))}


def of_type(events, type_):
    return [e for e in events if e.type == type_]


@pytest.fixture
async def brain(tmp_path, token_file):
    b = Brain(brain_cfg(tmp_path, token_file))
    await b.start()
    yield b
    await b.stop()


async def eventually(condition, timeout=3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


# --- ping / pong ---------------------------------------------------------------


async def test_ping_pong_round_trip(tmp_path, token_file, brain):
    lines = []
    rtts = await run_mac(mac_cfg(tmp_path, token_file, brain.port), count=2, every_s=0.05, out=lines.append)
    assert len(rtts) == 2 and all(0 < r < 1000 for r in rtts)
    assert any("ping 1:" in line and "round trip" in line for line in lines)

    [mac_events] = logged(tmp_path, "mac").values()
    await eventually(lambda: len(of_type(list(logged(tmp_path, "brain").values())[0], "pong")) == 2)
    [brain_events] = logged(tmp_path, "brain").values()
    # The exit criterion: both raw logs contain both the pings and the pongs.
    for events in (mac_events, brain_events):
        assert len(of_type(events, "ping")) == 2
        assert len(of_type(events, "pong")) == 2
    marks = [e for e in of_type(mac_events, "latency_mark") if e.payload.stage == "ping_rtt"]
    assert len(marks) == 2


async def test_brain_ignores_its_own_pings(brain):
    pongs = brain.bus.subscribe(ev.Pong)
    brain.bus.emit(ev.Ping, ev.PingPayload(seq=1))
    await asyncio.sleep(0.1)
    assert pongs._queue.empty()


async def test_startup_and_shutdown_are_logged(tmp_path, token_file, brain):
    await run_mac(mac_cfg(tmp_path, token_file, brain.port), count=1, every_s=0.05, out=lambda _: None)
    [mac_events] = logged(tmp_path, "mac").values()
    kinds = [e.payload.kind for e in of_type(mac_events, "system")]
    assert "startup" in kinds and "shutdown" in kinds
    [brain_events] = logged(tmp_path, "brain").values()
    assert "startup" in [e.payload.kind for e in of_type(brain_events, "system")]


async def test_wrong_token_stops_the_mac(tmp_path, token_file, brain):
    bad = tmp_path / "bad_token"
    generate_token(bad)
    cfg = mac_cfg(tmp_path, bad, brain.port)
    with pytest.raises(LinkAuthError):
        await asyncio.wait_for(run_mac(cfg, count=1, out=lambda _: None), 3)


# --- new day ---------------------------------------------------------------------


async def test_new_day_switches_the_running_brain_and_mac(tmp_path, token_file, brain):
    mac = asyncio.create_task(
        run_mac(mac_cfg(tmp_path, token_file, brain.port), every_s=0.05, out=lambda _: None)
    )
    await eventually(lambda: len(logged(tmp_path, "mac")) == 1)

    token = token_file.read_text().strip()
    result = await send_command(admin_url(f"ws://127.0.0.1:{brain.port}"), token, "new_day")
    assert result.ok and "day-0002_" in result.message
    assert StateStore(tmp_path / "brain-state").load_or_create().day == 2

    # The Mac keeps pinging (the admin connection didn't knock it off), and its
    # new events land in the new day's file on both machines.
    await eventually(lambda: len(logged(tmp_path, "mac")) == 2)
    await eventually(lambda: any(of_type(list(logged(tmp_path, "brain").values())[-1], "pong")))

    mac.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await mac


async def test_unknown_admin_command(brain, token_file):
    token = token_file.read_text().strip()
    result = await send_command(admin_url(f"ws://127.0.0.1:{brain.port}"), token, "explode")
    assert not result.ok and "unknown command" in result.message


async def test_admin_needs_the_token(brain):
    with pytest.raises(LinkAuthError):
        await send_command(admin_url(f"ws://127.0.0.1:{brain.port}"), "x" * 64, "new_day")


# --- the command line ---------------------------------------------------------------


def test_new_day_cli_when_the_brain_is_not_running(tmp_path, token_file, capsys):
    path = write_brain_toml(tmp_path, port=1, token_file=token_file)  # nothing listens on port 1
    cli.main(["new-day", "--config", str(path)])
    assert "brain isn't running" in capsys.readouterr().out
    assert StateStore(tmp_path / "brain-state").load_or_create().day == 2


def test_run_without_a_token_explains_what_to_do(tmp_path):
    path = write_brain_toml(tmp_path, port=0, token_file=tmp_path / "missing")
    with pytest.raises(SystemExit) as caught:
        cli.main(["run", "--role", "brain", "--config", str(path)])
    assert "jason gen-token" in str(caught.value.code)


def test_run_with_the_wrong_role_config_fails_clearly(tmp_path, token_file):
    path = write_brain_toml(tmp_path, port=0, token_file=token_file)
    with pytest.raises(SystemExit) as caught:
        cli.main(["run", "--role", "mac", "--config", str(path)])
    assert "role" in str(caught.value.code)


def test_sleep_is_a_stub(capsys):
    cli.main(["sleep"])
    assert "M6" in capsys.readouterr().out


def test_default_configs_exist():
    assert cli.default_config("mac").exists()
    assert cli.default_config("brain").exists()
    assert cli.default_config("mac").parent == Path(cli.__file__).resolve().parent.parent / "config"
