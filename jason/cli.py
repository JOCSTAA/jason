"""Command-line entry point.

    jason run --role brain [--config PATH]
    jason run --role body [--config PATH] [--text [--speak] | --listen | --ping [--count N] [--every S]]
                          [--voice NAME] [--speed X]   (default: a spoken conversation)
    jason new-day [--role brain|body] [--config PATH]
    jason sleep [--config PATH]
    jason gen-token [--file PATH] [--force]
    jason audio-check [--config PATH] [--echo-delay-ms MS]

Without --config, each command uses config/<role>.toml from the repo.
"""

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from jason.core.config import (
    DEFAULT_TOKEN_FILE,
    BodyConfig,
    BrainConfig,
    TokenError,
    TtsConfig,
    generate_token,
    load_config,
    load_token,
)
from jason.core.state import StateStore
from jason.net.admin import admin_url, send_command
from jason.net.link import LinkAuthError, LinkVersionError

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def default_config(role: str) -> Path:
    return CONFIG_DIR / f"{role}.toml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jason")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one machine's role")
    run.add_argument("--role", choices=["body", "brain"], required=True)
    run.add_argument("--config", type=Path, help="TOML config (default: config/<role>.toml)")
    run.add_argument("--ping", action="store_true", help="body only: the M0 pinger instead of a conversation")
    run.add_argument("--count", type=int, help="body only, pinger: stop after this many pings (implies --ping)")
    run.add_argument("--every", type=float, help="body only, pinger: seconds between pings (implies --ping)")
    run.add_argument("--text", action="store_true", help="body only: chat with Jason by typing (no audio)")
    run.add_argument("--speak", action="store_true", help="body only, with --text: Jason says his replies out loud")
    run.add_argument("--voice", help="body only: Kokoro voice for this run, e.g. am_puck (default: [tts] voice in the config)")
    run.add_argument("--speed", type=float, help="body only: speaking speed for this run, e.g. 1.1 (default: [tts] speed)")
    run.add_argument("--listen", action="store_true",
                     help="body only: print speech start, pauses and end-of-turn probability as you talk")

    new_day = sub.add_parser("new-day", help="end today's session and start the next")
    new_day.add_argument("--role", choices=["brain", "body"], default="brain",
                         help="which machine's config to use (default: brain)")
    new_day.add_argument("--config", type=Path)

    sleep = sub.add_parser("sleep", help="run the nightly sleep jobs (M6)")
    sleep.add_argument("--config", type=Path)

    check = sub.add_parser("audio-check", help="body: test the mic, speaker and echo cancellation")
    check.add_argument("--config", type=Path)
    check.add_argument("--echo-delay-ms", type=float, help="try this echo delay hint instead of the config's")

    gen = sub.add_parser("gen-token", help="create the link auth token")
    gen.add_argument("--file", type=Path, default=DEFAULT_TOKEN_FILE, help="where to write it")
    gen.add_argument("--force", action="store_true", help="replace an existing token")

    args = parser.parse_args(argv)
    # Write each printed line immediately, even when output goes to a file or
    # to Docker's logs rather than a terminal.
    with contextlib.suppress(AttributeError):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "run":
            _run(args)
        elif args.command == "new-day":
            _new_day(args)
        elif args.command == "audio-check":
            _audio_check(args)
        elif args.command == "sleep":
            print("jason sleep: not implemented until M6; nothing to do.")
        else:
            _gen_token(args.file, args.force)
    except (TokenError, LinkAuthError, LinkVersionError) as e:
        sys.exit(f"jason: {e}")
    except FileNotFoundError as e:
        sys.exit(f"jason: file not found: {e.filename}")
    except ValidationError as e:
        sys.exit(f"jason: invalid config:\n{e}")
    except ValueError as e:  # e.g. a config file for the other role
        sys.exit(f"jason: {e}")
    except KeyboardInterrupt:
        print("\njason: stopped")


async def _until_stopped[T](coro: Coroutine[Any, Any, T]) -> T | None:
    """Run coro; on SIGTERM (what `docker stop` sends) cancel it so its cleanup
    runs, the same as Ctrl+C does. Returns None if it was stopped that way."""
    task = asyncio.ensure_future(coro)
    with contextlib.suppress(NotImplementedError):  # Windows has no SIGTERM handlers
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await task
    except asyncio.CancelledError:
        print("jason: stopped")
        return None


def _run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config or default_config(args.role), args.role)
    if isinstance(cfg, BodyConfig) and (args.voice or args.speed):
        # A one-run override of the voice settings; the config file stays as it is.
        tts = cfg.tts.model_copy(update={k: v for k, v in (("voice", args.voice), ("speed", args.speed)) if v})
        cfg = cfg.model_copy(update={"tts": TtsConfig.model_validate(tts.model_dump())})
    if isinstance(cfg, BrainConfig):
        from jason.brain.main import run_brain

        asyncio.run(_until_stopped(run_brain(cfg)))
    else:
        from jason.body.main import run_body

        ping = args.ping or args.count is not None or args.every is not None
        voice = not (ping or args.text or args.listen)  # a spoken conversation is the default
        asyncio.run(_until_stopped(run_body(
            cfg, count=args.count, every_s=args.every or 5.0, text=args.text, listen=args.listen,
            speak=args.speak, voice=voice)))


def _new_day(args: argparse.Namespace) -> None:
    cfg = load_config(args.config or default_config(args.role), args.role)
    token = load_token(cfg.link.token_file)
    if isinstance(cfg, BrainConfig):
        url = admin_url(f"ws://127.0.0.1:{cfg.link.listen_port}")  # run on the brain itself
    else:
        url = admin_url(cfg.link.brain_url)
    try:
        result = asyncio.run(send_command(url, token, "new_day", timeout_s=cfg.link.timeout_s))
    except (OSError, TimeoutError):
        if not isinstance(cfg, BrainConfig):
            sys.exit(f"jason: can't reach the brain at {url}")
        # The brain isn't running: just advance its state file.
        state = StateStore(cfg.state_dir).new_day()
        print(f"brain isn't running; state file now on {state.session_id}")
        return
    if not result.ok:
        sys.exit(f"jason: {result.message}")
    print(result.message)


def _audio_check(args: argparse.Namespace) -> None:
    cfg = load_config(args.config or default_config("body"), "body")
    try:
        from jason.body.audio_check import run_check
    except ImportError as e:
        sys.exit(f"jason: the body's audio libraries aren't installed ({e.name}); run `uv sync --extra body`")
    audio = cfg.audio
    if args.echo_delay_ms is not None:
        audio = audio.model_copy(update={"echo_delay_ms": args.echo_delay_ms})
    echo, voice, onsets = asyncio.run(run_check(audio))
    if not (echo.ok and voice.ok and onsets.ok):
        sys.exit(1)


def _gen_token(file: Path, force: bool) -> None:
    try:
        token = generate_token(file, overwrite=force)
    except FileExistsError:
        sys.exit(f"{file.expanduser()} already exists. Use --force to replace it "
                 "(then copy the new token to the other machine).")
    print(f"Wrote a new link token to {file.expanduser()}:\n\n  {token}\n")
    print("Copy this same token into the token file on the other machine.")
