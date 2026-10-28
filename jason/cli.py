"""Command-line entry point: `jason run --role mac|brain`, `jason gen-token`."""

import argparse
import sys
from pathlib import Path

from jason.core.config import DEFAULT_TOKEN_FILE, generate_token


def main() -> None:
    parser = argparse.ArgumentParser(prog="jason")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one machine's role")
    run.add_argument("--role", choices=["mac", "brain"], required=True)
    run.add_argument("--config", help="path to a TOML config file")

    gen = sub.add_parser("gen-token", help="create the link auth token")
    gen.add_argument("--file", type=Path, default=DEFAULT_TOKEN_FILE, help="where to write it")
    gen.add_argument("--force", action="store_true", help="replace an existing token")

    args = parser.parse_args()
    if args.command == "gen-token":
        _gen_token(args.file, args.force)
    else:
        # Roles are wired up in step 10.
        print(f"jason: role={args.role} (not implemented yet)")


def _gen_token(file: Path, force: bool) -> None:
    try:
        token = generate_token(file, overwrite=force)
    except FileExistsError:
        sys.exit(f"{file.expanduser()} already exists. Use --force to replace it "
                 "(then copy the new token to the other machine).")
    print(f"Wrote a new link token to {file.expanduser()}:\n\n  {token}\n")
    print("Copy this same token into the token file on the other machine.")
