"""Command-line entry point: `jason run --role mac|brain`."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="jason")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one machine's role")
    run.add_argument("--role", choices=["mac", "brain"], required=True)
    run.add_argument("--config", help="path to a TOML config file")
    args = parser.parse_args()
    # Roles are wired up in step 10.
    print(f"jason: role={args.role} (not implemented yet)")
