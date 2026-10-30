"""TOML config loading, plus the link auth token.

Each role has one config file (config/body.toml, config/brain.toml). Settings
that change when the brain moves (its address, where state lives) are here,
not in code. The auth token is never in a config file or in git: it lives in
its own file, or in the JASON_LINK_TOKEN environment variable.
"""

import os
import secrets
import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

TOKEN_ENV = "JASON_LINK_TOKEN"
DEFAULT_TOKEN_FILE = Path("~/jason-data/link_token")
_MIN_TOKEN_CHARS = 32

# Same rule as the raw log: the machine name becomes a folder name.
MachineName = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$")]
Seconds = Annotated[float, Field(gt=0)]


class _Strict(BaseModel):
    # extra="forbid": a misspelled setting is an error, not silently ignored.
    model_config = ConfigDict(frozen=True, extra="forbid")


class BodyLink(_Strict):
    brain_url: Annotated[str, Field(pattern=r"^wss?://")]
    token_file: Path
    heartbeat_s: Seconds = 2.0  # how often to check the link is alive
    timeout_s: Seconds = 10.0  # this long with no reply means the link is dead
    reconnect_max_s: Seconds = 10.0  # longest wait between reconnect attempts
    clock_sync_s: Seconds = 5.0  # how often to measure clock offset and round trip


class BrainLink(_Strict):
    listen_host: str = "0.0.0.0"
    listen_port: Annotated[int, Field(ge=0, le=65535)] = 8765  # 0 = any free port (tests)
    token_file: Path
    heartbeat_s: Seconds = 2.0
    timeout_s: Seconds = 10.0


class LlmConfig(_Strict):
    """The brain's LLM component: where vLLM is and how to prompt it."""

    base_url: Annotated[str, Field(pattern=r"^https?://")]  # vLLM's OpenAI-compatible server
    model: str = "jason-llm"  # the --served-model-name
    # Which weights answer, written into every agent_utterance so nightly
    # training stays attributable. Change it whenever the served weights change.
    model_version: str
    persona_file: Path  # the system prompt (core blocks, design §8.1)
    history_turns: Annotated[int, Field(ge=0)] = 20  # user + Jason turns kept in the prompt
    max_tokens: Annotated[int, Field(gt=0)] = 256
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.7
    timeout_s: Seconds = 10.0  # longest wait to connect or for the next streamed piece


class BodyConfig(_Strict):
    role: Literal["body"]
    machine: MachineName
    state_dir: Path
    link: BodyLink


class BrainConfig(_Strict):
    role: Literal["brain"]
    machine: MachineName
    state_dir: Path
    link: BrainLink
    llm: LlmConfig | None = None  # no [llm] section: the brain runs without an LLM (tests)


Config = BodyConfig | BrainConfig


def _resolve(value: str, base: Path) -> str:
    # "~" is the home folder; a relative path is relative to the config file,
    # so the same config works whatever folder Jason is started from.
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def load_config(path: Path, role: Literal["body", "brain"]) -> Config:
    """Load and validate a config file, checking it is for the expected role."""
    path = Path(path)
    with open(path, "rb") as f:
        try:
            raw = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            # The usual cause on Windows: "C:\Users\..." in double quotes, where
            # TOML reads \U as the start of an escape code.
            raise ValueError(
                f"{path} is not valid TOML ({e}). If it contains a Windows path, write it "
                "with forward slashes (\"C:/Users/...\") or in single quotes ('C:\\Users\\...')."
            ) from None
    if raw.get("role") != role:
        raise ValueError(f"{path} is a config for role {raw.get('role')!r}, not {role!r}")
    base = path.parent
    if "state_dir" in raw:
        raw["state_dir"] = _resolve(raw["state_dir"], base)
    link = raw.get("link", {})
    if "token_file" in link:
        link["token_file"] = _resolve(link["token_file"], base)
    llm = raw.get("llm", {})
    if "persona_file" in llm:
        llm["persona_file"] = _resolve(llm["persona_file"], base)
    model = BodyConfig if role == "body" else BrainConfig
    return model.model_validate(raw)


class TokenError(Exception):
    pass


def load_token(token_file: Path) -> str:
    """The link token: JASON_LINK_TOKEN if set, otherwise the token file."""
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token and Path(token_file).is_dir():
        # What Docker leaves behind when it mounts a token file that didn't exist.
        raise TokenError(
            f"{token_file} is a folder, not a token file. If Docker created it, "
            "delete it on the host, run `jason gen-token`, and start again."
        )
    if not token:
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            raise TokenError(
                f"no link token at {token_file}. Run `jason gen-token` on one machine, "
                f"then copy the same token to the other (or set {TOKEN_ENV})."
            ) from None
    if len(token) < _MIN_TOKEN_CHARS:
        raise TokenError(f"link token is too short ({len(token)} chars, need {_MIN_TOKEN_CHARS}+)")
    return token


def generate_token(token_file: Path, overwrite: bool = False) -> str:
    """Write a new random token to token_file and return it.

    Refuses to replace an existing token unless overwrite=True, because the
    other machine would stop being able to connect until it gets the new one.
    """
    token_file = Path(token_file).expanduser()
    if token_file.exists() and not overwrite:
        raise FileExistsError(f"{token_file} already exists; pass overwrite=True to replace it")
    token = secrets.token_hex(32)  # 32 random bytes = 64 hex characters
    token_file.parent.mkdir(parents=True, exist_ok=True)
    # Created readable by the owner only (0600) from the first byte, so there is
    # no moment where other users could read it. Windows ignores the mode.
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token + "\n")
    return token
