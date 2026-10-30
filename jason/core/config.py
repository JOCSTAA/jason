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
    # Keep the GPU awake during a conversation: on the 4090 Laptop the first
    # token takes ~380 ms after 2+ s idle and ~40 ms when warm
    # (scripts/llm_latency_probe.py). A one-token request every warm_interval_s,
    # until warm_window_s after the user's last sign of activity. 0 turns it off.
    warm_interval_s: Annotated[float, Field(ge=0)] = 0.75
    warm_window_s: Annotated[float, Field(ge=0)] = 60.0


REPO_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"


class AudioConfig(_Strict):
    """The body's mic and speaker, and how it listens. Device names as
    `python -m sounddevice` lists them; empty means the system default."""

    input_device: str = ""
    output_device: str = ""
    # When Jason's voice reaches the mic after it's played, for the echo
    # canceller. PortAudio's own latency figures are unreliable (on the M5 Air
    # they add up to 228 ms; the real delay is ~95 ms), and a wrong hint makes
    # the canceller leak the start of every reply after a silence. Set it from
    # `jason audio-check`; None falls back to PortAudio's figures.
    echo_delay_ms: Annotated[float, Field(ge=0, le=1000)] | None = None
    # Model files and the voice-check clip (committed in the repo's assets/).
    assets_dir: Path | None = None
    # Voice activity (jason/body/listen.py): speech starts after start_chunks
    # 32 ms slices in a row at or above vad_threshold; a pause is pause_ms
    # below vad_off_threshold.
    vad_threshold: Annotated[float, Field(gt=0, lt=1)] = 0.5
    vad_off_threshold: Annotated[float, Field(gt=0, lt=1)] = 0.35
    start_chunks: Annotated[int, Field(ge=1)] = 3
    pause_ms: Annotated[float, Field(gt=0)] = 200.0
    preroll_ms: Annotated[float, Field(ge=0)] = 300.0  # audio kept from just before speech starts
    new_utterance_after_s: Seconds = 3.0  # silence that ends an utterance nobody acted on
    # Just after Jason starts speaking (from silence) the echo canceller can
    # leak his first word, which looked like the user cutting in: in the first
    # live barge-in test 7 of 15 barge-ins were his own first word (96-352 ms
    # of "speech", 0.3-0.5 s into his reply). For onset_guard_ms after he
    # starts, speech must last onset_guard_speech_ms to count.
    onset_guard_ms: Annotated[float, Field(ge=0)] = 600.0
    onset_guard_speech_ms: Annotated[float, Field(ge=0)] = 400.0

    def assets(self) -> Path:
        return self.assets_dir or REPO_ASSETS


class SttConfig(_Strict):
    """Speech-to-text (jason/body/stt.py). The backend is swappable so a
    non-Mac body can use another one; only Parakeet on MLX exists so far."""

    backend: Literal["parakeet-mlx"] = "parakeet-mlx"
    model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    revision: str = "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"  # pinned: a new upload can't change Jason's ears
    partial_every_s: Seconds = 0.5  # re-transcribe the utterance this often while the user speaks


class TtsConfig(_Strict):
    """Text-to-speech (jason/body/tts.py): swappable backend; only Kokoro on
    MLX exists so far."""

    backend: Literal["kokoro-mlx"] = "kokoro-mlx"
    model: str = "mlx-community/Kokoro-82M-bf16"
    revision: str = "a71e4d38b236d968966a2002c4c895dbd12b1c3c"
    voice: str = "am_puck"
    speed: Annotated[float, Field(gt=0.5, lt=2.0)] = 1.1
    # A clause after the first with fewer words than this is held and spoken
    # with the next one ("Josh." alone sounds choppy). The first never waits.
    min_words: Annotated[int, Field(ge=1)] = 3
    # Kokoro pads each clause with ~1 s of silence; it's trimmed and replaced by
    # a pause that fits the clause's last punctuation.
    lead_ms: Annotated[float, Field(ge=0)] = 30.0
    pause_comma_ms: Annotated[float, Field(ge=0)] = 120.0
    pause_sentence_ms: Annotated[float, Field(ge=0)] = 250.0
    pause_other_ms: Annotated[float, Field(ge=0)] = 80.0


class FloorConfig(_Strict):
    """The rule-based floor controller (jason/body/floor.py): how long a pause
    it waits for before answering, by how finished Smart Turn says the user
    sounds. M4 replaces these hand-picked numbers with a learned policy."""

    sure_prob: Annotated[float, Field(gt=0, lt=1)] = 0.7  # at or above: sounds finished
    unsure_prob: Annotated[float, Field(gt=0, lt=1)] = 0.3  # below: sounds mid-thought
    wait_sure_ms: Annotated[float, Field(ge=0)] = 200.0
    wait_unsure_ms: Annotated[float, Field(ge=0)] = 700.0
    wait_unlikely_ms: Annotated[float, Field(ge=0)] = 1500.0
    tick_ms: Annotated[float, Field(gt=0)] = 20.0  # how often it looks (100 ms cost ~40 ms per turn on average)
    policy_version: str = "rules-2"  # bump whenever the rule changes: logged decisions say which rule made them


class BodyConfig(_Strict):
    role: Literal["body"]
    machine: MachineName
    state_dir: Path
    link: BodyLink
    audio: AudioConfig = AudioConfig()
    stt: SttConfig = SttConfig()
    tts: TtsConfig = TtsConfig()
    floor: FloorConfig = FloorConfig()


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
    audio = raw.get("audio", {})
    if "assets_dir" in audio:
        audio["assets_dir"] = _resolve(audio["assets_dir"], base)
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
