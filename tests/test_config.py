import os
import stat
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from jason.core.config import (
    TOKEN_ENV,
    BrainConfig,
    BodyConfig,
    TokenError,
    generate_token,
    load_config,
    load_token,
)

REPO_CONFIG = Path(__file__).parent.parent / "config"

BODY_TOML = """
role = "body"
machine = "mac"
state_dir = "{state_dir}"
[link]
brain_url = "ws://jason-brain:8765"
token_file = "{token_file}"
"""


def write(tmp_path, text, name="cfg.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def body_cfg(tmp_path, state_dir="state", token_file="link_token", **replace):
    text = BODY_TOML.format(state_dir=state_dir, token_file=token_file)
    for old, new in replace.items():
        text = text.replace(old, new)
    return write(tmp_path, text)


# --- the committed config files ---------------------------------------------


def test_committed_body_config_loads():
    cfg = load_config(REPO_CONFIG / "body.toml", "body")
    assert isinstance(cfg, BodyConfig)
    assert cfg.state_dir == Path("~/jason-data/state").expanduser()
    assert cfg.link.brain_url == "ws://jason-brain:8765"


def test_committed_brain_config_loads():
    cfg = load_config(REPO_CONFIG / "brain.toml", "brain")
    assert isinstance(cfg, BrainConfig)
    # This config is for the Linux container, where it is exactly /state. Read on
    # Windows (no drive letter), "/state" becomes C:\state; still a /state folder.
    assert cfg.state_dir.as_posix().endswith("/state")
    assert cfg.link.listen_port == 8765
    # The LLM is reached inside compose, and the persona sits next to the config.
    assert cfg.llm.base_url == "http://llm:8000"
    assert cfg.llm.persona_file == REPO_CONFIG.resolve() / "persona.md"
    assert cfg.llm.persona_file.read_text(encoding="utf-8").strip()


def test_brain_runs_without_an_llm_section(tmp_path):
    path = tmp_path / "brain.toml"
    path.write_text('role = "brain"\nmachine = "brain"\nstate_dir = "s"\n[link]\ntoken_file = "t"\n')
    assert load_config(path, "brain").llm is None


# --- paths ------------------------------------------------------------------


def test_relative_paths_are_relative_to_the_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir("/")  # the working directory must not matter
    cfg = load_config(body_cfg(tmp_path), "body")
    assert cfg.state_dir == (tmp_path / "state").resolve()
    assert cfg.link.token_file == (tmp_path / "link_token").resolve()


def test_tilde_means_home(tmp_path):
    cfg = load_config(body_cfg(tmp_path, state_dir="~/somewhere"), "body")
    assert cfg.state_dir == Path.home() / "somewhere"


# --- validation -------------------------------------------------------------


def test_wrong_role_rejected():
    with pytest.raises(ValueError, match="role"):
        load_config(REPO_CONFIG / "brain.toml", "body")


def test_misspelled_setting_rejected(tmp_path):
    path = body_cfg(tmp_path)
    path.write_text(path.read_text() + "heartbeat_secs = 1.0\n")
    with pytest.raises(ValidationError):
        load_config(path, "body")


def test_brain_url_must_be_websocket(tmp_path):
    path = body_cfg(tmp_path, **{"ws://jason-brain:8765": "http://jason-brain:8765"})
    with pytest.raises(ValidationError):
        load_config(path, "body")


def test_unsafe_machine_name_rejected(tmp_path):
    path = body_cfg(tmp_path, **{'machine = "mac"': 'machine = "../mac"'})
    with pytest.raises(ValidationError):
        load_config(path, "body")


def test_bad_port_rejected(tmp_path):
    path = write(tmp_path, """
role = "brain"
machine = "brain"
state_dir = "state"
[link]
listen_port = 70000
token_file = "t"
""")
    with pytest.raises(ValidationError):
        load_config(path, "brain")


def test_negative_timeout_rejected(tmp_path):
    path = body_cfg(tmp_path)
    path.write_text(path.read_text() + "timeout_s = -1\n")
    with pytest.raises(ValidationError):
        load_config(path, "body")


# --- token ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_token_env(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)


def test_generate_and_load_token(tmp_path):
    file = tmp_path / "sub" / "link_token"
    token = generate_token(file)
    assert len(token) == 64 and int(token, 16) >= 0  # 64 hex characters
    assert load_token(file) == token


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no Unix file modes")
def test_token_file_is_private(tmp_path):
    file = tmp_path / "link_token"
    generate_token(file)
    assert stat.S_IMODE(os.stat(file).st_mode) == 0o600


def test_generate_refuses_to_overwrite(tmp_path):
    file = tmp_path / "link_token"
    first = generate_token(file)
    with pytest.raises(FileExistsError):
        generate_token(file)
    assert load_token(file) == first
    assert generate_token(file, overwrite=True) != first


def test_tokens_are_random(tmp_path):
    a = generate_token(tmp_path / "a")
    b = generate_token(tmp_path / "b")
    assert a != b


def test_env_var_overrides_file(tmp_path, monkeypatch):
    file = tmp_path / "link_token"
    generate_token(file)
    monkeypatch.setenv(TOKEN_ENV, "e" * 64)
    assert load_token(file) == "e" * 64


def test_missing_token_explains_what_to_do(tmp_path):
    with pytest.raises(TokenError, match="jason gen-token"):
        load_token(tmp_path / "nope")


def test_short_token_rejected(tmp_path):
    file = tmp_path / "link_token"
    file.write_text("password\n")
    with pytest.raises(TokenError, match="too short"):
        load_token(file)


def test_token_path_that_is_a_folder_explains_the_docker_cause(tmp_path):
    folder = tmp_path / "link_token"
    folder.mkdir()  # what Docker creates when the mounted file is missing
    with pytest.raises(TokenError, match="is a folder"):
        load_token(folder)


def test_windows_backslash_path_gets_a_helpful_error(tmp_path):
    path = write(tmp_path, BODY_TOML.format(state_dir="C:\\Users\\me\\state", token_file="t"))
    with pytest.raises(ValueError, match="forward slashes"):
        load_config(path, "body")


def test_windows_paths_in_forward_slashes_or_single_quotes_load(tmp_path):
    text = BODY_TOML.format(state_dir="C:/Users/me/state", token_file="t")
    text = text.replace('token_file = "t"', "token_file = 'C:\\Users\\me\\link_token'")
    load_config(write(tmp_path, text), "body")
