"""The committed model files are exactly the pinned ones (assets/README.md)."""

import hashlib

import pytest

from jason.core.config import REPO_ASSETS

PINNED = {
    "silero_vad.onnx": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    "smart-turn-v3.2-cpu.onnx": "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
    "voice_check.wav": "9b13e4e7d39ba472f34f76df0fbf1d0baa24cd652508b8e67dd2346f89181653",
}


@pytest.mark.parametrize("name", PINNED)
def test_asset_matches_its_pinned_hash(name):
    digest = hashlib.sha256((REPO_ASSETS / name).read_bytes()).hexdigest()
    assert digest == PINNED[name], f"{name} changed; if on purpose, update assets/README.md and this test"
