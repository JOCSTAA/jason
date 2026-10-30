"""Smart Turn's features (numpy log-mel) and model. Needs the body extra."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")

from jason.body.turn import SECONDS, SR, SmartTurn, log_mel, mel_filters  # noqa: E402
from jason.core.config import REPO_ASSETS  # noqa: E402

REFERENCE = Path(__file__).parent / "data" / "whisper_logmel_reference.npz"


def test_matches_transformers_whisper_features():
    # Made once with transformers' WhisperFeatureExtractor(chunk_length=8), as
    # Smart Turn's reference code uses it, on 5 s of the voice-check clip.
    ref = np.load(REFERENCE)
    audio = ref["audio_int16"].astype(np.float32) / 32767
    got = log_mel(audio)
    assert got.shape == ref["features"].shape == (1, 80, 800)
    assert np.max(np.abs(got - ref["features"])) < 1e-4


def test_mel_filters_shape_and_equal_area():
    f = mel_filters()
    assert f.shape == (201, 80)
    assert np.all(f >= 0)
    # Slaney normalization: every band has the same area (2 / bandwidth × half the bandwidth).
    hz_per_bin = (SR / 2) / 200
    assert np.allclose(f.sum(axis=0) * hz_per_bin, 1.0, rtol=0.15)


def test_short_audio_is_padded_in_front_and_long_audio_keeps_the_end():
    rng = np.random.default_rng(0)
    speech = rng.standard_normal(SR).astype(np.float32) * 0.1
    short = log_mel(speech)
    # The first 7 s are padding: steady over time in every band (normalization
    # shifts the zeros, so each band sits at its own constant level); the last
    # second carries the sound and varies.
    assert np.ptp(short[0, :, :600], axis=1).max() < 1e-6
    assert np.ptp(short[0, :, -90:], axis=1).max() > 0.5
    long = np.concatenate([np.zeros(SECONDS * SR, dtype=np.float32), speech])
    assert np.allclose(log_mel(long), short, atol=1e-5)  # only the last 8 s count


def test_smart_turn_gives_a_probability():
    model = SmartTurn(REPO_ASSETS / "smart-turn-v3.2-cpu.onnx")
    ref = np.load(REFERENCE)
    p = model(ref["audio_int16"].astype(np.float32) / 32767)
    assert 0.0 <= p <= 1.0
