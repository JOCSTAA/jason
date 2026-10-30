"""The body's audio I/O, driven by a fake PortAudio stream (no hardware).

Needs the body's audio libraries (`uv sync --extra body`); skipped without them.
The real mic and speaker are tested by `jason audio-check`, opt-in below.
"""

import asyncio
import os
import threading
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
soxr = pytest.importorskip("soxr")

from jason.body.audio import BLOCK, IO_SR, MIC_SR, AudioIO, Playback  # noqa: E402
from jason.body.audio_check import delay_ms, score, speechlike_signal, LEAD_S, SIGNAL_S, TAIL_S  # noqa: E402


class FakeStream:
    """Stands in for sounddevice.Stream: the test calls the callback itself."""

    latency = (0.02, 0.03)

    def __init__(self, callback, **kw):
        self.callback, self.kw = callback, kw
        self.started = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def block(self, mic, now=1.0):
        """Run one 10 ms block: mic samples in, what the speaker plays out."""
        out = np.zeros((BLOCK, 2), dtype=np.float32)
        t = SimpleNamespace(currentTime=now, outputBufferDacTime=now + 0.03, inputBufferAdcTime=now - 0.02)
        self.callback(np.asarray(mic, dtype=np.float32).reshape(-1, 1), out, BLOCK, t, None)
        return out


class FakeCanceller:
    """Records what the echo canceller was given; 'cleans' by halving."""

    def __init__(self, delay_hint_ms):
        self.delay_hint_ms = delay_hint_ms
        self.far_blocks, self.near_blocks = [], []

    def far(self, block):
        self.far_blocks.append(block.copy())

    def near(self, block):
        self.near_blocks.append(block.copy())
        return block * 0.5


@pytest.fixture
async def io():
    streams = []

    def factory(**kw):
        streams.append(FakeStream(**kw))
        return streams[-1]

    audio = AudioIO(stream_factory=factory, canceller=FakeCanceller)
    audio.start()
    audio.fake = streams[0]
    yield audio
    audio.stop()


def silence():
    return np.zeros(BLOCK, dtype=np.float32)


# --- the stream -------------------------------------------------------------------


async def test_stream_is_48k_in_10ms_blocks_mono_in_stereo_out(io):
    kw = io.fake.kw
    assert (kw["samplerate"], kw["blocksize"], kw["channels"]) == (IO_SR, BLOCK, (1, 2))
    assert io.fake.started
    assert io.canceller.delay_hint_ms == 50  # the stream's input + output latency


async def test_what_plays_is_the_echo_reference(io):
    tone = np.sin(np.arange(BLOCK * 3) / 10).astype(np.float32) * 0.3
    io.play("r1", tone, IO_SR)
    outs = [io.fake.block(silence())[:, 0] for _ in range(3)]
    assert np.allclose(np.concatenate(outs), tone)
    assert np.allclose(np.concatenate(io.canceller.far_blocks), tone)
    assert np.allclose(io.fake.block(silence())[:, 1], 0)  # queue empty: silence, both channels


async def test_mic_audio_is_cleaned_then_resampled_to_16k(io):
    n_blocks = 50
    t = np.arange(BLOCK * n_blocks) / IO_SR
    mic = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    for i in range(n_blocks):
        io.fake.block(mic[i * BLOCK:(i + 1) * BLOCK])
    await asyncio.sleep(0)  # deliveries are scheduled onto the loop
    frames = []
    async for f in io.frames():
        frames.append(f)
        if len(frames) == n_blocks:
            break
    got = np.concatenate([f.samples for f in frames])
    # Same as resampling the whole cleaned signal at once: block edges leave no seams.
    expected = soxr.resample(mic * 0.5, IO_SR, MIC_SR)
    n = min(len(got), len(expected)) - 200
    assert abs(len(got) - len(mic) * MIC_SR // IO_SR) < 200
    assert np.max(np.abs(got[200:n] - expected[200:n])) < 0.01


async def test_mic_frames_arrive_from_the_audio_thread(io):
    thread = threading.Thread(target=lambda: io.fake.block(silence() + 0.1))
    thread.start()
    thread.join()
    frame = await asyncio.wait_for(anext(io.frames()), 1)
    assert frame.audio_ns > 0


async def test_a_slow_consumer_loses_the_oldest_audio(tmp_path):
    streams = []
    audio = AudioIO(stream_factory=lambda **kw: streams.append(FakeStream(**kw)) or streams[-1],
                    canceller=FakeCanceller, mic_queue_size=3)
    audio.start()
    for i in range(5):
        streams[0].block(silence() + i / 10)
    await asyncio.sleep(0)
    assert audio.stats.mic_frames_dropped == 2
    audio.stop()


# --- playback and the playhead ---------------------------------------------------------


def test_playhead_counts_what_reached_the_speaker():
    pb = Playback()
    pb.add("r1", np.ones(BLOCK * 2 + 100, dtype=np.float32))
    pb.next_block(BLOCK, dac_ns=5_000)
    p = pb.played("r1")
    assert (p.played, p.first_ns, p.finished) == (BLOCK, 5_000, False)
    pb.next_block(BLOCK, dac_ns=15_000)
    pb.next_block(BLOCK, dac_ns=25_000)
    p = pb.played("r1")
    assert (p.played, p.finished, p.first_ns) == (BLOCK * 2 + 100, True, 5_000)
    assert p.played_ms == pytest.approx((BLOCK * 2 + 100) * 1000 / IO_SR)


def test_first_sample_time_accounts_for_its_position_in_the_block():
    pb = Playback()
    pb.add("a", np.ones(BLOCK // 2, dtype=np.float32))
    pb.add("b", np.ones(BLOCK, dtype=np.float32))
    pb.next_block(BLOCK, dac_ns=0)
    # "b" starts halfway through the block: 5 ms after it reaches the speaker.
    assert pb.played("b").first_ns == 5_000_000


def test_pieces_play_in_order_across_tags():
    pb = Playback()
    pb.add("a", np.full(300, 1.0, dtype=np.float32))
    pb.add("b", np.full(300, 2.0, dtype=np.float32))
    out = pb.next_block(BLOCK, dac_ns=0)
    assert list(out[:300]) == [1.0] * 300 and list(out[300:480]) == [2.0] * 180
    assert pb.played("b").played == 180


def test_stop_drops_queued_audio_and_marks_it():
    pb = Playback()
    pb.add("r1", np.ones(BLOCK * 5, dtype=np.float32))
    pb.next_block(BLOCK, dac_ns=0)
    pb.stop()
    assert not pb.busy
    assert np.all(pb.next_block(BLOCK, dac_ns=0) == 0)
    p = pb.played("r1")
    assert (p.played, p.stopped, p.finished) == (BLOCK, True, False)


def test_stop_one_tag_keeps_the_others():
    pb = Playback()
    pb.add("old", np.ones(BLOCK, dtype=np.float32))
    pb.add("new", np.full(BLOCK, 2.0, dtype=np.float32))
    pb.stop("old")
    assert np.all(pb.next_block(BLOCK, dac_ns=0) == 2.0)
    assert pb.played("old").stopped and not pb.played("new").stopped


async def test_play_resamples_to_48k(io):
    io.play("r1", np.zeros(24_000, dtype=np.float32), 24_000)  # Kokoro's rate
    assert io.playback.played("r1").total == pytest.approx(IO_SR, abs=10)


# --- audio-check scoring --------------------------------------------------------------


def test_delay_is_found_from_the_raw_mic():
    sig = speechlike_signal(2.0)
    lag = int(0.111 * IO_SR)
    mic = np.concatenate([np.zeros(lag), sig[:-lag]]) * 0.1
    assert delay_ms(sig, mic) == pytest.approx(111, abs=1)


def test_score_passes_good_cancellation_and_fails_bad():
    rng = np.random.default_rng(1)
    n = int((LEAD_S + SIGNAL_S + TAIL_S) * IO_SR)
    timeline = np.zeros(n, dtype=np.float32)
    timeline[int(LEAD_S * IO_SR):int(LEAD_S * IO_SR) + int(SIGNAL_S * IO_SR)] = speechlike_signal()
    noise = rng.normal(0, 0.002, n).astype(np.float32)  # about -54 dBFS, like the spike's room
    raw = np.roll(timeline, int(0.1 * IO_SR)) * 0.1 + noise
    good = score(timeline, raw, noise * 0.9, status_flags=0, blocks=1000)
    assert good.ok and good.reduction_db > 15
    bad = score(timeline, raw, raw * 0.5, status_flags=0, blocks=1000)
    assert not bad.ok


# --- real hardware (opt-in) ----------------------------------------------------------


@pytest.mark.skipif(os.environ.get("JASON_AUDIO_TESTS") != "1",
                    reason="set JASON_AUDIO_TESTS=1 to play sound and use the mic (stay quiet)")
async def test_real_audio_check():
    from jason.body.audio_check import run_check
    from jason.core.config import AudioConfig

    result = await run_check(AudioConfig(), out=lambda _: None)
    assert result.ok, "\n".join(result.lines())
