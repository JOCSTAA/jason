"""Text-to-speech: clauses become audio in order, tagged by reply; short
clauses after the first are merged; reply_heard once playback ends. Fake
backend and a real Playback queue; real Kokoro is opt-in at the end."""

import asyncio
import os
import threading
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("soxr")

from jason.body.audio import BLOCK, IO_SR, Playback  # noqa: E402
from jason.body.tts import Speaker, trim  # noqa: E402
from jason.core import events as ev  # noqa: E402
from jason.core.bus import EventBus  # noqa: E402
from jason.core.config import TtsConfig  # noqa: E402
from jason.log.rawlog import RawLog  # noqa: E402

SESSION = "day-0001_2026-10-30"


class FakeBackend:
    """Each word becomes 10 ms of audio at 48 kHz, so lengths are predictable."""

    sample_rate = IO_SR

    def __init__(self, delay=0.0):
        self.delay, self.texts, self.threads = delay, [], set()

    def load(self):
        pass

    def synthesize(self, text):
        self.threads.add(threading.current_thread().name)
        self.texts.append(text)
        time.sleep(self.delay)
        return np.ones(BLOCK * len(text.split()), dtype=np.float32)


class FakeAudio:
    """The real Playback queue; tests play it by calling tick()."""

    def __init__(self):
        self.playback = Playback()

    def play(self, tag, samples, sample_rate):
        assert sample_rate == IO_SR
        self.playback.add(tag, samples)

    def tick(self, blocks=1):
        for _ in range(blocks):
            self.playback.next_block(BLOCK, dac_ns=time.monotonic_ns())


@pytest.fixture
async def setup(tmp_path):
    rawlog = RawLog(tmp_path, "mac")
    bus = EventBus("mac", rawlog, session_id=SESSION)
    out = bus.subscribe(ev.LatencyMark, ev.ReplyHeard)
    audio, backend = FakeAudio(), FakeBackend()
    # No added pauses, so the audio lengths below are exact (trim() is tested on its own).
    speaker = Speaker(bus, audio, backend, TtsConfig(lead_ms=0, pause_comma_ms=0, pause_sentence_ms=0, pause_other_ms=0))
    task = asyncio.create_task(speaker.run())
    await asyncio.sleep(0)
    yield bus, audio, backend, out
    task.cancel()
    rawlog.close()


def chunk(bus, action, index, text, last=False):
    bus.publish(ev.ReplyChunk(machine="brain", session_id=SESSION, payload=ev.ReplyChunkPayload(
        action_id=action, index=index, text=text, last=last)))


async def settle():
    for _ in range(20):
        await asyncio.sleep(0.01)


async def test_clauses_are_spoken_in_order_under_the_replys_tag(setup):
    bus, audio, backend, out = setup
    chunk(bus, "a1", 0, "Sure thing,")
    chunk(bus, "a1", 1, "I can help with that.")
    chunk(bus, "a1", 2, "", last=True)
    await settle()
    assert backend.texts == ["Sure thing,", "I can help with that."]
    assert audio.playback.played("a1").total == BLOCK * 7  # 2 + 5 words


async def test_a_short_clause_after_the_first_is_spoken_with_the_next(setup):
    bus, audio, backend, out = setup
    chunk(bus, "a1", 0, "Got it,")  # first: never held, even though short
    chunk(bus, "a1", 1, "Josh.")
    chunk(bus, "a1", 2, "That sounds like a specific check.")
    chunk(bus, "a1", 3, "Okay.", last=True)  # the last is never held either
    await settle()
    assert backend.texts == ["Got it,", "Josh. That sounds like a specific check.", "Okay."]


async def test_first_chunk_latency_is_logged_once_per_reply(setup):
    bus, audio, backend, out = setup
    chunk(bus, "a1", 0, "Hello there,")
    chunk(bus, "a1", 1, "how are you today?", last=True)
    await settle()
    marks = [e for e in list(out._queue._queue) if isinstance(e, ev.LatencyMark)]
    assert [(m.payload.stage, m.payload.ref_id) for m in marks] == [(ev.LatencyStage.TTS_FIRST_CHUNK, "a1")]


async def test_reply_heard_comes_once_the_last_sample_has_played(setup):
    bus, audio, backend, out = setup
    chunk(bus, "a1", 0, "Hello there,")
    chunk(bus, "a1", 1, "how are you?", last=True)
    await settle()
    assert out._queue.qsize() == 1  # only the latency mark: nothing played yet
    audio.tick(3)
    await settle()
    assert out._queue.qsize() == 1  # 3 of 5 blocks played
    audio.tick(2)
    heard = None
    for _ in range(50):
        await asyncio.sleep(0.01)
        items = [e for e in list(out._queue._queue) if isinstance(e, ev.ReplyHeard)]
        if items:
            heard = items[0]
            break
    assert heard is not None
    assert (heard.payload.text, heard.payload.complete) == ("Hello there, how are you?", True)
    assert heard.payload.played_ms == pytest.approx(50.0)


async def test_a_newer_reply_stops_the_old_one(setup):
    bus, audio, backend, out = setup
    chunk(bus, "old", 0, "This is a long answer that")
    await settle()
    chunk(bus, "new", 0, "Actually, never mind.", last=True)
    await settle()
    assert audio.playback.played("old").stopped
    audio.tick(10)
    assert audio.playback.played("new").finished


async def test_synthesis_runs_on_one_thread(setup):
    bus, audio, backend, out = setup
    for i in range(4):
        chunk(bus, "a1", i, f"Clause number {i} here,", last=i == 3)
    await settle()
    assert len(backend.threads) == 1 and next(iter(backend.threads)).startswith("tts")


# --- trimming Kokoro's padding ------------------------------------------------------


def padded(sound_s=1.0, before_s=0.4, after_s=0.6, sr=24_000):
    return np.concatenate([np.zeros(int(before_s * sr)), 0.5 * np.ones(int(sound_s * sr)),
                           np.zeros(int(after_s * sr))]).astype(np.float32)


@pytest.mark.parametrize("text,pause_ms", [("Nice one,", 120), ("Serious work ahead.", 250),
                                           ("Really?", 250), ("and then", 80)])
def test_trim_keeps_the_sound_and_a_pause_that_fits_the_punctuation(text, pause_ms):
    sr = 24_000
    out = trim(padded(), sr, text, TtsConfig())
    # 30 ms lead + 1 s of sound + 30 ms decay + the pause, instead of 2 s with Kokoro's padding.
    assert len(out) / sr == pytest.approx(0.030 + 1.0 + 0.030 + pause_ms / 1000, abs=0.002)
    assert np.count_nonzero(out) == sr  # all of the sound survives


def test_trim_keeps_quiet_onsets():
    sr = 24_000
    x = padded()
    onset = int(0.4 * sr)
    x[onset - int(0.05 * sr):onset] = 0.01  # a soft "s" just before the loud part (2% of peak)
    out = trim(x, sr, "Sure.", TtsConfig())
    assert np.count_nonzero(out) == sr + int(0.05 * sr)


def test_trim_of_silence_is_empty():
    assert len(trim(np.zeros(1000, dtype=np.float32), 24_000, "Hm.", TtsConfig())) == 0


@pytest.mark.skipif(os.environ.get("JASON_TTS_TESTS") != "1",
                    reason="set JASON_TTS_TESTS=1 to run real Kokoro (downloads 330 MB the first time)")
def test_real_kokoro_speaks():
    from jason.body.tts import make_backend

    tts = make_backend(TtsConfig())
    tts.load()
    t = time.perf_counter()
    audio = tts.synthesize("Sure thing, I can help with that.")
    elapsed = time.perf_counter() - t
    assert len(audio) / tts.sample_rate > 1.0  # about two seconds of speech
    assert elapsed < 1.0
