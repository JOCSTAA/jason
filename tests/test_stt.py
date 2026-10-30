"""Speech-to-text: partials while speaking, a final at every pause. Fake
backend and listener make it exact; real Parakeet is opt-in at the end."""

import asyncio
import os
import threading
import time

import pytest

np = pytest.importorskip("numpy")

from jason.body.stt import Transcriber  # noqa: E402
from jason.core import events as ev  # noqa: E402
from jason.core.bus import EventBus  # noqa: E402
from jason.core.config import SttConfig  # noqa: E402
from jason.log.rawlog import RawLog  # noqa: E402

SESSION = "day-0001_2026-10-30"


class FakeBackend:
    """'Transcribes' audio as how long it is; remembers which thread it ran on."""

    def __init__(self, delay=0.0, text=None):
        self.delay, self.text = delay, text
        self.calls, self.threads = [], set()

    def load(self):
        self.threads.add(threading.current_thread().name)

    def transcribe(self, audio):
        self.threads.add(threading.current_thread().name)
        self.calls.append(len(audio))
        time.sleep(self.delay)
        return self.text if self.text is not None else f"{len(audio)} samples"


class FakeListener:
    def __init__(self):
        self.utterance_id = "u1"
        self.speaking = False
        self.audio = np.zeros(16_000, dtype=np.float32)

    def utterance_audio(self):
        return self.audio


@pytest.fixture
async def setup(tmp_path):
    rawlog = RawLog(tmp_path, "mac")
    bus = EventBus("mac", rawlog, session_id=SESSION)
    out = bus.subscribe(ev.UserPartial, ev.UserFinal, ev.LatencyMark)
    tasks = []

    async def start(backend=None, every=0.02):
        backend = backend or FakeBackend()
        listener = FakeListener()
        t = Transcriber(bus, listener, backend, SttConfig(partial_every_s=every))
        await t.load()
        tasks.append(asyncio.create_task(t.run()))
        await asyncio.sleep(0)
        return bus, listener, backend, out

    yield start
    for task in tasks:
        task.cancel()
    rawlog.close()


def pause(bus, uid="u1"):
    bus.publish(ev.SpeechEnd(machine="mac", session_id=SESSION, payload=ev.SpeechEndPayload(
        utterance_id=uid, audio_ns=1, speech_ms=900.0)))


async def next_of(out, cls, timeout=2.0):
    async with asyncio.timeout(timeout):
        while True:
            e = await out.get()
            if isinstance(e, cls):
                return e


async def test_a_pause_gives_a_final_transcript_and_its_latency(setup):
    bus, listener, backend, out = await setup()
    pause(bus)
    final = await next_of(out, ev.UserFinal)
    assert (final.payload.utterance_id, final.payload.text) == ("u1", "16000 samples")
    mark = await next_of(out, ev.LatencyMark)
    assert (mark.payload.stage, mark.payload.ref_id) == (ev.LatencyStage.STT_FINAL, final.id)


async def test_partials_come_while_speaking_and_only_when_the_text_changes(setup):
    bus, listener, backend, out = await setup()
    listener.speaking = True
    first = await next_of(out, ev.UserPartial)
    await asyncio.sleep(0.1)  # same audio, same text: no new partials
    assert out._queue.empty()
    listener.audio = np.zeros(24_000, dtype=np.float32)  # the user said more
    second = await next_of(out, ev.UserPartial)
    assert (first.payload.text, second.payload.text) == ("16000 samples", "24000 samples")


async def test_no_partials_when_not_speaking(setup):
    bus, listener, backend, out = await setup()
    await asyncio.sleep(0.1)
    assert backend.calls == []


async def test_every_pause_gets_a_final_and_the_later_one_covers_more(setup):
    bus, listener, backend, out = await setup()
    pause(bus)
    first = await next_of(out, ev.UserFinal)
    listener.audio = np.zeros(40_000, dtype=np.float32)  # carried on after the pause
    pause(bus)
    second = await next_of(out, ev.UserFinal)
    assert first.payload.utterance_id == second.payload.utterance_id
    assert second.payload.text == "40000 samples"


async def test_noise_that_transcribes_to_nothing_emits_nothing(setup):
    bus, listener, backend, out = await setup(FakeBackend(text=""))
    pause(bus)
    await asyncio.sleep(0.1)
    assert not any(isinstance(e, ev.UserFinal) for e in list(out._queue._queue))


async def test_a_partial_that_finishes_after_a_final_is_dropped(setup):
    bus, listener, backend, out = await setup(FakeBackend(delay=0.15), every=0.01)
    listener.speaking = True
    await asyncio.sleep(0.03)  # a slow partial is now running
    listener.speaking = False
    listener.audio = np.zeros(32_000, dtype=np.float32)
    pause(bus)
    final = await next_of(out, ev.UserFinal, timeout=3)
    await asyncio.sleep(0.2)
    stale = [e for e in list(out._queue._queue) if isinstance(e, ev.UserPartial)]
    assert final.payload.text == "32000 samples"
    assert stale == []


async def test_the_model_only_ever_runs_on_one_thread(setup):
    bus, listener, backend, out = await setup()
    listener.speaking = True
    await next_of(out, ev.UserPartial)
    pause(bus)
    await next_of(out, ev.UserFinal)
    assert len(backend.threads) == 1 and next(iter(backend.threads)).startswith("stt")


@pytest.mark.skipif(os.environ.get("JASON_STT_TESTS") != "1",
                    reason="set JASON_STT_TESTS=1 to run real Parakeet (downloads 2.5 GB the first time)")
# parakeet-mlx's loader opens config.json without closing it (its bug, harmless).
@pytest.mark.filterwarnings("ignore::ResourceWarning", "ignore::pytest.PytestUnraisableExceptionWarning")
def test_real_parakeet_on_the_voice_clip():
    import soxr

    from jason.body.audio_check import read_wav
    from jason.body.stt import make_backend
    from jason.core.config import REPO_ASSETS

    voice, sr = read_wav(REPO_ASSETS / "voice_check.wav")
    stt = make_backend(SttConfig())
    stt.load()
    text = stt.transcribe(soxr.resample(voice, sr, 16_000).astype(np.float32)).lower()
    for word in ("echo", "canceller", "microphone", "interrupt"):
        assert word in text, text
