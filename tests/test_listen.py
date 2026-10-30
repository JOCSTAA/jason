"""The body's listener: VAD hysteresis, utterances, and Smart Turn at pauses.

A fake VAD (loud slice = speech) and a fake Smart Turn make the rules exact;
the real Silero is tested on the voice clip at the end. Needs the body extra.
"""

import asyncio
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")
pytest.importorskip("soxr")

from jason.body.audio import MicFrame  # noqa: E402
from jason.body.listen import CHUNK, CHUNK_NS, Listener, Silero, VoiceActivity  # noqa: E402
from jason.core import events as ev  # noqa: E402
from jason.core.bus import EventBus  # noqa: E402
from jason.core.config import REPO_ASSETS, AudioConfig  # noqa: E402
from jason.log.rawlog import RawLog  # noqa: E402

SESSION = "day-0001_2026-10-30"
CFG = AudioConfig()  # 0.5 / 0.35, 3 slices to start, 200 ms pause, 3 s to a new utterance


# --- hysteresis -----------------------------------------------------------------------


def feed(va, probs, t0=0):
    out = []
    for i, p in enumerate(probs):
        t = va.update(p, t0 + i * CHUNK_NS)
        if t:
            out.append((i, t))
    return out


def test_speech_needs_three_loud_slices_in_a_row():
    va = VoiceActivity(CFG)
    assert feed(va, [0.9, 0.1, 0.9, 0.9, 0.1]) == []  # a cough, then two: not enough
    [(i, t)] = feed(VoiceActivity(CFG), [0.1, 0.9, 0.9, 0.9])
    # Reported when the third slice arrives, but timed from the first one.
    assert (i, t.kind, t.audio_ns) == (3, "start", 1 * CHUNK_NS)


def test_a_pause_is_200ms_below_the_lower_threshold():
    pause = round(200 / 32)  # 6 slices
    va = VoiceActivity(CFG)
    events = feed(va, [0.9] * 5 + [0.2] * (pause - 1) + [0.4] + [0.2] * pause)
    # The 0.4 dip-and-back is above 0.35: speech continues, so only one end.
    starts = [t for _, t in events if t.kind == "start"]
    ends = [(i, t) for i, t in events if t.kind == "end"]
    assert len(starts) == 1 and len(ends) == 1
    i, end = ends[0]
    last_voiced = 5 + pause - 1  # the 0.4 slice
    assert end.audio_ns == (last_voiced + 1) * CHUNK_NS  # the end of the last voiced slice
    assert end.speech_ms == pytest.approx((last_voiced + 1) * CHUNK_NS / 1e6)


# --- the listener --------------------------------------------------------------------


class FakeAudio:
    def __init__(self):
        self.playback = SimpleNamespace(busy=False)


def loud(prob_audio):
    return 0.9 if float(np.mean(np.abs(prob_audio))) > 0.05 else 0.0


@pytest.fixture
def setup(tmp_path):
    rawlog = RawLog(tmp_path, "mac")
    bus = EventBus("mac", rawlog, session_id=SESSION)
    out = bus.subscribe()
    turn_inputs = []

    def turn(audio):
        turn_inputs.append(audio)
        return 0.8

    audio = FakeAudio()
    listener = Listener(bus, audio, CFG, vad=loud, turn=turn)
    yield SimpleNamespace(bus=bus, out=out, listener=listener, audio=audio, turn_inputs=turn_inputs)
    rawlog.close()


def say(listener, pattern, t0_ns=1_000_000_000):
    """Feed slices: 1 = speech, 0 = silence, in 160-sample frames as AudioIO sends them."""
    x = np.concatenate([np.full(CHUNK, 0.3 if p else 0.0, dtype=np.float32) for p in pattern])
    for i in range(0, len(x), 160):
        listener.process(MicFrame(x[i:i + 160], t0_ns + i * 1_000_000_000 // 16_000))


def drain(out):
    events = []
    while not out._queue.empty():
        events.append(out._queue.get_nowait())
    return events


async def test_speech_then_pause_gives_start_end_and_a_turn_probability(setup):
    say(setup.listener, [0] * 12 + [1] * 20 + [0] * 8)
    await asyncio.sleep(0.05)  # Smart Turn runs in a thread
    events = drain(setup.out)
    start, end, turn = [e for e in events if e.type in ("speech_start", "speech_end", "turn_end_prob")]
    assert (start.type, end.type, turn.type) == ("speech_start", "speech_end", "turn_end_prob")
    assert start.payload.utterance_id == end.payload.utterance_id == turn.payload.utterance_id
    assert start.payload.audio_ns == 1_000_000_000 + 12 * CHUNK_NS  # when the sound began
    assert end.payload.audio_ns == 1_000_000_000 + 32 * CHUNK_NS  # when it stopped
    assert turn.payload.prob == 0.8
    # Smart Turn heard the speech plus ~300 ms from just before it (pre-roll).
    heard = setup.turn_inputs[0]
    assert len(heard) >= 20 * CHUNK + 9 * CHUNK


async def test_a_pause_does_not_start_a_new_utterance(setup):
    say(setup.listener, [1] * 5 + [0] * 8 + [1] * 5 + [0] * 8)
    await asyncio.sleep(0.05)
    ids = {e.payload.utterance_id for e in drain(setup.out) if e.type == "speech_start"}
    assert len(ids) == 1


async def test_silence_long_enough_starts_a_new_utterance(setup):
    three_s = int(3.1 * 16_000 / CHUNK)
    say(setup.listener, [1] * 5 + [0] * three_s + [1] * 5 + [0] * 8)
    await asyncio.sleep(0.05)
    ids = [e.payload.utterance_id for e in drain(setup.out) if e.type == "speech_start"]
    assert len(ids) == 2 and ids[0] != ids[1]


async def test_jason_acting_on_an_utterance_closes_it(setup):
    task = asyncio.create_task(setup.listener._watch_actions())
    await asyncio.sleep(0)
    say(setup.listener, [1] * 5 + [0] * 8)
    first = setup.listener.utterance_id
    setup.bus.publish(ev.AgentAction(machine="mac", session_id=SESSION, payload=ev.AgentActionPayload(
        action=ev.Action.RESPOND, action_probs={ev.Action.RESPOND: 1.0}, utterance_id=first)))
    await asyncio.sleep(0)
    assert setup.listener.utterance_id is None
    say(setup.listener, [1] * 5 + [0] * 8)
    assert setup.listener.utterance_id not in (None, first)
    task.cancel()


async def test_deciding_to_keep_listening_does_not_close_the_utterance(setup):
    task = asyncio.create_task(setup.listener._watch_actions())
    await asyncio.sleep(0)
    say(setup.listener, [1] * 5 + [0] * 8)
    uid = setup.listener.utterance_id
    setup.bus.publish(ev.AgentAction(machine="mac", session_id=SESSION, payload=ev.AgentActionPayload(
        action=ev.Action.SILENT, action_probs={ev.Action.SILENT: 1.0}, utterance_id=uid)))
    await asyncio.sleep(0)
    assert setup.listener.utterance_id == uid
    task.cancel()


async def test_still_talking_when_jason_acts_opens_a_new_utterance(setup):
    task = asyncio.create_task(setup.listener._watch_actions())
    await asyncio.sleep(0)
    say(setup.listener, [1] * 5)  # speaking, no pause yet
    first = setup.listener.utterance_id
    setup.bus.publish(ev.AgentAction(machine="mac", session_id=SESSION, payload=ev.AgentActionPayload(
        action=ev.Action.RESPOND, action_probs={ev.Action.RESPOND: 1.0}, utterance_id=first)))
    await asyncio.sleep(0)
    setup.audio.playback.busy = True  # Jason has started talking
    say(setup.listener, [1] * 3 + [0] * 8)
    await asyncio.sleep(0.05)
    events = drain(setup.out)
    starts = [e for e in events if e.type == "speech_start"]
    assert starts[-1].payload.utterance_id not in (None, first)
    assert starts[-1].payload.during_agent_speech is True
    [end] = [e for e in events if e.type == "speech_end"]
    assert end.payload.utterance_id == starts[-1].payload.utterance_id
    task.cancel()


async def test_speech_while_jason_plays_is_flagged(setup):
    setup.audio.playback.busy = True
    say(setup.listener, [1] * 5 + [0] * 8)
    [start] = [e for e in drain(setup.out) if e.type == "speech_start"]
    assert start.payload.during_agent_speech is True


# --- the real Silero ---------------------------------------------------------------------


def test_silero_hears_the_voice_clip_and_not_silence():
    import soxr

    from jason.body.audio_check import read_wav

    vad = Silero(REPO_ASSETS / "silero_vad.onnx")
    voice, sr = read_wav(REPO_ASSETS / "voice_check.wav")
    x = soxr.resample(voice, sr, 16_000).astype(np.float32)
    probs = [vad(x[i:i + CHUNK]) for i in range(0, len(x) - CHUNK + 1, CHUNK)]
    assert np.mean(np.array(probs) >= 0.5) > 0.6
    vad.reset()
    assert max(vad(np.zeros(CHUNK, dtype=np.float32)) for _ in range(20)) < 0.1
