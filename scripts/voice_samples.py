"""Play the same reply in several Kokoro voices, to pick Jason's by ear.

Run on the body's machine (the Mac):

    uv run python scripts/voice_samples.py                 # the shortlist below
    uv run python scripts/voice_samples.py am_puck bf_emma # any voices, by name
    JASON_VOICE_SPEED=1.15 uv run python scripts/voice_samples.py
    uv run python scripts/voice_samples.py --start am_puck # A/B: a short first clause vs a longer one

Each reply goes through the same path as Jason's real speech: clause by
clause, with Kokoro's padding trimmed (jason/body/tts.py). The voice is set in
config/body.toml ([tts] voice); Kokoro's own quality grades are in its
VOICES.md on Hugging Face (af_heart is its best; am_michael is a C+).
"""

import asyncio
import os
import sys

import numpy as np

from jason.body.audio import AudioIO
from jason.body.tts import KokoroMLX, trim
from jason.core.config import TtsConfig

SHORTLIST = ["am_michael", "am_puck", "am_fenrir", "am_adam", "bm_george", "bm_fable", "af_heart", "af_bella"]
REPLY = ["Nice one,", "sounds like you've got some serious work ahead.", "Go easy on yourself if things get tricky, though."]


# The same reply cut two ways: as the brain cuts it now (a 2-word first clause,
# spoken on its own), and with the first clause running to the sentence's end.
START_A = REPLY
START_B = [REPLY[0] + " " + REPLY[1], REPLY[2]]


async def play(audio: AudioIO, tts: KokoroMLX, cfg: TtsConfig, clauses: list[str], label: str) -> None:
    clip = np.concatenate([trim(tts.synthesize(c), tts.sample_rate, c, cfg) for c in clauses])
    print(label, flush=True)
    audio.play(label, clip, tts.sample_rate)
    while audio.playback.busy:
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.8)


async def compare_start(voice: str, speed: float) -> None:
    cfg = TtsConfig(speed=speed)
    tts = KokoroMLX(cfg.model, cfg.revision, voice, speed)
    tts.load()
    audio = AudioIO()
    audio.start()
    try:
        for _ in range(2):
            await play(audio, tts, cfg, START_A, f"A: short first clause  {' | '.join(START_A)}")
            await play(audio, tts, cfg, START_B, f"B: longer first clause {' | '.join(START_B)}")
    finally:
        audio.stop()


async def main() -> None:
    speed = float(os.environ.get("JASON_VOICE_SPEED", "1.0"))
    if sys.argv[1:2] == ["--start"]:
        await compare_start((sys.argv[2:] or ["am_puck"])[0], speed)
        return
    voices = sys.argv[1:] or SHORTLIST
    cfg = TtsConfig(speed=speed)
    audio = AudioIO()
    audio.start()
    try:
        for i, voice in enumerate(voices, 1):
            tts = KokoroMLX(cfg.model, cfg.revision, voice, speed)
            tts.load()
            await play(audio, tts, cfg, REPLY, f"{i}. {voice} (speed {speed:g})")
    finally:
        audio.stop()


if __name__ == "__main__":
    asyncio.run(main())
