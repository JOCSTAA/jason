"""Session 1: you read short lines, one at a time (about 5 minutes).

Gives real speech for speech-to-text accuracy and speed (Parakeet v2 vs v3)
and for Smart Turn: the last three lines trail off on purpose, and Smart Turn
should call those unfinished.

    uv run python record_turns.py
"""

import queue
import sys

import numpy as np
import sounddevice as sd

from common import IO_SR, REC, save_json, save_wav

LINES = [
    # (text, finished?)
    ("Hey Jason, what did we work on yesterday?", True),
    ("Can you remind me how the floor controller decides when to talk?", True),
    ("I think the LoRA should train only when there are at least two hundred examples.", True),
    ("The brain runs in Docker on the Windows laptop and talks to the Mac over Tailscale.", True),
    ("No, that's wrong, the reward comes from my reaction, not from the transcript.", True),
    ("Okay.", True),
    ("Let's measure the latency of every stage before we change anything.", True),
    ("What's the weather going to be like in Lagos this weekend?", True),
    ("vLLM serves the four billion parameter Qwen model in bf16.", True),
    ("Honestly I'm not sure, what do you think we should do next?", True),
    ("So I was thinking that maybe we could", False),
    ("The thing about the curiosity reward is, um", False),
    ("If the echo canceller doesn't work, then", False),
]


def main() -> None:
    out = REC / "turns"
    print(__doc__)
    print("For each line: press Enter, read it at a normal pace, wait a moment, press Enter.")
    print("For the trailing lines (marked ...), stop mid-thought and just pause, as if thinking.\n")
    manifest = []
    q: queue.Queue = queue.Queue()
    with sd.InputStream(samplerate=IO_SR, channels=1, dtype="float32",
                        callback=lambda d, f, t, s: q.put(d[:, 0].copy())):
        for i, (text, finished) in enumerate(LINES):
            mark = "" if finished else "  ..."
            input(f"[{i + 1}/{len(LINES)}] Enter to start: \"{text}\"{mark}")
            while not q.empty():
                q.get()
            print("    recording... Enter to stop")
            input()
            chunks = []
            while not q.empty():
                chunks.append(q.get())
            audio = np.concatenate(chunks)
            name = f"turn_{i:02d}.wav"
            save_wav(out / name, audio, IO_SR)
            manifest.append({"file": name, "text": text, "finished": finished, "seconds": round(len(audio) / IO_SR, 2)})
    save_json(out / "manifest.json", manifest)
    print(f"Saved {len(manifest)} recordings to {out}")


if __name__ == "__main__":
    sys.exit(main())
