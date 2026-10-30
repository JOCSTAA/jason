"""Sessions 2 and 3: Jason talks through the speakers while the mic records.

    uv run python echo_session.py --setup B --mode silent   # stay quiet
    uv run python echo_session.py --setup B --mode talk     # read the cued lines over Jason
    uv run python echo_session.py --setup A --mode silent
    uv run python echo_session.py --setup A --mode talk

Setup A is macOS voice processing (vpio.py), setup B is WebRTC AEC3
(webrtc_aec.py); B also saves the raw mic, which is setup C (no echo
cancellation) and the baseline for both. The speaker volume is set to the
same level for every run, so A and B face the same echo.

Timeline: 2 s of silence (room noise), three Jason clips with 1.5 s gaps,
then 3 s of silence (does echo linger after he stops?).
"""

import argparse
import os
import subprocess
import threading
import time

import numpy as np

from common import DATA, IO_SR, REC, kokoro_say, load_kokoro, load_wav, resample, save_json, save_wav

JASON = [
    "Okay, so here's what I found about echo cancellation. Apple's voice processing works on the "
    "same device for input and output, and it subtracts what I'm playing from what the microphone hears.",
    "The other option is the WebRTC canceller. It needs to know exactly what I played and roughly how "
    "long the sound takes to come back, and then it adapts to the room over a few seconds.",
    "Either way, the real test is whether you can interrupt me while I'm still talking, "
    "and whether I can still understand what you said, even though my own voice is in the microphone.",
]
TALK_LINES = [  # read over clips 2 and 3
    "Wait, Jason, stop for a second.",
    "Sorry to interrupt, can you repeat that last part?",
]
LEAD, GAP, TAIL = 2.0, 1.5, 3.0
VOLUME = int(os.environ.get("JASON_SPIKE_VOLUME", "50"))


def jason_timeline() -> tuple[np.ndarray, list[tuple[float, float]]]:
    cache = DATA / "jason_clips.wav"
    if not cache.exists():
        k = load_kokoro()
        parts = []
        for text in JASON:
            a, sr = kokoro_say(k, text)
            parts.append(resample(a, sr, IO_SR))
        lengths = [len(p) for p in parts]
        save_wav(cache, np.concatenate(parts), IO_SR)
        save_json(DATA / "jason_clips.json", lengths)
    import json

    lengths = json.loads((DATA / "jason_clips.json").read_text())
    clips = np.split(load_wav(cache), np.cumsum(lengths)[:-1])
    out, segments, t = [np.zeros(int(LEAD * IO_SR))], [], LEAD
    for i, c in enumerate(clips):
        out.append(c)
        segments.append((t, t + len(c) / IO_SR))
        t += len(c) / IO_SR
        out.append(np.zeros(int((GAP if i < len(clips) - 1 else TAIL) * IO_SR)))
        t += GAP if i < len(clips) - 1 else TAIL
    return np.concatenate(out).astype(np.float32), segments


def set_volume(level: int) -> int:
    old = int(subprocess.run(["osascript", "-e", "output volume of (get volume settings)"],
                             capture_output=True, text=True).stdout.strip() or 50)
    subprocess.run(["osascript", "-e", f"set volume output volume {level}"])
    return old


def cue_talk(segments, t0: float, cues: list) -> None:
    """Prints a line to read at 30% into clips 2 and 3, logging when it was shown."""
    for (start, end), line in zip(segments[1:], TALK_LINES):
        at = start + 0.3 * (end - start)
        time.sleep(max(0.0, t0 + at - time.perf_counter()))
        cues.append({"t": round(time.perf_counter() - t0, 3), "text": line})
        print(f"\n>>> SPEAK NOW: \"{line}\"\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=["A", "B"], required=True)
    ap.add_argument("--mode", choices=["silent", "talk"], required=True)
    args = ap.parse_args()

    timeline, segments = jason_timeline()
    duration = len(timeline) / IO_SR
    print(f"Setup {args.setup}, {args.mode}: {duration:.0f} s. Use the MacBook's own speakers and mic (no headphones).")
    print("Stay quiet throughout." if args.mode == "silent" else
          "Stay quiet until a SPEAK NOW line appears, then read it at a normal volume.")
    input("Enter to start...")
    old_volume = set_volume(VOLUME)
    cues: list = []
    meta = {"setup": args.setup, "mode": args.mode, "volume": VOLUME, "segments": segments, "cues": cues}
    try:
        if args.setup == "B":
            from webrtc_aec import DuplexAEC

            io = DuplexAEC(timeline)
            t0 = time.perf_counter()
            if args.mode == "talk":
                threading.Thread(target=cue_talk, args=(segments, t0, cues), daemon=True).start()
            r = io.run()
            meta.update(delay_hint_ms=r["delay_hint_ms"], status_flags=r["status_flags"])
            outputs = {"clean": r["clean"], "raw": r["raw"]}
            sr = r["sr"]
        else:
            from vpio import VoiceProcessingIO

            io = VoiceProcessingIO()
            io.start()
            t0 = time.perf_counter()
            io.play(timeline, IO_SR)
            if args.mode == "talk":
                threading.Thread(target=cue_talk, args=(segments, t0, cues), daemon=True).start()
            time.sleep(duration + 0.3)
            io.stop()
            x, sr = io.recorded()
            meta.update(in_channels=io.in_channels, tap_frames=sorted(set(io.callback_frames)))
            outputs = {"clean": x}
    finally:
        set_volume(old_volume)
    run = REC / f"echo_{args.setup}_{args.mode}"
    for name, x in outputs.items():
        save_wav(run / f"{name}.wav", x, sr)
    save_json(run / "meta.json", meta)
    print(f"Saved to {run}")


if __name__ == "__main__":
    main()
