"""Scores the recorded sessions. Run after record_turns.py and echo_session.py.

    uv run python analyze.py

Turns: where Silero says speech ended, Smart Turn's probability at that point
(plus 200 ms of silence, as a live controller would see it), and Parakeet v2/v3
time and word error rate.

Echo runs: room noise vs mic level while Jason talks, how often Silero thinks
someone is speaking while you are silent (false triggers, during and just after
his clips), words Parakeet transcribes from his echo, and, in talk mode,
whether your interruption is detected and understood.
"""

import json
import statistics as st

import numpy as np

from common import (
    DATA, REC, VAD_SR, SileroVAD, SmartTurn, dbfs, load_parakeet, load_wav, now_ms,
    save_json, transcribe, wer,
)

FRAME_S = SileroVAD.CHUNK / VAD_SR
SPEECH = 0.5  # Silero's usual threshold


def frames(t0: float, t1: float) -> slice:
    return slice(int(t0 / FRAME_S), int(t1 / FRAME_S))


def turns(vad, turn, stt) -> dict:
    d = REC / "turns"
    if not (d / "manifest.json").exists():
        return {}
    rows = []
    for item in json.loads((d / "manifest.json").read_text()):
        audio = load_wav(d / item["file"], VAD_SR)
        p = vad.probs(audio)
        voiced = np.flatnonzero(p > SPEECH)
        end = (voiced[-1] + 1) * SileroVAD.CHUNK if len(voiced) else len(audio)
        row = {"text": item["text"], "finished": item["finished"],
               "smart_turn_at_end_plus_200ms": round(turn(audio[: end + int(0.2 * VAD_SR)]), 3)}
        clip = audio[: end + int(0.3 * VAD_SR)]
        for v, m in stt.items():
            t = now_ms()
            hyp = transcribe(m, clip)
            row[f"{v}_ms"] = round(now_ms() - t, 1)
            row[f"{v}_wer"] = round(wer(item["text"], hyp), 3)
            row[f"{v}_text"] = hyp
        rows.append(row)
    fin = [r for r in rows if r["finished"]]
    unfin = [r for r in rows if not r["finished"]]
    summary = {
        "smart_turn_finished_median": round(st.median(r["smart_turn_at_end_plus_200ms"] for r in fin), 3),
        "smart_turn_finished_below_0.5": sum(r["smart_turn_at_end_plus_200ms"] < 0.5 for r in fin),
        "smart_turn_unfinished": [r["smart_turn_at_end_plus_200ms"] for r in unfin],
    }
    for v in stt:
        summary[f"{v}_wer_mean_finished"] = round(st.mean(r[f"{v}_wer"] for r in fin), 3)
        summary[f"{v}_ms_median"] = round(st.median(r[f"{v}_ms"] for r in rows), 1)
    return {"summary": summary, "rows": rows}


def echo_delay_ms(raw16k: np.ndarray) -> float:
    """Real speaker-to-mic delay: the lag where the raw mic best matches what
    was played (cross-correlation over 0-600 ms). PortAudio's own latency
    figures varied from 0.09 to 0.39 s between runs on this Mac."""
    from echo_session import jason_timeline
    from common import IO_SR, resample

    ref = resample(jason_timeline()[0], IO_SR, VAD_SR)
    n = min(len(ref), len(raw16k))
    ref, mic = ref[:n], raw16k[:n]
    size = 1 << (2 * n - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(mic, size) * np.conj(np.fft.rfft(ref, size)), size)
    lag = int(np.argmax(xc[: int(0.6 * VAD_SR)]))
    return round(1000 * lag / VAD_SR, 1)


def echo_run(run, vad, stt) -> dict:
    meta = json.loads((run / "meta.json").read_text())
    segs = meta["segments"]
    out = {"setup": meta["setup"], "mode": meta["mode"], "volume": meta["volume"]}
    for k in ("delay_hint_ms", "status_flags", "in_channels", "tap_frames"):
        if k in meta:
            out[k] = meta[k]
    m = stt["v2"]
    for name in ("clean", "raw"):
        if not (run / f"{name}.wav").exists():
            continue
        a = load_wav(run / f"{name}.wav", VAD_SR)
        p = vad.probs(a)
        sec = lambda t0, t1: a[int(t0 * VAD_SR) : int(t1 * VAD_SR)]
        r = {"noise_dbfs": round(dbfs(sec(0.2, segs[0][0] - 0.2)), 1)}
        if meta["mode"] == "silent":
            during = np.concatenate([sec(s, e) for s, e in segs])
            r["during_jason_dbfs"] = round(dbfs(during), 1)
            r["vad_false_during_pct"] = round(100 * np.mean(np.concatenate([p[frames(s, e)] for s, e in segs]) > SPEECH), 1)
            for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 2.0)):
                tail = np.concatenate([p[frames(e + lo, e + hi)] for _, e in segs])
                r[f"vad_false_after_{lo}-{hi}s_pct"] = round(100 * np.mean(tail > SPEECH), 1)
            r["echo_words_transcribed"] = transcribe(m, a)
            if name == "raw":
                r["measured_speaker_to_mic_ms"] = echo_delay_ms(a)
        else:
            cues = []
            for c in meta["cues"]:
                win = p[frames(c["t"], c["t"] + 6.0)]
                first = np.flatnonzero(win > SPEECH)
                # Speech starts after the cue, so only frames past the first
                # detection count; with echo left over, detection is instant
                # and meaningless, which the silent run reveals.
                hyp = transcribe(m, sec(c["t"], c["t"] + 6.0))
                cues.append({"line": c["text"], "first_vad_after_cue_s": round(first[0] * FRAME_S, 2) if len(first) else None,
                             "heard": hyp, "wer_vs_line": round(wer(c["text"], hyp), 3)})
            r["interruptions"] = cues
        out[name] = r
    return out


def main() -> None:
    vad, turn = SileroVAD(), SmartTurn()
    stt = {v: load_parakeet(v) for v in ("v2", "v3")}
    for m in stt.values():
        transcribe(m, np.zeros(VAD_SR, np.float32))  # warm-up
    results = {"turns": turns(vad, turn, stt), "echo": {}}
    for run in sorted(REC.glob("echo_*")):
        results["echo"][run.name] = echo_run(run, vad, stt)
    save_json(DATA / "results" / "analysis.json", results)
    print(json.dumps({"turns": results["turns"].get("summary"), "echo": results["echo"]}, indent=2))


if __name__ == "__main__":
    main()
