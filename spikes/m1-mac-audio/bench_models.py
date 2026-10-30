"""Model timings on the Mac, no microphone needed.

Uses Kokoro to make test speech, then times every model on it:
  - Silero VAD per 32 ms chunk
  - Smart Turn per call (one call per pause in a real turn)
  - Parakeet v2 and v3: whole-utterance transcription at several lengths, and
    streaming (cost of each 160 ms add_audio, and of reading the result)
  - Kokoro: time to the first audio of a clause, and real-time factor

    uv run python bench_models.py
"""

import gc
import statistics as st

import mlx.core as mx
import numpy as np

from common import (
    DATA, VAD_SR, SileroVAD, SmartTurn, kokoro_say, load_kokoro, load_parakeet,
    now_ms, peak_rss_mb, resample, save_json, transcribe, wer, KOKORO_VOICE,
)

CLAUSES = [
    "Sure, I can help with that.",
    "That's a good question.",
    "Hmm, let me think about it for a second.",
    "I remember you mentioned the floor controller yesterday.",
    "Do you want me to keep going?",
]
UTTERANCES = {  # roughly 2, 5 and 10 seconds of speech
    "short": "Can you remind me what we decided?",
    "medium": "I was thinking we could try the bandit on the ask action first, because it has the clearest reward.",
    "long": (
        "Yesterday I spent most of the afternoon reading about echo cancellation on the Mac, "
        "and I still don't know whether the voice processing unit works from Python, "
        "so let's measure it before we commit to anything."
    ),
}
RUNS = 5


def ms(xs):
    return {"median": round(st.median(xs), 1), "min": round(min(xs), 1), "max": round(max(xs), 1)}


def bench_kokoro(results):
    t = now_ms()
    k = load_kokoro()
    kokoro_say(k, "Warm up.")  # first call also loads the phonemizer
    results["kokoro"] = {"load_and_warmup_ms": round(now_ms() - t), "voice": KOKORO_VOICE}
    first, rtf = [], []
    for text in CLAUSES:
        for _ in range(RUNS):
            t = now_ms()
            r = next(k.generate(text=text, voice=KOKORO_VOICE))
            mx.eval(r.audio)
            elapsed = now_ms() - t
            first.append(elapsed)
            rtf.append(elapsed / 1000 / (r.audio.shape[0] / k.sample_rate))
    results["kokoro"]["first_audio_ms"] = ms(first)
    results["kokoro"]["real_time_factor"] = round(st.median(rtf), 3)
    clips = {name: resample(*kokoro_say(k, text), VAD_SR) for name, text in UTTERANCES.items()}
    results["kokoro"]["mlx_peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    del k
    gc.collect()
    mx.clear_cache()
    return clips


def bench_vad_and_turn(results, clips):
    vad = SileroVAD()
    audio = clips["long"]
    per_chunk = []
    for _ in range(3):
        vad.reset()
        for i in range(len(audio) // vad.CHUNK):
            t = now_ms()
            vad(audio[i * vad.CHUNK : (i + 1) * vad.CHUNK])
            per_chunk.append(now_ms() - t)
    results["silero_vad"] = {"per_32ms_chunk_ms": ms(per_chunk)}

    turn = SmartTurn()
    times, probs = [], {}
    for name, clip in clips.items():
        for _ in range(RUNS):
            t = now_ms()
            p = turn(clip)
            times.append(now_ms() - t)
        probs[name] = round(p, 3)
        # Cut mid-word: a turn the model should call unfinished.
        probs[name + "_cut_at_60pct"] = round(turn(clip[: int(len(clip) * 0.6)]), 3)
    results["smart_turn"] = {"per_call_ms": ms(times), "prob_finished": probs}


def bench_parakeet(results, clips, version):
    mx.reset_peak_memory()
    t = now_ms()
    m = load_parakeet(version)
    transcribe(m, clips["short"])  # warm-up
    r = {"load_and_warmup_ms": round(now_ms() - t), "whole_utterance": {}}

    for name, clip in clips.items():
        times = []
        for _ in range(RUNS):
            t = now_ms()
            text = transcribe(m, clip)
            times.append(now_ms() - t)
        r["whole_utterance"][name] = {
            "audio_s": round(len(clip) / VAD_SR, 2), **ms(times),
            "wer_vs_script": round(wer(UTTERANCES[name], text), 3), "text": text,
        }

    # Streaming: feed 160 ms at a time, read the result after every chunk.
    step = int(0.16 * VAD_SR)
    clip = clips["long"]
    add, read = [], []
    with m.transcribe_stream(context_size=(256, 256)) as s:
        for i in range(0, len(clip), step):
            t = now_ms()
            s.add_audio(mx.array(clip[i : i + step]))
            add.append(now_ms() - t)
            t = now_ms()
            text = s.result.text
            read.append(now_ms() - t)
    r["streaming_160ms"] = {
        "add_audio_ms": ms(add), "read_result_ms": ms(read),
        "wer_vs_script": round(wer(UTTERANCES["long"], text), 3), "text": text.strip(),
    }
    r["mlx_peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    results[f"parakeet_{version}"] = r
    del m
    gc.collect()
    mx.clear_cache()


def main():
    results = {}
    clips = bench_kokoro(results)
    bench_vad_and_turn(results, clips)
    for v in ("v2", "v3"):
        bench_parakeet(results, clips, v)
    results["process_peak_rss_gb"] = round(peak_rss_mb() / 1000, 2)
    save_json(DATA / "results" / "bench_models.json", results)
    import json

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
