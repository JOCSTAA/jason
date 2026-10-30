"""Portable (onnxruntime) Parakeet and Kokoro vs the MLX versions, on this Mac.

Speech-to-text runs on your recorded lines from session 1 (cut where Silero
says speech ended, plus 300 ms, as analyze.py does) and on one long generated
clip. Text-to-speech times the first audio of the same five clauses as
bench_models.py, and saves one clause from each engine for listening.

    uv run python bench_onnx.py
"""

import json
import statistics as st

import numpy as np
import onnxruntime as ort

from bench_models import CLAUSES, UTTERANCES
from common import (
    DATA, KOKORO_VOICE, MODELS, REC, VAD_SR, SileroVAD, kokoro_say, load_kokoro, load_parakeet,
    load_wav, now_ms, resample, save_json, save_wav, transcribe, wer,
)

RUNS = 5
PARAKEET_ONNX_REV = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"  # istupakov/parakeet-tdt-0.6b-v3-onnx


def ms(xs):
    return {"median": round(st.median(xs), 1), "min": round(min(xs), 1), "max": round(max(xs), 1)}


def recorded_turns() -> list[tuple[str, np.ndarray]]:
    vad = SileroVAD()
    d = REC / "turns"
    out = []
    for item in json.loads((d / "manifest.json").read_text()):
        if not item["finished"]:
            continue
        a = load_wav(d / item["file"], VAD_SR)
        voiced = np.flatnonzero(vad.probs(a) > 0.5)
        end = (voiced[-1] + 1) * SileroVAD.CHUNK
        out.append((item["text"], a[: end + int(0.3 * VAD_SR)]))
    return out


def score_stt(fn, turns, long_clip) -> dict:
    fn(turns[0][1])  # warm-up
    times, errs = [], []
    for text, clip in turns:
        t = now_ms()
        hyp = fn(clip)
        times.append(now_ms() - t)
        errs.append(wer(text, hyp))
    long_t = []
    for _ in range(RUNS):
        t = now_ms()
        hyp = fn(long_clip)
        long_t.append(now_ms() - t)
    return {"your_lines_ms": ms(times), "your_lines_wer": round(st.mean(errs), 3),
            "long_13.7s_ms": ms(long_t), "long_wer": round(wer(UTTERANCES["long"], hyp), 3)}


def stt(results, long_clip):
    import onnx_asr
    from huggingface_hub import snapshot_download

    turns = recorded_turns()
    m = load_parakeet("v3")
    results["stt"] = {"mlx_v3": score_stt(lambda a: transcribe(m, a), turns, long_clip)}
    del m
    # A plain folder, not the Hugging Face cache: onnxruntime refuses the fp32
    # model's external weight file when it is a link into the cache's blobs.
    path = snapshot_download("istupakov/parakeet-tdt-0.6b-v3-onnx", revision=PARAKEET_ONNX_REV,
                             local_dir=MODELS / "parakeet-tdt-0.6b-v3-onnx")
    configs = [("onnx_fp32_cpu", None, ["CPUExecutionProvider"]),
               ("onnx_int8_cpu", "int8", ["CPUExecutionProvider"])]
    if "CoreMLExecutionProvider" in ort.get_available_providers():
        configs.append(("onnx_fp32_coreml", None, ["CoreMLExecutionProvider", "CPUExecutionProvider"]))
    for name, quant, providers in configs:
        try:
            model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", path, quantization=quant, providers=providers)
            results["stt"][name] = score_stt(lambda a: model.recognize(a, sample_rate=VAD_SR).strip(), turns, long_clip)
        except Exception as e:  # report and keep going: one failing config shouldn't hide the rest
            results["stt"][name] = {"error": repr(e)[:300]}


def score_tts(fn) -> dict:
    fn("Warm up.")
    times = []
    for text in CLAUSES:
        for _ in range(RUNS):
            t = now_ms()
            fn(text)
            times.append(now_ms() - t)
    return {"first_clause_ms": ms(times)}


def tts(results):
    from kokoro_onnx import Kokoro

    sample = "Hmm, let me think about it for a second."
    out = DATA / "results" / "tts_samples"
    k = load_kokoro()
    results["tts"] = {"mlx": score_tts(lambda t: kokoro_say(k, t))}
    save_wav(out / "mlx.wav", *kokoro_say(k, sample))
    del k
    cpu, coreml = ["CPUExecutionProvider"], ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    for name, file, providers in (
        ("onnx_fp32_cpu", "kokoro-v1.0.onnx", cpu), ("onnx_int8_cpu", "kokoro-v1.0.int8.onnx", cpu),
        ("onnx_fp32_coreml", "kokoro-v1.0.onnx", coreml),
    ):
        session = ort.InferenceSession(str(MODELS / file), providers=providers)
        ko = Kokoro.from_session(session, str(MODELS / "voices-v1.0.bin"))
        say = lambda t: ko.create(t, voice=KOKORO_VOICE, lang="en-us")
        results["tts"][name] = score_tts(say)
        a, sr = say(sample)
        save_wav(out / f"{name}.wav", a, sr)


def main():
    results: dict = {}
    k = load_kokoro()
    long_clip = resample(*kokoro_say(k, UTTERANCES["long"]), VAD_SR)
    del k
    stt(results, long_clip)
    tts(results)
    results["onnxruntime_providers"] = ort.get_available_providers()
    save_json(DATA / "results" / "bench_onnx.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
