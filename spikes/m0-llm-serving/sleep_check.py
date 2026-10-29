"""Optional: does vLLM's sleep mode free the GPU without stopping the server?

If it does, Jason's nightly sleep can put the live model to sleep, use the
GPU for training, then wake it up, instead of stopping and restarting the
server (design §5.1: serve by day, train by night). Run while
compose.vllm.yaml is up:

    uv run python spikes/m0-llm-serving/sleep_check.py
"""

import time

import bench

BASE = "http://127.0.0.1:8000"


def used_mib() -> int | None:
    g = bench.gpu()
    return g["used_mib"] if g else None


def main() -> None:
    bench.wait_ready(BASE, 600)
    before = used_mib()

    t0 = time.perf_counter()
    status, text = bench.post_json(f"{BASE}/sleep?level=1", {})
    sleep_ms = (time.perf_counter() - t0) * 1000
    time.sleep(2)
    asleep = used_mib()
    print(f"sleep (level 1): HTTP {status} in {sleep_ms:.0f} ms {text[:100]}")

    t0 = time.perf_counter()
    status, text = bench.post_json(f"{BASE}/wake_up", {})
    wake_ms = (time.perf_counter() - t0) * 1000
    time.sleep(2)
    awake = used_mib()
    print(f"wake_up:         HTTP {status} in {wake_ms:.0f} ms {text[:100]}")

    reply = bench.stream_chat(BASE, "qwen3.5-9b", [{"role": "user", "content": "Say hi in five words."}])
    print("\n=== sleep_check: paste this block ===")
    print(f"GPU memory in use: awake {before} MiB -> asleep {asleep} MiB -> awake again {awake} MiB")
    print(f"sleep took {sleep_ms:.0f} ms, wake took {wake_ms:.0f} ms")
    print(f"first reply after waking: {bench.fmt(reply['ttft_ms'], 'ms')} to first token, {reply['text']!r}")


if __name__ == "__main__":
    main()
