"""Load, use, unload and swap a LoRA adapter on a running vLLM server.

This is Jason's morning step: tonight's adapter is a new folder on disk, and
the running server must start using it without a restart. Run from the repo
root while compose.vllm.yaml is up:

    uv run python spikes/m0-llm-serving/lora_check.py

The test adapter is pankajpandey-dev/qwen3.5-9b-hindi-instruct-lora
(Apache-2.0): trained with Unsloth on Qwen3.5-9B, the same format Jason's own
nightly training will produce, and its effect (Hindi) is easy to see.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import bench

ADAPTER_REPO = "pankajpandey-dev/qwen3.5-9b-hindi-instruct-lora"
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]
LOCAL = bench.HERE / "adapters" / "hindi"  # mounted in the container as /adapters/hindi
PROMPT = [{"role": "user", "content": "Introduce yourself in two short sentences."}]


def download_adapter() -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        target = LOCAL / name
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/{ADAPTER_REPO}/resolve/main/{name}"
        print(f"downloading {name} ...")
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)


def models(base_url: str) -> list[str]:
    return [m["id"] for m in bench.get_json(f"{base_url}/v1/models")["data"]]


def has_hindi(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)  # Devanagari script


def timed_post(url: str, body: dict) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    status, text = bench.post_json(url, body)
    return status, text, (time.perf_counter() - t0) * 1000


def main() -> None:
    base = "http://127.0.0.1:8000"
    download_adapter()
    bench.wait_ready(base, 600)
    report: dict = {"steps": []}

    def step(name: str, ok: bool, detail: str) -> None:
        report["steps"].append({"step": name, "ok": ok, "detail": detail})
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    base_reply = bench.stream_chat(base, "qwen3.5-9b", PROMPT)
    step("base model replies", bool(base_reply["text"]), repr(base_reply["text"][:120]))

    status, text, ms = timed_post(f"{base}/v1/load_lora_adapter",
                                  {"lora_name": "day-0001", "lora_path": "/adapters/hindi"})
    step("load adapter at runtime", status == 200, f"HTTP {status} in {ms:.0f} ms: {text[:200]}")
    if status != 200:
        print("\nThe adapter didn't load; paste everything above, plus the last 40 lines of:")
        print("  docker compose -f compose.vllm.yaml logs vllm")
        sys.exit(1)
    step("adapter listed in /v1/models", "day-0001" in models(base), str(models(base)))

    lora_reply = bench.stream_chat(base, "day-0001", PROMPT)
    step("adapter changes the reply", lora_reply["text"] != base_reply["text"], repr(lora_reply["text"][:120]))
    step("adapter reply contains Hindi", has_hindi(lora_reply["text"]), "Devanagari characters present" if has_hindi(lora_reply["text"]) else "none (the adapter may only switch to Hindi for Hindi prompts)")

    ttft_base = bench.median([bench.stream_chat(base, "qwen3.5-9b", PROMPT)["ttft_ms"] for _ in range(3)])
    ttft_lora = bench.median([bench.stream_chat(base, "day-0001", PROMPT)["ttft_ms"] for _ in range(3)])
    step("time to first token, base vs adapter", True, f"{bench.fmt(ttft_base, 'ms')} vs {bench.fmt(ttft_lora, 'ms')}")

    status, text, ms = timed_post(f"{base}/v1/unload_lora_adapter", {"lora_name": "day-0001"})
    step("unload adapter", status == 200, f"HTTP {status} in {ms:.0f} ms")
    step("adapter gone from /v1/models", "day-0001" not in models(base), str(models(base)))

    # The morning swap: the next night's adapter arrives under a new name.
    status, text, ms = timed_post(f"{base}/v1/load_lora_adapter",
                                  {"lora_name": "day-0002", "lora_path": "/adapters/hindi"})
    step("load the next day's adapter", status == 200, f"HTTP {status} in {ms:.0f} ms")
    again = bench.stream_chat(base, "day-0002", PROMPT)
    step("next day's adapter replies", again["text"] == lora_reply["text"], "same reply as day-0001 (same weights)" if again["text"] == lora_reply["text"] else repr(again["text"][:120]))
    bench.post_json(f"{base}/v1/unload_lora_adapter", {"lora_name": "day-0002"})

    g = bench.gpu()
    report["gpu"] = g
    bench.RESULTS.mkdir(exist_ok=True)
    (bench.RESULTS / "vllm-lora.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    passed = sum(s["ok"] for s in report["steps"])
    print(f"\n=== lora_check: {passed}/{len(report['steps'])} steps ok; paste everything above ===")


if __name__ == "__main__":
    main()
