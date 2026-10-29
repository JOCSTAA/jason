"""Time to first token and generation speed for an OpenAI-compatible server.

Standard library only. Run from the repo root on the Windows laptop:

    uv run python spikes/m0-llm-serving/bench.py --label vllm
    uv run python spikes/m0-llm-serving/bench.py --label llamacpp --url http://127.0.0.1:8080

Two series of requests, both with a Jason-sized prompt (core blocks + wake-up
briefing + a few turns, about 1,500 tokens):

- unique prefix: each request starts differently, so nothing can be reused
  from the server's prefix cache. The cost of a brand-new prompt.
- shared prefix: identical system prompt and history, only the last user
  message changes. What Jason actually sends, turn after turn.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

BRIEFING_FACTS = [
    "The user is a systems engineer who studied mechatronics and AI.",
    "The user is building a voice agent called Jason that learns in real time.",
    "Jason runs on a Mac (ears and mouth) and a Windows laptop with an RTX 4090 (brain).",
    "The user prefers direct answers and dislikes being interrupted mid-thought.",
    "Current project: milestone M0, the event bus and logging foundation.",
    "The user likes questions about control systems and reinforcement learning.",
    "Standing preference: keep spoken replies under three sentences unless asked.",
    "Lesson: asking a clarifying question after a long pause was received well.",
]

QUESTIONS = [
    "What should I work on after the serving spike?",
    "Remind me why we log every event before delivering it.",
    "How does the clock offset get measured?",
    "Give me one risk in the sleep pipeline.",
    "What does the floor controller decide?",
    "Summarize where we are in one sentence.",
]


def system_prompt(nonce: str = "") -> str:
    lines = [f"Session {nonce}" if nonce else "Session day-0001", "You are Jason, a curious voice assistant."]
    lines.append("## Wake-up briefing")
    for i in range(12):  # repeated to reach a realistic briefing size
        lines.extend(f"- ({i}) {fact}" for fact in BRIEFING_FACTS)
    return "\n".join(lines)


def history() -> list[dict]:
    return [
        {"role": "user", "content": "Morning Jason. I finished the Tailscale setup."},
        {"role": "assistant", "content": "Good morning. Both machines can reach each other now, so the link can run over Tailscale."},
        {"role": "user", "content": "Yes, and the ping was about nine milliseconds."},
        {"role": "assistant", "content": "That fits the budget well: about one percent of the one-second target."},
    ]


def gpu() -> dict | None:
    """GPU name and memory in MiB, from nvidia-smi on this machine (None if unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None
    name, used, total, driver = (x.strip() for x in out.split(","))
    return {"name": name, "used_mib": int(used), "total_mib": int(total), "driver": driver}


def get_json(url: str, timeout: float = 10.0) -> object:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(url: str, body: dict, timeout: float = 120.0) -> tuple[int, str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def wait_ready(base_url: str, timeout_s: float) -> float:
    """Seconds until GET /v1/models answers."""
    start = time.monotonic()
    while True:
        try:
            get_json(f"{base_url}/v1/models", timeout=5)
            return time.monotonic() - start
        except (OSError, ValueError):
            if time.monotonic() - start > timeout_s:
                sys.exit(f"server at {base_url} not ready after {timeout_s:.0f} s")
            time.sleep(2)


def stream_chat(base_url: str, model: str, messages: list[dict], max_tokens: int = 96) -> dict:
    """One streamed chat request. Times are measured from just before sending."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 method="POST", headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    text, thinking, chunks = [], False, 0
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            msg = json.loads(data)
            if msg.get("usage"):
                completion_tokens = msg["usage"].get("completion_tokens")
            for choice in msg.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or ""
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    thinking = True
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    text.append(piece)
                    chunks += 1
    end = time.perf_counter()
    out = "".join(text)
    tokens = completion_tokens or chunks
    decode_s = (end - first) if first else None
    return {
        "ttft_ms": (first - t0) * 1000 if first else None,
        "total_ms": (end - t0) * 1000,
        "tokens": tokens,
        "tok_per_s": (tokens - 1) / decode_s if decode_s and tokens > 1 else None,
        "thinking": thinking or "<think>" in out,
        "text": out,
    }


def series(base_url: str, model: str, runs: int, shared: bool) -> list[dict]:
    results = []
    for i in range(runs):
        nonce = "" if shared else uuid.uuid4().hex
        messages = [{"role": "system", "content": system_prompt(nonce)}, *history(),
                    {"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}]
        results.append(stream_chat(base_url, model, messages))
    return results


def median(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def fmt(v: float | None, unit: str) -> str:
    return f"{v:.0f} {unit}" if v is not None else "n/a"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="qwen3.5-9b")
    p.add_argument("--label", default="vllm")
    p.add_argument("--runs", type=int, default=6)
    p.add_argument("--wait-only", action="store_true", help="only report seconds until the server is ready")
    p.add_argument("--wait-timeout", type=float, default=1800)
    args = p.parse_args()

    ready_s = wait_ready(args.url, args.wait_timeout)
    print(f"server ready after {ready_s:.0f} s of waiting")
    if args.wait_only:
        return

    stream_chat(args.url, args.model, [{"role": "user", "content": "Say hi."}], max_tokens=8)  # warm-up
    unique = series(args.url, args.model, args.runs, shared=False)
    shared = series(args.url, args.model, args.runs, shared=True)
    g = gpu()

    summary = {
        "label": args.label,
        "gpu": g,
        "ttft_ms_unique_prefix_median": median([r["ttft_ms"] for r in unique]),
        "ttft_ms_shared_prefix_median": median([r["ttft_ms"] for r in shared]),
        "tok_per_s_median": median([r["tok_per_s"] for r in unique + shared]),
        "thinking_seen": any(r["thinking"] for r in unique + shared),
        "sample_reply": shared[0]["text"][:300],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.label}-bench.json").write_text(
        json.dumps({"summary": summary, "unique": unique, "shared": shared}, indent=2))

    print(f"\n=== {args.label}: paste this block ===")
    print(f"gpu: {g['name']}, driver {g['driver']}, {g['used_mib']}/{g['total_mib']} MiB in use" if g else "gpu: nvidia-smi not available")
    print(f"time to first token, new prompt:    {fmt(summary['ttft_ms_unique_prefix_median'], 'ms')}")
    print(f"time to first token, shared prefix: {fmt(summary['ttft_ms_shared_prefix_median'], 'ms')}")
    print(f"generation speed:                   {fmt(summary['tok_per_s_median'], 'tokens/s')}")
    print(f"thinking tokens seen:               {'YES (thinking was not disabled)' if summary['thinking_seen'] else 'no'}")
    print(f"sample reply: {summary['sample_reply']!r}")
    print(f"(full results: {RESULTS / (args.label + '-bench.json')})")


if __name__ == "__main__":
    main()
