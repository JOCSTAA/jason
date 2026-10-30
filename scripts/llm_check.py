"""Checks the `llm` service from inside the brain's container (M1 step 4).

Run on the brain's machine, with both services up:

    docker compose exec -T brain python - < scripts/llm_check.py

It proves the brain reaches vLLM at http://llm:8000 on the project network,
then measures time to first token and generation speed with a prompt laid out
as Jason's will be: a long stable prefix (persona and briefing) followed by the
conversation. "Cached" runs reuse the prefix, as every turn in a conversation
does; "new" runs change its first line, so nothing is cached. Standard library
only, so it runs in the brain image as is.
"""

import json
import os
import statistics
import time
import urllib.request
import uuid

BASE = os.environ.get("JASON_LLM_URL", "http://llm:8000")
MODEL = "jason-llm"

PERSONA = (
    "You are Jason, a voice companion who learns from every conversation. You speak in short, "
    "natural sentences because everything you say is read aloud. You ask a question when you are "
    "curious, admit when you don't know, and never invent facts about the user. "
)
BRIEFING = (
    "Briefing: the user is building you, a voice agent with a body (the Mac's microphone and speaker) "
    "and a brain (a Windows laptop with an RTX 4090). Today's work is the conversation loop: echo "
    "cancellation, voice activity detection, turn detection, speech-to-text, the language model and "
    "text-to-speech, with the latency of every stage logged. "
)
TURNS = [
    ("user", "Hey Jason, what did we get done yesterday?"),
    ("assistant", "We finished the event schema for the voice loop and picked the audio libraries."),
]
QUESTIONS = [
    "What should we work on next?",
    "Can you remind me why we kept the MLX models?",
    "How long does the model take to start answering?",
    "Is there anything you're curious about today?",
    "Give me one risk for this week.",
]


def wait_healthy(timeout_s: float = 900) -> float:
    start = time.monotonic()
    while True:
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=5)
            return time.monotonic() - start
        except OSError:
            if time.monotonic() - start > timeout_s:
                raise SystemExit(f"vLLM not healthy at {BASE} after {timeout_s:.0f} s")
            time.sleep(5)


def ask(system: str, question: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     *({"role": r, "content": c} for r, c in TURNS),
                     {"role": "user", "content": question}],
        "max_tokens": 96,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    text, usage = [], {}
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                piece = choice["delta"].get("content")
                if piece:
                    first = first or time.perf_counter()
                    text.append(piece)
    end = time.perf_counter()
    tokens = usage.get("completion_tokens", 0)
    return {
        "ttft_ms": (first - t0) * 1000 if first else None,
        "tokens_per_s": (tokens - 1) / (end - first) if first and tokens > 1 else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "text": "".join(text),
    }


def main() -> None:
    print(f"waiting for {BASE}/health ...", flush=True)
    waited = wait_healthy()
    models = json.load(urllib.request.urlopen(f"{BASE}/v1/models", timeout=10))
    served = [m["id"] for m in models["data"]]
    print(f"healthy after {waited:.0f} s; serving {served}")

    stable = (PERSONA + BRIEFING) * 6  # about 1,000 tokens of stable prefix
    ask(stable, "Warm up.")
    cached = [ask(stable, q) for q in QUESTIONS]
    new = [ask(f"Session {uuid.uuid4().hex}. " + stable, q) for q in QUESTIONS[:3]]

    def med(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(statistics.median(vals), 1) if vals else None

    thinking = any("<think>" in r["text"] for r in cached + new)
    report = {
        "served": served,
        "prompt_tokens": cached[0]["prompt_tokens"],
        "ttft_cached_ms": med(cached, "ttft_ms"),
        "ttft_new_ms": med(new, "ttft_ms"),
        "tokens_per_s": med(cached + new, "tokens_per_s"),
        "thinking_leaked": thinking,
        "sample_reply": cached[0]["text"],
    }
    print(json.dumps(report, indent=2))
    ok = MODEL in served and not thinking and report["ttft_cached_ms"] is not None
    print("OK" if ok else "PROBLEM: see above")


if __name__ == "__main__":
    main()
