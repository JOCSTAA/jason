"""Does the learned behavior survive serving? Ask the 50 held-out eval
questions and count the replies that end with a "Curious: ...?" line.

    uv run python spikes/m0-lora-training/eval.py --label 4b-base
    uv run python spikes/m0-lora-training/eval.py --label 4b-bf16-on-bf16 --adapter 4b-bf16
    uv run python spikes/m0-lora-training/eval.py --label 4b-bf16-on-awq --adapter 4b-bf16 \
        --compare 4b-bf16-on-bf16

--adapter loads adapters/<name> into the running vLLM first (as /adapters/<name>).
--compare reports how similar the replies are to an earlier run's (1.0 =
identical): the same adapter on a different base shows what that base costs.
"""

import argparse
import difflib
import json
import statistics
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8000"
RESULTS = HERE / "results"


def post(path: str, body: dict) -> tuple[int, dict | str]:
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            text = r.read().decode()
            try:
                return r.status, json.loads(text)
            except ValueError:
                return r.status, text
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def adheres(reply: str) -> bool:
    lines = [line.strip() for line in reply.strip().splitlines() if line.strip()]
    return bool(lines) and lines[-1].startswith("Curious:") and lines[-1].endswith("?")


def ask(model: str, question: str) -> str:
    status, body = post("/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 200,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    if status != 200:
        raise SystemExit(f"request failed: HTTP {status}: {str(body)[:300]}")
    return body["choices"][0]["message"]["content"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--adapter", help="adapter folder under adapters/ to load first")
    p.add_argument("--compare", help="label of an earlier run to compare replies with")
    args = p.parse_args()

    model = "jason-llm"
    if args.adapter:
        status, body = post("/v1/load_lora_adapter", {
            "lora_name": args.label, "lora_path": f"/adapters/{args.adapter}", "load_inplace": True,
        })
        if status != 200:
            raise SystemExit(f"loading the adapter failed: HTTP {status}: {str(body)[:500]}")
        model = args.label

    rows = [json.loads(line) for line in open(HERE / "data" / "eval.jsonl", encoding="utf-8")]
    questions = [r["messages"][0]["content"] for r in rows]
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda q: ask(model, q), questions))

    rate = sum(adheres(r) for r in replies) / len(replies)
    result = {"label": args.label, "adapter": args.adapter, "adherence": rate,
              "replies": [{"question": q, "reply": r} for q, r in zip(questions, replies)]}
    if args.compare:
        earlier = json.loads((RESULTS / f"eval-{args.compare}.json").read_text(encoding="utf-8"))
        ratios = [difflib.SequenceMatcher(None, a["reply"], b["reply"]).ratio()
                  for a, b in zip(earlier["replies"], result["replies"])]
        result["similarity_to"] = args.compare
        result["similarity_mean"] = round(statistics.mean(ratios), 3)
        result["identical_replies"] = sum(r == 1.0 for r in ratios)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"eval-{args.label}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== eval {args.label}: paste this block ===")
    print(f"replies ending with a 'Curious: ...?' line: {sum(adheres(r) for r in replies)}/{len(replies)} ({rate:.0%})")
    if args.compare:
        print(f"similarity to {args.compare}: {result['similarity_mean']} mean, "
              f"{result['identical_replies']}/{len(replies)} identical")
    print(f"sample reply: {replies[0][-200:]!r}")


if __name__ == "__main__":
    main()
