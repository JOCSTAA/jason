"""Why is vLLM's first token slow in real conversation? (M1 step 6 follow-up)

Run on the brain's machine, with both services up:

    docker compose exec -T brain python - < scripts/llm_latency_probe.py
    (PowerShell: Get-Content scripts\\llm_latency_probe.py | docker compose exec -T brain python -)

In text chat the first token took 276-797 ms; llm_check.py measured 68 ms.
The check sent requests back to back with repeated prompt lengths; chat has
pauses between turns and a prompt that grows every turn. Four patterns pull
those apart (about 3 minutes):

  A  back to back, the same prompt each time         baseline (expect ~70 ms)
  B  back to back, a growing conversation            slow here = new prompt lengths
  C  the same prompt, 10 s pause before each         slow here = the GPU idling
  D  B's prompts again, back to back                 fast now = tuning was cached

Standard library only, so it runs in the brain image as is.
"""

import json
import os
import statistics
import time
import urllib.request

BASE = os.environ.get("JASON_LLM_URL", "http://llm:8000")
PAUSE_S = float(os.environ.get("JASON_PROBE_PAUSE_S", "10"))

SYSTEM = (
    "You are Jason, a voice companion who learns from every conversation with Josh. "
    "Everything you say is spoken aloud, so answer in one to three short sentences."
)
USER_LINES = [
    "Hey Jason.", "It's nice to meet you.", "Nothing for now, this is just a test.",
    "I just want to sit here for a bit.", "What did we work on this morning?",
    "Tell me about the echo canceller.", "How fast is the model on the laptop?",
    "Do you remember my name?",
]
REPLY = "Sure, I can do that. Let me know if anything comes to mind."


def first_token_ms(messages: list[dict[str, str]]) -> float:
    body = {"model": "jason-llm", "messages": messages, "max_tokens": 2, "temperature": 0, "stream": True}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data: {"):
                choices = json.loads(line[6:]).get("choices") or [{}]
                if choices[0].get("delta", {}).get("content"):
                    ms = (time.perf_counter() - t0) * 1000
                    resp.read()  # drain, so the connection closes cleanly
                    return ms
    raise RuntimeError("no token streamed")


def conversation(turns: int) -> list[dict[str, str]]:
    """A chat that has gone `turns` exchanges: a new prompt length each time."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for line in USER_LINES[:turns]:
        msgs += [{"role": "user", "content": line}, {"role": "assistant", "content": REPLY}]
    return msgs + [{"role": "user", "content": USER_LINES[turns % len(USER_LINES)]}]


def run(name: str, prompts: list, pause_s: float = 0.0) -> list[float]:
    times = []
    for p in prompts:
        if pause_s:
            time.sleep(pause_s)
        times.append(round(first_token_ms(p), 1))
    print(f"{name}: median {statistics.median(times):6.1f} ms   each: {times}", flush=True)
    return times


def main() -> None:
    first_token_ms(conversation(0))  # wake it up once
    same = conversation(2)
    growing = [conversation(n) for n in range(len(USER_LINES))]
    results = {
        "A_back_to_back_same": run("A back to back, same prompt ", [same] * 8),
        "B_back_to_back_growing": run("B back to back, growing     ", growing),
        "C_paused_same": run(f"C {PAUSE_S:g} s pause, same prompt  ", [same] * 6, pause_s=PAUSE_S),
        "D_growing_again": run("D B's prompts again         ", growing),
    }
    print(json.dumps({k: statistics.median(v) for k, v in results.items()}))


if __name__ == "__main__":
    main()
