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

Result (2026-10-30): A 35 ms, B 42, C 356, D 39. The pause is the cause.
Second mode, JASON_PROBE=wake, measures what a fix needs (about 4 minutes):

  decay  first token after pauses of 0.5 to 10 s      how fast it goes cold
  wake   10 s pause, a one-token warm-up, then wait    does warming up first
         0.2 / 0.5 / 1 s before the real request       work, and how early

    PowerShell: $env:JASON_PROBE="wake"; ... | docker compose exec -T -e JASON_PROBE brain python -

Standard library only, so it runs in the brain image as is.
"""

import json
import os
import statistics
import time
import urllib.request

BASE = os.environ.get("JASON_LLM_URL", "http://llm:8000")
PAUSE_S = float(os.environ.get("JASON_PROBE_PAUSE_S", "10"))
WAKE_IDLE_S = float(os.environ.get("JASON_PROBE_IDLE_S", "10"))

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


def wake_mode() -> None:
    same = conversation(2)
    first_token_ms(same)
    decay = {}
    for pause in (0.5, 1, 2, 3, 5, WAKE_IDLE_S):
        decay[pause] = run(f"decay: after {pause:>4g} s pause  ", [same] * 3, pause_s=pause)
    wake = {}
    for lead in (0.2, 0.5, 1.0):
        times = []
        for _ in range(3):
            time.sleep(WAKE_IDLE_S)
            first_token_ms(conversation(0))  # the warm-up, as speech_start would trigger
            time.sleep(lead)
            times.append(round(first_token_ms(same), 1))
        wake[lead] = times
        print(f"wake: {WAKE_IDLE_S:g} s idle, warm-up, {lead:g} s later: median {statistics.median(times):6.1f} ms   each: {times}", flush=True)
    print(json.dumps({"decay": {k: statistics.median(v) for k, v in decay.items()},
                      "wake": {k: statistics.median(v) for k, v in wake.items()}}))


def main() -> None:
    if os.environ.get("JASON_PROBE") == "wake":
        wake_mode()
        return
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
