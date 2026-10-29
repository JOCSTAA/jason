"""Build the spike's training and eval data from the running base model.

The behavior to teach is easy to measure: every reply ends with a short
follow-up question on its own line, starting "Curious:". The base model never
does this, so after training, the share of eval replies that do is a clean
measure of whether the lesson was learned and survived serving.

Answers come from the base model itself (served by compose.yaml's vllm, with
thinking off); the Curious line is appended. Answers that hit the token limit
are dropped, so no example teaches "stop mid-sentence, then ask". Eval topics are not in training,
so the eval tests whether the behavior generalizes. Run from the repo root:

    uv run python spikes/m0-lora-training/make_data.py
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8000/v1/chat/completions"

TOPICS = [
    "sourdough baking", "rocket propulsion", "chess openings", "beekeeping", "jazz improvisation",
    "solar panels", "marathon training", "origami", "volcanoes", "bonsai trees",
    "electric guitars", "coffee roasting", "tidal energy", "medieval castles", "birdwatching",
    "3D printing", "the stock market", "sign language", "glassblowing", "black holes",
    "composting", "rock climbing", "film photography", "the human immune system", "wind turbines",
    "board game design", "fermentation", "ocean currents", "calligraphy", "electric cars",
    "the Roman Empire", "home networking", "mushroom foraging", "public speaking", "sailing",
    "watercolor painting", "earthquakes", "chocolate making", "language learning", "drones",
    # Held out for eval: never seen in training.
    "cheese making", "the moon landings", "woodworking", "coral reefs", "cryptography",
    "knitting", "hurricanes", "robotics competitions", "tea ceremonies", "deep sea creatures",
]
EVAL_TOPICS = set(TOPICS[40:])

QUESTIONS = [
    "Can you explain {t} in a few sentences?",
    "What's one surprising fact about {t}?",
    "How would someone get started with {t}?",
    "What is a common misconception about {t}?",
    "Why does {t} matter?",
]
FOLLOW_UPS = [
    "Curious: what got you interested in {t}?",
    "Curious: have you tried any of this yourself?",
    "Curious: which part of {t} would you like to dig into next?",
    "Curious: is this for something you're working on?",
    "Curious: how much do you already know about {t}?",
]


def base_answer(question: str) -> tuple[str, bool]:
    """The base model's answer, and whether it finished (didn't hit the token limit)."""
    body = {
        "model": "jason-llm",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 512,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        choice = json.loads(r.read())["choices"][0]
    return choice["message"]["content"].strip(), choice.get("finish_reason") != "length"


def main() -> None:
    items = [(t, i, q.format(t=t)) for t in TOPICS for i, q in enumerate(QUESTIONS)]
    print(f"asking the base model {len(items)} questions ...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(lambda it: base_answer(it[2]), items))

    out = HERE / "data"
    out.mkdir(exist_ok=True)
    counts = {"train": 0, "eval": 0, "dropped": 0}
    with open(out / "train.jsonl", "w", encoding="utf-8") as train, \
         open(out / "eval.jsonl", "w", encoding="utf-8") as ev:
        for (topic, i, question), (answer, finished) in zip(items, answers):
            if not finished and topic not in EVAL_TOPICS:
                # A cut-off training answer would teach "stop mid-sentence, then
                # ask". Eval keeps every question: it only uses the questions.
                counts["dropped"] += 1
                continue
            follow_up = FOLLOW_UPS[(i + TOPICS.index(topic)) % len(FOLLOW_UPS)].format(t=topic)
            row = {"topic": topic, "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"{answer}\n\n{follow_up}"},
            ]}
            split = "eval" if topic in EVAL_TOPICS else "train"
            (ev if split == "eval" else train).write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    print(f"wrote {counts['train']} training and {counts['eval']} eval examples to {out} "
          f"({counts['dropped']} training examples dropped: the answer didn't finish within 512 tokens)")


if __name__ == "__main__":
    main()
