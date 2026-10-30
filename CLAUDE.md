# Voice Agent

## Overview
A voice agent that learns and grows in real time. Core priorities, in order:
1. Real-time learning using reinforcement learning principles
2. Memory: absorbs information as it arrives, stores and uses it
3. Proactivity: can speak out of turn and ask questions
4. Natural voice conversation (mic → STT → agent → TTS) with low latency

- Design doc: @docs/design.md
- Decision log: @docs/decisions.md
- Survey of current stack: @docs/stack-survey.md
- Review of old work: @docs/old-repos-review.md
- Serving engine decision: @docs/spikes/m0-llm-serving.md

## Architecture and layout
Two roles, one Python package (`jason/`). The **body** role (the Mac today, possibly another machine later, so no macOS assumptions outside swappable backends) runs audio and the floor controller; the **brain** role (Windows RTX 4090 today, possibly a cloud GPU later; runs in Docker) runs the LLM, memory, learner and sleep. Each machine has an in-process **event bus**; every event published on it is written to that machine's **raw log**, and a **WebSocket link** (over Tailscale, token-authenticated) carries events between the two buses.

```
jason/
  cli.py            the `jason` command (run, new-day, sleep, gen-token)
  core/events.py    versioned event schema (pydantic); SCHEMA_VERSION
  core/bus.py       pub/sub; publish() logs first, then delivers
  core/clock.py     monotonic time (durations) vs wall time (humans)
  core/config.py    TOML config per role; link token
  core/state.py     brain's state.json: day counter + current session
  log/rawlog.py     append-only JSONL, one file per session
  net/link.py       LinkServer (brain) / LinkClient (body); auth, reconnect, clock sync
  net/protocol.py   messages on the wire; PROTOCOL_VERSION
  net/clocksync.py  NTP-style clock offset and round trip
  net/admin.py      one-shot commands to a running brain (/admin)
  body/main.py      body role (M0: pinger)
  brain/main.py     brain role (M0: ping responder)
config/body.toml, config/brain.toml  one config per role (committed, no secrets)
Dockerfile, compose.yaml             the brain's container
tests/                               pytest; test_container.py is opt-in
```

A **session** is one of Jason's days (wake → sleep), not a calendar date: one raw-log file per session, ids like `day-0007_2026-11-03`, owned by the brain and sent to the body on connect.

Jason's data lives **outside the repo** in `~/jason-data/` (`state/` for raw logs and later memory; `link_token`). On Windows: `C:\Users\<you>\jason-data`, mounted into the container.

## About me
I'm a systems engineer (ex-Google) with some experience in inference performance, I studied mechatronics engineering for my bachelors so i am somewhat familiar with real time controls and automation, for my masters i studied computer science with a focus on artificial intelligence, during that time i learned alot about reinforcement learning, properties like DQN, PPO, Actor critic etc. for my career i worked under a performance analysis and optimization for AI team at google and learned a bit on optimizing performance of models on google tpus. My knowledge on these topics are very limited and might be outdated so please be patient with me

## Working style
- Be direct: if an approach is weak, say so and suggest a better one
- For the learning system, be explicit about what is learning: the reward signal, what gets updated, and how often
- Check current docs before using a library API you're unsure of; this stack changes fast
- Flag any change that conflicts with docs/decisions.md

## Commands
- Install: `uv sync` (uv installs Python 3.12 and the locked dependencies into `.venv/`)
- Tests: `uv run pytest` (strict: `uv run pytest -W error`; container test: `JASON_DOCKER_TESTS=1 uv run pytest tests/test_container.py`)
- Token (once): `uv run jason gen-token`, then copy `~/jason-data/link_token` to the other machine
- Run the brain: `docker compose up --build -d` (logs: `docker compose logs -f brain`; stop: `docker compose down`). `config/brain.toml` uses the container's paths, so outside Docker pass a config with local paths
- Run the body (on the Mac): `uv run jason run --role body` (`--count N` to stop after N pings)
- New day (until sleep exists): `docker compose exec brain jason new-day`
- Without `--config`, commands use `config/<role>.toml`

## Conventions
- Log learning signals, rewards and updates so learning is observable
- Log per-stage latency in the voice pipeline
- Work one milestone at a time (see design doc §13)
- Small, focused commits with clear messages
- Show the changes you plan to make in the staging area before committing, so i can read and approve them

## Code conventions
- Components are asyncio tasks that talk only through the event bus (`bus.emit` / `bus.subscribe`), never by calling each other
- Every event type is a frozen pydantic model in `core/events.py` (`extra="forbid"`); changing a field's meaning bumps `SCHEMA_VERSION`, changing the wire format bumps `PROTOCOL_VERSION`
- The bus stamps `seq` and `run_id` on publish: never set them yourself, and use the event `publish()`/`emit()` return
- The raw log is append-only: never edit, rewrite or delete it
- No hard-coded addresses, ports or paths: they belong in `config/*.toml` (design §3.4 portability rules)
- Durations use `clock.monotonic_ns()`; wall time is for humans. Cross-machine timing goes through the clock offset
- Never commit Jason's data or the link token
- Every module has tests; run the full suite before staging. Comments explain why, in plain language

## Rules for Claude
- Read docs/design.md and docs/decisions.md before proposing design changes
- Propose a plan before any multi-file change
- Ask before adding a new dependency or external service
- Explain non-obvious design choices, especially in the learning system
- Don't change the design or stack without flagging it; record agreed changes in docs/decisions.md
