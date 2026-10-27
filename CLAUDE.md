# Voice Agent

## Overview
A voice agent that learns and grows in real time. Core priorities, in order:
1. Real-time learning using reinforcement learning principles
2. Memory: absorbs information as it arrives, stores and uses it
3. Proactivity: can speak out of turn and ask questions
4. Natural voice conversation (mic → STT → agent → TTS) with low latency

- Design doc: @docs/design.md
- Decision log: @docs/decisions.md

## Commands
<!-- Fill in once the stack is chosen -->
- Install: `...`
- Run: `...`
- Tests: `...`

## Conventions
- Log learning signals, rewards and updates so learning is observable
- Log per-stage latency in the voice pipeline
- Work one milestone at a time (see design doc §9)
- Small, focused commits with clear messages

## Rules for Claude
- Propose a plan before any multi-file change
- Ask before adding a new dependency or external service
- Explain non-obvious design choices, especially in the learning system
- Don't change the design or stack without flagging it; record agreed changes in docs/decisions.md
