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

## About me
I'm a systems engineer (ex-Google) with some experience in inference performance, I studied mechatronics engineering for my bachelors so i am somewhat familiar with real time controls and automation, for my masters i studied computer science with a focus on artificial intelligence, during that time i learned alot about reinforcement learning, properties like DQN, PPO, Actor critic etc. for my career i worked under a performance analysis and optimization for AI team at google and learned a bit on optimizing performance of models on google tpus. My knowledge on these topics are very limited and might be outdated so please be patient with me

## Working style
- Be direct: if an approach is weak, say so and suggest a better one
- For the learning system, be explicit about what is learning: the reward signal, what gets updated, and how often
- Check current docs before using a library API you're unsure of; this stack changes fast
- Flag any change that conflicts with docs/decisions.md

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
