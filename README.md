# Tokimon v2

Tokimon v2 is a token monster that improves itself by eating tokens.

The v2 reboot starts with three product rules:

1. Tokimon automatically spends available tokens on self-improvement when idle.
2. When given learning directions, Tokimon prioritizes those directions in future improvement cycles.
3. When given a task, Tokimon should use what it has learned, identify gaps, and improve task-relevant skills before answering when tokens are available.

This repository intentionally starts smaller than v1. The first goal is to make the core behavior executable and testable before adding heavier orchestration, model integrations, or UI layers.

## Why this is different from v1

`archive/tokimon-v1/` preserves the earlier system. v2 resets the root repo around a smaller core loop:

- stateful token budget
- directed learning priorities
- idle self-improvement
- task-time preparation
- benchmark-first quality goals

The "better than Claude or Codex" ambition is treated as a benchmark target, not a hardcoded claim. Tokimon needs to earn that through measured task performance.

## Quickstart

Create a virtual environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Initialize Tokimon state:

```bash
tokimon init
```

Feed Tokimon tokens:

```bash
tokimon feed --tokens 10
```

Tell Tokimon what to learn:

```bash
tokimon direct --skill python --priority 10 --note "Ship better backend code"
tokimon direct --skill testing --priority 9 --note "Improve reliability and regression coverage"
```

Let Tokimon improve itself while idle:

```bash
tokimon idle --max-cycles 5
```

Ask Tokimon to prepare for a task:

```bash
tokimon task --summary "Build a small API and test plan" --skill python --skill testing
```

Check status:

```bash
tokimon status
```

Run tests:

```bash
pytest
```

## AI Docs

The active AI-facing operating docs for v2 live at:

- `AGENTS.md`
- `docs/helix.md`
- `docs/repository-guidelines.md`
- `docs/c4/`

## State Model

Tokimon stores its working state in `.tokimon/state.json` by default. The state tracks:

- available tokens and lifetime tokens eaten
- skills and their levels
- directed learning priorities
- improvement history
- task history
- a derived quality score

## Next Steps

The current implementation is intentionally simple. The next layers after this foundation are:

- real task execution against benchmark suites
- retrieval-backed memory instead of only skill levels
- autonomous background loops
- model/tool integrations
- evaluation reports that compare Tokimon against baseline agents
