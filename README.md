# Tokimon

Doc-led hierarchical manager/worker agent system with workflow orchestration, novelty-gated retries, offline benchmarks, and a self-improvement mode.

The CLI entrypoint is `tokimon`.

## Install
Run from the repository root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e src[dev]
```

## Use the CLI
Show commands:
```bash
source .venv/bin/activate && tokimon --help
```

Run the benchmark suite (baseline vs hierarchical) and write a report under `runs/`:
```bash
source .venv/bin/activate && tokimon run-suite
```

Run a single benchmark task:
```bash
source .venv/bin/activate && tokimon run-task --task-id stats-summary
source .venv/bin/activate && tokimon run-task --task-id lru-cache --runner baseline
```
Task ids live in `src/benchmarks/tasks/*/task.json`.

Inspect or resume a run:
```bash
source .venv/bin/activate && tokimon inspect-run --run-path runs/run-<id>
source .venv/bin/activate && tokimon resume-run --run-path runs/run-<id>
```

List skills (built-in + generated):
```bash
source .venv/bin/activate && tokimon list-skills
```

Start the local chat UI (open the printed URL in a browser):
```bash
source .venv/bin/activate && tokimon chat-ui

# Use a real model via Codex CLI:
source .venv/bin/activate && tokimon chat-ui --llm codex

# Or via Claude Code CLI:
source .venv/bin/activate && tokimon chat-ui --llm claude
```

Self-improve (experimental; uses the current working directory as the "master" workspace):
```bash
source .venv/bin/activate && tokimon self-improve --goal "Improve tokimon based on docs and failing tests."

# Use a real model via Codex CLI (requires `codex` on PATH, or set CODEX_CLI=/path/to/codex):
source .venv/bin/activate && tokimon self-improve --llm codex --goal "Improve tokimon based on docs and failing tests."

# Use a real model via Claude Code CLI (requires `claude` on PATH, or set CLAUDE_CODE_CLI=/path/to/claude):
source .venv/bin/activate && tokimon self-improve --llm claude --goal "Improve tokimon based on docs and failing tests."

# Mixed-provider self-improve (exact claude:codex=1:4; requires --sessions multiple of 5):
source .venv/bin/activate && tokimon self-improve --llm mixed --sessions 5 --goal "Improve tokimon based on docs and failing tests."
```

Note: by default, self-improve uses `MockLLMClient` (no real model calls). To use an agent CLI as the LLM, pass `--llm codex|claude|mixed` (or set `TOKIMON_LLM`).

## Development
Run tests:
```bash
source .venv/bin/activate && pytest --maxfail=1 -c src/pyproject.toml src/tests
```

## Docs
- Architecture/specs: `docs/c4/`
- Requirements: `docs/c4/level-3-component/tokimon/requirements.md`
- Test plan: `docs/c4/level-3-component/tokimon/test-plan.md`
- AI agent workflow rules: `AGENTS.md` and `docs/helix.md`
