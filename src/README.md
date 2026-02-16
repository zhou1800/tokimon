# Tokimon

Production-grade manager/worker agent system with workflow orchestration, novelty-gated retries, and offline benchmarks.

## Quickstart
- Run commands from the repository root.
- Specs:
  - `../docs/c4/level-2-container/tokimon.md`
  - `../docs/c4/level-3-component/tokimon/requirements.md`
  - `../docs/c4/level-3-component/tokimon/test-plan.md`
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -e src[dev]`
- Run a benchmark suite: `source .venv/bin/activate && tokimon run-suite`
- Run a single task: `source .venv/bin/activate && tokimon run-task --task-id stats-summary`
- Self-improve (real LLM via Codex CLI): `source .venv/bin/activate && tokimon self-improve --llm codex --goal "Improve tokimon based on docs and failing tests."`
- Run tests: `source .venv/bin/activate && pytest --maxfail=1 -c src/pyproject.toml src/tests`
