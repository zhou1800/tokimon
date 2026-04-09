# Repository Guidelines

These rules apply to the active Tokimon v2 root repository.

## Active Scope

- Root files and `src/tokimon/` are the active app.
- `archive/` is read-only historical context unless the user explicitly asks to revive something from it.

## Runtime

- Python package: `tokimon`
- Minimum Python: `3.11`
- CLI entrypoint: `tokimon`
- Default state file: `.tokimon/state.json`

## Common Commands

Create an environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run tests:

```bash
pytest
```

Run the CLI without installation during local iteration:

```bash
PYTHONPATH=src python3 -m tokimon --help
```

## Current Source Map

- `src/tokimon/models.py`: persistent state and record types
- `src/tokimon/engine.py`: token feeding, idle self-improvement, directed learning, task preparation
- `src/tokimon/cli.py`: user-facing command layer
- `tests/test_engine.py`: executable baseline for the v2 product rules

## Doc Update Standard

- Keep docs short and operational.
- Prefer updating existing active docs over adding new broad narratives.
- When behavior changes, update the smallest doc that future agents would actually consult.
- If a doc does not help the next agent make a correct change, delete or compress it.
