# Tokimon v2: Container View

## Containers

### CLI

- File: `src/tokimon/cli.py`
- Responsibility: parse commands, load/save state, expose user operations

### Learning Engine

- File: `src/tokimon/engine.py`
- Responsibility: choose learning targets, spend tokens, prepare for tasks, compute quality score

### State Model

- Files: `src/tokimon/models.py`, `.tokimon/state.json`
- Responsibility: represent persistent skills, directions, task history, and improvement history

### Verification Layer

- File: `tests/test_engine.py`
- Responsibility: prove the three baseline product rules still hold

## Main Data Flow

1. User invokes the CLI.
2. CLI loads Tokimon state from disk or creates a default state.
3. Engine mutates state according to the requested operation.
4. CLI writes updated state back to disk.
5. Tests and future benchmarks validate that the behavior matches the product rules.

## Near-Term Extension Points

- benchmark runner for measurable quality claims
- background improvement loop
- richer task execution beyond preparation guidance
- memory and retrieval beyond flat skill levels
