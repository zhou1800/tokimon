# Tokimon v2: Container View

## Containers

### CLI

- File: `src/tokimon/cli.py`
- Responsibility: parse commands, load state/settings, expose user operations

### Learning Engine

- File: `src/tokimon/engine.py`
- Responsibility: choose learning targets, spend tokens, prepare for tasks, recompute derived runtime values

### Persistence Boundary

- File: `src/tokimon/persistence.py`
- Responsibility: validate and migrate the durable snapshot, merge settings layers, and write state safely with backups

### State Model

- Files: `src/tokimon/models.py`, `.tokimon/state.json`, `tokimon.settings.json`, `.tokimon/user-settings.json`
- Responsibility: keep durable learning facts separate from settings and runtime-only session data

### Verification Layer

- Files: `tests/test_engine.py`, `tests/test_persistence.py`
- Responsibility: prove the product rules and persistence contract still hold

## Main Data Flow

1. User invokes the CLI.
2. CLI loads the durable snapshot from disk, layers project and user settings, and rebuilds runtime-only derived values.
3. Engine mutates only the durable learning snapshot plus ephemeral runtime session data for the current command.
4. CLI writes the updated snapshot back through the persistence boundary.
5. Tests and future benchmarks validate that the behavior matches the product rules.

## Near-Term Extension Points

- benchmark runner for measurable quality claims
- background improvement loop
- append-only audit or replay ledger alongside the bounded snapshot histories
- richer task execution beyond preparation guidance
- memory and retrieval beyond flat skill levels
