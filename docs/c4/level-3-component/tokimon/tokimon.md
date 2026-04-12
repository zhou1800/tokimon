# Tokimon v2: Component View

## Components

### Command Router

- File: `src/tokimon/cli.py`
- Exposes `init`, `feed`, `direct`, `idle`, `task`, and `status`
- Owns user input parsing, settings-aware defaults, and output formatting

### Durable Snapshot

- File: `src/tokimon/models.py`
- Type: `TokimonSnapshot`
- Survives CLI runs through `.tokimon/state.json`
- Stores only durable learning facts: token balances, cumulative counters, skill records, directions, bounded task/improvement history, and lifecycle timestamps

### Settings Layer

- Files: `src/tokimon/models.py`, `src/tokimon/persistence.py`
- Type: `TokimonSettings`
- Merges safe project defaults from `tokimon.settings.json` with user-owned overrides from `.tokimon/user-settings.json`
- Shared settings may influence defaults, but risky runtime flags do not auto-enable without the user-owned layer

### Runtime Session

- File: `src/tokimon/models.py`
- Type: `TokimonSessionState`
- Never serialized
- Holds current command context, caches, derived rankings, temporary approvals, and future daemon coordination data

### Persistence Boundary

- File: `src/tokimon/persistence.py`
- Functions: `load_state`, `save_state`, `load_settings`, `migrate_snapshot`
- Validates file shape, applies explicit versioned migrations, rebuilds runtime-only derived data, and writes with safe replacement plus rolling backups

### Decision Policy

- File: `src/tokimon/policy.py`
- Functions: `choose_improvement_target`, `extract_task_skills`, `plan_task_preparation`, `assess_confidence`
- Pure layer only: accepts immutable policy inputs, ranks candidates, allocates bounded prep spend, produces focus/confidence outputs, and returns allowed capability metadata without touching durable state

### Improvement Loop

- File: `src/tokimon/engine.py`
- Functions: `feed_tokens`, `spend_token_on_improvement`, `run_idle_cycle`
- Orchestration only: loads mutable state, calls the policy layer, applies token spending, and appends bounded improvement history

### Task Preparation

- File: `src/tokimon/engine.py`
- Function: `prepare_for_task`
- Orchestrates task runs: builds policy inputs, applies the returned prep plan to state, appends task history, and formats runtime advice

### Quality Heuristic

- File: `src/tokimon/engine.py`
- Function: `refresh_quality_score`
- Produces a local score from token consumption, skill depth, and breadth
- Recomputed after load into runtime-only session state and not persisted as source-of-truth data
- This score is not a benchmark and must not be treated as proof of superiority

## Current Gaps

- No real benchmark harness yet
- No automated background daemon
- No append-only replay ledger beyond the bounded snapshot histories
- No external model or tool integrations in the active root repo
- No retrieval memory beyond stateful skill records
