# Tokimon v2 Test Plan

## Current Executable Coverage

The baseline product rules and persistence contract are currently enforced in `tests/test_engine.py` and `tests/test_persistence.py`.

### T1. Idle Self-Improvement

- Feed Tokimon tokens
- Run a bounded idle cycle
- Verify tokens are consumed and skill levels increase

### T2. Directed Learning Priority

- Feed Tokimon tokens
- Add at least two directions with different priorities
- Run an idle cycle
- Verify the highest-priority direction is selected first

### T3. Task-Time Preparation

- Feed Tokimon tokens
- Add a direction and request a task with explicit skills
- Run task preparation
- Verify task-relevant skills are selected and tokens are spent from the preparation budget

### T4. Durable Snapshot Round-Trip

- Save a non-trivial Tokimon state to disk
- Load it back through the persistence boundary
- Verify durable learning facts survive and derived runtime data is rebuilt

### T5. Snapshot Migration

- Load a version 2 state file
- Verify it migrates to the current snapshot version
- Verify removed persisted fields such as `quality_score` do not survive rewrite

### T6. Malformed File Recovery

- Attempt to load a malformed state file
- Verify Tokimon fails safely instead of overwriting the file with defaults

### T7. Runtime-Only Serialization Boundary

- Populate command context, caches, rankings, approvals, and daemon coordination fields
- Save state
- Verify runtime-only fields are absent from the durable snapshot

## Planned Coverage

- CLI integration tests
- benchmark harness tests once evaluation exists
- regression tests for future autonomous background learning
