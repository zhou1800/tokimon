# Tokimon v2 Test Plan

## Current Executable Coverage

The baseline product rules are currently enforced in `tests/test_engine.py`.

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

## Planned Coverage

- state load/save round-trip tests
- CLI integration tests
- benchmark harness tests once evaluation exists
- regression tests for future autonomous background learning
