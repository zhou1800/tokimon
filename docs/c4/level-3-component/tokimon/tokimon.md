# Tokimon v2: Component View

## Components

### Command Router

- File: `src/tokimon/cli.py`
- Exposes `init`, `feed`, `direct`, `idle`, `task`, and `status`
- Owns user input parsing and output formatting

### State Persistence

- File: `src/tokimon/engine.py`
- Functions: `create_state`, `load_state`, `save_state`
- Ensures Tokimon can resume learning across runs

### Learning Target Selection

- File: `src/tokimon/engine.py`
- Functions: `choose_improvement_target`, `skill_priority`, `extract_task_skills`
- Chooses between directed skills, relevant task skills, and fallback general learning

### Improvement Loop

- File: `src/tokimon/engine.py`
- Functions: `feed_tokens`, `spend_token_on_improvement`, `run_idle_cycle`
- Converts tokens into skill growth and improvement history

### Task Preparation

- File: `src/tokimon/engine.py`
- Function: `prepare_for_task`
- Uses a bounded token budget to improve task-relevant skills before giving guidance

### Quality Heuristic

- File: `src/tokimon/engine.py`
- Function: `refresh_quality_score`
- Produces a local score from token consumption, skill depth, and breadth
- This score is not a benchmark and must not be treated as proof of superiority

## Current Gaps

- No real benchmark harness yet
- No automated background daemon
- No external model or tool integrations in the active root repo
- No retrieval memory beyond stateful skill records
