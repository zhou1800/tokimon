# Tokimon Test Plan

This document maps requirements to automated tests.

## Unit Tests
- Workflow persistence and resume: serialize/deserialize DAG state and resume execution.
- Workflow early termination: when a worker signals `metrics.terminate_workflow`, remaining steps are marked skipped and are not executed.
- Planner workflow generation: Planner output is converted into a multi-step workflow when provided.
- Retry novelty gating: refuse identical retry without a Lesson and changed strategy.
- Failure signature de-dup: detect repeated failures via hash (task_id, call_signature, failure_signature).
- Cycle detection: detect delegation cycles and repeated subtrees without new artifacts.
- Memory staged retrieval: Stage 1/2/3 selection logic with deterministic lexical index.
- Dynamic skill registration: register only after tests pass; hot reload behavior.
- Parallel execution correctness: basic ordering, backpressure, and cancellation.
- Tool schemas: FileTool path traversal protection, PatchTool validation, PytestTool parsing, WebTool URL validation and network policy (allowlists + domain secrets).
- Worker tool loop: tool calls execute and are reflected in worker metrics (model/tool call counts).
- Trace loop unrolling: worker model/tool calls are recorded to `trace.jsonl` with bounded payload sizes.
- Codex CLI prompt rendering: deterministic prompt envelope with stable tool ordering and explicit context sections.

## Integration Tests
- End-to-end run of at least two benchmark tasks using the mock model:
  - Validate artifacts, Lessons, and reports are produced.
  - Verify baseline and hierarchical runners both execute.

- Chat UI smoke test (mock model):
  - Start `tokimon chat-ui` (or the server module) on an ephemeral port.
  - Assert `GET /healthz` returns `{"ok": true}` (or equivalent).
  - Assert `POST /api/send` with a simple message returns a structured JSON reply.
  - Shut the server down cleanly.

- Self-improvement batch:
  - Creates multiple isolated session workspaces from a master root.
  - Uses `git worktree` (detached HEAD) for all session workspaces and aborts with an actionable error when the master is not a clean git checkout.
  - Evaluates each session and selects a winner deterministically.
  - Merges the winner back to master and re-runs evaluation.
  - Winner merge uses `git merge --squash` and commits only on passing evaluation.
  - Merges are safe queued merges via an OS-level lock; merge conflicts are auto-resolved (prefer winner changes).
  - Runs all configured batches even when:
    - merge is disabled (`--no-merge` / report-only mode), or
    - a batch fails to produce a mergeable winner (evaluation fails).
  - Passes `pytest_args` into the hierarchical runner so per-step progress metrics include test counts.
  - Auto-detects URL(s) in `--goal` and fetches content as session input when `--input` is not provided.
  - URL fetching respects the WebTool network policy (org allowlist + optional request allowlist).

## Benchmark Task Tests
- Each benchmark task includes pytest acceptance tests.
- Intermediate scoring signals are asserted by tests (e.g., partial score thresholds).

## Test Commands
- From repo root: `source .venv/bin/activate && pytest --maxfail=1 -c src/pyproject.toml src/tests`

## Coverage Expectations
- All key requirements in `docs/c4/level-3-component/tokimon/requirements.md` have corresponding tests.
