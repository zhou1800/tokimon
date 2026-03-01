# Tokimon Test Plan

This document maps requirements to automated tests.

## Unit Tests
- Workflow persistence and resume: serialize/deserialize DAG state and resume execution.
- Workflow early termination: when a worker signals `metrics.terminate_workflow`, remaining steps are marked skipped and are not executed.
- Workflow termination baseline guard: `metrics.terminate_workflow` is ignored for baseline steps (step id starts with `baseline`) so clean baselines do not skip planned work.
- Planner workflow generation: Planner output is converted into a multi-step workflow when provided.
- Retry novelty gating: refuse identical retry without a Lesson and changed strategy.
- Memory-informed retry gate: consult retrieved Lessons and stop or force a strategy change when repeating a known failed pattern without novelty (see `src/tests/test_retry_gate.py`).
- Failure signature de-dup: detect repeated failures via hash (task_id, call_signature, failure_signature).
- Cycle detection: detect delegation cycles and repeated subtrees without new artifacts.
- Memory staged retrieval: Stage 1/2/3 selection logic with deterministic lexical index, requiring `component`, `retrieval_tags`, and `failure_signature` inputs (see `src/tests/test_memory_retrieval.py`).
- Memory charter: Lesson schema validation for `lesson_type in {failure,retry}` and deterministic secret redaction/denial (see `src/tests/test_memory_charter.py`).
- Dynamic skill registration: register only after tests pass; hot reload behavior (see `src/tests/test_skill_builder.py`).
- Skill gap detection triggers: repeated subtask patterns, repeated retry failures, and repeatedly re-derived tool workflows create a candidate skill draft + Lesson, without registering the skill.
- Skill metadata validation: Prompt Skill and Code Skill assets require the full metadata set (name, purpose, contract inputs/outputs, preconditions, required_tools, retrieval_prefs, failure_modes, safety_notes hard/soft, cost/energy notes, validation method, version, deprecation policy).
- Skill promotion gate: register/load skills only when validation passes; failed validation leaves a candidate draft unregistered and records a Lesson.
- Skill budget/red lines: unsafe skills are rejected; sprawl guards prevent promoting redundant low-ROI skills.
- Parallel execution correctness: basic ordering, backpressure, and cancellation.
- CLI auto routing: `tokimon auto "<prompt>"` uses an LLM router to produce a validated argv list (tests stub the router/LLM for determinism and cover fallback to heuristic routing) (see `src/tests/test_cli_auto.py`).
- CLI help surface: default `--help` output hides advanced flags while still accepting them (see `src/tests/test_cli_auto.py`).
- Self-improve CLI LLM default: `--llm` defaults to `$TOKIMON_LLM` when set, else `mixed` (see `src/tests/test_cli_auto.py`).
- CLI gateway subcommands: `tokimon gateway run|health|call|probe` parse and client behavior (see `src/tests/test_gateway_cli.py`).
- CLI logs: `tokimon logs` returns Gateway log entries over the WebSocket RPC (see `src/tests/test_logs_cli.py`).
- CLI memory: `tokimon memory status/index/search` supports deterministic JSON output, query precedence (`--query` wins), and indexing/search via `--root`, `--deep`, `--index`, `--limit` (see `src/tests/test_memory_cli.py`).
- CLI sessions: `tokimon sessions` supports deterministic JSON output and `--active` filtering (see `src/tests/test_sessions_cli.py`).
- CLI status: `tokimon status --json` emits stable section keys and probes gateway WS health against an ephemeral `GatewayServer` (see `src/tests/test_status_cli.py`).
- CLI doctor: `tokimon doctor` checks and `--json` output are deterministic under dependency injection / monkeypatch (see `src/tests/test_doctor.py`).
- Tool schemas: FileTool path traversal protection, PatchTool validation + hunk header normalization, PytestTool parsing, GrepTool bounded output + default excludes, WebTool URL validation and network policy (allowlists + domain secrets).
- Worker tool loop: tool calls execute and are reflected in worker metrics (model/tool call counts).
- Worker output schema enforcement: final structured outputs validate against the per-step success schema; invalid outputs trigger bounded repair (max 2) and produce a deterministic schema-related `failure_signature` on exhaustion.
- Tool call correlation: tool calls with `call_id` are echoed into tool results and recorded in `tool_call_records`.
- Trace loop unrolling: worker model/tool calls are recorded to `trace.jsonl` with bounded payload sizes.
- Codex CLI prompt rendering: deterministic prompt envelope with stable tool ordering and explicit context sections.
- Codex CLI ripgrep guard: guard on/off, guard config contents, `RIPGREP_CONFIG_PATH` override/preservation, max-columns default and disable=0.
- Codex CLI delegation markers: subprocess env includes `TOKIMON_DELEGATED=1`, increments `TOKIMON_DELEGATION_DEPTH`, and prompt context reflects delegation depth.
- Claude CLI adapter: subprocess args include non-interactive flags (`--print`, `--input-format text`, `--output-format json`) and delegation markers are set in the subprocess environment (see `src/tests/test_claude_cli_client.py`).
- Self-improve mixed provider schedule: deterministic session_id→provider assignment and validation that mixed mode requires `--sessions` multiple of 5 (see `src/tests/test_self_improve_provider_mix.py`).
- Constitution doc exists and includes required headings (see `src/tests/test_self_improve_constitution.py`).
- Entry-point prompt includes Constitution Acknowledgement, Immutable Invariants, and `## Evaluation Plan (Required)` (see `src/tests/test_self_improve_constitution.py`).
- Winner selection tie-breaker uses lowest `session_id` when scores tie (see `src/tests/test_self_improve_constitution.py`).
- Report markdown includes constitution headings and planned vs actual energy with correct actual sum (see `src/tests/test_self_improve_constitution.py`).
- Entry-point prompt includes Evaluation-First Experiment Loop requirements (baseline, post-change, delta, causal mechanism, pass condition) (see `src/tests/test_self_improve_constitution.py`).
- Report markdown includes Evaluation-First Experiment Loop summary (baseline, post-change, delta, causal mechanism, pass condition) and includes a bounded list of failing test identifiers in baseline/post-change summaries (see `src/tests/test_self_improve_constitution.py`).
- Resource Safety Directive:
  - Entry-point prompt declares resource plan, risk register, and stop conditions (see `src/tests/test_resource_safety_directive.py`).
  - Report markdown includes the same sections and an audit log that records attempted actions, refused actions (with reasons), and mitigations applied (see `src/tests/test_resource_safety_directive.py`).
  - Hard red line goal screening refuses unsafe goals (see `src/tests/test_resource_safety_directive.py`).
  - Soft red line mitigation is logged (reduce concurrency + shorten context) and supports `PARTIAL` early-stop when verification is not feasible (see `src/tests/test_resource_safety_directive.py`).
- Parallel exploration protocol: deterministic `path_charter` per session, diversity check (pairwise differences in >= 2 dimensions), enforced per-attempt experiment summary fields (`plan`, `path_charter`, `self_critique`, `lessons`), deterministic winner selection by pre-declared score, and report includes required protocol sections/table (see `src/tests/test_parallel_exploration_protocol.py`).

## Integration Tests
- End-to-end run of at least two benchmark tasks using the mock model:
  - Validate artifacts, Lessons, and reports are produced.
  - Verify baseline and hierarchical runners both execute.

- Chat UI smoke test (mock model):
  - Start `tokimon chat-ui` (or the server module) on an ephemeral port.
  - Assert `GET /healthz` returns `{"ok": true}` (or equivalent).
  - Assert `POST /api/send` with a simple message returns a structured JSON reply.
  - Shut the server down cleanly.

- Gateway smoke test (mock model):
  - Start `tokimon gateway` (or `GatewayServer`) on an ephemeral port.
  - Assert `GET /healthz` returns `{"ok": true}`.
  - Establish a WebSocket connection to `/gateway`.
  - Assert the server emits `connect.challenge`.
  - Send a `connect` request and then validate `health` and `send` responses.
  - Shut the server down cleanly (see `src/tests/test_gateway_ws.py`).

- CLI health smoke test:
  - Start `GatewayServer` on an ephemeral port.
  - Assert `tokimon health --json --url ws://<host>:<port>/gateway` returns `{"ok": true}` and exits 0.
  - Assert an unreachable port returns `{"ok": false}` and a non-zero exit code (see `src/tests/test_health.py`).

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
  - Entry-point loop behavior:
    - Ambiguous goals block immediately with clarifying questions.
    - Verification failures trigger a retry from prompt generation in the same session.
    - Winners are merged only when verification succeeds.
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
