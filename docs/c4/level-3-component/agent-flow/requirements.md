# Agent-Flow Requirements

## Overview
Agent-Flow is a production-grade manager/worker (hierarchical) agent system that orchestrates multi-step workflows, enforces novelty-gated retries, and persists long-term memory to disk with staged retrieval. It also provides a benchmark harness to compare a baseline single-agent runner against the hierarchical system.

## Goals
- Orchestrate workflows with typed step contracts, dependencies, parallel execution, and resume support.
- Enforce meaningful retries with explicit novelty, progress metrics, and Lesson artifacts.
- Provide file-based long-term memory with staged retrieval and local lexical indexing.
- Support dynamic skill creation, registration, and hot reloading with tests.
- Offer a robust CLI, structured logging, and artifact persistence.
- Include a benchmark suite of hard, self-contained tasks with automated tests and scoring signals.

## Non-goals
- No cyber exploitation, credential theft, malware, or harmful capabilities.
- No benchmark tasks that require hacking, bypassing security, or collecting private data.
- No dependency on a single vendor SDK for model integration.

## User Stories
- As a developer, I can define a workflow in JSON/YAML or Python and resume it after interruption.
- As a researcher, I can compare baseline and hierarchical agents across a benchmark suite with reports.
- As a maintainer, I can add or generate new skills and register them only when tests pass.
- As an operator, I can inspect runs, artifacts, Lessons, and traces for debugging and auditing.

## Functional Requirements
### Hierarchical Agents
- Manager agent produces executable workflows with step contracts and dependencies.
- Workers are specialized (Planner, Implementer, Debugger, Reviewer, TestTriager, SkillBuilder).
- Worker outputs are structured and include status, summary, artifacts, metrics, next_actions, failure_signature.
- Manager tracks delegation graph, avoids cycles, and enforces progress-based continuation.
- Retries require a Lesson artifact and a changed strategy/context.

### Dynamic Skills
- SkillRegistry loads built-in and generated skills and supports hot reload.
- SkillBuilder pipeline creates a new skill module plus manifest entry, includes tests when feasible, and registers only on passing tests.
- Generated skills live under `src/skills_generated/`.
- Skill creation requires a justification and a Lesson capturing gap and expected benefit.

### Workflow Orchestration
- Workflow engine supports DAG steps, typed input/output schemas, and persistent state with resume.
- Parallel execution of independent steps.
- Artifact store for per-step outputs.
- DSL: JSON or YAML for workflows; Python API for programmatic construction.

### Restart/Retry Controls
- Novelty-gated retries require a Lesson and a change in at least one meaningful dimension.
- call_signature hash includes (goal, step_id, worker_type, key inputs, strategy_id, retrieval_stage).
- Block identical reruns without measurable progress.
- Failure signature de-dup hashes (task id, call_signature, failure_signature).
- Progress metrics are explicit, logged, and required to justify retry.
- Cycle detection for delegation graph and repeated subtrees with no new artifacts.
- Manager refuses to retry without a concrete Lesson and plan change.

### Long-term Memory
- Lessons persisted as Markdown with a small JSON header (metadata).
- Artifacts indexed with metadata and producing steps.
- Staged retrieval: Stage 1 (tight), Stage 2 (broaden), Stage 3 (targeted by tags/components/failure signatures).
- Default lexical index (sqlite FTS or BM25-like) with an interface for optional embeddings later.

### Tools
- FileTool: safe read/write within workspace and prevents path traversal.
- PatchTool: apply unified diffs with validation.
- PytestTool: run pytest, capture output, pass/fail counts, failing tests list.
- GrepTool: search within repo.
- Tools expose structured schemas and outputs.

#### Tool Invocation Protocol (Worker ↔ Tools)
- Workers may request tool calls by returning `tool_calls` in the model response:
  - `{"tool_calls": [{"tool": "grep", "action": "search", "args": {"pattern": "...", "path": "..."}}]}`
- A response is considered **final** when it includes `status` (SUCCESS|FAILURE|BLOCKED|PARTIAL).
- Tool results are fed back into the worker loop as structured records; workers report:
  - `metrics.model_calls`, `metrics.tool_calls`, `metrics.elapsed_ms`, and `metrics.iteration_count`

#### Planner Output Contract (Goal → Workflow)
- The Planner may return a multi-step workflow in either of these shapes:
  - `{"workflow": {"steps": [ ...step dicts... ]}}`
  - `{"task_steps": [ ...step dicts... ]}`
- Each step dict must include at least:
  - `id` (string), plus optional `name`, `description`, `worker`, `depends_on`, `inputs_schema`, `outputs_schema`, `inputs`
- If no valid steps are returned, the manager falls back to a single-step workflow.

### Parallel Execution
- Asyncio task queue with configurable concurrency and backpressure.
- Robust cancellation and timeouts at tool level.
- Deterministic run folder layout with per-step and per-worker logs and a consolidated run trace (jsonl).

### Model Integration
- Abstract `LLMClient.send(messages, tools=None, response_schema=None)`.
- Provide stub adapter, deterministic mock adapter, and a documented placeholder for a real adapter.
- Optional real adapter: Codex CLI-backed client that shells out to `codex exec` and returns structured JSON (controlled via `AGENT_FLOW_LLM=codex` or CLI flags).
- No hard dependency on a vendor SDK.

### Benchmarks & Harness
- At least 8 self-contained tasks with task specs, starter code, pytest acceptance tests, and intermediate scoring signals.
- Mix of debugging, refactor, algorithms, concurrency/stateful, parsing/evaluator, and performance constraints.
- Harness compares BaselineRunner vs HierarchicalRunner, records metrics, and writes JSON + Markdown reports with links to artifacts.
  - Metrics include at least: pass/fail counts, wall time, model calls, tool calls, and Lessons produced.

### CLI
- Commands: run-task, run-suite, resume-run, inspect-run, list-skills, build-skill, self-improve.
- CLI outputs are structured and point to run artifacts.

### Self-Improvement Mode (Multi-Session / Batch)
- When invoked with a self-improvement goal, the system can accept optional “inputs”:
  - URL (http/https), local file path, or inline text (or none).
- The system runs a batch of N independent improvement sessions in parallel.
- Each session:
  - Clones the master workspace into a session workspace (isolated copy).
  - Runs the hierarchical agent system within that workspace to attempt improvements.
  - Evaluates the result (pytest by default; optionally benchmark suite).
  - Produces a session report, metrics, and a diff/changed-file set.
  - After each batch:
    - A comparer selects a winner by deterministic criteria (tests passing is primary).
    - A merger applies the winner back onto master (restricted to configured paths; defaults should include `src/` and `docs/`).
    - Master is re-evaluated; if evaluation fails, the merge is rolled back/refused.
- The system may run multiple batches for a single improvement request.

## Repository Layout
- Agent-Flow project root lives under `src/` in this repository.
- Python sources live directly under `src/` (e.g., `cli.py`, `agents/`, `workflow/`).
- Tests live under `src/tests/`.
- Generated skills live under `src/skills_generated/`.
- Benchmarks live under `src/benchmarks/` with per-task folders.
- Project configuration lives under `src/pyproject.toml`.

## Non-functional Requirements
- Deterministic, auditable runs with structured logs and trace files.
- Safe file operations with path traversal protections.
- Local operation for memory retrieval; benchmarks are self-contained and do not require network access.
- Defaults favor returning PARTIAL with best artifacts rather than looping.

## Acceptance Criteria
- `pytest` passes for framework and benchmarks.
- `run-suite` executes end-to-end and generates JSON + Markdown reports.
- Baseline vs hierarchical comparison is produced with artifacts and Lessons.
- Parallel worker execution is implemented and configurable to very high concurrency.
- Retries are novelty-gated, log progress metrics, and persist Lessons for each retry.
- Dynamic skills are generated, tested, and registered only on passing tests.
