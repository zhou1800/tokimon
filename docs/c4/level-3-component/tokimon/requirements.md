# Tokimon Requirements

## Overview
Tokimon is a production-grade manager/worker (hierarchical) agent system that orchestrates multi-step workflows, enforces novelty-gated retries, and persists long-term memory to disk with staged retrieval. It also provides a benchmark harness to compare a baseline single-agent runner against the hierarchical system.

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
- As a developer, I can interact with Tokimon in a local chat UI to submit goals and view responses without repeatedly invoking CLI commands.

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

#### Skill Asset Protocol: Prompt Skills + Code Skills with Metadata
- Skill gap detection triggers (SkillBuilder SHOULD propose a candidate skill when any trigger fires):
  - Repeated subtask pattern across runs: the same normalized subtask plan/subtree is repeatedly emitted across distinct runs without measurable improvement.
  - Repeated retry failures: the same `failure_signature` recurs across novelty-gated retries for the same step or task family.
  - Repeatedly re-derived tool workflow: the same normalized tool-call workflow is re-derived across runs instead of being reused as an asset.
- Skill forms:
  - Prompt Skill: a prompt-only asset (no executable code) that standardizes a workflow, rubric, or template response and is loaded at runtime.
  - Code Skill: an executable Python skill module (plus tests) that implements deterministic logic and/or orchestrates tool use and is registered with SkillRegistry.
- Required metadata for all new skills (Prompt Skill and Code Skill):
  - `name`: unique, stable identifier (kebab-case).
  - `purpose`: why the skill exists and what it achieves.
  - `contract`: explicit `inputs` and `outputs` (and side effects when applicable).
  - `preconditions`: assumptions that MUST hold before invocation (files present, environment, permissions, etc.).
  - `required_tools`: the exact tool names the skill is permitted to call (empty if none).
  - `retrieval_prefs`: how to retrieve supporting context (tags/components, preferred staged retrieval, what to avoid).
  - `failure_modes`: known ways the skill can fail, detection signals, and required fallback behavior.
  - `safety_notes`:
    - hard: MUST NOT behaviors (disallowed actions), including any behavior that violates Non-goals.
    - soft: SHOULD NOT behaviors and escalation/stop conditions.
  - `cost_energy_notes`: expected cost profile (model calls, tool calls, runtime) and when to avoid using the skill.
  - `validation_method`: how promotion is validated (tests to run, expected signals, any manual checks).
  - `version`: semantic version string.
  - `deprecation_policy`: how and when the skill is retired, replaced, or migrated (including any compatibility window).
- Promotion gate:
  - A skill MUST NOT be registered (or loaded by SkillRegistry) unless `validation_method` passes.
  - If validation fails, the skill remains a candidate draft (not registered) and the system MUST record a Lesson capturing the gap, the attempted skill, and why validation failed.
- Budget and red lines:
  - No unsafe skills: a skill MUST NOT expand Tokimon into disallowed capabilities (see Non-goals) or weaken safety controls (path traversal protections, network allowlists, novelty gates, secret redaction/denial).
  - Avoid skill sprawl: prefer generalizable skills, merge or deprecate duplicates, and require a documented ROI (time saved and/or failure reduction) before promotion.

### Workflow Orchestration
- Workflow engine supports DAG steps, typed input/output schemas, and persistent state with resume.
- Parallel execution of independent steps.
- Workflow engine supports early termination when a worker signals the overall goal is satisfied, marking remaining steps as skipped to avoid redundant work.
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
- Before retrying, the system consults retrieved Lessons and either stops or forces a strategy change when repeating a known failed pattern without evidence of novelty/progress.

### Long-term Memory
- Memory is treated as an **asset**: it is persisted, retrievable, and directly influences retry/stop decisions to avoid repeating known failures.

- Tokimon supports three memory types:
  - **Episodic memory**: run/step-specific events (attempts, failures, retries, progress signals) persisted as Lessons and traces.
  - **Semantic memory**: durable facts and requirements about the repo/system (docs, conventions, invariants) captured as Lessons or doc references.
  - **Procedural memory**: reusable how-to workflows (tool sequences, safe fixes, skills) captured as Lessons and/or Skills with validation gates.

- Lessons persisted as Markdown with a small JSON header (metadata).
- For `lesson_type in {failure,retry}`, Lesson metadata MUST include:
  - `failure_signature` (string; placeholder allowed, but MUST be non-empty)
  - `root_cause_hypothesis` (string; concise)
  - `strategy_change` (string; what changed or why it did not)
  - `evidence_of_novelty` (string; why this retry is not identical)
  - `retrieval_tags` (list of strings; MUST include at least: task, component, tool/workflow)
- Lesson persistence MUST deterministically deny or redact secrets in both metadata and body.
- Artifacts indexed with metadata and producing steps.
- Staged retrieval:
  - Stage 1: narrow by `step_id` and `component` (high precision).
  - Stage 2: broaden by `retrieval_tags` plus adjacent components.
  - Stage 3: target by `failure_signature` similarity (and tags/components).
- Retrieval callers MUST supply `component`, `retrieval_tags` (or tags), and `failure_signature`.
- Default lexical index (sqlite FTS or BM25-like) with an interface for optional embeddings later.

- Memory-influenced action:
  - On failure, the system MUST write a `lesson_type=failure` Lesson (kept concise) that includes the required charter fields and retrieval tags.
  - Before retrying, the system MUST consult staged retrieval and either stop or force a strategy change when repeating a known failed pattern without evidence of novelty/progress.
  - When memory indicates a repeated failure, the system SHOULD prefer known safe fixes and existing Skills over re-deriving risky actions.

### Tools
- FileTool: safe read/write within workspace and prevents path traversal.
- PatchTool: apply unified diffs with validation.
- PytestTool: run pytest, capture output, pass/fail counts, failing tests list.
- GrepTool: search within repo with bounded output.
  - Uses `rg` (ripgrep) when available, otherwise falls back to a Python regex scan.
  - Output is bounded by default to prevent OOM and oversized traces:
    - Config surface: `TOKIMON_GREP_MAX_BYTES` (default: 200_000; `0` disables the cap).
    - Tool result includes `data.output` (string) and `data.truncated` (bool).
  - Default excludes apply only when `path` is omitted (repo-wide search). Excluded by default:
    - `**/runs/**`, `**/.tokimon-tmp/**`, `**/.venv/**`, `**/node_modules/**`, `**/dist/**`, `**/build/**`, `**/*.jsonl`, `**/*.ndjson`
    - Supplying an explicit `path` disables these default excludes so targeted searches (e.g., within `runs/`) still work.
- WebTool: fetch URL content (and optional lightweight search) with byte/time limits.
  - Networking supports a two-layer allowlist model: an operator-configured org allowlist (maximum destinations) plus an optional request allowlist (must be a subset).
  - WebTool may inject per-domain secret headers from environment-backed configuration (domain secrets) without exposing raw credential values in tool outputs.
  - Default configuration surface: `TOKIMON_WEB_ORG_ALLOWLIST`, `TOKIMON_WEB_REQUEST_ALLOWLIST`, and `TOKIMON_WEB_DOMAIN_SECRETS_JSON`.
- Tools expose structured schemas and outputs.

#### Tool Invocation Protocol (Worker ↔ Tools)
- Workers may request tool calls by returning `tool_calls` in the model response:
  - `{"tool_calls": [{"tool": "grep", "action": "search", "args": {"pattern": "...", "path": "..."}, "call_id": "call_1"}]}`
  - `call_id` is optional but, when present, MUST be echoed back in the tool result payloads so models can correlate results.
- A response is considered **final** when it includes `status` (SUCCESS|FAILURE|BLOCKED|PARTIAL).
- Tool results are fed back into the worker loop as structured records; workers report:
  - `metrics.model_calls`, `metrics.tool_calls`, `metrics.elapsed_ms`, and `metrics.iteration_count`
- Workers may request early workflow termination by setting:
  - `metrics.terminate_workflow: true` (and optional `metrics.terminate_reason`)
  - The runner marks remaining steps as `SKIPPED` and completes the workflow when safe to do so.

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

### Trace & Loop Unrolling
- `trace.jsonl` captures workflow state transitions plus unrolled worker loops (model calls + tool calls/results).
- Trace events include stable identifiers when available (task_id, step_id, worker role, call_id, tool_call_id, call_signature) and use bounded payload sizes (truncate large fields).

### Model Integration
- Abstract `LLMClient.send(messages, tools=None, response_schema=None)`.
- Provide stub adapter, deterministic mock adapter, and a documented placeholder for a real adapter.
- Optional real adapter: Codex CLI-backed client that shells out to `codex exec` and returns structured JSON (controlled via `TOKIMON_LLM=codex` or CLI flags).
- Optional real adapter: Claude Code CLI-backed client that shells out to `claude` and returns structured JSON (controlled via `TOKIMON_LLM=claude` or CLI flags).
- Claude Code CLI invocation: send prompts via stdin in `--print` mode (`claude --print --input-format text --output-format json`) and optionally pass a settings file via `--settings <path>` (mirrors `~/clover/joey-playground/apps/ai-agent-cli`).
  - Config surface (env): `CLAUDE_CODE_CLI` (binary override), `TOKIMON_CLAUDE_MODEL`, `TOKIMON_CLAUDE_TIMEOUT_S`, `TOKIMON_CLAUDE_SETTINGS_PATH` or `TOKIMON_CLAUDE_SETTINGS_JSON`, `TOKIMON_CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS`, `TOKIMON_CLAUDE_ARGS`.
- Codex CLI prompt rendering is deterministic and caching-friendly (stable tool ordering; explicit sections such as `<permissions instructions>` and `<environment_context>`).
- No hard dependency on a vendor SDK.
- Delegation recursion safety: when Tokimon launches an agent CLI (Codex or Claude), it MUST mark the subprocess environment with `TOKIMON_DELEGATED=1` and increment `TOKIMON_DELEGATION_DEPTH` (defaulting from 0 to 1), and include the delegation depth in prompt context.
- Codex CLI ripgrep guard: when launching Codex CLI, Tokimon MUST set `RIPGREP_CONFIG_PATH` for the Codex subprocess to a workspace-local guard config at `<workspace>/.tokimon-tmp/tokimon-codex.ripgreprc` to prevent OOM from scanning generated artifacts.
  - Preserve user config: if the incoming environment has `RIPGREP_CONFIG_PATH` pointing to a readable file, prepend its contents to the generated guard config before Tokimon guard flags.
  - Default exclusions (minimum): `**/runs/**`, `**/.tokimon-tmp/**`, `**/.venv/**`, `**/node_modules/**`, `**/dist/**`, `**/build/**`, `**/*.jsonl`, `**/*.ndjson`.
  - Output bound: `--max-columns=<N>` and `--max-columns-preview` (default `N=4096`; `N=0` disables max-columns flags).
  - Config surface: `TOKIMON_CODEX_RIPGREP_GUARD` (default: true) enables/disables the guard; `TOKIMON_CODEX_RIPGREP_MAX_COLUMNS` (default: 4096; `0` disables) controls max-columns behavior.

### Benchmarks & Harness
- At least 8 self-contained tasks with task specs, starter code, pytest acceptance tests, and intermediate scoring signals.
- Mix of debugging, refactor, algorithms, concurrency/stateful, parsing/evaluator, and performance constraints.
- Harness compares BaselineRunner vs HierarchicalRunner, records metrics, and writes JSON + Markdown reports with links to artifacts.
  - Metrics include at least: pass/fail counts, wall time, model calls, tool calls, and Lessons produced.

### CLI
- Commands: auto, run-task, run-suite, resume-run, inspect-run, list-skills, build-skill, self-improve, chat-ui, gateway, memory, sessions, doctor, health.
- Prompt-driven entrypoint: `tokimon auto "<prompt>"` routes to the appropriate mode by asking an AI router (Codex/Claude) to return a concrete Tokimon argv list.
  - Output contract: the router returns JSON containing `argv: string[]` (argv excludes the leading `tokimon`).
  - Validation: Tokimon MUST validate the router argv against the CLI parser (unknown commands/options are rejected) and MUST prevent `auto` recursion.
  - Fallback: if the router fails (missing CLI, timeout, invalid JSON, invalid argv), Tokimon falls back to deterministic heuristic routing; prompts that ask to learn/improve route to `tokimon self-improve`.
- Default `--help` output minimizes option surface by hiding advanced flags while still accepting them for power users.
- CLI outputs are structured and point to run artifacts.
- Memory (OpenClaw-inspired, Phase 1): `tokimon memory` manages local lesson indexing and search.
  - Subcommands: `status`, `index`, `search`.
  - Common flags:
    - `--root PATH` (default: `<workspace>/memory`)
    - `--json` emits stable machine-readable JSON
    - `--verbose` prints additional diagnostics (must not break `--json`)
  - `memory status` flags:
    - `--deep` includes additional index/file reconciliation details
    - `--index` triggers a reindex when the store is dirty
  - `memory search`:
    - Query input: positional `[query]` or `--query <text>`; if both are provided, `--query` wins; if neither is provided, exit non-zero.
    - `--limit N` limits the number of lesson ids returned.
- Sessions (OpenClaw-inspired, Phase 1A): `tokimon sessions` lists self-improve runs (default: `<workspace>/runs/self-improve`).
  - Flags:
    - `--root PATH` overrides the runs root
    - `--active MINUTES` filters to runs modified recently
    - `--json` emits stable machine-readable JSON
- Doctor (OpenClaw-inspired, Phase 1): `tokimon doctor` runs local readiness checks and returns non-zero when any required check fails.
  - Output: default human output; `--json` emits a stable machine-readable report.
  - Repairs: `--repair` / `--fix` attempts only safe, non-destructive repairs; otherwise it reports suggested manual remediation.
  - Minimum checks:
    - Git/worktree readiness: clean checkout (no uncommitted changes) and writable worktree.
    - Codex CLI availability: `codex` on PATH and `codex --version` succeeds.
    - Port availability: report availability/conflicts for Chat UI default port 8765 and Gateway default port 8765.
    - Required docs present: `AGENTS.md`, `docs/helix.md`, `docs/repository-guidelines.md`.

- Health (OpenClaw-inspired, Phase 1): `tokimon health` checks a running Tokimon Gateway's WebSocket `health` RPC and exits non-zero on failure.
  - Flags:
    - `--url` (default: `ws://127.0.0.1:8765/gateway`)
    - `--timeout-ms`
    - `--json` emits stable machine-readable JSON: `{ok, url, elapsed_ms, error, details}`
    - `--verbose` prints additional diagnostic steps (must not break `--json` output).
  - Protocol:
    - Connect to the Gateway WebSocket endpoint.
    - Receive the `connect.challenge` event.
    - Send a `connect` request with `minProtocol=1`, `maxProtocol=1`, plus `client`, `role`, and `scopes`.
    - Send a `health` request and require an `ok:true` response with payload `{ok:true}`.
  - Exit codes:
    - `0` when the Gateway health is ok.
    - `1` otherwise.

### Chat UI
- `tokimon chat-ui` starts a local web server (binds loopback by default) that serves a single-page chat UI.
- Health endpoint: `GET /healthz` returns JSON indicating the server is running.
- Chat endpoint: `POST /api/send` accepts JSON `{message: string, history?: [{role, content}]}` and returns a structured JSON reply including `status` and a human-readable assistant message (in `reply` or `summary`).
- The chat handler uses the same tool set as the hierarchical runner (file, grep, patch, pytest, web).
- Default LLM provider is `mock`; `--llm codex` / `--llm claude` (or `TOKIMON_LLM=codex|claude`) enables the corresponding CLI-backed client.

### Gateway Server (OpenClaw-Inspired, Phase 1)
- `tokimon gateway` starts a local server that supports:
  - Existing Chat UI HTTP endpoints: `GET /healthz`, `POST /api/send`
  - A WebSocket control-plane endpoint at `GET /gateway` (WS upgrade)
- The WebSocket endpoint uses a minimal OpenClaw-inspired framing:
  - Request: `{type:"req", id, method, params}`
  - Response: `{type:"res", id, ok, payload|error}`
  - Event: `{type:"event", event, payload}`
- Handshake:
  - On socket open, the server emits `connect.challenge`.
  - The first client request MUST be `connect` and MUST pass protocol version validation.
- Phase 1 methods:
  - `health`: returns `{ok:true}`
  - `send`: invokes the same logic as `/api/send` and requires an idempotency key.
- The Gateway protocol surface and Phase 2 TODOs are documented in `docs/gateway.md`.

### Self-Improvement Mode (Multi-Session / Batch)
- When invoked with a self-improvement goal, the system can accept optional “inputs”:
  - URL (http/https), local file path, or inline text (or none).
- If `--input` is not provided, the system may auto-detect URL(s) embedded in the `--goal` text and fetch at least the first URL as the session input payload (bounded by byte/time limits and the WebTool network policy).
- The system runs a batch of N independent improvement sessions in parallel.
- Self-improve CLI LLM default: `--llm` defaults to `$TOKIMON_LLM` when set; otherwise it defaults to `mixed`.
- Mixed-provider mode: when `--llm mixed`, enforce a deterministic `claude:codex=1:4` session mix by assigning Claude to session indices 1, 6, 11, ... (i.e., `(index - 1) % 5 == 0`) and Codex to the other sessions. `--sessions` MUST be a multiple of 5 (default: 5).
- Before launching each batch, the system runs an evaluation on the current master workspace (pytest by default) and passes a compact summary (pass/fail counts + failing test ids) into every session as context.
- Each session:
  - Materializes the master workspace into an isolated session workspace using `git worktree` (detached HEAD) so sessions can run in parallel without colliding on files.
  - Self-improve requires the master workspace to be a clean git checkout (no local changes) so worktrees and merges are deterministic; otherwise it aborts with an actionable error.
  - Runs the hierarchical agent system within that workspace to attempt improvements.
  - Runs the configured evaluation command after each workflow step when `pytest_args` are provided, so retry/progress gating has objective signals.
  - Evaluates the result (pytest by default; optionally benchmark suite).
  - Produces a session report, metrics, and a diff/changed-file set.
  - After each batch:
    - A comparer selects a winner by deterministic criteria (tests passing is primary).
    - A merger applies the winner back onto master (restricted to configured paths; defaults should include `src/` and `docs/`) using a conflict-aware git integration:
      - Create a temporary commit capturing the winner changes.
      - Apply via `git merge --squash` onto master.
      - Re-evaluate master; on success, commit the squashed changes.
      - Use an OS-level lock so multiple self-improve runs perform safe queued merges into the same checkout.
      - On merge conflicts, automatically resolve (prefer winner changes) and continue; on failing evaluation, abort and leave master unchanged.
- The system runs up to the configured number of batches, even when merge is disabled (report-only mode) or when a batch fails to produce a mergeable winner.

#### Parallel Exploration Protocol (Diverse Paths, Deterministic Selection)
- For each batch, Tokimon MUST run N parallel “paths” (sessions) that are meaningfully different in at least 2 dimensions:
  - decomposition (plan granularity),
  - root-cause hypothesis,
  - tool sequence,
  - skill usage.
- Each path MUST be assigned a deterministic `path_charter` derived from `session_id` and recorded in:
  - the session entry-point prompt,
  - the session report JSON,
  - the per-attempt experiment summary artifact.
- Each path MUST write a per-attempt experiment summary artifact that includes (minimum):
  - `plan` (short plan),
  - `path_charter` (dimensions above),
  - `self_critique` (1 paragraph; failure modes + confidence),
  - `lessons` (list of Lesson strings produced by the path),
  - plus the evaluation-first experiment-loop fields (baseline, post-change, delta, causal mechanism, pass condition).
- The batch report MUST include:
  - a diversity summary/check (pass/fail + details),
  - the declared deterministic scoring function (declared before scoring),
  - a per-path comparison table,
  - winner rationale by score (not narrative preference),
  - why non-winners lost and what Lesson(s) they produced.
- Winner selection MUST remain deterministic and aligned with `docs/tokimon-constitution.md` (do not use energy as a selector).

#### Self-Improve Session Context Contract
- Each session receives:
  - The raw `goal` string.
  - A session strategy hint (diverse across sessions).
  - The optional input payload content (URL/file/text) when provided.
  - The master evaluation summary for the batch (pytest counts + failing test ids).
- Workers are expected to:
  - Clarify assumptions inline when the goal is underspecified (ask questions only when strictly necessary).
  - Produce a structured plan (workflow steps) with verifiable subgoals.
  - Prefer writing/expanding tests and updating docs before code changes (Helix).
  - Use tools for repo context retrieval (grep/file), patching, and evaluation.

#### Self-Improve Entry-Point Request Loop
- For entry-point self-improve tasks, each session follows this loop:
  - Understand the user request and ask clarifying questions immediately when it is ambiguous.
  - Once clear, generate the task prompt for the attempt.
  - Run the agent workflow with that prompt.
  - Monitor and report progress per attempt in the session report artifact.
  - Verify the outcome using workflow status plus evaluation checks.
  - If verification fails, restart from prompt generation (retry loop) until success or attempt budget is exhausted.

#### Self-Improve Evaluation-First Experiment Loop (Required)
- Self-improvement MUST be treated as an experiment loop:
  1) Run a baseline evaluation and summarize results.
  2) Propose the smallest candidate change(s) that could move the evaluation signal.
  3) Apply changes in short, verifiable steps.
  4) After each step, re-run evaluation signals and log progress.
  5) If progress stalls, change strategy (do not repeat identical attempts).
  6) If evaluation regresses, undo the change and record a Lesson describing the regression.
- Session reports and the batch report MUST include:
  - baseline evaluation summary,
  - post-change evaluation summary,
  - delta (improvement or regression),
  - a brief causal mechanism hypothesis linking the change to the delta,
  - an explicit pass condition for the run (chosen deterministically for auditability).
- Reporting format requirements (minimum):
  - Baseline evaluation summary MUST be taken before any session changes and include `ok`, `passed`, `failed`, and a bounded list of `failing_tests` identifiers.
  - Post-change evaluation summary MUST be taken after the final accepted change and include the same fields.
  - Delta MUST be computed as `post-change - baseline` at minimum for `passed` and `failed`.
  - Causal mechanism hypothesis MUST be written by the agent as a short, falsifiable explanation connecting the change(s) to the delta (do not invent mechanisms not supported by artifacts).
- Pass condition MUST be chosen deterministically from the baseline (e.g., if `failed > 0` then “Reduce failing tests by >= 1”; else an energy-budget/quality maintenance condition).

#### Resource Safety Directive: Hard vs Soft Constraints (Self-Improve)
- The self-improve entry-point prompt MUST declare:
  - a resource plan (planned time, planned memory, planned energy, and planned concurrency),
  - a risk register (top risks, triggers, mitigations),
  - explicit stop conditions (hard vs soft, including `PARTIAL` behavior).
- Self-improve Markdown reports MUST include the same sections (resource plan, risk register, stop conditions) and MUST include an audit log that records attempted actions, refused/blocked actions (with reasons), and mitigations applied.
- Hard red line enforcement:
  - Unsafe goals (e.g., requests for cyber exploitation, credential theft, malware, or data exfiltration) MUST be refused immediately (status `BLOCKED`) before invoking any agent execution.
  - The refusal MUST be logged in the audit log with a clear reason.
- Soft red line mitigation:
  - When soft limits trigger (e.g., repeated timeouts/tool failures, repeated retries without novelty, evaluation regression), Tokimon MUST apply deterministic degradations (minimum: reduce concurrency and shorten context) and MUST log the mitigation and outcome.
  - If mitigations are exhausted or remaining budget is too low to verify safely, Tokimon MUST stop early and return `PARTIAL` with best artifacts plus an actionable next-step plan.
- Planned vs actual reporting:
  - Reports MUST track planned vs actual time (elapsed), memory (best-effort), and energy (`model_calls + tool_calls`), and MUST record which stop condition (hard/soft) fired when returning `BLOCKED` or `PARTIAL`.

#### Constitution Enforcement
- The Tokimon Constitution at `docs/tokimon-constitution.md` is binding for all self-improve runs.
- `src/self_improve/orchestrator.py:_entrypoint_prompt` must begin with a 1-paragraph Constitution Acknowledgement, then list the Immutable Invariants, and include the exact heading `## Evaluation Plan (Required)`.
- Winner selection is deterministic and does not depend on completion order; if scores tie, the winner is the lowest `session_id` (lexicographic).
- `src/self_improve/orchestrator.py:_report_to_markdown` must include the headings `## Constitution Acknowledgement`, `## Scoring Rubric`, `## Energy Budget`, and `## Audit Log`.
- Energy budget reporting must include planned vs actual energy; actual energy is the sum of `(model_calls + tool_calls)` across all sessions in the report.
- `src/self_improve/orchestrator.py:_entrypoint_prompt` and `src/self_improve/orchestrator.py:_report_to_markdown` MUST reflect the Evaluation-First Experiment Loop requirements above (baseline, post-change, delta, causal mechanism, pass condition).

## Repository Layout
- Tokimon project root lives under `src/` in this repository.
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
- `tokimon chat-ui` starts a local server where `GET /healthz` and `POST /api/send` return successful JSON responses.
- `tokimon doctor` runs the Phase-1 checks, supports `--json`, and has deterministic unit tests for checks and JSON output.
