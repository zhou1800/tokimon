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
- Worker outputs are structured and include status, summary, artifacts, metrics, next_actions, failure_signature, and optional `ui_blocks`.
- Step results (per step) are represented as a deterministic JSON object with this schema:
  - `status`: `"SUCCESS" | "FAILURE" | "BLOCKED" | "PARTIAL"`
  - `summary`: string (human-readable)
  - `artifacts`: array of object (JSON-serializable)
  - `metrics`: object (JSON-serializable)
  - `next_actions`: array of string
  - `failure_signature`: string (empty when none)
  - `ui_blocks` (optional): array of UIBlock objects (see below)
- UIBlock schema (deterministic):
  - Base fields:
    - `type`: `"text" | "json" | "component"`
    - `title` (optional): string
  - When `type="text"`: `text` (string) is required.
  - When `type="json"`: `data` (any JSON-serializable value) is required.
  - When `type="component"`:
    - `component`: `"Text" | "Json" | "Panel" | "Chart" | "Form"`
    - `props`: object (JSON-serializable; validated by the frontend via Zod)
    - Component prop schemas:
      - `Text`: `{text: string}`
      - `Json`: `{data: any}`
      - `Panel`: `{title?: string, blocks: UIBlock[]}`
      - `Chart`: `{kind: "bar" | "line", title?: string, labels: string[], values: number[]}`
      - `Form`: `{title?: string, submit_label?: string, fields: [{name: string, label?: string, type: "string" | "number" | "boolean" | "json", required?: boolean, placeholder?: string}]}`
- Worker final outputs MUST validate against a per-step success schema (type checks + required keys, not just JSON parsing).
- On schema validation failure, the worker MUST attempt bounded repair by asking the model to re-emit a schema-valid final object (max 2 repair attempts). If repair fails, the step MUST return a deterministic schema-related `failure_signature` (prefix `worker-output-schema-invalid`).
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
  - Per step, Tokimon persists first-class artifacts under `<run_root>/artifacts/steps/<step_id>/`:
    - `step_result.json`: full structured step result (including any `ui_blocks`).
    - `outputs.json`: engine step outputs (may be a subset of the step result, used for workflow state).
    - `artifacts.json`: the `artifacts` list (mirrors `step_result.json.artifacts` for convenience).
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
- PatchTool: apply unified diffs with validation (and deterministically repairs mismatched hunk header line-counts when possible).
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
- Tool calls MUST be policy-audited:
  - Every tool call record MUST include a `policy_decision` object with at least:
    - `decision`: `"allow" | "deny" | "confirm"`
    - `risk_tier`: `"low" | "medium" | "high"`
    - `reason`: string (bounded)
    - `policy_id`: string (stable identifier, e.g., `"default-v1"`)
  - Phase 1 default policy may be allow-all, but policy decisions MUST still be recorded deterministically.
- Side-effect tool idempotency (Phase 1):
  - For side-effectful actions (`file.write`, `patch.apply`), repeated calls within a single worker step attempt with the same `(tool, action, args)` MUST be deduped and MUST NOT execute twice.
  - Deduped tool call records MUST be marked with `cached=true` and MUST return the same tool result payload as the first execution.
- A response is considered **final** when it includes `status` (SUCCESS|FAILURE|BLOCKED|PARTIAL).
- Tool results are fed back into the worker loop as structured records; workers report:
  - `metrics.model_calls`, `metrics.tool_calls`, `metrics.elapsed_ms`, and `metrics.iteration_count`
- Workers may request early workflow termination by setting:
  - `metrics.terminate_workflow: true` (and optional `metrics.terminate_reason`)
  - The runner marks remaining steps as `SKIPPED` and completes the workflow when safe to do so.
  - Termination signals are ignored for baseline evaluation steps (`step_id` starting with `baseline`) to avoid skipping planned work based solely on a clean baseline.

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

### Replay & Audit (OpenClaw-inspired, Phase 1)
- Tokimon MUST persist a per-step replay artifact that enables deterministic, offline replay with all tools mocked (no side effects).
- Replay artifact location (relative to `<run_root>`): `artifacts/steps/<step_id>/replay.json`
- Replay artifact schema (stable JSON with sorted keys):
  - `schema_version`: string (current: `"1.0"`)
  - `recorded_at`: string (ISO-8601 UTC)
  - `step_id`: string
  - `worker_role`: string
  - `goal`: string (may be truncated)
  - `inputs`: object (JSON-serializable; may be truncated)
  - `memory`: list of strings (may be truncated)
  - `model_script`: list of model responses (JSON objects) in call order (may be truncated, but MUST preserve `tool_calls` and final `status` payloads).
  - `tool_script`: list of invoked tool calls in call order:
    - `tool`, `action`, optional `call_id`
    - `args_hash`: sha256 of stable JSON of args
    - `args_preview`: truncated args object (for debugging only)
    - `result`: ToolResult-like object (`ok`, `summary`, `data`, `error`, `elapsed_ms`) with bounded payload sizes
  - `final_result`: object containing the final worker output fields (`status`, `summary`, `failure_signature`) plus a bounded subset of `metrics`.
- Redaction requirements (minimum): replay artifacts MUST redact bearer tokens of the form `Authorization: Bearer <token>` anywhere in recorded strings.
- Offline replay CLI:
  - `tokimon replay --run-path <run_root>` MUST replay every step replay artifact using the recorded `model_script` and `tool_script`, with tools mocked so no filesystem or network side effects occur.
  - Replay MUST validate that invoked tool calls match recorded `tool_script` entries by comparing `args_hash`.
  - Replay MUST exit `0` when all steps replay and the computed final outputs match `final_result`; otherwise exit non-zero and report the first mismatch deterministically.

### Operator & Safety (OpenClaw-inspired, Phase 2 - MVP)
- Tokimon MUST centralize risk classification for tool calls in a single “dangerous tools” registry.
  - The registry MUST map `(tool, action)` → metadata including:
    - `risk_tier`: `"low" | "medium" | "high"`
    - `requires_approval`: bool
    - `notes`: string (bounded)
  - Worker `policy_decision.risk_tier` MUST be derived from this registry when an entry exists (fallback rules are allowed for unknown tools/actions).
- Tokimon MUST harden dynamic code loading for skills (treat as a supply-chain surface):
  - `SkillRegistry` MUST refuse to import generated code skills whose `module` is outside the default safe prefixes:
    - `skills_builtin`
    - `skills_generated`
  - Optional allowlist: operators may extend allowed module prefixes via `TOKIMON_SKILL_MODULE_ALLOWLIST` (comma-separated prefixes).
  - Refused skills MUST be skipped (not imported) deterministically.
  - Loaded code skills MUST record module provenance (best-effort):
    - `module` (string), `module_file` (string), and `sha256` of the module file contents when readable.
- Tokimon MUST append an audit log entry when writing config-like skill assets:
  - Applies at minimum to writes of:
    - `src/skills_generated/manifest.json`
    - `src/skills_generated/*.py` (promoted code skills)
    - `src/skills_generated/prompts/*.md` and `*.validation.md` (promoted prompt skills)
  - Audit log format: append-only JSONL under `<workspace_root>/.tokimon-tmp/audit/config.jsonl` (workspace root is the repo root, sibling of `src/` and `docs/`)
  - Each entry MUST include: `ts` (ISO-8601 UTC), `path`, `action`, `sha256_before`, `sha256_after`, and bounded `reason`.
  - Audit MUST NOT log raw secrets; preserve `${ENV}` patterns (do not expand env vars into files as part of auditing).
- Tokimon MUST extend `tokimon doctor` with state-integrity checks (safe, non-destructive):
  - Detect missing or non-writable state directories (at minimum: `.tokimon-tmp/`, `runs/`, `memory/`, `src/skills_generated/`).
  - Detect invalid `src/skills_generated/manifest.json` shapes (non-dict, missing skills list) and surface actionable remediation.

### Operator & Safety (OpenClaw-inspired, Phase 3)
- Tokimon SHOULD provide opt-in tool-loop detection guardrails (disabled by default):
  - Goal: prevent runaway repeated tool-call loops that do not make progress.
  - Config surface (env):
    - `TOKIMON_TOOL_LOOP_DETECTION_ENABLED` (default: false)
    - `TOKIMON_TOOL_LOOP_HISTORY_SIZE` (default: 20)
    - `TOKIMON_TOOL_LOOP_REPEAT_THRESHOLD` (default: 3)
    - `TOKIMON_TOOL_LOOP_CRITICAL_THRESHOLD` (default: 6)
  - Detection (minimum):
    - Repeated identical tool call signature (same tool, action, args_hash) >= `repeat_threshold`
    - Repeated failures for the same tool call signature >= `repeat_threshold`
  - When detected, the worker MUST stop executing further tool calls for the step attempt and MUST return a deterministic final output:
    - `status=PARTIAL`
    - `failure_signature` prefix: `worker-tool-loop-detected`
    - Include bounded evidence in `metrics` (e.g., last N tool call signatures, counts, trigger reason).
- Tokimon SHOULD provide an opt-in approval gate for high-risk tool calls:
  - High-risk tool calls are identified by `policy_decision.requires_approval=true` (derived from the dangerous tools registry).
  - Config surface (env): `TOKIMON_TOOL_APPROVAL_MODE`:
    - `off` (default): do not block tool calls based on approval requirements.
    - `block`: when a tool call requires approval, do not execute it; return a deterministic `BLOCKED` final output for the step attempt.
    - `deny`: when a tool call requires approval, do not execute it; record a deterministic tool error and continue the loop.
  - When blocking, the worker MUST include a deterministic `approval_request` payload in `metrics`:
    - `approval_id` (stable hash), `tool`, `action`, `args_hash`, and bounded `args_preview`.
    - `reason` (bounded) describing why approval was required.
  - No-UI fallback: when approvals are not supported by the current runtime, `block` mode MUST fail closed by leaving the step `BLOCKED` with actionable remediation.

### Operator & Safety (OpenClaw-inspired, Phase 4)
- Tokimon SHOULD support approval allowlists so operators can pre-approve specific high-risk tool calls without requiring interactive approval.
  - Allowlist sources (checked in order; first match wins):
    - Environment variable: `TOKIMON_TOOL_APPROVAL_ALLOWLIST` (comma-separated list of `approval_id` hashes).
    - File: `.tokimon-tmp/approvals/allowlist.json` (JSON object with `{"allowlist": ["<approval_id>", ...]}`) relative to workspace root.
  - Matching is deterministic: a tool call is considered pre-approved when its computed `approval_id` (stable SHA-256 hash of `{tool, action, args_hash}`) appears in the merged allowlist.
  - Behavior with `TOKIMON_TOOL_APPROVAL_MODE`:
    - `off`: allowlist has no effect (all tool calls proceed without approval checks).
    - `block`: if the tool call's `approval_id` is in the allowlist, the call proceeds as if approved; otherwise the step is `BLOCKED` as before.
    - `deny`: if the tool call's `approval_id` is in the allowlist, the call proceeds as if approved; otherwise a deterministic tool error is recorded and the loop continues.
  - When a pre-approved call proceeds, the `policy_decision` MUST include `"pre_approved": true` and `"allowlist_source"` (either `"env"` or `"file"`) so the decision is auditable.
  - Allowlist loading MUST be deterministic and fail-safe: malformed JSON in the file or missing file is treated as an empty allowlist (no error raised).
- Tokimon SHOULD provide an operator CLI for managing approval allowlist files without manual JSON edits.
  - Command group: `tokimon approvals`
  - Subcommands:
    - `tokimon approvals list` prints the effective allowlist set (env + file) and indicates the source for each entry.
    - `tokimon approvals add <approval_id>` adds an approval id to the file allowlist (idempotent) and prints the updated allowlist.
    - `tokimon approvals remove <approval_id>` removes an approval id from the file allowlist (no-op when missing) and prints the updated allowlist.
    - `tokimon approvals clear` clears the file allowlist and prints the updated allowlist.
  - Output:
    - Default: human-readable.
    - `--json` emits stable machine-readable JSON with sorted keys and deterministic ordering.
  - Persistence:
    - The CLI MUST manage the file allowlist at `.tokimon-tmp/approvals/allowlist.json` relative to the Tokimon workspace root.
    - When adding/removing, the CLI MUST create parent directories as needed.

### Observability: Metrics & Dashboards
- Tokimon MUST persist canonical run/step metrics and a self-contained dashboard artifact for every BaselineRunner, HierarchicalRunner, and Chat UI run.
- Persistence locations (relative to `<run_root>`):
  - Canonical metrics: `reports/metrics.json`
  - Dashboard artifact: `reports/dashboard.html` (self-contained HTML; no external CDN)
- `metrics.json` schema (stable JSON with sorted keys):
  - `schema_version`: string (current: `"1.0"`)
  - `run`: run-level summary metrics (object)
  - `steps`: list of step result metric objects (one per step attempt when available)
- Canonical step metrics (per element of `steps`) with types/units:
  - `step_id`: string
  - `attempt_id`: integer (1-based; 0 allowed when a step has no attempts)
  - `status`: string (`"SUCCESS" | "FAILURE" | "BLOCKED" | "PARTIAL"` when available)
  - `elapsed_ms`: number (milliseconds; may be null when unavailable)
  - `model_calls`: integer (count; may be null when unavailable)
  - `tool_calls`: integer (count; may be null when unavailable)
  - `energy`: integer (count; `model_calls + tool_calls` when both available)
  - `iteration_count`: integer (count; may be null when unavailable)
  - `schema_repairs`: integer (count; may be null when unavailable)
  - `artifact_count`: integer (count; may be null when unavailable)
  - `touched_files_count`: integer (count; may be null when unavailable)
  - `tool_errors`: integer (count of non-ok tool calls; may be null when unavailable)
  - `failure_signature`: string
- Canonical run summary metrics (under `run`) with types/units:
  - `run_id`: string
  - `runner`: string (`"baseline" | "hierarchical" | "chat-ui"`)
  - `wall_time_s`: number (seconds; may be null when unavailable)
  - `model_calls`: integer (count; sum of step metrics when available)
  - `tool_calls`: integer (count; sum of step metrics when available)
  - `energy`: integer (count; `model_calls + tool_calls` when available)
  - `steps_total`: integer (count)
  - `steps_by_status`: object mapping `status -> count` (counts)
  - `tests_passed`: integer (count; best observed when tests are run; null otherwise)
  - `tests_failed`: integer (count; best observed when tests are run; null otherwise)
- Determinism requirements:
  - Metric normalization MUST be deterministic (stable key names, stable coercion rules, stable ordering).
  - `dashboard.html` MUST be generated deterministically from the persisted `metrics.json` and MUST NOT require network access (no external JS/CSS/CDN).

### Model Integration
- Abstract `LLMClient.send(messages, tools=None, response_schema=None)`.
- Provide a stub adapter and a deterministic scripted adapter for tests/offline replay.
  - The scripted adapter MUST NOT synthesize placeholder content; if the script is exhausted it MUST return a deterministic `FAILURE` with actionable remediation.
- Primary adapter: Codex CLI-backed client that shells out to `codex exec` and returns structured JSON (controlled via `TOKIMON_LLM=codex` or CLI flags).
  - Codex model selection MUST NOT rely on the user's global Codex config. Tokimon MUST pass an explicit `--model` on every Codex invocation.
  - Default Codex model for general tasks is `gpt-5.4` unless overridden (env: `TOKIMON_CODEX_MODEL`, or Chat UI request field: `model`).
  - If Codex CLI rejects a request-selected model as unsupported for the current auth mode, Tokimon MUST retry that call once with the built-in default model `gpt-5.4` before surfacing a failure.
- Optional adapter: Claude Code CLI-backed client that shells out to `claude` and returns structured JSON (controlled via `TOKIMON_LLM=claude` or CLI flags).
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
- Commands: auto, run-task, run-suite, resume-run, inspect-run, list-skills, build-skill, self-improve, chat-ui, gateway, memory, sessions, status, doctor, health, logs, approvals.
- Prompt-driven entrypoint: `tokimon auto "<prompt>"` routes to the appropriate mode by asking an AI router (Codex/Claude) to return a concrete Tokimon argv list.
  - Output contract: the router returns JSON containing `argv: string[]` (argv excludes the leading `tokimon`).
  - Validation: Tokimon MUST validate the router argv against the CLI parser (unknown commands/options are rejected) and MUST prevent `auto` recursion.
  - Fallback: if the router fails (missing CLI, timeout, invalid JSON, invalid argv), Tokimon falls back to deterministic heuristic routing; prompts that ask to learn/improve route to `tokimon self-improve`.
- Default `--help` output minimizes option surface by hiding advanced flags while still accepting them for power users.
- Help ergonomics (operator-facing):
  - `tokimon --help` MUST include one-line summaries for each subcommand (so the command list is self-describing).
  - When a group command is invoked without a required subcommand (e.g., `tokimon memory`), Tokimon MUST print a short usage + examples section and exit `0`.
  - Common operator errors MUST be handled without stack traces and with actionable remediation:
    - Port already in use when starting `tokimon gateway` or `tokimon chat-ui`.
    - Gateway connectivity failures for `tokimon health` / `tokimon logs` (unreachable port, wrong service speaking HTTP instead of WS).
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

- Status (OpenClaw-inspired, Phase 1): `tokimon status` prints a concise diagnostic overview of Tokimon.
  - Includes at least: doctor readiness, gateway WebSocket health, memory status, and sessions summary.
  - Flags:
    - `--json` emits stable machine-readable JSON with sections `{ok, doctor, gateway, memory, sessions, usage?}`.
    - `--all` includes more detail (full doctor report, deep memory reconciliation, full sessions list).
    - `--deep` may run live probes (doctor, gateway health, memory index reconciliation) but must not require external services.
    - `--usage` emits process/runtime resource snapshots when feasible.
    - `--url` and `--timeout-ms` configure the gateway WebSocket probe (mirrors `tokimon health`).

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

- Logs (OpenClaw-inspired, Phase 1): `tokimon logs` tails Gateway logs over the WebSocket RPC (no SSH).
  - Flags:
    - `--url` (default: `ws://127.0.0.1:8765/gateway`)
    - `--follow` keeps tailing until interrupted
    - `--json` emits machine-readable JSON
    - `--limit N` limits the number of entries returned
    - `--local-time` renders timestamps in your local timezone
  - Default behavior is a one-shot tail; with `--follow`, it polls until interrupted.

### Chat UI
- `tokimon chat-ui` starts a local web server (binds loopback by default) that serves a single-page chat UI.
- The chat UI frontend is a React + TypeScript app built with Vite under `ui/`.
- The primary chat composer is keyboard-first: the message textarea auto-focuses on initial load and after each completed send, `Enter` submits the current message, and `Shift+Enter` inserts a newline.
- `GET /` serves `ui/dist/index.html` and static assets when present; otherwise it serves an HTML page explaining that the UI build is missing.
- Health endpoint: `GET /healthz` returns JSON indicating the server is running.
- Chat endpoint: `POST /api/send` accepts JSON `{message: string, history?: [{role, content}], model?: string}` and returns a structured JSON reply including the step result fields (`status`, `summary`, `artifacts`, `metrics`, `next_actions`, `failure_signature`) plus optional `ui_blocks`, and a human-readable assistant message (in `reply`).
  - When `model` is provided and the active LLM provider is a CLI-backed provider (Codex/Claude), Tokimon MUST use it for that request.
  - For Codex-backed requests, if the requested model is rejected as unsupported for the current auth mode, Tokimon MUST retry once with `gpt-5.4` and still return a normal structured reply when the fallback succeeds.
- The frontend renders `ui_blocks` using `@tambo-ai/react` (`TamboRegistryProvider` + `ComponentRenderer`) with a local registry (no Tambo cloud / no API key).
- Chat UI persists each `/api/send` result under `<workspace_dir>/runs/chat-ui/run-<run_id>/artifacts/steps/chat-<N>/step_result.json`.
- The chat handler uses the same tool set as the hierarchical runner (file, grep, patch, pytest, web).
- Default LLM provider for `chat-ui` and `gateway` is `codex`; `--llm claude` (or `TOKIMON_LLM=claude`) selects the Claude CLI-backed client instead.
- When `chat-ui` or `gateway` runs with the Codex provider, the interactive Codex defaults MUST be writable: if `TOKIMON_CODEX_SANDBOX` is unset, use `workspace-write`; if `TOKIMON_CODEX_APPROVAL` is unset, use `never`. Explicit environment values still override these defaults.

### Gateway Server (OpenClaw-Inspired, Phase 1/2/3)
- `tokimon gateway` starts a local server that supports:
  - Existing Chat UI HTTP endpoints: `GET /healthz`, `POST /api/send`
  - A WebSocket control-plane endpoint at `GET /gateway` (WS upgrade)
- Bind safety:
  - Gateway MUST refuse non-loopback binds (e.g. `0.0.0.0`) unless `--dangerously-expose` is set AND `TOKIMON_GATEWAY_AUTH_TOKEN` is configured.
- `tokimon gateway run` is an explicit alias for starting the Gateway server.
- `tokimon gateway health` is an alias of `tokimon health` (same flags).
- `tokimon gateway call <method>` performs a single WebSocket RPC call and prints the response.
  - Flags: `--url`, `--timeout-ms`, `--json`, `--params <json>`
- `tokimon gateway probe` connects, performs handshake, runs a health call, and exits non-zero on failure.
  - Flags: `--url`, `--timeout-ms`, `--json`
- The WebSocket endpoint uses a minimal OpenClaw-inspired framing:
  - Request: `{type:"req", id, method, params}`
  - Response: `{type:"res", id, ok, payload|error}`
  - Event: `{type:"event", event, payload}`
- Handshake:
  - On socket open, the server emits `connect.challenge`.
  - The first client request MUST be `connect` and MUST pass protocol negotiation:
    - The server supports protocol versions `1..3`.
    - The server selects the highest common supported version within `[connect.params.minProtocol, connect.params.maxProtocol]`.
    - `hello-ok.payload.protocol` MUST be the selected version.
    - On mismatch, the server responds with `ok=false` and closes the socket.
  - The client MUST echo the `connect.challenge` nonce in `connect.params.challenge.nonce`; mismatches MUST be rejected.
  - When `TOKIMON_GATEWAY_AUTH_TOKEN` is configured, the server MUST require `connect.params.auth` to be either `{mode:"token", credential:"..."}` or `{token:"..."}` and verify via constant-time compare.
  - Device identity + challenge signing (protocol v3):
    - For negotiated protocol versions `>= 3`, the server MUST require `connect.params.device` unless `TOKIMON_GATEWAY_DANGEROUSLY_DISABLE_DEVICE_AUTH=1` is set.
    - Required device fields: `id`, `publicKey`, `signature`, `signedAt`, `nonce`.
    - Validation MUST be deterministic and OpenClaw-compatible:
      - `device.id == sha256(device.publicKey_raw_bytes).hexdigest()`.
      - `device.signedAt` within ±2 minutes of server time.
      - `device.nonce` non-blank and equals the `connect.challenge` nonce.
      - `device.signature` verifies Ed25519 base64url over the OpenClaw payload string (try v3 then v2).
        - Token field uses `connect.params.auth.token` if present, else `connect.params.auth.credential` when `auth.mode == "token"`, else `""`.
    - On failure, the server MUST respond `ok=false` with OpenClaw-compatible `error.details.code`/`reason` (DEVICE_* codes) and close the socket.
  - The server MUST accept optional `connect.params` fields: `caps`, `commands`, `permissions`, `locale`, `userAgent`, `device` (type-check deterministically; protocol v3 enforces device auth as above).
- Methods:
  - `health`: returns `{ok:true}`
  - `methods.list`: returns the server-supported methods (excluding `connect`) in deterministic order, gated by negotiated protocol (protocol v1 keeps the Phase 1 list; protocol v3 adds `system-presence`).
  - `system-presence` (protocol v3 only): returns a deterministic snapshot (stable ordering) of active WS connections including device id, role, scopes, and client metadata.
  - `tools.catalog`: returns a deterministic tool/action risk catalog derived from `src/policy/dangerous_tools.py`.
  - `send`: invokes the same logic as `/api/send` and requires an idempotency key.
  - `logs.tail`: returns recent log entries from an in-memory ring buffer.
- The Gateway protocol surface and Phase 3 requirements are documented in `docs/gateway.md`.
- Acceptance tests:
  - `src/tests/test_gateway_ws.py::test_gateway_ws_handshake_health_and_send` asserts a v1 connect flow with `hello-ok.payload.protocol == 1` and `methods.list` returning the Phase 1 list.
  - `src/tests/test_gateway_ws.py::test_gateway_ws_protocol_v3_methods_and_presence` asserts protocol negotiation selects v3, `methods.list` includes `system-presence`, and `system-presence` includes the current connection.
  - `src/tests/test_gateway_ws.py::test_gateway_ws_protocol_v3_device_auth_success` asserts a protocol v3 connect succeeds with a valid device identity + signature.
  - `src/tests/test_gateway_ws.py::test_gateway_ws_protocol_v3_device_auth_failure_codes` asserts each required device-auth failure mode returns the expected OpenClaw-compatible `error.details.code`/`reason`.

### Self-Improvement Mode (Multi-Session / Batch)
- When invoked with a self-improvement goal, the system can accept optional “inputs”:
  - URL (http/https), local file path, or inline text (or none).
- If `--input` is not provided, the system may auto-detect URL(s) embedded in the `--goal` text and fetch at least the first URL as the session input payload (bounded by byte/time limits and the WebTool network policy).
- When the self-improvement goal is to update Tokimon itself, the default execution strategy MUST be a batch of N independent improvement sessions in parallel so multiple worktree-backed candidates can be explored at once.
- Self-improve CLI LLM default: `--llm` defaults to `$TOKIMON_LLM` when set; otherwise it defaults to `mixed`.
- Self-improve provider timeout policy:
  - When `TOKIMON_CODEX_TIMEOUT_S` / `TOKIMON_CLAUDE_TIMEOUT_S` are unset, `tokimon self-improve` MUST inherit the provider client's default timeout behavior.
  - `tokimon self-improve` MUST NOT silently shorten provider timeouts with a self-improve-specific override.
- Mixed-provider mode: when `--llm mixed`, enforce a deterministic `claude:codex=1:4` session mix by assigning Claude to session indices 1, 6, 11, ... (i.e., `(index - 1) % 5 == 0`) and Codex to the other sessions. `--sessions` MUST be a multiple of 5 (default: 5).
- Before launching each batch, the system runs an evaluation on the current canonical `main` checkout (the master workspace for the run) and passes a compact summary (pass/fail counts + failing test ids) into every session as context.
- Each session:
  - Materializes the canonical `main` checkout into an isolated session workspace using `git worktree` (detached HEAD) so sessions can run in parallel without colliding on files.
  - Self-improve requires the canonical `main` checkout to be a clean git checkout (no local changes) so worktrees and merges are deterministic; otherwise it aborts with an actionable error.
  - Runs the hierarchical agent system within that workspace to attempt improvements.
  - Runs the configured evaluation command after each workflow step when `pytest_args` are provided, so retry/progress gating has objective signals.
  - Evaluates the result (pytest by default; optionally benchmark suite).
  - Produces a session report, metrics, and a diff/changed-file set versus `main`.
  - After each batch:
    - A comparer selects the best verified winner by deterministic criteria (tests passing is primary).
    - The selected winner MUST exist as a commit before merge, either because the session already committed it or because the merger materializes a temporary candidate commit from the verified diff.
    - A merger applies the committed winner back onto the canonical `main` checkout (restricted to configured paths; defaults should include `src/` and `docs/`) using a conflict-aware git integration:
      - Apply via `git merge --squash` onto `main`.
      - Re-evaluate `main`; on success, commit the squashed changes.
      - Use an OS-level lock so multiple self-improve runs perform safe queued merges into the same checkout.
      - On merge conflicts, automatically resolve (prefer winner changes) and continue; on failing evaluation, abort and leave `main` unchanged.
    - Once required reports and session artifacts are persisted, the system MUST delete every session worktree it created for the batch; cleanup failures MUST be reported explicitly.
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
  - The `main` evaluation summary for the batch (pytest counts + failing test ids).
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

#### Nested Agent Worktree Rule (Self-Improve, Session-Local)
- This rule applies inside each self-improve session workspace. It does not replace the outer Tokimon batch merge back onto the canonical `main` checkout, which still follows the winner-merge contract above.
- When the active self-improve runtime can execute git/shell commands inside the session workspace, the self-improve entry-point prompt MUST instruct the agent to isolate its own edits in a nested git worktree rooted at `temp/codex-worktrees/`.
- The prompt MUST tell the agent to treat the default session checkout as read-only, reuse the most recent `Worktree: <absolute-path>` from the same thread when present, and report `Worktree: <absolute-path>` whenever it makes changes.
- When a session creates multiple nested agent threads, each thread MUST use its own nested worktree so candidate solutions stay isolated.
- In self-improve mode, the agent MUST manage the nested worktree lifecycle without waiting for extra human instructions: create before editing, compare multiple candidates deterministically when more than one nested worktree exists, commit the selected improvement it intends to keep, merge committed verified work back into the session checkout, then delete every nested worktree it created once winners and losers are safely merged or discarded.
- Allowed mutating git commands are limited to the `Create`, `Commit`, `Merge`, and `Delete` sequences below. Here, `<main-checkout>` means the current self-improve session workspace root.
- If the runtime cannot execute git/shell commands, it MUST say so plainly instead of claiming the worktree lifecycle happened.

Create:
1) `mkdir -p temp/codex-worktrees`
2) `WT_DIR="$(mktemp -d temp/codex-worktrees/wt-XXXXXX)"`
3) `BRANCH="ai/$(basename "$WT_DIR")"`
4) `git worktree add -b "$BRANCH" "$WT_DIR" HEAD`

Commit:
1) When an improvement is selected to keep, verify the change is complete enough to preserve.
2) `cd <worktree-path>`
3) `git add -A`
4) `git -c user.email=tokimon@local -c user.name=Tokimon commit -m "<scoped summary of the selected improvement>"`

Merge:
1) Verify the work is committed and there is something to merge; otherwise warn and stop.
2) `cd <worktree-path>`
3) `git rebase <base-branch>`
4) If conflicts happen, resolve them with `git add <resolved-files>` and `git rebase --continue`; use `git rebase --abort` if the merge cannot be completed safely.
5) `cd <main-checkout>`
6) `git merge --ff-only <worktree-branch>`

Delete:
1) Verify there is nothing to merge and no uncommitted code change; otherwise refuse and warn.
2) `git worktree remove <worktree-path>`
3) `git branch -d <worktree-branch>`
4) If the user asks to `delete` again after that refusal:
5) `git worktree remove --force <worktree-path>`
6) `git branch -D <worktree-branch>`

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
