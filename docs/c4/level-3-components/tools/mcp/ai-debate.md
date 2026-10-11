---
title: "AI Debate MCP - Component"
summary: "L3 component contract for AI Debate MCP commands, run state, retry behavior, and transcripts."
read_when:
  - Changing AI Debate MCP tools, provider adapters, retry state, transcript format, or command surface.
  - Reviewing AI Debate MCP behavior and compatibility contracts.
---

# AI Debate MCP - Component

## Contents

- [Overview](#overview)
- [Components](#components)
- [Interfaces/Contracts](#interfacescontracts)
- [Data Ownership](#data-ownership)
- [Failure Modes](#failure-modes)
- [Acceptance Checks](#acceptance-checks)
- [References](#references)

## Overview

AI Debate MCP is Tokimon's adversarial multi-round debate and final-judgment tool.
At L3, this document owns the component contracts and state transitions that implement that protocol.
Its primary outcome is a resumable, inspectable judged debate run that can return structured results without hiding provider failures.

## Components

- **MCP server:** Owns newline-delimited stdio JSON-RPC handling, supported MCP methods, exposed tool names, and MCP text-content response shape.
- **Direct CLI wrapper:** Owns health, run, resume, status, and log-read command behavior for local operator use and tests.
- **Provider access runner:** Owns local Codex/Claude CLI access, prompt delivery, schema enforcement when available, stdout/stderr capture, and final JSON extraction.
- **Debate state manager:** Owns `state.json`, retry metadata, transcript updates, run resumption, and final-result persistence.
- **Artifact writer:** Owns run artifacts under the default `.tokimon/` runtime root or a caller-selected allowed `output_root`.

## Interfaces/Contracts

Canonical machine-readable tool contracts are the MCP `tools/list` response produced by `.venv/bin/python -m tokimon.tools.mcp.ai_debate serve` and the health payload from `.venv/bin/python -m tokimon.tools.mcp.ai_debate health --json`.

### Command Surface

```bash
.venv/bin/python -m tokimon.tools.mcp.ai_debate serve
.venv/bin/python -m tokimon.tools.mcp.ai_debate health --json
.venv/bin/python -m tokimon.tools.mcp.ai_debate run --topic TEXT --side-a TEXT --side-b TEXT
.venv/bin/python -m tokimon.tools.mcp.ai_debate resume RUN_DIR
.venv/bin/python -m tokimon.tools.mcp.ai_debate status RUN_DIR --json
.venv/bin/python -m tokimon.tools.mcp.ai_debate turn-start RUN_DIR --json
.venv/bin/python -m tokimon.tools.mcp.ai_debate turn-status RUN_DIR --json
.venv/bin/python -m tokimon.tools.mcp.ai_debate turn-finalize RUN_DIR --json
.venv/bin/python -m tokimon.tools.mcp.ai_debate read-log RUN_DIR
```

The `serve` command reads newline-delimited JSON-RPC requests from stdin and writes newline-delimited JSON-RPC responses to stdout.

### MCP Protocol Methods

The server supports the minimum MCP stdio methods needed by Codex-compatible clients:

- `initialize`
- `notifications/initialized`
- `ping`
- `tools/list`
- `tools/call`

`initialize` returns the service's declared protocol version, including for unsupported, missing, or malformed client proposals.
Missing or non-string methods return `-32600`.
Messages with a string method and no `id` are notifications: they produce no reply and cannot invoke request-only methods or tools.
Explicit request ids, including `0`, an empty string, and `null`, keep their response correlation.
Wire responses escape unpaired Unicode surrogates as JSON escapes and preserve ordinary Unicode text.
Tool text and error messages also escape malformed characters before wrapping, so decoded display text is valid UTF-8.
Strict UTF-8 output remains valid for ids, errors, and tool content, and the next request still receives its response.
Malformed JSON, interpreter integer limits, and decoder recursion errors return `-32700`, log a correlated parse failure, and leave stdio ready for the next line.

Tool names exposed through `tools/list`:

- `ai_debate_health`
- `ai_debate_run`
- `ai_debate_start`
- `ai_debate_resume`
- `ai_debate_resume_start`
- `ai_debate_turn_start`
- `ai_debate_turn_status`
- `ai_debate_turn_finalize`
- `ai_debate_status`
- `ai_debate_read_log`

Tool responses are returned as MCP text content blocks containing formatted JSON.
Retryable provider exhaustion is returned as a normal tool result with `status=waiting_retry`; callers should inspect `retry.retry_after_at` and call `ai_debate_resume` later with the same `run_dir`.
Non-retryable failures during a created run are returned as MCP tool errors, but the JSON payload includes top-level `run_dir` plus the run summary so callers do not lose the created run.
Because `ai_debate_run` and `ai_debate_resume` execute local provider CLI turns synchronously, a platform or connector `tools/call` timeout can still prevent the caller from receiving that response.
Timeout-sensitive workflows should call `ai_debate_start` first, persist the returned `run_dir`, then call `ai_debate_resume` from a local agent context.
The async turn lifecycle tools remain listed for compatibility but fail fast because AI Debate is local-only and has no durable broker job handle.

### Tool Input Contracts

#### `ai_debate_run`, `ai_debate_start`

Required:

- `topic`: non-empty string
- `side_a`: non-empty string
- `side_b`: non-empty string

Optional:

- `side_b_provider`: `claude` or `codex`; required from settings unless supplied for the invocation
- `access`: `local-cli`; required from settings unless supplied for the invocation
- `max_rounds`: positive integer, default `24`
- `model`, `effort`: Codex model and effort shared by the Codex roles, read following `docs/guidelines/configuration-guidelines.md`
- `agent_a_model`, `agent_b_model`, `judge_model`: optional role-specific model overrides
- `agent_b_effort`: optional Agent B effort override
- `turn_idle_timeout_sec`: positive integer; defaults to `agent_execution.idle_timeout_sec`
- `turn_timeout_sec`: compatibility alias for `turn_idle_timeout_sec`
- `output_root`: path within an allowed Tokimon repository or runtime root; default `.tokimon/tools/mcp/ai-debate/runs`

`ai_debate_run` creates a fresh run and executes synchronously until completion, waiting retry, or failure.
`ai_debate_start` creates the same fresh run artifacts and returns the run summary immediately without executing provider turns.
The returned `run_dir` is immediately valid for `ai_debate_status` and `ai_debate_resume`.

#### `ai_debate_resume`, `ai_debate_resume_start`, `ai_debate_status`, `ai_debate_read_log`

Required:

- `run_dir`: absolute path or repository-relative path to a debate run directory.

Optional for `ai_debate_read_log`:

- `max_chars`: positive integer limit for returned transcript text.

Retry metadata is not caller-supplied.
`ai_debate_run` and `ai_debate_resume` classify quota, rate-limit, and token-exhaustion provider failures automatically and return without sleeping.
`ai_debate_resume_start` validates an existing `run_dir` and returns the current summary without executing provider turns.
It is a cheap handle/status handshake for callers that already have a run handle; it does not discover handles lost before a previous response.
Valid historical `mcp.remote-agent-broker` runs remain available for status, resume-start, and transcript reads.
Completed runs may also return their summary through resume, but incomplete remote runs cannot execute under the local-only runtime.

#### `ai_debate_turn_start`, `ai_debate_turn_status`, `ai_debate_turn_finalize`

Required:

- `run_dir`: absolute path or repository-relative path to a debate run directory.

The async lifecycle is unsupported for local-only AI Debate.
Lifecycle calls fail fast because local CLI is a synchronous placement with no durable active job handle.

### Settings and Precedence

AI Debate reads required machine-local settings from `.tokimon/settings.local.json`.
The canonical shared shape and recommended values live in `docs/guidelines/configuration-guidelines.md` and `tokimon/settings.example.json`.
AI Debate requires `ai_debate` placement and shared agent-execution deadlines before provider execution.
Documented invocation arguments may override individual values for one run; settings are otherwise required.
Resolved access, provider, and supplied model/effort are recorded in `run_config.json`.
Omitted model/effort are recorded as `default`; neither provider command receives that marker as a flag value.
Agent B inherits the shared options only when its provider is Codex.
A Claude Agent B uses its explicit role overrides or the Claude CLI defaults for omitted values.
Run creation, execution, and artifact reads require no `agents` settings section.

### Provider Access

Execution supports only `access=local-cli`; historical `mcp.remote-agent-broker` artifacts are compatibility inputs, not a remote execution path.
Provider executables resolve in this order:

1. `CODEX_CLI` or `CLAUDE_CODE_CLI` environment variable when set.
2. `codex` or `claude` from `PATH`.

Health reports executable discovery and version capture only.
It marks provider authentication as not checked; an installed executable does not prove that the account has an active subscription or can complete a model call.

Provider adapters run debate turns without repository-write or external-tool authority, deliver prompts without relying on command-line argument size, and request structured results.
Exact provider flags and schema-delivery mechanics remain adapter-owned so they can follow supported CLI versions without changing the debate contract.

Each task stores its prompt at `prompts/<task-id>.md` and its expected schema, stdout, stderr, and normalized result under `results/`.
Debater result schemas require `citations` and `files_created` as arrays for provider structured-output compatibility; callers should treat empty arrays as omitted optional metadata.
Observable output refreshes the idle deadline; idle and optional wall deadlines bound execution and final output capture.
A result is parsed only after complete capture, and timeout or capture failure terminates remaining provider work and records a visible provider failure.

### Debate State Machine

`state.json` stores:

- `status`: `in_progress`, `waiting_retry`, `failed`, or `completed`
- `current_round`
- `next_step`: `agent_a`, `agent_b`, `final_judgment`, or `completed`
- completed round records
- final result when available
- last error when a provider or validation failure stops the run
- `retry` when status is `waiting_retry`
- `last_failure` with machine-readable provider failure metadata
- historical remote-broker lifecycle metadata in `active_task` and `task_attempts`; local runs use `null` and `{}`

Resume restarts at `next_step` and reuses existing transcript and run config.
It refuses execution when a historical `active_task` remains or a historical remote run is incomplete; inspection remains available.
If a provider fails before returning a valid turn payload, no turn is appended, `current_round` is not advanced, and `next_step` remains the failed step.
A retryable Agent B failure therefore resumes by retrying the same Agent B turn in the same debate.
Persisted `state.json` is authoritative for completed turns and the final result.
Before resumed local execution invokes another provider, the transcript is reconciled from committed state so interrupted or repeated resumes cannot duplicate or override a committed turn.
Historical lifecycle metadata must describe a task already reached by persisted state and is rejected on local runs.

Retry metadata fields:

- `retryable`: boolean
- `reason`: `rate_limit`, `quota_exhausted`, `token_exhausted`, or `provider_failure`
- `provider`: `codex` or `claude`
- `task_id`: provider task id such as `round-01-agent-b`
- `role`: `agent_a`, `agent_b`, or `judge`
- `next_step`: persisted step to retry
- `message`: short provider failure detail
- `retry_after_sec`: integer delay for retryable failures
- `retry_after_at`: UTC ISO timestamp for retryable failures
- `detected_at`: UTC ISO timestamp
- optional `stdout_path`, `stderr_path`, and `cli_exit_code`

The default retry delay is one hour when provider output does not contain a parseable retry-after hint.
Persisted retry state is valid only when `retry_after_at` equals `detected_at + retry_after_sec`.
The MCP call returns immediately with `status=waiting_retry`; waiting is the responsibility of the batch runner or caller.
For workflows that must survive MCP connector call deadlines, the caller should use `ai_debate_start` before long execution and then resume from a local agent context.

## Data Ownership

- Owns local debate run artifacts under `.tokimon/tools/mcp/ai-debate/runs` by default.
- Owns prompt files, per-task JSON Schemas, provider stdout/stderr captures, debate state, retry metadata, transcript logs, and final result artifacts for each run.
- Consumes read-only `.tokimon/settings.local.json`, provider CLI executables, and provider output as external inputs.
- Derives structured debate state, retry state, transcript text, and final result JSON from provider turns.
- The default path is local and git-ignored through `.tokimon/`; callers choosing another accepted [`output_root`](#tool-input-contracts) own its ignore policy.
- Reads and writes require canonical artifact paths contained by the accepted output root and run directory; missing, malformed, symlinked, or escaping paths are rejected.
- Run loading cross-validates lifecycle, provider, artifact, and result metadata before reuse.
- Compatibility permits older failure results without `model` and historical remote-broker results without newer schemas or with legacy citation objects.

## Failure Modes

AI Debate recovery preserves the run directory and retry metadata first, then either resumes the same persisted step, fixes provider settings, or starts a new run when the current state cannot be trusted.

- Provider executable, settings, access, or Peer B configuration is missing or invalid.
- Stalled or incomplete provider execution is terminated; invalid structured output is recorded as a provider failure with task and stream metadata and preserves the current step for recovery.
- A platform or MCP connector times out while a synchronous `ai_debate_run` or `ai_debate_resume` call is still executing; timeout-sensitive callers should start the run first and resume locally.
- Async lifecycle calls and resume of active or incomplete historical remote runs fail fast because local CLI has no durable broker job handle; historical runs remain inspectable.
- Provider quota, rate-limit, token-exhaustion, or transient failures stop the current turn and preserve retry metadata when retryable.
- A provider fails before returning a valid turn payload; no turn is appended, the round does not advance, and resume retries the same step.
- The run directory or an artifact is missing, malformed, inconsistent, outside its allowed boundary, or uses an unsafe link or file type.
- MCP requests are malformed or call unknown tools.

## Acceptance Checks

Focused tests should cover:

- command and MCP surfaces, settings precedence, and registry readiness
- protocol negotiation, silent notifications, explicit request ids, and stdio recovery through `tests/tools/mcp/test_protocol.py`
- strict UTF-8 byte-stream replies for surrogate ids, error messages, and tool content through `tests/tools/mcp/test_wire_encoding.py`
- debate outcomes, retry timing, resumability, transcript consistency, and idempotent state transitions
- provider isolation, deadline handling, output validation, and visible failure evidence
- artifact boundaries and historical compatibility without restoring remote execution

Verification:

```bash
.venv/bin/python -m unittest discover -s tests/tools/mcp/ai-debate -t . -p 'test_*.py'
.venv/bin/python -m unittest discover -s tests -t . -p 'test_*.py'
git diff --check
```

Expected outcome: focused tests pass for MCP tools, state-machine behavior, retry handling, parser coverage, and registry availability; broader tests and `git diff --check` complete without failures.

Ask an AI coding agent to verify the component contract:

```text
Review docs/c4/level-3-components/tools/mcp/ai-debate.md, tokimon/tools/mcp/ai_debate/, tests/tools/mcp/ai-debate/, and tests/tools/registry/.
Confirm that the documented MCP methods, tool inputs, provider invocation, retry state, artifact ownership, failure modes, and tests match the implementation.
Report any drift with file and line evidence.
```

Expected outcome: the agent reports no contract drift, or lists precise mismatches with recommended owner documents or implementation files.

## References

- L1 Context: `docs/c4/level-1-context/tools-context.md`
- L2 Container: `docs/c4/level-2-containers/tools/mcp/ai-debate.md`
- Configuration Guidelines: `docs/guidelines/configuration-guidelines.md`
