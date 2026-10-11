---
title: "Remote Agent Broker MCP - Component"
summary: "Component contract for the MCP server that manages remote shell and AI CLI jobs."
read_when:
  - Changing Remote Agent Broker transport, lifecycle, runner, or artifact behavior.
  - Adding or testing Remote Agent Broker MCP tools.
  - Reviewing job state, runner adapter, artifact, or CLI command contracts.
---

# Remote Agent Broker MCP - Component

## Contents

- [Overview](#overview)
- [Components](#components)
- [Interfaces/Contracts](#interfacescontracts)
- [Data Ownership](#data-ownership)
- [Trust and Transport](#trust-and-transport)
- [Limits](#limits)
- [Failure Modes](#failure-modes)
- [Acceptance Checks](#acceptance-checks)
- [References](#references)

## Overview

Remote Agent Broker MCP is Tokimon's trusted remote execution bridge for shell jobs and AI CLI jobs.
At L3, this document owns the MCP transport contract, runner lifecycle, runner adapter boundary, artifact contract, cleanup behavior, trust boundary, failure behavior, and acceptance checks for the `mcp.remote-agent-broker` capability.
Its primary outcome is an auditable remote job lifecycle that local agents can start, observe, interact with, cancel, and inspect through one stable MCP interface.

The key invariant is:

```text
one job lifecycle -> many runner adapters
```

The local coordinating agent should learn one lifecycle and pass runner-specific configuration only through `runner_options`.

## Components

- **MCP transports:** Own stdio and `/mcp` HTTP transport, session handling, JSON-RPC method support, and request-correlation context.
- **Tool dispatcher:** Owns the canonical MCP tool declarations, tool-input validation, and dispatch into the broker lifecycle API.
- **Job lifecycle manager:** Owns process tracking, cancellation, idle-timeout handling, and recovery of completed, failed, cancelled, lost, and expired jobs.
- **Job state store:** Owns atomic per-job state serialization and monotonic status transitions.
- **Runner adapters:** Own runner-specific executable discovery, prompt delivery, CLI flags, output parsing, and interactive input for shell, Copilot, and Claude runners.
- **Artifact manager:** Owns job directories, transcripts, logs, diffs, final payloads, artifact inventory, and safe concurrent access under the local runtime root.
- **Cleanup manager:** Owns per-job TTL cleanup without traversing outside the runtime root or deleting artifacts in active use.

## Interfaces/Contracts

Canonical machine-readable tool contracts are the MCP `tools/list` response produced by `.venv/bin/python -m tokimon.tools.mcp.remote_agent_broker serve` and the health payload from `.venv/bin/python -m tokimon.tools.mcp.remote_agent_broker health --json`.

### Command Surface

```bash
.venv/bin/python -m tokimon.tools.mcp.remote_agent_broker serve
.venv/bin/python -m tokimon.tools.mcp.remote_agent_broker serve-http --host 127.0.0.1 --port 8938
.venv/bin/python -m tokimon.tools.mcp.remote_agent_broker health --json
.venv/bin/python -m tokimon.tools.mcp.remote_agent_broker cleanup --ttl-days 7
```

`serve` starts a stdio MCP endpoint for target-platform managed MCP startup.
Stdout is reserved for line-delimited JSON-RPC responses; runtime logs go to stderr.
`serve-http` starts a Streamable-HTTP-style MCP endpoint at `/mcp`.
JSON-RPC requests are sent with `POST /mcp`; responses are sent as Server-Sent Events with the `mcp-session-id` response header.
Clients that require a shared event stream can open `GET /mcp`, which returns a long-lived Server-Sent Events stream for the MCP session.
`DELETE /mcp` is accepted for MCP session shutdown and does not delete job artifacts.

### MCP JSON-RPC Contract

Supported JSON-RPC methods:

- `initialize`
- `notifications/initialized`
- `ping`
- `tools/list`
- `tools/call`

Tool names exposed through `tools/list`:

- `remote_agent_health`
- `remote_agent_list_agents`
- `remote_agent_start`
- `remote_agent_status`
- `remote_agent_read`
- `remote_agent_send_input`
- `remote_agent_cancel`
- `remote_agent_diff`
- `remote_agent_artifacts`
- `remote_shell_start`

Tool responses are MCP text content blocks containing formatted JSON.
Broker tool failures are returned as MCP tool errors with `isError=true`.
Unsupported agent names, including `codex`, are rejected at this boundary before a job directory is allocated.
The declarations returned by `tools/list` are the validation owner.
Before dispatch or artifact allocation, validation rejects undeclared fields, wrong JSON types, invalid enum values, and numeric values below declared minima.

### Tool Inputs

`remote_agent_start` inputs:

- `agent`: required string, one of `copilot` or `claude`.
- `cwd`: required working directory on the remote host.
- `prompt`: required user instruction for the runner.
- `runner_options`: optional object interpreted by the selected runner adapter.
- `idle_timeout_sec`: optional no-output timeout for AI-agent jobs; defaults to shared `agent_execution.idle_timeout_sec`.
- `timeout_sec`: compatibility alias for `idle_timeout_sec` on AI-agent jobs.
- `max_wall_time_sec`: optional total runtime limit for AI-agent jobs.
- `ttl_days`: optional artifact retention override.
- `capture_limit_bytes`: optional output capture limit.

`remote_shell_start` inputs:

- `command`: required command string or argv array.
- `cwd`: optional working directory on the remote host.
- `async`: optional boolean, default `false` for short shell convenience.
- `timeout_sec`: optional process timeout.
- `env`: optional per-process environment additions.
- `capture_limit_bytes`: optional output capture limit.

Lifecycle inputs:

- `remote_agent_status`: `job_id`.
- `remote_agent_read`: `job_id`, optional `stream`, `since_offset`, `max_bytes`.
- `remote_agent_send_input`: `job_id`, `text`.
- `remote_agent_cancel`: `job_id`, optional `signal`.
- `remote_agent_diff`: `job_id`, optional `base_ref`.
- `remote_agent_artifacts`: `job_id`.

### Job State

Every started command or agent run receives a `job_id`.
Job ids are unique under the broker runtime root and map to:

```text
.tokimon/tools/mcp/remote-agent/jobs/<job_id>/
```

Allowed job states:

- `queued`: accepted but process not started yet.
- `running`: process is active.
- `waiting_input`: process is active and the adapter believes follow-up input is needed.
- `completed`: process exited successfully.
- `failed`: process exited non-zero, timed out, or produced an adapter failure.
- `cancelled`: broker cancellation completed.
- `lost`: broker state exists but the process cannot be found.
- `expired`: job artifacts are past retention and may be removed by cleanup.

All updates to one job's `job.json` pass through one authoritative state boundary so concurrent lifecycle activity cannot overwrite newer state.
Different jobs remain independent and may progress concurrently.

Terminal states are sticky.
A transition from an active state (`queued`, `running`, or `waiting_input`) to any terminal state is allowed once.
No later update may replace one terminal state with another.

Cancellation is accepted only while the process and persisted job are active.
The broker durably transitions the job to `cancelled` before signalling the process.
The monitor then records exit code, completion time, duration, diff metadata, and final artifacts.
It does not change `cancelled`.

If signal delivery fails while the process remains alive, the failure is persisted.
A later cancellation retries delivery without replacing `cancelled`.
If the monitor commits another terminal state first, a late cancellation reports that existing terminal result.
It does not overwrite the persisted state.

Interactive input and cancellation are serialized per runtime.
Input is rejected after persisted state leaves the active set.

`job.json` stores:

- `job_id`
- `kind`: `shell` or `agent`
- `agent`: `shell`, `copilot`, or `claude`
- `cwd`
- `command`
- `runner_options`: redacted resolved runner options after settings and invocation overrides have been applied
- `prompt_summary`: redacted bounded summary; the full prompt is not persisted in job state
- `request_context`: the originating `correlation_id`, JSON-RPC `request_id`, and transport (`stdio-mcp`, `http-mcp`, or direct `broker` call)
- `status`
- `pid`
- `created_at`
- `started_at`
- `updated_at`
- `completed_at`
- `exit_code`
- `duration_sec`
- `timeout_sec`
- `idle_timeout_sec`
- `max_wall_time_sec`
- `last_output_at`
- `ttl_days`
- `capture_limit_bytes`
- `artifacts`
- `diff`: optional captured-diff metadata written during terminal finalization
- `last_error`
- `cancellation_signal_error`: optional signal-delivery diagnostic retained until a later cancellation attempt succeeds.

### Runner Adapters

Runner adapters translate the unified lifecycle into CLI-specific process invocations.
Each adapter owns executable discovery, prompt delivery, CLI flags, output parsing, and interaction support for its runner.

Supported runners:

- `shell`
  - Runs shell commands under the remote user account.
  - Supports synchronous short command execution, asynchronous jobs, and interactive input while the asynchronous process remains active.
- `copilot`
  - Uses GitHub Copilot CLI.
  - Runs non-interactively and rejects follow-up input through the lifecycle boundary.
  - Accepts invocation options through the runner adapter.
- `claude`
  - Uses Claude Code on the remote host.
  - Runs non-interactively and rejects follow-up input through the lifecycle boundary.
  - Supports runner options for model, permission mode, output format, resume behavior, and interaction mode where supported.

Runner options are intentionally not flattened into the MCP top-level schema.
Read the calling main thread's active model and effort following `docs/guidelines/configuration-guidelines.md`, then pass both through `runner_options`.
Supplied `runner_options` are persisted in `job.json`.
Health and job launch require no provider model/effort settings.
Health records this as `agent_settings.model_settings_required=false`.
Examples:

```json
{
  "agent": "claude",
  "cwd": "/workspace/project",
  "prompt": "Review this diff and report bugs only.",
  "runner_options": {
    "output_format": "json",
    "permission_mode": "bypassPermissions"
  }
}
```

Exact CLI flag mapping remains adapter-owned and is revalidated against supported CLI versions.
Adapters should avoid exposing secrets in process arguments when a runner supports safer prompt delivery.

### Artifacts

Each job directory may contain:

- `job.json`: machine-readable state and metadata.
- `stdout.log`: captured stdout.
- `stderr.log`: captured stderr.
- `transcript.md`: human-readable transcript across prompts, input, output, and status transitions.
- `final.json`: adapter-normalized final result when available.
- `diff.patch`: git diff captured from the job working tree when available.

Artifact writes are append-safe while a job is running.
Readers may observe partial logs and should use offsets for incremental reads.
For AI-agent jobs, stdout/stderr output updates `last_output_at` and refreshes the idle timeout.

Stdout and stderr preserve character order within each stream and remain incrementally readable while the job runs.
Malformed UTF-8 is represented visibly rather than terminating capture.
Apart from malformed-byte replacement, secret redaction, and overlong-line truncation, recorded output remains unchanged.
Incremental reads use byte offsets but return only complete UTF-8 characters.
A caller-provided `since_offset` must identify a character boundary, and `max_bytes` must be large enough for the next complete character.
Terminal state and final artifacts publish only after output capture completes or a bounded, visible capture failure is recorded.
After capture, `transcript.md` records bounded stdout and stderr tails under `Output` before the terminal status section.
Artifact reads, diff generation, inventory, and cleanup cannot delete or replace a job directory while another artifact operation is using it.
Observable stream activity refreshes the idle deadline without delaying lifecycle transitions or causing unbounded state writes.

Artifacts must not persist:

- environment-variable dumps;
- tokens or API keys;
- cookies or browser storage;
- SSH private keys;
- passwords;
- device codes or auth codes.

Adapters should redact common secret patterns before writing transcript or final artifacts where practical.

### Agent Execution Policy

Claude and Copilot jobs use shared agent execution settings from `.tokimon/settings.local.json` where applicable.
The default `idle_timeout_sec` comes from the shared configuration contract.
This timeout measures absence of observable stdout/stderr progress, not total runtime.
`max_wall_time_sec` is optional and disabled when unset.
Shell jobs keep their existing short-command wall-clock timeout behavior.

## Data Ownership

Default runtime root:

```text
.tokimon/tools/mcp/remote-agent/
```

Job artifacts live under `jobs/<job_id>/`; other broker working data is transient and not part of the caller contract.

Cleanup requires positive retention values.
Without a CLI override, cleanup uses the `ttl_days` persisted in each job's state.
An explicit cleanup `--ttl-days` value overrides every job for that invocation.
Cleanup must never traverse outside the runtime root.
Job-directory symbolic links are unsafe cleanup inputs even when they resolve within the runtime root.
On concurrent-ownership, unsafe-link, boundary, or filesystem failure, cleanup returns `ok: false`, reports the failure, and the CLI exits nonzero; unsafe paths are detected before eligible directories are removed.

- Owns job-local state, transcripts, logs, diffs, final payloads, artifact manifests, process metadata, and transient broker working data under this root.
- Consumes read-only runner executable output, existing job artifacts during status/read/artifact calls, and configured runner settings.
- Consumes mutable runtime inputs supplied per job, including the requested remote working directory, prompt, command, environment additions, timeout settings, and `runner_options`.
- Derives job status, log tails, artifact lists, diff summaries, and final payloads for MCP callers.

## Trust and Transport

The broker runs inside a trusted remote account.
It should not block arbitrary shell commands or arbitrary working directories by default.
Optional guard modes may add path allowlists, command deny lists, or confirmation checks, but the default execution path prioritizes convenience and auditability.

The broker must bind to `127.0.0.1` by default.
Local callers should connect through an SSH tunnel:

```bash
ssh -N -L 8938:127.0.0.1:8938 <remote-host>
```

HTTP transport accepts only loopback binds and local clients, including clients connected through an SSH tunnel.
Each request requires a loopback `Host` authority and no `Origin` header; browser requests and DNS-rebinding hostnames receive HTTP 403 before dispatch.
HTTP POST requires `application/json`, one positive decimal `Content-Length` of at most 1 MiB, and no `Transfer-Encoding`.
Invalid framing, incomplete bodies, and unsupported content types fail before dispatch.
HTTP connections have a ten-second read timeout.
Use stdio or an SSH tunnel for remote access; unauthenticated non-loopback HTTP is refused before socket creation.

Each transport request creates one correlation context.
When a tool call starts a job, that context is persisted in `job.json` and included with the generated `job_id` in structured `job_accepted` and `job_completed` broker events.
This preserves the chain from an HTTP MCP session or stdio JSON-RPC request to the asynchronous job and its terminal artifacts.

## Limits

- Runner-specific flag mappings must remain compatible with supported CLI versions and be revalidated after CLI upgrades.
- Cross-machine file sync, PR creation, and workflow orchestration policy are separate workflow concerns.
- Artifact redaction is best-effort and does not replace avoiding secret-producing commands.

## Failure Modes

- **Runner unavailable or unsupported:** job start is refused before creating a running process.
  Recover by fixing the executable path or installed CLI version, then rerun `remote_agent_health` before starting another job.
- **Runner exits non-zero or times out:** the job enters `failed`, preserving `stdout.log`, `stderr.log`, `transcript.md`, `job.json`, and any partial `final.json` or `diff.patch`.
  Recover by inspecting `remote_agent_status`, `remote_agent_read`, and `remote_agent_artifacts`, then resubmit a new job only after preserving the failed job id for review.
- **Cancelled job:** the job enters `cancelled` and keeps artifacts written before cancellation.
  Recover by checking artifact state, then either submit a fresh job or run cleanup after the retained evidence is no longer needed.
- **Concurrent completion and cancellation:** whichever terminal transition commits first remains authoritative.
  An accepted cancellation is committed before signalling, while a late cancellation after completion returns the completed terminal state.
  Recover by using the returned and persisted terminal status rather than inferring outcome from the process signal alone.
- **Cancellation signal failure:** the accepted cancellation remains `cancelled`, records the signal-delivery error, and keeps the live runtime available for another cancellation attempt.
  Recover by inspecting `last_error`, retrying cancellation, and verifying that monitor metadata records the eventual process exit.
- **Output capture cannot reach end-of-stream:** the Artifacts contract's bounded drain procedure publishes a visible capture failure instead of waiting indefinitely.
  Recover by inspecting capture diagnostics and surviving artifacts, then investigate orphan descendant processes before rerunning the job.
- **Lost process after broker restart:** the job enters `lost` when state exists but the process cannot be found.
  Recover by inspecting `job.json` and logs, confirming whether the underlying process is gone, then resubmit rather than editing the lost job state.
- **Expired or malformed artifact sink:** `remote_agent_status` and artifact reads report the missing, expired, or malformed sink.
  Recover by preserving the remaining job directory, using `remote_agent_artifacts` to identify available files, and rerunning the job when required evidence is gone.
- **Unsupported interactive input:** `remote_agent_send_input` returns a tool error without mutating transcript state for runners that cannot accept input.
  Recover by cancelling or resubmitting with a non-interactive prompt.
- **Secret-bearing output risk:** artifact redaction is best-effort and cannot make secret-producing commands safe.
  Recover by stopping the job, rotating any exposed secret outside the broker, and removing or quarantining local artifacts according to operator policy.
- **Concurrent cleanup or boundary failure:** cleanup fails closed if another artifact operation owns the job or a path falls outside the runtime root.
  Recover by preserving the cleanup diagnostic, waiting for active work to finish or correcting the unsafe path, then rerun cleanup.
- **Unsafe cleanup link or filesystem failure:** cleanup rejects job-directory symbolic links before deletion and reports filesystem failures instead of leaking a raw exception.
  Recover by preserving the diagnostic, replacing the unsafe entry with a verified regular job directory or quarantining it, then rerun cleanup.
- **Unsafe HTTP access:** non-loopback binds are refused, and browser, foreign-host, or malformed requests fail before tool dispatch.
  Recover by using stdio or a local HTTP client through an SSH tunnel with the documented JSON framing.

## Acceptance Checks

Unit tests should cover:

- command, MCP transport, and declared tool-schema behavior
- HTTP Origin/Host/client/bind rejection, bounded JSON body framing, malformed UTF-8, and refusal of non-loopback startup before socket creation
- runner discovery, settings resolution, supported-runner rejection, and shell/AI job start
- status, incremental read, input, cancellation, diff, and artifact operations
- request correlation, monotonic terminal state, concurrent completion/cancellation, and restart recovery
- ordered output capture, character-safe offsets, deadlines, and visible capture failures
- retention, concurrent artifact access, cleanup containment, and unsafe-link rejection
- registry availability

Verification:

```bash
.venv/bin/python -m unittest discover -s tests/tools/mcp/remote_agent_broker -t . -p 'test_remote_agent_broker.py'
.venv/bin/python -m unittest discover -s tests/tools/registry -t . -p 'test_registry.py'
git diff --check
```

Expected outcome: focused tests pass for MCP tools, runner adapters, job state recovery, cleanup boundaries, and registry availability; `git diff --check` completes without failures.

Ask an AI coding agent to verify the component contract:

```text
Review docs/c4/level-3-components/tools/mcp/remote-agent-broker.md, tokimon/tools/mcp/remote_agent_broker/, tests/tools/mcp/remote_agent_broker/, and tests/tools/registry/.
Confirm that the documented MCP transport contract, runner inputs, job state, artifact ownership, cleanup boundary, trust boundary, failure modes, and tests match the implementation.
Report any drift with file and line evidence.
```

Expected outcome: the agent reports no contract drift, or lists precise mismatches with recommended owner documents or implementation files.

## References

- L2 Container: `docs/c4/level-2-containers/tools/mcp/remote-agent-broker.md`
- L1 Context: `docs/c4/level-1-context/tools-context.md`
