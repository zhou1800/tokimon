---
title: "Tokimon Configuration Guidelines"
summary: "Repo-wide rules for machine-local Tokimon settings under `.tokimon/settings.local.json`."
read_when:
  - Adding or changing shared machine-local configuration.
  - Deciding whether a workflow, tool, or skill should read `.tokimon/settings.local.json`.
  - Reviewing AI-agent placement and model or effort propagation.
---

# Tokimon Configuration Guidelines

This file owns shared machine-local configuration rules for Tokimon.
It is intentionally small so future sections can be added without creating a configuration subsystem before one is needed.

## Contents

- [Settings File](#settings-file)
- [AI-Agent Settings](#ai-agent-settings)
- [Current Sections](#current-sections)

## Settings File

Tokimon shared local settings live at:

```text
.tokimon/settings.local.json
```

Rules:

- The file is local runtime configuration.
  AI-agent execution features that need provider access settings must fail fast when it is missing.
- The file must be a JSON object.
  Tokimon does not use YAML for local settings because the repository has no YAML runtime dependency.
- The file is local-only runtime state under `.tokimon/` and must not be committed.
  The tracked template is `tokimon/settings.example.json`.
- Do not store secrets, credentials, or session material in settings.
- Top-level keys are feature sections.
  Keep sections flat until a real consumer needs deeper structure.
- Invocation arguments may override individual settings for one run; settings remain required for features that own provider placement.
- Do not add code defaults for AI-agent provider or access choices.
  If required settings are missing, refuse to execute.

## AI-Agent Settings

Read the calling main thread's active model and effort and pass them only to CLI agents from the same provider.
For a different provider, use explicit provider-specific model and effort overrides or omit both to use that CLI's defaults.
Do not translate model names or effort values between providers.
For Codex, use `CODEX_THREAD_ID` to locate the matching session JSONL under `$CODEX_HOME/sessions` (`~/.codex/sessions` when `CODEX_HOME` is unset), then read `model` and `effort` from its latest `turn_context` record.
Tokimon does not store provider model or effort choices in local settings.
Run and job artifacts record requested options; their model fields do not identify the actual provider model unless independently verified.

## Current Sections

The example file at `tokimon/settings.example.json` is the complete current settings shape.
Add new sections only when a checked-in consumer needs them.

### `ai_debate`

`ai_debate` stores the required placement choices for AI Debate:

```json
{
  "ai_debate": {
    "access": "local-cli",
    "peer_b": "codex"
  }
}
```

- `access`: must be `local-cli`.
- `peer_b`: `claude` or `codex`.
- AI Debate run arguments may override both fields for one invocation.
- AI Debate requires this section before provider execution starts.

### `agent_execution`

`agent_execution` stores shared local execution policy for Codex and Claude agent work:

```json
{
  "agent_execution": {
    "idle_timeout_sec": 3600,
    "poll_interval_sec": 5.0,
    "max_wall_time_sec": null
  }
}
```

- `idle_timeout_sec`: seconds without observable output before an agent process is timed out.
- `poll_interval_sec`: seconds between output/activity checks.
- `max_wall_time_sec`: optional total runtime limit; `null` means no wall-clock limit.

LLM agent calls should stream stdout/stderr, broker output, or browser-visible generation state when the platform exposes it.
Active output refreshes the idle timer.
The default one-hour idle timeout is a fallback for silent processes, not a limit on normal long thinking.

### `pair`

`pair` stores the Pair peer placement required before a Pair run can prompt Peers:

```json
{
  "pair": {
    "access": "native-subagent",
    "peer_b": "codex",
    "cwd": "<tokimon-checkout-path>"
  }
}
```

- `access`: Codex Peer A placement: `native-subagent` (default) or `local-cli`.
- `peer_b`: `codex`; `local-cli` also allows `claude`.
  Both Peers use the configured `access`.
- `cwd`: fallback task checkout; `select-peers --cwd <path>` overrides it for one run.

Pair support commands require the `pair` section before a run starts.
One-run Pair overrides belong on Pair invocation surfaces or explicit alternate Tokimon settings paths.

### AI Harness Interface Setup

AI Harness Interface setup writes these same local settings through an explicit `configure` command.
Use explicit flags or `--yes`, then use AI Harness post-install readiness output for remaining work and next steps.

### `installation`

`installation.docker_features` is `auto` by default, or `disabled` to keep Docker-dependent capabilities off.
AI Harness generation resolves `auto` from local executable discovery and records the effective package selection.
Installation never installs external dependency software.
