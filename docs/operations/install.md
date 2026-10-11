---
title: "Install Operation"
summary: "Operator procedure for installing or reinstalling Tokimon into a target AI platform from live repository sources."
read_when:
  - Installing, reinstalling, refreshing, or updating Tokimon for a target AI platform.
  - Handling requests such as "install Tokimon" or "reinstall Tokimon using plugin-creator".
---

# Install Operation

This document owns Tokimon install and reinstall procedure.
For Tokimon, install means refresh the target-platform package from live repository sources before account-level platform installation.

## Contents

- [Scope](#scope)
- [Codex](#codex)
- [Claude](#claude)
- [Acceptance](#acceptance)

## Scope

Use this operation when the operator asks to install, reinstall, refresh, or update Tokimon for an AI platform.

Do not use cachebuster-only reinstall unless the operator explicitly asks to skip package generation.
Tokimon package output is generated from canonical live sources, so reinstalling stale generated output is incomplete.
Do not install, upgrade, or download external dependency software, provider CLIs, or container images to complete Tokimon installation.
Report essential missing prerequisites as blockers and disable unavailable optional capabilities.

## Codex

Codex install composes two owners:

- AI Harness Interface owns Tokimon package generation, validation, route data, and post-install readiness.
- The current `plugin-creator` skill and official OpenAI plugin documentation own Codex package shape, marketplace shape, and local installation guidance.
  Legacy system-skill helpers are optional authoring tools, not a prerequisite for installing a supported local plugin.

Run the steps in this order:

1. If the generated Codex package scaffold does not exist yet, bootstrap it:

```bash
.venv/bin/python -m tokimon.ai_harness_interface bootstrap-codex --json
```

This uses available legacy helpers or built-in authoring of the [official supported compatibility format](https://developers.openai.com/plugins/build/plugins).
Missing old system-skill scripts do not block normal installation.

2. Configure repository-local AI Debate and Pair defaults without interactive MCP selection.

```bash
.venv/bin/python -m tokimon.ai_harness_interface configure --yes --json
```

3. Generate the Codex package from live Tokimon sources.

The generated package selects supported surfaces from the local installation policy.
Missing Docker automatically excludes Remote Browser Broker and Container Executor.
Use `configure --docker-features disabled --yes` to keep them off even on a host with Docker.

```bash
.venv/bin/python -m tokimon.ai_harness_interface generate --json
```

4. Use the current `plugin-creator` skill to inspect the generated package and follow its local install flow.
   Run Tokimon package validation; unavailable legacy validators are reported as skipped.
   Register the generated marketplace with `codex plugin marketplace add`, then use the installed CLI's supported plugin install command or the desktop Plugins UI.
   Confirm that the target host accepts the package. Do not edit installed cache files.

Tokimon handoff facts for plugin-creator:

- plugin id: `tokimon`
- marketplace name: `tokimon-local`
- generated plugin root: `.tokimon/ai-harness-interface/codex/plugins/tokimon`
- generated marketplace path: `.tokimon/ai-harness-interface/codex/.agents/plugins/marketplace.json`
- generated marketplace root: `.tokimon/ai-harness-interface/codex`

Resolve relative paths from the Tokimon checkout running the install operation.

5. Run post-install readiness.

```bash
.venv/bin/python -m tokimon.ai_harness_interface post-install --json
```

Docker-free packages omit Docker-dependent services and skills automatically.
Use the native browser order in `skills/using-tokimon/skill.md`.
In-app Browser is the first choice; it does not require the broker.

## Claude

Claude Code install composes two owners:

- AI Harness Interface owns Tokimon package generation, validation, route data, and post-install readiness.
- The Claude Code plugin CLI owns account-level marketplace registration, plugin install, update, and uninstall mechanics.

Run the steps in this order:

1. Configure repository-local AI Debate and Pair defaults without interactive MCP selection.

```bash
.venv/bin/python -m tokimon.ai_harness_interface configure --yes --json
```

2. Generate and validate the Claude Code package from live Tokimon sources.

The same installation policy selects the Claude Code package surfaces.

```bash
.venv/bin/python -m tokimon.ai_harness_interface generate --target-platform claude --json
.venv/bin/python -m tokimon.ai_harness_interface validate --target-platform claude --json
```

3. Run the account-level Claude Code install with operator approval.
   Resolve relative paths from the Tokimon checkout running the install operation.
   Keep the leading `./` on the marketplace path.
   Without it, the Claude Code CLI parses the argument as a GitHub `owner/repo` source and attempts a git clone.

```bash
claude plugin marketplace add ./.tokimon/ai-harness-interface/claude
claude plugin install tokimon@tokimon-local
```

For reinstall or refresh after regeneration, update instead of re-adding:

```bash
claude plugin marketplace update tokimon-local
claude plugin update tokimon@tokimon-local
```

Tokimon handoff facts for the Claude Code plugin CLI:

- plugin id: `tokimon`
- marketplace name: `tokimon-local`
- generated plugin root: `.tokimon/ai-harness-interface/claude/plugins/tokimon`
- generated marketplace path: `.tokimon/ai-harness-interface/claude/.claude-plugin/marketplace.json`
- generated marketplace root: `.tokimon/ai-harness-interface/claude`

4. Run post-install readiness.

```bash
.venv/bin/python -m tokimon.ai_harness_interface post-install --target-platform claude --json
```

## Acceptance

Installation is complete only when:

- repository-local setup is configured, any required Codex scaffold exists, and package generation succeeded before reinstall or cachebuster
- package inspection through the current `plugin-creator` skill and Tokimon validation completed for Codex, or Claude Code CLI validation completed for Claude; skipped optional legacy helpers are reported
- post-install readiness reports the plugin installed and enabled
- MCP registration and route-back health are reported
- every policy-selected MCP entry is present and its registration state is reported; disabled optional surfaces are listed
- missing external MCP prerequisites are reported as non-blocking capability setup
- any warnings or remaining work are reported to the operator

When package interface files changed, tell the operator to start a new target-platform thread so refreshed skills and MCP entries load.
