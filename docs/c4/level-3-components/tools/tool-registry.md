---
title: "Tool Registry - Component"
summary: "L3 component contract for the shared tool catalog, host detection, connector inventory, readiness diagnostics, fail-fast requirements, and verification."
read_when:
  - Changing tool catalog or registry command behavior, capability declarations, status values, connector inventory, or require semantics.
  - Reviewing the exact Tool Registry operator/API contract.
---

# Tool Registry - Component

## Contents

- [Overview](#overview)
- [Components](#components)
- [Interfaces/Contracts](#interfacescontracts)
- [Data Ownership](#data-ownership)
- [Failure Modes](#failure-modes)
- [Acceptance Checks](#acceptance-checks)
- [References](#references)

## Overview

Tool Registry is the read-only capability discovery component for Tokimon tools.
At L3, this document owns the exact operator commands, Python API behavior, status meanings, connector-backed availability contract, failure behavior, and acceptance checks for the Tool Registry component.
Its primary outcome is one capability catalog plus reliable read-only readiness reporting and early failure when a workflow requires an unknown or unavailable capability.

## Components

### Host Profile Detector

- **Responsibility:** Reports local host traits that affect tool availability.
- **Key surfaces:** operating-system facts, GUI availability, and declared available MCP connector ids.

### Capability Definitions

- **Responsibility:** Defines the known tools-layer capability ids and their static metadata.
- **Key surfaces:** capability id, kind, package route, readiness requirements, command surface, health check, and C4 references.

### Connector Inventory Loader

- **Responsibility:** Reads optional local MCP connector inventory without treating connector metadata as committed configuration.
- **Key surfaces:** `TOKIMON_MCP_CONNECTORS_JSON`, `.tokimon/tools/mcp/connectors.json`, connector id, capability id, source, status, transport, and exposed tool names.

### Requirement API

- **Responsibility:** Lets workflows stop before domain work starts when a required capability cannot run on the current host.
- **Key surfaces:** `require_capability(capability_id)`, `ToolCapabilityRequirementError`, and `UnknownToolCapabilityError`.

### CLI Renderer

- **Responsibility:** Provides human-readable and JSON diagnostics for operators, agents, tests, and workflows.
- **Key surfaces:** `host`, `list`, `show`, and `require` commands.

## Interfaces/Contracts

Canonical machine-readable readiness and requirement contracts are emitted by `.venv/bin/python -m tokimon.tools.registry list --json`, `show <capability-id> --json`, and `require <capability-id> --json`.

### Commands

```bash
.venv/bin/python -m tokimon.tools.registry host
.venv/bin/python -m tokimon.tools.registry list --json
.venv/bin/python -m tokimon.tools.registry show mcp.telegram-bot --json
.venv/bin/python -m tokimon.tools.registry show mcp.ai-debate --json
.venv/bin/python -m tokimon.tools.registry show mcp.remote-browser-broker --json
.venv/bin/python -m tokimon.tools.registry show mcp.remote-agent-broker --json
.venv/bin/python -m tokimon.tools.registry require mcp.remote-agent-broker --json
```

Command behavior:

- `host` shows the detected host profile.
- `list` lists all known tool capabilities and their current status.
- `show <capability_id>` shows one capability by id.
- `show <capability_id> --json` returns an `ok: false` error object when the id is unknown.
- `require <capability_id>` succeeds only when the capability exists and is `available`.
- `require <capability_id>` returns actionable diagnostics and a non-zero exit code when the capability is unknown or `unavailable`.

### Python API

The public read-only registry API consists of:

- `tokimon.tools.registry.detect_host_profile`: returns host traits and connector ids declared available by the inventory.
- `tokimon.tools.registry.list_capabilities`: returns ordered `ToolCapability` records for `tokimon.tools.catalog.MCP_CATALOG`.
- `tokimon.tools.registry.find_capability`: returns one matching capability or `None`.
- `tokimon.tools.registry.require_capability`: returns an available capability, raises `UnknownToolCapabilityError` for an unknown id, and raises `ToolCapabilityRequirementError` for a known unavailable capability.
- `tokimon.tools.registry.capability_map`: returns the same host and capability records as JSON-compatible data.
- `tokimon.tools.registry.load_mcp_connectors`: returns validated local `MCPConnector` records.

Callers may supply `workspace_root` and environment mappings without changing process-global state.
`require_capability` also accepts an optional pre-detected `HostProfile` positional argument for compatibility with existing callers.
Malformed or unreadable connector inventory raises `MCPConnectorInventoryError`; local prerequisite failures resolve the affected capability to `unavailable`.

`UnknownToolCapabilityError` has exit code `2`; `ToolCapabilityRequirementError` has exit code `3` for known unavailable capabilities.

### Shared Catalog Mapping

`tokimon.tools.catalog.MCP_CATALOG` is the canonical static metadata source.
Registry `list --json` renders its complete capability set.
`tokimon.tools.catalog.package_entries()` exposes the ordered, closed set eligible for packaging and returned by AI Harness Interface `contract --json`.
The AI Harness Interface installation policy selects that set for each generated package; hosts without Docker omit the Remote Browser Broker.

| Capability id | Purpose | Server id | Live module | Target-platform package |
|---|---|---|---|---|
| `mcp.telegram-bot` | Send-only Telegram Bot API MCP | `tokimon-telegram-bot` | `tokimon.tools.mcp.telegram_bot` | included |
| `mcp.ai-debate` | Persisted adversarial debate and judgment MCP | `tokimon-ai-debate` | `tokimon.tools.mcp.ai_debate` | included |
| `mcp.remote-browser-broker` | Isolated remote browser session broker | `tokimon-remote-browser-broker` | `tokimon.tools.mcp.remote_browser_broker` | included |
| `mcp.remote-agent-broker` | Persistent remote agent job broker | `tokimon-remote-agent-broker` | `tokimon.tools.mcp.remote_agent_broker` | not included |
| `mcp.container-executor` | Independent container command/file execution | `tokimon-container-executor` | `tokimon.tools.mcp.container_executor` | not included |

Catalog records also carry the package route, commands, health check, C4 references, host requirements, artifact policy, failure modes, and connector presentation metadata.
C4-reference integrity is a repository check, not a host-readiness input.

### Status Values

- `available`
  - Local prerequisites are present or connector inventory declares the capability available.
- `unavailable`
  - Capability is known, but its local launcher, host requirements, local configuration, or connector availability are missing.

Registry checks do not verify live service connectivity or authentication.
The host field `reachable_mcp_endpoints` contains connector ids declared available by the inventory.

Local browser readiness resolves `TOKIMON_RB_DOCKER` (default `docker`) using `PATH`, matching the browser runtime.
An explicit empty or unresolved override makes the local capability unavailable.

### Connector Inventory

To report connector-backed MCP capabilities, provide local runtime connector inventory through `TOKIMON_MCP_CONNECTORS_JSON` or `.tokimon/tools/mcp/connectors.json`.

```bash
TOKIMON_MCP_CONNECTORS_JSON='{"connectors":[{"id":"tokimon-remote-browser-broker","capability_id":"mcp.remote-browser-broker","source":"codex","status":"available","transport":"codex-mcp","tools":["browser_ensure"]}]}' \
  .venv/bin/python -m tokimon.tools.registry list --json
```

Expected outcome: the host profile reports `tokimon-remote-browser-broker` under `reachable_mcp_endpoints` and `mcp.remote-browser-broker` is available through the configured connector.
Use the same inventory format for connector-backed AI Debate and remote agent brokers by setting `capability_id` to `mcp.ai-debate` or `mcp.remote-agent-broker`.

## Data Ownership

- Owns no persistent state.
- Owns tool capability discovery, status reporting, and requirement enforcement.
- Consumes read-only local runtime MCP connector inventory from `TOKIMON_MCP_CONNECTORS_JSON` or `.tokimon/tools/mcp/connectors.json`.
- May inspect local Telegram token presence from `.tokimon/secrets.local.json` or `TELEGRAM_BOT_TOKEN` without printing token values.
- May inspect AI Debate provider CLI presence and required local settings without invoking either provider.
- Derives read-only diagnostics from host facts, capability declarations, connector inventory, and local prerequisite checks.
- Does not own browser session state, Telegram bot tokens, MCP credentials, or workflow artifacts.

## Failure Modes

Tool Registry recovery fixes the local connector inventory, host prerequisite, or capability declaration and reruns the read-only registry command; the registry owns no mutable state to clean up.

- A known capability is unavailable; inspect `show <capability_id> --json`, restore its local launcher, token, settings, provider CLI, Docker, or connector prerequisite, and retry the workflow.
- A capability id is unknown; correct the caller to use one of the declared capability ids or add the capability through the owning tools-layer contract first.
- A connector inventory entry is missing, unavailable, malformed, or unreadable; fix `TOKIMON_MCP_CONNECTORS_JSON` or `.tokimon/tools/mcp/connectors.json` and rerun `list --json`.
- Provider model settings are not a Remote Agent Broker health prerequisite; callers supply execution options following `docs/guidelines/configuration-guidelines.md`.

## Acceptance Checks

1. Verify registry tests:
   ```bash
   .venv/bin/python -m unittest discover -s tests/tools/registry -t . -p 'test_registry.py'
   ```
   Expected outcome: tests pass for host detection, MCP connector inventory, capability readiness (including custom Docker commands and invalid overrides), catalog C4 reference integrity, shared-catalog alignment, and requirement success, unknown-id, unavailable-capability, JSON, and CLI parser behavior.

2. Verify diagnostics:
   ```bash
   .venv/bin/python -m tokimon.tools.registry list --json
   ```
   Expected outcome: output includes the complete shared catalog with statuses matching the current host and local connector configuration.

3. Verify connector-backed diagnostics:
   ```bash
   TOKIMON_MCP_CONNECTORS_JSON='{"connectors":[{"id":"tokimon-remote-browser-broker","capability_id":"mcp.remote-browser-broker","source":"codex","status":"available","transport":"codex-mcp","tools":["browser_ensure"]}]}' \
     .venv/bin/python -m tokimon.tools.registry list --json
   ```
   Expected outcome: `reachable_mcp_endpoints` includes `tokimon-remote-browser-broker` and `mcp.remote-browser-broker` is `available`.

4. Ask an AI coding agent to verify the component contract:
   ```text
   Review docs/c4/level-3-components/tools/tool-registry.md, tokimon/tools/catalog.py, tokimon/tools/registry.py, and tests/tools/registry/.
   Confirm that the documented capability catalog, CLI/API surfaces, readiness checks, connector inventory, status values, and requirement failures match the implementation and tests.
   Report any drift with file and line evidence.
   ```
   Expected result: no drift between the Tool Registry contract, implementation, and tests.

## References

- L1 Context: `docs/c4/level-1-context/tools-context.md`
- L2 Container: `docs/c4/level-2-containers/tools/tool-registry.md`
