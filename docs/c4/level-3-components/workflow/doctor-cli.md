---
title: "Doctor - CLI"
summary: "L3 command contract for Tokimon's read-only readiness report."
read_when:
  - Changing doctor CLI commands, JSON output, status handling, or check coverage.
  - Reviewing doctor workflow verification and command behavior.
---

# Doctor - CLI

## Contents

- [Overview](#overview)
- [Components](#components)
- [Workflow Package Declarations](#workflow-package-declarations)
- [Interfaces/Contracts](#interfacescontracts)
- [Operator Usage](#operator-usage)
- [Output Contract](#output-contract)
- [Safety Rules](#safety-rules)
- [Data Ownership](#data-ownership)
- [Failure Modes](#failure-modes)
- [Acceptance Checks](#acceptance-checks)
- [References](#references)

## Overview

The doctor CLI is the owned operator interface for read-only Tokimon readiness reporting.
It lists memory usage, tools availability, and workflow readiness without mutating repository files, local runtime state, memory, tools, or workflow artifacts.

## Components

- **Mem Palace usage reporter:** Owns read-only local memory presence and schema diagnostics.
- **Tools-registry readiness reporter:** Owns read-only capability metadata loading for tool availability.
- **Workflow readiness reporter:** Owns discovery and aggregation of workflow-owned `WORKFLOW_READINESS` declarations.
- **Foundation checker:** Owns required tracked file, layer directory, `.tokimon/` ignore policy, and `.venv` presence checks.
  For an installed wheel workspace, it checks the owning Python environment instead of requiring a nested `.venv`.
- **Renderers:** Own human-readable and JSON readiness output.

## Interfaces/Contracts

### Workflow Package Declarations

- **Workflow id:** `doctor`
- **Python package:** `tokimon/workflow/doctor/`
- **Canonical code entrypoint:** `tokimon.workflow.doctor.cli:main`
- **Operator entrypoint:** `.venv/bin/python -m tokimon.workflow.doctor`
- **Contract declarations:** `tokimon/workflow/doctor/contract.py`
- **Workflow readiness declaration:** `tokimon/workflow/doctor/contract.py` `WORKFLOW_READINESS`
- **Declared local state root:** `.tokimon/workflow/doctor/`
- **Memory use:** none
- **Durable learning target:** `scope=workflow`, `room=doctor` for Codex-written durable repeated mistakes
- **Required tool capability:** none

Canonical machine-readable workflow contract: `.venv/bin/python -m tokimon.workflow.doctor contract --json`. This section summarizes the contract and JSON readiness output.
Commands emit human-readable text by default and JSON with `--json`.

**Commands**
- `check`
  - Run read-only readiness reporting.
  - This is also the default command when no subcommand is provided.
- `contract`
  - Emit workflow contract declarations.

**Global arguments**
- `--workspace-root <path>`
  - Optional checkout root to inspect.
  - Defaults to the live Tokimon repository root that owns the package.
- `--json`
  - Emit machine-readable JSON for the selected command.

**`check` arguments**

**`contract` arguments**

### Operator Usage

```bash
.venv/bin/python -m tokimon.workflow.doctor check --json
```

```bash
.venv/bin/python -m tokimon.workflow.doctor
```

```bash
.venv/bin/python -m tokimon.workflow.doctor contract --json
```

### Output Contract

`check --json` includes:
- `workflow_id`
- `status`
- `ok`
- `summary`
- `memory`
- `tools`
- `workflows`
- `readiness`
- `checks`

Command status values:
- `pass`
  - No foundation check failed and every readiness item is ready.
- `warn`
  - No foundation check failed, but at least one check warned or one readiness item is partial or blocked.
- `fail`
  - A foundation check found a repository health problem that makes `ok` false.

Readiness status values:
- `ready`
  - The capability or workflow can run according to cheap local diagnostics.
- `partial`
  - A useful mode can run, but an optional, mode-specific, or externally inspected dependency is missing or not locally verifiable.
- `blocked`
  - Required local diagnostics show the capability or workflow cannot run now.

Exit behavior:
- Exit `0` when no foundation check has status `fail`.
- Exit `2` when one or more foundation checks have status `fail`.

Each readiness item includes:
- `id`
- `layer`
- `kind`
- `status`
- `summary`
- `reasons`
- `data`

Workflow readiness data includes the selected mode and evaluated mode list from the source workflow's `WORKFLOW_READINESS` declaration.
New workflows appear in Doctor after they add `WORKFLOW_READINESS` to their `contract.py` module.

Workflow readiness declarations use these fields:
- Declaration fields: `id`, `workflow_status`, `entrypoint`, and `modes`.
- Workflow status values: `implemented`, `support`, `planned`, and `deprecated`.
- Mode fields: `id`, `role`, `summary`, `required_tool_capabilities`, `any_tool_capability_groups`, `agent_connector_capabilities`, and `memory_requirement`.
- Mode roles: `primary` and `fallback`.
- Readiness statuses: `ready`, `partial`, and `blocked`.
- A workflow is `ready` when a primary mode is ready, `partial` when a primary mode is partial or only a fallback mode can run, and `blocked` when no declared mode can run.

Each foundation check includes:
- `id`
- `status`
- `summary`
- `details`
- `data`

### Safety Rules

- The command is read-only.
- The command opens Mem Palace with a read-only SQLite connection when the database exists.
- The command calls tools-registry metadata APIs but does not run tool health-check commands.
- Unavailable host-specific tool capabilities are reported as blocked readiness, not treated as command failures.
- The command does not run tests or repair commands.

## Data Ownership

- Owns no persistent repository or local runtime data.
- Consumes read-only tracked repository file/directory presence, `.gitignore`, `.venv` existence, Mem Palace schema metadata, tools registry metadata, and workflow readiness declarations read-only.
- Derives read-only human and JSON readiness reports.
- Does not write Mem Palace, tools, workflow artifacts, generated files, or repair outputs.

## Failure Modes

Doctor recovery is read-only: fix the reported repository, memory, tools-registry, or workflow-readiness input, then rerun `check` without editing Doctor artifacts.

- Required repository files or directories are missing; restore the tracked repository shape before treating readiness output as authoritative.
- `.gitignore` does not ignore `.tokimon/`; restore the local runtime ignore policy and rerun `check`.
- `.venv/bin/python` is missing; recreate the repository virtual environment before running workflows that require it.
- Mem Palace exists but cannot be opened read-only; inspect the database path and continue with memory reported as blocked until it is readable.
- Mem Palace exists but required schema tables are missing; run the owning memory workflow or restore the database before relying on memory readiness.
- Tools-registry metadata cannot be loaded; fix the tools registry contract or import path and rerun `check`.

## Acceptance Checks

```bash
.venv/bin/python -m unittest discover -s tests/workflow/doctor -t . -p 'test_*.py'
```

```bash
.venv/bin/python -m tokimon.workflow.doctor check --json
```

Expected outcome: tests pass and diagnostics emit the documented JSON shape without writing artifacts.

Ask an AI coding agent to verify the component contract:

```text
Review docs/c4/level-3-components/workflow/doctor-cli.md, tokimon/workflow/doctor/, tests/workflow/doctor/, and tokimon/tools/registry.py.
Confirm that the documented check/contract commands, read-only data boundary, readiness output shape, failure modes, and tests match the implementation.
Report any drift with file and line evidence.
```

Expected outcome: the agent reports no contract drift, or lists precise mismatches with recommended owner documents or implementation files.

## References

- L2 Container: `docs/c4/level-2-containers/workflow/doctor.md`
