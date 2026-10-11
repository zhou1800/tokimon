---
title: "Memory Maintenance - CLI"
summary: "L3 command contract for Mem Palace maintenance snapshots, safe mutations, backups, run logs, and repair."
read_when:
  - Changing memory maintenance CLI commands, mutation safety, backup behavior, run logs, or repair checks.
  - Reviewing memory maintenance workflow verification and command behavior.
---

# Memory Maintenance - CLI

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

The memory maintenance CLI is the owned operator interface for periodic Mem Palace cleanup.
It exposes deterministic facts and safe mutations while leaving fuzzy cleanup decisions to Codex.

## Components

- **Snapshot builder:** Owns factual Mem Palace record output and aggregate counts.
- **Exact duplicate detector:** Owns strict content-hash grouping only.
- **Safe mutation commands:** Own delete and weight-update behavior while refusing pinned records.
- **Backup and run-log writer:** Owns SQLite backups and JSONL run logs for every mutation.
- **Repair command:** Owns deterministic index and schema checks.

## Interfaces/Contracts

### Workflow Package Declarations

- **Workflow id:** `memory-maintenance`
- **Python package:** `tokimon/workflow/memory_maintenance/`
- **Canonical code entrypoint:** `tokimon.workflow.memory_maintenance.cli:main`
- **Operator entrypoint:** `.venv/bin/python -m tokimon.workflow.memory_maintenance`
- **Contract declarations:** `tokimon/workflow/memory_maintenance/contract.py`
- **Workflow readiness declaration:** `tokimon/workflow/memory_maintenance/contract.py` `WORKFLOW_READINESS`
- **Default local state root:** `.tokimon/workflow/memory-maintenance/`
- **Run log root:** `.tokimon/workflow/memory-maintenance/runs/`
- **Backup root:** `.tokimon/workflow/memory-maintenance/backups/`
- **Memory storage:** Mem Palace `scope=memory`, `room=maintenance`, default database `.tokimon/memory/memory.db`
- **Required tool capability:** none

Canonical command surfaces are exposed by `.venv/bin/python -m tokimon.workflow.memory_maintenance --help` plus subcommand `--help`; JSON command outputs own the machine-readable result contracts. This section summarizes the stable command and artifact families.
Commands emit human-readable text by default and JSON with `--json`.

**Commands**
- `snapshot`
  - Emit factual memory records and aggregate statistics.
- `delete`
  - Delete one unpinned memory after backup and write a run log.
- `update-weight`
  - Update one unpinned memory's `importance` and `confidence` after backup and write a run log.
- `repair`
  - Run deterministic store maintenance, rebuild FTS, and report before/after status.

**Global arguments**
- `--db <path>`
  - Optional Mem Palace database path. Defaults to `.tokimon/memory/memory.db`.
- `--state-dir <path>`
  - Optional local workflow artifact root for backups and run logs. Defaults to `.tokimon/workflow/memory-maintenance/`.

**`snapshot` arguments**
- `--scope <scope>`
  - Optional scope filter.
- `--room <room>`
  - Optional room filter.

**`delete` arguments**
- `--id <memory-id>`
  - Required target memory id.
- `--reason <reason>`
  - Required human/Codex-readable reason.
- `--run-id <id>`
  - Optional run id used for grouping run-log and backup filenames.

**`update-weight` arguments**
- `--id <memory-id>`
  - Required target memory id.
- `--importance <0-5>`
  - New retrieval weight.
- `--confidence <0-1>`
  - New confidence value.
- `--reason <reason>`
  - Required human/Codex-readable reason.
- `--run-id <id>`
  - Optional run id used for grouping run-log and backup filenames.

**`repair` arguments**
- `--run-id <id>`
  - Optional run id used for grouping run-log and backup filenames.

### Operator Usage

```bash
.venv/bin/python -m tokimon.workflow.memory_maintenance snapshot --json
```

```bash
.venv/bin/python -m tokimon.workflow.memory_maintenance delete \
  --id 42 \
  --reason "Redundant with memory #17 after full-record review" \
  --json
```

```bash
.venv/bin/python -m tokimon.workflow.memory_maintenance update-weight \
  --id 43 \
  --importance 1 \
  --confidence 0.6 \
  --reason "Maybe useful but stale and low signal" \
  --json
```

```bash
.venv/bin/python -m tokimon.workflow.memory_maintenance repair --json
```

### Output Contract

`snapshot --json` includes:
- `records`: raw memory rows plus simple text-length fields.
- `stats`: total, pinned, unpinned, kind, scope, and room counts.
- `exact_duplicate_groups`: strict content-hash groups only.

Mutation commands include:
- `backup.path`
- `run_log_path`
- target id and reason
- before/after or deleted record evidence

### Safety Rules

- Pinned memories are read-only for this workflow.
- Every mutation writes a SQLite backup first, including committed data still in the WAL while another connection remains open.
- Every mutation writes a JSONL run log.
- Python does not return `suggested_keep` or fuzzy duplicate judgments.
- Soft delete/state is intentionally absent; recovery uses backup plus run log.

## Data Ownership

- Owns local workflow artifacts under `.tokimon/workflow/memory-maintenance/`.
- Owns mutation backups under `.tokimon/workflow/memory-maintenance/backups/` and JSONL run logs under `.tokimon/workflow/memory-maintenance/runs/`.
- Reads Mem Palace records from `.tokimon/memory/memory.db` for snapshots and applies approved maintenance mutations through the documented database path.
- Derives factual snapshots, exact duplicate groups, mutation before/after evidence, backup paths, run-log paths, and repair status.
- Does not own fuzzy duplicate judgments, semantic keep/delete decisions, or pinned-memory mutation authority.

## Failure Modes

- Target memory id does not exist; rerun `snapshot --json` and choose an existing unpinned id.
- Target memory is pinned; leave it untouched unless the owning memory contract changes the pin policy.
- Target memory is referenced by temporal relationships; `delete` reports the reference count and refuses the operation before creating backup or run-log artifacts.
  Keep the memory while any relationship retains its `source_memory_id`; ending relationship validity does not remove that provenance reference.
- Reason is empty; provide a concrete human/Codex-readable reason and retry the mutation.
- The Mem Palace database cannot be opened or upgraded; stop the mutation path and restore or repair the database before retrying.
- FTS rebuild is unavailable on the local SQLite build; repair still reports the limitation, and operators recover by using a SQLite build with FTS support or leaving FTS unchanged.
- SQLite lock or timeout failures leave the database unchanged or partially backed up but not semantically reviewed. Recover by preserving the backup and run log, confirming no active writer owns the lock, then rerun the failed command.
- Delete and weight-update commands create a pre-mutation backup and append a run log. Recover from an incorrect approved mutation by restoring the backup outside the CLI, then recording the restoration evidence in the next maintenance run.
- Stale maintenance artifacts under `.tokimon/workflow/memory-maintenance/` may be removed only after the backup and run log needed for review are preserved.

## Acceptance Checks

1. Verify memory maintenance workflow tests:
   ```bash
   .venv/bin/python -m unittest discover -s tests/workflow/memory_maintenance -t . -p 'test_*.py'
   ```
   Expected outcome: tests pass for snapshot output, strict duplicate grouping, mutation safety, backup/run-log behavior, and repair checks.

2. Verify read-only and repair command surfaces:
   ```bash
   .venv/bin/python -m tokimon.workflow.memory_maintenance snapshot --json
   .venv/bin/python -m tokimon.workflow.memory_maintenance repair --json
   ```
   Expected outcome: `snapshot` emits records/stats/exact duplicate groups; `repair` emits before/after status and writes backup/run-log evidence only for repair behavior.

3. Ask an AI coding agent to verify the component contract:
   ```text
   Review docs/c4/level-3-components/workflow/memory-maintenance-cli.md, tokimon/workflow/memory_maintenance/, tests/workflow/memory_maintenance/, and docs/c4/level-3-components/memory/mem-palace.md.
   Confirm that the documented snapshot, delete, update-weight, repair, backup, run-log, pinned-memory safety, data ownership, failure modes, and tests match the implementation.
   Report any drift with file and line evidence.
   ```
   Expected outcome: the agent reports no contract drift, or lists precise mismatches with recommended owner documents or implementation files.

## References

- L2 Container: `docs/c4/level-2-containers/workflow/memory-maintenance.md`
- Memory Component: `docs/c4/level-3-components/memory/mem-palace.md`
