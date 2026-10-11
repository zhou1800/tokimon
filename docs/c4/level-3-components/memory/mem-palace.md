---
title: "Mem Palace - Components"
summary: "L3 component contract for the Mem Palace CLI, durable store, exchange, ingest, backup, and eval behavior."
read_when:
  - Changing Mem Palace CLI commands, Python APIs, persistence, or memory exchange.
  - Reviewing Mem Palace command contracts and failure modes.
---

# Mem Palace - Components

## Contents

- [Overview](#overview)
- [Components](#components)
- [Interfaces/Contracts](#interfacescontracts)
- [Operator Usage](#operator-usage)
- [Data Ownership](#data-ownership)
- [Failure Modes](#failure-modes)
- [Acceptance Checks](#acceptance-checks)
- [References](#references)

## Overview

Mem Palace is the local durable memory body for Tokimon.
At L3, this document owns the component responsibilities, command contracts, data ownership, failure behavior, and acceptance checks for durable local recall.
Its primary outcome is bounded, inspectable memory that helps agents load relevant context and preserve durable local decisions without using an external service.

## Components

### Command Interface

- **Responsibility:** Exposes the operator-facing command surface and Python facade for storing memories, retrieving bounded context, and maintaining temporal relationship facts.
- **Key surfaces:** bounded briefing, search, durable memory CRUD, temporal relationship operations, and delegation to exchange, ingestion, backup, and eval behavior.

### Exchange and Backup

- **Responsibility:** Creates portable, verifiable local snapshots and import/export streams without adding an external service dependency.
- **Key surfaces:** SQLite backup, JSONL export, manifest validation, duplicate handling, and single-transaction import/rollback behavior.

### Artifact Ingestion

- **Responsibility:** Converts explicitly named local text artifacts into bounded `artifact` memories.
- **Key surfaces:** explicit file/directory selection, ignored-runtime filtering, code-ingestion opt-in, path/hash metadata, and duplicate-resistant storage.

### Recall Evaluation

- **Responsibility:** Evaluates recall quality against small operator-owned fixtures before retrieval behavior changes.
- **Key surfaces:** query fixtures, expected/absent match checks, deterministic pass/fail reports, and evidence for future retrieval changes.
- **Retrieval change guardrail:** Add embeddings only after eval cases show lexical or FTS misses, and keep embeddings behind `MemPalace.recall(...)`.

### Local Durable Store

- **Responsibility:** Persists this machine's local durable memory rows and relationship facts.
- **Key surfaces:** durable memory records, relationship evidence, schema metadata, retrieval ranking inputs, and duplicate-resistant identity metadata.

## Interfaces/Contracts

Canonical command surfaces are exposed by `.venv/bin/python -m tokimon.memory.mem_palace.mem_palace --help` plus subcommand `--help`; JSON-producing commands own their machine-readable result contracts. This section summarizes the stable command, data, and artifact families.

**Commands**
- `briefing`
  - Reads `docs/guidelines/repository-guidelines.md`, essential local memory, and recent durable memory together.
  - Required before substantive Tokimon-enabled work unless the session or task already has a current briefing.
  - Tokimon callers pass `--scope tokimon` so unrelated local memory scopes are excluded.
  - Installed Tokimon callers pass `--project-root <current-task-repository-root> --include-scope global`; the adopting project is the task subject while the Tokimon checkout remains the command/runtime provider.
  - The generic CLI keeps `--scope` optional and defaults to an unscoped, cross-scope briefing for explicit general-purpose use.
- `recent`
  - Lists recent durable memory entries.
  - Supports `--scope` or `--project-root`, plus repeatable `--include-scope` for explicitly named additional scopes.
- `search <query>`
  - Searches durable memory with FTS/plain-text candidates plus deterministic reranking.
  - Supports `--scope` or `--project-root`, repeatable `--include-scope`, and `--room` and `--entity` filters.
- `add`
  - Stores a durable scoped memory entry.
  - Supports `--scope` or `--project-root`, plus `--room`, `--entities`, `--importance`, `--confidence`, `--metadata-json`, duplicate suppression, and local entity hint extraction.
  - Project-specific installed writes use `--project-root`; `--scope global` is reserved for genuinely cross-project memory.
  - Default duplicate suppression returns an existing row only when stable content identity and semantic attributes match; otherwise it reports a conflict without changing the row.
  - `--allow-duplicate` explicitly permits a second row with the same content identity.
- `show <id>`
  - Shows one durable memory entry by identifier.
- `link <subject> <predicate> <object>`
  - Stores a temporal entity relationship.
- `entity <name>`
  - Queries outgoing, incoming, or both directions for one entity, optionally `--as-of` a date.
- `invalidate <subject> <predicate> <object>`
  - Ends a current temporal relationship by setting `valid_to`.
- `timeline [entity]`
  - Shows chronological relationship history globally or for one entity.
- `stats`
  - Shows store counts, scope/room breakdowns, relationship types, and FTS availability.
- `backup`
  - Writes a consistent SQLite copy of the current store. The default destination is `.tokimon/memory/backups/`.
  - This command is a local SQLite snapshot utility. Host-level durability is outside Tokimon.
- `export --output <path>`
  - Exports memories, entities, and relationships as portable JSONL.
  - Writes a sibling `manifest.json` next to the JSONL.
  - The manifest contains `schema_version`, `source_instance_id`, `exported_at`, `streams`, `counts`, and `jsonl_sha256`.
- `import <path> [--no-manifest]`
  - Imports a Mem Palace JSONL export. Exact duplicate memories are skipped by default.
  - Requires the sibling manifest by default and validates manifest schema version, JSONL SHA-256, stream counts, and per-row schema before writing state.
  - Recomputes canonical memory content identity; a supplied `content_hash` must match, and an omitted legacy hash is populated before insertion.
  - A same-content semantic-attribute conflict fails the import transaction instead of silently discarding attributes.
  - `--keep-duplicate-memories` explicitly permits a second row with the same content identity.
  - Without `--replace`, an occupied exported memory integer identifier allocates a fresh destination identifier instead of skipping the imported memory.
  - Every inserted or exact-skipped memory contributes to a source-to-destination identifier map used to rewrite relationship `source_memory_id` provenance.
  - `--replace` keeps the exported identifier and replaces the destination row at that identifier.
  - `--no-manifest` is only for legacy JSONL exports and still validates every row schema.
  - Applies rows inside one SQLite transaction; any validation or apply error rolls back the import.
- `ingest <paths...>`
  - Ingests explicit text files/directories as scoped artifact memories.
  - Supports `--scope` or `--project-root`, plus `--room`, `--tags`, `--include-code`, file-size/detail limits, and duplicate control.
- `eval <cases_path>`
  - Evaluates recall against JSON or JSONL fixture cases.
- `migrate`
  - Runs schema initialization/migrations and reports resulting store stats.

**Python API**
- `MemPalace(db_path=..., c4_overview_path=...)`
- `MemPalace.remember(kind=..., summary=..., scope=..., room=...) -> MemoryRecord`
- `MemPalace.recall(query, scope=None, room=None, entity=None) -> list[SearchResult]`
- `MemPalace.recent(...) -> list[MemoryRecord]`
- `MemPalace.essential(...) -> list[MemoryRecord]`
- `MemPalace.add_fact(subject, predicate, object, valid_from=None, ...) -> str`
- `MemPalace.entity_facts(name, as_of=None, direction="both") -> list[RelationshipFact]`
- `MemPalace.timeline(entity=None) -> list[RelationshipFact]`
- `MemPalace.stats() -> dict`
- `MemPalace.backup(output_path=None) -> dict`
- `MemPalace.export_jsonl(output_path) -> dict`
- `MemPalace.import_jsonl(input_path, replace=False, skip_duplicate_memories=True, require_manifest=True) -> dict`
- `MemPalace.ingest_paths(paths, scope=..., room=...) -> dict`
- `MemPalace.evaluate_recall(cases_path) -> dict`

**Memory supersession**
- A successor memory may set `metadata.supersedes_memory_ids` to a list of existing older memory identifiers.
- `briefing`, `recent`, `essential`, and `search` omit effectively superseded rows by default. `show`/`get`, statistics, backup, export, and import retain the complete history.
- The reserved list is normalized to unique positive integer identifiers. Booleans, non-integers, nonpositive values, self-references, dangling references, and forward references are ineffective and cannot become active after a later insert.
- Effective supersession edges always point from a higher memory identifier to a lower one, preventing cycles. Retrieval also enforces that invariant for rows created outside the supported CLI/API boundary.
- Import processes memory rows in source-identifier order, rewrites supersession references through the complete source-to-destination memory map, and allocates a successor above all mapped predecessors when an occupied identifier would otherwise invert the edge.

**Durable memory schema**
- Stored fields:
  - `kind`
  - `summary`
  - `details`
  - `tags`
  - `source`
  - `scope`
  - `room`
  - `entities`
  - `content_hash`
  - `importance`
  - `confidence`
  - `metadata`
  - `pinned`
  - `created_at`
  - `updated_at`
  - `use_count`
  - `last_used_at`
- Allowed `kind` values:
  - `decision`
  - `preference`
  - `context`
  - `task`
  - `fact`
  - `note`
  - `event`
  - `milestone`
  - `problem`
  - `instruction`
  - `avoid`
  - `artifact`
- `avoid` records store durable anti-patterns or recurring mistakes to avoid, with details explaining the trigger, consequence, and preferred behavior.

**Duplicate identity**
- `content_hash` is the stable content identity derived from stored `kind`, whitespace-normalized `summary` and `details`, normalized `tags`, trimmed `source`, and lowercased `scope` and `room`.
- `pinned`, `entities`, `importance`, `confidence`, and `metadata` remain outside `content_hash` so content identity stays compatible with existing stores and exports.
- Duplicate suppression compares those semantic attributes separately: entities are whitespace-normalized and order-insensitive, metadata uses canonical JSON-object equality, and numeric/boolean values use their stored forms.
- If any semantic attribute differs and no exact attribute match exists for the content identity, add/import fails with the existing memory identifiers and conflicting field names; attribute values are not echoed.
- `created_at`, `updated_at`, `use_count`, and `last_used_at` are lifecycle metadata, not duplicate semantic attributes.
- `--allow-duplicate` and `--keep-duplicate-memories` are the explicit operator overrides for intentionally preserving same-content attribute variants.

**Temporal fact schema**
- Entity fields:
  - `id`
  - `name`
  - `kind`
  - `aliases`
  - `metadata`
- Relationship fields:
  - `subject_id`
  - `predicate`
  - `object_id`
  - `valid_from`
  - `valid_to`
  - `confidence`
  - `source_memory_id`
  - `source`

**Default paths**
- Durable memory DB: `.tokimon/memory/memory.db`
- repository guidelines: `docs/guidelines/repository-guidelines.md`

**Project-scope resolution**
- `--project-root` and base `--scope` are mutually exclusive.
- A Git project's resolved common Git directory is its local identity, so linked worktrees share one project scope.
- A non-Git project's resolved directory is its local identity.
- The live Tokimon repository and its linked worktrees map to the compatibility scope `tokimon`.
- Other projects map deterministically to `project:<slug>:<12-hex-digest>` without exposing the absolute path in the scope value.
- Invalid project roots fail rather than falling back to unscoped or Tokimon memory.
- Repeatable `--include-scope` adds only explicitly named scopes to retrieval; it never enables an implicit cross-scope fallback.

### Operator Usage

```bash
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace briefing --scope tokimon --limit 8
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace briefing --project-root /path/to/adopting-project --include-scope global --limit 8
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace recent --limit 8
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace search "memory system" --scope memory
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace add \
  --kind decision \
  --summary "Use Mem Palace for durable local Tokimon memory" \
  --details "Future sessions should review recent memory and search before answering history-dependent questions." \
  --tags memory,codex,tokimon \
  --scope memory \
  --room startup \
  --importance 5 \
  --source "local decision" \
  --pinned
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace link "Mem Palace" stores "local durable memory" --valid-from 2026-04-24
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace entity "Mem Palace"
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace stats
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace backup
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace export --output .tokimon/memory/export.jsonl
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace import .tokimon/memory/export.jsonl
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace import .tokimon/memory/legacy-export.jsonl --no-manifest
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace ingest docs/operations --scope docs --room operations
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace eval .tokimon/memory/recall-eval.jsonl
```

Python consumers should use the facade rather than table-level helpers:

```python
from tokimon.memory.mem_palace import MemPalace

with MemPalace() as palace:
    palace.remember(
        kind="event",
        summary="Example workflow wrote a run summary",
        scope="workflow",
        room="example-workflow",
        tags=["example-workflow", "workflow-run"],
    )
    results = palace.recall("run summary", scope="workflow", room="example-workflow")
```

## Data Ownership

- Mem Palace CLI owns the durable memory command interface.
- Mem Palace CLI owns the SQLite schema used by Tokimon local durable memory.
- Consumes read-only explicit operator commands, local memory rows, JSONL exchange files, ingestion targets, and recall fixtures.
- Derives bounded briefings, search results, portable exports, import manifests, artifact memories, and eval reports.
- Owns only local durable memory, not repository architecture facts.
- Does not store secrets, access tokens, passwords, cookies, or browser session state.

## Failure Modes

Mem Palace recovery preserves the local database first, then uses validation errors, import stages, backups, or explicit CLI commands to repair the named input without bypassing the CLI/API boundary.

- If SQLite FTS5 is unavailable, search falls back to plain text matching.
- Empty searches or invalid memory kinds return validation errors.
- Missing/non-directory project roots and simultaneous `--scope`/`--project-root` selection return validation errors without querying or writing memory.
- An empty memory store returns `No memories found.` rather than failing.
- Existing v1 memory databases are updated in place by adding new memory columns and relationship tables.
- Exact duplicate memory adds return an existing content-and-attribute match unless `--allow-duplicate` is used.
- Same-content adds with conflicting pinned state, entities, importance, confidence, or metadata fail with identifiers and field names instead of returning a row that would discard the requested attributes.
- Import requires a sibling manifest by default. Legacy JSONL requires `--no-manifest`.
- Import rejects unsupported manifest schema versions, JSONL hash mismatches, stream count mismatches, malformed records, and apply errors with a failed stage plus stream and record offset when available.
- Import validates canonical memory content hashes and populates omitted legacy hashes before insertion.
- Import skips exact content-and-attribute duplicates by default, rejects same-content attribute conflicts, and permits variants only with `--keep-duplicate-memories`.
- Non-replacing import remaps occupied exported memory identifiers to fresh destination identifiers and preserves relationship provenance plus memory supersession through the complete memory identifier map.
- Replacing import retains exported memory identifiers and replaces destination rows at those identifiers.
- Invalid, dangling, self, or forward `metadata.supersedes_memory_ids` entries are ignored at add/import time and do not hide a later unrelated memory.
- Import uses a single SQLite transaction so failed validation or apply leaves the destination store unchanged.
- Memory add/import rejects high-confidence secret-like content without echoing the value.
- Ingestion ignores unsupported, binary, empty, oversized, and runtime-state files instead of failing the full run.
- Ingestion skips high-confidence secret-like artifacts instead of storing their contents.
- Recall evaluation reports failed cases rather than mutating the memory store.
- A missing local memory database loads zero local entries.

## Acceptance Checks

1. Verify memory tests:
   ```bash
   .venv/bin/python -m unittest discover -s tests/memory/mem_palace -t . -p 'test_*.py'
   ```
   Expected outcome: tests pass for CLI command surface, durable store behavior, scoped recall, and exchange behavior.

2. Verify the startup briefing manually:
   ```bash
   .venv/bin/python -m tokimon.memory.mem_palace.mem_palace briefing --scope tokimon --limit 8
   .venv/bin/python -m tokimon.memory.mem_palace.mem_palace stats --json
   .venv/bin/python -m tokimon.memory.mem_palace.mem_palace export --output .tokimon/tmp/mem-palace.jsonl --json
   ```
   Expected outcome: briefing output shows the repository guidelines first, then essential and recent durable Mem Palace entries scoped to `tokimon`, with unrelated local scopes excluded; stats output remains inspectable.
   Export output includes `manifest_path`, `jsonl_sha256`, and per-stream counts, and writes `.tokimon/tmp/manifest.json`.

3. Ask an AI coding agent to verify the component contract:
   ```text
   Review docs/c4/level-3-components/memory/mem-palace.md, tokimon/memory/mem_palace/, and tests/memory/mem_palace/.
   Confirm that the documented command surface, briefing and recall behavior, exchange/import/export behavior, ingestion boundary, durable store ownership, failure modes, and tests match the implementation.
   Report any drift with file and line evidence.
   ```
   Expected outcome: the agent reports no contract drift, or lists precise mismatches with recommended owner documents or implementation files.

## References

- Repository Guidelines: `docs/guidelines/repository-guidelines.md`
- L1 Context: `docs/c4/level-1-context/memory-context.md`
- L2 Containers: `docs/c4/level-2-containers/memory/mem-palace.md`
