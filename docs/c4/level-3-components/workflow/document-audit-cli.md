---
title: "Document Audit - CLI"
summary: "L3 command contract for document audit rules, audit units, checklist matrices, clean-unit rechecks, report coverage, and finding recording."
read_when:
  - Changing document-audit CLI commands, JSON output, artifact paths, report schema, finding schema, matrix behavior, or ledger behavior.
  - Reviewing document-audit workflow verification and command behavior.
---

# Document Audit - CLI

## Contents

- [Overview](#overview)
- [Components](#components)
- [Workflow Package Declarations](#workflow-package-declarations)
- [Interfaces/Contracts](#interfacescontracts)
- [Project Selection](#project-selection)
- [Checklist Matrix Schema](#checklist-matrix-schema)
- [Audit Unit and Lock Schemas](#audit-unit-and-lock-schemas)
- [Structured Report Schema](#structured-report-schema)
- [Finding Schema](#finding-schema)
- [Ledger Contract](#ledger-contract)
- [Data Ownership](#data-ownership)
- [Safety Rules](#safety-rules)
- [Failure Modes](#failure-modes)
- [Acceptance Checks](#acceptance-checks)
- [References](#references)

## Overview

The document audit CLI is the deterministic support surface for Tokimon documentation audits.
It extracts live guideline rules, writes complete checklist matrices, persists support-only clean and settled unit states, validates targeted recheck plans, and converts fresh plus reused coverage into support-only row evaluations.
It summarizes the current ledger, including missing-document health signals, and records auditor-accepted findings with exact evidence.

## Components

- **Rule extractor:** Owns guideline-owned rule ID, source path, source line, description, and hash extraction.
- **Checklist matrix writer:** Owns complete document/rule evaluation rows, status counts, CSV output, and summary artifacts.
- **Clean-unit planner:** Owns audit-unit definitions, clean-unit lock signatures, invalidation reasons, and targeted recheck plans.
- **Report coverage verifier:** Owns exact fresh-report checks, validated reusable coverage, waiting-row exclusion, and final-completion signaling.
- **Report compiler:** Owns conversion from structured row reports into support-only row evaluation artifacts.
- **Current ledger reader/writer:** Owns compact latest-state finding records, fingerprint updates, and read-only missing-document health signals.
- **Run artifact writer:** Owns audit-unit, matrix, summary, CSV, structured report, clean-unit lock, recheck-plan, support evaluation, and accepted ledger artifacts.
- **Renderers:** Own human-readable and JSON command output.

Project selection and preserved-review validation share their implementation with Code Audit.

## Interfaces/Contracts

### Workflow Package Declarations

- **Workflow id:** `document-audit`
- **Skill id:** `document-audit`
- **Consumer skills:** `document-audit`, `document-audit-fix`
- **Python package:** `tokimon/workflow/document_audit/`
- **Canonical code entrypoint:** `tokimon.workflow.document_audit.cli:main`
- **Operator entrypoint:** `.venv/bin/python -m tokimon.workflow.document_audit`
- **Contract declarations:** `tokimon/workflow/document_audit/contract.py`
- **Workflow readiness declaration:** `tokimon/workflow/document_audit/contract.py` `WORKFLOW_READINESS`
- **Default local state root:** `.tokimon/workflow/document-audit/projects/<project-id>/`
- **Default run root:** `.tokimon/workflow/document-audit/projects/<project-id>/runs/`
- **Current ledger:** `.tokimon/workflow/document-audit/projects/<project-id>/ledger/current.json`
- **Report directory name:** `reports`
- **Compiled evaluations artifact:** `evaluations-from-reports.json`
- **Reports summary artifact:** `reports-summary.json`
- **Audit units artifact:** `audit-units.json`
- **Clean-unit locks artifact:** `clean-unit-locks.json`
- **Recheck plan artifact:** `recheck-plan.json`
- **Guideline rule sources:** `docs/guidelines/document-guidelines.md`, `docs/guidelines/c4-document-guidelines.md`
- **Required tool capability:** none

Canonical machine-readable workflow contract: `.venv/bin/python -m tokimon.workflow.document_audit contract --json`. This section summarizes the contract and the structured artifacts it names.
Commands emit human-readable text by default and JSON with `--json`.

Entrypoint: `.venv/bin/python -m tokimon.workflow.document_audit`.

Persisted artifacts use UTF-8 and preserve non-ASCII text.
All audits use `docs/guidelines/document-guidelines.md`; C4 audits also use `docs/guidelines/c4-document-guidelines.md`.
Split mixed target sets by source set.

**Commands**
- `contract`
  - Emit workflow contract declarations.
- `rules`
  - Extract live guideline rule IDs.
- `matrix`
  - Write audit-unit definitions and the full rule/document checklist matrix.
- `clean-unit-locks`
  - Persist support-only clean and accepted terminal-disposition states, plus dependency waits.
- `recheck-plan`
  - Validate prior unit states and identify clean, settled, waiting, and fresh-required units and rows.
- `verify-reports`
  - Verify every currently fresh-required row has exactly one report and reject reports for non-fresh rows.
- `compile-reports`
  - Compile structured reports into support-only evaluations.
- `ledger-summary`
  - Summarize the compact current ledger.
  - Report missing document paths, including the subset still marked `open`, without mutating entries.
- `record-findings`
  - Merge auditor-accepted findings into the current ledger and write one run artifact.

**`contract` arguments**

**`rules` arguments**
- `--source <path>`
  - Optional guideline source path; may repeat.
  - Defaults to both general and C4 document guidelines.
- `--source-ref <git-ref>`
  - Optional committed guideline snapshot.

**`matrix` arguments**
- `--scope-id <id>`
  - Required stable audit scope id.
- `--run-id <id>`
  - Optional audit run id.
  - Defaults to `document-audit-<UTC timestamp>`.
- `--run-root <path>`
  - Optional run artifact root.
  - Defaults to the selected project's run root.
- `--source <path>`
  - Optional guideline source path; may repeat.
- `--source-ref <git-ref>`
  - Optional committed guideline snapshot.
- `--document <path>`
  - Document under audit; may repeat.
- `--document-glob <glob>`
  - Document glob under audit; may repeat.
- `--documents-from <path>`
  - Text or JSON file of document paths.
- `--units-from <path>`
  - Optional JSON audit-unit definitions.
  - Defaults to one audit unit per document.
- `--evaluations <path>`
  - Optional evaluated row JSON keyed by `file_path` and `rule_id`.
- `--artifact-prefix <name>`
  - Optional filename prefix.
  - Defaults to `checklist-matrix`.
  - The default matrix artifacts are `checklist-matrix.json`, `checklist-matrix.csv`, and `checklist-summary.json`.
- `--expect-rule-count <n>`, `--expect-document-count <n>`, `--expect-matrix-count <n>`
  - Optional hard count guards.
- `--require-complete`
  - Fail when any row remains `unchecked`.

**`verify-reports` arguments**
- `--matrix <path>`
  - Required checklist matrix JSON path.
- `--reports-dir <path>`
  - Required directory containing report JSON files.
- `--recheck-plan <path>`
  - Optional validated recheck plan whose unchanged locked rows count as coverage.
- `--require-complete`
  - Exit non-zero when any currently required report is missing, invalid, duplicate, extra, or supplied for a non-fresh row.
  - A successful current wave can still have waiting rows; `final_coverage_complete` is false until none remain.

**`compile-reports` arguments**
- `--matrix <path>`
  - Required checklist matrix JSON path.
- `--reports-dir <path>`
  - Required directory containing report JSON files.
- `--output-dir <path>`
  - Optional output directory.
  - Defaults to the matrix file's directory.
- `--recheck-plan <path>`
  - Optional validated recheck plan used to combine locked evaluations with fresh reports.
- `--require-complete`
  - Exit non-zero without compiling when current-wave report coverage is incomplete.

**`clean-unit-locks` arguments**
- `--matrix <path>`
  - Required full target matrix.
- `--reports-dir <path>`
  - Required directory containing fresh reports.
- `--output-dir <path>`
  - Required destination for `clean-unit-locks.json`.
- `--ledger <path>`
  - Optional current ledger path used to reject locks for units with open findings.
- `--recheck-plan <path>`
  - Optional plan whose validated locked rows complete the current coverage set.

**`recheck-plan` arguments**
- `--matrix <path>`
  - Required current full target matrix.
- `--prior-clean-unit-locks <path>`
  - Required prior `clean-unit-locks.json` path.
- `--output-dir <path>`
  - Required destination for `recheck-plan.json`.
- `--ledger <path>`
  - Optional current ledger path used to invalidate units with open findings.
- `--activate-unit <id>`
  - Optional explicit unit invalidation; may repeat.

**`ledger-summary` arguments**
- `--ledger <path>`
  - Optional current ledger path.
  - Defaults to the selected project's current ledger.
- `--scope-id <id>`
  - Optional stable audit scope filter.

**`record-findings` arguments**
- `--accepted-findings <path>`
  - Required auditor-owned JSON file containing either a finding list or an object with `findings`.
  - This must not be a `compile-reports` support artifact.
  - Inputs with `audit_authority: false` are rejected.
- `--scope-id <id>`
  - Required stable audit scope id.
- `--run-id <id>`
  - Optional audit run id.
  - Defaults to `document-audit-<UTC timestamp>`.
  - Use the source report run id when recording dispositions that will create a new settled-unit state.
- `--ledger <path>`
  - Optional current ledger path.
- `--run-root <path>`
  - Optional run artifact root.
- `--source <path>`
  - Optional guideline source path for rule validation; may repeat.
  - Use the same sources loaded by the audit.
- `--source-ref <git-ref>`
  - Optional committed guideline snapshot.
- `--allow-unknown-rules`
  - Bypass live-rule validation for migration or emergency repair.

### Project Selection

Every command accepts `--project-root <directory>` before or after its command name.
New audits default to the caller's Git checkout root, or the caller's directory outside Git.
Rules, documents, globs, bound inputs, document paths inside supplied manifests, and Git refs resolve against that project.
Tokimon code and persistent runtime remain in the Tokimon checkout and its primary worktree runtime.
Default state is partitioned by a project ID derived from the primary checkout path; linked worktrees share a ledger while matrices retain their exact audited checkout.
Explicit runtime paths remain supported; project-bound ledgers reject a different project even at an explicitly shared path.
`contract --json` returns the selected project and resolved default paths.

Schema 3 matrices store `project_id` and `project_root` in `project`, covered by their input snapshot.
Commands consuming a matrix use its recorded project when the flag is omitted and reject a conflicting explicit checkout.
Relative artifact and manifest arguments resolve from the caller's working directory before selecting the recorded project.
Returned artifact paths are absolute so another working directory can consume them unchanged.
Locks bind the project and checkout, so another project's or checkout's state cannot supply coverage.
Legacy matrices and reports remain readable for historical verification but cannot supply current-schema reports or reusable locks.
Legacy unbound ledgers are readable; recording into a non-empty unbound ledger requires a prior explicit migration that preserves source records and binds their verified project.

### Checklist Matrix Schema

Each row records one document/rule pair:
`audit_unit_id`, `file_path`, `rule_id`, `status`, `rule_source_path`, `rule_line`, `rule_hash`, and `rule_description`.

Schema version 3 also records `project`, `run_id`, `scope_id`, `audit_snapshot_id`, the resolved guideline source ref, and content identities for guideline sources, documents, and bound inputs.
The snapshot ID covers these inputs, normalized audit units, and applicable rule rows.
Commands that consume a current matrix reject it if any captured live input changed; regenerate the matrix before accepting new reports.
When a completed matrix is rebuilt from compiled evaluations, their `compiled_for_run_id`, `compiled_for_scope_id`, and `compiled_for_audit_snapshot_id` must match that matrix.

`status` is one of `pass`, `fail`, `not_applicable`, or `unchecked`.
Optional evaluation fields are `summary`, `severity`, `finding_status`, `reason`, `notes`, `recommended_action`, `verification`, and `evidence`.
The summary artifact records counts, completion status, and artifact paths.
Completed audit reports must use `--require-complete`.

### Audit Unit and Lock Schemas

Each audit unit records a stable `audit_unit_id`, one or more `document_paths`, and optional `bound_input_paths` for owner or dependency documents that affect the unit's audit result.
Every matrix document belongs to exactly one unit.
The matrix command writes normalized definitions to `audit-units.json`.

A clean-unit state binds the unit definition, current document and bound-input content hashes, applicable matrix row rule hashes, guideline snapshots, and hashes and provenance of its source reports.
A unit is `clean` only when all rows are `pass` or `not_applicable` and no open ledger finding in the audit scope names one of its documents.
A unit is `settled` when every failed report finding has an exact matching `deferred`, `rejected`, or `blocked` ledger entry; settled state is reusable scheduling state but is not a clean audit result.
Creating a settled state from fresh reports requires those terminal dispositions to have been accepted under the source report run id, so an older terminal decision cannot silently settle a newly invalidated audit result.
Re-recording the same terminal disposition may update ledger timestamps or counters without invalidating the settled state; a changed finding identity or terminal status does invalidate it.
A unit is `waiting_on_dependency` when it depends on another active unit outside its mutual dependency group. It receives no report until the blocking unit is clean or settled, after which the dependent becomes fresh-required once against stable upstream content.
Mutually dependent active units are treated as one dependency wave so a cycle cannot leave every unit waiting.
`clean-unit-locks.json` and `recheck-plan.json` use `audit_authority: false`.

The recheck plan revalidates every bound value and preserves valid clean or settled states independently.
A changed or explicitly activated unit requires fresh reports for all of its rows; explicit activation applies only to the named unit, while dependents follow bound-input invalidation and waiting rules.
`verify-reports`, `compile-reports`, and `clean-unit-locks` recompute plan validity before accepting reused coverage.
`complete` means the reports required by the current plan are complete. A full target is finished only when `final_coverage_complete` is true and the plan has zero active and waiting units.

### Structured Report Schema

Each report records one completed audit for one checklist row:

- `schema_version`: current report schema version 3.
- `workflow_id`: `document-audit`.
- `run_id`, `scope_id`, and `audit_snapshot_id`: exact provenance values from the owning matrix.
- `file_path`: exact path from the matrix row.
- `rule_id`: exact rule ID from the matrix row.
- `status`: one of `pass`, `fail`, or `not_applicable`.
- `summary`: concise result for the row.
- `reason`: audit rationale for the classification.
- `review`: `reviewer_id` in `subagent:<id>` form naming the assigned subagent, plus `source_path` and `source_sha256` identifying its preserved JSON result.
- `findings`: required non-empty list for `fail`; omitted or empty for `pass` and `not_applicable`.

The review source is an object with the same `reviewer_id` and a `reports` list containing the exact row result before its `review` pointer was attached.
One source may cover a document group; source paths are absolute or relative to the row report's directory.
Validation checks the source bytes, reviewer, unique row, and complete row result, and preserves this reference through compilation and lock reuse.
A metadata declaration alone does not establish actual reviewer work; the agent preserves the assigned subagent's returned result and keeps preliminary sweeps outside `reports/`.

Each finding inside a report uses the Finding Schema below and must match the report's `file_path` and `rule_id`.
`verify-reports --require-complete` rejects missing, invalid, duplicate, extra, stale-provenance, and unexpected non-fresh reports after validated clean and settled coverage is applied.
`compile-reports --require-complete` writes support-only `evaluations-from-reports.json` and `reports-summary.json` after current-wave coverage is complete; `final_coverage_complete` remains false while dependency rows wait.
Compiled evaluations retain source-report provenance and separately name the current matrix snapshot for which lock and report coverage was validated.
Those artifacts include `audit_authority: false`.
Compile output is not evidence that the document audit is clean.

### Finding Schema

Each recorded finding must include:

- `rule_id`: live guideline rule ID.
- `file_path`: repo-relative path under audit.
- `summary`: stable finding claim.
- `severity`: one of `must_fix`, `should_fix`, or `optional`.
- `status`: one of `open`, `fixed`, `deferred`, `rejected`, `blocked`, or `stale`.
- `evidence`: non-empty list of exact evidence objects.
  Each evidence object must include a positive `line` or `start_line`, plus one non-empty `text`, `excerpt`, `quote`, or `details` field.

Optional fields include `fingerprint`, `normalized_subject`, `recommended_action`, `resolution`, and `rule_hash`.
When `fingerprint` is omitted, the workflow computes one from `scope_id + rule_id + file_path + normalized_subject`.
Line numbers are not part of the fingerprint because they drift during doc edits.

### Ledger Contract

The current ledger stores one latest-state entry per fingerprint.
Schema version 3 binds the ledger to `project_id`; versions 1 and 2 remain readable as historical ledgers.
Migration preserves original dispositions, timestamps, counters, fingerprints, and artifact provenance in source snapshots; it does not manufacture reviewer metadata, snapshots, or clean locks.
Current entries may normalize historical checkout paths while retaining `migration_sources` pointing to the unchanged source entries and run artifacts.
It keeps compact fields needed by the next run: fingerprint, scope id, rule id, file path, normalized subject, current severity and status, first seen run, last seen run, seen count, current summary, optional recommended action, optional resolution, last rule hash, and last run artifact pointer.

Full evidence stays in per-run artifacts.
Future runs read only the current ledger by default; old run artifacts are for debugging and traceability.
`ledger-summary` derives `missing_file_count`, `open_missing_file_count`, and affected entry details from the current checkout.
The summary is healthy only when no open entry points to a missing document; health reporting never changes an accepted finding disposition.

## Data Ownership

- Owns project-isolated document-audit artifacts under `.tokimon/workflow/document-audit/projects/<project-id>/`.
- Owns audit-unit definitions, run matrices, matrix summaries, matrix CSVs, structured reports, clean-unit locks, recheck plans, support evaluations, report summaries, accepted per-run findings, and the compact current ledger.
- Consumes read-only tracked documentation, guideline rule sources, optional committed source refs, and evaluator-produced row/finding JSON as inputs.
- Derives rule inventories, complete document/rule checklist matrices, status counts, finding fingerprints, and current ledger summaries.
- Does not own the audited documents themselves; `document-audit-fix` owns bounded documentation edits when the user explicitly requests remediation.

## Safety Rules

- The CLI writes only through these commands:
  - `matrix`
  - `clean-unit-locks`
  - `recheck-plan`
  - `compile-reports`
  - `record-findings`
- `matrix` writes only under the configured run root.
- `clean-unit-locks` and `recheck-plan` write support-only artifacts under their configured output directories.
- `compile-reports` writes only support artifacts under the configured output directory or the matrix file's directory.
- `record-findings` writes one accepted finding run artifact and the configured current ledger.
- `contract`, `rules`, `verify-reports`, and `ledger-summary` are read-only.
- The CLI does not read secrets, cookies, API keys, browser sessions, MCP configs, or provider credentials.
- The CLI does not call external network services.

## Failure Modes

- A selected project conflicts with the matrix or ledger; select the recorded checkout or create a new audit for the intended project.
- A review source is missing, changed, or disagrees with its row; recover the original subagent result or obtain a fresh review.

- Guideline rule IDs are duplicated or the requested source ref cannot be read; fix the rule source selection before creating a matrix.
- The matrix snapshot is stale or a report names a different run, scope, or snapshot; regenerate the matrix and audit only its fresh-required rows.
- Matrix document scope is empty or expected counts do not match; correct the document list, glob, or count guards and regenerate the matrix.
- Report coverage is incomplete, invalid, duplicated, or outside the checklist; repair only the reported gaps and rerun `verify-reports`.
- A clean-unit lock is missing, stale, tampered, or names a unit with an open finding; audit only the affected unit rows identified by `recheck-plan`.
- The whole lock artifact is invalid or unavailable; the recheck plan requires fresh reports for the full matrix rather than trusting partial state.
- Report-support compilation or matrix evaluation is incomplete under `--require-complete`; finish every row before producing support artifacts.
- Finding rows omit exact evidence; add file and line evidence before recording them.
- Accepted finding input is marked `audit_authority: false`; use an auditor-owned disposition file instead of support output.
- Finding rows use an unknown rule ID; fix the rule-source mismatch or use `--allow-unknown-rules` only for an intentional migration/emergency repair.
- Duplicate accepted fingerprints or ownership-mismatched supplied fingerprints make `record-findings` reject the batch before any run or ledger write; correct the input and retry.
- Ledger schema is unsupported or run artifact paths cannot be written; inspect the configured run root and ledger path before retrying.
- Open ledger entries point to documents that no longer exist; use the reported fingerprints to re-audit and explicitly mark them `fixed`, `rejected`, or `stale` instead of treating the ledger as current.
- Interrupted `matrix`, `compile-reports`, or `record-findings` writes leave local run artifacts as recovery evidence. Recover by preserving the partial run directory, rerunning the command with the same explicit run root when idempotent, or generating a fresh run id if the artifact set cannot be trusted.
- Reusing a reports directory across matrix snapshots produces stale or unexpected coverage. Preserve the prior directory as lock evidence, use a new empty reports directory for the current snapshot, and rerun only fresh-required rows.

## Acceptance Checks

```bash
.venv/bin/python -m unittest discover -s tests/workflow/document_audit -t . -p 'test_*.py'
```

```bash
.venv/bin/python -m tokimon.workflow.document_audit contract --json
```

```bash
.venv/bin/python -m tokimon.workflow.document_audit rules --source docs/guidelines/document-guidelines.md --json
```

```bash
.venv/bin/python -m tokimon.workflow.document_audit rules \
  --source docs/guidelines/document-guidelines.md \
  --source docs/guidelines/c4-document-guidelines.md \
  --json
```

```bash
.venv/bin/python -m tokimon.workflow.document_audit matrix \
  --scope-id docs-c4-l1-context \
  --run-id document-audit-smoke \
  --document-glob 'docs/c4/level-1-context/*.md' \
  --run-root .tokimon/tmp/document-audit-smoke \
  --json
```

```bash
.venv/bin/python -m tokimon.workflow.document_audit ledger-summary --json
```

Expected outcome: tests pass and support commands emit the documented JSON shape.
The smoke matrix writes JSON, CSV, and summary artifacts under the supplied local run root.
Structured-report smoke checks should also be covered by workflow tests:
`verify-reports --require-complete` rejects incomplete current-wave coverage, and `compile-reports --require-complete` writes support evaluations plus a summary only for complete current-wave coverage.
Clean-unit workflow tests must also prove that unchanged clean and settled states replace fresh reports, active dependencies make downstream units wait, stable upstream content activates them once, stale snapshots and tampered source reports fail closed, and unexpected reports for non-fresh rows are rejected.

AI coding agent verification prompt:

```text
Review docs/c4/level-3-components/workflow/document-audit-cli.md and tokimon/workflow/document_audit/.
Confirm that documented commands, default paths, audit-unit and clean-unit lock schemas, recheck behavior, structured report schema, finding schema, fingerprint inputs, ledger compaction behavior, and tests match implementation.
Confirm that `compile-reports` output is not documented or shaped as authoritative audit results.
Report any drift with file and line evidence.
```

Expected result: no drift between the Document Audit workflow contract, CLI behavior, ledger behavior, and tests.

## References

- L2 Containers: `docs/c4/level-2-containers/workflow/document-audit.md`
- L1 Context: `docs/c4/level-1-context/workflow-context.md`
- Guidelines: `docs/guidelines/document-guidelines.md`
- Guidelines: `docs/guidelines/c4-document-guidelines.md`
