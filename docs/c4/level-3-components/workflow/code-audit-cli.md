---
title: "Code Audit - CLI"
summary: "L3 command contract for code and unit-test guideline rule extraction, subject validation, checklist matrices, pass-lock rechecks, structured report coverage, current ledger summaries, and finding recording."
read_when:
  - Changing code-audit CLI commands, JSON output, artifact paths, subject schema, report schema, evaluation schema, matrix behavior, or ledger behavior.
  - Reviewing code-audit workflow verification and command behavior.
---

# Code Audit - CLI

## Contents

- [Overview](#overview)
- [Components](#components)
- [Workflow Package Declarations](#workflow-package-declarations)
- [Interfaces/Contracts](#interfacescontracts)
- [Subject Schema](#subject-schema)
- [Checklist Matrix Schema](#checklist-matrix-schema)
- [Audit Report Schema](#audit-report-schema)
- [Evaluation Rules](#evaluation-rules)
- [Ledger Contract](#ledger-contract)
- [Data Ownership](#data-ownership)
- [Safety Rules](#safety-rules)
- [Failure Modes](#failure-modes)
- [Acceptance Checks](#acceptance-checks)
- [References](#references)

## Overview

The code audit CLI is the deterministic support surface for Tokimon implementation audits.
It extracts live code and unit-test guideline rules, validates explicit audit subjects, writes complete checklist matrices, validates pass-lock recheck plans, verifies structured report coverage and provenance, compiles reports into support-only row evaluations, records auditor-accepted current row dispositions, and summarizes current ledger health.

## Components

- **Rule extractor:** Owns code and unit-test guideline rule ID, source path, source line, description, and hash extraction.
- **Subject validator:** Owns caller-provided audit subject normalization and path validation.
- **Checklist matrix writer:** Owns complete subject/rule evaluation rows, frozen project and input snapshots, status counts, CSV output, and summary artifacts.
- **Pass-lock planner:** Owns support-only pass-lock artifacts and lock-aware recheck plans.
- **Report coverage verifier:** Owns exactly-one-report coverage for fresh-required rows, validated locked coverage, and matrix `run_id`/`scope_id` provenance checks.
- **Report compiler:** Owns conversion from structured row reports into support-only row evaluation artifacts.
- **Current ledger reader/writer:** Owns compact latest-state finding records, fingerprint updates, row-level closure, and health signals.
- **Run artifact writer:** Owns subject, matrix, summary, CSV, structured report, support evaluation, and accepted ledger artifacts.
- **Renderers:** Own human-readable and JSON command output.

## Interfaces/Contracts

### Workflow Package Declarations

- **Workflow id:** `code-audit`
- **Skill id:** `code-audit`
- **Consumer skills:** `code-audit`, `code-audit-fix`
- **Python package:** `tokimon/workflow/code_audit/`
- **Canonical code entrypoint:** `tokimon.workflow.code_audit.cli:main`
- **Operator entrypoint:** `.venv/bin/python -m tokimon.workflow.code_audit`
- **Contract declarations:** `tokimon/workflow/code_audit/contract.py`
- **Workflow readiness declaration:** `tokimon/workflow/code_audit/contract.py` `WORKFLOW_READINESS`
- **Default local state root:** `.tokimon/workflow/code-audit/projects/<project-id>/`
- **Default run root:** `.tokimon/workflow/code-audit/projects/<project-id>/runs/`
- **Current ledger:** `.tokimon/workflow/code-audit/projects/<project-id>/ledger/current.json`
- **Report directory name:** `reports`
- **Compiled evaluations artifact:** `evaluations-from-reports.json`
- **Reports summary artifact:** `reports-summary.json`
- **Pass-lock artifact:** `pass-locks.json`
- **Recheck-plan artifact:** `recheck-plan.json`
- **Guideline rule sources:**
  - `docs/guidelines/code-guidelines.md`
  - `docs/guidelines/unit-testing-guidelines.md`
- **Required tool capability:** none

Canonical machine-readable workflow contract: `.venv/bin/python -m tokimon.workflow.code_audit contract --json`. This section summarizes the contract and the structured artifacts it names.
Commands emit human-readable text by default and JSON with `--json`.

`--project-root <checkout>` is accepted before or after the command and defaults to the caller's repository or directory.
Guideline sources and subject boundary paths resolve in that project; supplied artifact and manifest arguments resolve from the caller's working directory.
Matrix consumers use the recorded checkout unless an explicit project was supplied; a conflicting checkout is rejected.
Default state lives under `.tokimon/workflow/code-audit/projects/<project-id>/` in Tokimon's primary worktree.
The project ID derives from the audited repository's primary checkout, so linked worktrees share the ledger while matrices and reusable results bind the specific checkout.
Project selection and preserved-review validation share their implementation with Document Audit.

Entrypoint: `.venv/bin/python -m tokimon.workflow.code_audit`.

Persisted artifacts use UTF-8 and preserve non-ASCII text.
Audits use `docs/guidelines/code-guidelines.md` and `docs/guidelines/unit-testing-guidelines.md` by default.

**Commands**
- `contract`
  - Emit workflow contract declarations.
- `rules`
  - Extract live guideline rule IDs.
- `subjects`
  - Validate and normalize caller-provided audit subjects.
- `matrix`
  - Write the full rule/subject checklist matrix.
- `pass-locks`
  - Write support-only pass-locks for completed `pass` and `not_applicable` rows.
- `recheck-plan`
  - Write a support-only lock-aware target recheck plan from current matrix rows and prior pass-locks.
- `verify-reports`
  - Verify one structured report exists for every fresh-required checklist row and optionally count validated pass-locks.
- `compile-reports`
  - Compile structured reports and optional validated pass-locks into support-only evaluated matrix rows.
  - Emit `pass-locks.json` only for complete coverage; incomplete compiles produce no usable locks.
- `record-findings`
  - Record auditor-accepted evaluated rows as current dispositions.
  - `fail` and `needs_human_review` open or update findings.
  - `pass` resolves prior open findings for the same scope/rule/subject row as `fixed`; `not_applicable` resolves them as `stale`.
- `ledger-summary`
  - Summarize the current code-audit ledger and report missing evidence paths as health signals.

**`contract` arguments**

**`rules` arguments**
- `--source <path>`
  - Optional guideline source path; may repeat.
  - Defaults to the audited project's code and unit-testing guideline docs.

**`subjects` arguments**
- `--subjects-from <path>`
  - Required JSON file containing audit subjects.
- `--subject <id>`
  - Optional audit subject id filter; may repeat.

**`matrix` arguments**
- `--scope-id <id>`
  - Required stable audit scope id.
- `--run-id <id>`
  - Optional audit run id.
  - Defaults to `code-audit-<UTC timestamp>`.
  - When `--evaluations` is supplied and `--run-id` is omitted, the one shared evaluation `run_id` is used.
- `--run-root <path>`
  - Optional run artifact root.
  - Defaults to `.tokimon/workflow/code-audit/projects/<project-id>/runs/`.
- `--source <path>`
  - Optional guideline source path; may repeat.
- `--subjects-from <path>`
  - Required JSON file containing audit subjects.
- `--subject <id>`
  - Optional audit subject id filter; may repeat.
- `--evaluations <path>`
  - Optional evaluated row JSON keyed by `audit_subject_id` and `rule_id`.
- `--artifact-prefix <name>`
  - Optional filename prefix.
  - Defaults to `checklist-matrix`.
- `--expect-rule-count <n>`, `--expect-subject-count <n>`, `--expect-matrix-count <n>`
  - Optional hard count guards.
- `--require-complete`
  - Fail when any row remains `unchecked`.

**`verify-reports` arguments**
- `--matrix <path>`
  - Required checklist matrix JSON path.
- `--reports-dir <path>`
  - Required directory containing report JSON files.
- `--recheck-plan <path>`
  - Optional recheck-plan JSON path.
  - When present, validated locked `pass` and `not_applicable` rows count as coverage.
  - If the plan cannot be validated, locked rows do not count and complete fresh reports remain required.
- `--require-complete`
  - Exit non-zero when any report is missing, invalid, duplicated, outside the matrix, or supplied for a validated locked row.

**`pass-locks` arguments**
- `--matrix <path>`
  - Required checklist matrix JSON path.
- `--reports-dir <path>`
  - Required directory containing report JSON files.
- `--output-dir <path>`
  - Required output directory for `pass-locks.json`.
- `--recheck-plan <path>`
  - Optional recheck-plan JSON path for lock-aware completed coverage.

**`recheck-plan` arguments**
- `--matrix <path>`
  - Required current checklist matrix JSON path.
- `--prior-pass-locks <path>`
  - Required prior `pass-locks.json` path.
- `--output-dir <path>`
  - Required output directory for `recheck-plan.json`.

**`compile-reports` arguments**
- `--matrix <path>`
  - Required checklist matrix JSON path.
- `--reports-dir <path>`
  - Required directory containing report JSON files.
- `--recheck-plan <path>`
  - Optional recheck-plan JSON path.
  - Validated locked `pass` and `not_applicable` rows may be compiled as support-only evaluations with `evaluation_source: validated_prior_pass_lock`.
- `--output-dir <path>`
  - Optional output directory.
  - Defaults to the matrix file's directory.
- `--require-complete`
  - Exit non-zero without compiling when fresh report plus validated lock coverage is incomplete.
  - Complete coverage is required for any emitted pass-lock to be usable by a later recheck.

**`record-findings` arguments**
- `--accepted-evaluations <path>`
  - Required auditor-owned JSON file containing accepted evaluated matrix rows.
  - This must not be raw `compile-reports` support output.
  - Inputs with `audit_authority: false` are rejected.
- `--scope-id <id>`
  - Required stable audit scope id.
- `--run-id <id>`
  - Optional audit run id.
  - When omitted, the one shared accepted-evaluation `run_id` is used; empty inputs fall back to `code-audit-<UTC timestamp>`.
- `--ledger <path>`
  - Optional current ledger path.
- `--run-root <path>`
  - Optional run artifact root.
- `--source <path>`
  - Optional guideline source path for rule validation; may repeat.
  - Use the same sources loaded by the audit.
- `--allow-unknown-rules`
  - Bypass live-rule validation for migration or emergency repair.

**`ledger-summary` arguments**
- `--ledger <path>`
  - Optional current ledger path.
  - Defaults to `.tokimon/workflow/code-audit/projects/<project-id>/ledger/current.json`.
- `--scope-id <id>`
  - Optional stable audit scope filter.

### Subject Schema

An audit subject is a bounded implementation or behavior surface.
Each subject requires `audit_subject_id`, `title`, and at least one boundary field:
`implementation_paths`, `test_paths`, `owning_specs`, `verification_commands`, `dependency_edges`, or `notes`.
When an owning component doc is available, the skill normally writes a runtime subject entry using that doc as an `owning_specs` boundary.
That is a default convention, not a separate source-of-truth document.

```json
{
  "subjects": [
    {
      "audit_subject_id": "tokimon.workflow.document_audit",
      "title": "Document Audit workflow",
      "implementation_paths": ["tokimon/workflow/document_audit"],
      "test_paths": ["tests/workflow/document_audit"],
      "owning_specs": ["docs/c4/level-3-components/workflow/document-audit-cli.md"],
      "verification_commands": [
        ".venv/bin/python -m unittest discover -s tests/workflow/document_audit -t ."
      ],
      "dependency_edges": [
        {"kind": "reads", "target": "docs/guidelines/document-guidelines.md"}
      ]
    }
  ]
}
```

### Checklist Matrix Schema

Each row records one rule/subject pair:
`scope_id`, `audit_subject_id`, `rule_id`, `status`, `rule_source_path`, `rule_line`, `rule_hash`, and `rule_description`.

`status` is one of `pass`, `fail`, `not_applicable`, `needs_human_review`, or `unchecked`.
Optional evaluation fields are `summary`, `reason`, `evidence`, `verification`, and `review_prompt`.
Completed audit reports must use `--require-complete`.

Current matrices use schema 3 and include `project`, `input_snapshots`, and `audit_snapshot_id`.
The snapshot binds scope/run, subject definitions, rules, and each subject's implementation, test, owner-document, guideline, and verification-command inputs.
Reports and compiled evaluations must identify that snapshot.
Changed inputs invalidate fresh coverage and finalization; rebuild the matrix before reviewing the affected rows.
Completed-matrix regeneration preserves the snapshot identity when only row dispositions change.
An existing matrix path cannot be overwritten with another project, checkout, or input snapshot.

### Audit Report Schema

Each report records one completed checklist row.
Required fields:

- `schema_version`
- `workflow_id`
- `run_id`
- `scope_id`
- `audit_snapshot_id`
- `audit_subject_id`
- `rule_id`
- `status`: one of `pass`, `fail`, `not_applicable`, or `needs_human_review`
- `summary`
- `reason`
- `implementation_paths_examined`
- `test_paths_examined`
- `owning_specs_examined`
- `review`: `reviewer_id`, `source_path`, and `source_sha256` for the preserved result returned by the assigned subagent.

The review source is JSON with the same `reviewer_id` and a `reports` list containing each original row body before its `review` reference was attached.
The reviewer ID must name the assigned `subagent:<id>`; the source hash, reviewer, unique subject/rule key, and full row body must match.
Relative source paths resolve from the row report's directory.
Preserve the source under the run's `reviews/` directory and keep preliminary notes outside `reports/`.
If the subagent cannot write shared files, save its returned JSON unchanged before attaching review references.
Metadata validates consistency and does not substitute for actual review work.

Optional fields are `evidence`, `verification`, `review_prompt`, and `normalized_subject`.
`run_id`, `scope_id`, and `audit_snapshot_id` must exactly match the matrix that owns the row; a mismatch is an invalid report, not fresh coverage.
`verify-reports --require-complete` rejects missing, invalid, duplicate, extra, and unexpected reports for validated locked rows.
`compile-reports --require-complete` writes support-only `evaluations-from-reports.json` and `reports-summary.json` only after coverage is complete.
Those artifacts include `audit_authority: false`.
Compile output is not evidence that the implementation audit is clean.

### Pass-Lock Schema

`pass-locks.json` is a support-only artifact with `audit_authority: false`; its historical name is retained for compatibility.
It contains one lock per completed `pass` or `not_applicable` row that can be deterministically hashed.
Each lock binds:

- `project`, including the specific checkout
- `audit_snapshot_id` of the preserved source report
- `scope_id`
- `run_id` of the source report that supplies the evidence
- `status` (`pass` or `not_applicable`)
- `audit_subject_id`
- `rule_id`
- `rule_hash`
- `subject_definition_hash`
- `implementation_path_hashes`
- `test_path_hashes`
- `owning_spec_path_hashes`
- `guideline_path_hashes`
- `verification_commands`
- `verification_command_hash`
- `pass_report_path`
- `pass_report_hash`
- `verification_evidence_pointers`

Every file or directory hash records its `hash_basis`.
Repository directories use `git_worktree_files_v2`, covering tracked files plus non-ignored untracked files so pre-commit additions participate in the lock. Each entry binds file type, permission mode, symlink target when present, and content bytes. Git failure or an in-repository directory with no worktree files makes that row ineligible for a pass lock instead of silently switching inputs.
Directories outside Git use the explicit `filesystem_tree_v2` basis with the same per-file identity fields. Individual files use `file_identity_v2`.

Only `pass` and `not_applicable` reports produce locks.
`fail` and `needs_human_review` rows do not lock.
Lock creation validates the frozen reviewed inputs before reading their signatures; it cannot attach an old report to newly changed code.
Reused reports keep their original run, snapshot, and review-source references.
Source results are validated again during reuse, so changing a preserved review result invalidates its lock.

### Recheck-Plan Schema

`recheck-plan.json` is a support-only artifact with `audit_authority: false`.
It records `matrix_path`, `prior_pass_locks_path`, lock validation availability, summary counts, and one row per current matrix row.
Each row has `action: use_validated_pass_lock` or `fresh_report_required`, reason codes, and lock/report pointers when a lock is valid.

A row uses a validated pass-lock only when the prior lock and current row bind the same project and checkout, scope id, subject id, rule id, rule hash, subject definition hash, relevant path hashes, guideline hashes, verification command hash/list, source report path/hash/status, and verification pointers.
Any changed, missing, unreadable, duplicate, unsupported, invalid, or tampered bound input makes the row `fresh_report_required`.
If lock validation is unavailable, the plan requires complete fresh reports and locked coverage must not be claimed.

### Evaluation Rules

Every accepted or compiled evaluation carries `run_id` and `scope_id` matching the matrix/run being recorded.

- `pass` requires evidence or verification.
- `fail` requires summary and exact evidence.
- `not_applicable` requires reason.
- `needs_human_review` requires exact evidence and `review_prompt`.
  Use it for real blockers to supported judgment, not for ordinary human review of final audit findings.
- `unchecked` is valid only for generated matrix rows and is not allowed in evaluations.

Each exact evidence object must include:

- `path`
- one positive `line`, `line_start`, or `start_line`
- one non-empty `text`, `excerpt`, `quote`, or `details`

### Ledger Contract

The current ledger records open and resolved finding history.
Schema 3 ledgers bind a `project_id`; a foreign project's ledger is rejected even at an explicit path.
Legacy schema 1/2 ledgers remain readable, but a nonempty unbound ledger must be migrated and bound before recording new dispositions.
Migration preserves original records and run artifacts; entry `migration_sources` remain attached through later updates and closures.
`fail` and `needs_human_review` are open states; `needs_human_review` is a durable review item, not a confirmed failure.
When an accepted `pass` row is recorded, every open ledger entry with the same `scope_id`, `rule_id`, and `audit_subject_id` becomes `fixed` and records the resolving run.
When an accepted `not_applicable` row is recorded, the same matching open entries become `stale`.
A pass or not-applicable row without a matching open entry does not create a synthetic finding.
Partial accepted-evaluation inputs affect only the rows they contain.

Each ledger entry keeps fingerprint, scope id, rule id, audit subject id, normalized subject, status, summary, reason, evidence, verification, review prompt, last rule hash, first/last/resolved run metadata, last run artifact pointer, and timestamps.
When `fingerprint` is omitted, the workflow computes one from:

```text
scope_id + rule_id + audit_subject_id + normalized_subject
```

Line numbers are not part of the fingerprint because they drift during code edits.
`ledger-summary` reports open and resolved counts plus missing evidence paths; missing evidence makes the health result false without mutating the ledger.

Legacy schema 1/2 reports and matrices remain readable as historical evidence.
They cannot cover a new schema 3 matrix or create current pass-locks; migration must not invent reviewer results or input snapshots for old records.

## Data Ownership

- Owns local code-audit artifacts under `.tokimon/workflow/code-audit/projects/<project-id>/`.
- Owns run subjects, matrices, matrix summaries, matrix CSVs, structured reports, support evaluations, report summaries, accepted per-run findings, and the current ledger.
- Owns support-only pass-lock and recheck-plan artifacts under run directories.
- Consumes read-only tracked guideline rule sources, audited repository paths, and evaluator-produced row JSON as inputs.
- Derives rule inventories, normalized audit subjects, subject/rule matrices, status counts, finding fingerprints, and current ledger summaries.
- Does not own audited source code, tests, product runtime behavior, Pair transcripts, ChatGPT conversations, or human review decisions.

## Safety Rules

- The CLI writes only through these commands:
  - `matrix`
  - `pass-locks`
  - `recheck-plan`
  - `compile-reports`
  - `record-findings`
- `matrix` writes only under the configured run root.
- `pass-locks` and `recheck-plan` write only under the configured output directory.
- `compile-reports` writes only support artifacts under the configured output directory or the matrix file's directory.
- `record-findings` writes one accepted disposition run artifact and the configured current ledger.
- `contract`, `rules`, `subjects`, `verify-reports`, and `ledger-summary` are read-only.
- The CLI does not read secrets, cookies, API keys, browser sessions, MCP configs, or provider credentials.
- The CLI does not call external network services.

## Failure Modes

- Guideline rule IDs are duplicated; fix the owning guideline source before using the extracted rule set.
- Subject input is missing required fields, names an unknown subject, or points to missing paths; correct the subject JSON and rerun `subjects`.
- Matrix counts do not match expected guards; recheck the declared subjects and live rule sources before trusting the run.
- Report coverage is incomplete, invalid, duplicated, outside the checklist, or includes a report for a validated locked row; repair only the reported gaps and rerun `verify-reports`.
- Prior pass-locks are missing, stale, duplicated, unreadable, unsupported, or tampered; rerun the unlocked rows and do not claim locked coverage for them.
- Matrix evaluation is incomplete or omits required exact evidence; fix the evaluation rows before using `--require-complete` or recording findings.
- Accepted evaluation input is marked `audit_authority: false`; use an auditor-owned disposition file instead of support output.
- Evaluation uses an unknown rule ID; fix the rule source mismatch or use `--allow-unknown-rules` only for an intentional migration/emergency repair.
- Duplicate rule/subject dispositions or supplied fingerprints with mismatched scope/rule/subject ownership make `record-findings` reject the batch before any run or ledger write; correct the input and retry.
- Ledger schema is unsupported or run artifact paths cannot be written; inspect the configured run root and ledger path before retrying.
- Report `run_id` or `scope_id` does not match the matrix; regenerate the report for the current matrix instead of copying a prior row.
- Reusing a reports directory across recheck matrices mixes prior evidence with current fresh coverage; keep prior reports at their lock paths and use a new empty reports directory for each recheck matrix.
- A repository directory cannot be enumerated through Git or has no tracked inputs; keep the row fresh rather than silently creating a lock from a different file set.
- Recording an accepted row writes per-run dispositions and updates the current ledger; recover by preserving the run artifact, fixing the audited surface or review prompt, rerunning the audit row, and recording the new disposition rather than hand-editing ledger fingerprints.

## Acceptance Checks

```bash
.venv/bin/python -m unittest discover -s tests/workflow/code_audit -t . -p 'test_*.py'
```

```bash
.venv/bin/python -m tokimon.workflow.code_audit contract --json
```

```bash
.venv/bin/python -m tokimon.workflow.code_audit rules --json
```

```bash
.venv/bin/python -m tokimon.workflow.code_audit ledger-summary --json
```

Expected outcome: tests pass and support commands emit the documented JSON shape.

Structured-report smoke commands should also be covered by workflow tests:
`verify-reports --require-complete` rejects incomplete coverage, and `compile-reports --require-complete` writes support evaluations plus a summary only for complete coverage.
Recheck smoke coverage should prove that `compile-reports --require-complete` emits `pass-locks.json`, `recheck-plan` validates unchanged pass and not-applicable rows, and lock-aware `verify-reports`/`compile-reports` count each locked status correctly without accepting stale locks or new reports for locked rows.

AI coding agent verification prompt:

```text
Review docs/c4/level-3-components/workflow/code-audit-cli.md and tokimon/workflow/code_audit/.
Confirm that documented commands, default paths, subject schema, matrix schema,
report schema, ledger semantics, runtime artifact ownership, and tests match implementation.
Confirm that `compile-reports` output is not documented or shaped as authoritative audit results.
Report any drift with file and line evidence.
```

Expected result: no drift between the Code Audit workflow contract, CLI behavior, ledger behavior, and tests.

## References

- L2 Containers: `docs/c4/level-2-containers/workflow/code-audit.md`
- L1 Context: `docs/c4/level-1-context/workflow-context.md`
- Guidelines: `docs/guidelines/code-guidelines.md`
- Guidelines: `docs/guidelines/unit-testing-guidelines.md`
