---
name: document-audit
description: Audit repository documentation for ownership, alignment, and audit readiness without editing audited documents.
---

# Document Audit

Run a current-repository audit of tracked documentation without editing the audited documents.
Use the Document Audit workflow to load live rule IDs and write checklist matrices.
Use it to verify structured reports and update the ledger.
Read enough surrounding context to understand the document's role.
Operator entrypoint: `{{tokimonCommand}} -m tokimon.workflow.document_audit`.
Pass `--project-root <task-checkout>`; use `contract --json` for the selected project and its runtime paths.

Rule source selection:
- All documentation audits use `docs/guidelines/document-guidelines.md`.
- C4 audits under `docs/c4/` also use `docs/guidelines/c4-document-guidelines.md`.
- Split mixed target sets by rule source set.

## Contents
- [Scope](#scope)
- [Steps](#steps)
- [Findings](#findings)
- [Matrix Report Shape](#matrix-report-shape)
- [Guardrails](#guardrails)

## Scope

Audit tracked documentation surfaces only unless the user explicitly widens scope.
This includes C4 docs, operations docs, guidelines, skills, and root entrypoints such as `AGENTS.md`.
Treat prompt and embedded payload text as context unless the user explicitly targets that payload behavior.

## Steps

1. Identify the requested document type, target scope, and rule source set.
2. Read the workflow support contract once with `<operator-entrypoint> contract --json`.
   Use its `operator_entrypoint`, `run_root`, artifact names, statuses, and schema constraints.
3. Load live rules from the selected sources.
4. Load the compact current ledger for the target scope.
5. Identify each requested document's role, owner, audience, and dependencies.
   Treat one document as the default audit unit; group documents only when they must be judged or changed together.
   When a coherent group or external owner/dependency input affects the result, pass audit-unit JSON with `document_paths` and `bound_input_paths` through `--units-from`.
6. Create the runtime checklist matrix before delegating audit work.
   Preserve the normalized audit-unit artifact written with the matrix.
   The matrix captures one `audit_snapshot_id` over its exact guideline sources, documents, bound inputs, scope, and unit definitions.
   Use exact expected rule, document, and matrix counts when they are known or requested.
7. Delegate one document or small coherent document group per subagent task.
   Each task reads its assigned documents once and audits every matrix rule for those documents.
   Run the work in bounded batches when the matrix is large, but keep the output contract one report per row.
8. Write reports only for rows actually reviewed by the assigned subagent; keep preliminary sweeps outside the reports directory.
   Each subagent preserves its result under `<run-dir>/reviews/` and writes one report per assigned row under `<run-dir>/reports/`.
   If it cannot write shared runtime files, save its returned JSON unchanged before attaching review references.
   Required report fields:
   - `schema_version`
   - `workflow_id`
   - `run_id`
   - `scope_id`
   - `audit_snapshot_id`: exact value from the matrix
   - `file_path`
   - `rule_id`
   - `status`: `pass`, `fail`, or `not_applicable`
   - `summary`
   - `reason`
   - `findings`: non-empty only for `fail`
   - `review`: actual `reviewer_id`, preserved result `source_path`, and `source_sha256`, as defined by the report contract
   Keep `pass` and `not_applicable` reports concise instead of repeating shared document context.
9. Verify report coverage after each batch.
   If any report is missing, invalid, duplicate, or outside the checklist, return to step 7 for only the reported gaps.
   If the matrix snapshot is stale, regenerate it before continuing; never attach a report from an older snapshot to changed content.
10. Compile complete reports into support-only row evaluations.
    Treat these artifacts as support only; they do not prove the audit is clean and do not close known findings.
11. Write the completed matrix from compiled evaluations and require completion.
    Use the same run, scope, documents, audit units, and rule-source selection as the unchecked matrix so compiled provenance matches.
12. If live rules and the requested committed snapshot differ, run both matrices.
    Use the live sources and the committed `--source-ref HEAD` sources.
13. Decide the final audit disposition before updating the ledger.
    Known findings must be named as `open`, `fixed`, `deferred`, `rejected`, `blocked`, or `stale`.
    The accepted findings file must be auditor-owned and separate from compile output.
14. Record accepted findings with `record-findings --accepted-findings`.
    Use the same rule sources.

## Findings

Lead with findings and number them continuously across the audit.
Each finding must include:

- rule ID from the live guideline rules
- path evidence
  Use `evidence` as a non-empty list of line-level objects.
  Each object uses a positive `line` or `start_line`.
  Each object includes one non-empty `text`, `excerpt`, `quote`, or `details` field.
- issue
- severity: `must_fix`, `should_fix`, or `optional`
- status: `open`, `fixed`, `deferred`, `rejected`, `blocked`, or `stale`
- recommended action
- blast radius when material
- verification

Previously deferred, rejected, or blocked findings stay suppressed unless the evidence or rule meaning changed.

## Matrix Report Shape

For matrix-backed audits, report:

- rule count, document count, and exact matrix count
- matrix JSON, matrix CSV, and summary artifact paths
- report coverage counts and support artifact paths
- status counts for `pass`, `fail`, `not_applicable`, and `unchecked`
- whether the matrix was run with `--require-complete`
- final disposition for every previously known finding

If both committed-snapshot and live-rule matrices were required, list both.
Do not report a clean audit from `verify-reports`, `compile-reports`, full coverage, or zero failed rows.
Report accepted findings only after the final disposition step.
Do not turn `not_applicable` rows into findings.

## Guardrails

Preserve valid differences between document types, useful user wording, and declared scope.
Change them only when the user asks.
Do not force skills, prompts, or embedded payloads into C4 templates.
An audit edits no repository content; it may only write runtime artifacts under `.tokimon/workflow/document-audit/`.
Use `document-audit-fix` when the user asks to implement fixes for document-audit findings.

End with the recommended next step.
