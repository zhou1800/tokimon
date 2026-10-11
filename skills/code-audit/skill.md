---
name: code-audit
description: Audit repository implementation against owning contracts, code guidelines, unit-test guidelines, and verification evidence.
---

# Code Audit

Run implementation audits for the current repository without mutating the audited implementation surface.
Use a repository routing document when present to choose owning intended-behavior documents.
Use `docs/guidelines/code-guidelines.md` and `docs/guidelines/unit-testing-guidelines.md` as canonical audit standards.
Use code and tests as live evidence.
Use the Code Audit workflow to load live rule IDs, validate subject input, and write checklist matrices.
Use it to verify structured reports, compile evaluations, and update the ledger.
Do not judge code in isolation.
Read enough neighboring code, tests, owner docs, and verification evidence to understand the audited surface.
Operator entrypoint: `{{tokimonCommand}} -m tokimon.workflow.code_audit`.
Pass `--project-root <task-checkout>` to select the audited repository and its ledger.
Use CLI `--help` and `contract --json` for command discovery.

## Contents
- [Scope](#scope)
- [Method](#method)
- [Candidate Types](#candidate-types)
- [Findings](#findings)

## Scope

Audit repository-owned implementation code and tests, including untracked additions in the requested worktree, unless the user explicitly narrows or expands scope.
Do not edit audited source files, docs, tests, generated artifacts, or local memories during an audit.
Do not edit runtime state outside the code-audit workflow root.
Every matrix row in the requested scope needs a fresh report.
Do not stop mid-audit for human approval or review; finish the read-only audit and present findings to the human at the end.

## Method

1. Identify the audited surface, consumers, protected public contract, relevant verification commands, and dependency edges.
   Select owning C4 docs and other owner docs through the audited repository's routing documents when they are audit evidence for the requested surface.
2. Define bounded audit subjects.
   A subject is any bounded implementation or behavior surface under audit.
   When the requested scope has clear owner docs, prefer one subject per owner-doc boundary.
   Otherwise define subjects directly around the bounded surface under audit.
   Keep each subject's `implementation_paths` and `test_paths` as narrow and non-overlapping as ownership permits; group surfaces that must be evaluated together instead of assigning the same broad directory to multiple subjects.
   Write the subject list as a runtime artifact under the run folder; do not create or maintain a separate source-controlled subject inventory.
3. Read the workflow support contract once with `<operator-entrypoint> contract --json`.
   Use its `operator_entrypoint`, `run_root`, artifact names, statuses, and schema constraints.
   Create one runtime folder at `<run-root>/<run-id>/`.
4. Validate subject input with `subjects` and generate the unchecked checklist matrix with `matrix`.
   The subject source is always ordinary JSON, whether it was derived from owner docs or authored for another bounded surface.
   Use exact expected rule, subject, and matrix counts when they are known or requested.
5. Delegate one audit subject or small coherent subject group per subagent task.
   Each task loads its subject context once and audits every matrix rule for those subjects.
   The subagent must compare owning C4 docs and other owner docs when they are audit evidence, implementation, tests, and verification evidence.
   Run the work in bounded batches when the matrix is large, but keep the output contract one report per row.
6. Preserve each subagent's original JSON result (`reviewer_id` and `reports`) under `<run-dir>/reviews/`, then write one report per row under `<run-dir>/reports/` with a reference to that result.
   Write reports only for rows actually reviewed by the assigned subagent; keep preliminary sweeps outside the reports directory.
   If the subagent cannot write shared files, save its returned JSON unchanged before attaching review references.
   Report fields:
   - `schema_version`
   - `workflow_id`
   - `run_id`
   - `scope_id`
   - `audit_snapshot_id` from the assigned matrix
   - `audit_subject_id`
   - `rule_id`
   - `status`: `pass`, `fail`, `not_applicable`, or `needs_human_review`
   - `summary`
   - `reason`
   - `implementation_paths_examined`
   - `test_paths_examined`
   - `owning_specs_examined`
   - `evidence` when required by status
   - `verification` when verification was run or named
   - `review_prompt` for `needs_human_review`
   - `review`: `reviewer_id`, `source_path`, and `source_sha256` of the preserved result
   Keep `pass` and `not_applicable` reports concise, and reuse unchanged verification evidence across applicable rows.
7. Verify report coverage with `verify-reports --require-complete` after each batch and before compiling support artifacts.
   If bound inputs changed after the matrix was created, rebuild it and review the changed inputs before compiling.
   If any report is missing, invalid, duplicated, or outside the matrix, return to step 5.
   Audit only the gap rows.
8. Compile complete fresh reports into support-only row evaluations with `compile-reports --require-complete`.
   Treat these artifacts as support only; they do not prove the implementation is clean and do not close known findings.
9. Regenerate the final matrix from compiled evaluations and require completion.
   Use the same scope, run, and subjects as the unchecked matrix, then pass `--evaluations <run-dir>/evaluations-from-reports.json` and `--require-complete`.
   Representative final matrix command:

   ```bash
   {{tokimonCommand}} -m tokimon.workflow.code_audit matrix \
     --project-root <task-checkout> \
     --scope-id <scope-id> \
     --run-id <run-id> \
     --subjects-from <run-dir>/subjects-input.json \
     --evaluations <run-dir>/evaluations-from-reports.json \
     --require-complete \
     --json
   ```

10. Decide the accepted disposition for every freshly audited row before updating the ledger.
    The accepted-dispositions file must be auditor-owned and separate from compile output.
    Include evidence-backed `fail` and `needs_human_review` rows to open or update findings, plus accepted `pass` and `not_applicable` rows so prior open findings for the exact scope/rule/subject row can close as `fixed` or `stale`.
11. Record the auditor-owned accepted dispositions with `record-findings --accepted-evaluations`.
    Do not pass raw compiled support output as audit authority, and do not treat compilation itself as acceptance.
12. Summarize the current ledger with `ledger-summary`.

## Candidate Types

Classify each candidate as `delete`, `merge`, `downgrade`, `promote`, `rename`, or `defer`.
These are recommendation labels, not workflow statuses.
Favor candidates that reduce real complexity without weakening readability, contracts, behavior, or verification evidence.
Treat C4, code, and test mismatches as findings before deciding which surface should change.

## Findings

Lead with findings and number candidates continuously.
For each candidate, include:

- surface
- path evidence
- issue and why it matters under the relevant guideline rule ID
- recommended action
- blast radius
- verification command
- expected result
- remaining manual checks

Use design-method names only as references to guideline rule IDs; keep labels subordinate to path evidence.
Use `needs_human_review` only for a real audit blocker: the row cannot be judged because required owner intent, external evidence, or safety-critical context is unavailable.
For `needs_human_review`, include exact evidence, the missing information, and a focused review prompt.
Do not invent evidence or mark semantic uncertainty as `pass` or `fail` without support.
Report the final matrix counts and fresh report coverage counts with every audit.

End with a concise disposition: `modify now`, `defer`, `keep`, or `make no change`.
