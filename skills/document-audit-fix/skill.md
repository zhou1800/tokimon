---
name: document-audit-fix
description: Automate scoped fixes for document-audit findings, including documentation edits, verification, and fresh re-audit loops, stopping only on real blockers.
---

# Document Audit Fix

Run a bounded remediation loop for document-audit findings.
The loop is bounded by the declared scope and monotonic unit scheduling state, not by an arbitrary pass count.
Use the `document-audit` skill as the read-only audit primitive, then fix evidence-backed documentation findings, verify the changed scope, and re-audit until findings are fixed, deferred, rejected with rationale, or blocked by a concrete reason.

## Scope

Edit only the chosen remediation scope; report required scope expansion as a blocker.
For each round, freeze the exact findings and files selected for remediation before editing.
Every edit hunk must be traceable to one selected finding as its direct fix or a required consistency update; do not add unrelated cleanup or unreviewed fixes discovered while editing.
Preserve document-type differences, user wording, prompts, payloads, product definitions, supported behavior, ownership, and other contract meaning.
Change meaning only when the finding and an owning source or explicit user decision make the intended result unambiguous; otherwise defer the finding or report a blocker.
Prioritize `must_fix` and `should_fix` findings; fix `optional` findings only while their audit unit is already active and the edit is local and meaning-preserving.
Do not change document-audit workflow or ledger semantics, and treat compiled reports as support artifacts only.
Proceed without human approval for ordinary scoped documentation fixes, verification, and re-audits.
Route git mutations through the `git` skill and only when explicitly requested.

## Method

1. Identify the target queue, current promise, owning docs, expected verification commands, and any existing document-audit run artifacts.
   Freeze each target's document list and audit units in `audit-units.json`; use one document per unit unless documents must be judged or changed together.
   Declare owner or dependency documents that can change a unit's audit result as `bound_input_paths`.
2. Run `document-audit` for the current target or read an existing complete document-audit run when the user supplied one.
3. Record accepted finding dispositions under the same run id as the reports, then run `clean-unit-locks` for the complete result so clean units and accepted terminal dispositions become durable scheduling state.
4. Choose fixes from evidence-backed `fail` rows, auditor-accepted open findings, and explicit user decisions already present in the task context.
   Treat deferred, rejected, or blocked findings as settled unless new evidence, a changed rule, or a user decision reopens them.
5. Apply the smallest scoped documentation fix that preserves document ownership, dependency direction, and the accepted product contract.
6. Run relevant verification and record exact commands and outcomes.
7. Create the current full-target matrix under a new run id with the same audit units, then run `recheck-plan` from the newest `clean-unit-locks.json`.
   Use `--activate-unit` only for a named locked or settled unit selected for change; dependent units are not activated speculatively.
8. Follow the plan exactly: re-audit only rows marked `fresh_report_required`; write no report for `use_validated_clean_unit_lock`, `use_validated_terminal_disposition`, or `waiting_on_dependency` rows.
   Use a new empty reports directory for each matrix snapshot; prior source reports remain referenced by the state artifact and must not be copied into the current wave.
   Pass the plan to `verify-reports` and `compile-reports` so validated clean and settled rows complete the current wave's coverage.
9. Record the new accepted finding dispositions under the current report run id, then run `clean-unit-locks` with the recheck plan to preserve valid states, add newly clean or settled units, and retain dependency waits.
10. Rebuild the current matrix and plan. A dependent unit that was waiting becomes active only after its upstream units are stable, so it is audited once against their settled content rather than once per upstream edit.
11. If a fix reveals a material product, design, or owner-intent problem, stop and report the smallest decision needed.
12. Repeat steps 4-10 while the plan has active or waiting units. Finish only when `active_unit_count` and `waiting_unit_count` are both zero and report verification says `final_coverage_complete: true`.
13. Move to the next target only after recording the target outcome, artifact paths, verification results, fresh, locked, settled, and waiting counts, and every remaining deferred, rejected, or blocked item.

## Lock Rules

Treat `clean-unit-locks.json` and `recheck-plan.json` as workflow support artifacts, never audit authority.
Every current-schema report must carry the matrix's exact `run_id`, `scope_id`, and `audit_snapshot_id`; regenerate a stale matrix before auditing rather than binding old reports to changed inputs.
A clean unit has only `pass` or `not_applicable` rows and no open finding in the audit scope.
A settled unit has non-clean rows whose findings all have exact `deferred`, `rejected`, or `blocked` ledger dispositions; settled is terminal scheduling state, not a clean result.
A unit whose result depends on another active unit is `waiting_on_dependency`; do not audit it until the blocker becomes clean or settled.
Do not re-audit or edit a clean or settled unit unless `recheck-plan` invalidates its content, applicable rules, bound inputs, source reports, or terminal ledger dispositions, or it is explicitly activated for recorded new evidence.
Before changing a clean or settled unit, record the triggering finding and activate only that unit; preserve every unrelated mechanically validated state.
Mechanically revalidate all remaining states before declaring the queue complete.

## Output

Report completed targets, documentation fixes made, verification results, audit artifact paths, clean, settled, waiting, and active unit counts, invalidations with reasons, accepted finding dispositions, and remaining deferred, rejected, or blocked items.
