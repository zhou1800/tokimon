---
name: code-audit-fix
description: Automate scoped fixes for code-audit findings, including verification and re-audit loops, stopping only on real blockers.
---

# Code Audit Fix

Run a bounded remediation loop for code-audit findings.
The loop is bounded by the declared scope and reusable row state, not by an arbitrary pass count.
Use the `code-audit` skill as the read-only audit primitive, then fix evidence-backed findings, verify the changed scope, and re-audit until findings are fixed, deferred, rejected with rationale, or blocked by a concrete reason.

## Scope

Edit only the chosen remediation scope.
Do not broaden the target queue; if broadening is required to satisfy the promise, report it as a blocker.
For each round, freeze the exact finding rows and files selected for remediation before editing.
Every edit hunk must be traceable to one selected finding as its fix, required contract update, or verification; do not add unrelated refactors or unreviewed fixes discovered while editing.
Preserve owning contracts, public behavior, and intended test behavior unless the finding and an owning source or explicit user decision make the change unambiguous.
Do not weaken tests to make verification pass; defer or block when the correct behavior is unclear.
Do not change ledger semantics; the code-audit ledger keeps `fail` and `needs_human_review` rows open and preserves accepted `pass` or `not_applicable` row dispositions as `fixed` or `stale` closures for the exact scope/rule/subject row.
Treat compiled reports, pass-locks, and locked coverage as support artifacts, not audit authority.
Do not ask for human approval before ordinary scoped fixes, verification, or re-audits.
This bounded automation covers source edits in scope, verification, and re-audits only.
Route staging, commits, branch changes, merges, pushes, and other git mutations through the `git` skill.
Do not merge, force-push, stage unrelated changes, or mutate another branch unless the user explicitly requested that git action.

## Method

1. Identify the target queue, current promise, owning docs, expected verification commands, and any existing code-audit run artifacts.
   Freeze each target's subject and file set before auditing it.
2. Run `code-audit` for the current target or read an existing complete code-audit run when the user supplied one.
3. Choose fixes from evidence-backed `fail` rows and any explicit user decisions already present in the task context.
   Treat `needs_human_review` rows as investigation targets first; stop for the human only when missing owner intent, external evidence, destructive action, or safety-critical uncertainty is a real blocker.
4. Apply the smallest scoped fix that preserves contracts, tests, and public behavior.
5. Run relevant verification and record exact commands and outcomes.
6. Re-audit the same target.
   For every recheck, create the matrix under a new run id and use the newest `pass-locks.json` from `compile-reports` to generate `recheck-plan` before launching row workers.
7. Re-audit only rows marked `fresh_report_required`, scheduling them by affected subject so each subject's context is loaded once.
   Use a new empty reports directory for each recheck matrix; prior source reports remain referenced by pass-locks and must not be copied into the current recheck.
   Skip prior `pass` and `not_applicable` rows only when `recheck-plan`, `verify-reports --recheck-plan`, and `compile-reports --recheck-plan` validate unchanged pass-lock coverage mechanically.
8. If an individual lock is stale or missing, re-audit only its fresh-required row; run complete fresh reports only when plan-wide lock validation is unavailable or invalid.
9. Repeat fix, verify, and targeted re-audit until the target has no new `fail` or `needs_human_review` rows except items explicitly deferred, rejected with rationale, or blocked with exact evidence.
10. Create an auditor-owned accepted-dispositions artifact for every current row represented by the complete re-audit result.
    Keep it separate from compile support output, then record it with `record-findings --accepted-evaluations` so accepted pass/not-applicable rows close prior exact-row findings while fail/human-review rows remain open.
11. Move to the next target only after recording the target outcome, artifact paths, verification results, fresh report counts, locked row counts, unlocked row counts, lock artifact paths, every unlocked-row reason, and the resulting ledger dispositions.

## Pass-Lock Rules

Use pass-locks only within a remediation recheck after a previous complete matrix run.
A lock is valid only because a completed report row reached `pass` or `not_applicable` against the matrix's captured inputs and those inputs remain unchanged; the lock remains support state rather than audit authority.
Only a changed bound implementation, test, owner doc, guideline rule, subject definition, verification command, or source report invalidates that row's lock; unrelated changes and findings do not.
`fail` and `needs_human_review` rows never lock.
Do not relaunch workers for validated unchanged locked `pass` or `not_applicable` rows.
When multiple fresh rows share a mutable input, finish all selected fixes to that input in one edit wave before rebuilding the matrix and recheck plan; the next plan will invalidate every row whose declared input hash changed.
Carry every still-valid lock into the next recheck; an individual stale, missing, or tampered row lock must not discard the rest or restart the target.

## Output

Report completed targets, fixes made, verification results, audit artifact paths, fresh report counts, locked row counts, unlocked row counts with reasons, and remaining deferred, rejected, or blocked items.
