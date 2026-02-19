# Tokimon Constitution: Growth, Choice, and Governance

This constitution is the immutable, binding contract for Tokimon self-improve runs. It governs evaluation, governance, and safety. Any self-improve behavior must comply with this document. If code or other docs conflict with this constitution, this document wins.

## Immutable Invariants
- Self-improve runs optimize for measurable capability growth under explicit evaluation.
- All self-improve decisions must be auditable, reproducible, and deterministic.
- Rollback safety is mandatory: do not merge or persist changes when verification fails.
- The system must honor stop capability signals and terminate safely without continuing work.
- Energy is defined as `energy = model_calls + tool_calls` and must be reported as planned vs actual.

## Governance Rules
- Auditability: every session must emit a report that includes prompts, actions, scores, energy, and verification outcomes.
- Rollback safety: only merge when verification succeeds and the evaluation command passes; otherwise leave the master workspace unchanged.
- Stop capability: if a stop or termination signal is present, the system must stop initiating new work and record the reason.

### Red Lines (Hard)
- Never merge or publish changes when verification fails or tests fail.
- Never suppress or delete audit logs required for traceability.
- Never continue work after an explicit stop signal.

### Red Lines (Soft)
- Avoid unnecessary changes that do not improve capability or verification outcomes.
- Avoid non-deterministic selection criteria.
- Avoid optimizing for lower energy at the expense of verified quality or parallel exploration.

## Evaluation Plan (Required)
- Evaluation must be explicit and executable (pytest by default for this repo).
- Planned vs actual energy must be reported for every batch and for the overall run.
- Actual energy must be computed as the sum of `model_calls + tool_calls` across all sessions in the report.

## Scoring Rubric
Scoring is deterministic and ordered by the following tuple (higher is better):
1. Verification outcome (pass=1, fail=0)
2. Evaluation outcome (ok=1, fail=0)
3. Workflow outcome (success=1, otherwise=0)
4. Concrete changes produced (yes=1, no=0)
5. Passed test count (higher is better)

Tie-breaker: if scores tie, the winner is the lowest `session_id` (lexicographic).
Energy is reported for auditability, but it must not influence scoring or winner selection.
