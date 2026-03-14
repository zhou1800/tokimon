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

## Resource Safety Directive: Hard vs Soft Constraints

This directive defines how Tokimon plans, enforces, and reports resource usage and safety boundaries during self-improve runs. It is binding for self-improve orchestration and reporting.

### Budgets (Plan vs Actual)
- Time: every entry-point prompt and report MUST declare a planned time budget and report actual elapsed time (best-effort, deterministic measurement).
- Memory: every entry-point prompt and report MUST declare a planned memory budget and report actual peak memory usage when available (best-effort; if unavailable, report `unknown` deterministically).
- Energy: every entry-point prompt and report MUST declare a planned energy budget and report actual energy computed as `model_calls + tool_calls` (as defined in Immutable Invariants). Energy reporting is required for auditability; it MUST NOT influence winner selection.
- Concurrency: every entry-point prompt MUST declare the planned concurrency (worker/session concurrency) and the report MUST record the actual concurrency used (and any changes due to mitigations).

### Hard Red Lines (Enforcement = Refuse/Block)
Hard red lines are non-negotiable. If triggered, Tokimon MUST immediately refuse or block execution (no agent execution) and log the refusal with a clear reason.
- Unsafe goals: requests that fall under repo Non-goals (e.g., cyber exploitation, credential theft, malware, or data exfiltration) MUST be refused.
- Governance violations: never merge when verification fails; never suppress required audit logs; never continue after an explicit stop signal.

### Soft Red Lines (Mitigation = Degrade, Then Stop)
Soft red lines indicate elevated risk or stalled progress. When triggered, Tokimon MUST apply deterministic, auditable mitigations and continue only if verification remains feasible.
- Triggers (examples): repeated tool failures/timeouts, repeated retries with no novel progress, evaluation regression, or insufficient remaining budget for verification.
- Auto-mitigations (minimum set):
  - Reduce concurrency (down to 1).
  - Shorten context (reduce prompt/input payload size deterministically).
- If mitigations do not stabilize progress (or remaining budget is too low to verify safely), Tokimon MUST stop early and return `PARTIAL`, preserving best artifacts plus a concrete next-step plan.

### Risk Register (Required)
Every entry-point prompt and report MUST include a risk register that names the top risks, their triggers, and mitigations (at minimum: OOM/overscan risk, non-determinism risk, and retry-loop risk).

### Stop Conditions (Required)
Every entry-point prompt and report MUST include explicit stop conditions:
- Hard stop: stop immediately on hard red line violations or explicit stop signals.
- Soft stop: stop early with `PARTIAL` after mitigations are exhausted or when verification is not feasible within remaining budget.
- Rollback semantics: when stopping with `PARTIAL` or `BLOCKED`, master workspace MUST remain unchanged.

### Audit Log Requirements
Self-improve Markdown reports MUST include an audit log that records:
- Attempted actions (what was tried, with key parameters like concurrency and evaluation command).
- Refused actions (what was refused/blocked, with the hard-red-line reason).
- Mitigations applied (what changed, why, and the observed stabilization outcome when available).
- The stop condition that fired (hard/soft) for any non-success outcome.

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
- Self-improve reports must include baseline evaluation summary, post-change evaluation summary, delta, a causal mechanism hypothesis, and an explicit pass condition.
- The pass condition must be chosen deterministically for auditability and be reported verbatim in the entry-point prompt and report.

## Scoring Rubric
Scoring is deterministic and ordered by the following tuple (higher is better):
1. Verification outcome (pass=1, fail=0)
2. Evaluation outcome (ok=1, fail=0)
3. Workflow outcome (success=1, otherwise=0)
4. Concrete changes produced (yes=1, no=0)
5. Passed test count (higher is better)

Tie-breaker: if scores tie, the winner is the lowest `session_id` (lexicographic).
Energy is reported for auditability, but it must not influence scoring or winner selection.
