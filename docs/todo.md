# Tokimon TODO

This document tracks prioritized follow-up work to reduce OOM risk and improve runtime reliability.

1. [ ] Schema-driven structured outputs (Owner: TBD)
   - Acceptance criteria: Define a per-step "success schema" and enforce schema-valid structured results (not just valid JSON), with bounded repair on validation failures.
2. [ ] Persist and render structured results + UI blocks (Owner: TBD)
   - Acceptance criteria: Persist the full structured step result (including any UI blocks) as first-class run artifacts/outputs, and render them in the chat UI rather than only printing a text summary.
3. [ ] Observability-ready metrics and dashboards (Owner: TBD)
   - Acceptance criteria: Standardize a small set of run/step metrics (with types and units) and produce an importable dashboard artifact so Tokimon runs are measurable and easy to visualize.
4. [ ] Non-trivial upgrade: move chat UI to React + Tambo (Owner: TBD)
   - Acceptance criteria: Migrate the chat UI to a React frontend using Tambo to render Tokimon UI blocks (charts/forms/panels), add the required JS build+serve workflow, and keep `/healthz` and `/api/send` stable.
5. [x] Self-improve: enforce evaluation-first experiment reporting (Owner: TBD)
   - Acceptance criteria: Self-improve prompts and reports always include baseline/post-change/delta, causal mechanism hypothesis, and an explicit pass condition (including planned vs actual energy).
   - Follow-up complete: Winner causal mechanism hypothesis is now loaded from per-attempt experiment JSON and rendered directly in the top-level batch report.
6. [ ] Self-improve: include failing test IDs in baseline/post-change summaries (Owner: TBD)
   - Acceptance criteria: Batch report shows a bounded list of `failing_tests` for baseline and winner post-change evaluations to make deltas auditable.
7. [ ] Self-improve: parallel exploration protocol (Owner: TBD)
   - Acceptance criteria: Deterministic per-session `path_charter`, enforced per-attempt experiment summary fields (`plan`, `path_charter`, `self_critique`, `lessons` + experiment-loop fields), report includes diversity check + scoring function + per-path comparison + loser reasons/Lessons, and deterministic winner selection aligned with the Constitution.
8. [ ] Self-improve: Memory-as-Asset charter (Owner: TBD)
   - Acceptance criteria: Persist Lessons for failure/retry decisions with required fields, deny/redact secrets deterministically, and use staged retrieval to stop or force strategy changes on repeated failures.
9. [ ] Skill Asset Protocol: Prompt Skills + Code Skills with metadata (Owner: TBD)
   - Acceptance criteria: Detect skill gaps (repeat subtask patterns, repeat retry failures, repeat tool workflows), require metadata for both Prompt Skills and Code Skills, keep failing skills as candidate drafts + record Lessons, and register only on passing validation with sprawl/safety guardrails.
