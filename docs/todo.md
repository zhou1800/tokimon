# Tokimon TODO

This document tracks prioritized follow-up work to reduce OOM risk and improve runtime reliability.

1. [x] Schema-driven structured outputs (Owner: TBD)
   - Acceptance criteria: Define a per-step "success schema" and enforce schema-valid structured results (not just valid JSON), with bounded repair on validation failures.
2. [x] Persist and render structured results + UI blocks (Owner: TBD)
   - Acceptance criteria: Persist the full structured step result (including any UI blocks) as first-class run artifacts/outputs, and render them in the chat UI rather than only printing a text summary.
3. [x] Observability-ready metrics and dashboards (Owner: TBD)
   - Acceptance criteria: Standardize a small set of run/step metrics (with types and units) and produce an importable dashboard artifact so Tokimon runs are measurable and easy to visualize.
4. [x] Non-trivial upgrade: move chat UI to React + Tambo (Owner: TBD)
   - Acceptance criteria: Migrate the chat UI to a React frontend using Tambo to render Tokimon UI blocks (charts/forms/panels), add the required JS build+serve workflow, and keep `/healthz` and `/api/send` stable.
