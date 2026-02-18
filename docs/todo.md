# Tokimon TODO

This document tracks prioritized follow-up work to reduce OOM risk and improve runtime reliability.

1. [ ] OOM and reliability hardening (Owner: TBD)
   - Acceptance criteria: Use Tokimon to identify, prioritize, and implement the required safeguards and validation to prevent OOM regressions and improve runtime reliability.
2. [x] Relocate `runs/` to repo root (Owner: TBD)
   - Acceptance criteria: Store `runs/` as a repo-root sibling of `src/` and `docs/`, and update all setup and usage instructions accordingly.
3. [ ] Self-improve from Codex agent loop article (Owner: TBD)
   - Acceptance criteria: Use Tokimon to extract improvements from https://openai.com/index/unrolling-the-codex-agent-loop/ and apply at least one validated enhancement.
4. [ ] Self-improve from skills shell tips article (Owner: TBD)
   - Acceptance criteria: Use Tokimon to extract improvements from https://developers.openai.com/blog/skills-shell-tips and apply at least one validated enhancement.
5. [x] Chat UI-first interaction flow (Owner: TBD)
   - Acceptance criteria: Deliver a Tokimon chat UI flow that supports routine interaction without CLI invocation for each step.
6. [x] Generalized arbitrary-goal self-improvement mode (Owner: TBD)
   - Acceptance criteria: Provide a super-concise workflow that takes an arbitrary goal and executes end-to-end self-improvement with measurable output.
7. [ ] Claude CLI learning + mixed-provider self-improve ratio (Owner: TBD)
   - Acceptance criteria: Learn from `~/clover/joey-playground/apps/` how to use Claude CLI, then enforce a `claude:codex` session mix of `1:4` in every self-improve task.
8. [x] Ensure self-improve uses the agent loop (Owner: TBD)
   - Acceptance criteria: Confirm and enforce that Tokimon self-improve tasks run through the full agent loop (iterative model/tool cycle), not a one-shot execution path.
9. [x] Self-improve entry-point request handling loop (Owner: TBD)
   - Acceptance criteria: For Tokimon entry-point tasks, enforce this flow: (1) understand the user's request and ask clarifying questions immediately if it is unclear or ambiguous, (2) once the request is clear move to step 3, (3) generate a prompt, (4) run the agent with the prompt, (5) monitor and report progress, and (6) verify the final outcome; if step 6 fails, restart from step 3.
   - Verification: `pytest --maxfail=1 -c src/pyproject.toml src/tests` (passes); see `src/tests/test_self_improve_entrypoint_loop.py` and `src/self_improve/orchestrator.py`.
10. [ ] Schema-driven structured outputs (Owner: TBD)
   - Acceptance criteria: Define a per-step "success schema" and enforce schema-valid structured results (not just valid JSON), with bounded repair on validation failures.
11. [ ] Persist and render structured results + UI blocks (Owner: TBD)
   - Acceptance criteria: Persist the full structured step result (including any UI blocks) as first-class run artifacts/outputs, and render them in the chat UI rather than only printing a text summary.
12. [ ] Observability-ready metrics and dashboards (Owner: TBD)
   - Acceptance criteria: Standardize a small set of run/step metrics (with types and units) and produce an importable dashboard artifact so Tokimon runs are measurable and easy to visualize.
