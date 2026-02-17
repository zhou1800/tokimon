# Tokimon TODO

This document tracks prioritized follow-up work to reduce OOM risk and improve runtime reliability.

1. [ ] GrepTool safety limits (Owner: TBD)
   - Acceptance criteria: Enforce configurable max file-size and match-count limits in GrepTool, with tests covering truncation and fail-safe behavior.
2. [ ] Large-runs regression test (Owner: TBD)
   - Acceptance criteria: Add a regression test for large runs that reproduces historical memory pressure and fails on OOM-related regressions.
3. [ ] Ripgrep guard docs (Owner: TBD)
   - Acceptance criteria: Document ripgrep guard behavior, defaults, and override guidance in `docs/` with one concrete example.
4. [ ] Self-improve OOM smoke scenario (Owner: TBD)
   - Acceptance criteria: Add a smoke scenario for self-improve flow that validates completion without OOM under CI-representative settings.
5. [ ] Relocate `runs/` to repo root (Owner: TBD)
   - Acceptance criteria: Store `runs/` as a repo-root sibling of `src/` and `docs/`, and update all setup and usage instructions accordingly.
6. [ ] Self-improve from Codex agent loop article (Owner: TBD)
   - Acceptance criteria: Use Tokimon to extract improvements from https://openai.com/index/unrolling-the-codex-agent-loop/ and apply at least one validated enhancement.
7. [ ] Self-improve from skills shell tips article (Owner: TBD)
   - Acceptance criteria: Use Tokimon to extract improvements from https://developers.openai.com/blog/skills-shell-tips and apply at least one validated enhancement.
8. [ ] Chat UI-first interaction flow (Owner: TBD)
   - Acceptance criteria: Deliver a Tokimon chat UI flow that supports routine interaction without CLI invocation for each step.
9. [ ] Generalized arbitrary-goal self-improvement mode (Owner: TBD)
   - Acceptance criteria: Provide a super-concise workflow that takes an arbitrary goal and executes end-to-end self-improvement with measurable output.
10. [ ] Claude CLI learning + mixed-provider self-improve ratio (Owner: TBD)
   - Acceptance criteria: Learn from `~/clover/joey-playground/apps/` how to use Claude CLI, then enforce a `claude:codex` session mix of `1:4` in every self-improve task.
11. [ ] Ensure self-improve uses the agent loop (Owner: TBD)
   - Acceptance criteria: Confirm and enforce that Tokimon self-improve tasks run through the full agent loop (iterative model/tool cycle), not a one-shot execution path.
