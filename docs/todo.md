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
