# AI Entry Point (Start Here)

Start sequence (every chat turn):
1) Read `AGENTS.md` (this file).
2) Read `docs/helix.md` — the canonical method and operating instructions.
3) Read `docs/repository-guidelines.md` — repo-specific commands and conventions.

Non-negotiables
- All method rules, principles, and the 4-step loop are defined in `docs/helix.md`.
- All repo-specific conventions (build commands, testing, commit format) are in `docs/repository-guidelines.md`.
- This file only defines reading order.
- Delegation recursion guard: delegate to Tokimon only for top-level human requests.
- Marker contract for delegated runs: Tokimon sets `TOKIMON_DELEGATED=1` and `TOKIMON_DELEGATION_DEPTH>=1`.
- Recursion rule: if `TOKIMON_DELEGATED=1` or `TOKIMON_DELEGATION_DEPTH>=1`, do not delegate again; execute directly.
- If markers are absent, treat the request as top-level and delegation is allowed (supports multiple human-started Tokimon runs).
