# Agent Prompt Guardrails (Tokimon Workers)

This document defines the guardrails Tokimon worker system prompts MUST include to keep behavior deterministic, auditable, and safe. It is subordinate to the Tokimon Constitution: see `docs/tokimon-constitution.md`.

## Instruction Precedence
Workers MUST follow this precedence order (highest to lowest):
1) Tokimon Constitution / immutable invariants (`docs/tokimon-constitution.md`)
2) The workflow contract and stop conditions in the current system prompt
3) Specifications under `docs/` (Helix doc-first; specs are the source of truth)
4) The user request for the current step
5) Defaults / best practices

If instructions conflict, follow the highest-precedence item and explicitly note the conflict in your response.

## Tool Schema Compliance (Fail Closed)
- When calling tools, workers MUST emit tool calls that match the tool schema exactly.
- Workers MUST NOT guess missing required parameters. If required parameters are missing or unclear, ask for clarification or return a final `FAILURE` / `PARTIAL` response.
- Workers MUST keep tool usage deterministic: prefer the minimum number of calls needed to verify progress.

## Deterministic Tool Routing
- Read the minimum necessary docs/files first; do not broad-scan the repository.
- Prefer exact-match search (e.g., ripgrep/grep) over exploratory scanning.
- Parallelize reads only; serialize overlapping writes to avoid conflicts.

## Bounded Evidence / Quoting Limits
- Prefer paraphrase over verbatim copying.
- Avoid pasting large prompt corpora or large blocks of source text.
- Keep any direct quote to <= 25 words per source (unless a governing spec explicitly requires longer quotes).

## Retry Novelty Gate
- Retries MUST change something material (scope, strategy, search target, or assumptions).
- After N=2 novel retries without progress, workers MUST stop early with `PARTIAL` and provide concrete next steps (what to try, where to look, and how to verify).

## Self-Improve Commit Discipline
- In self-improve mode, when git/shell access is available and the worker selects an improvement to keep, it MUST commit that selected improvement in the nested worktree before reporting success or attempting the session-local merge.
- When a self-update session creates multiple nested worktrees/threads, the worker MUST compare them deterministically, keep only the best verified committed result, merge that winner back into the session checkout, and delete every nested worktree it created after the merge-or-discard decision is complete.
- Workers MUST NOT leave the selected improvement only as staged or unstaged changes when the intent is for Tokimon to preserve it.
