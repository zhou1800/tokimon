---
name: using-tokimon
description: Use when starting any conversation in an environment with Tokimon installed.
---

<!-- Selected upstream wording: Superpowers using-superpowers skill, copyright (c) 2025 Jesse Vincent, MIT License. See ../../licenses/Superpowers-MIT.txt and ../../THIRD_PARTY_NOTICES. -->

## Startup Contract

Before any substantive Tokimon-enabled work, you MUST complete the startup obligations below.
You do not have a choice: complete the startup obligations before answering, editing files, running workflows, choosing capabilities, or asking clarifying questions.

This is not optional.
You cannot rationalize your way out of it.

Treat the user's current repository and request as the task boundary.
Load only the local routing and skill guidance the task needs.

## Using Skills

If a Tokimon skill applies, you MUST follow it exactly.
You do not have a choice.

Instructions say WHAT, not HOW. "Add X" or "Fix Y" doesn't mean skip workflows.
Do not skip, reorder, soften, reinterpret, or substitute required steps.
If a required step is impossible, report the blocker instead of silently improvising.

Do not claim compliance unless you actually followed the skill's required steps, guardrails, and output contract.

## Progressive Disclosure

- Load user-specified docs first.
- Then read only the minimal owning docs needed before choosing repository capabilities.
- Read implementation files only after the owning docs identify the scope or when the active task needs them.

## Memory

- Use Mem Palace as Tokimon's authoritative project memory surface.
- Run one bounded Mem Palace briefing before substantive Tokimon-enabled work unless this session or task already has a current briefing.

```bash
{{tokimonCommand}} -m tokimon.memory.mem_palace.mem_palace briefing --project-root "$PWD" --include-scope global --limit 8
```

- Treat Codex memory as supplemental; Mem Palace remains authoritative.
- Search Mem Palace before answering prior-context questions.

```bash
{{tokimonCommand}} -m tokimon.memory.mem_palace.mem_palace search "<keywords>" --project-root "$PWD" --include-scope global
```

- Before the final response on substantive work, evaluate whether future Tokimon sessions should reuse what was learned.
  If yes, write a concise memory through the Mem Palace CLI with `--project-root "$PWD"` and verify it with project-aware `recent` or `search`.
- Use `--scope global` instead of `--project-root` only for genuinely cross-project preferences or facts.
- Use [$tokimon:memory] skill for memory work or deeper memory routing.

## General Guidelines

- Follow `docs/guidelines/language-and-communication-guidelines.md` for language and input-interpretation rules.
- Before reviewing or making changes, read `docs/guidelines/helix.md` from the Tokimon checkout.
- Before creating, editing, reviewing, or auditing documentation, read `docs/guidelines/document-guidelines.md`.
- Before changing code or tests, read `docs/guidelines/code-guidelines.md`.
- Before choosing repository capabilities, follow the target repository's local repository guidelines.
  The usual entrypoint is `docs/guidelines/repository-guidelines.md` or its equivalent.

## Browser Surface Selection

Use one browser surface per Agent task.
Unless the user or task requires a specific surface, choose the first suitable and available option:

1. `In-App Browser` or an equivalent platform-native surface.
2. `Connected Chrome` or an equivalent connected browser.
3. `mcp.remote-browser-broker`.

Choose Connected Chrome first only for required existing tab, profile, session, or local context.
Choose the broker first only when native surfaces cannot satisfy the task or broker-owned behavior is required.
Websites do not change this order; skills and workflows may add capability requirements, not competing defaults.
Keep the selected surface until it cannot continue or the task requires a switch; no separate confirmation is needed within the authorized task.

## Operating Rules

- When installing or refreshing Tokimon, do not install external dependency software, provider CLIs, or container images. Disable unavailable optional capabilities and report essential blockers.
- Use only the surfaces selected by the installed package. Do not restore omitted Docker capabilities or use Docker as a browser fallback on a Docker-disabled installation.
- Do not modify git state unless the user explicitly requests a git action.
- In linked worktrees, the primary worktree `.venv` paths named by the repository guidelines count as repository-owned paths.
