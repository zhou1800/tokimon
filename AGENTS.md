# Tokimon Repository Entry

This file routes startup, memory, guideline, and operating rules for Tokimon repository work.

## Progressive Disclosure

- Load user-specified docs first.
- Then read only the minimal owning docs needed before choosing repository capabilities.
- Read implementation files only after the owning docs identify the scope or when the user explicitly points to them.

## Memory

- Use Mem Palace as Tokimon's authoritative project memory surface.
- Run one bounded Mem Palace briefing before substantive Tokimon-enabled work unless this session or task already has a current briefing.

```bash
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace briefing --scope tokimon --limit 8
```

- Treat Codex memory as supplemental; Mem Palace remains authoritative.
- Search Mem Palace before answering prior-context questions.

```bash
.venv/bin/python -m tokimon.memory.mem_palace.mem_palace search "<keywords>"
```

- Before the final response on substantive work, evaluate whether future Tokimon sessions should reuse what was learned.
  If yes, write a concise memory through the Mem Palace CLI and verify it with `recent` or `search`.
- Load `skills/memory/skill.md` for memory work or deeper memory routing.

## General Guidelines

- Follow `docs/guidelines/language-and-communication-guidelines.md` for language and input-interpretation rules.
- Before making changes, read `docs/guidelines/helix.md`.
- Before installing, reinstalling, refreshing, or updating Tokimon for an AI platform, read `docs/operations/install.md`.
- Before creating, editing, reviewing, or auditing documentation, read `docs/guidelines/document-guidelines.md`.
- Before changing code or tests, read `docs/guidelines/code-guidelines.md`.
- Read `docs/guidelines/repository-guidelines.md` before choosing repository capabilities.

## Operating Rules

- Follow the repository-level `.venv`, `pip`, and `.tokimon/` runtime policy.
- Follow the browser-surface selection policy in `skills/using-tokimon/skill.md`.
- Do not modify files, install anything, or change system settings outside this repository root.
  Only do so when the task explicitly requires it.
- In linked worktrees, the primary worktree `.venv` and `.tokimon/` paths named by the repository guidelines count as repository-owned paths.
