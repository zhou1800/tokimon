---
title: "Agent Skills - Context"
summary: "L1 context for repo-shareable agent skills."
read_when:
  - Adding, moving, renaming, or using repo-shareable agent skills.
  - Deciding whether repeatable agent operating guidance belongs in skills or another Tokimon surface.
---

# Agent Skills - Context

This document owns the L1 context boundary for repo-shareable agent skills.

## Overview & Mission

Agent skills are repo-shareable operating guides for AI agents.
They help agents find and apply repeatable practices that should travel with the repository.

## Actors

### User

The user may ask an agent to use a skill explicitly or may describe work that matches a skill.

### Agent

The agent discovers matching repo skills through `AGENTS.md`, then reads the matching skill entrypoint before planning or editing.

### C4 Docs

C4 owns the shared skill design and contract boundary.
Guidelines own shared document-authoring conventions.
Skills are self-contained operating manuals for normal use.

## Dependencies

### Repository Components

- **Workflow Contracts (workflow):** own repeatable process behavior that workflow-backed skills route to instead of duplicating.

### Agent Interfaces

- **Skill Entrypoints (skills):** provide the repo-shareable `skills/<skill-name>/skill.md` guides that agents load when requested or matched.

## Scope Boundaries

### In Scope

- Storing repo-shareable agent operating guides under `skills/`.
- Defining where agents find skill entrypoints.
- Keeping skill layout rules shareable across machines.
- Keeping skill entrypoints self-contained for normal use.
- Declaring Codex presentation, implicit-invocation policy, and tool dependencies in optional `agents/openai.yaml` metadata.
- Keeping skill-specific local artifacts under `.tokimon/skills/<skill-name>/`.

### Out of Scope

- Defining a fourth implementation layer.
- Replacing C4 architecture or workflow contracts.
- Replacing tools registry capability declarations.
- Replacing tests or verification commands.
- Owning machine-local preferences that would make skills non-shareable.

## Traceability

- `AGENTS.md`
- `docs/guidelines/repository-guidelines.md`
- `docs/c4/level-2-containers/skills/skills.md`
- `skills/<skill-name>/skill.md`
