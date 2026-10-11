---
title: "Memory Layer - Context"
summary: "L1 context for the Tokimon memory layer, C4 repository self-description, and Mem Palace local durable memory."
read_when:
  - Starting memory architecture work or deciding where durable context belongs.
  - Reviewing Mem Palace, C4 repository memory, or memory-layer boundaries.
---

# Memory Layer - Context

This document owns the L1 context boundary for the Tokimon memory layer.

## Overview & Mission

The memory layer is the first layer of the Tokimon architecture.
It gives agents enough durable context to understand the active repository and recover stable local decisions without scanning prior chat transcripts.

The memory layer has two meaningful memory bodies:

- **C4 repository self-description:** the git-tracked architecture docs under `docs/c4/`.
- **Mem Palace:** the accepted local durable memory system for scoped records, ranked recall, bounded briefing, temporal entity facts, exchange, explicit artifact ingestion, and recall evaluation.

## Actors

### User

The user expresses durable preferences, decisions, and corrections that may need to be retrieved in future sessions.
The user does not read or write Mem Palace directly.
Durable input reaches memory through the Codex session and Mem Palace CLI, then returns through ranked recall in later sessions.

### Codex Session

The Codex session reads the repository guidelines plus a bounded Mem Palace briefing at startup, searches memory for history-dependent questions, and stores durable local facts through the Mem Palace CLI.

### Future Codex Sessions

Future sessions consume C4 and Mem Palace to recover stable context cheaply and consistently.

## Dependencies

### Repository Components

- **C4 repository self-description (c4-repository-memory):** provides the git-tracked architecture context used as the repository memory body.
- **Mem Palace (mem-palace):** provides the local durable memory body for scoped recall and continuity.

## Scope Boundaries

### In Scope

- Reading repository guidance and bounded Mem Palace memory at startup.
- Reading and writing local durable memory through Mem Palace.
- Querying scoped memory records and temporal entity facts through the Mem Palace CLI or Python API.
- Backing up, exporting, importing, explicitly ingesting artifacts into, and evaluating the local memory store.
- Providing bounded, explicit search and briefing commands for local continuity.

### Out of Scope

- Re-describing C4 as a memory container at L2 or L3.
- Treating repository code as memory.
- Treating workflow docs or tool docs as separate base memory bodies.
- Automatic ingestion of every repo file or prior chat.
- Owning generic repository security policy.

## Traceability

- `docs/guidelines/repository-guidelines.md`
