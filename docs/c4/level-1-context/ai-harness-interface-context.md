---
title: "AI Harness Interface - Context"
summary: "Tokimon plugin packages for Codex and Claude Code."
read_when:
  - Starting AI Harness Interface architecture work.
  - Checking package, activation, route injection, or live-checkout routing boundaries.
---

# AI Harness Interface - Context

## Overview & Mission

AI Harness Interface packages Tokimon skills and MCP entries for Codex and Claude Code.
Installed skills call live Tokimon code; copied package changes require a platform refresh.

## Actors

### Operator

Authorizes installation or refresh, follows the returned instructions, and verifies package readiness.

### Target AI Platform Agent

Uses installed skills and MCP entries to invoke live Tokimon code.

## Dependencies

### External Systems

- **Codex and Claude Code (target-ai-platform):** Load installed packages.

### Repository Components

- **Root Skills (root-skill-inventory):** Supply installed guidance, including startup and memory.
- **MCP Tool Contracts (mcp-tool-contracts):** Supply packaged tool entries.

## Scope Boundaries

### In Scope

- Generate platform packages with supported skills and MCP entries selected by the installation policy; hosts without Docker omit Docker-dependent surfaces.

### Out of Scope

- Tool execution, service data, and credentials stay with their owning tools and accounts.

## Traceability

- [Installation](../../operations/install.md).
