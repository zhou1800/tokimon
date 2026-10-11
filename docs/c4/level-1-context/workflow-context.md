---
title: "Workflow Layer - Context"
summary: "L1 context for the Tokimon workflow layer and repeatable domain process boundaries."
read_when:
  - Adding, changing, or reviewing workflow packages.
  - Deciding whether repeated work belongs in workflow code, tools, memory, skills, or operations docs.
---

# Workflow Layer - Context

This document owns the L1 context boundary for the Tokimon workflow layer.

## Overview & Mission

The workflow layer turns repeated work into reusable domain processes.
It is the third layer of the Tokimon architecture and owns repeatable process contracts plus their local runtime artifacts.
Code and document audits share pure Markdown rule scanning and report-file reading.
Each audit owns its source selection, provenance, convergence rules, and ledger semantics.

## Actors

### User

The user chooses which workflow to run, reviews produced artifacts, and makes final decisions where automation must remain conservative.

### Codex Session

The Codex session runs or follows workflow implementation.
It searches durable memory and uses declared or task-selected tools when needed.
It reports workflow outputs and durable learnings.

### Tools Layer

The tools layer receives capability requests without requiring workflows to hard-code host assumptions.

### Memory Layer

The memory layer receives explicit recall and durable-memory writes from workflows.

## Dependencies

### Repository Components

- **Repository Guidelines (repository-guidelines):** route workflow readers to owning C4, operations, and guideline documents.
- **Memory Layer (memory):** supplies persisted context and explicit recall.

### Agent Interfaces

- **Tools Layer (tools):** supplies registered local or MCP execution capabilities and their readiness contracts.

## Scope Boundaries

### In Scope

- Declaring workflow-specific tool use where the activating skill or package metadata does not already own it.
- Searching memory explicitly when prior run context matters.
- Owning workflow-specific docs, prompts, schemas, replay cases, and artifacts.
- Defining workflow outputs and review contracts.
- Keeping generated workflow state and artifacts under `.tokimon/workflow/<workflow-id>/` by default.
- Recording durable learnings in their owning documentation or local memory.

### Out of Scope

- Owning generic local or MCP tools.
- Owning base memory retrieval policy.
- Treating every script as a workflow before it has a repeatable process contract.
- Hiding domain-specific prompts or rules in C4 when they belong to a workflow package.
- Writing workflow credentials, cookies, browser session state, or API tokens into repository files.

## Traceability

- `docs/guidelines/repository-guidelines.md`
- `docs/c4/level-1-context/memory-context.md`
- `docs/c4/level-1-context/tools-context.md`
- `docs/c4/level-2-containers/tools/tool-registry.md`
