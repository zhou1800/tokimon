---
title: "AI Harness Design Philosophy"
summary: "Repository-wide design philosophy for responsibility across harness layers and decision authority in documentation."
read_when:
  - Deciding how much detail belongs in a document and which decisions it should own.
  - Deciding whether responsibility belongs in workflows/tools, skills/AI agents, or humans.
  - Designing or reviewing AI Harness responsibility boundaries.
---

# AI Harness Design Philosophy

## Responsibility Across Harness Layers

AI Harness design separates deterministic machinery, agent judgment, and human authority.

The goal is to move repeatable responsibility into stable workflow and tool programs, keep the AI agent in the driver seat for judgment-heavy work, and reserve humans for direction, goals, and authorization.

### Workflow and Tool Layer

Deterministic programs provide consistent operations, command discovery, validation, and clear error output.

This layer should be as fat as practical because every reliable workflow or tool reduces agent context load and makes results more repeatable.

### AI Agent Layer

Skills define what the AI agent should do.

The agent follows skills and uses workflows or tools whenever deterministic support can reduce context pollution, improve reliability, or make progress auditable.

The agent is smart but unreliable, so it should keep only the responsibilities that cannot reasonably be moved into deterministic programs.

### Human Layer

Humans provide goals, direction, and authorization.

AI Harness Interface should keep this layer as thin as risk allows.

More human involvement lowers risk but slows the process; moving more work to the agent increases speed and autonomy while increasing risk.

AI Harness design balances these layers so humans are not overused, agents are not trusted with unnecessary deterministic work, and deterministic programs carry as much stable responsibility as possible.

## Documentation and Decision Authority

A document's meaning emerges from the interaction of its text, its context, and those who interpret it.
That meaning evolves with time and use.

Good documentation creates shared direction while preserving room to adapt.
Precision stabilizes what needs continuity; openness keeps judgment responsive to circumstances.

The right balance depends on where decision authority belongs.
Each document settles what belongs at its level and leaves the rest to whoever is responsible for deciding it.
