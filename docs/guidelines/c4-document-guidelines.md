---
title: "C4 Document Guidelines"
summary: "Machine-loadable C4 authoring rule catalog for L1, L2, and L3 C4 docs."
read_when:
  - Loading the canonical C4 rule source for automated review.
  - Creating, editing, reviewing, or auditing C4 documents under `docs/c4/`.
  - Deciding L1/L2/L3 structure, diagrams, container fields, data stores, references, or C4 layer boundaries.
---

# C4 Document Guidelines

This document defines layer-specific authoring conventions for C4 docs.
Follow `docs/guidelines/document-guidelines.md` first for repo-wide documentation rules, then apply this file for C4 artifacts.
This file is also the canonical machine-loadable C4 rule source for automated review.
Load the core rules plus only the C4 layer section that matches the target document.

## Contents
1. [Core Rules](#core-rules)
2. [Layer-Separated Documentation](#layer-separated-documentation)
3. [L1 Context Conventions](#l1-context-conventions)
4. [L2 Container Conventions](#l2-container-conventions)
5. [L3 Component Conventions](#l3-component-conventions)

## Core Rules
- **C4-LAYER-001:** Use C4 to separate concerns so humans and AI can load only relevant context.
- **C4-REQ-001:** Write each capability requirement once in the module that owns the visible behavior.
  Avoid duplicated requirements and cross-reference by links/IDs.
- **C4-TARGET-001:** C4 documents describe accepted target architecture, ownership boundaries, and committed responsibilities.
  Do not use C4 documents as migration-progress, current-implementation-status, or process ledgers.

## Layer-Separated Documentation
- **C4-LAYER-002:** Keep major capability families in separate C4 document sets when they have distinct ownership, runtime boundaries, or audiences.
  Separation still applies when multiple families implement one end-to-end capability.
- **C4-LAYER-003:** Default scope: each C4 document describes the ownership boundary for its C4 layer.
  It also describes the responsibilities owned at that boundary.
- **C4-LAYER-004:** Describe cross-layer behavior by mapping contracts and interactions.
  Link to the owning docs instead of merging every layer's runtime details into one document.

## L1 Context Conventions

### L1 Context Document Structure
- **C4-L1-STRUCT-001:** C4 L1 context docs live under `docs/c4/level-1-context/<capability>.md`.
- **C4-L1-STRUCT-002:** Title should end with `Context` (for example, `# Payments - Context`).
- **C4-L1-STRUCT-003:** Use this top-level section order:
  - `## Overview & Mission`
  - `## Actors`
  - `## Dependencies`
  - `## Scope Boundaries`
  - `## Traceability`
- **C4-L1-STRUCT-004:** Under `## Scope Boundaries`, include both `### In Scope` and `### Out of Scope`.

### L1 Overview & Mission Convention
- **C4-L1-OVERVIEW-001:** Start with what the capability is responsible for and why it exists.

### L1 Actors Convention
- **C4-L1-ACTOR-001:** Under `## Actors`, define each actor as a separate `###` subsection.
- **C4-L1-ACTOR-002:** Actor headings may include role qualifiers in parentheses (for example, `(Primary)` or `(Secondary)`).
- **C4-L1-ACTOR-003:** For each actor, document goals and interaction boundaries with the capability.
  Call out indirect access paths when applicable.
  For example, an operator workflow may use an integration registry rather than hard-coding provider paths.

### L1 Dependency Convention
- **C4-L1-DEP-001:** In `## Dependencies`, group dependencies by role and omit empty groups.
  - `### External Systems`: third-party endpoints, host-local services, or dependencies outside the workspace.
  - `### Platform Services`: cloud, SaaS, data, model, compute, or managed-service dependencies.
  - `### Repository Components`: local repositories, scripts, docs, generated artifacts, or component outputs consumed by the capability.
  - `### Operational Workflows`: repeatable runbooks, CI jobs, or human/operator workflows.
  - `### Agent Interfaces`: prompts, skills, tools, MCP servers, plugins, connectors, or inbound integration surfaces used by the capability.
- **C4-L1-DEP-002:** Prefer system bullets in this form:
  - `- **System Name (capability-id):** relationship purpose.`

### L1 Scope Boundaries Convention
- **C4-L1-SCOPE-001:** `In Scope` bullets capture committed responsibilities and measurable expectations.
  Include performance/NFR targets when relevant.
- **C4-L1-SCOPE-002:** `Out of Scope` bullets capture explicit exclusions, deferred work, and non-goals.
- **C4-L1-SCOPE-003:** Keep boundary statements capability-specific and avoid repeating generic repository rules.

### L1 Traceability Convention
- **C4-L1-TRACE-001:** When story IDs or TODO IDs exist, map them explicitly in the traceability bullets.
- **C4-L1-TRACE-002:** Add extra references only when they materially clarify ownership boundaries.
  Links should be to owning guidelines, operations docs, and related C4 docs.

## L2 Container Conventions

### L2 Container Document Structure
- **C4-L2-STRUCT-001:** C4 L2 container docs live under `docs/c4/level-2-containers/<capability>.md`.
  A multi-file container set may live under `docs/c4/level-2-containers/<capability>/*.md`.
- **C4-L2-STRUCT-002:** Title should end with `Containers` (for example, `# Payments - Containers`).
- **C4-L2-STRUCT-003:** Use this top-level section order:
  - `## Overview`
  - `## Container Diagram`
  - `## Containers`
  - `## Key Interactions`
  - `## Security & Trust`
  - `## Assumptions` (optional; include when assumptions materially affect architecture or contracts).
  - `## Data Stores`
  - `## References`
- **C4-L2-STRUCT-004:** L2 docs must be readable without opening deeper-level C4 docs.

### L2 Overview Convention
- **C4-L2-OVERVIEW-001:** Start with 1-2 sentences that name the capability (and capability ID when applicable) and the primary boundary.
- **C4-L2-OVERVIEW-002:** Add one `At a high level, ...` paragraph.
  Summarize local/runtime scope plus the main containers and integration paths in 2-4 short sentences.
- **C4-L2-OVERVIEW-003:** If the overview needs an inventory (more than 3 items), use a bullet list instead of a single sentence.
- **C4-L2-OVERVIEW-004:** Include `**Responsibilities**` and `**Non-responsibilities**` blocks to make ownership boundaries explicit.

### L2 Container Diagram Convention
- **C4-L2-DIAGRAM-001:** Use topology labels for cross-boundary subgraphs when they clarify container relationships:
  - `External Systems`
  - `Repository foundations`
  - `Upstream producers`
  - `Downstream consumers`
  - `Agent access path`
- **C4-L2-DIAGRAM-002:** Prefer `<Capability> - <capability-id>` for the primary capability subgraph label when a capability ID exists.
- **C4-L2-DIAGRAM-003:** Use solid arrows (`-->`) for runtime request, command, artifact, event, and data flows.
- **C4-L2-DIAGRAM-004:** Use dashed arrows (`-.->`) only for non-runtime/static relationships.
  Representative relationships include command-contract mappings, package wiring, and model-contract wiring.
- **C4-L2-DIAGRAM-005:** Arrow labels are optional.
  Add labels when they materially improve clarity and omit them for obvious relationships.

### L2 Container Convention

**Type**
- **C4-L2-TYPE-001:** Use only these values for `- **Type:**` in L2 docs:
  - `Infrastructure`
  - `CLI Runtime`
  - `API Runtime`
  - `Workflow Runtime`
  - `Tool Runtime`
  - `Tool Adapter`
  - `Import Adapter`
  - `Agent Operating Guide Collection`
  - `Inbound Integration Interface Collection`
  - `Data Store`
  - `Managed Service`
  - `External System`
  - `Tooling/Test`
- **C4-L2-TYPE-002:** Put extra qualifiers in `Responsibility`/`Technology`, not in `Type`.
  Representative qualifiers include "local-only", "MCP-backed", "workflow-backed", and "agent-facing".

**Responsibilities**
- **C4-L2-RESP-001:** Keep `Responsibilities` and `Non-responsibilities` runtime/container-specific.
- **C4-L2-RESP-002:** Avoid restating L1 `Scope Boundaries` in L2 responsibility sections.

**Technology**
- **C4-L2-TECH-001:** Every container in L2 docs must include `- **Technology:**`.
- **C4-L2-TECH-002:** Use a single-line technology summary when the container has one primary technology.
- **C4-L2-TECH-003:** Use a `- **Technology:**` label followed by indented bullets when the container uses multiple technologies.
- **C4-L2-TECH-004:** Use consistent terminology for repeated technologies across L2 docs.
  Representative technologies include cloud providers, data platforms, MCP, infrastructure tooling, container runtimes, programming languages, and document formats.

**Data field**
- **C4-L2-DATA-001:** Use the exact field label `- **Data:**` for every container entry.
  For key-pattern enumerations, use `- **Data:**` followed by one-level bullets.
- **C4-L2-DATA-002:** Keep container `Data` content scoped to the data surfaces handled by that container.
- **C4-L2-DATA-003:** Include access mode/ownership only when it improves clarity.
  Allowed values are `Owned`, `Read-only`, `Read/write`, `Emitted`, and `Transient`.

**Key interfaces**
- **C4-L2-INTERFACE-001:** Use `Key interfaces` to document stable integration surfaces that other containers or agents depend on.
- **C4-L2-CATALOG-001:** Capture the concrete integration surfaces other containers depend on.
  Representative surfaces include commands, package entrypoints, capability IDs, artifact paths, schemas, runtime roots, and key patterns.
  Wrap stable identifiers/literals in backticks.
- **C4-L2-CATALOG-002:** In L2, document interface families and the integration identifiers consumers need at the container level.
- **C4-L2-CATALOG-003:** When listing identifiers, put the family in one bullet and list identifiers as second-level bullets.
  Avoid inline parenthetical lists.
- **C4-L2-CATALOG-004:** Put full catalogs in the owning L3 component doc; do not inline them in L2.
  Representative full catalogs include complete command lists, schema inventories, artifact inventories, and key-pattern enumerations.
- **C4-L2-CATALOG-005:** Use either a single inline sentence after the label or one or more second-level bullets.
  If none exist, state `- None.` or `- None at runtime.`.

### L2 Key Interactions Diagram Convention
- **C4-L2-FLOW-001:** In `## Key Interactions`, each `###` flow subsection must include its own Mermaid sequence diagram.
- **C4-L2-FLOW-002:** Place the diagram directly below the numbered flow steps for that subsection.
- **C4-L2-FLOW-003:** Do not use one combined sequence diagram to represent multiple `###` flow subsections.
- **C4-L2-CONSUME-001:** End `## Key Interactions` with `### Consumption Patterns` (may be prose-only).
- **C4-L2-CONSUME-002:** In `### Consumption Patterns`, use exactly two labels in order.
  The labels are `**Primary consumers:**` and `**Dependencies consumed:**`.
- **C4-L2-CONSUME-003:** Keep `Consumption Patterns` scoped to cross-cutting consumption surfaces.
  Do not restate flows already covered by the `### Key Interactions` flow subsections above.
- **C4-L2-CONSUME-004:** Under `**Dependencies consumed:**`, use the L1 dependency labels in L1 order; omit empty groups.
- **C4-L2-CONSUME-005:** For each label, use either one sentence or second-level bullets (match `Key interfaces` blocks).
- **C4-L2-CONSUME-006:** In `**Primary consumers:**`, state who consumes what (consumer + interface/artifact/contract consumed).

### L2 Security & Trust Convention
- **C4-L2-SEC-001:** Every L2 container doc must include a `## Security & Trust` section.
- **C4-L2-SEC-002:** Keep L2 security bullets capability-specific.
  Do not restate baseline controls that already live in `AGENTS.md`, `docs/guidelines/helix.md`, or the owning guideline document.
- **C4-L2-SEC-003:** Document explicit baseline exceptions in the owning L2 file.
  Representative exceptions include cross-machine execution, authenticated browser access, and local-only runtime state.

### L2 Assumptions Convention
- **C4-L2-ASSUME-001:** Write assumptions as a short bullet list where each item starts with `- **Assumption:** ...`.
- **C4-L2-ASSUME-002:** Keep assumptions concrete and checkable.
  Representative assumptions include required local setup, required command contracts, and dependency availability.

### L2 Data Stores Convention
- **C4-L2-STORE-001:** List `Owned stores` first.
- **C4-L2-STORE-002:** Include `Dependency stores` only when needed.
- **C4-L2-STORE-003:** For each `Owned stores` or `Dependency stores` label, use one inline sentence or second-level bullets.
  Match the style used by `Key interfaces` blocks.
- **C4-L2-STORE-004:** Avoid boundary-only statements without concrete persistence surfaces.
- **C4-L2-STORE-005:** Keep container-level `Data:` fields focused on per-container handled data.
  Keep `Data Stores` focused on persistent ownership and dependencies.

### L2 References Convention
- **C4-L2-REF-001:** List references as bullets using a category prefix and a repo-relative doc path in backticks.
  Representative prefixes include `L1 Context`, `Guidelines`, `Operations`, and `Contracts`.
- **C4-L2-REF-002:** Do not add deeper-level C4 document references to L2 unless the L2 document is the routing owner for a capability set.
  Deeper-level docs should reference their owning L2 doc.

## L3 Component Conventions

### L3 Component Document Structure
- **C4-L3-STRUCT-001:** C4 L3 component docs live under `docs/c4/level-3-components/<capability>/*.md`.
- **C4-L3-STRUCT-002:** Every L3 doc must include these core top-level sections, in this order relative to each other:
  - `## Overview`
  - `## Components`
  - `## Interfaces/Contracts`
  - `## Data Ownership`
  - `## Failure Modes`
  - `## Acceptance Checks`
  - `## References`
- **C4-L3-DUPE-001:** Avoid duplicating catalogs in non-`/docs` READMEs.
  Prefer a short pointer to the owning L3 doc and keep the catalog in `## Interfaces/Contracts`.

### L3 Overview Convention
- **C4-L3-OVERVIEW-001:** Include exactly what a reader needs to orient.
  Cover purpose, boundary/ownership, and the primary outcome the component provides.
- **C4-L3-OVERVIEW-002:** Keep `## Overview` stable over time and independent of implementation details.
  Do not mention source paths, class names, or framework wiring.
- **C4-L3-OVERVIEW-003:** Avoid catalogs and long lists in `## Overview`.
  Put exact identifiers and enumerations in `## Interfaces/Contracts` and `## Data Ownership`.

### L3 Components Convention
- **C4-L3-COMP-001:** Use `## Components` to explain how the component is composed at a conceptual level.
  Do not use it as a code/module inventory.
- **C4-L3-COMP-002:** Each component entry must state its responsibility and the key surfaces it touches.
  Prefer pointing to the owning catalogs over inlining identifiers.
- **C4-L3-COMP-003:** When the component list is long, group by responsibility areas rather than listing helpers.

### L3 Interfaces/Contracts Convention
- **C4-L3-INTERFACE-001:** Treat catalogs as complete lists of stable identifiers.
  Put them in L3 `## Interfaces/Contracts` and keep them owned and non-duplicated.
- **C4-L3-INTERFACE-002:** Prefer stable interface families, such as commands, package APIs, artifact families, and event families.
  Add the full identifier list only when consumers require enumeration.
- **C4-L3-INTERFACE-003:** Link to canonical machine-readable contract artifacts and treat the artifact as the source of truth.
  Representative artifacts include JSON Schema, package declarations, CLI contracts, and generated artifacts.
- **C4-L3-INTERFACE-004:** Wrap stable identifiers in backticks.
  Representative identifiers include commands, package names, capability IDs, artifact paths, environment variables, and key patterns.

### L3 Data Ownership Convention
- **C4-L3-DATA-001:** In `## Data Ownership`, state what data the component owns and what it consumes read-only.
  Also state what data the component derives or transforms.
- **C4-L3-DATA-002:** When the component owns no persistent data, state that explicitly.

### L3 Failure Modes Convention
- **C4-L3-FAIL-001:** Document expected degraded behavior and recovery paths for the component.
  Representative cases include timeouts, retries, idempotency, local runtime cleanup, and manual ops hooks.
- **C4-L3-FAIL-002:** If a runtime has an error sink, document what triggers it and how operators recover.
  Representative sinks include quarantine files, failed run artifacts, dead-letter events, and rejected outputs.

### L3 Acceptance Checks Convention
- **C4-L3-ACCEPT-001:** In `## Acceptance Checks`, provide at least one copy-pasteable verification command.
- **C4-L3-ACCEPT-002:** In `## Acceptance Checks`, also provide at least one copy-pasteable verification prompt.
  The prompt should let an AI coding agent validate the component.
- **C4-L3-ACCEPT-003:** For each check, state the expected observable outcome.
  Representative observable outcomes include files produced, logs printed, resources created, exit code, and important output fields.
- **C4-L3-VERIFY-001:** If an L3 doc includes verification guidance outside `## Acceptance Checks`, keep it subordinate.
  Use copy-pasteable commands only when behavior can be checked locally.
- **C4-L3-VERIFY-002:** Verification commands outside `## Acceptance Checks` should follow `C4-L3-ACCEPT-003`.
  Apply that rule's observable-outcome requirement.
  Do not repeat outcomes already covered there.

### L3 References Convention
- **C4-L3-REF-001:** Include the owning L2 container doc (required).
- **C4-L3-REF-003:** Include the L1 context doc when it materially clarifies scope, contracts, or NFRs.
  Include relevant operations or guidelines docs under the same condition.
- **C4-L3-REF-004:** Include guideline references only when relevant.
  Prefer the single canonical reference over repeating the same link set in every doc.
