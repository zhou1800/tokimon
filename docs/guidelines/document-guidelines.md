---
title: "Documentation Guidelines"
summary: "Repo-wide documentation governance, live audit rules, metadata, readability, and cross-document conventions."
read_when:
  - Creating, editing, reviewing, or auditing repository documentation.
  - Deciding document ownership, duplication boundaries, or general document shape.
---

# Documentation Guidelines

This file owns general documentation rules and audit-ready rule definitions.
For C4 artifacts, follow `docs/guidelines/c4-document-guidelines.md` after this file.

## Contents

1. [Core Rules](#core-rules)
2. [Clear Writing](#clear-writing)
3. [Machine-Readable Artifacts](#machine-readable-artifacts)
4. [Frontmatter Metadata](#frontmatter-metadata)

## Core Rules
- **DOC-RESP-001:** State the document's responsibility early.
- **DOC-MOD-001:** Keep docs modular: prefer small, focused files that can be loaded independently.
- **DOC-SIGNAL-001:** Keep documentation high-signal and simple.
  Remove common sense, duplication, low-value detail, and content readers can infer from owning docs.
  Treat repository-wide defaults as implicit; document only local deltas or exceptions.
  For feature removal or demotion, delete obsolete material first; add only contract text the remaining docs cannot imply.
  Prefer concise examples over verbose explanations.
  Brevity is not a defect when behavior, context, and constraints stay clear.
- **DOC-SIGNAL-002:** Treat the context window as a shared budget.
  **Default assumption: Codex is already very smart.**
  Only add context Codex doesn't already have.
  Challenge each piece of information:
  "Does Codex really need this explanation?"
  "Does this paragraph justify its token cost?"
- **DOC-OUTCOME-001:** Shorter, outcome-first documents usually work better than process-heavy documentation stacks.
  Define the outcome, success criteria, constraints, and available context.
  Describe the destination rather than every step unless the sequence is required.
  Reserve absolute rules for non-negotiable constraints the document owns.
- **DOC-OUTCOME-002:** Architecture, contract, and catalog documents describe the accepted target behavior in present tense.
  Use process or current-status language only in documents that own execution plans, status reports, runtime diagnostics, or historical records.
- **DOC-PRECISION-001:** Match precision to the document's responsibility.
  Provide shared direction while preserving room to adapt.
- **DOC-OWN-001:** Keep durable content in one owning document.
  When editing related docs, find the owner first and change only what must change.
  Link to that owner instead of repeating it.
  Repeat only when local context is needed to make the doc usable or safe.
- **DOC-PROMPT-001:** Treat prompts and embedded payload blocks as owned by their enclosing document.
  Review them as context unless the task explicitly targets their behavior.
- **DOC-REF-001:** Keep references in dependency direction.
  Point specific docs to the definitions they follow.
  Definition docs should not depend on their implementations or consumers.
- **DOC-REF-002:** Avoid deeply nested references.
  Keep reference files one level deep from the entrypoint or routing document that asks agents to load them.
  Link such files directly from that entrypoint instead of requiring another broad reference walk.
- **DOC-REF-003:** Structure longer reference files.
  For reference files longer than 100 lines, include a table of contents at the top.
  The table of contents lets Codex see the full scope when previewing.
- **DOC-ORCH-001:** Keep orchestration references one-way.
  Orchestrators may reference what they compose.
  Reusable parts should not depend on a specific orchestrator.
- **DOC-SEC-001:** Do not store secrets, credentials, or session material in docs, skills, AI Harness Interface files, tests, or memory.
  This includes repository entrypoints and generated interface files.

## Clear Writing
- **DOC-CLEAR-001:** Write for human review first, then AI review.
  Use simple words, clear priority, and concise structure.
- **DOC-CLEAR-002:** Prefer progressive disclosure.
  Summarize first, then expand with details in-place.
- **DOC-CLEAR-003:** State requirements before mechanisms.
  Explain non-obvious constraints.
- **DOC-CLEAR-004:** Keep prose easy to scan.
  Avoid mega-sentences, nested clauses, and forced line wrapping.
  Use one sentence per line when it improves reviewability.
- **DOC-CLEAR-005:** Use structure only when it improves clarity.
  Use bullets for comparable items and tables for real comparisons.
- **DOC-CLEAR-006:** Prefer scope and expected behavior over lists of examples or exclusions.
  Do not use partial lists that look complete.
  Use concrete lists only when they are complete or owned here.
  If examples help, mark them non-exhaustive and keep them representative.
- **DOC-CLEAR-007:** Keep headings and labels clear.
  Use short Title Case headings.
  Use sentence case for bold label keys that end with a colon.
- **DOC-CLEAR-008:** Prefer domain framing and durable meaning.
  Avoid implementation-first inventories.
  Do not document code-obvious details; they drift.
- **DOC-CLEAR-009:** Preserve useful wording and tone when refining existing docs.
  Do not rewrite working documentation solely for formatting preferences.
- **DOC-CLEAR-010:** Write and revise technical documentation at about 80% ASD-STE100.
  Use short, direct sentences with one main idea.
  Prefer active voice.
  Use plain words and clear references.
  Use structure only when it improves understanding.
- **DOC-CLEAR-011:** Use one consistent term for each concept.
  Preserve technical meaning, including important constraints and uncertainty.
  Keep identifiers and literal values unchanged during prose edits.
- **DOC-CLEAR-012:** State requirements and expected behavior explicitly.
  Make relevant conditions and consequences clear.

## Machine-Readable Artifacts
- **DOC-MACHINE-001:** Use machine-readable contract artifacts where they fit naturally.
  Treat artifacts as canonical and link to them from prose.
- **DOC-MACHINE-002:** Use prose to explain what artifacts cannot express clearly.
- **DOC-MACHINE-003:** Support traceability and automation with explicit structure.
  Do not contort prose for automation or model parsing.

## Frontmatter Metadata

Use frontmatter only as stable routing metadata for stable documents.
The file body remains the source of truth.

- **DOC-FM-001:** Stable docs may use `title`, `summary`, and `read_when`.
  `summary` states what the document owns.
  `read_when` states when an agent should read it.
- **DOC-FM-002:** Skill entrypoints under `skills/<skill-name>/skill.md` use `name` and `description`.
- **DOC-FM-003:** `AGENTS.md` may stay frontmatter-free because it is a portable repository entrypoint.
