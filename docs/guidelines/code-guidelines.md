---
title: "Code Guidelines"
summary: "Baseline rules for writing, changing, reviewing, and verifying repository code."
read_when:
  - Before reading, reviewing, or changing code or tests.
  - Deciding public-contract boundaries, runtime behavior, structure, or verification expectations.
---

# Code Guidelines

## Purpose
- Define baseline rules for writing, changing, reviewing, and verifying code so work honors public contracts and improves maintainability, clarity, and safety.

## Engineering Principles
- **CODE-PRINCIPLE-001:** Keep each durable fact in one authoritative owner.
  Duplicate only for explicit compatibility or local safety.
- **CODE-PRINCIPLE-002:** Use responsibility, dependency, and interface boundaries where they reduce coupling or verification cost.
- **CODE-PRINCIPLE-003:** Prefer the simplest design that preserves the public contract and expected verification path.
- **CODE-PRINCIPLE-004:** Add future-facing hooks only when an active request or owned contract needs them.
- **CODE-PRINCIPLE-005:** Fail Fast: surface invalid state and unrecoverable errors immediately instead of masking them with silent defaults.
- **CODE-PRINCIPLE-006:** Keep responsibilities separate enough that one behavior can be changed and tested without unrelated rewrites.
- **CODE-PRINCIPLE-007:** Drive and test application behavior through public ports, not runtime devices or databases.
- **CODE-PRINCIPLE-008:** Make runtime state observable through traces, metrics, logs, and contract-visible outcomes.
- **CODE-PRINCIPLE-009:** Grant users, services, and processes only the privileges needed for assigned tasks.
- **CODE-PRINCIPLE-010:** Add abstractions only when they remove real duplication or isolate a real boundary.

## Public Contracts
- **CODE-CONTRACT-001:** Treat behavior depended on by callers, operators, or automation as a public contract.
  Examples include APIs, commands, schemas, persisted artifact formats, generated outputs, import surfaces, exports, metadata, and CI/operator workflows.
- **CODE-CONTRACT-002:** Keep public surfaces stable unless an explicit request updates the owning contract.
  Shrink or remove them only with compatibility evidence, a migration path, or an explicit contract update.
- **CODE-CONTRACT-003:** Treat owning C4 documents as contract inputs for implementation review.
  If C4, code, or tests disagree, record a finding and decide whether the owning contract or implementation should change before editing.

## Structure
- **CODE-STRUCT-001:** Expose the smallest stable interface that serves the contract.
  Keep implementation details behind the owner that controls them.
- **CODE-STRUCT-002:** Keep hierarchy and seams only for real boundaries.
  Representative boundaries include domain, lifecycle, I/O, policy, verification, or reuse.
- **CODE-STRUCT-003:** Avoid false seams, pass-through modules, test-only production APIs, and shims without active compatibility value.
- **CODE-STRUCT-004:** Keep separate responsibilities in separate places.
  Representative responsibilities include parsing, validation, provider clients, orchestration, domain policy, and operator-output/report/artifact shaping.
- **CODE-STRUCT-005:** Hide external dependencies behind adapters when doing so reduces coupling or verification cost.
  Representative dependencies include external services, files, browsers, and tools.

## Runtime Behavior
- **CODE-RUNTIME-001:** Use fallbacks and defaults only when they are explicit contract or compatibility behavior; otherwise fail fast.
  Log once at the boundary, then re-raise with minimal, actionable context.
- **CODE-RUNTIME-002:** Runtime services must emit structured, correlated logs through configured loggers.
  Configure a single logger at cold start.
  Do not use `print()` for runtime logging.
- **CODE-RUNTIME-003:** CLI/operator scripts may use `print()` for intentional stdout/stderr user output.
- **CODE-RUNTIME-004:** Log records must carry durable operational meaning.
  Include timestamp, service/runtime identity, correlation, run, or task identifier, severity, and message fields.
- **CODE-RUNTIME-005:** Always propagate a correlation, run, or task identifier across boundaries.

## Code Hygiene
- **CODE-HYGIENE-001:** Favor self-explanatory code.
  Use brief comments to explain why, not what.
- **CODE-HYGIENE-002:** Preserve behavior while deleting dead/duplicate code and unused imports/vars.
- **CODE-HYGIENE-003:** Remove boilerplate and redundant branches; flatten deep nesting.
- **CODE-HYGIENE-004:** Extract shared logic into small utilities only when it clearly reduces duplication.

## Change Discipline
- **CODE-CHANGE-001:** Prefer small, incremental refactors; keep tests green at each step.
- **CODE-CHANGE-002:** Follow existing project style and tooling.
  Representative tooling includes formatters, linters, and type checks.
  Do not introduce new dependencies.

## Tests & Verification
- **CODE-VERIFY-001:** Test public behavior through the relevant public contract, not implementation shape.
- **CODE-VERIFY-002:** Keep behavior tests focused, deterministic, and cheap to run when practical.
- **CODE-VERIFY-003:** Expand coverage when change risk crosses a public or shared boundary.
  Representative triggers include shared behavior, public contracts, state formats, artifact formats, cross-layer boundaries, and external integrations.
- **CODE-VERIFY-004:** Do not add production surface only for test convenience.
- **CODE-VERIFY-005:** Name exact verification commands, expected results, and any manual checks that remain.
