# Refactoring Guideline Review

This document reviews `tmp/refactoring-guidelines.md` against Tokimon's governing specs:

- `docs/tokimon-constitution.md`
- `docs/helix.md`
- `docs/repository-guidelines.md`

It records what Tokimon should adopt as guidance for future refactors. This is a doc-only learning-round change and does not alter runtime behavior.

## Adopt now

- Preserve observable behavior; keep public APIs and user-visible behavior stable unless explicitly requested.
- Prefer small, incremental refactors; keep tests green at each step.
- Address one significant concern per change set; avoid broad, cross-cutting rewrites.
- Follow existing project style and tooling; do not introduce new dependencies for refactoring work.
- Behavior-preserving cleanup is encouraged: delete dead/duplicate code, remove unused imports/vars, reduce redundant branches, and flatten unnecessary nesting.
- Prefer straightforward designs (KISS/YAGNI); extract utilities only when duplication is clearly reduced.
- Comments should explain "why" rather than "what". Avoid stale TODOs in code; move work items into `docs/` (or the task plan) if they matter.
- Exception handling: fail fast; do not hide failures behind silent fallbacks or defaults.

## Defer (needs Tokimon-specific spec)

- "Structured JSON logs", "single logger at cold-start", "never use `print()`", and "propagate correlation IDs": adopt only after Tokimon defines an observability/logging spec (formats, boundaries, and how correlation IDs are generated and carried).
- "Exactly one stack trace per failure tied to a correlation ID": too prescriptive without a defined error boundary model and tracing design.
- "Hexagonal boundaries where applicable": apply only when an architecture doc explicitly calls for those boundaries in a specific component.

## Ignore / conflicts

- CDK-specific guidance ("CDK stacks/constructs: avoid custom logging") is out of scope unless Tokimon introduces AWS CDK infrastructure code.
- Any guideline that would require changing user-visible behavior or public APIs without a prior doc change conflicts with Helix's doc-first rule and this round's "runtime unchanged" constraint.
