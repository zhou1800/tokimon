# Helix Lite

Helix Lite is Tokimon v2's AI-facing development method.

It keeps the useful part of doc-driven development: explicit architecture, contracts, and evaluation criteria that an AI can retrieve quickly. It removes the brittle part: treating unreviewed prose as a higher authority than executable behavior.

## Truth Order

When sources disagree, resolve them in this order:

1. User instruction
2. Executable checks: tests, benchmarks, schemas, CLI contracts
3. Active AI-facing docs under `docs/`
4. Current implementation details
5. Archived v1 materials under `archive/`

Docs should explain and constrain the system, but they do not outrank verified behavior unless the user explicitly says the docs are the source to follow.

## Core Loop

Use this loop for normal work:

1. Read the minimum relevant docs and code.
2. Identify the behavior, contract, or evaluation that matters.
3. Implement the smallest coherent change.
4. Verify with tests, benchmarks, or a direct CLI run.
5. Update docs so the next agent sees the current truth.

## Rules

- Do not write ceremonial docs. Every doc section should do at least one of these things:
  - define behavior
  - define a contract
  - define an evaluation rule
  - compress architecture for future agents
- Do not block on missing docs when the user intent is clear and the behavior can be encoded in tests or code.
- Do not preserve stale docs just because they existed first.
- Prefer small patches over large regeneration unless a real generator exists and is already part of the repo.
- Claims like "better than mainstream AI" must be backed by benchmarks, not prose.
- If you make an assumption because nobody will review the docs manually, keep it narrow and encode it in an executable check when possible.

## C4 For AI

Tokimon uses C4 as compact retrieval context for AI agents, not as documentation theater.

Keep C4 docs:

- short
- current
- specific to the live root repo
- explicit about file ownership and behavior

For this repo, the required levels are:

- Context: what Tokimon is and what success means
- Container: the major runtime pieces and data flow
- Component: the main logic slices inside the current implementation

Code-level diagrams are optional and should be skipped unless they add real retrieval value.

## Missing Or Conflicting Information

When information is incomplete:

- If the user intent is clear, proceed with the smallest safe assumption.
- Encode the assumption in tests, behavior, or a small doc update.
- Call out the assumption in the final response.

When information conflicts:

- Prefer executable behavior over prose.
- If neither side is trustworthy, follow the user's most recent instruction.

## Definition Of Done

A change is not done until:

- the implementation works
- the relevant test or verification step has been run, or the inability to run it is explicit
- the active docs reflect the post-change behavior

## Mantra

If it is not testable, contract-bound, or useful retrieval context for the next agent, it should not live in the docs.
