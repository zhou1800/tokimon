---
title: "Unit Testing Guidelines"
summary: "Repository rules for writing and running fast, deterministic, meaningful unit tests."
read_when:
  - Before running, reviewing, or changing unit tests.
  - Deciding unit-test scope, coverage expectations, mocks, or verification commands.
---

# Unit Testing Guidelines

## Purpose & Scope

- **UNIT-SCOPE-001:** Define how contributors write and run fast, deterministic, meaningful unit tests.
  This rule applies across repository unit-test targets, including memory, tools, workflow packages, and other repository-owned packages.
- **UNIT-SCOPE-002:** Use `.venv/bin/python -m unittest discover ...` for unit-test orchestration.
  This exercises repository import roots and discovery behavior.
  Raw runners are for focused debugging after the required orchestration path remains covered.
- **UNIT-SCOPE-003:** Goal: unit-test failures signal real regressions and surface fast, precise, actionable feedback.

## Core Principles

- **UNIT-BEHAVIOR-001:** Behavior-focused: assert observable outcomes, not internals.
- **UNIT-COVERAGE-001:** Coverage with substance: success, failure, and boundary paths; avoid trivial/no-op tests.
- **UNIT-FAST-001:** Fast by default (~<500ms/test where practical).
  Eliminate sleeps and external I/O.
- **UNIT-DETERMINISM-001:** Deterministic: fixed seeds, isolated state, no order coupling, no hidden globals.
- **UNIT-FOCUS-001:** Single logical scenario per test; multiple assertions stay scoped to that scenario.
- **UNIT-BOUNDARY-001:** Mock/stub true external boundaries.
  Representative boundaries include network, clock, and filesystem.
  Keep the unit under test real.

## Fast Feedback Workflow

Run the smallest relevant unit-test command first, then rerun the full affected suite before handoff.
When file mutation is in scope, run lint/format commands when the owning package defines them.

Representative commands:

- All tests: `.venv/bin/python -m unittest discover -s tests -t . -p 'test_*.py'`
- One workflow area: `.venv/bin/python -m unittest discover -s tests/workflow/<workflow-name> -t . -p 'test_*.py'`
- One test module: `.venv/bin/python -m unittest <dotted.test.module>`

Adjust `-s` to the affected `tests/<area>` path while preserving `-t .`.
The top-level argument keeps imports anchored at the repository root.

## Tooling & Conventions

- Python (unittest): tests live under `tests/` and mirror the package or workflow they verify (`test_*.py`).
  Use fixtures/helpers sparingly; introduce them only after genuine reuse (>2-3 repeats) and favor shared helpers.
- Tool, browser, or provider-facing units should use standard-library fakes and temporary directories.
  Run them through repository configuration invoked by `.venv/bin/python -m unittest discover ...`.

## Guardrails

- **UNIT-GUARD-001:** Unit-test work must not deploy, mutate external services, or depend on live external state.
  Representative external state includes cloud, browser, broker, provider, and messaging systems.
- **UNIT-GUARD-002:** No live network calls in unit tests.
  Mock or fake them explicitly.
- **UNIT-GUARD-003:** No new skipped tests without a referenced issue.
- **UNIT-GUARD-004:** Do not change production code solely to satisfy coverage.
  Change production code only to fix real defects or clarify brittle paths discovered while testing.
- **UNIT-GUARD-005:** Treat flakiness as a defect: investigate root causes immediately and stabilize before handoff.
- **UNIT-GUARD-006:** Refactor noisy or brittle tests promptly so suites stay clear, maintainable, and trustworthy.
