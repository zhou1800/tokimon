---
title: "Learning - CLI"
summary: "L3 command contract for Learning workflow candidate validation, intake package creation, activation, and acceptance support."
read_when:
  - Changing Learning support commands, JSON output, artifact structure, or validation behavior.
  - Debugging Learning candidate lifecycle, intake packages, activation records, or acceptance validation.
---

# Learning - CLI

## Contents

- [Overview](#overview)
- [Components](#components)
- [Workflow Package Declarations](#workflow-package-declarations)
- [Interfaces/Contracts](#interfacescontracts)
- [Agent CLI Usage](#agent-cli-usage)
- [Output Contract](#output-contract)
- [Validation Boundary](#validation-boundary)
- [Data Ownership](#data-ownership)
- [Failure Modes](#failure-modes)
- [Acceptance Checks](#acceptance-checks)
- [References](#references)

## Overview

The Learning CLI is the deterministic support surface for an agent-driven Learning workflow.
It owns artifact structure, validation, lifecycle records, intake package creation, activation records, and acceptance status.
It does not decide what to learn, inspect sources, publish repository guidance, or activate product behavior by itself.

## Components

- **Contract renderer:** Emits workflow declarations and readiness metadata.
- **Run initializer:** Creates Reactive, Active, and acceptance run directories.
- **Candidate validator:** Checks candidate packets against mode-specific evidence, scope, review, and approval requirements.
- **Lifecycle recorder:** Records reviewer decisions and operator intake approval.
- **Intake packager:** Creates local product-development intake packages that normal product-development may activate as Skill, MCP, workflow, or existing-surface updates.
- **Activation recorder:** Records product-development proof after the learned capability is active.
- **Acceptance validator:** Evaluates held-out acceptance status from Reactive, Active, and activation evidence.

## Interfaces/Contracts

### Workflow Package Declarations

- **Workflow id:** `learning`
- **Product id:** `learning`
- **Skill id:** `learning`
- **Python package:** `tokimon/workflow/learning/`
- **Canonical code entrypoint:** `tokimon.workflow.learning.cli:main`
- **Operator entrypoint:** `.venv/bin/python -m tokimon.workflow.learning`
- **Contract declarations:** `tokimon/workflow/learning/contract.py`
- **Workflow readiness declaration:** `tokimon/workflow/learning/contract.py` `WORKFLOW_READINESS`
- **Default local state root:** `.tokimon/workflow/learning/`
- **Default run root:** `.tokimon/workflow/learning/runs/`
- **Default intake package root:** `.tokimon/workflow/learning/intake-packages/`
- **Memory use:** explicit search when prior learning evidence matters
- **Required tool capability:** none

Canonical machine-readable workflow contract: `.venv/bin/python -m tokimon.workflow.learning contract --json`. This section summarizes the contract and the structured artifacts it names.
Commands emit human-readable text by default and JSON with `--json`.

**Commands**
- `contract`
- `start-reactive`
- `start-active`
- `start-acceptance`
- `create-reactive-candidate`
- `create-active-candidate`
- `validate-reactive-candidate`
- `validate-active-candidate`
- `record-review`
- `approve-intake`
- `create-intake-package`
- `record-activation`
- `validate-acceptance`

**Common arguments**
- `--run-root <path>`
  - Optional directory that contains Learning run directories.
  - Defaults to `.tokimon/workflow/learning/runs/`.

**Run initialization arguments**
- `start-reactive --trigger <trigger> [--slug <slug>]`
- `start-active --target-capability <name> --source <path-or-ref> [--slug <slug>]`
- `start-acceptance --target-capability <name> [--slug <slug>]`

**Candidate arguments**
- `create-reactive-candidate --input <packet.json> [--run-id <run_id>]`
- `create-active-candidate --input <packet.json> [--run-id <run_id>]`
- `validate-reactive-candidate <packet.json>`
- `validate-active-candidate <packet.json>`

**Lifecycle arguments**
- `record-review <packet.json> --reviewer-id <id> --decision accept|reject`
- `approve-intake <packet.json> --operator-id <id> --approval-note <text>`

**Promotion and activation arguments**
- `create-intake-package --packet <packet.json> [--proposed-surface skill|mcp|skill_and_mcp|workflow|existing_surface_update] [--intake-id <id>] [--run-id <run_id>]`
- `record-activation --intake-package <package.json> --activated-path <path> --evidence <evidence.json> --run-id <run_id>`
- `validate-acceptance <run_id>`

### Agent CLI Usage

```bash
.venv/bin/python -m tokimon.workflow.learning contract --json
```

```bash
.venv/bin/python -m tokimon.workflow.learning start-active --target-capability dreaming --source /path/to/source --slug dreaming --json
```

```bash
.venv/bin/python -m tokimon.workflow.learning validate-active-candidate /path/to/candidate.json --json
```

```bash
.venv/bin/python -m tokimon.workflow.learning create-intake-package --packet /path/to/candidate.json --proposed-surface skill_and_mcp --run-id <run_id> --json
```

```bash
.venv/bin/python -m tokimon.workflow.learning record-activation --intake-package /path/to/package.json --activated-path skills/dreaming/skill.md --evidence /path/to/evidence.json --run-id <run_id> --json
```

```bash
.venv/bin/python -m tokimon.workflow.learning validate-acceptance <run_id> --json
```

### Output Contract

`contract --json` emits:
- `ok`
- `contract`

The `contract` object includes:
- `product_id`
- `skill_id`
- `workflow_id`
- `python_package`
- `workflow_status`
- `canonical_entrypoint`
- `operator_entrypoint`
- `local_state_root`
- `run_root`
- `intake_package_root`
- `required_tool_capabilities`
- `commands`
- `run_id_pattern`
- `valid_run_modes`
- `valid_candidate_statuses`
- `valid_review_decisions`
- `valid_operator_intake_statuses`
- `valid_intake_package_statuses`
- `valid_target_surfaces`
- `required_candidate_fields`
- `required_reactive_evidence_fields`
- `required_active_evidence_fields`
- `required_activation_evidence_fields`
- `required_product_development_evidence_fields`
- `required_accepted_artifact_fields`
- `required_verification_evidence_fields`
- `workflow_readiness`

Run initialization commands emit:
- `ok`
- `run_id`
- `run_path`
- `state_path`
- `state`

Candidate creation and validation commands emit:
- `ok`
- `run_id`, when the command writes a packet to a run
- `packet_id`
- `mode`
- `status`
- `errors`
- `missing_fields`
- `packet_path`, when the command writes a packet
- `validation_report_path`, when the command writes a report

Lifecycle commands emit:
- `ok`
- `packet_id`
- `candidate_review_status`
- `operator_intake_status`
- `intake_package_status`
- `artifact_path`

Intake package and activation commands emit:
- `ok`
- `run_id`, for activation commands
- `intake_id`
- `artifact_path` or `record_path`
- `package`, for intake package creation
- `activated`, for activation recording
- `record`, for activation recording

Acceptance validation emits:
- `ok`
- `run_id`
- `status`
- `errors`
- `reactive_ready`
- `active_ready`
- `activation_ready`

Exit behavior:
- Exit `0` when a command succeeds or validation passes.
- Exit `2` when inputs are invalid, required evidence is missing, lifecycle state blocks the requested transition, or acceptance is incomplete.

### Validation Boundary

Validation is structural and contract-based.
It checks:
- accepted trigger categories for Reactive Learning
- explicit operator request, target capability, selected sources, and exact source spans for Active Learning
- pressure scenario, candidate guidance, scope, negative scope, and claim boundary
- acceptance expectations for product-development intake
- reviewer decision status and operator intake approval before intake package creation
- optional non-binding surface hypotheses for Skill, MCP, Skill+MCP, workflow, and existing-surface update packages
- run-local linkage between activation records, intake packages, and accepted candidate evidence
- activation evidence path, activated surface type, and normal product-development accepted status, summary, artifacts, and verification evidence
- held-out acceptance coverage for Reactive, Active, and activation evidence

Validation does not:
- judge whether source code copied from a reference repository is legally or technically safe
- decide product scope for a new Skill or MCP
- activate a Skill or MCP surface
- turn candidate packets into canonical repository guidance

## Data Ownership

Learning owns `.tokimon/workflow/learning/runs/<run_id>/` for runtime evidence.
Learning owns `.tokimon/workflow/learning/intake-packages/` for product-development intake packages.
Learning consumes review reports, source files, test logs, and product-development artifacts read-only unless the operator explicitly asks to edit their owning surface.
The durable output of Learning is the activated Tokimon capability and its owning C4/code/tests, not Learning notes.

## Failure Modes

- **Invalid candidate packet:** validation writes an error report and leaves the packet unusable for intake-package creation. Recover by revising the source evidence, pressure scenario, scope, negative scope, or claim boundary, then rerun the relevant validation command.
- **Reviewer rejection:** the candidate remains local evidence but intake creation stays blocked. Recover by either preserving the rejection as final evidence, revising and resubmitting a new candidate, or deleting stale runtime artifacts only after the rejection is recorded elsewhere if needed.
- **Missing operator intake approval:** `create-intake-package` remains blocked. Recover by obtaining explicit operator approval and recording it with `approve-intake`, or close the run without an intake package.
- **Blocked or stale intake package:** product-development may reject or ignore the package. Recover by creating a new package from accepted run-local evidence, or preserve and retire the stale package instead of editing it in place.
- **Activation recording mismatch:** `record-activation` fails when the intake package is outside the target run or not linked to accepted evidence. Recover by selecting the run-local package or regenerating the intake package from the accepted candidate before retrying.
- **Missing or summary-only activation evidence:** `validate-acceptance` reports incomplete acceptance. Recover by recording product-development evidence that names the activated surface and includes accepted status, summary, artifacts, and verification evidence, or keep the Learning run open.

## Acceptance Checks

```bash
.venv/bin/python -m tokimon.workflow.learning contract --json
```

Expected outcome: JSON output contains `ok: true`, `workflow_id: learning`, and `workflow_readiness`.

```bash
.venv/bin/python -m unittest discover -s tests/workflow/learning -t . -p 'test_*.py'
```

```bash
.venv/bin/python -m unittest discover -s tests/workflow -t . -p 'test_package_contract.py'
```

Expected outcome: deterministic Learning tests and shared workflow package contract tests pass.

Verification prompt:

```text
Review the Learning workflow against docs/c4/level-2-containers/workflow/learning.md and docs/c4/level-3-components/workflow/learning-cli.md. Confirm whether candidate validation, intake packages, and activation records prove both Reactive Learning and Active Learning can produce an activated Skill, MCP, workflow, or existing-surface update through normal product-development gates.
```

Expected outcome: the reviewer reports no blocking mismatch between the docs, implementation, tests, and acceptance evidence.

## References

- L2 Container: `docs/c4/level-2-containers/workflow/learning.md`
- Guidelines: `docs/guidelines/document-guidelines.md`
- Guidelines: `docs/guidelines/c4-document-guidelines.md`
