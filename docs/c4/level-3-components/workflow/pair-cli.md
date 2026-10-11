---
title: "Pair - CLI"
summary: "L3 command contract for Pair workflow deterministic state and artifact support."
read_when:
  - Changing Pair support commands, JSON output, run artifact structure, or structural validation.
  - Debugging Pair run initialization, status loading, or artifact validation.
---

# Pair - CLI

## Contents

- [Overview](#overview)
- [Components](#components)
- [Workflow Package Declarations](#workflow-package-declarations)
- [Interfaces/Contracts](#interfacescontracts)
- [Operator Usage](#operator-usage)
- [Output Contract](#output-contract)
- [Validation Boundary](#validation-boundary)
- [Safety Rules](#safety-rules)
- [Data Ownership](#data-ownership)
- [Failure Modes](#failure-modes)
- [Acceptance Checks](#acceptance-checks)
- [References](#references)

## Overview

The Pair CLI is a deterministic support surface for the human-invoked Pair workflow.
It declares workflow metadata, initializes run directories, records configured Peer selection, reads state, and validates required artifact structure.
The Pair workflow contract owns Peer-selection policy.

## Components

- **Contract renderer:** Owns deterministic output from the Pair workflow contract declarations.
- **Run initializer:** Owns creation of Pair run directories and initial state.
- **Peer-selection resolver:** Owns configured Peer resolution and state recording.
- **State reader:** Owns read-only `state.json` loading for status output.
- **Structural validator:** Owns required file, directory, JSON-shape, status-value, and terminal-artifact checks.
- **Renderers:** Own human-readable and JSON command output.

## Interfaces/Contracts

### Workflow Package Declarations

- **Workflow id:** `pair`
- **Python package:** `tokimon/workflow/pair/`
- **Canonical code entrypoint:** `tokimon.workflow.pair.cli:main`
- **Operator entrypoint:** `.venv/bin/python -m tokimon.workflow.pair`
- **Contract declarations:** `tokimon/workflow/pair/contract.py`
- **Workflow readiness declaration:** `tokimon/workflow/pair/contract.py` `WORKFLOW_READINESS`
- **Default local state root:** `.tokimon/workflow/pair/`
- **Default run root:** `.tokimon/workflow/pair/runs/`
- **Settings file:** `.tokimon/settings.local.json`
- **Settings section:** `pair`
- **Settings keys consumed:** `pair.access`, `pair.peer_b`, `pair.cwd`
- **Peer capabilities:** `native-subagent`, `local-cli`
- **Peer providers:** Peer A is Codex; Peer B is Codex for `native-subagent`, or Codex/Claude for `local-cli`.
- **Memory use:** explicit search when prior Pair context matters
- **Durable learning target:** `scope=workflow`, `room=pair` for Codex-written durable repeated mistakes
- **Required tool capability:** none

Canonical machine-readable workflow contract: `.venv/bin/python -m tokimon.workflow.pair contract --json`. This section summarizes the contract and the structured artifacts it names.
Commands emit human-readable text by default and JSON with `--json`.

**Commands**
- `contract`
  - Emit workflow contract declarations.
  - This is the default command when no subcommand is provided.
- `init`
  - Initialize a Pair run artifact directory.
- `select-peers`
  - Resolve the configured Peers and write them into run state.
- `status`
  - Read a Pair run state file.
- `validate`
  - Validate Pair run artifact structure.

**`contract` arguments**

**`init` arguments**
- `--task-summary <text>`
  - Required short human-readable task summary.
- `--slug <text>`
  - Optional human-readable run-id slug.
- `--max-rounds <int>`
  - Optional positive round limit, default `5`, recorded in run state.
  - The Mediator pauses at the limit without consensus; the support CLI does not execute or stop Peers.
- `--state-root <path>`
  - Optional directory that contains Pair run directories.
  - Defaults to `.tokimon/workflow/pair/runs/`.

**`select-peers` arguments**
- `--run-id <id>`
  - Required run id.
- `--settings <path>`
  - Optional Tokimon settings path.
  - Defaults to `.tokimon/settings.local.json`.
- `--task-behavior <behavior>`
  - Required task behavior, one of `read_only`, `shared_mutable_output`, `unique_irreversible_side_effect`, or `independent_outputs`.
- `--cwd <path>`
  - Current task checkout supplied to both Peers for this run; must name an existing directory.
  - Overrides `pair.cwd` without changing settings. Omission retains the configured directory for existing callers.
- `--state-root <path>`
  - Optional directory that contains Pair run directories.

**`status` arguments**
- `--run-id <id>`
  - Required run id.
- `--state-root <path>`
  - Optional directory that contains Pair run directories.

**`validate` arguments**
- `--run-id <id>`
  - Required run id.
- `--state-root <path>`
  - Optional directory that contains Pair run directories.

### Operator Usage

```bash
.venv/bin/python -m tokimon.workflow.pair contract --json
```

```bash
.venv/bin/python -m tokimon.workflow.pair init --task-summary "Pair plan review" --slug pair-plan-review --json
```

```bash
.venv/bin/python -m tokimon.workflow.pair select-peers --run-id <run_id> --task-behavior read_only --cwd <task-checkout> --json
```

```bash
.venv/bin/python -m tokimon.workflow.pair status --run-id <run_id> --json
```

```bash
.venv/bin/python -m tokimon.workflow.pair validate --run-id <run_id> --json
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
- `default_max_rounds`
- `settings_path`
- `settings_section`
- `peer_capabilities`
- `codex_claude_peer_capabilities`
- `codex_claude_peer_agents`
- `peer_a_agent`
- `peer_b_agents`
- `required_tool_capabilities`
- `commands`
- `run_id_pattern`
- `valid_statuses`
- `task_behaviors`
- `terminal_outcomes`
- `round_signals`
- `pause_reasons`
- `external_clarification_fields`
- `required_initial_artifacts`
- `required_directories`
- `round_note_files`
- `consensus_artifact`
- `no_consensus_artifact`
- `required_state_fields`
- `no_consensus_fields`
- `consensus_status_fields`
- `consensus_independent_note_fields`
- `consensus_exchange_fields`
- `consensus_peer_acceptance_fields`
- `mutable_work_fields`
- `workflow_readiness`

`init --json` emits:
- `ok`
- `run_id`
- `run_path`
- `state_path`
- `artifact_manifest_path`
- `transcript_path`
- `state`

Initial `state` includes:
- `run_id`
- `workflow_id`
- `task_summary`
- `status`
- `created_at`
- `updated_at`
- `current_round_index`
- `max_rounds`
- `peers`
- `peer_selection_config`
- `peer_selection_rationale`
- `task_behavior`
- `active_prompt_artifact`
- `transcript_artifact`
- `artifact_manifest`
- `external_clarification`
- `consensus_status`
- `final_delivery_bundle`

`max_rounds` is a positive integer; runs created before this field was introduced use the default of `5` when it is absent.
Reading an older run does not rewrite its state or imply that its evidence passes current validation.

`external_clarification` is required while paused for external clarification.
When present, it records non-empty `status`, `requested_by`, and `question`; the requester must identify one of the selected Peers or alias `a`/`b`.
The Mediator records the answer in state or the transcript and sends it to both Peers before resuming.

When `state.status` is `consensus`, `consensus_status` is the required terminal evidence object.
It includes:
- `result`
- `rounds_completed`
- `final_text_artifact`
- `final_text_sha256`
- `independent_notes`
- `exchange`
- `peer_acceptances`

`independent_notes` records round `1` plus `peer_a_artifact` and `peer_b_artifact`.
`exchange` records a later round, the artifact each Peer received from the other Peer, and both Peer response artifacts.
`peer_acceptances.a` and `peer_acceptances.b` record `accepted`, `final_text_artifact`, and `final_text_sha256`.
For `task_behavior` `shared_mutable_output`, `consensus_status.mutable_work` records `writer_peer`, `reviewer_peer`, `write_round`, `review_round`, `writer_evidence_artifact`, `reviewer_evidence_artifact`, and `reviewer_approved`.

`artifact_manifest` `artifact_contract` exposes `required_consensus_status_fields`, `required_consensus_independent_note_fields`, `required_consensus_exchange_fields`, `required_consensus_peer_acceptance_fields`, and `required_mutable_work_fields`.

When `state.status` is `no_consensus`, `no_consensus` records `unresolved_issue`, `peer_a_position`, `peer_b_position`, `peer_a_evidence`, `peer_b_evidence`, `human_decision_needed`, and `continuation_options`.

`select-peers --json` emits:
- `ok`
- `run_id`
- `peers`
- `peer_selection_config`
- `state`

The `peers` object records the selected placement capability.
`pair.access` selects the Codex Peer A placement.
With `pair.peer_b=codex`, `native-subagent` selects two Codex-native sibling subagents.
It is the recommended default for Codex and uses ids such as `native-a-codex` and `native-b-codex`.
The Mediator owns native thread creation and recovery; ephemeral native thread ids are not part of structural validity.
`local-cli` remains the explicit compatibility and rollback capability for Codex or Claude Peers.
`local-cli` peers use ids such as `local-a-codex` and `local-b-claude`.

`peer_selection_config` includes:
- `source_path`
- `settings_section`
- `selected_peer_ids`
- `selected_at`

`status --json` emits:
- `ok`
- `run_id`
- `run_path`
- `state`
- `artifacts`

`validate --json` emits:
- `ok`
- `run_id`
- `run_path`
- `errors`
- `checked`

Exit behavior:
- Exit `0` when a command succeeds or validation passes.
- Exit `2` when initialization input is invalid, state cannot be loaded, or validation fails.

### Validation Boundary

Validation is structural only.
It checks:
- required directories
- required files
- valid JSON object shape for `state.json` and `artifacts.json`
- required state fields
- configured Peer selection for active and terminal runs
- exactly Peer slots `a` and `b`, with distinct non-empty identities
- valid Codex or Claude Peer placement
- valid external-clarification fields and selected requester
- matching `workflow_id` and `run_id`
- known status values
- terminal artifact presence for `consensus` and `no_consensus`
- non-empty `consensus.md` when status is `consensus`
- `current_round_index >= 2` when status is `consensus`
- non-negative integer current round and positive configured round limit
- `consensus_status.result == "consensus"`, `rounds_completed >= 2`, `final_text_artifact == "consensus.md"`, and `final_text_sha256` matching the SHA-256 of `consensus.md`
- `rounds_completed` equal to the current round, with exchange and write/review rounds no later than the completed round
- round-1 independent note artifacts for Peer A and Peer B
- a later exchange round where each Peer received the other Peer independent note and both response artifacts exist
- `peer_acceptances.a` and `peer_acceptances.b` accepted flags, consensus artifact references, and final-text hash agreement
- `mutable_work` writer/reviewer evidence, distinct Peer slots, ordered write/review rounds, and reviewer approval for `shared_mutable_output`
- required no-consensus fields when status is `no_consensus`
- artifact manifest `artifact_contract` fields that describe required initial artifacts, directories, round note files, terminal artifacts, state fields, no-consensus fields, consensus-status fields, and mutable-work fields

For `in_progress`, `paused_external_clarification`, `consensus`, or `no_consensus` state, validation requires `peer_selection_config` to match `peers`.

Validation does not:
- decide whether Peer reasoning is correct
- decide whether Peer disagreement is substantive
- infer consensus from transcript prose
- prove real independence beyond recorded artifacts
- run external Agents
- inspect provider conversations or remote runner availability
- scrape provider or browser state
- validate provider model, effort, or credential availability for selected Peer capabilities
- prove that the current Agent runtime exposes native subagent controls or sufficient concurrent slots
- mutate Peer notes
- decide whether objections are substantive

### Safety Rules

- The CLI writes only when `init` or `select-peers` is called.
- `init` writes only inside the configured run root.
- `select-peers` writes only to the requested Pair run `state.json`.
- `status` and `validate` are read-only.
- The CLI does not read secrets, cookies, API keys, browser sessions, MCP configs, or provider credentials.
- The CLI does not call external network services.

## Data Ownership

- Owns Pair run artifact state under `.tokimon/workflow/pair/runs/` or the caller-supplied state root.
- Owns `state.json`, `artifacts.json`, settings-derived Peer selection metadata, terminal artifact path validation, and structural validation output created by deterministic Pair support commands.
- Consumes read-only `.tokimon/settings.local.json` section `pair`, task summary, and existing run artifacts as inputs.
- Derives run ids, configured Peer records, status output, artifact inventory validation, and terminal-state validation results.
- Does not own provider conversations, Peer notes, consensus judgment, external Agent execution, selected-capability settings, or semantic review content.

## Failure Modes

Pair recovery preserves the run directory, transcript, Peer notes, and terminal artifact first, then fixes run state or Peer-selection inputs before rerunning structural validation.

- Tokimon settings or the `pair` section is missing, malformed, or inconsistent with active run state; fix the settings or recorded selection before prompting Peers.
- Native controls or required Peer slots are unavailable; select `local-cli` explicitly or resume from a capable Codex task.
- A native Peer thread is interrupted or unavailable after restart; recreate that Peer from the persisted prompt and round artifacts without treating the old thread id as durable state.
- The target run id already exists during initialization; choose a new slug/run id or resume the existing run.
- The requested run directory is missing; pass the correct `--state-root`/`--run-id` pair or initialize the run first.
- Required directories or files are missing; regenerate or repair the run artifact structure before continuing.
- `state.json` or `artifacts.json` is invalid JSON; fix the artifact syntax and rerun validation.
- State fields are missing or inconsistent; reconcile state with the Pair protocol before delivering final output.
- Terminal status lacks its required terminal artifact; write the matching terminal artifact before marking the run complete.
- Consensus state has only one round, missing `consensus_status`, missing referenced artifacts, mismatched SHA-256 values, incomplete Peer acceptances, or invalid mutable-work evidence; repair the recorded evidence before delivering consensus.

## Acceptance Checks

```bash
.venv/bin/python -m unittest discover -s tests/workflow/pair -t . -p 'test_*.py'
```

```bash
.venv/bin/python -m tokimon.workflow.pair contract --json
```

```bash
.venv/bin/python -m tokimon.workflow.pair init --task-summary "Pair smoke test" --slug smoke-test --state-root .tokimon/tmp/pair-smoke --json
```

```bash
.venv/bin/python -m tokimon.workflow.pair select-peers --run-id <run_id> --state-root .tokimon/tmp/pair-smoke --settings .tokimon/settings.local.json --task-behavior read_only --json
```

```bash
.venv/bin/python -m tokimon.workflow.pair validate --run-id <run_id> --state-root .tokimon/tmp/pair-smoke --json
```

Expected outcome: tests pass and support commands emit the documented JSON shape without running any Agent provider.

Ask an AI coding agent to verify the component contract:

```text
Review docs/c4/level-3-components/workflow/pair-cli.md, tokimon/workflow/pair/, tests/workflow/pair/, and skills/pair/skill.md.
Confirm that the documented contract/init/select-peers/status/validate commands, artifact structure, data boundary, failure modes, and tests match the implementation.
Report any drift with file and line evidence.
```

Expected outcome: the agent reports no contract drift, or lists precise mismatches with recommended owner documents or implementation files.

## References

- L2 Container: `docs/c4/level-2-containers/workflow/pair.md`
- L1 Context: `docs/c4/level-1-context/workflow-context.md`
