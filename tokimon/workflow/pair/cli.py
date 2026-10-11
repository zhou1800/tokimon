"""Operator support CLI for the Pair workflow."""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping

from tokimon.settings import SETTINGS_LOCAL_PATH
from tokimon.workflow.pair import contract, peer_config, state


def required_tool_capabilities() -> tuple[str, ...]:
    return contract.REQUIRED_TOOL_CAPABILITIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pair workflow deterministic state and artifact support.")
    subparsers = parser.add_subparsers(dest="command")

    contract_parser = subparsers.add_parser("contract", help="Emit workflow contract declarations")
    contract_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    init_parser = subparsers.add_parser("init", help="Initialize a Pair run artifact directory")
    init_parser.add_argument("--task-summary", required=True, help="Short human-readable task summary")
    init_parser.add_argument("--slug", help="Optional human-readable run-id slug")
    init_parser.add_argument("--max-rounds", type=int, default=contract.DEFAULT_MAX_ROUNDS)
    init_parser.add_argument("--state-root", help="Directory that contains Pair run directories")
    init_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    select_peers_parser = subparsers.add_parser(
        "select-peers",
        help="Resolve the required Peer pair from Tokimon settings",
    )
    select_peers_parser.add_argument("--run-id", required=True)
    select_peers_parser.add_argument(
        "--settings",
        default=None,
        help=f"Optional Tokimon settings path; defaults to {SETTINGS_LOCAL_PATH}",
    )
    select_peers_parser.add_argument("--task-behavior", choices=contract.TASK_BEHAVIORS, required=True)
    select_peers_parser.add_argument("--cwd", help="Current task checkout for both Peers in this run")
    select_peers_parser.add_argument("--state-root", help="Directory that contains Pair run directories")
    select_peers_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    status_parser = subparsers.add_parser("status", help="Read a Pair run state file")
    status_parser.add_argument("--run-id", required=True)
    status_parser.add_argument("--state-root", help="Directory that contains Pair run directories")
    status_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    validate_parser = subparsers.add_parser("validate", help="Validate Pair run artifact structure")
    validate_parser.add_argument("--run-id", required=True)
    validate_parser.add_argument("--state-root", help="Directory that contains Pair run directories")
    validate_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    return parser


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _contract_payload() -> dict[str, Any]:
    return {
        "product_id": contract.PRODUCT_ID,
        "skill_id": contract.SKILL_ID,
        "workflow_id": contract.WORKFLOW_ID,
        "python_package": contract.PYTHON_PACKAGE,
        "workflow_status": contract.WORKFLOW_STATUS,
        "canonical_entrypoint": contract.CANONICAL_ENTRYPOINT,
        "operator_entrypoint": contract.OPERATOR_ENTRYPOINT,
        "local_state_root": str(contract.LOCAL_STATE_ROOT),
        "run_root": str(contract.RUN_ROOT),
        "default_max_rounds": contract.DEFAULT_MAX_ROUNDS,
        "settings_path": str(SETTINGS_LOCAL_PATH),
        "settings_section": contract.SETTINGS_SECTION,
        "peer_capabilities": list(contract.PEER_CAPABILITIES),
        "codex_claude_peer_capabilities": list(contract.CODEX_CLAUDE_PEER_CAPABILITIES),
        "codex_claude_peer_agents": list(contract.CODEX_CLAUDE_PEER_AGENTS),
        "peer_a_agent": contract.PEER_A_AGENT,
        "peer_b_agents": list(contract.PEER_B_AGENTS),
        "required_tool_capabilities": list(contract.REQUIRED_TOOL_CAPABILITIES),
        "commands": list(contract.COMMANDS),
        "run_id_pattern": contract.RUN_ID_PATTERN,
        "valid_statuses": list(contract.VALID_STATUSES),
        "task_behaviors": list(contract.TASK_BEHAVIORS),
        "terminal_outcomes": list(contract.TERMINAL_OUTCOMES),
        "round_signals": list(contract.ROUND_SIGNALS),
        "pause_reasons": list(contract.PAUSE_REASONS),
        "external_clarification_fields": list(contract.EXTERNAL_CLARIFICATION_FIELDS),
        "required_initial_artifacts": list(contract.REQUIRED_INITIAL_ARTIFACTS),
        "required_directories": list(contract.REQUIRED_DIRECTORIES),
        "round_note_files": list(contract.ROUND_NOTE_FILES),
        "consensus_artifact": contract.CONSENSUS_ARTIFACT,
        "no_consensus_artifact": contract.NO_CONSENSUS_ARTIFACT,
        "required_state_fields": list(contract.REQUIRED_STATE_FIELDS),
        "no_consensus_fields": list(contract.NO_CONSENSUS_FIELDS),
        "consensus_status_fields": list(contract.REQUIRED_CONSENSUS_STATUS_FIELDS),
        "consensus_independent_note_fields": list(contract.REQUIRED_CONSENSUS_INDEPENDENT_NOTE_FIELDS),
        "consensus_exchange_fields": list(contract.REQUIRED_CONSENSUS_EXCHANGE_FIELDS),
        "consensus_peer_acceptance_fields": list(contract.REQUIRED_CONSENSUS_PEER_ACCEPTANCE_FIELDS),
        "mutable_work_fields": list(contract.REQUIRED_MUTABLE_WORK_FIELDS),
        "workflow_readiness": contract.WORKFLOW_READINESS,
    }


def _render_contract_text(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            f"workflow_id: {payload['workflow_id']}",
            f"status: {payload['workflow_status']}",
            f"entrypoint: {payload['operator_entrypoint']}",
            f"run_root: {payload['run_root']}",
        )
    )


def _render_init_text(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            f"pair run initialized: {payload['run_id']}",
            f"run_path: {payload['run_path']}",
            f"state_path: {payload['state_path']}",
            f"transcript_path: {payload['transcript_path']}",
        )
    )


def _render_status_text(payload: Mapping[str, Any]) -> str:
    state_payload = payload["state"]
    return "\n".join(
        (
            f"run_id: {payload['run_id']}",
            f"status: {state_payload['status']}",
            f"current_round_index: {state_payload['current_round_index']}",
            f"max_rounds: {state_payload.get('max_rounds', contract.DEFAULT_MAX_ROUNDS)}",
            f"run_path: {payload['run_path']}",
        )
    )


def _render_select_peers_text(payload: Mapping[str, Any]) -> str:
    selected_ids = ", ".join(payload["peer_selection_config"]["selected_peer_ids"])
    return "\n".join(
        (
            f"pair peers selected: {payload['run_id']}",
            f"peers: {selected_ids}",
            f"config: {payload['peer_selection_config']['source_path']}",
        )
    )


def _render_validate_text(payload: Mapping[str, Any]) -> str:
    if payload["ok"]:
        return f"pair run valid: {payload['run_id']}"
    lines = [f"pair run invalid: {payload['run_id']}"]
    lines.extend(f"- {error}" for error in payload["errors"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "contract"

    if command == "contract":
        payload = {"ok": True, "contract": _contract_payload()}
        if getattr(args, "json", False):
            _print_json(payload)
        else:
            print(_render_contract_text(payload["contract"]))
        return 0

    if command == "init":
        try:
            initialized = state.init_run(
                args.task_summary,
                slug=args.slug,
                max_rounds=args.max_rounds,
                state_root=args.state_root,
            )
        except (OSError, ValueError) as exc:
            payload = {"ok": False, "error": str(exc)}
            if args.json:
                _print_json(payload)
            else:
                print(str(exc))
            return 2
        payload = {"ok": True, **initialized}
        if args.json:
            _print_json(payload)
        else:
            print(_render_init_text(payload))
        return 0

    if command == "select-peers":
        try:
            peers, selection = peer_config.load_and_select(
                settings_path=args.settings,
                cwd=args.cwd,
            )
            rationale = (
                f"Selected Peers from Tokimon settings at {selection['source_path']} "
                f"section {selection['settings_section']!r}. "
                "The configured Peer pair is a hard Pair requirement."
            )
            updated_state = state.update_state(
                args.run_id,
                state_root=args.state_root,
                status="in_progress",
                peers=peers,
                peer_selection_config=selection,
                peer_selection_rationale=rationale,
                task_behavior=args.task_behavior,
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError, peer_config.PeerSelectionConfigError) as exc:
            payload = {"ok": False, "run_id": args.run_id, "error": str(exc)}
            if args.json:
                _print_json(payload)
            else:
                print(str(exc))
            return 2
        payload = {
            "ok": True,
            "run_id": args.run_id,
            "peers": peers,
            "peer_selection_config": selection,
            "state": updated_state,
        }
        if args.json:
            _print_json(payload)
        else:
            print(_render_select_peers_text(payload))
        return 0

    if command == "status":
        try:
            payload = state.status_payload(args.run_id, state_root=args.state_root)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            error_payload = {"ok": False, "run_id": args.run_id, "error": str(exc)}
            if args.json:
                _print_json(error_payload)
            else:
                print(str(exc))
            return 2
        if args.json:
            _print_json(payload)
        else:
            print(_render_status_text(payload))
        return 0

    if command == "validate":
        payload = state.validate_run(args.run_id, state_root=args.state_root)
        if args.json:
            _print_json(payload)
        else:
            print(_render_validate_text(payload))
        return 0 if payload["ok"] else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
