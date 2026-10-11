"""Operator support CLI for the Learning workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tokimon.workflow.learning import artifacts, contract, validation


def required_tool_capabilities() -> tuple[str, ...]:
    return contract.REQUIRED_TOOL_CAPABILITIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learning workflow deterministic support.")
    subparsers = parser.add_subparsers(dest="command")

    contract_parser = subparsers.add_parser("contract", help="Emit workflow contract declarations")
    contract_parser.add_argument("--json", action="store_true")

    start_reactive = subparsers.add_parser("start-reactive", help="Initialize a Reactive Learning run")
    start_reactive.add_argument("--trigger", required=True)
    start_reactive.add_argument("--slug")
    start_reactive.add_argument("--run-root")
    start_reactive.add_argument("--json", action="store_true")

    start_active = subparsers.add_parser("start-active", help="Initialize an Active Learning run")
    start_active.add_argument("--target-capability", required=True)
    start_active.add_argument("--source", required=True)
    start_active.add_argument("--slug")
    start_active.add_argument("--run-root")
    start_active.add_argument("--json", action="store_true")

    start_acceptance = subparsers.add_parser("start-acceptance", help="Initialize a Learning acceptance run")
    start_acceptance.add_argument("--target-capability", required=True)
    start_acceptance.add_argument("--slug")
    start_acceptance.add_argument("--run-root")
    start_acceptance.add_argument("--json", action="store_true")

    create_reactive = subparsers.add_parser("create-reactive-candidate")
    create_reactive.add_argument("--input", required=True)
    create_reactive.add_argument("--run-id")
    create_reactive.add_argument("--run-root")
    create_reactive.add_argument("--json", action="store_true")

    create_active = subparsers.add_parser("create-active-candidate")
    create_active.add_argument("--input", required=True)
    create_active.add_argument("--run-id")
    create_active.add_argument("--run-root")
    create_active.add_argument("--json", action="store_true")

    validate_reactive = subparsers.add_parser("validate-reactive-candidate")
    validate_reactive.add_argument("packet")
    validate_reactive.add_argument("--json", action="store_true")

    validate_active = subparsers.add_parser("validate-active-candidate")
    validate_active.add_argument("packet")
    validate_active.add_argument("--json", action="store_true")

    review = subparsers.add_parser("record-review")
    review.add_argument("packet")
    review.add_argument("--reviewer-id", required=True)
    review.add_argument("--decision", choices=contract.VALID_REVIEW_DECISIONS, required=True)
    review.add_argument("--json", action="store_true")

    approve = subparsers.add_parser("approve-intake")
    approve.add_argument("packet")
    approve.add_argument("--operator-id", required=True)
    approve.add_argument("--approval-note", required=True)
    approve.add_argument("--json", action="store_true")

    package = subparsers.add_parser("create-intake-package")
    package.add_argument("--packet", required=True)
    package.add_argument("--proposed-surface", choices=contract.VALID_TARGET_SURFACES)
    package.add_argument("--intake-id")
    package.add_argument("--run-id")
    package.add_argument("--run-root")
    package.add_argument("--json", action="store_true")

    activation = subparsers.add_parser("record-activation")
    activation.add_argument("--intake-package", required=True)
    activation.add_argument("--activated-path", required=True)
    activation.add_argument("--evidence", required=True)
    activation.add_argument("--run-id", required=True)
    activation.add_argument("--run-root")
    activation.add_argument("--json", action="store_true")

    acceptance = subparsers.add_parser("validate-acceptance")
    acceptance.add_argument("run_id")
    acceptance.add_argument("--run-root")
    acceptance.add_argument("--json", action="store_true")

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
        "intake_package_root": str(contract.INTAKE_PACKAGE_ROOT),
        "required_tool_capabilities": list(contract.REQUIRED_TOOL_CAPABILITIES),
        "commands": list(contract.COMMANDS),
        "run_id_pattern": contract.RUN_ID_PATTERN,
        "valid_run_modes": list(contract.VALID_RUN_MODES),
        "valid_candidate_statuses": list(contract.VALID_CANDIDATE_STATUSES),
        "valid_review_decisions": list(contract.VALID_REVIEW_DECISIONS),
        "valid_operator_intake_statuses": list(contract.VALID_OPERATOR_INTAKE_STATUSES),
        "valid_intake_package_statuses": list(contract.VALID_INTAKE_PACKAGE_STATUSES),
        "valid_target_surfaces": list(contract.VALID_TARGET_SURFACES),
        "required_candidate_fields": list(contract.REQUIRED_CANDIDATE_FIELDS),
        "required_reactive_evidence_fields": list(contract.REQUIRED_REACTIVE_EVIDENCE_FIELDS),
        "required_active_evidence_fields": list(contract.REQUIRED_ACTIVE_EVIDENCE_FIELDS),
        "required_activation_evidence_fields": list(contract.REQUIRED_ACTIVATION_EVIDENCE_FIELDS),
        "required_product_development_evidence_fields": list(contract.REQUIRED_PRODUCT_DEVELOPMENT_EVIDENCE_FIELDS),
        "required_accepted_artifact_fields": list(contract.REQUIRED_ACCEPTED_ARTIFACT_FIELDS),
        "required_verification_evidence_fields": list(contract.REQUIRED_VERIFICATION_EVIDENCE_FIELDS),
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


def _write_packet(path: str | Path, packet: Mapping[str, Any]) -> None:
    artifacts.write_json(Path(path), packet)


def _emit(payload: Mapping[str, Any], *, json_output: bool, text: str | None = None) -> None:
    if json_output:
        _print_json(payload)
    else:
        print(text or json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _read_packet(path: str | Path) -> dict[str, Any]:
    return artifacts.read_json(path)


def _start_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "start-reactive":
        return artifacts.start_run(
            mode="reactive",
            trigger=args.trigger,
            slug=args.slug or args.trigger,
            run_root=args.run_root,
        )
    if args.command == "start-active":
        return artifacts.start_run(
            mode="active",
            target_capability=args.target_capability,
            source=args.source,
            slug=args.slug or args.target_capability,
            run_root=args.run_root,
        )
    return artifacts.start_run(
        mode="acceptance",
        target_capability=args.target_capability,
        slug=args.slug or args.target_capability,
        run_root=args.run_root,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "contract"

    try:
        if command == "contract":
            payload = {"ok": True, "contract": _contract_payload()}
            _emit(payload, json_output=getattr(args, "json", False), text=_render_contract_text(payload["contract"]))
            return 0

        if command in {"start-reactive", "start-active", "start-acceptance"}:
            payload = {"ok": True, **_start_payload(args)}
            _emit(payload, json_output=args.json, text=f"learning run initialized: {payload['run_id']}")
            return 0

        if command == "create-reactive-candidate":
            payload = artifacts.write_candidate(_read_packet(args.input), mode="reactive", run_id=args.run_id, run_root=args.run_root)
            _emit(payload, json_output=args.json, text=f"reactive candidate {payload['status']}: {payload.get('packet_id')}")
            return 0 if payload["ok"] else 2

        if command == "create-active-candidate":
            payload = artifacts.write_candidate(_read_packet(args.input), mode="active", run_id=args.run_id, run_root=args.run_root)
            _emit(payload, json_output=args.json, text=f"active candidate {payload['status']}: {payload.get('packet_id')}")
            return 0 if payload["ok"] else 2

        if command == "validate-reactive-candidate":
            payload = validation.validate_candidate_packet(_read_packet(args.packet), expected_mode="reactive")
            _emit(payload, json_output=args.json, text=f"reactive candidate {payload['status']}: {payload.get('packet_id')}")
            return 0 if payload["ok"] else 2

        if command == "validate-active-candidate":
            payload = validation.validate_candidate_packet(_read_packet(args.packet), expected_mode="active")
            _emit(payload, json_output=args.json, text=f"active candidate {payload['status']}: {payload.get('packet_id')}")
            return 0 if payload["ok"] else 2

        if command == "record-review":
            packet = artifacts.record_review(_read_packet(args.packet), reviewer_id=args.reviewer_id, decision=args.decision)
            _write_packet(args.packet, packet)
            payload = {
                "ok": True,
                "packet_id": packet.get("packet_id"),
                "candidate_review_status": packet.get("candidate_review_status"),
                "operator_intake_status": packet.get("operator_intake_status"),
                "intake_package_status": packet.get("intake_package_status"),
                "artifact_path": args.packet,
            }
            _emit(payload, json_output=args.json)
            return 0

        if command == "approve-intake":
            packet = artifacts.approve_intake(
                _read_packet(args.packet),
                operator_id=args.operator_id,
                approval_note=args.approval_note,
            )
            _write_packet(args.packet, packet)
            payload = {
                "ok": True,
                "packet_id": packet.get("packet_id"),
                "candidate_review_status": packet.get("candidate_review_status"),
                "operator_intake_status": packet.get("operator_intake_status"),
                "intake_package_status": packet.get("intake_package_status"),
                "artifact_path": args.packet,
            }
            _emit(payload, json_output=args.json)
            return 0

        if command == "create-intake-package":
            packet = _read_packet(args.packet)
            payload = artifacts.create_intake_package(
                packet,
                proposed_surface=args.proposed_surface,
                intake_id=args.intake_id,
                run_id=args.run_id,
                run_root=args.run_root,
            )
            if payload["ok"]:
                _write_packet(args.packet, packet)
            _emit(payload, json_output=args.json)
            return 0 if payload["ok"] else 2

        if command == "record-activation":
            payload = artifacts.record_activation(
                _read_packet(args.intake_package),
                activated_path=args.activated_path,
                evidence=_read_packet(args.evidence),
                run_id=args.run_id,
                run_root=args.run_root,
            )
            _emit(payload, json_output=args.json)
            return 0 if payload["ok"] else 2

        if command == "validate-acceptance":
            payload = artifacts.validate_acceptance(args.run_id, run_root=args.run_root)
            _emit(payload, json_output=args.json, text=f"learning acceptance {payload['status']}: {args.run_id}")
            return 0 if payload["ok"] else 2

    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {"ok": False, "error": str(exc)}
        _emit(payload, json_output=getattr(args, "json", False), text=str(exc))
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
