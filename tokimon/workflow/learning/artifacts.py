"""File and lifecycle helpers for Learning workflow artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Mapping

from tokimon.runtime import resolve_runtime_path, source_root
from tokimon.workflow.learning import contract, models, validation


WORKSPACE_ROOT = source_root()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return models.require_mapping(json.load(handle), label=str(path))


def resolve_run_root(run_root: str | Path | None = None) -> Path:
    path = Path(run_root) if run_root is not None else contract.RUN_ROOT
    return resolve_runtime_path(path, workspace_root=WORKSPACE_ROOT)


def run_path(run_id: str, run_root: str | Path | None = None) -> Path:
    if (
        not isinstance(run_id, str) or not run_id.strip()
        or run_id in {".", ".."} or "/" in run_id or "\\" in run_id
        or Path(run_id).is_absolute()
    ):
        raise ValueError("run_id must be a non-empty path component")
    root = resolve_run_root(run_root)
    target = root / run_id
    try:
        contained = target.resolve().is_relative_to(root.resolve())
    except RuntimeError as exc:
        raise ValueError("run_id cannot resolve its configured path") from exc
    if not contained:
        raise ValueError("run_id escapes its configured run root")
    return target


def _owned_path(root: Path, relative: str | Path) -> Path:
    """Keep generated artifact paths and scan inputs inside their owning root."""
    path = root / relative
    try:
        contained = path.resolve().is_relative_to(root.resolve())
    except RuntimeError as exc:
        raise ValueError("Learning artifact path cannot be resolved") from exc
    if not contained:
        raise ValueError("Learning artifact path must stay inside its owning directory")
    return path


def _run_json_paths(target: Path, directory: str) -> Iterator[Path]:
    folder = _owned_path(target, directory)
    for path in folder.glob("*.json"):
        yield _owned_path(target, path.relative_to(target))


def state_path(run_id: str, run_root: str | Path | None = None) -> Path:
    return _owned_path(run_path(run_id, run_root), "state.json")


def generate_run_id(slug: str | None = None, *, now=None) -> str:
    slug_part = models.slugify(slug, fallback="learning-run")
    return f"{contract.WORKFLOW_ID}-{models.run_id_timestamp(now)}-{slug_part}"


def _artifact(kind: str, path: str) -> dict[str, str]:
    return {"kind": kind, "path": path}


def artifact_manifest(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "workflow_id": contract.WORKFLOW_ID,
        "artifact_contract": {
            "required_initial_artifacts": list(contract.REQUIRED_INITIAL_ARTIFACTS),
            "required_directories": list(contract.REQUIRED_DIRECTORIES),
            "required_state_fields": list(contract.REQUIRED_STATE_FIELDS),
        },
        "artifacts": [
            _artifact("state", "state.json"),
            _artifact("acceptance_plan", "acceptance-plan.json"),
            _artifact("artifact_manifest", "artifacts.json"),
            _artifact("candidate_directory", "candidates/"),
            _artifact("validation_report_directory", "validation-reports/"),
            _artifact("intake_package_directory", "intake-packages/"),
            _artifact("activation_record_directory", "activation-records/"),
        ],
    }


def initial_state(
    run_id: str,
    *,
    mode: str,
    trigger: str | None = None,
    target_capability: str | None = None,
    source: str | None = None,
    now=None,
) -> dict[str, Any]:
    stamp = models.timestamp_text(now)
    return {
        "run_id": run_id,
        "workflow_id": contract.WORKFLOW_ID,
        "mode": mode,
        "status": "initialized",
        "trigger": trigger,
        "target_capability": target_capability,
        "source": source,
        "created_at": stamp,
        "updated_at": stamp,
        "artifact_manifest": "artifacts.json",
        "acceptance_plan": "acceptance-plan.json",
    }


def acceptance_plan(run_id: str, *, target_capability: str | None = None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "target_capability": target_capability,
        "reactive_ready": False,
        "active_ready": False,
        "activation_ready": False,
        "held_out_target": True,
        "status": "incomplete",
        "errors": [
            "reactive learning evidence is missing",
            "active learning evidence is missing",
            "activation evidence is missing",
        ],
    }


def start_run(
    *,
    mode: str,
    slug: str | None = None,
    trigger: str | None = None,
    target_capability: str | None = None,
    source: str | None = None,
    run_root: str | Path | None = None,
    now=None,
) -> dict[str, Any]:
    if mode not in contract.VALID_RUN_MODES:
        raise ValueError("mode is not valid")
    run_id = generate_run_id(slug or target_capability or trigger or mode, now=now)
    target = run_path(run_id, run_root)
    target.mkdir(parents=True, exist_ok=False)
    for directory in contract.REQUIRED_DIRECTORIES:
        _owned_path(target, directory).mkdir(parents=True, exist_ok=True)
    state = initial_state(
        run_id,
        mode=mode,
        trigger=trigger,
        target_capability=target_capability,
        source=source,
        now=now,
    )
    plan = acceptance_plan(run_id, target_capability=target_capability)
    write_json(_owned_path(target, "state.json"), state)
    write_json(_owned_path(target, "acceptance-plan.json"), plan)
    write_json(_owned_path(target, "artifacts.json"), artifact_manifest(run_id))
    return {
        "run_id": run_id,
        "run_path": str(target),
        "state_path": str(target / "state.json"),
        "state": state,
    }


def _packet_file_name(packet: Mapping[str, Any]) -> str:
    packet_id = models.slugify(str(packet.get("packet_id") or "candidate"), fallback="candidate")
    return f"{packet_id}.json"


def write_candidate(
    packet: Mapping[str, Any],
    *,
    mode: str,
    run_id: str | None = None,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    result = validation.validate_candidate_packet(packet, expected_mode=mode)
    if not result["ok"]:
        return {**result, "packet_path": None, "validation_report_path": None}
    if run_id is None:
        run_payload = start_run(mode=mode, slug=str(packet.get("packet_id") or mode), run_root=run_root)
        run_id = run_payload["run_id"]
    elif not state_path(run_id, run_root).is_file():
        raise ValueError("run_id must name an initialized Learning run")
    target = run_path(run_id, run_root)
    candidate_path = _owned_path(target, Path("candidates") / _packet_file_name(packet))
    report_path = _owned_path(target, Path("validation-reports") / f"{candidate_path.stem}-validation.json")
    write_json(candidate_path, packet)
    write_json(report_path, result)
    return {
        **result,
        "run_id": run_id,
        "packet_path": str(candidate_path),
        "validation_report_path": str(report_path),
    }


def record_review(packet: dict[str, Any], *, reviewer_id: str, decision: str, now=None) -> dict[str, Any]:
    if decision not in contract.VALID_REVIEW_DECISIONS:
        raise ValueError("decision must be accept or reject")
    record = {
        "reviewer_id": reviewer_id,
        "decision": decision,
        "reviewed_at": models.timestamp_text(now),
    }
    records = packet.setdefault("reviewer_records", [])
    if not isinstance(records, list):
        raise ValueError("reviewer_records must be a list")
    records.append(record)
    packet["candidate_review_status"] = "accepted" if decision == "accept" else "rejected"
    return packet


def approve_intake(packet: dict[str, Any], *, operator_id: str, approval_note: str, now=None) -> dict[str, Any]:
    if not approval_note.strip():
        raise ValueError("approval_note is required")
    if packet.get("candidate_review_status") != "accepted":
        raise ValueError("candidate_review_status must be accepted before intake approval")
    packet["operator_intake_status"] = "approved"
    packet["operator_approval"] = {
        "operator_id": operator_id,
        "approval_note": approval_note,
        "approved": True,
        "approved_at": models.timestamp_text(now),
    }
    return packet


def product_intake_from_packet(packet: Mapping[str, Any], *, now=None) -> dict[str, Any]:
    learning_mode = str(packet.get("learning_mode") or "")
    evidence = packet.get("reactive_evidence") if learning_mode == "reactive" else packet.get("active_evidence")
    active_evidence = packet.get("active_evidence") if isinstance(packet.get("active_evidence"), Mapping) else {}
    return {
        "packet_id": packet.get("packet_id"),
        "learning_mode": learning_mode,
        "target_capability": active_evidence.get("target_capability"),
        "title": str(packet.get("candidate_guidance") or packet.get("packet_id") or "Learning product-development intake"),
        "pressure_scenario": packet.get("pressure_scenario"),
        "product_intent": packet.get("candidate_guidance"),
        "evidence_refs": {
            "source_evidence": evidence,
            "reviewer_records": packet.get("reviewer_records", []),
            "operator_approval": packet.get("operator_approval", {}),
        },
        "constraints": {
            "scope": packet.get("scope"),
            "negative_scope": packet.get("negative_scope"),
            "claim_boundary": packet.get("claim_boundary"),
        },
        "acceptance_expectations": packet.get("acceptance_expectations"),
        "created_at": models.timestamp_text(now),
    }


def create_intake_package(
    packet: dict[str, Any],
    *,
    proposed_surface: str | None = None,
    intake_id: str | None = None,
    package_root: str | Path | None = None,
    run_id: str | None = None,
    run_root: str | Path | None = None,
    now=None,
) -> dict[str, Any]:
    if proposed_surface is not None and proposed_surface not in contract.VALID_TARGET_SURFACES:
        raise ValueError("proposed_surface is not valid")
    packet_mode = str(packet.get("learning_mode") or "")
    expected_mode = packet_mode if packet_mode in {"reactive", "active"} else None
    validation_result = validation.validate_candidate_packet(
        packet,
        expected_mode=expected_mode,
        require_intake_gate=True,
    )
    if not validation_result["ok"]:
        return {**validation_result, "artifact_path": None, "package": None}
    packet_id = str(packet.get("packet_id") or "")
    if not packet_id:
        raise ValueError("packet_id is required")
    product_intake = product_intake_from_packet(packet, now=now)
    package_id = models.slugify(intake_id or f"{packet_id}-product-development-intake", fallback="intake")
    if package_root is not None:
        root = Path(package_root)
    elif run_id is not None:
        if not state_path(run_id, run_root).is_file():
            raise ValueError("run_id must name an initialized Learning run")
        root = _owned_path(run_path(run_id, run_root), "intake-packages")
    else:
        root = resolve_runtime_path(contract.INTAKE_PACKAGE_ROOT, workspace_root=WORKSPACE_ROOT)
    package = {
        "schema_id": "learning.product_development_intake.v1",
        "intake_id": package_id,
        "packet_id": packet_id,
        "source_candidate_id": packet_id,
        "target_capability": product_intake.get("target_capability") or packet_id,
        "normal_product_development_required": True,
        "activation_not_permitted_by_learning_workflow": True,
        "product_intent": product_intake["product_intent"],
        "pressure_scenario": product_intake["pressure_scenario"],
        "evidence_refs": product_intake["evidence_refs"],
        "constraints": product_intake["constraints"],
        "acceptance_expectations": product_intake["acceptance_expectations"],
        "activation_candidate": {
            "proposed_surface": proposed_surface,
            "surface_is_non_binding": True,
            "activation_goal": product_intake["product_intent"],
        },
        "handoff": {
            "status": "ready_for_product_development",
            "recommended_product_development_phase": "3-productize-the-experiment",
            "open_questions": [],
            "next_decision": "Run product-development against this intake package.",
        },
        "product_intake": product_intake,
        "created_at": models.timestamp_text(now),
    }
    path = (
        _owned_path(run_path(run_id, run_root), Path("intake-packages") / f"{package_id}.json")
        if run_id is not None and package_root is None
        else _owned_path(root, f"{package_id}.json")
    )
    write_json(path, package)
    packet["intake_package_status"] = "package_created"
    return {"ok": True, "intake_id": package_id, "artifact_path": str(path), "package": package}


def _record_id(prefix: str, key: str | None = None, *, now=None) -> str:
    base = models.slugify(key, fallback=prefix)
    return f"{prefix}-{models.run_id_timestamp(now)}-{base}"


def record_activation(
    intake_package: Mapping[str, Any],
    *,
    activated_path: str,
    evidence: Mapping[str, Any],
    run_id: str | None = None,
    run_root: str | Path | None = None,
    now=None,
) -> dict[str, Any]:
    if run_id is None:
        raise ValueError("run_id is required for activation records")
    if not state_path(run_id, run_root).is_file():
        raise ValueError("run_id must name an initialized Learning run")
    package_link_result = validate_run_intake_package_link(intake_package, run_id=run_id, run_root=run_root)
    if not package_link_result["ok"]:
        return {
            **package_link_result,
            "record_path": None,
            "intake_id": intake_package.get("intake_id"),
            "activated": False,
        }
    activation_candidate = intake_package.get("activation_candidate")
    activation_candidate = activation_candidate if isinstance(activation_candidate, Mapping) else {}
    record = {
        "activation_id": _record_id("activation", str(intake_package.get("intake_id") or activated_path), now=now),
        "intake_id": intake_package.get("intake_id"),
        "target_surface": evidence.get("target_surface") or intake_package.get("target_surface") or activation_candidate.get("proposed_surface"),
        "activated_path": activated_path,
        "product_development_evidence": dict(evidence),
        "activation_summary": evidence.get("summary") or "Activated through product-development evidence.",
        "created_at": models.timestamp_text(now),
    }
    result = validation.validate_activation_record(record)
    if not result["ok"]:
        return {**result, "record_path": None, "intake_id": record["intake_id"], "activated": False}
    path = _owned_path(run_path(run_id, run_root), Path("activation-records") / f"{record['activation_id']}.json")
    write_json(path, record)
    return {
        "ok": True,
        "run_id": run_id,
        "record_path": str(path),
        "intake_id": record["intake_id"],
        "activated": True,
        "record": record,
    }


def validate_run_intake_package_link(
    intake_package: Mapping[str, Any],
    *,
    run_id: str,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    expected_intake_id = str(intake_package.get("intake_id") or "")
    expected_source_id = str(intake_package.get("source_candidate_id") or intake_package.get("packet_id") or "")
    if not expected_intake_id:
        return {"ok": False, "errors": ["intake package must include intake_id"]}
    if not expected_source_id:
        return {"ok": False, "errors": ["intake package must include source_candidate_id or packet_id"]}

    target = run_path(run_id, run_root)
    linked_candidate_ids = accepted_candidate_ids(target)
    for package_path in _run_json_paths(target, "intake-packages"):
        package = read_json(package_path)
        package_intake_id = str(package.get("intake_id") or "")
        package_source_id = str(package.get("source_candidate_id") or package.get("packet_id") or "")
        if package_intake_id == expected_intake_id and package_source_id == expected_source_id:
            if package_source_id in linked_candidate_ids:
                return {"ok": True, "errors": []}
            return {"ok": False, "errors": ["intake package is not linked to an accepted candidate in the run"]}
    return {"ok": False, "errors": ["intake package must exist in the run before activation recording"]}


def accepted_candidate_ids(target: Path) -> set[str]:
    ids: set[str] = set()
    for candidate_path in _run_json_paths(target, "candidates"):
        packet = read_json(candidate_path)
        packet_mode = str(packet.get("learning_mode") or "")
        expected_mode = packet_mode if packet_mode in {"reactive", "active"} else None
        result = validation.validate_candidate_packet(
            packet,
            expected_mode=expected_mode,
            require_intake_gate=True,
        )
        if result["ok"] and packet.get("packet_id"):
            ids.add(str(packet["packet_id"]))
    return ids


def validate_acceptance(run_id: str, *, run_root: str | Path | None = None) -> dict[str, Any]:
    target = run_path(run_id, run_root)
    errors: list[str] = []
    if not target.exists():
        return {
            "ok": False,
            "run_id": run_id,
            "status": "incomplete",
            "errors": [f"missing run directory: {target}"],
            "reactive_ready": False,
            "active_ready": False,
            "activation_ready": False,
        }

    reactive_ready = False
    active_ready = False
    linked_candidate_ids: set[str] = set()
    for candidate_path in _run_json_paths(target, "candidates"):
        packet = read_json(candidate_path)
        packet_mode = str(packet.get("learning_mode") or "")
        expected_mode = packet_mode if packet_mode in {"reactive", "active"} else None
        result = validation.validate_candidate_packet(
            packet,
            expected_mode=expected_mode,
            require_intake_gate=True,
        )
        if not result["ok"]:
            continue
        if packet.get("packet_id"):
            linked_candidate_ids.add(str(packet["packet_id"]))
        if packet.get("learning_mode") == "reactive":
            reactive_ready = True
        if packet.get("learning_mode") == "active":
            active_ready = True

    linked_intake_ids: set[str] = set()
    for package_path in _run_json_paths(target, "intake-packages"):
        package = read_json(package_path)
        source_id = str(package.get("source_candidate_id") or package.get("packet_id") or "")
        intake_id = str(package.get("intake_id") or "")
        if intake_id and source_id in linked_candidate_ids:
            linked_intake_ids.add(intake_id)

    activation_ready = False
    valid_activation_seen = False
    for activation_path in _run_json_paths(target, "activation-records"):
        record = read_json(activation_path)
        result = validation.validate_activation_record(record)
        valid_activation_seen = valid_activation_seen or result["ok"]
        activation_ready = activation_ready or (result["ok"] and str(record.get("intake_id") or "") in linked_intake_ids)

    if not reactive_ready:
        errors.append("reactive learning evidence is missing")
    if not active_ready:
        errors.append("active learning evidence is missing")
    if not activation_ready and valid_activation_seen:
        errors.append("activation evidence is not linked to an accepted intake package")
    elif not activation_ready:
        errors.append("activation evidence is missing")

    status = "accepted" if not errors else "incomplete"
    payload = {
        "ok": not errors,
        "run_id": run_id,
        "status": status,
        "errors": errors,
        "reactive_ready": reactive_ready,
        "active_ready": active_ready,
        "activation_ready": activation_ready,
    }
    write_json(_owned_path(target, "acceptance-plan.json"), {**acceptance_plan(run_id), **payload})
    return payload
