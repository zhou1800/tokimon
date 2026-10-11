"""Contract validation for Learning workflow artifacts."""

from __future__ import annotations

from typing import Any, Mapping

from tokimon.workflow.learning import contract, models


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_object(value: object) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _has_list(value: object) -> bool:
    return isinstance(value, list) and bool(value)


def _missing_text_fields(payload: Mapping[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if not _has_text(payload.get(field))]


def _object_list_errors(
    payload: Mapping[str, Any],
    *,
    field: str,
    required_fields: tuple[str, ...],
    label: str,
) -> list[str]:
    value = payload.get(field)
    if not _has_list(value):
        return [f"{label}.{field} must be a non-empty list"]

    errors: list[str] = []
    for index, item in enumerate(value):
        item_label = f"{label}.{field}[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{item_label} must be an object")
            continue
        for item_field in required_fields:
            if not _has_text(item.get(item_field)):
                errors.append(f"{item_label}.{item_field} must be a non-empty string")
    return errors


def validate_candidate_packet(
    packet: Mapping[str, Any],
    *,
    expected_mode: str | None = None,
    require_intake_gate: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    missing_fields = [field for field in contract.REQUIRED_CANDIDATE_FIELDS if field not in packet]

    mode = str(packet.get("learning_mode") or "")
    if expected_mode is not None and mode != expected_mode:
        errors.append(f"learning_mode must be {expected_mode}")
    if mode not in {"reactive", "active"}:
        errors.append("learning_mode must be reactive or active")

    for field in contract.REQUIRED_CANDIDATE_FIELDS:
        if field in {
            "candidate_review_status",
            "operator_intake_status",
            "intake_package_status",
            "reviewer_records",
            "operator_approval",
        }:
            continue
        if not _has_text(packet.get(field)):
            errors.append(f"{field} must be a non-empty string")

    review_status = packet.get("candidate_review_status")
    if review_status not in contract.VALID_CANDIDATE_STATUSES:
        errors.append("candidate_review_status is not valid")
    operator_status = packet.get("operator_intake_status")
    if operator_status not in contract.VALID_OPERATOR_INTAKE_STATUSES:
        errors.append("operator_intake_status is not valid")
    intake_package_status = packet.get("intake_package_status")
    if intake_package_status not in contract.VALID_INTAKE_PACKAGE_STATUSES:
        errors.append("intake_package_status is not valid")
    if not isinstance(packet.get("reviewer_records"), list):
        errors.append("reviewer_records must be a list")
    if not isinstance(packet.get("operator_approval"), Mapping):
        errors.append("operator_approval must be an object")

    if mode == "reactive":
        errors.extend(validate_reactive_evidence(packet).get("errors", []))
    if mode == "active":
        errors.extend(validate_active_evidence(packet).get("errors", []))

    if require_intake_gate:
        errors.extend(intake_gate_errors(packet))

    return {
        "ok": not errors,
        "packet_id": packet.get("packet_id"),
        "mode": mode,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "missing_fields": missing_fields,
    }


def validate_reactive_evidence(packet: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    trigger = models.normalize_token(packet.get("trigger"))
    if trigger in contract.REACTIVE_EXCLUDED_TRIGGER_VALUES:
        errors.append(f"reactive trigger is excluded: {trigger}")
    if trigger not in contract.REACTIVE_TRIGGER_VALUES:
        errors.append("reactive trigger must be a concrete failure, correction, test failure, code review finding, or document review finding")

    evidence = packet.get("reactive_evidence")
    if not isinstance(evidence, Mapping):
        return {"ok": False, "errors": [*errors, "reactive_evidence must be an object"]}

    missing = _missing_text_fields(evidence, contract.REQUIRED_REACTIVE_EVIDENCE_FIELDS)
    errors.extend(f"reactive_evidence.{field} must be a non-empty string" for field in missing)

    failure_type = models.normalize_token(evidence.get("failure_type"))
    if failure_type in contract.REACTIVE_EXCLUDED_TRIGGER_VALUES:
        errors.append(f"reactive failure_type is excluded: {failure_type}")
    if failure_type not in contract.REACTIVE_TRIGGER_VALUES:
        errors.append("reactive_evidence.failure_type must be an accepted reactive evidence type")

    return {"ok": not errors, "errors": errors}


def validate_active_evidence(packet: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    trigger = models.normalize_token(packet.get("trigger"))
    if trigger in contract.ACTIVE_EXCLUDED_TRIGGER_VALUES:
        errors.append(f"active trigger is excluded: {trigger}")

    evidence = packet.get("active_evidence")
    if not isinstance(evidence, Mapping):
        return {"ok": False, "errors": [*errors, "active_evidence must be an object"]}

    text_fields = (
        "operator_learning_request",
        "target_capability",
        "target_question",
        "baseline_pressure_scenario",
        "before_after_delta",
    )
    missing = _missing_text_fields(evidence, text_fields)
    errors.extend(f"active_evidence.{field} must be a non-empty string" for field in missing)

    sources = evidence.get("selected_sources")
    if not _has_list(sources) or any(not _has_text(source) for source in sources):
        errors.append("active_evidence.selected_sources must be a non-empty list of non-empty source strings")
    if not _has_list(evidence.get("source_spans")):
        errors.append("active_evidence.source_spans must be a non-empty list")
    else:
        for index, span in enumerate(evidence["source_spans"]):
            if not isinstance(span, Mapping) or not _has_text(span.get("source")) or not _has_text(span.get("span")):
                errors.append(f"active_evidence.source_spans[{index}] must include source and span")

    if not _has_text(packet.get("pressure_scenario")):
        errors.append("pressure_scenario is required for active learning")

    return {"ok": not errors, "errors": errors}


def intake_gate_errors(packet: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if packet.get("candidate_review_status") != "accepted":
        errors.append("candidate_review_status must be accepted before intake package creation")
    if packet.get("operator_intake_status") != "approved":
        errors.append("operator_intake_status must be approved before intake package creation")
    records = packet.get("reviewer_records")
    if not _has_list(records):
        errors.append("reviewer_records must contain at least one review before intake package creation")
    else:
        for record in records:
            if (
                not isinstance(record, Mapping)
                or _missing_text_fields(record, ("reviewer_id", "reviewed_at"))
                or record.get("decision") not in contract.VALID_REVIEW_DECISIONS
            ):
                errors.append("reviewer_records must contain valid reviewer IDs, decisions and review timestamps")
                break
        if not isinstance(records[-1], Mapping) or records[-1].get("decision") != "accept":
            errors.append("reviewer_records must end with an accepting review before intake package creation")
    approval = packet.get("operator_approval")
    if not _has_object(approval) or approval.get("approved") is not True:
        errors.append("operator_approval must record approved=true before intake package creation")
    if not isinstance(approval, Mapping) or _missing_text_fields(
        approval, ("operator_id", "approval_note", "approved_at")
    ):
        errors.append("operator_approval must contain an operator ID, approval note and approval timestamp")
    return errors


def validate_activation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    for field in contract.REQUIRED_ACTIVATION_EVIDENCE_FIELDS:
        if field not in record:
            errors.append(f"activation record missing field: {field}")
    for field in ("activation_id", "intake_id", "target_surface", "activated_path", "activation_summary", "created_at"):
        if field in record and not _has_text(record.get(field)):
            errors.append(f"{field} must be a non-empty string")
    if record.get("target_surface") not in contract.VALID_TARGET_SURFACES:
        errors.append("target_surface is not valid")
    product_development_evidence = record.get("product_development_evidence")
    if not _has_object(product_development_evidence):
        errors.append("product_development_evidence must be a non-empty object")
    else:
        errors.extend(_validate_product_development_evidence(product_development_evidence))
    return {"ok": not errors, "errors": errors}


def _validate_product_development_evidence(evidence: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if evidence.get("acceptance_status") != "accepted":
        errors.append("product_development_evidence.acceptance_status must be accepted")
    if not _has_text(evidence.get("acceptance_summary")):
        errors.append("product_development_evidence.acceptance_summary must be a non-empty string")
    errors.extend(
        _object_list_errors(
            evidence,
            field="accepted_artifacts",
            required_fields=contract.REQUIRED_ACCEPTED_ARTIFACT_FIELDS,
            label="product_development_evidence",
        )
    )
    errors.extend(
        _object_list_errors(
            evidence,
            field="verification_evidence",
            required_fields=contract.REQUIRED_VERIFICATION_EVIDENCE_FIELDS,
            label="product_development_evidence",
        )
    )
    return errors
