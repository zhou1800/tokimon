"""Pass-lock and recheck-plan support for code-audit runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tokimon.workflow.audit_project import contained_child
from tokimon.workflow.code_audit import contract, inputs, ledger
from tokimon.workflow.code_audit.report_evaluation import evaluation_from_report


LOCK_ACTION = "use_validated_pass_lock"
FRESH_ACTION = "fresh_report_required"
REUSABLE_STATUSES = frozenset(("pass", "not_applicable"))


def _display_path(path: Path) -> str:
    return ledger.display_path(path)


def _load_json(path: Path | str) -> Any:
    return json.loads(ledger.repo_path(path).read_text(encoding="utf-8"))


def _load_matrix(matrix_path: Path | str) -> dict[str, Any]:
    payload = _load_json(matrix_path)
    if not isinstance(payload, Mapping):
        raise ValueError("matrix input must be a JSON object")
    if payload.get("schema_version") != contract.CHECKLIST_MATRIX_SCHEMA_VERSION:
        raise ValueError(f"unsupported matrix schema_version: {payload.get('schema_version')}")
    if payload.get("workflow_id") != contract.WORKFLOW_ID:
        raise ValueError(f"unexpected matrix workflow_id: {payload.get('workflow_id')}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("matrix input must contain rows")
    inputs.validate_snapshot(payload, check_current=False)
    return dict(payload)


def _subject_index(matrix_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    subjects = matrix_payload.get("audit_subjects", [])
    if not isinstance(subjects, list):
        raise ValueError("matrix audit_subjects must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for subject in subjects:
        if not isinstance(subject, Mapping):
            raise ValueError("matrix audit_subjects entries must be objects")
        subject_id = str(subject.get("audit_subject_id", "")).strip()
        if not subject_id:
            raise ValueError("matrix audit_subjects entries require audit_subject_id")
        indexed[subject_id] = dict(subject)
    return indexed


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    subject_id = str(row.get("audit_subject_id", "")).strip()
    rule_id = str(row.get("rule_id", "")).strip()
    if not subject_id or not rule_id:
        raise ValueError("matrix rows require audit_subject_id and rule_id")
    return subject_id, rule_id


def _row_signature(
    *,
    matrix_payload: Mapping[str, Any],
    row: Mapping[str, Any],
    subject_signatures_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    subject_id, rule_id = _row_key(row)
    subject_signature = subject_signatures_by_id.get(subject_id)
    if subject_signature is None:
        raise ValueError(f"matrix row references unknown audit_subject_id: {subject_id}")
    return {
        "project": matrix_payload["project"],
        "scope_id": str(row.get("scope_id") or matrix_payload.get("scope_id") or "").strip(),
        "audit_subject_id": subject_id,
        "rule_id": rule_id,
        "rule_hash": str(row.get("rule_hash", "")).strip(),
        **subject_signature,
    }


def _validate_prior_pass_report(lock: Mapping[str, Any]) -> dict[str, Any]:
    lock_core = {
        key: value for key, value in lock.items() if key not in {"lock_id", "status"}
    }
    if str(lock.get("lock_id", "")) != inputs.canonical_hash(lock_core):
        raise ValueError("pass lock lock_id is invalid")
    report_path = str(lock.get("pass_report_path", "")).strip()
    if not report_path:
        raise ValueError("pass lock missing pass_report_path")
    if not ledger.repo_path(report_path).is_file():
        raise ValueError("pass lock report path is missing")
    expected_hash = str(lock.get("pass_report_hash", "")).strip()
    content = ledger.repo_path(report_path).read_bytes()
    actual_hash = hashlib.sha256(content).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError("pass lock report hash changed")
    report = json.loads(content)
    if not isinstance(report, Mapping):
        raise ValueError("pass lock report must be a JSON object")
    from tokimon.workflow.code_audit.reports import validate_report
    validated = validate_report(report, report_path=ledger.repo_path(report_path))
    if report.get("schema_version") != contract.AUDIT_REPORT_SCHEMA_VERSION:
        raise ValueError("pass lock report schema_version is unsupported")
    if report.get("audit_snapshot_id") != lock.get("audit_snapshot_id"):
        raise ValueError("pass lock source snapshot changed")
    if report.get("workflow_id") != contract.WORKFLOW_ID:
        raise ValueError("pass lock report workflow_id is invalid")
    report_status = str(report.get("status", "")).strip()
    lock_status = str(lock.get("status", "")).strip()
    if report_status not in REUSABLE_STATUSES:
        raise ValueError("pass lock report status is not reusable")
    if report_status != lock_status:
        raise ValueError("pass lock report status changed")
    if str(report.get("audit_subject_id", "")).strip() != str(lock.get("audit_subject_id", "")).strip():
        raise ValueError("pass lock report audit_subject_id changed")
    if str(report.get("rule_id", "")).strip() != str(lock.get("rule_id", "")).strip():
        raise ValueError("pass lock report rule_id changed")
    try:
        evaluation = evaluation_from_report(report)
        ledger.validate_evaluation(
            evaluation,
            expected_scope_id=str(lock.get("scope_id", "")).strip(),
            expected_run_id=str(lock.get("run_id", "")).strip(),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"pass lock report is invalid: {exc}") from exc
    return validated


def _lock_from_report(
    *,
    matrix_payload: Mapping[str, Any],
    row: Mapping[str, Any],
    report: Mapping[str, Any],
    subject_signatures_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    status = str(report.get("status", "")).strip()
    if status not in REUSABLE_STATUSES:
        return None
    signature = _row_signature(
        matrix_payload=matrix_payload,
        row=row,
        subject_signatures_by_id=subject_signatures_by_id,
    )
    report_path = str(report.get("report_path", "")).strip()
    if not report_path:
        return None
    content = ledger.repo_path(report_path).read_bytes()
    from tokimon.workflow.code_audit.reports import validate_report
    validated = validate_report(json.loads(content), report_path=ledger.repo_path(report_path))
    if evaluation_from_report(validated) != evaluation_from_report(report):
        raise ValueError("source report changed while compiling")
    pass_report_hash = hashlib.sha256(content).hexdigest()
    lock_core = {
        **signature,
        "run_id": str(report.get("run_id", "")).strip(),
        "audit_snapshot_id": report["audit_snapshot_id"],
        "pass_report_path": report_path,
        "pass_report_hash": pass_report_hash,
        "verification_evidence_pointers": list(report.get("verification", [])),
    }
    return {
        "lock_id": inputs.canonical_hash(lock_core),
        "status": status,
        **lock_core,
    }


def write_pass_locks(
    *,
    matrix_path: Path | str,
    reports: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
    reports_dir: Path | str,
    recheck_plan_path: Path | str | None = None,
) -> dict[str, Any]:
    matrix_payload = _load_matrix(matrix_path)
    inputs.validate_snapshot(matrix_payload)
    reported_subject_ids = {
        str(report.get("audit_subject_id", "")).strip()
        for report in reports
        if str(report.get("status", "")).strip() in REUSABLE_STATUSES
    }
    frozen = matrix_payload["input_snapshots"]
    subject_signatures_by_id = {
        subject_id: frozen[subject_id]["signature"] for subject_id in reported_subject_ids
        if "signature" in frozen.get(subject_id, {})
    }
    subject_signature_errors = {
        subject_id: frozen[subject_id]["error"] for subject_id in reported_subject_ids
        if "error" in frozen.get(subject_id, {})
    }
    rows_by_key = {_row_key(row): row for row in matrix_payload["rows"]}
    locks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for report in reports:
        key = (str(report.get("audit_subject_id", "")).strip(), str(report.get("rule_id", "")).strip())
        row = rows_by_key.get(key)
        if row is None:
            continue
        if key[0] in subject_signature_errors:
            skipped.append(
                {
                    "audit_subject_id": key[0],
                    "rule_id": key[1],
                    "reason": subject_signature_errors[key[0]],
                }
            )
            continue
        if report.get("run_id") == matrix_payload["run_id"] and report.get("audit_snapshot_id") != matrix_payload["audit_snapshot_id"]:
            raise ValueError("fresh source report does not match matrix snapshot")
        lock = _lock_from_report(
            matrix_payload=matrix_payload,
            row=row,
            report=report,
            subject_signatures_by_id=subject_signatures_by_id,
        )
        if lock is not None:
            locks.append(lock)
    target_dir = ledger.repo_path(output_dir)
    locks_path = contained_child(target_dir, contract.PASS_LOCKS_NAME, label="pass locks output")
    payload = {
        "schema_version": contract.PASS_LOCKS_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "artifact_role": "pass_lock_support",
        "audit_authority": False,
        "created_at": ledger.utc_now(),
        "matrix_path": _display_path(ledger.repo_path(matrix_path)),
        "reports_dir": _display_path(ledger.repo_path(reports_dir)),
        "recheck_plan_path": _display_path(ledger.repo_path(recheck_plan_path)) if recheck_plan_path else None,
        "lock_count": len(locks),
        "skipped_count": len(skipped),
        "locks": locks,
        "skipped": skipped,
    }
    ledger.write_json(locks_path, payload)
    return {
        "pass_locks_path": _display_path(locks_path),
        "pass_lock_count": len(locks),
        "pass_lock_skipped_count": len(skipped),
    }


def _load_pass_locks(path: Path | str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = _load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, Mapping):
        return None, "pass-locks input must be a JSON object"
    if payload.get("schema_version") != contract.PASS_LOCKS_SCHEMA_VERSION:
        return None, f"unsupported pass-locks schema_version: {payload.get('schema_version')}"
    if payload.get("workflow_id") != contract.WORKFLOW_ID:
        return None, f"unexpected pass-locks workflow_id: {payload.get('workflow_id')}"
    if payload.get("audit_authority") is not False:
        return None, "pass-locks artifact must be support-only"
    return dict(payload), None


def _lock_index(payload: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    locks = payload.get("locks", [])
    if not isinstance(locks, list):
        raise ValueError("pass-locks locks must be a list")
    for lock in locks:
        if not isinstance(lock, Mapping):
            raise ValueError("pass-lock entries must be objects")
        if str(lock.get("status", "")).strip() not in REUSABLE_STATUSES:
            continue
        key = (
            str(lock.get("scope_id", "")).strip(),
            str(lock.get("audit_subject_id", "")).strip(),
            str(lock.get("rule_id", "")).strip(),
        )
        if not all(key):
            continue
        if key in indexed:
            raise ValueError(f"duplicate pass lock: {' '.join(key)}")
        indexed[key] = dict(lock)
    return indexed


def _compare_signature(current: Mapping[str, Any], lock: Mapping[str, Any]) -> list[str]:
    reason_codes: list[str] = []
    for field in (
        "project",
        "scope_id",
        "audit_subject_id",
        "rule_id",
        "rule_hash",
        "subject_definition_hash",
        "implementation_path_hashes",
        "test_path_hashes",
        "owning_spec_path_hashes",
        "guideline_path_hashes",
        "verification_commands",
        "verification_command_hash",
    ):
        if current.get(field) != lock.get(field):
            reason_codes.append(f"{field}_changed")
    return reason_codes


def build_recheck_plan_payload(
    *,
    matrix_path: Path | str,
    prior_pass_locks_path: Path | str,
) -> dict[str, Any]:
    matrix_payload = _load_matrix(matrix_path)
    subjects_by_id = _subject_index(matrix_payload)
    pass_locks, lock_error = _load_pass_locks(prior_pass_locks_path)
    lock_validation_available = pass_locks is not None
    lock_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    if pass_locks is not None:
        try:
            lock_index = _lock_index(pass_locks)
        except ValueError as exc:
            lock_error = str(exc)
            lock_validation_available = False
    locked_subject_ids = {
        str(lock.get("audit_subject_id", "")).strip() for lock in lock_index.values()
    }
    subject_signatures_by_id, subject_signature_errors = (
        inputs.subject_signature_results(
            matrix_payload=matrix_payload,
            subjects_by_id={
                subject_id: subject
                for subject_id, subject in subjects_by_id.items()
                if subject_id in locked_subject_ids
            },
        )
        if lock_validation_available and locked_subject_ids
        else ({}, {})
    )

    rows: list[dict[str, Any]] = []
    for row in matrix_payload["rows"]:
        subject_id, rule_id = _row_key(row)
        scope_id = str(row.get("scope_id") or matrix_payload.get("scope_id") or "").strip()
        reason_codes: list[str] = []
        lock = lock_index.get((scope_id, subject_id, rule_id))
        prior_report: dict[str, Any] | None = None
        if not lock_validation_available:
            reason_codes.append("pass_lock_validation_unavailable")
        elif lock is None:
            reason_codes.append("no_prior_pass_lock")
        else:
            try:
                if subject_id in subject_signature_errors:
                    raise ValueError(subject_signature_errors[subject_id])
                signature = _row_signature(
                    matrix_payload=matrix_payload,
                    row=row,
                    subject_signatures_by_id=subject_signatures_by_id,
                )
                reason_codes.extend(_compare_signature(signature, lock))
                prior_report = _validate_prior_pass_report(lock)
            except (OSError, ValueError) as exc:
                reason_codes.append(str(exc).replace(" ", "_"))
        action = LOCK_ACTION if lock is not None and not reason_codes else FRESH_ACTION
        row_plan: dict[str, Any] = {
            "scope_id": scope_id,
            "audit_subject_id": subject_id,
            "rule_id": rule_id,
            "action": action,
            "reason_codes": reason_codes or ["inputs_unchanged"],
        }
        if lock is not None:
            row_plan["lock_id"] = lock.get("lock_id")
            row_plan["prior_report_path"] = lock.get("pass_report_path")
            row_plan["prior_report_hash"] = lock.get("pass_report_hash")
        if action == LOCK_ACTION and prior_report is not None:
            prior_evaluation = evaluation_from_report(prior_report)
            row_plan["locked_evaluation"] = {
                **prior_evaluation,
                "project": matrix_payload["project"],
                "compiled_for_audit_snapshot_id": matrix_payload["audit_snapshot_id"],
                "run_id": str(matrix_payload.get("run_id", "")).strip(),
                "scope_id": scope_id,
                "rule_hash": row.get("rule_hash"),
                "evaluation_source": "validated_prior_pass_lock",
                "prior_run_id": prior_evaluation["run_id"],
                "prior_report_path": lock.get("pass_report_path"),
                "prior_report_hash": lock.get("pass_report_hash"),
                "lock_id": lock.get("lock_id"),
            }
        rows.append(row_plan)

    counts = Counter(row["action"] for row in rows)
    locked = counts[LOCK_ACTION]
    fresh = counts[FRESH_ACTION]
    return {
        "schema_version": contract.RECHECK_PLAN_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "artifact_role": "recheck_plan_support",
        "audit_authority": False,
        "created_at": ledger.utc_now(),
        "scope_id": matrix_payload.get("scope_id"),
        "run_id": matrix_payload.get("run_id"),
        "matrix_path": _display_path(ledger.repo_path(matrix_path)),
        "prior_pass_locks_path": _display_path(ledger.repo_path(prior_pass_locks_path)),
        "lock_validation_available": lock_validation_available,
        "lock_validation_error": lock_error,
        "complete_fresh_reports_required": not lock_validation_available,
        "matrix_count": len(rows),
        "locked_row_count": locked,
        "fresh_required_row_count": fresh,
        "rows": rows,
    }


def write_recheck_plan(
    *,
    matrix_path: Path | str,
    prior_pass_locks_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    output_path = contained_child(ledger.repo_path(output_dir), contract.RECHECK_PLAN_NAME, label="recheck plan output")
    payload = build_recheck_plan_payload(matrix_path=matrix_path, prior_pass_locks_path=prior_pass_locks_path)
    artifact_paths = {"recheck_plan_json": _display_path(output_path)}
    payload["artifact_paths"] = artifact_paths
    ledger.write_json(output_path, payload)
    return {
        "workflow_id": contract.WORKFLOW_ID,
        "artifact_paths": artifact_paths,
        "summary": {
            "schema_version": contract.RECHECK_PLAN_SCHEMA_VERSION,
            "workflow_id": contract.WORKFLOW_ID,
            "audit_authority": False,
            "scope_id": payload["scope_id"],
            "run_id": payload["run_id"],
            "matrix_count": payload["matrix_count"],
            "locked_row_count": payload["locked_row_count"],
            "fresh_required_row_count": payload["fresh_required_row_count"],
            "lock_validation_available": payload["lock_validation_available"],
            "complete_fresh_reports_required": payload["complete_fresh_reports_required"],
        },
    }


def validated_locked_rows(
    *,
    recheck_plan_path: Path | str,
    matrix_path: Path | str,
) -> dict[tuple[str, str], dict[str, Any]]:
    plan = _load_json(recheck_plan_path)
    if not isinstance(plan, Mapping):
        raise ValueError("recheck-plan input must be a JSON object")
    if plan.get("schema_version") != contract.RECHECK_PLAN_SCHEMA_VERSION:
        raise ValueError(f"unsupported recheck-plan schema_version: {plan.get('schema_version')}")
    if plan.get("workflow_id") != contract.WORKFLOW_ID:
        raise ValueError(f"unexpected recheck-plan workflow_id: {plan.get('workflow_id')}")
    if plan.get("audit_authority") is not False:
        raise ValueError("recheck-plan artifact must be support-only")
    prior_path = str(plan.get("prior_pass_locks_path", "")).strip()
    if not prior_path:
        raise ValueError("recheck-plan missing prior_pass_locks_path")
    recomputed = build_recheck_plan_payload(matrix_path=matrix_path, prior_pass_locks_path=prior_path)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in recomputed["rows"]:
        if row["action"] != LOCK_ACTION:
            continue
        evaluation = row.get("locked_evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        rows[(row["audit_subject_id"], row["rule_id"])] = dict(row)
    return rows
