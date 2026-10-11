"""Structured report coverage for code-audit runs."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tokimon.workflow import audit_review, report_coverage
from tokimon.workflow.audit_project import contained_child
from tokimon.workflow.code_audit import contract, inputs, ledger, recheck_plan
from tokimon.workflow.code_audit.report_evaluation import evaluation_from_report


def _display_path(path: Path) -> str:
    return ledger.display_path(path)


def _expected_index(matrix_path: Path | str) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (key[0], key[1]): row
        for key, row in report_coverage.expected_index(
            matrix_path,
            key_fields=("audit_subject_id", "rule_id"),
            repo_path=ledger.repo_path,
        ).items()
    }


def _require_text(report: Mapping[str, Any], field: str, report_path: Path) -> str:
    value = str(report.get(field, "")).strip()
    if not value:
        raise ValueError(f"{_display_path(report_path)} missing required field: {field}")
    return value


def _string_list(report: Mapping[str, Any], field: str, report_path: Path) -> list[str]:
    value = report.get(field)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{_display_path(report_path)} {field} must be a list")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{_display_path(report_path)} {field} must contain only non-empty strings")
        normalized.append(item.strip())
    return normalized


def _optional_string_list(report: Mapping[str, Any], field: str, report_path: Path) -> list[str]:
    if field not in report:
        return []
    return _string_list(report, field, report_path)


def _optional_evidence_list(report: Mapping[str, Any], report_path: Path) -> list[dict[str, Any]]:
    if "evidence" not in report:
        return []
    value = report["evidence"]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{_display_path(report_path)} evidence must be a list")
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"{_display_path(report_path)} evidence item {index} must be an object")
        evidence.append(dict(item))
    return evidence


def validate_report(report: Mapping[str, Any], *, report_path: Path) -> dict[str, Any]:
    if "schema_version" not in report:
        raise ValueError(f"{_display_path(report_path)} missing required field: schema_version")
    schema_version = report["schema_version"]
    if schema_version not in (contract.AUDIT_REPORT_SCHEMA_VERSION, *contract.LEGACY_AUDIT_REPORT_SCHEMA_VERSIONS):
        raise ValueError(f"{_display_path(report_path)} unsupported report schema_version: {schema_version}")
    workflow_id = _require_text(report, "workflow_id", report_path)
    if workflow_id != contract.WORKFLOW_ID:
        raise ValueError(f"{_display_path(report_path)} unexpected workflow_id: {workflow_id}")
    run_id = _require_text(report, "run_id", report_path) if schema_version >= 2 else report.get("run_id")
    scope_id = _require_text(report, "scope_id", report_path) if schema_version >= 2 else report.get("scope_id")
    subject_id = _require_text(report, "audit_subject_id", report_path)
    rule_id = _require_text(report, "rule_id", report_path)
    status = _require_text(report, "status", report_path)
    if status not in contract.AUDIT_REPORT_STATUSES:
        raise ValueError(f"{_display_path(report_path)} invalid report status: {status}")
    summary = _require_text(report, "summary", report_path)
    reason = _require_text(report, "reason", report_path)
    normalized = {
        "schema_version": schema_version,
        "workflow_id": workflow_id,
        "run_id": run_id,
        "scope_id": scope_id,
        "audit_subject_id": subject_id,
        "rule_id": rule_id,
        "status": status,
        "summary": summary,
        "reason": reason,
        "implementation_paths_examined": _string_list(report, "implementation_paths_examined", report_path),
        "test_paths_examined": _string_list(report, "test_paths_examined", report_path),
        "owning_specs_examined": _string_list(report, "owning_specs_examined", report_path),
        "evidence": _optional_evidence_list(report, report_path),
        "verification": _optional_string_list(report, "verification", report_path),
        "report_path": _display_path(report_path),
    }
    for optional in ("review_prompt", "normalized_subject"):
        if optional in report:
            normalized[optional] = report[optional]
    if schema_version == contract.AUDIT_REPORT_SCHEMA_VERSION:
        normalized["audit_snapshot_id"] = _require_text(report, "audit_snapshot_id", report_path)
        normalized["review"] = audit_review.validate_review(
            report, report_path=ledger.repo_path(report_path), key_fields=("audit_subject_id", "rule_id"),
        )
    elif "review" in report:
        normalized["review"] = report["review"]
    if schema_version >= 2:
        ledger.validate_evaluation(evaluation_from_report(normalized))
    return normalized


def read_reports(reports_dir: Path | str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    return report_coverage.read_reports(
        reports_dir,
        repo_path=ledger.repo_path,
        display_path=_display_path,
        validate_report=lambda report, path: validate_report(report, report_path=path),
    )


def _matrix_identity(matrix_path: Path | str) -> dict[str, Any]:
    payload = report_coverage.load_json(matrix_path, repo_path=ledger.repo_path)
    if not isinstance(payload, Mapping):
        raise ValueError("matrix input must be a JSON object")
    version = payload.get("schema_version")
    if version not in (contract.CHECKLIST_MATRIX_SCHEMA_VERSION, *contract.LEGACY_CHECKLIST_MATRIX_SCHEMA_VERSIONS):
        raise ValueError(f"unsupported matrix schema_version: {version}")
    if version == contract.CHECKLIST_MATRIX_SCHEMA_VERSION:
        inputs.validate_snapshot(payload)
    if payload.get("workflow_id") != contract.WORKFLOW_ID:
        raise ValueError(f"unexpected matrix workflow_id: {payload.get('workflow_id')}")
    run_id = str(payload.get("run_id", "")).strip()
    scope_id = str(payload.get("scope_id", "")).strip()
    if not run_id or not scope_id:
        raise ValueError("matrix input requires run_id and scope_id")
    return {"schema_version": version, "run_id": run_id, "scope_id": scope_id,
            "audit_snapshot_id": payload.get("audit_snapshot_id"), "project": payload.get("project")}


def collect_report_coverage(
    *,
    matrix_path: Path | str,
    reports_dir: Path | str,
    recheck_plan_path: Path | str | None = None,
) -> dict[str, Any]:
    expected = _expected_index(matrix_path)
    identity = _matrix_identity(matrix_path)
    matrix_run_id, matrix_scope_id = identity["run_id"], identity["scope_id"]
    for row in expected.values() if identity["schema_version"] >= 2 else ():
        if str(row.get("run_id", "")).strip() != matrix_run_id:
            raise ValueError("matrix row run_id does not match matrix run_id")
        if str(row.get("scope_id", "")).strip() != matrix_scope_id:
            raise ValueError("matrix row scope_id does not match matrix scope_id")
    parsed_reports, invalid_reports = read_reports(reports_dir)
    report_file_count = len(parsed_reports) + len(invalid_reports)
    reports: list[dict[str, Any]] = []
    for report in parsed_reports:
        mismatch: list[str] = []
        if identity["schema_version"] >= 2 and report["run_id"] != matrix_run_id:
            mismatch.append(f"run_id expected {matrix_run_id}; got {report['run_id']}")
        if identity["schema_version"] >= 2 and report["scope_id"] != matrix_scope_id:
            mismatch.append(f"scope_id expected {matrix_scope_id}; got {report['scope_id']}")
        if identity["schema_version"] == contract.CHECKLIST_MATRIX_SCHEMA_VERSION:
            if report["schema_version"] != contract.AUDIT_REPORT_SCHEMA_VERSION:
                mismatch.append("current matrices require current-schema reports")
            if report.get("audit_snapshot_id") != identity["audit_snapshot_id"]:
                mismatch.append("audit_snapshot_id does not match matrix")
        if mismatch:
            invalid_reports.append(
                {"report_path": report["report_path"], "error": "report provenance mismatch: " + "; ".join(mismatch)}
            )
            continue
        reports.append(report)
    covered: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_map: dict[tuple[str, str], list[str]] = {}
    extra_reports: list[dict[str, str]] = []

    for report in reports:
        key = (report["audit_subject_id"], report["rule_id"])
        if key not in expected:
            extra_reports.append(
                {
                    "report_path": report["report_path"],
                    "audit_subject_id": key[0],
                    "rule_id": key[1],
                }
            )
            continue
        if key in covered:
            duplicate_map.setdefault(key, [covered[key]["report_path"]]).append(report["report_path"])
            continue
        covered[key] = report

    lock_validation_error = None
    locked_rows: dict[tuple[str, str], dict[str, Any]] = {}
    if recheck_plan_path:
        try:
            locked_rows = recheck_plan.validated_locked_rows(
                recheck_plan_path=recheck_plan_path,
                matrix_path=matrix_path,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            lock_validation_error = str(exc)
            locked_rows = {}
    unexpected_reports = [
        {
            "report_path": covered[key]["report_path"],
            "audit_subject_id": key[0],
            "rule_id": key[1],
            "planned_action": recheck_plan.LOCK_ACTION,
        }
        for key in sorted(set(covered) & set(locked_rows))
    ]
    for key in set(covered) & set(locked_rows):
        del covered[key]
    locked_coverage = {
        key: row
        for key, row in locked_rows.items()
        if key in expected
    }

    missing_reports = [
        {"audit_subject_id": key[0], "rule_id": key[1]}
        for key in expected
        if key not in covered and key not in locked_coverage
    ]
    duplicate_reports = [
        {
            "audit_subject_id": key[0],
            "rule_id": key[1],
            "report_paths": paths,
        }
        for key, paths in sorted(duplicate_map.items())
    ]
    status_counts = {status: 0 for status in contract.AUDIT_REPORT_STATUSES}
    status_counts.update(Counter(report["status"] for report in covered.values()))
    for row in locked_coverage.values():
        evaluation = row.get("locked_evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        status = str(evaluation.get("status", "")).strip()
        if status in status_counts:
            status_counts[status] += 1
    complete = not (
        missing_reports
        or invalid_reports
        or duplicate_reports
        or extra_reports
        or unexpected_reports
    )
    verification = {
        "workflow_id": contract.WORKFLOW_ID,
        "matrix_schema_version": identity["schema_version"],
        "audit_snapshot_id": identity["audit_snapshot_id"],
        "project": identity["project"],
        "run_id": matrix_run_id,
        "scope_id": matrix_scope_id,
        "matrix_path": _display_path(ledger.repo_path(matrix_path)),
        "reports_dir": _display_path(ledger.repo_path(reports_dir)),
        "recheck_plan_path": _display_path(ledger.repo_path(recheck_plan_path)) if recheck_plan_path else None,
        "expected_report_count": len(expected),
        "expected_fresh_report_count": len(expected) - len(locked_coverage),
        "report_file_count": report_file_count,
        "valid_report_count": len(reports),
        "covered_report_count": len(covered),
        "locked_row_count": len(locked_coverage),
        "coverage_count": len(covered) + len(locked_coverage),
        "missing_count": len(missing_reports),
        "invalid_count": len(invalid_reports),
        "duplicate_count": len(duplicate_reports),
        "extra_count": len(extra_reports),
        "unexpected_count": len(unexpected_reports),
        "lock_validation_error": lock_validation_error,
        "status_counts": status_counts,
        "complete": complete,
        "missing_reports": missing_reports,
        "invalid_reports": invalid_reports,
        "duplicate_reports": duplicate_reports,
        "extra_reports": extra_reports,
        "unexpected_reports": unexpected_reports,
        "locked_rows": list(locked_coverage.values()),
    }
    return {
        "verification": verification,
        "expected": expected,
        "fresh_reports": list(covered.values()),
        "covered_reports": covered,
        "locked_rows": locked_coverage,
    }


def verify_report_coverage(
    *,
    matrix_path: Path | str,
    reports_dir: Path | str,
    recheck_plan_path: Path | str | None = None,
) -> dict[str, Any]:
    return collect_report_coverage(
        matrix_path=matrix_path,
        reports_dir=reports_dir,
        recheck_plan_path=recheck_plan_path,
    )["verification"]


def _compiled_rows(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = snapshot["expected"]
    report_index = snapshot["covered_reports"]
    locked_index = snapshot["locked_rows"]
    evaluations: list[dict[str, Any]] = []
    for key, row in expected.items():
        report = report_index.get(key)
        locked_row = locked_index.get(key)
        if report:
            evaluation = evaluation_from_report(report)
        elif locked_row and isinstance(locked_row.get("locked_evaluation"), Mapping):
            evaluation = dict(locked_row["locked_evaluation"])
        else:
            continue
        evaluation["rule_hash"] = row.get("rule_hash")
        verification = snapshot["verification"]
        if verification["audit_snapshot_id"]:
            evaluation["compiled_for_audit_snapshot_id"] = verification["audit_snapshot_id"]
            evaluation["project"] = verification["project"]
        evaluations.append(evaluation)
    return evaluations


def compile_reports(
    *,
    matrix_path: Path | str,
    reports_dir: Path | str,
    output_dir: Path | str | None = None,
    recheck_plan_path: Path | str | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    snapshot = collect_report_coverage(
        matrix_path=matrix_path,
        reports_dir=reports_dir,
        recheck_plan_path=recheck_plan_path,
    )
    verification = snapshot["verification"]
    if require_complete and not verification["complete"]:
        return {"workflow_id": contract.WORKFLOW_ID, "compiled": False, "verification": verification}
    evaluations = _compiled_rows(snapshot)
    target_dir = ledger.repo_path(output_dir) if output_dir else ledger.repo_path(matrix_path).parent
    evaluations_path = contained_child(target_dir, contract.COMPILED_EVALUATIONS_NAME, label="compiled evaluations output")
    summary_path = contained_child(target_dir, contract.REPORTS_SUMMARY_NAME, label="reports summary output")
    fresh_reports = snapshot["fresh_reports"]
    locked_reports = [
        dict(
            row["locked_evaluation"],
            run_id=row["locked_evaluation"].get("prior_run_id", row["locked_evaluation"]["run_id"]),
            report_path=row["prior_report_path"],
        )
        for row in verification.get("locked_rows", [])
        if isinstance(row.get("locked_evaluation"), Mapping)
    ]
    lock_source_reports = [*fresh_reports, *locked_reports] if verification["complete"] else []
    if verification["matrix_schema_version"] == contract.CHECKLIST_MATRIX_SCHEMA_VERSION:
        pass_lock_result = recheck_plan.write_pass_locks(
            matrix_path=matrix_path,
            reports=lock_source_reports,
            output_dir=target_dir,
            reports_dir=reports_dir,
            recheck_plan_path=recheck_plan_path,
        )
    else:
        pass_lock_result = {"pass_locks_path": None, "pass_lock_count": 0}
    artifact_paths = {
        "evaluations_json": _display_path(evaluations_path),
        "summary_json": _display_path(summary_path),
        "pass_locks_json": pass_lock_result["pass_locks_path"],
    }
    summary = {
        "schema_version": verification["matrix_schema_version"],
        "workflow_id": contract.WORKFLOW_ID,
        "run_id": verification["run_id"],
        "audit_snapshot_id": verification["audit_snapshot_id"],
        "project": verification["project"],
        "scope_id": verification["scope_id"],
        "artifact_role": "report_aggregation_support",
        "audit_authority": False,
        "matrix_path": verification["matrix_path"],
        "reports_dir": verification["reports_dir"],
        "recheck_plan_path": verification["recheck_plan_path"],
        "expected_report_count": verification["expected_report_count"],
        "expected_fresh_report_count": verification["expected_fresh_report_count"],
        "covered_report_count": verification["covered_report_count"],
        "locked_row_count": verification["locked_row_count"],
        "coverage_count": verification["coverage_count"],
        "status_counts": verification["status_counts"],
        "complete": verification["complete"],
        "evaluation_count": len(evaluations),
        "pass_lock_count": pass_lock_result["pass_lock_count"],
        "artifact_paths": artifact_paths,
        "gaps": {
            "missing_reports": verification["missing_reports"],
            "invalid_reports": verification["invalid_reports"],
            "duplicate_reports": verification["duplicate_reports"],
            "extra_reports": verification["extra_reports"],
            "unexpected_reports": verification["unexpected_reports"],
        },
    }
    ledger.write_json(
        evaluations_path,
        {
            "schema_version": verification["matrix_schema_version"],
            "workflow_id": contract.WORKFLOW_ID,
            "run_id": verification["run_id"],
            "audit_snapshot_id": verification["audit_snapshot_id"],
            "project": verification["project"],
            "scope_id": verification["scope_id"],
            "artifact_role": "row_evaluations_from_reports",
            "audit_authority": False,
            "evaluations": evaluations,
        },
    )
    ledger.write_json(summary_path, summary)
    return {
        "workflow_id": contract.WORKFLOW_ID,
        "compiled": True,
        "artifact_paths": artifact_paths,
        "summary": summary,
    }
