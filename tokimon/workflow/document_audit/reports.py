"""Structured report coverage for document-audit runs."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tokimon.workflow import audit_review, report_coverage
from tokimon.workflow.audit_project import contained_child
from tokimon.workflow.document_audit import contract, ledger, matrix as matrix_module


def _display_path(path: Path) -> str:
    return ledger.display_path(path)


def _expected_index(matrix_path: Path | str) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (key[0], key[1]): row
        for key, row in report_coverage.expected_index(
            matrix_path,
            key_fields=("file_path", "rule_id"),
            repo_path=ledger.repo_path,
        ).items()
    }


def _require_text(report: Mapping[str, Any], field: str, report_path: Path) -> str:
    value = str(report.get(field, "")).strip()
    if not value:
        raise ValueError(f"{_display_path(report_path)} missing required field: {field}")
    return value


def _validate_report_findings(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    file_path: str,
    rule_id: str,
    status: str,
) -> list[dict[str, Any]]:
    raw_findings = report.get("findings", [])
    if raw_findings is None:
        raw_findings = []
    if not isinstance(raw_findings, list):
        raise ValueError(f"{_display_path(report_path)} findings must be a list")
    findings = [dict(finding) for finding in raw_findings]
    if status == "fail" and not findings:
        raise ValueError(f"{_display_path(report_path)} fail reports require at least one finding")
    if status != "fail" and findings:
        raise ValueError(f"{_display_path(report_path)} non-fail reports must not include findings")
    for finding in findings:
        ledger.validate_finding(finding)
        if str(finding["file_path"]) != file_path:
            raise ValueError(f"{_display_path(report_path)} finding file_path does not match report")
        if str(finding["rule_id"]) != rule_id:
            raise ValueError(f"{_display_path(report_path)} finding rule_id does not match report")
    return findings


def validate_report(report: Mapping[str, Any], *, report_path: Path) -> dict[str, Any]:
    schema_version = report.get("schema_version", contract.LEGACY_AUDIT_REPORT_SCHEMA_VERSIONS[0])
    supported_versions = {
        contract.AUDIT_REPORT_SCHEMA_VERSION,
        *contract.LEGACY_AUDIT_REPORT_SCHEMA_VERSIONS,
    }
    if schema_version not in supported_versions:
        raise ValueError(f"{_display_path(report_path)} unsupported report schema_version: {schema_version}")
    workflow_id = report.get("workflow_id", contract.WORKFLOW_ID)
    if workflow_id != contract.WORKFLOW_ID:
        raise ValueError(f"{_display_path(report_path)} unexpected workflow_id: {workflow_id}")
    file_path = _require_text(report, "file_path", report_path)
    rule_id = _require_text(report, "rule_id", report_path)
    status = _require_text(report, "status", report_path)
    if status not in contract.AUDIT_REPORT_STATUSES:
        raise ValueError(f"{_display_path(report_path)} invalid report status: {status}")
    summary = _require_text(report, "summary", report_path)
    reason = _require_text(report, "reason", report_path)
    findings = _validate_report_findings(
        report,
        report_path=report_path,
        file_path=file_path,
        rule_id=rule_id,
        status=status,
    )
    normalized = {
        "schema_version": schema_version,
        "workflow_id": workflow_id,
        "file_path": file_path,
        "rule_id": rule_id,
        "status": status,
        "summary": summary,
        "reason": reason,
        "findings": findings,
        "report_path": _display_path(report_path),
    }
    if schema_version in (2, contract.AUDIT_REPORT_SCHEMA_VERSION):
        normalized.update(
            {
                "run_id": _require_text(report, "run_id", report_path),
                "scope_id": _require_text(report, "scope_id", report_path),
                "audit_snapshot_id": _require_text(report, "audit_snapshot_id", report_path),
            }
        )
    if schema_version == contract.AUDIT_REPORT_SCHEMA_VERSION:
        normalized["review"] = audit_review.validate_review(report, report_path=ledger.repo_path(report_path), key_fields=("file_path", "rule_id"))
    elif "review" in report:
        normalized["review"] = report["review"]
    for optional in ("notes", "recommended_action", "verification", "evidence"):
        if optional in report:
            normalized[optional] = report[optional]
    return normalized


def _matrix_provenance(matrix_path: Path | str) -> dict[str, str] | None:
    payload = report_coverage.load_json(matrix_path, repo_path=ledger.repo_path)
    if not isinstance(payload, Mapping):
        raise ValueError("matrix input must be a JSON object")
    schema_version = payload.get(
        "schema_version",
        contract.LEGACY_CHECKLIST_MATRIX_SCHEMA_VERSIONS[0],
    )
    if schema_version in contract.LEGACY_CHECKLIST_MATRIX_SCHEMA_VERSIONS:
        return None
    if schema_version != contract.CHECKLIST_MATRIX_SCHEMA_VERSION:
        raise ValueError(f"unsupported matrix schema_version: {schema_version}")
    if payload.get("workflow_id") != contract.WORKFLOW_ID:
        raise ValueError(f"unexpected matrix workflow_id: {payload.get('workflow_id')}")
    matrix_module.validate_audit_snapshot(payload)
    provenance = {
        "run_id": str(payload.get("run_id", "")).strip(),
        "scope_id": str(payload.get("scope_id", "")).strip(),
        "audit_snapshot_id": str(payload.get("audit_snapshot_id", "")).strip(),
    }
    if not all(provenance.values()):
        raise ValueError("matrix input requires run_id, scope_id, and audit_snapshot_id")
    return provenance


def _read_reports(reports_dir: Path | str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    return report_coverage.read_reports(
        reports_dir,
        repo_path=ledger.repo_path,
        display_path=_display_path,
        validate_report=lambda report, path: validate_report(report, report_path=path),
    )


def collect_report_coverage(
    *,
    matrix_path: Path | str,
    reports_dir: Path | str,
    recheck_plan_path: Path | str | None = None,
) -> dict[str, Any]:
    expected = _expected_index(matrix_path)
    provenance = _matrix_provenance(matrix_path)
    parsed_reports, invalid_reports = _read_reports(reports_dir)
    reports: list[dict[str, Any]] = []
    for report in parsed_reports:
        if provenance is None:
            reports.append(report)
            continue
        mismatch = [
            f"{field} expected {expected_value}; got {report.get(field)}"
            for field, expected_value in provenance.items()
            if report.get(field) != expected_value
        ]
        if report.get("schema_version") != contract.AUDIT_REPORT_SCHEMA_VERSION:
            mismatch.append("current matrices require current-schema reports")
        if mismatch:
            invalid_reports.append(
                {
                    "report_path": report["report_path"],
                    "error": "report provenance mismatch: " + "; ".join(mismatch),
                }
            )
            continue
        reports.append(report)
    lock_validation_error = None
    plan_rows: dict[tuple[str, str], dict[str, Any]] = {}
    if recheck_plan_path:
        try:
            from tokimon.workflow.document_audit import recheck_plan

            plan_rows = recheck_plan.validated_plan_rows(
                recheck_plan_path=recheck_plan_path,
                matrix_path=matrix_path,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            lock_validation_error = str(exc)
            plan_rows = {}

    from tokimon.workflow.document_audit import recheck_plan as recheck_plan_module

    expected_actions = {
        key: str(plan_rows.get(key, {}).get("action", recheck_plan_module.FRESH_ACTION))
        for key in expected
    }
    covered: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_map: dict[tuple[str, str], list[str]] = {}
    extra_reports: list[dict[str, str]] = []
    unexpected_reports: list[dict[str, str]] = []
    for report in reports:
        key = (report["file_path"], report["rule_id"])
        if key not in expected:
            extra_reports.append(
                {
                    "report_path": report["report_path"],
                    "file_path": key[0],
                    "rule_id": key[1],
                }
            )
            continue
        if expected_actions[key] != recheck_plan_module.FRESH_ACTION:
            unexpected_reports.append(
                {
                    "report_path": report["report_path"],
                    "file_path": key[0],
                    "rule_id": key[1],
                    "planned_action": expected_actions[key],
                }
            )
            continue
        if key in covered:
            duplicate_map.setdefault(key, [covered[key]["report_path"]]).append(report["report_path"])
            continue
        covered[key] = report

    locked_coverage = {
        key: row
        for key, row in plan_rows.items()
        if key in expected and row.get("action") == recheck_plan_module.LOCK_ACTION
    }
    settled_coverage = {
        key: row
        for key, row in plan_rows.items()
        if key in expected and row.get("action") == recheck_plan_module.SETTLED_ACTION
    }
    waiting_rows = {
        key: row
        for key, row in plan_rows.items()
        if key in expected and row.get("action") == recheck_plan_module.WAIT_ACTION
    }
    missing_reports = [
        {"file_path": key[0], "rule_id": key[1]}
        for key in expected
        if expected_actions[key] == recheck_plan_module.FRESH_ACTION and key not in covered
    ]
    duplicate_reports = [
        {
            "file_path": key[0],
            "rule_id": key[1],
            "report_paths": paths,
        }
        for key, paths in sorted(duplicate_map.items())
    ]
    status_counts = {status: 0 for status in contract.AUDIT_REPORT_STATUSES}
    status_counts.update(Counter(report["status"] for report in covered.values()))
    for row in [*locked_coverage.values(), *settled_coverage.values()]:
        evaluation = row.get("locked_evaluation") or row.get("settled_evaluation") or {}
        status = str(evaluation.get("status", "")) if isinstance(evaluation, Mapping) else ""
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
        "run_id": provenance["run_id"] if provenance else None,
        "scope_id": provenance["scope_id"] if provenance else None,
        "audit_snapshot_id": provenance["audit_snapshot_id"] if provenance else None,
        "matrix_path": _display_path(ledger.repo_path(matrix_path)),
        "reports_dir": _display_path(ledger.repo_path(reports_dir)),
        "recheck_plan_path": (
            _display_path(ledger.repo_path(recheck_plan_path)) if recheck_plan_path else None
        ),
        "expected_report_count": len(expected),
        "expected_fresh_report_count": sum(
            action == recheck_plan_module.FRESH_ACTION for action in expected_actions.values()
        ),
        "report_file_count": len(reports) + len(invalid_reports),
        "valid_report_count": len(reports),
        "covered_report_count": len(covered),
        "locked_row_count": len(locked_coverage),
        "settled_row_count": len(settled_coverage),
        "waiting_row_count": len(waiting_rows),
        "coverage_count": len(covered) + len(locked_coverage) + len(settled_coverage),
        "missing_count": len(missing_reports),
        "invalid_count": len(invalid_reports),
        "duplicate_count": len(duplicate_reports),
        "extra_count": len(extra_reports),
        "unexpected_count": len(unexpected_reports),
        "lock_validation_error": lock_validation_error,
        "status_counts": status_counts,
        "complete": complete,
        "final_coverage_complete": complete and not waiting_rows,
        "missing_reports": missing_reports,
        "invalid_reports": invalid_reports,
        "duplicate_reports": duplicate_reports,
        "extra_reports": extra_reports,
        "unexpected_reports": unexpected_reports,
        "locked_rows": list(locked_coverage.values()),
        "settled_rows": list(settled_coverage.values()),
        "waiting_rows": list(waiting_rows.values()),
    }
    return {
        "verification": verification,
        "expected": expected,
        "fresh_reports": list(covered.values()),
        "covered_reports": covered,
        "locked_rows": locked_coverage,
        "settled_rows": settled_coverage,
        "waiting_rows": waiting_rows,
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


def _severity_rank(severity: str) -> int:
    return {"must_fix": 3, "should_fix": 2, "optional": 1}.get(severity, 0)


def evaluation_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = {
        "file_path": report["file_path"],
        "rule_id": report["rule_id"],
        "status": report["status"],
        "summary": report["summary"],
        "reason": report["reason"],
    }
    for provenance_field in ("run_id", "scope_id", "audit_snapshot_id"):
        if str(report.get(provenance_field, "")).strip():
            evaluation[provenance_field] = report[provenance_field]
    if "review" in report:
        evaluation["review"] = report["review"]
    findings = report.get("findings", [])
    if findings:
        highest = max(findings, key=lambda finding: _severity_rank(str(finding["severity"])))
        evaluation["severity"] = highest["severity"]
        evaluation["finding_status"] = highest["status"]
        evaluation["recommended_action"] = highest.get("recommended_action", "")
        evaluation["verification"] = highest.get("verification", "")
        evaluation["evidence"] = highest.get("evidence", [])
    for optional in ("notes", "recommended_action", "verification", "evidence"):
        if optional in report and optional not in evaluation:
            evaluation[optional] = report[optional]
    return evaluation


def _compiled_rows(
    *,
    matrix_path: Path | str,
    reports_dir: Path | str,
    recheck_plan_path: Path | str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    snapshot = collect_report_coverage(
        matrix_path=matrix_path,
        reports_dir=reports_dir,
        recheck_plan_path=recheck_plan_path,
    )
    verification = snapshot["verification"]
    expected = snapshot["expected"]
    report_index = snapshot["covered_reports"]
    locked_index = snapshot["locked_rows"]
    settled_index = snapshot["settled_rows"]
    evaluations: list[dict[str, Any]] = []
    for key in expected:
        report = report_index.get(key)
        locked_row = locked_index.get(key)
        settled_row = settled_index.get(key)
        if report:
            evaluation = evaluation_from_report(report)
        elif locked_row and isinstance(locked_row.get("locked_evaluation"), Mapping):
            evaluation = dict(locked_row["locked_evaluation"])
        elif settled_row and isinstance(settled_row.get("settled_evaluation"), Mapping):
            evaluation = dict(settled_row["settled_evaluation"])
        else:
            continue
        if verification.get("audit_snapshot_id"):
            evaluation.update(
                {
                    "compiled_for_run_id": verification["run_id"],
                    "compiled_for_scope_id": verification["scope_id"],
                    "compiled_for_audit_snapshot_id": verification["audit_snapshot_id"],
                }
            )
        evaluations.append(evaluation)
    return verification, evaluations


def compile_reports(
    *,
    matrix_path: Path | str,
    reports_dir: Path | str,
    output_dir: Path | str | None = None,
    recheck_plan_path: Path | str | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    verification, evaluations = _compiled_rows(
        matrix_path=matrix_path,
        reports_dir=reports_dir,
        recheck_plan_path=recheck_plan_path,
    )
    if require_complete and not verification["complete"]:
        return {"compiled": False, "verification": verification}
    target_dir = ledger.repo_path(output_dir) if output_dir else ledger.repo_path(matrix_path).parent
    evaluations_path = contained_child(target_dir, contract.COMPILED_EVALUATIONS_NAME, label="compiled evaluations output")
    summary_path = contained_child(target_dir, contract.REPORTS_SUMMARY_NAME, label="reports summary output")
    artifact_paths = {
        "evaluations_json": _display_path(evaluations_path),
        "summary_json": _display_path(summary_path),
    }
    summary = {
        "schema_version": contract.AUDIT_REPORT_SCHEMA_VERSION if verification["audit_snapshot_id"] else 1,
        "workflow_id": contract.WORKFLOW_ID,
        "run_id": verification["run_id"],
        "scope_id": verification["scope_id"],
        "audit_snapshot_id": verification["audit_snapshot_id"],
        "artifact_role": "report_aggregation_support",
        "audit_authority": False,
        "matrix_path": verification["matrix_path"],
        "reports_dir": verification["reports_dir"],
        "recheck_plan_path": verification["recheck_plan_path"],
        "expected_report_count": verification["expected_report_count"],
        "expected_fresh_report_count": verification["expected_fresh_report_count"],
        "covered_report_count": verification["covered_report_count"],
        "locked_row_count": verification["locked_row_count"],
        "settled_row_count": verification["settled_row_count"],
        "waiting_row_count": verification["waiting_row_count"],
        "coverage_count": verification["coverage_count"],
        "status_counts": verification["status_counts"],
        "complete": verification["complete"],
        "final_coverage_complete": verification["final_coverage_complete"],
        "evaluation_count": len(evaluations),
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
            "schema_version": contract.AUDIT_REPORT_SCHEMA_VERSION if verification["audit_snapshot_id"] else 1,
            "workflow_id": contract.WORKFLOW_ID,
            "run_id": verification["run_id"],
            "scope_id": verification["scope_id"],
            "audit_snapshot_id": verification["audit_snapshot_id"],
            "artifact_role": "row_evaluations_from_reports",
            "audit_authority": False,
            "evaluations": evaluations,
        },
    )
    ledger.write_json(summary_path, summary)
    return {
        "workflow_id": contract.WORKFLOW_ID,
        "artifact_paths": artifact_paths,
        "summary": summary,
    }
