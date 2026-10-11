"""Checklist matrix artifacts for code-audit runs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tokimon.workflow import audit_project as project
from tokimon.workflow.code_audit import contract, inputs, ledger, rules, subjects


def _rule_source_path(source: Path) -> str:
    return subjects.display_path(project.root() / source if not source.is_absolute() else source)


def _evaluation_key(evaluation: Mapping[str, Any]) -> tuple[str, str]:
    subject_id = str(evaluation.get("audit_subject_id", "")).strip()
    rule_id = str(evaluation.get("rule_id", "")).strip()
    if not subject_id or not rule_id:
        raise ValueError("evaluation rows require audit_subject_id and rule_id")
    return subject_id, rule_id


def _evaluation_index(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    audit_subject_ids: set[str],
    rule_ids: set[str],
    scope_id: str,
    run_id: str,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for evaluation in evaluations:
        key = _evaluation_key(evaluation)
        if key in indexed:
            raise ValueError(f"duplicate evaluation row: {key[0]} {key[1]}")
        if key[0] not in audit_subject_ids:
            raise ValueError(f"evaluation references unknown audit_subject_id: {key[0]}")
        if key[1] not in rule_ids:
            raise ValueError(f"evaluation references unknown rule_id: {key[1]}")
        ledger.validate_evaluation(
            evaluation,
            known_rule_ids=rule_ids,
            known_subject_ids=audit_subject_ids,
            expected_scope_id=scope_id,
            expected_run_id=run_id,
        )
        indexed[key] = evaluation
    return indexed


def _copy_evaluation_fields(row: dict[str, Any], evaluation: Mapping[str, Any]) -> None:
    for field in ("summary", "reason", "evidence", "verification", "review_prompt", "review"):
        if field in evaluation:
            row[field] = evaluation[field]


def build_matrix_payload(
    *,
    scope_id: str,
    run_id: str,
    audit_subjects: Sequence[Mapping[str, Any]],
    rule_sources: Sequence[Path] | None = None,
    evaluations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    extracted_rules = rules.extract_rules(rule_sources)
    rule_ids = {rule["rule_id"] for rule in extracted_rules}
    audit_subject_ids = {subject["audit_subject_id"] for subject in audit_subjects}
    evaluation_index = _evaluation_index(
        evaluations or (),
        audit_subject_ids=audit_subject_ids,
        rule_ids=rule_ids,
        scope_id=scope_id,
        run_id=run_id,
    )

    rows: list[dict[str, Any]] = []
    for audit_subject in audit_subjects:
        subject_id = audit_subject["audit_subject_id"]
        for rule in extracted_rules:
            evaluation = evaluation_index.get((subject_id, rule["rule_id"]), {})
            row = {
                "run_id": run_id,
                "scope_id": scope_id,
                "audit_subject_id": subject_id,
                "rule_id": rule["rule_id"],
                "status": evaluation.get("status", "unchecked"),
                "rule_source_path": rule["source_path"],
                "rule_line": rule["line"],
                "rule_hash": rule["rule_hash"],
                "rule_description": rule["description"],
            }
            _copy_evaluation_fields(row, evaluation)
            rows.append(row)

    status_counts = {status: 0 for status in contract.AUDIT_STATUSES}
    status_counts.update(Counter(str(row["status"]) for row in rows))
    payload = inputs.bind_snapshot({
        "schema_version": contract.CHECKLIST_MATRIX_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "run_id": run_id,
        "scope_id": scope_id,
        "created_at": ledger.utc_now(),
        "rule_sources": [_rule_source_path(source) for source in (rule_sources or contract.GUIDELINE_RULE_SOURCES)],
        "audit_subjects": list(audit_subjects),
        "rule_count": len(extracted_rules),
        "audit_subject_count": len(audit_subjects),
        "matrix_count": len(rows),
        "expected_matrix_count": len(extracted_rules) * len(audit_subjects),
        "status_counts": status_counts,
        "complete": status_counts["unchecked"] == 0,
        "rows": rows,
    })
    for evaluation in evaluations or ():
        snapshot_id = evaluation.get("compiled_for_audit_snapshot_id", evaluation.get("audit_snapshot_id"))
        if snapshot_id != payload["audit_snapshot_id"]:
            raise ValueError("evaluation does not match the current matrix snapshot")
    return payload


def validate_matrix_expectations(
    payload: Mapping[str, Any],
    *,
    expect_rule_count: int | None = None,
    expect_subject_count: int | None = None,
    expect_matrix_count: int | None = None,
    require_complete: bool = False,
) -> None:
    checks = (
        ("rule_count", expect_rule_count),
        ("audit_subject_count", expect_subject_count),
        ("matrix_count", expect_matrix_count),
    )
    for field, expected in checks:
        if expected is not None and payload[field] != expected:
            raise ValueError(f"expected {field}={expected}; got {payload[field]}")
    if payload["matrix_count"] != payload["expected_matrix_count"]:
        raise ValueError(
            f"matrix_count does not match rule_count x audit_subject_count: "
            f"{payload['matrix_count']} != {payload['expected_matrix_count']}"
        )
    if require_complete and not payload["complete"]:
        unchecked = payload["status_counts"]["unchecked"]
        raise ValueError(f"matrix has {unchecked} unchecked rows")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "audit_subject_id",
        "rule_id",
        "status",
        "rule_source_path",
        "rule_line",
        "rule_description",
        "summary",
        "reason",
        "review_prompt",
        "verification",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_artifacts(
    payload: Mapping[str, Any],
    *,
    run_root: Path | str,
    artifact_prefix: str,
) -> dict[str, Any]:
    project.validate_component(artifact_prefix, label="artifact-prefix")
    inputs.validate_snapshot(payload)
    run_dir = project.contained_child(ledger.repo_path(run_root), payload["run_id"], label="run_id")
    subjects_path = project.contained_child(run_dir, "audit-subjects.json", label="artifact")
    matrix_path = project.contained_child(run_dir, f"{artifact_prefix}.json", label="artifact")
    if matrix_path.exists():
        previous = json.loads(matrix_path.read_text(encoding="utf-8"))
        if previous.get("project") != payload["project"] or previous.get("audit_snapshot_id") != payload["audit_snapshot_id"]:
            raise ValueError("existing matrix belongs to another project or input snapshot; use a new run")
    csv_path = project.contained_child(run_dir, f"{artifact_prefix}.csv", label="artifact")
    summary_name = (
        "checklist-summary.json"
        if artifact_prefix == "checklist-matrix"
        else f"{artifact_prefix}-summary.json"
    )
    summary_path = project.contained_child(run_dir, summary_name, label="artifact")
    artifact_paths = {
        "audit_subjects_json": ledger.display_path(subjects_path),
        "matrix_json": ledger.display_path(matrix_path),
        "matrix_csv": ledger.display_path(csv_path),
        "summary_json": ledger.display_path(summary_path),
    }
    matrix_payload = dict(payload, artifact_paths=artifact_paths)
    summary_payload = {
        "schema_version": contract.CHECKLIST_MATRIX_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "run_id": payload["run_id"],
        "scope_id": payload["scope_id"],
        "project": payload["project"],
        "audit_snapshot_id": payload["audit_snapshot_id"],
        "created_at": payload["created_at"],
        "rule_sources": payload["rule_sources"],
        "rule_count": payload["rule_count"],
        "audit_subject_count": payload["audit_subject_count"],
        "matrix_count": payload["matrix_count"],
        "expected_matrix_count": payload["expected_matrix_count"],
        "status_counts": payload["status_counts"],
        "complete": payload["complete"],
        "artifact_paths": artifact_paths,
    }
    subjects_payload = {
        "schema_version": contract.SUBJECTS_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "run_id": payload["run_id"],
        "scope_id": payload["scope_id"],
        "subject_count": payload["audit_subject_count"],
        "subjects": payload["audit_subjects"],
    }

    ledger.write_json(subjects_path, subjects_payload)
    ledger.write_json(matrix_path, matrix_payload)
    _write_csv(csv_path, payload["rows"])
    ledger.write_json(summary_path, summary_payload)
    return {
        "workflow_id": contract.WORKFLOW_ID,
        "run_id": payload["run_id"],
        "scope_id": payload["scope_id"],
        "artifact_paths": artifact_paths,
        "summary": summary_payload,
    }
