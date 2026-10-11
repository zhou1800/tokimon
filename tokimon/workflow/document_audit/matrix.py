"""Checklist matrix artifacts for document-audit runs."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tokimon.workflow import audit_project as project
from tokimon.workflow.document_audit import audit_units, contract, ledger, rules


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project.root()).as_posix()
    except ValueError:
        return path.as_posix()


def _rule_source_path(source: Path) -> str:
    return _display_path(project.root() / source if not source.is_absolute() else source)


def _read_documents_from(path: Path | str) -> list[str]:
    source = ledger.repo_path(path)
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        documents = payload.get("documents") if isinstance(payload, Mapping) else payload
        if documents is None and isinstance(payload, Mapping):
            documents = payload.get("document_paths")
        if not isinstance(documents, list):
            raise ValueError("documents-from JSON must be a list or contain documents/document_paths")
        return [str(document) for document in documents]
    return [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _expand_document_glob(pattern: str) -> list[str]:
    search_pattern = pattern if Path(pattern).is_absolute() else str(project.root() / pattern)
    matches = sorted(Path(match) for match in glob.glob(search_pattern))
    return [_display_path(path) for path in matches if path.is_file()]


def _document_path(document: str) -> str:
    path = ledger.repo_path(document)
    if not path.is_file():
        raise ValueError(f"document does not exist: {document}")
    return _display_path(path)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=contract.JSON_ENSURE_ASCII,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _snapshot_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "audit_unit_id",
        "file_path",
        "rule_id",
        "rule_source_path",
        "rule_line",
        "rule_hash",
        "rule_description",
    )
    return sorted(
        ({field: row.get(field) for field in fields} for row in rows),
        key=lambda row: (str(row["file_path"]), str(row["rule_id"])),
    )


def _audit_snapshot_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "project": payload.get("project"),
        "schema_version": payload.get("schema_version"),
        "workflow_id": payload.get("workflow_id"),
        "run_id": payload.get("run_id"),
        "scope_id": payload.get("scope_id"),
        "source_ref": payload.get("source_ref"),
        "resolved_source_ref": payload.get("resolved_source_ref"),
        "rule_source_snapshots": payload.get("rule_source_snapshots"),
        "document_snapshots": payload.get("document_snapshots"),
        "bound_input_snapshots": payload.get("bound_input_snapshots"),
        "audit_units": payload.get("audit_units"),
        "rows": _snapshot_rows(payload.get("rows", [])),
    }


def audit_snapshot_id(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(_audit_snapshot_core(payload))).hexdigest()


def validate_audit_snapshot(payload: Mapping[str, Any]) -> None:
    """Reject a matrix whose captured audit inputs are no longer current."""

    project.validate_binding(payload)
    snapshot_id = str(payload.get("audit_snapshot_id", "")).strip()
    if not snapshot_id:
        raise ValueError("matrix input is missing audit_snapshot_id")
    if snapshot_id != audit_snapshot_id(payload):
        raise ValueError("matrix audit_snapshot_id does not match its captured inputs")

    document_paths = payload.get("document_paths")
    units = payload.get("audit_units")
    rule_sources = payload.get("rule_sources")
    if not isinstance(document_paths, list) or not isinstance(units, list):
        raise ValueError("matrix snapshot requires document_paths and audit_units")
    if not isinstance(rule_sources, list):
        raise ValueError("matrix snapshot requires rule_sources")

    current_documents = audit_units.path_identities([str(path) for path in document_paths])
    if current_documents != payload.get("document_snapshots"):
        raise ValueError("matrix document snapshot is stale")
    bound_inputs = list(
        dict.fromkeys(
            str(path)
            for unit in units
            if isinstance(unit, Mapping)
            for path in unit.get("bound_input_paths", [])
        )
    )
    current_bound_inputs = audit_units.path_identities(bound_inputs)
    if current_bound_inputs != payload.get("bound_input_snapshots"):
        raise ValueError("matrix bound-input snapshot is stale")

    source_ref = payload.get("source_ref")
    resolved_source_ref = payload.get("resolved_source_ref")
    if source_ref:
        current_resolved_ref = rules.resolve_source_ref(str(source_ref))
        if current_resolved_ref != resolved_source_ref:
            raise ValueError("matrix source_ref no longer resolves to its captured commit")
    current_rule_sources = rules.source_snapshots(
        [Path(str(path)) for path in rule_sources],
        source_ref=str(resolved_source_ref) if resolved_source_ref else None,
    )
    if current_rule_sources != payload.get("rule_source_snapshots"):
        raise ValueError("matrix guideline source snapshot is stale")


def collect_document_paths(
    *,
    documents: Sequence[str] | None = None,
    document_globs: Sequence[str] | None = None,
    documents_from: str | None = None,
) -> list[str]:
    collected: list[str] = []
    for document in documents or ():
        collected.append(_document_path(document))
    for pattern in document_globs or ():
        collected.extend(_expand_document_glob(pattern))
    if documents_from:
        for document in _read_documents_from(documents_from):
            collected.append(_document_path(document))

    deduped = list(dict.fromkeys(collected))
    if not deduped:
        raise ValueError("matrix requires at least one document")
    return deduped


def load_evaluations(path: Path | str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(ledger.repo_path(path).read_text(encoding="utf-8"))
    evaluations = payload.get("evaluations") if isinstance(payload, Mapping) else payload
    if evaluations is None and isinstance(payload, Mapping):
        evaluations = payload.get("rows")
    if not isinstance(evaluations, list):
        raise ValueError("evaluations input must be a list or contain evaluations/rows")
    normalized = [dict(evaluation) for evaluation in evaluations]
    if isinstance(payload, Mapping) and payload.get("artifact_role") == "row_evaluations_from_reports":
        provenance = {
            "compiled_for_run_id": str(payload.get("run_id", "")).strip(),
            "compiled_for_scope_id": str(payload.get("scope_id", "")).strip(),
            "compiled_for_audit_snapshot_id": str(payload.get("audit_snapshot_id", "")).strip(),
        }
        if not all(provenance.values()):
            raise ValueError(
                "compiled evaluations are missing current snapshot provenance; "
                "regenerate them from a current-schema matrix and reports"
            )
        for evaluation in normalized:
            if any(evaluation.get(field) != value for field, value in provenance.items()):
                raise ValueError("compiled evaluation provenance does not match its artifact")
    return normalized


def _evaluation_key(evaluation: Mapping[str, Any]) -> tuple[str, str]:
    file_path = str(evaluation.get("file_path", "")).strip()
    rule_id = str(evaluation.get("rule_id", "")).strip()
    if not file_path or not rule_id:
        raise ValueError("evaluation rows require file_path and rule_id")
    status = str(evaluation.get("status", "")).strip()
    if status not in contract.CHECKLIST_STATUSES:
        raise ValueError(f"invalid checklist status: {status}")
    return _document_path(file_path), rule_id


def _evaluation_index(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    document_paths: Sequence[str],
    rule_ids: set[str],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    allowed_documents = set(document_paths)
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for evaluation in evaluations:
        key = _evaluation_key(evaluation)
        if key in indexed:
            raise ValueError(f"duplicate evaluation row: {key[0]} {key[1]}")
        if key[0] not in allowed_documents:
            raise ValueError(f"evaluation references unknown document: {key[0]}")
        if key[1] not in rule_ids:
            raise ValueError(f"evaluation references unknown rule_id: {key[1]}")
        indexed[key] = evaluation
    return indexed


def _copy_evaluation_fields(row: dict[str, Any], evaluation: Mapping[str, Any]) -> None:
    for field in (
        "review",
        "summary",
        "severity",
        "finding_status",
        "reason",
        "notes",
        "recommended_action",
        "verification",
        "evidence",
    ):
        if field in evaluation:
            row[field] = evaluation[field]


def _validate_evaluation_provenance(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    scope_id: str,
    snapshot_id: str,
) -> None:
    expected = {
        "compiled_for_run_id": run_id,
        "compiled_for_scope_id": scope_id,
        "compiled_for_audit_snapshot_id": snapshot_id,
    }
    for evaluation in evaluations:
        supplied = {field: evaluation.get(field) for field in expected if field in evaluation}
        if not supplied:
            continue
        if set(supplied) != set(expected):
            raise ValueError("evaluation provenance is incomplete")
        mismatched = [field for field, value in expected.items() if supplied[field] != value]
        if mismatched:
            raise ValueError(
                "evaluation provenance does not match the current matrix snapshot: "
                + ", ".join(mismatched)
            )


def build_matrix_payload(
    *,
    scope_id: str,
    run_id: str,
    document_paths: Sequence[str],
    unit_definitions: Sequence[Mapping[str, Any]] | None = None,
    rule_sources: Sequence[Path] | None = None,
    source_ref: str | None = None,
    evaluations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    document_paths = list(dict.fromkeys(_document_path(path) for path in document_paths))
    selected_rule_sources = tuple(rule_sources or contract.GUIDELINE_RULE_SOURCES)
    resolved_source_ref = rules.resolve_source_ref(source_ref) if source_ref else None
    rule_source_snapshots = rules.source_snapshots(
        selected_rule_sources,
        source_ref=resolved_source_ref,
    )
    extracted_rules = rules.extract_rules(selected_rule_sources, source_ref=resolved_source_ref)
    if rule_source_snapshots != rules.source_snapshots(
        selected_rule_sources,
        source_ref=resolved_source_ref,
    ):
        raise ValueError("guideline sources changed while the matrix snapshot was being created")
    rule_ids = {rule["rule_id"] for rule in extracted_rules}
    normalized_units = (
        audit_units.normalize_units(unit_definitions)
        if unit_definitions is not None
        else audit_units.default_units(document_paths)
    )
    audit_units.validate_document_coverage(normalized_units, document_paths)
    unit_by_document = {
        document_path: unit["audit_unit_id"]
        for unit in normalized_units
        for document_path in unit["document_paths"]
    }
    evaluation_index = _evaluation_index(
        evaluations or (),
        document_paths=document_paths,
        rule_ids=rule_ids,
    )

    rows: list[dict[str, Any]] = []
    for document_path in document_paths:
        for rule in extracted_rules:
            evaluation = evaluation_index.get((document_path, rule["rule_id"]), {})
            row = {
                "audit_unit_id": unit_by_document[document_path],
                "file_path": document_path,
                "rule_id": rule["rule_id"],
                "status": evaluation.get("status", "unchecked"),
                "rule_source_path": rule["source_path"],
                "rule_line": rule["line"],
                "rule_hash": rule["rule_hash"],
                "rule_description": rule["description"],
            }
            _copy_evaluation_fields(row, evaluation)
            rows.append(row)

    status_counts = {status: 0 for status in contract.CHECKLIST_STATUSES}
    status_counts.update(Counter(str(row["status"]) for row in rows))
    bound_input_paths = list(
        dict.fromkeys(
            str(path)
            for unit in normalized_units
            for path in unit.get("bound_input_paths", [])
        )
    )
    payload = {
        "project": project.current().to_dict(),
        "schema_version": contract.CHECKLIST_MATRIX_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "run_id": run_id,
        "scope_id": scope_id,
        "created_at": ledger.utc_now(),
        "source_ref": source_ref,
        "resolved_source_ref": resolved_source_ref,
        "rule_sources": [_rule_source_path(source) for source in selected_rule_sources],
        "rule_source_snapshots": rule_source_snapshots,
        "document_paths": list(document_paths),
        "document_snapshots": audit_units.path_identities(document_paths),
        "bound_input_snapshots": audit_units.path_identities(bound_input_paths),
        "audit_units": normalized_units,
        "audit_unit_count": len(normalized_units),
        "rule_count": len(extracted_rules),
        "document_count": len(document_paths),
        "matrix_count": len(rows),
        "expected_matrix_count": len(extracted_rules) * len(document_paths),
        "status_counts": status_counts,
        "complete": status_counts["unchecked"] == 0,
        "rows": rows,
    }
    payload["audit_snapshot_id"] = audit_snapshot_id(payload)
    validate_audit_snapshot(payload)
    _validate_evaluation_provenance(
        evaluations or (),
        run_id=run_id,
        scope_id=scope_id,
        snapshot_id=payload["audit_snapshot_id"],
    )
    return payload


def validate_matrix_expectations(
    payload: Mapping[str, Any],
    *,
    expect_rule_count: int | None = None,
    expect_document_count: int | None = None,
    expect_matrix_count: int | None = None,
    require_complete: bool = False,
) -> None:
    checks = (
        ("rule_count", expect_rule_count),
        ("document_count", expect_document_count),
        ("matrix_count", expect_matrix_count),
    )
    for field, expected in checks:
        if expected is not None and payload[field] != expected:
            raise ValueError(f"expected {field}={expected}; got {payload[field]}")
    if payload["matrix_count"] != payload["expected_matrix_count"]:
        raise ValueError(
            f"matrix_count does not match rule_count x document_count: "
            f"{payload['matrix_count']} != {payload['expected_matrix_count']}"
        )
    if require_complete and not payload["complete"]:
        unchecked = payload["status_counts"]["unchecked"]
        raise ValueError(f"matrix has {unchecked} unchecked rows")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "audit_unit_id",
        "file_path",
        "rule_id",
        "status",
        "rule_source_path",
        "rule_line",
        "rule_description",
        "summary",
        "severity",
        "finding_status",
        "reason",
        "recommended_action",
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
    run_dir = project.contained_child(ledger.repo_path(run_root), payload["run_id"], label="run_id")
    units_path = project.contained_child(run_dir, contract.AUDIT_UNITS_NAME, label="artifact")
    matrix_path = project.contained_child(run_dir, f"{artifact_prefix}.json", label="artifact")
    if matrix_path.exists():
        prior = json.loads(matrix_path.read_text(encoding="utf-8"))
        if prior.get("project") != payload.get("project"):
            raise ValueError("run artifacts belong to a different audit checkout")
    csv_path = project.contained_child(run_dir, f"{artifact_prefix}.csv", label="artifact")
    summary_name = (
        "checklist-summary.json"
        if artifact_prefix == "checklist-matrix"
        else f"{artifact_prefix}-summary.json"
    )
    summary_path = project.contained_child(run_dir, summary_name, label="artifact")
    artifact_paths = {
        "audit_units_json": ledger.display_path(units_path),
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
        "created_at": payload["created_at"],
        "audit_snapshot_id": payload["audit_snapshot_id"],
        "source_ref": payload["source_ref"],
        "resolved_source_ref": payload["resolved_source_ref"],
        "rule_sources": payload["rule_sources"],
        "audit_unit_count": payload["audit_unit_count"],
        "rule_count": payload["rule_count"],
        "document_count": payload["document_count"],
        "matrix_count": payload["matrix_count"],
        "expected_matrix_count": payload["expected_matrix_count"],
        "status_counts": payload["status_counts"],
        "complete": payload["complete"],
        "artifact_paths": artifact_paths,
    }

    ledger.write_json(
        units_path,
        audit_units.payload(
            payload["audit_units"],
            scope_id=str(payload["scope_id"]),
            run_id=str(payload["run_id"]),
        ),
    )
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
