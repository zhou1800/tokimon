"""Audit-unit validation for document-audit matrices and clean-unit locks."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tokimon.workflow import audit_project as project
from tokimon.workflow.document_audit import contract


def repo_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project.root() / candidate


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project.root()).as_posix()
    except ValueError:
        return path.as_posix()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_identity(path_value: Path | str) -> dict[str, str]:
    """Return the content identity used by matrix snapshots and unit locks."""

    path = repo_path(path_value)
    if not path.is_file():
        raise ValueError(f"audit input path is not a file: {path_value}")
    identity = {
        "path": display_path(path),
        "sha256": _file_hash(path),
    }
    if path.is_symlink():
        identity["symlink_target"] = os.readlink(path)
    return identity


def path_identities(paths: Sequence[Path | str]) -> list[dict[str, str]]:
    return sorted((path_identity(path) for path in paths), key=lambda item: item["path"])


def _path_list(unit: Mapping[str, Any], field: str, unit_id: str, *, required: bool) -> list[str]:
    value = unit.get(field)
    if value is None:
        if required:
            raise ValueError(f"audit unit {unit_id} requires {field}")
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"audit unit {unit_id} {field} must be a list")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"audit unit {unit_id} {field} must contain only non-empty paths")
        path = repo_path(item.strip())
        if not path.is_file():
            raise ValueError(f"audit unit {unit_id} {field} path is not a file: {item}")
        normalized.append(display_path(path))
    deduped = list(dict.fromkeys(normalized))
    if required and not deduped:
        raise ValueError(f"audit unit {unit_id} requires at least one {field} item")
    return deduped


def normalize_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    unit_id = str(unit.get("audit_unit_id", "")).strip()
    if not unit_id:
        raise ValueError("audit unit missing required field: audit_unit_id")
    normalized = {
        "audit_unit_id": unit_id,
        "document_paths": _path_list(unit, "document_paths", unit_id, required=True),
        "bound_input_paths": _path_list(unit, "bound_input_paths", unit_id, required=False),
    }
    title = str(unit.get("title", "")).strip()
    if title:
        normalized["title"] = title
    return normalized


def normalize_units(raw_units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if any(not isinstance(unit, Mapping) for unit in raw_units):
        raise ValueError("audit-units entries must be objects")
    units = [normalize_unit(unit) for unit in raw_units]
    if not units:
        raise ValueError("audit units must contain at least one unit")
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    seen_documents: set[str] = set()
    duplicate_documents: set[str] = set()
    for unit in units:
        unit_id = unit["audit_unit_id"]
        if unit_id in seen_ids:
            duplicate_ids.add(unit_id)
        seen_ids.add(unit_id)
        for document_path in unit["document_paths"]:
            if document_path in seen_documents:
                duplicate_documents.add(document_path)
            seen_documents.add(document_path)
    if duplicate_ids:
        raise ValueError(f"duplicate audit_unit_id values: {', '.join(sorted(duplicate_ids))}")
    if duplicate_documents:
        raise ValueError(
            "documents may belong to only one audit unit: " + ", ".join(sorted(duplicate_documents))
        )
    return units


def load_units(path: Path | str) -> list[dict[str, Any]]:
    payload = json.loads(repo_path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        schema_version = payload.get("schema_version", contract.AUDIT_UNITS_SCHEMA_VERSION)
        if schema_version != contract.AUDIT_UNITS_SCHEMA_VERSION:
            raise ValueError(f"unsupported audit-units schema_version: {schema_version}")
        workflow_id = payload.get("workflow_id", contract.WORKFLOW_ID)
        if workflow_id != contract.WORKFLOW_ID:
            raise ValueError(f"unexpected audit-units workflow_id: {workflow_id}")
    raw_units = payload.get("units") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_units, list):
        raise ValueError("audit-units input must be a list or contain a units list")
    if any(not isinstance(unit, Mapping) for unit in raw_units):
        raise ValueError("audit-units entries must be objects")
    return normalize_units(raw_units)


def default_units(document_paths: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "audit_unit_id": document_path,
            "document_paths": [document_path],
            "bound_input_paths": [],
        }
        for document_path in document_paths
    ]


def validate_document_coverage(
    units: Sequence[Mapping[str, Any]],
    document_paths: Sequence[str],
) -> None:
    unit_documents = {
        str(document_path)
        for unit in units
        for document_path in unit.get("document_paths", [])
    }
    matrix_documents = set(document_paths)
    missing = sorted(matrix_documents - unit_documents)
    extra = sorted(unit_documents - matrix_documents)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing from audit units: " + ", ".join(missing))
        if extra:
            details.append("outside matrix documents: " + ", ".join(extra))
        raise ValueError("audit-unit document coverage mismatch: " + "; ".join(details))


def payload(units: Sequence[Mapping[str, Any]], *, scope_id: str, run_id: str) -> dict[str, Any]:
    return {
        "schema_version": contract.AUDIT_UNITS_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "run_id": run_id,
        "scope_id": scope_id,
        "unit_count": len(units),
        "units": list(units),
    }
