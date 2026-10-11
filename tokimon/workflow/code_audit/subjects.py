"""Audit subject validation and normalization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tokimon.workflow import audit_project as project
from tokimon.workflow.code_audit import contract


def repo_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project.root() / candidate


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project.root()).as_posix()
    except ValueError:
        return path.as_posix()


def _path_string(value: Any, *, field: str, subject_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"subject {subject_id} has invalid {field} item")
    path = repo_path(value.strip())
    if not path.exists():
        raise ValueError(f"subject {subject_id} {field} path does not exist: {value}")
    return display_path(path)


def _path_list(subject: Mapping[str, Any], field: str, subject_id: str) -> list[str]:
    value = subject.get(field)
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"subject {subject_id} {field} must be a list")
    return [_path_string(item, field=field, subject_id=subject_id) for item in value]


def _string_list(subject: Mapping[str, Any], field: str, subject_id: str) -> list[str]:
    value = subject.get(field)
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"subject {subject_id} {field} must be a list")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"subject {subject_id} {field} must contain only non-empty strings")
        normalized.append(item.strip())
    return normalized


def _normalize_dependency_edges(subject: Mapping[str, Any], subject_id: str) -> list[dict[str, str]]:
    value = subject.get("dependency_edges", [])
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"subject {subject_id} dependency_edges must be a list")
    edges: list[dict[str, str]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"subject {subject_id} dependency edge {index} must be an object")
        kind = str(item.get("kind", "")).strip()
        target = str(item.get("target", "")).strip()
        if not kind or not target:
            raise ValueError(f"subject {subject_id} dependency edge {index} requires kind and target")
        edge = {"kind": kind, "target": target}
        if str(item.get("source", "")).strip():
            edge["source"] = str(item["source"]).strip()
        edges.append(edge)
    return edges


def normalize_subject(subject: Mapping[str, Any]) -> dict[str, Any]:
    subject_id = str(subject.get("audit_subject_id", "")).strip()
    if not subject_id:
        raise ValueError("subject missing required field: audit_subject_id")
    title = str(subject.get("title", "")).strip()
    if not title:
        raise ValueError(f"subject {subject_id} missing required field: title")
    normalized = {
        "audit_subject_id": subject_id,
        "title": title,
        "implementation_paths": _path_list(subject, "implementation_paths", subject_id),
        "test_paths": _path_list(subject, "test_paths", subject_id),
        "owning_specs": _path_list(subject, "owning_specs", subject_id),
        "verification_commands": _string_list(subject, "verification_commands", subject_id),
        "dependency_edges": _normalize_dependency_edges(subject, subject_id),
    }
    if str(subject.get("notes", "")).strip():
        normalized["notes"] = str(subject["notes"]).strip()
    if str(subject.get("subject_source", "")).strip():
        normalized["subject_source"] = str(subject["subject_source"]).strip()
    has_boundary = any(
        normalized.get(field)
        for field in ("implementation_paths", "test_paths", "owning_specs", "verification_commands", "dependency_edges")
    ) or bool(normalized.get("notes"))
    if not has_boundary:
        raise ValueError(
            f"subject {subject_id} needs at least one boundary field: "
            f"{', '.join(contract.SUBJECT_BOUNDARY_FIELDS)}"
        )
    return normalized


def load_subjects(path: Path | str, *, selected_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    payload = json.loads(repo_path(path).read_text(encoding="utf-8"))
    raw_subjects = payload.get("subjects") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_subjects, list):
        raise ValueError("subjects input must be a list or contain a subjects list")
    subjects = [normalize_subject(subject) for subject in raw_subjects]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for subject in subjects:
        subject_id = subject["audit_subject_id"]
        if subject_id in seen:
            duplicates.add(subject_id)
        seen.add(subject_id)
    if duplicates:
        raise ValueError(f"duplicate audit_subject_id values: {', '.join(sorted(duplicates))}")
    if selected_ids:
        selected = set(selected_ids)
        unknown = selected - seen
        if unknown:
            raise ValueError(f"unknown audit_subject_id values: {', '.join(sorted(unknown))}")
        subjects = [subject for subject in subjects if subject["audit_subject_id"] in selected]
    if not subjects:
        raise ValueError("subjects input must resolve at least one audit subject")
    return subjects


def subjects_payload(path: Path | str, *, selected_ids: Sequence[str] | None = None) -> dict[str, Any]:
    loaded = load_subjects(path, selected_ids=selected_ids)
    return {
        "schema_version": contract.SUBJECTS_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "subjects_source": display_path(repo_path(path)),
        "subject_count": len(loaded),
        "subjects": loaded,
    }
