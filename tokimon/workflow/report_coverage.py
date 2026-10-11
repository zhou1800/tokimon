"""Shared structured report coverage helpers for audit workflows."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


RepoPath = Callable[[Path | str], Path]
DisplayPath = Callable[[Path], str]
ReportValidator = Callable[[Mapping[str, Any], Path], dict[str, Any]]


def load_json(path: Path | str, *, repo_path: RepoPath) -> Any:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def matrix_rows(matrix_path: Path | str, *, repo_path: RepoPath) -> list[dict[str, Any]]:
    payload = load_json(matrix_path, repo_path=repo_path)
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("matrix input must contain rows")
    return [dict(row) for row in rows]


def expected_index(
    matrix_path: Path | str,
    *,
    key_fields: Sequence[str],
    repo_path: RepoPath,
) -> dict[tuple[str, ...], dict[str, Any]]:
    expected: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in matrix_rows(matrix_path, repo_path=repo_path):
        key = tuple(str(row.get(field, "")).strip() for field in key_fields)
        missing = [field for field, value in zip(key_fields, key, strict=True) if not value]
        if missing:
            raise ValueError(f"matrix rows require {', '.join(key_fields)}")
        if key in expected:
            label = " ".join(key)
            raise ValueError(f"duplicate matrix row: {label}")
        expected[key] = row
    if not expected:
        raise ValueError("matrix contains no rows")
    return expected


def report_paths(reports_dir: Path | str, *, repo_path: RepoPath) -> list[Path]:
    root = repo_path(reports_dir)
    if not root.exists():
        raise ValueError(f"reports directory does not exist: {reports_dir}")
    if not root.is_dir():
        raise ValueError(f"reports path is not a directory: {reports_dir}")
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def read_reports(
    reports_dir: Path | str,
    *,
    repo_path: RepoPath,
    display_path: DisplayPath,
    validate_report: ReportValidator,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for report_path in report_paths(reports_dir, repo_path=repo_path):
        try:
            payload = load_json(report_path, repo_path=repo_path)
            if not isinstance(payload, Mapping):
                raise ValueError("report must be a JSON object")
            valid.append(validate_report(payload, report_path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid.append({"report_path": display_path(report_path), "error": str(exc)})
    return valid, invalid
