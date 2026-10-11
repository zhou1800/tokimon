"""Frozen code-audit input identities shared by matrices and recheck locks."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tokimon.workflow import audit_project as project
from tokimon.workflow.code_audit import contract, ledger


def _display_path(path: Path) -> str:
    return ledger.display_path(path)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=contract.JSON_ENSURE_ASCII,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(project.root()).as_posix()
    except ValueError:
        return path.as_posix()


def _is_repository_path(path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(project.root().resolve())
        return (project.root() / ".git").exists()
    except ValueError:
        return False


def _git_worktree_files(path: Path) -> list[Path]:
    relative = _repo_relative(path)
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(project.root()),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                relative,
            ],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        raise ValueError(
            f"cannot enumerate Git worktree lock inputs for {relative}: {exc}"
        ) from exc
    if result.returncode != 0:
        error = result.stderr.strip() or f"git exited {result.returncode}"
        raise ValueError(
            f"cannot enumerate Git worktree lock inputs for {relative}: {error}"
        )
    files = [
        project.root() / line for line in result.stdout.split("\0") if line.strip()
    ]
    if not files:
        raise ValueError(
            f"repository lock input directory has no Git worktree files: {relative}"
        )
    return files


def _fallback_tree_files(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and candidate.suffix != ".pyc"
    )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    """Hash behavior-relevant file identity, not bytes alone."""

    file_stat = path.lstat()
    mode = stat.S_IMODE(file_stat.st_mode)
    if stat.S_ISLNK(file_stat.st_mode):
        target = os.readlink(path)
        if not path.is_file():
            raise ValueError(
                f"lock input symlink does not resolve to a regular file: {_display_path(path)}"
            )
        file_type = "symlink"
    elif stat.S_ISREG(file_stat.st_mode):
        target = None
        file_type = "regular_file"
    else:
        raise ValueError(f"unsupported lock input file type: {_display_path(path)}")
    identity: dict[str, Any] = {
        "path": _display_path(path),
        "file_type": file_type,
        "mode": format(mode, "04o"),
        "sha256": file_hash(path),
    }
    if target is not None:
        identity["symlink_target"] = target
    return identity


def path_hash(path_value: str) -> dict[str, Any]:
    path = ledger.repo_path(path_value)
    if path.is_file() or path.is_symlink():
        identity = _file_identity(path)
        return {
            **identity,
            "kind": "file",
            "hash_basis": "file_identity_v2",
        }
    if path.is_dir():
        repository_path = _is_repository_path(path)
        files = (
            _git_worktree_files(path) if repository_path else _fallback_tree_files(path)
        )
        entries = [
            _file_identity(file_path)
            for file_path in sorted(files, key=lambda item: _display_path(item))
            if file_path.is_file() or file_path.is_symlink()
        ]
        return {
            "path": _display_path(path),
            "kind": "directory",
            "mode": format(stat.S_IMODE(path.lstat().st_mode), "04o"),
            "hash_basis": (
                "git_worktree_files_v2" if repository_path else "filesystem_tree_v2"
            ),
            "file_count": len(entries),
            "sha256": canonical_hash(entries),
        }
    raise ValueError(f"lock input path does not exist: {path_value}")


def _input_identity(value: str) -> dict[str, Any]:
    try:
        return path_hash(value)
    except (OSError, ValueError) as exc:
        return {"path": value, "error": str(exc)}


def _path_hashes(subject: Mapping[str, Any], field: str) -> list[dict[str, Any]]:
    values = subject.get(field, [])
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"subject {field} must be a list")
    return [_input_identity(str(value)) for value in values]


def _guideline_hashes(matrix_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = matrix_payload.get("rule_sources", [])
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise ValueError("matrix rule_sources must be a list")
    return [_input_identity(str(source)) for source in sources]


def _verification_commands(subject: Mapping[str, Any]) -> list[str]:
    values = subject.get("verification_commands", [])
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("subject verification_commands must be a list")
    return [str(value).strip() for value in values if str(value).strip()]


def _subject_signature(
    *,
    subject: Mapping[str, Any],
    guideline_path_hashes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    commands = _verification_commands(subject)
    return {
        "subject_definition_hash": canonical_hash(subject),
        "implementation_path_hashes": _path_hashes(subject, "implementation_paths"),
        "test_path_hashes": _path_hashes(subject, "test_paths"),
        "owning_spec_path_hashes": _path_hashes(subject, "owning_specs"),
        "guideline_path_hashes": list(guideline_path_hashes),
        "verification_commands": commands,
        "verification_command_hash": canonical_hash(commands),
    }


def subject_signature_results(
    *,
    matrix_payload: Mapping[str, Any],
    subjects_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    signatures: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    try:
        guideline_path_hashes = _guideline_hashes(matrix_payload)
    except (OSError, ValueError) as exc:
        return {}, {subject_id: str(exc) for subject_id in subjects_by_id}
    for subject_id, subject in subjects_by_id.items():
        try:
            signatures[subject_id] = _subject_signature(
                subject=subject,
                guideline_path_hashes=guideline_path_hashes,
            )
            invalid = [
                item["error"]
                for field, value in signatures[subject_id].items()
                if field.endswith("path_hashes")
                for item in value
                if "error" in item
            ]
            if invalid:
                errors[subject_id] = "; ".join(invalid)
        except (OSError, ValueError) as exc:
            errors[subject_id] = str(exc)
    return signatures, errors


def capture(payload: Mapping[str, Any]) -> dict[str, Any]:
    subjects = payload.get("audit_subjects")
    if not isinstance(subjects, list):
        raise ValueError("matrix audit_subjects must be a list")
    indexed = {}
    for subject in subjects:
        if not isinstance(subject, Mapping) or not subject.get("audit_subject_id"):
            raise ValueError("matrix subjects require audit_subject_id")
        subject_id = str(subject["audit_subject_id"])
        if subject_id in indexed:
            raise ValueError(f"duplicate audit_subject_id: {subject_id}")
        indexed[subject_id] = subject
    with project.selected(project.current()):
        signatures, errors = subject_signature_results(
            matrix_payload=payload,
            subjects_by_id=indexed,
        )
    return {
        subject_id: {
            **(
                {"signature": signatures[subject_id]}
                if subject_id in signatures
                else {}
            ),
            **({"error": errors[subject_id]} if subject_id in errors else {}),
        }
        for subject_id in sorted(indexed)
    }


def _snapshot_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    row_fields = (
        "audit_subject_id",
        "rule_id",
        "rule_hash",
        "rule_source_path",
        "rule_line",
        "rule_description",
    )
    return {
        **{
            field: payload.get(field)
            for field in (
                "project",
                "run_id",
                "scope_id",
                "rule_sources",
                "audit_subjects",
                "input_snapshots",
            )
        },
        "rows": [
            {field: row.get(field) for field in row_fields} for row in payload["rows"]
        ],
    }


def bind_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "project" in payload:
        project.validate_binding(payload)
    bound = dict(payload, project=project.current().to_dict())
    bound["input_snapshots"] = capture(bound)
    bound["audit_snapshot_id"] = canonical_hash(_snapshot_core(bound))
    bound["rows"] = [
        dict(row, audit_snapshot_id=bound["audit_snapshot_id"]) for row in bound["rows"]
    ]
    return bound


def validate_snapshot(
    payload: Mapping[str, Any], *, check_current: bool = True
) -> None:
    if payload.get("schema_version") != contract.CHECKLIST_MATRIX_SCHEMA_VERSION:
        raise ValueError("current input snapshots require a current-schema matrix")
    project.validate_binding(payload)
    recorded = payload.get("audit_snapshot_id")
    if not recorded or recorded != canonical_hash(_snapshot_core(payload)):
        raise ValueError("audit input snapshot identity is invalid")
    if any(row.get("audit_snapshot_id") != recorded for row in payload["rows"]):
        raise ValueError("matrix row audit_snapshot_id does not match matrix")
    if check_current and capture(payload) != payload.get("input_snapshots"):
        raise ValueError(
            "audit input snapshot is stale; rebuild the matrix before reviewing changed inputs"
        )
