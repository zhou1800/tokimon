"""Rule extraction from guideline documents."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from tokimon.workflow.rule_extraction import rule_hash, rules_from_lines
from tokimon.workflow import audit_project as project
from tokimon.workflow.document_audit import contract


RULE_RE = re.compile(contract.RULE_LINE_PATTERN)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project.root()).as_posix()
    except ValueError:
        return path.as_posix()


def _rules_from_lines(source_path: Path, lines: list[str], *, source_label: str | None = None) -> list[dict[str, Any]]:
    return rules_from_lines(source_label or _repo_relative(source_path), lines, rule_pattern=RULE_RE)


def extract_rules_from_file(path: Path) -> list[dict[str, Any]]:
    source_path = path if path.is_absolute() else project.root() / path
    lines = source_path.read_text(encoding="utf-8").splitlines()
    return _rules_from_lines(source_path, lines)


def _repo_relative_source(source: Path) -> str:
    source_path = source if source.is_absolute() else project.root() / source
    try:
        return source_path.resolve().relative_to(project.root()).as_posix()
    except ValueError as exc:
        raise ValueError("source_ref requires guideline sources under the repository root") from exc


def resolve_source_ref(source_ref: str) -> str:
    """Resolve a possibly moving tree-ish to an immutable commit id."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{source_ref}^{{commit}}"],
            cwd=project.root(),
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or str(exc)
        raise ValueError(f"could not resolve source_ref {source_ref}: {message}") from exc
    return result.stdout.strip()


def _source_bytes(source: Path, *, source_ref: str | None) -> bytes:
    if source_ref is None:
        source_path = source if source.is_absolute() else project.root() / source
        return source_path.read_bytes()
    repo_relative = _repo_relative_source(source)
    try:
        result = subprocess.run(
            ["git", "show", f"{source_ref}:{repo_relative}"],
            cwd=project.root(),
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode(errors="replace").strip() or str(exc)
        raise ValueError(f"could not read {repo_relative} at {source_ref}: {message}") from exc
    return result.stdout


def source_snapshots(
    sources: Iterable[Path] | None = None,
    *,
    source_ref: str | None = None,
) -> list[dict[str, str]]:
    """Hash the exact guideline source bytes used by a matrix."""

    snapshots: list[dict[str, str]] = []
    for source in sources or contract.GUIDELINE_RULE_SOURCES:
        source_path = source if source.is_absolute() else project.root() / source
        label = _repo_relative_source(source) if source_ref else _repo_relative(source_path)
        snapshots.append(
            {
                "path": label,
                "sha256": hashlib.sha256(_source_bytes(source, source_ref=source_ref)).hexdigest(),
            }
        )
    return sorted(snapshots, key=lambda item: item["path"])


def extract_rules_from_git_ref(path: Path, source_ref: str) -> list[dict[str, Any]]:
    repo_relative = _repo_relative_source(path)
    try:
        result = subprocess.run(
            ["git", "show", f"{source_ref}:{repo_relative}"],
            cwd=project.root(),
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or str(exc)
        raise ValueError(f"could not read {repo_relative} at {source_ref}: {message}") from exc
    source_path = project.root() / repo_relative
    return _rules_from_lines(source_path, result.stdout.splitlines(), source_label=repo_relative)


def extract_rules(sources: Iterable[Path] | None = None, *, source_ref: str | None = None) -> list[dict[str, Any]]:
    rule_sources = tuple(sources or contract.GUIDELINE_RULE_SOURCES)
    extracted: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for source in rule_sources:
        source_rules = (
            extract_rules_from_git_ref(source, source_ref)
            if source_ref
            else extract_rules_from_file(source)
        )
        for rule in source_rules:
            if rule["rule_id"] in seen:
                duplicates.add(rule["rule_id"])
            seen.add(rule["rule_id"])
            extracted.append(rule)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate rule ids: {duplicate_list}")
    return extracted


def rules_payload(sources: Iterable[Path] | None = None, *, source_ref: str | None = None) -> dict[str, Any]:
    extracted = extract_rules(sources, source_ref=source_ref)
    return {
        "workflow_id": contract.WORKFLOW_ID,
        "rule_sources": [
            _repo_relative(project.root() / source if not source.is_absolute() else source)
            for source in (sources or contract.GUIDELINE_RULE_SOURCES)
        ],
        "source_ref": source_ref,
        "rule_count": len(extracted),
        "rules": extracted,
    }
