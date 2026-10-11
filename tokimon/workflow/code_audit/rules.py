"""Rule extraction from code and unit-test guideline documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from tokimon.workflow import audit_project as project
from tokimon.workflow.code_audit import contract
from tokimon.workflow.rule_extraction import rule_hash, rules_from_lines


RULE_RE = re.compile(contract.RULE_LINE_PATTERN)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project.root()).as_posix()
    except ValueError:
        return path.as_posix()


def _rules_from_lines(source_path: Path, lines: list[str]) -> list[dict[str, Any]]:
    return rules_from_lines(_repo_relative(source_path), lines, rule_pattern=RULE_RE)


def extract_rules_from_file(path: Path) -> list[dict[str, Any]]:
    source_path = path if path.is_absolute() else project.root() / path
    lines = source_path.read_text(encoding="utf-8").splitlines()
    return _rules_from_lines(source_path, lines)


def extract_rules(sources: Iterable[Path] | None = None) -> list[dict[str, Any]]:
    rule_sources = tuple(sources or contract.GUIDELINE_RULE_SOURCES)
    extracted: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for source in rule_sources:
        for rule in extract_rules_from_file(source):
            if rule["rule_id"] in seen:
                duplicates.add(rule["rule_id"])
            seen.add(rule["rule_id"])
            extracted.append(rule)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate rule ids: {duplicate_list}")
    return extracted


def rules_payload(sources: Iterable[Path] | None = None) -> dict[str, Any]:
    rule_sources = tuple(sources or contract.GUIDELINE_RULE_SOURCES)
    extracted = extract_rules(rule_sources)
    return {
        "workflow_id": contract.WORKFLOW_ID,
        "rule_sources": [
            _repo_relative(project.root() / source if not source.is_absolute() else source)
            for source in rule_sources
        ],
        "rule_count": len(extracted),
        "rules": extracted,
    }
