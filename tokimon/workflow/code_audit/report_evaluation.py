"""Shared conversion from validated code-audit reports to evaluations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def evaluation_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the provenance-preserving evaluation represented by a report."""

    evaluation = {
        "run_id": report["run_id"],
        "scope_id": report["scope_id"],
        "audit_subject_id": report["audit_subject_id"],
        "rule_id": report["rule_id"],
        "status": report["status"],
        "summary": report["summary"],
        "reason": report["reason"],
        "evidence": list(report.get("evidence", [])),
        "verification": list(report.get("verification", [])),
        "implementation_paths_examined": list(report["implementation_paths_examined"]),
        "test_paths_examined": list(report["test_paths_examined"]),
        "owning_specs_examined": list(report["owning_specs_examined"]),
    }
    for optional in ("review_prompt", "normalized_subject"):
        if str(report.get(optional, "")).strip():
            evaluation[optional] = str(report[optional]).strip()
    for field in ("audit_snapshot_id", "review"):
        if field in report:
            evaluation[field] = report[field]
    return evaluation
