"""Synthetic current-schema fixtures; these do not represent real audit work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tokimon.workflow.code_audit import inputs


def reviewed_report(
    report: dict[str, Any], report_path: Path, *, source_dir: Path | None = None
) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key != "review"}
    body.setdefault("audit_snapshot_id", "synthetic-validator-only")
    reviewer = "subagent:synthetic-test-fixture"
    source = (
        source_dir or report_path.parent / "reviews"
    ) / f"{report_path.parent.name}-{report_path.stem}.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps({"reviewer_id": reviewer, "reports": [body]}) + "\n")
    return dict(
        body,
        review={
            "reviewer_id": reviewer,
            "source_path": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    )


def write_fixture(path: Path, payload: Any) -> None:
    if (
        isinstance(payload, dict)
        and payload.get("schema_version") == 3
        and payload.get("workflow_id") == "code-audit"
    ):
        if "rows" in payload and "locks" not in payload:
            payload = dict(payload)
            payload.setdefault("rule_sources", [])
            payload.setdefault(
                "audit_subjects",
                [
                    {
                        "audit_subject_id": subject,
                        "title": subject,
                        "notes": "Synthetic coverage fixture",
                    }
                    for subject in sorted(
                        {row["audit_subject_id"] for row in payload["rows"]}
                    )
                ],
            )
            payload = inputs.bind_snapshot(payload)
        elif "status" in payload and "audit_subject_id" in payload:
            payload = dict(payload)
            matrices = sorted(path.parent.parent.glob("*matrix*.json"))
            if matrices:
                matrix = json.loads(matrices[0].read_text())
                payload["audit_snapshot_id"] = matrix["audit_snapshot_id"]
            payload = reviewed_report(
                payload, path, source_dir=path.parent.parent / "reviews"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
