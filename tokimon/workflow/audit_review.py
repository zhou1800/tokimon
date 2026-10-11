"""Validate row results against preserved output from their assigned reviewer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def validate_review(
    report: Mapping[str, Any], *, report_path: Path, key_fields: Sequence[str]
) -> dict[str, str]:
    review = report.get("review")
    if not isinstance(review, Mapping):
        raise ValueError(f"{report_path} missing required review field")
    for field in ("reviewer_id", "source_path", "source_sha256"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError(f"{report_path} missing required review.{field}")
    reviewer_id = review["reviewer_id"].strip()
    if (
        not reviewer_id.startswith("subagent:")
        or not reviewer_id.removeprefix("subagent:").strip()
    ):
        raise ValueError("reviewer_id must identify the assigned subagent")
    source = Path(review["source_path"])
    if not source.is_absolute():
        source = report_path.parent / source
    source = source.resolve()
    content = source.read_bytes()
    digest = review["source_sha256"].strip()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("review source hash does not match preserved result")
    payload = json.loads(content)
    if not isinstance(payload, Mapping) or payload.get("reviewer_id") != reviewer_id:
        raise ValueError("review source reviewer does not match report")
    rows = payload.get("reports")
    if not isinstance(rows, list):
        raise ValueError("review source requires a reports list")
    matched = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and all(row.get(field) == report.get(field) for field in key_fields)
    ]
    body = {key: value for key, value in report.items() if key != "review"}
    if len(matched) != 1 or matched[0] != body:
        raise ValueError("review source must contain the exact unique row result")
    return {
        "reviewer_id": reviewer_id,
        "source_path": str(source),
        "source_sha256": digest,
    }
