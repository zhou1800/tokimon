"""Synthetic reviewer results for document-audit contract tests."""

import hashlib
import json
from pathlib import Path


def reviewed_report(payload: dict, directory: Path, *, reviewer_id: str = "subagent:fixture") -> dict:
    content = json.dumps({"reviewer_id": reviewer_id, "reports": [payload]}, sort_keys=True).encode()
    digest = hashlib.sha256(content).hexdigest()
    source = directory / f"{digest}.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    return dict(payload, review={"reviewer_id": reviewer_id, "source_path": str(source), "source_sha256": digest})
