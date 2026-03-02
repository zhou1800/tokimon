"""Append-only audit log for config-like writes.

Phase 2 (OpenClaw-inspired) focuses on skill asset writes (manifest, modules, prompts).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_MAX_REASON_CHARS = 500


@dataclass(frozen=True)
class ConfigAuditEntry:
    ts: str
    path: str
    action: str
    sha256_before: str | None
    sha256_after: str | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ts": self.ts,
            "path": self.path,
            "action": self.action,
            "sha256_before": self.sha256_before,
            "sha256_after": self.sha256_after,
            "reason": self.reason,
        }


def audit_path(repo_root: Path) -> Path:
    return repo_root / ".tokimon-tmp" / "audit" / "config.jsonl"


def append_entry(repo_root: Path, entry: ConfigAuditEntry) -> None:
    path = audit_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_dict(), sort_keys=True))
        handle.write("\n")


def write_text_with_audit(repo_root: Path, path: Path, content: str, *, reason: str) -> None:
    before_bytes: bytes | None = None
    if path.exists():
        try:
            before_bytes = path.read_bytes()
        except Exception:
            before_bytes = None
    sha_before = _sha256(before_bytes) if before_bytes is not None else None
    encoded = content.encode("utf-8")
    sha_after = _sha256(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    entry = ConfigAuditEntry(
        ts=datetime.now(timezone.utc).isoformat(),
        path=_relpath(repo_root, path),
        action="write",
        sha256_before=sha_before,
        sha256_after=sha_after,
        reason=_bound_reason(reason),
    )
    append_entry(repo_root, entry)


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _bound_reason(reason: str) -> str:
    reason = str(reason or "").strip()
    if len(reason) <= _MAX_REASON_CHARS:
        return reason
    return reason[:_MAX_REASON_CHARS] + f"...(truncated {len(reason) - _MAX_REASON_CHARS} chars)"


def _relpath(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path)
