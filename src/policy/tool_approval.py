"""Opt-in approval gate for high-risk tool calls."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal


ToolApprovalMode = Literal["off", "block", "deny"]

_ALLOWLIST_FILE_REL = ".tokimon-tmp/approvals/allowlist.json"

def approval_allowlist_file_path(workspace_root: str | Path | None = None) -> Path:
    """Return the Phase 4 approval allowlist file path under *workspace_root*."""
    if workspace_root is None:
        workspace_root = Path.cwd()
    return Path(workspace_root) / _ALLOWLIST_FILE_REL


def write_allowlist_file(file_ids: set[str], workspace_root: str | Path | None = None) -> Path:
    """Write the file allowlist (canonical JSON, stable ordering).

    Creates parent directories as needed.
    """
    path = approval_allowlist_file_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"allowlist": sorted(file_ids)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def tool_approval_mode_from_env(env: dict[str, str] | None = None) -> ToolApprovalMode:
    if env is None:
        env = os.environ  # pragma: no cover
    raw = str(env.get("TOKIMON_TOOL_APPROVAL_MODE", "off") or "off").strip().lower()
    if raw not in {"off", "block", "deny"}:
        raw = "off"
    return raw  # type: ignore[return-value]


def approval_id_for(tool: str, action: str, args_hash: str) -> str:
    payload = json.dumps(
        {"tool": str(tool or ""), "action": str(action or ""), "args_hash": str(args_hash or "")},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_approval_request(
    *,
    tool: str,
    action: str,
    args_hash: str,
    args_preview: dict[str, Any],
    reason: str,
    max_reason_chars: int = 200,
) -> dict[str, Any]:
    bounded_reason = str(reason or "").strip()
    if len(bounded_reason) > max_reason_chars:
        bounded_reason = bounded_reason[:max_reason_chars] + "...(truncated)"
    return {
        "approval_id": approval_id_for(tool, action, args_hash),
        "tool": str(tool or ""),
        "action": str(action or ""),
        "args_hash": str(args_hash or ""),
        "args_preview": args_preview,
        "reason": bounded_reason,
    }


# ---------------------------------------------------------------------------
# Phase 4: Approval allowlists
# ---------------------------------------------------------------------------


def _load_allowlist_from_env(env: dict[str, str] | None = None) -> set[str]:
    """Return approval_id set from TOKIMON_TOOL_APPROVAL_ALLOWLIST (comma-separated)."""
    if env is None:
        env = os.environ  # pragma: no cover
    raw = str(env.get("TOKIMON_TOOL_APPROVAL_ALLOWLIST", "") or "").strip()
    if not raw:
        return set()
    return {tok.strip() for tok in raw.split(",") if tok.strip()}


def _load_allowlist_from_file(workspace_root: str | Path | None = None) -> set[str]:
    """Return approval_id set from .tokimon-tmp/approvals/allowlist.json.

    Fail-safe: missing or malformed file -> empty set.
    """
    path = approval_allowlist_file_path(workspace_root)
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            return set()
        items = data.get("allowlist")
        if not isinstance(items, list):
            return set()
        return {str(item).strip() for item in items if isinstance(item, str) and str(item).strip()}
    except Exception:
        return set()


def load_approval_allowlist(
    env: dict[str, str] | None = None,
    workspace_root: str | Path | None = None,
) -> tuple[set[str], set[str]]:
    """Load allowlists from env and file.

    Returns (env_ids, file_ids) so callers can determine the source of a match.
    """
    return _load_allowlist_from_env(env), _load_allowlist_from_file(workspace_root)


def check_allowlist(
    approval_id: str,
    env_ids: set[str],
    file_ids: set[str],
) -> tuple[bool, str]:
    """Check if *approval_id* is pre-approved.

    Returns (is_approved, source).  source is ``"env"`` or ``"file"`` when
    approved, ``""`` otherwise.  Env is checked first (first-match wins).
    """
    if approval_id in env_ids:
        return True, "env"
    if approval_id in file_ids:
        return True, "file"
    return False, ""
