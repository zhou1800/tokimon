"""Shared machine-local settings helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from tokimon.runtime import resolve_runtime_path, source_root

SETTINGS_LOCAL_PATH = Path(".tokimon") / "settings.local.json"
SETTINGS_TEMPLATE_PATH = Path("tokimon") / "settings.example.json"
DEFAULT_AGENT_IDLE_TIMEOUT_SEC = 3600
DEFAULT_AGENT_POLL_INTERVAL_SEC = 5.0


class TokimonSettingsError(ValueError):
    """Raised when Tokimon local settings are malformed."""


def resolve_settings_path(
    path: str | Path | None = None,
    *,
    workspace_root: str | Path | None = None,
) -> Path:
    root = Path(workspace_root).expanduser() if workspace_root is not None else source_root()
    return resolve_runtime_path(path or SETTINGS_LOCAL_PATH, workspace_root=root).resolve()


def load_settings(
    path: str | Path | None = None,
    *,
    workspace_root: str | Path | None = None,
    required: bool = False,
) -> dict[str, Any]:
    resolved = resolve_settings_path(path, workspace_root=workspace_root)
    if not resolved.exists():
        if required:
            raise TokimonSettingsError(f"missing Tokimon settings: {resolved}")
        return {}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TokimonSettingsError(f"invalid JSON in Tokimon settings: {exc}") from exc
    if not isinstance(payload, dict):
        raise TokimonSettingsError("Tokimon settings must contain a JSON object")
    return dict(payload)


def load_settings_section(
    section: str,
    *,
    path: str | Path | None = None,
    workspace_root: str | Path | None = None,
    required: bool = False,
) -> dict[str, Any]:
    payload = load_settings(path, workspace_root=workspace_root, required=required)
    value = payload.get(section)
    if value is None:
        if required:
            raise TokimonSettingsError(f"missing Tokimon settings section: {section}")
        return {}
    if not isinstance(value, Mapping):
        raise TokimonSettingsError(f"Tokimon settings section {section!r} must be an object")
    return dict(value)


def resolve_agent_options(
    agent: str,
    options: Mapping[str, Any] | None,
    *,
    path: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Preserve caller options without selecting a provider model or effort."""
    return dict(options or {})


def resolve_agent_execution_options(
    options: Mapping[str, Any] | None = None,
    *,
    path: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    configured = load_settings_section(
        "agent_execution",
        path=path,
        workspace_root=workspace_root,
    )
    merged = dict(configured)
    merged.update(dict(options or {}))
    return {
        "idle_timeout_sec": _positive_int_option(
            merged,
            "idle_timeout_sec",
            default=DEFAULT_AGENT_IDLE_TIMEOUT_SEC,
        ),
        "poll_interval_sec": _positive_float_option(
            merged,
            "poll_interval_sec",
            default=DEFAULT_AGENT_POLL_INTERVAL_SEC,
        ),
        "max_wall_time_sec": _optional_positive_int_option(merged, "max_wall_time_sec"),
    }


def _positive_int_option(
    options: Mapping[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = options.get(key, default)
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise TokimonSettingsError(f"agent_execution.{key} must be an integer") from exc
    if resolved <= 0:
        raise TokimonSettingsError(f"agent_execution.{key} must be greater than zero")
    return resolved


def _optional_positive_int_option(options: Mapping[str, Any], key: str) -> int | None:
    value = options.get(key)
    if value is None:
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise TokimonSettingsError(f"agent_execution.{key} must be an integer") from exc
    if resolved <= 0:
        raise TokimonSettingsError(f"agent_execution.{key} must be greater than zero")
    return resolved


def _positive_float_option(
    options: Mapping[str, Any],
    key: str,
    *,
    default: float,
) -> float:
    value = options.get(key, default)
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TokimonSettingsError(f"agent_execution.{key} must be a number") from exc
    if not math.isfinite(resolved) or resolved <= 0:
        raise TokimonSettingsError(f"agent_execution.{key} must be finite and greater than zero")
    return resolved
