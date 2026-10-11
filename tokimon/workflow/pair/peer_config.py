"""Peer selection configuration for Pair workflow runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tokimon.runtime import source_root
from tokimon.settings import (
    SETTINGS_LOCAL_PATH,
    TokimonSettingsError,
    load_settings_section,
    resolve_settings_path,
)
from tokimon.workflow.pair import contract


WORKSPACE_ROOT = source_root()
CODEX_CLAUDE_CAPABILITIES = set(contract.CODEX_CLAUDE_PEER_CAPABILITIES)


class PeerSelectionConfigError(ValueError):
    """Raised when Pair settings are missing or invalid."""


def timestamp_text(now: datetime | None = None) -> str:
    current = now or datetime.now(tz=timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config(settings_path: str | Path | None = None) -> dict[str, Any]:
    try:
        return load_settings_section(
            contract.SETTINGS_SECTION,
            path=settings_path,
            workspace_root=WORKSPACE_ROOT,
            required=True,
        )
    except TokimonSettingsError as exc:
        raise PeerSelectionConfigError(str(exc)) from exc


def _required_text(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PeerSelectionConfigError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def select_peers(
    config: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    cwd: str | Path | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    unsupported_keys = set(config) - {"access", "peer_b", "cwd"}
    if unsupported_keys:
        keys = ", ".join(sorted(unsupported_keys))
        raise PeerSelectionConfigError(f"unsupported pair settings keys: {keys}")
    access = _required_text(config, "access", context=contract.SETTINGS_SECTION)
    if access not in CODEX_CLAUDE_CAPABILITIES:
        choices = " or ".join(contract.CODEX_CLAUDE_PEER_CAPABILITIES)
        raise PeerSelectionConfigError(f"pair.access must be {choices}")
    peer_b = _required_text(config, "peer_b", context=contract.SETTINGS_SECTION)
    if peer_b not in contract.PEER_B_AGENTS:
        raise PeerSelectionConfigError("pair.peer_b must be claude or codex")
    if access == contract.NATIVE_SUBAGENT_CAPABILITY and peer_b == "claude":
        raise PeerSelectionConfigError("pair.peer_b must be codex for native-subagent access")
    if cwd is None:
        peer_cwd = _required_text(config, "cwd", context=contract.SETTINGS_SECTION)
    else:
        if not str(cwd).strip():
            raise PeerSelectionConfigError("Pair task working directory must be an existing directory")
        try:
            task_directory = Path(cwd).expanduser().resolve()
            is_directory = task_directory.is_dir()
        except (OSError, RuntimeError) as exc:
            raise PeerSelectionConfigError("Pair task working directory cannot be resolved") from exc
        if not is_directory:
            raise PeerSelectionConfigError("Pair task working directory must be an existing directory")
        peer_cwd = str(task_directory)
    prefix = "native" if access == contract.NATIVE_SUBAGENT_CAPABILITY else "local"
    peers = {
        "a": {
            "id": f"{prefix}-a-{contract.PEER_A_AGENT}",
            "capability": access,
            "agent": contract.PEER_A_AGENT,
            "cwd": peer_cwd,
        },
        "b": {
            "id": f"{prefix}-b-{peer_b}",
            "capability": access,
            "agent": peer_b,
            "cwd": peer_cwd,
        },
    }
    peer_ids = [peers["a"]["id"], peers["b"]["id"]]

    selection = {
        "source_path": str(
            Path(source_path)
            if source_path is not None
            else resolve_settings_path(SETTINGS_LOCAL_PATH, workspace_root=WORKSPACE_ROOT)
        ),
        "settings_section": contract.SETTINGS_SECTION,
        "selected_peer_ids": peer_ids,
        "selected_at": timestamp_text(now),
    }
    return peers, selection


def load_and_select(
    *,
    settings_path: str | Path | None = None,
    cwd: str | Path | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = resolve_settings_path(settings_path or SETTINGS_LOCAL_PATH, workspace_root=WORKSPACE_ROOT)
    config = load_config(path)
    return select_peers(config, source_path=path, cwd=cwd, now=now)
