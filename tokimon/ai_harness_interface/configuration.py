"""Repository-local setup for AI Harness Interface generation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from tokimon import settings as settings_helpers
from tokimon.ai_harness_interface import installation_policy
from tokimon.workflow.pair import contract as pair_contract

AI_DEBATE_ACCESS_CHOICES: tuple[str, ...] = ("local-cli",)
PAIR_ACCESS_CHOICES: tuple[str, ...] = pair_contract.CODEX_CLAUDE_PEER_CAPABILITIES
PAIR_PEER_CHOICES: tuple[str, ...] = pair_contract.PEER_B_AGENTS
RETIRED_ACCESS_CHOICES: tuple[str, ...] = ("mcp.remote-agent-broker",)
PEER_CHOICES: tuple[str, ...] = ("claude", "codex")


class ConfigurationError(ValueError):
    """Raised when AI Harness setup is missing or invalid."""


def configure_settings(
    *,
    source_root: Path,
    settings_path: Path | None = None,
    ai_debate_access: str | None = None,
    ai_debate_peer_b: str | None = None,
    pair_access: str | None = None,
    pair_peer_b: str | None = None,
    docker_features: str | None = None,
    assume_yes: bool = False,
) -> dict[str, Any]:
    """Write repository-local settings needed by generated AI Harness entries."""

    root = source_root.expanduser().resolve(strict=False)
    resolved_settings_path = settings_helpers.resolve_settings_path(
        settings_path,
        workspace_root=root,
    )
    payload = _default_settings(source_root=root)
    existing = _load_existing_settings(resolved_settings_path)
    payload = _deep_merge(payload, existing)
    if docker_features is not None:
        payload.setdefault("installation", {})
        if not isinstance(payload["installation"], dict):
            raise ConfigurationError("installation settings must be an object")
        payload["installation"]["docker_features"] = docker_features
    _remove_retired_mcp_selection(payload)
    agents = payload.get("agents")
    if isinstance(agents, dict):
        agents.pop("codex", None)
        agents.pop("claude", None)
        if not agents:
            payload.pop("agents", None)
    payload.setdefault("ai_debate", {})
    payload.setdefault("pair", {})
    if not isinstance(payload["ai_debate"], dict):
        raise ConfigurationError("ai_debate settings must be an object")
    if not isinstance(payload["pair"], dict):
        raise ConfigurationError("pair settings must be an object")

    payload["pair"]["cwd"] = str(root)
    debate_access = _resolve_access(
        key="ai_debate.access",
        current=payload["ai_debate"].get("access"),
        explicit=ai_debate_access,
        choices=AI_DEBATE_ACCESS_CHOICES,
    )
    debate_peer_b = _resolve_choice(
        key="ai_debate.peer_b",
        current=payload["ai_debate"].get("peer_b"),
        explicit=ai_debate_peer_b,
        choices=PEER_CHOICES,
        assume_yes=assume_yes,
    )
    payload["ai_debate"] = {
        "access": debate_access,
        "peer_b": debate_peer_b,
    }
    payload["pair"]["access"] = _resolve_access(
        key="pair.access",
        current=payload["pair"].get("access"),
        explicit=pair_access,
        choices=PAIR_ACCESS_CHOICES,
    )
    payload["pair"]["peer_b"] = _resolve_choice(
        key="pair.peer_b",
        current=payload["pair"].get("peer_b"),
        explicit=pair_peer_b,
        choices=PAIR_PEER_CHOICES,
        assume_yes=assume_yes,
    )

    validate_settings_payload(payload, source_root=root)
    resolved_settings_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_settings_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "settings_path": str(resolved_settings_path),
        "settings": payload,
    }


def require_configured_settings(
    *,
    source_root: Path,
    settings_path: Path | None = None,
) -> dict[str, Any]:
    """Validate settings required before AI Harness package generation."""

    root = source_root.expanduser().resolve(strict=False)
    resolved_settings_path = settings_helpers.resolve_settings_path(
        settings_path,
        workspace_root=root,
    )
    try:
        payload = settings_helpers.load_settings(
            path=resolved_settings_path,
            workspace_root=root,
            required=True,
        )
    except settings_helpers.TokimonSettingsError as exc:
        raise ConfigurationError(
            f"AI Harness setup is required before generation: {exc}. "
            "Run `.venv/bin/python -m tokimon.ai_harness_interface configure`."
        ) from exc
    validate_settings_payload(payload, source_root=root)
    return {
        "ok": True,
        "settings_path": str(resolved_settings_path),
        "settings": payload,
    }


def validate_settings_payload(payload: Mapping[str, Any], *, source_root: Path) -> None:
    root = source_root.expanduser().resolve(strict=False)
    try:
        installation_policy.validate_settings(payload)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    ai_debate = _required_mapping(payload, "ai_debate")
    _required_choice(
        ai_debate,
        "access",
        choices=AI_DEBATE_ACCESS_CHOICES,
        context="ai_debate",
    )
    _required_choice(
        ai_debate,
        "peer_b",
        choices=PEER_CHOICES,
        context="ai_debate",
    )

    pair = _required_mapping(payload, "pair")
    pair_access = _required_choice(
        pair,
        "access",
        choices=PAIR_ACCESS_CHOICES,
        context="pair",
    )
    pair_peer_b = _required_choice(pair, "peer_b", choices=PAIR_PEER_CHOICES, context="pair")
    if pair_access == "native-subagent" and pair_peer_b == "claude":
        raise ConfigurationError("pair.peer_b must be codex for native-subagent access")
    cwd = Path(_required_text(pair, "cwd", context="pair")).expanduser().resolve(strict=False)
    if cwd != root:
        raise ConfigurationError(f"pair.cwd must match live checkout path: {root}")


def _default_settings(*, source_root: Path) -> dict[str, Any]:
    template_path = source_root / settings_helpers.SETTINGS_TEMPLATE_PATH
    try:
        payload = json.loads(template_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"unable to read settings template {template_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"settings template must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError("settings template must contain a JSON object")
    return payload


def _load_existing_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in Tokimon settings: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError("Tokimon settings must contain a JSON object")
    return payload


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(result.get(key), dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _remove_retired_mcp_selection(payload: dict[str, Any]) -> None:
    harness = payload.get("ai_harness_interface")
    if not isinstance(harness, dict):
        return
    harness.pop("mcp_capabilities", None)
    if not harness:
        payload.pop("ai_harness_interface", None)


def _resolve_choice(
    *,
    key: str,
    current: Any,
    explicit: str | None,
    choices: tuple[str, ...],
    assume_yes: bool,
) -> str:
    if explicit is not None:
        return _normalize_choice(explicit, key=key, choices=choices)
    current_choice = _normalize_choice(current, key=key, choices=choices)
    if assume_yes:
        return current_choice
    raise ConfigurationError(
        "AI Harness configure does not prompt during install. "
        f"Run with --yes or an explicit value for {key}; "
        "run `post-install` after installation for readiness and remaining work."
    )


def _resolve_access(
    *,
    key: str,
    current: Any,
    explicit: str | None,
    choices: tuple[str, ...],
) -> str:
    if explicit is not None:
        return _normalize_choice(explicit, key=key, choices=choices)
    if isinstance(current, str) and current.strip() in RETIRED_ACCESS_CHOICES:
        return "local-cli"
    return _normalize_choice(current, key=key, choices=choices)


def _normalize_choice(value: Any, *, key: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be one of: {', '.join(choices)}")
    choice = value.strip()
    if choice not in choices:
        raise ConfigurationError(f"{key} must be one of: {', '.join(choices)}")
    return choice


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str | None = None,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        prefix = f"{context}." if context else ""
        raise ConfigurationError(f"{prefix}{key} must be an object")
    return value


def _required_text(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _required_choice(
    payload: Mapping[str, Any],
    key: str,
    *,
    choices: tuple[str, ...],
    context: str,
) -> str:
    return _normalize_choice(payload.get(key), key=f"{context}.{key}", choices=choices)
