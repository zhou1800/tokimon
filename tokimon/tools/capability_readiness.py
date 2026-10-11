"""Resolve static tool declarations against local and connector readiness facts."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tokimon import settings as settings_helpers
from tokimon.tools._runtime import resolve_executable as _resolve_executable
from tokimon.runtime import resolve_runtime_path
from tokimon.tools.catalog import McpCatalogEntry
from tokimon.tools.connector_inventory import (
    MCPConnector,
    available_connector_for_capability,
)


DEFAULT_TELEGRAM_SECRETS_RELATIVE_PATH = Path(".tokimon") / "secrets.local.json"


@dataclass(frozen=True)
class ToolCapability:
    id: str
    name: str
    family: str
    status: str
    description: str
    implementation: str | None
    docs: str | None
    commands: tuple[str, ...]
    host_requirements: tuple[str, ...]
    health_check: str | None
    artifact_policy: str
    failure_modes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "status": self.status,
            "description": self.description,
            "implementation": self.implementation,
            "docs": self.docs,
            "commands": list(self.commands),
            "host_requirements": list(self.host_requirements),
            "health_check": self.health_check,
            "artifact_policy": self.artifact_policy,
            "failure_modes": list(self.failure_modes),
        }


@dataclass(frozen=True)
class AiDebateReadinessConfiguration:
    """Validated settings that determine AI Debate's required local providers."""

    access: str
    peer_b: str
    required_providers: tuple[str, ...]


def _telegram_token_configured(*, workspace_root: Path, env: Mapping[str, str]) -> bool:
    if str(env.get("TELEGRAM_BOT_TOKEN") or "").strip():
        return True
    secrets_path = resolve_runtime_path(
        DEFAULT_TELEGRAM_SECRETS_RELATIVE_PATH,
        workspace_root=workspace_root,
    )
    if not secrets_path.exists():
        return False
    try:
        payload = json.loads(secrets_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    telegram = payload.get("telegram", {})
    if not isinstance(telegram, Mapping):
        return False
    token = (
        telegram.get("bot_token")
        or telegram.get("botToken")
        or payload.get("telegram_bot_token")
        or payload.get("telegramBotToken")
    )
    return bool(str(token or "").strip())


def require_ai_debate_readiness_configuration(
    payload: Mapping[str, Any],
) -> AiDebateReadinessConfiguration:
    """Validate the settings shared by registry readiness and runtime health."""

    debate = payload.get("ai_debate")
    if not isinstance(debate, Mapping):
        raise settings_helpers.TokimonSettingsError("ai_debate must be an object")

    access = debate.get("access")
    if access != "local-cli":
        raise settings_helpers.TokimonSettingsError(
            "ai_debate.access must be local-cli"
        )
    peer_b = debate.get("peer_b")
    if not isinstance(peer_b, str) or peer_b not in {"claude", "codex"}:
        raise settings_helpers.TokimonSettingsError(
            "ai_debate.peer_b must be claude or codex"
        )

    required_providers = tuple(sorted({"codex", peer_b}))
    return AiDebateReadinessConfiguration(
        access=access,
        peer_b=peer_b,
        required_providers=required_providers,
    )


def _ai_debate_ready(*, workspace_root: Path, env: Mapping[str, str]) -> bool:
    try:
        payload = settings_helpers.load_settings(
            workspace_root=workspace_root,
            required=True,
        )
        configuration = require_ai_debate_readiness_configuration(payload)
    except settings_helpers.TokimonSettingsError:
        return False

    if _resolve_executable("codex", "CODEX_CLI", env) is None:
        return False
    if configuration.peer_b == "claude" and _resolve_executable(
        "claude",
        "CLAUDE_CODE_CLI",
        env,
    ) is None:
        return False
    return True


def _local_probe_ready(
    entry: McpCatalogEntry,
    *,
    workspace_root: Path,
    env: Mapping[str, str],
) -> bool:
    if not (workspace_root / entry.implementation_entrypoint).is_file():
        return False
    if entry.readiness_probe == "telegram-token":
        return _telegram_token_configured(workspace_root=workspace_root, env=env)
    if entry.readiness_probe == "docker":
        return shutil.which(
            env.get("TOKIMON_RB_DOCKER", "docker"), path=env.get("PATH")
        ) is not None
    if entry.readiness_probe == "container-executor-docker":
        return shutil.which("docker", path=env.get("PATH")) is not None
    if entry.readiness_probe == "ai-debate":
        return _ai_debate_ready(workspace_root=workspace_root, env=env)
    if entry.readiness_probe == "implementation":
        return True
    raise ValueError(
        f"Unsupported readiness probe {entry.readiness_probe!r} for {entry.capability_id}."
    )


def _connector_capability(
    entry: McpCatalogEntry,
    connector: MCPConnector,
) -> ToolCapability:
    tool_hint = ", ".join(connector.tools) or "tools/list"
    return ToolCapability(
        id=entry.capability_id,
        name=entry.name,
        family=entry.kind,
        status="available",
        description=entry.connector.description,
        implementation=f"{connector.transport}:{connector.id}",
        docs=entry.docs,
        commands=(f"Use configured MCP connector {connector.id} ({tool_hint})",),
        host_requirements=(
            f"Configured MCP connector {connector.id} from {connector.source}",
            "Connector status is available",
        ),
        health_check=f"Configured MCP connector {connector.id} exposes {tool_hint}",
        artifact_policy=entry.connector.artifact_policy,
        failure_modes=entry.connector.failure_modes,
    )


def _local_capability(
    entry: McpCatalogEntry,
    *,
    workspace_root: Path,
    env: Mapping[str, str],
) -> ToolCapability:
    return ToolCapability(
        id=entry.capability_id,
        name=entry.name,
        family=entry.kind,
        status=(
            "available"
            if _local_probe_ready(entry, workspace_root=workspace_root, env=env)
            else "unavailable"
        ),
        description=entry.description,
        implementation=entry.implementation,
        docs=entry.docs,
        commands=entry.commands,
        host_requirements=entry.host_requirements,
        health_check=entry.health_check,
        artifact_policy=entry.artifact_policy,
        failure_modes=entry.failure_modes,
    )


def resolve_capabilities(
    entries: tuple[McpCatalogEntry, ...],
    *,
    workspace_root: Path,
    env: Mapping[str, str] | None,
    connectors: tuple[MCPConnector, ...],
) -> tuple[ToolCapability, ...]:
    effective_env = dict(os.environ if env is None else env)
    capabilities: list[ToolCapability] = []
    for entry in entries:
        connector = available_connector_for_capability(entry.capability_id, connectors)
        if connector is not None:
            capabilities.append(_connector_capability(entry, connector))
        else:
            capabilities.append(
                _local_capability(
                    entry,
                    workspace_root=workspace_root,
                    env=effective_env,
                )
            )
    return tuple(capabilities)
