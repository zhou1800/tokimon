"""Resolve and record optional package surfaces without installing dependencies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Mapping

from tokimon.ai_harness_interface import contract

DOCKER_CAPABILITIES = frozenset({"mcp.remote-browser-broker", "mcp.container-executor"})
DOCKER_SKILLS = frozenset({"container-executor"})
POLICY_FILENAME = "installation-policy.json"


@dataclass(frozen=True)
class InstallationPolicy:
    docker_features: str
    docker_available: bool

    @property
    def docker_enabled(self) -> bool:
        return self.docker_features == "auto" and self.docker_available

    @property
    def mcp_entries(self) -> tuple[contract.McpEntry, ...]:
        return tuple(
            entry for entry in contract.MCP_ENTRIES
            if self.docker_enabled or entry.capability_id not in DOCKER_CAPABILITIES
        )

    def skill_ids(self, source_root: Path) -> tuple[str, ...]:
        return tuple(
            skill for skill in contract.package_skill_ids(source_root=source_root)
            if self.docker_enabled or skill not in DOCKER_SKILLS
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dependency_installation": "never",
            "docker_features": self.docker_features,
            "docker_available": self.docker_available,
            "docker_enabled": self.docker_enabled,
            "disabled_capabilities": [] if self.docker_enabled else sorted(DOCKER_CAPABILITIES),
            "disabled_skills": [] if self.docker_enabled else sorted(DOCKER_SKILLS),
            "browser_preference": "in-app",
        }


def validate_settings(settings: Mapping[str, Any]) -> str:
    section = settings.get("installation", {})
    if not isinstance(section, Mapping):
        raise ValueError("installation settings must be an object")
    mode = section.get("docker_features", "auto")
    if mode not in ("auto", "disabled"):
        raise ValueError("installation.docker_features must be auto or disabled")
    return mode


def resolve(settings: Mapping[str, Any]) -> InstallationPolicy:
    mode = validate_settings(settings)
    command = os.environ.get("TOKIMON_RB_DOCKER", "docker").strip()
    if not command:
        raise ValueError("TOKIMON_RB_DOCKER must not be blank")
    return InstallationPolicy(mode, bool(which(command)))


def from_json(payload: Any) -> InstallationPolicy:
    if not isinstance(payload, dict) or type(payload.get("docker_available")) is not bool:
        raise ValueError("invalid installation policy snapshot")
    mode = validate_settings({"installation": payload})
    policy = InstallationPolicy(mode, payload["docker_available"])
    if payload != policy.to_json():
        raise ValueError("installation policy snapshot does not match its declared selection")
    return policy
