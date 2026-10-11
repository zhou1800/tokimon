"""Tokimon-owned target-platform plugin manifests."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from tokimon.ai_harness_interface import contract, templates

SEMVER = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-((?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
)


def validate_version(version: str) -> None:
    if not SEMVER.fullmatch(version):
        raise ValueError("plugin version must be strict semantic versioning")


def apply_manifest_overlay(
    manifest: dict[str, Any],
    *,
    version: str = contract.PACKAGE_VERSION,
) -> dict[str, Any]:
    """Patch Tokimon values into a plugin-creator-seeded manifest."""
    payload = deepcopy(manifest)
    payload["name"] = contract.PACKAGE_ID
    payload["version"] = version
    overlay = templates.load_json_template("plugin-manifest-overlay.json")
    payload["description"] = overlay["description"]
    payload["license"] = overlay["license"]
    payload["author"] = {"name": "Tokimon"}
    payload["skills"] = "./skills/"
    payload["mcpServers"] = "./.mcp.json"
    payload.pop("hooks", None)
    payload.pop("apps", None)

    interface = payload.get("interface")
    if not isinstance(interface, dict):
        interface = {}
    overlay_interface = overlay.get("interface")
    if not isinstance(overlay_interface, dict):
        raise ValueError("plugin manifest overlay template must include interface object")
    interface.update(overlay_interface)
    payload["interface"] = interface
    return payload


CLAUDE_MANIFEST_FORBIDDEN_FIELDS: tuple[str, ...] = (
    "hooks",
    "commands",
    "agents",
    "skills",
    "interface",
    "apps",
)


def build_claude_manifest(*, version: str = contract.PACKAGE_VERSION) -> dict[str, Any]:
    """Build the Tokimon-owned Claude Code plugin manifest."""
    overlay = templates.load_json_template("claude-plugin-manifest-overlay.json")
    return {
        "name": contract.PACKAGE_ID,
        "displayName": overlay["displayName"],
        "version": version,
        "description": overlay["description"],
        "license": overlay["license"],
        "author": {"name": "Tokimon"},
        "mcpServers": "./.mcp.json",
    }
