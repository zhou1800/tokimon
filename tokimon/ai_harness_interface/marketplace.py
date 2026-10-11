"""Tokimon local marketplace content helpers."""

from __future__ import annotations

from typing import Any

from tokimon.ai_harness_interface import contract


def tokimon_plugin_entry() -> dict[str, Any]:
    return {
        "name": contract.PACKAGE_ID,
        "source": {
            "source": "local",
            "path": contract.MARKETPLACE_SOURCE_PATH,
        },
        "policy": {
            "installation": "AVAILABLE",
            "authentication": "ON_INSTALL",
        },
        "category": contract.MARKETPLACE_CATEGORY,
    }


def apply_tokimon_marketplace_content(payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(payload)
    updated["name"] = contract.MARKETPLACE_NAME
    interface = updated.get("interface")
    if not isinstance(interface, dict):
        interface = {}
    interface["displayName"] = contract.MARKETPLACE_DISPLAY_NAME
    updated["interface"] = interface

    plugins = updated.get("plugins")
    if not isinstance(plugins, list):
        plugins = []
    next_plugins = [
        plugin
        for plugin in plugins
        if not (isinstance(plugin, dict) and plugin.get("name") == contract.PACKAGE_ID)
    ]
    next_plugins.append(tokimon_plugin_entry())
    updated["plugins"] = next_plugins
    return updated


def claude_plugin_entry() -> dict[str, Any]:
    return {
        "name": contract.PACKAGE_ID,
        "source": contract.MARKETPLACE_SOURCE_PATH,
        "description": (
            "Expose Tokimon startup, root skill directories, and MCP entries "
            "while routing implementation work to the configured live checkout."
        ),
        "category": contract.CLAUDE_MARKETPLACE_CATEGORY,
    }


def build_claude_marketplace_content() -> dict[str, Any]:
    return {
        "name": contract.MARKETPLACE_NAME,
        "owner": {"name": contract.MARKETPLACE_OWNER_NAME},
        "description": "Local Tokimon marketplace generated from the live Tokimon checkout.",
        "plugins": [claude_plugin_entry()],
    }


def validate_claude_marketplace_content(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if payload.get("name") != contract.MARKETPLACE_NAME:
        issues.append("marketplace name mismatch")
    owner = payload.get("owner")
    if not isinstance(owner, dict) or owner.get("name") != contract.MARKETPLACE_OWNER_NAME:
        issues.append("marketplace owner mismatch")
    if not isinstance(payload.get("description"), str) or not payload["description"].strip():
        issues.append("marketplace description missing")
    plugins = payload.get("plugins")
    if not isinstance(plugins, list):
        issues.append("marketplace plugins must be a list")
        return issues
    matches = [plugin for plugin in plugins if isinstance(plugin, dict) and plugin.get("name") == contract.PACKAGE_ID]
    if len(matches) != 1:
        issues.append("marketplace must contain exactly one tokimon plugin entry")
        return issues
    expected = claude_plugin_entry()
    for key, value in expected.items():
        if matches[0].get(key) != value:
            issues.append(f"tokimon marketplace entry {key} mismatch")
    return issues


def validate_tokimon_marketplace_content(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if payload.get("name") != contract.MARKETPLACE_NAME:
        issues.append("marketplace name mismatch")
    interface = payload.get("interface")
    if not isinstance(interface, dict) or interface.get("displayName") != contract.MARKETPLACE_DISPLAY_NAME:
        issues.append("marketplace display name mismatch")
    plugins = payload.get("plugins")
    if not isinstance(plugins, list):
        issues.append("marketplace plugins must be a list")
        return issues
    matches = [plugin for plugin in plugins if isinstance(plugin, dict) and plugin.get("name") == contract.PACKAGE_ID]
    if len(matches) != 1:
        issues.append("marketplace must contain exactly one tokimon plugin entry")
        return issues
    expected = tokimon_plugin_entry()
    for key, value in expected.items():
        if matches[0].get(key) != value:
            issues.append(f"tokimon marketplace entry {key} mismatch")
    return issues
