#!/usr/bin/env python3
"""Read-only capability registry for the Tokimon tools layer."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tokimon.runtime import source_root
from tokimon.tools import catalog
from tokimon.tools.capability_readiness import ToolCapability, resolve_capabilities
from tokimon.tools.connector_inventory import (
    MCPConnector,
    MCPConnectorInventoryError,
)
from tokimon.tools.connector_inventory import load_mcp_connectors as _load_mcp_connectors


WORKSPACE_ROOT = source_root()


@dataclass(frozen=True)
class HostProfile:
    host_id: str
    os_name: str
    has_gui: bool
    active_codex_host: bool
    local_tool_families: tuple[str, ...]
    reachable_mcp_endpoints: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "host_id": self.host_id,
            "os_name": self.os_name,
            "has_gui": self.has_gui,
            "active_codex_host": self.active_codex_host,
            "local_tool_families": list(self.local_tool_families),
            "reachable_mcp_endpoints": list(self.reachable_mcp_endpoints),
        }


class ToolCapabilityRequirementError(RuntimeError):
    exit_code = 3

    def __init__(
        self,
        message: str,
        *,
        capability: ToolCapability | None = None,
    ) -> None:
        super().__init__(message)
        self.capability = capability


class UnknownToolCapabilityError(ToolCapabilityRequirementError):
    exit_code = 2


def load_mcp_connectors(
    *,
    workspace_root: Path = WORKSPACE_ROOT,
    env: Mapping[str, str] | None = None,
) -> tuple[MCPConnector, ...]:
    """Load connector inventory through the Registry's stable read-only API."""

    return _load_mcp_connectors(workspace_root=workspace_root, env=env)


def _host_profile(
    connectors: tuple[MCPConnector, ...],
    *,
    platform_name: str | None,
    host_id: str | None,
    env: Mapping[str, str] | None,
) -> HostProfile:
    effective_platform = platform_name or platform.system()
    effective_env = dict(os.environ if env is None else env)
    normalized_os = effective_platform.lower()
    is_macos = normalized_os == "darwin"
    is_windows = normalized_os == "windows"
    has_gui = is_macos or is_windows or bool(
        effective_env.get("DISPLAY") or effective_env.get("WAYLAND_DISPLAY")
    )

    local_tool_families: list[str] = ["shell", "filesystem"]
    if has_gui:
        local_tool_families.append("gui")
    return HostProfile(
        host_id=host_id or socket.gethostname(),
        os_name=effective_platform,
        has_gui=has_gui,
        active_codex_host=True,
        local_tool_families=tuple(local_tool_families),
        reachable_mcp_endpoints=tuple(
            connector.id for connector in connectors if connector.available
        ),
    )


def detect_host_profile(
    *,
    platform_name: str | None = None,
    host_id: str | None = None,
    env: Mapping[str, str] | None = None,
    workspace_root: Path = WORKSPACE_ROOT,
) -> HostProfile:
    connectors = load_mcp_connectors(workspace_root=workspace_root, env=env)
    return _host_profile(
        connectors,
        platform_name=platform_name,
        host_id=host_id,
        env=env,
    )


def list_capabilities(
    *,
    workspace_root: Path = WORKSPACE_ROOT,
    env: Mapping[str, str] | None = None,
) -> tuple[ToolCapability, ...]:
    connectors = load_mcp_connectors(workspace_root=workspace_root, env=env)
    return resolve_capabilities(
        catalog.MCP_CATALOG,
        workspace_root=workspace_root,
        env=env,
        connectors=connectors,
    )


def _find_resolved_capability(
    capability_id: str,
    capabilities: tuple[ToolCapability, ...],
) -> ToolCapability | None:
    return next(
        (capability for capability in capabilities if capability.id == capability_id),
        None,
    )


def find_capability(
    capability_id: str,
    *,
    workspace_root: Path = WORKSPACE_ROOT,
    env: Mapping[str, str] | None = None,
) -> ToolCapability | None:
    return _find_resolved_capability(
        capability_id,
        list_capabilities(workspace_root=workspace_root, env=env),
    )


def _resolve_registry(
    *,
    workspace_root: Path,
    env: Mapping[str, str] | None,
    host: HostProfile | None = None,
    platform_name: str | None = None,
    host_id: str | None = None,
) -> tuple[HostProfile, tuple[ToolCapability, ...]]:
    connectors = load_mcp_connectors(workspace_root=workspace_root, env=env)
    resolved_host = host or _host_profile(
        connectors,
        platform_name=platform_name,
        host_id=host_id,
        env=env,
    )
    capabilities = resolve_capabilities(
        catalog.MCP_CATALOG,
        workspace_root=workspace_root,
        env=env,
        connectors=connectors,
    )
    return resolved_host, capabilities


def _require_resolved_capability(
    capability_id: str,
    capabilities: tuple[ToolCapability, ...],
) -> ToolCapability:
    capability = _find_resolved_capability(capability_id, capabilities)
    if capability is None:
        raise UnknownToolCapabilityError(f"Unknown tool capability: {capability_id}")
    if capability.status != "available":
        raise ToolCapabilityRequirementError(
            render_capability_requirement_failure(capability),
            capability=capability,
        )
    return capability


def require_capability(
    capability_id: str,
    host: HostProfile | None = None,
    *,
    workspace_root: Path = WORKSPACE_ROOT,
    env: Mapping[str, str] | None = None,
) -> ToolCapability:
    """Return an available capability or fail before domain work starts."""

    _, capabilities = _resolve_registry(
        workspace_root=workspace_root,
        env=env,
        host=host,
    )
    return _require_resolved_capability(capability_id, capabilities)


def _capability_map_from_resolved(
    host: HostProfile,
    capabilities: tuple[ToolCapability, ...],
) -> dict[str, object]:
    return {
        "host": host.to_dict(),
        "capabilities": [capability.to_dict() for capability in capabilities],
    }


def capability_map(
    host: HostProfile | None = None,
    *,
    workspace_root: Path = WORKSPACE_ROOT,
    env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    host_id: str | None = None,
) -> dict[str, object]:
    resolved_host, capabilities = _resolve_registry(
        workspace_root=workspace_root,
        env=env,
        host=host,
        platform_name=platform_name,
        host_id=host_id,
    )
    return _capability_map_from_resolved(resolved_host, capabilities)


def render_capability_requirement_failure(capability: ToolCapability) -> str:
    lines = [
        f"Tool capability is not available: {capability.id} [{capability.status}]",
        f"Name: {capability.name}",
        f"Description: {capability.description}",
    ]
    if capability.docs:
        lines.append(f"Docs: {capability.docs}")
    if capability.health_check:
        lines.append(f"Health check: {capability.health_check}")
    if capability.host_requirements:
        lines.append("Host requirements:")
        lines.extend(f"- {requirement}" for requirement in capability.host_requirements)
    if capability.failure_modes:
        lines.append("Failure modes:")
        lines.extend(f"- {failure_mode}" for failure_mode in capability.failure_modes)
    return "\n".join(lines)


def render_capabilities(
    host: HostProfile,
    capabilities: tuple[ToolCapability, ...],
) -> str:
    lines = [
        f"Host: {host.host_id}",
        f"OS: {host.os_name}",
        f"GUI: {host.has_gui}",
        "",
        "Tool capabilities:",
    ]
    for capability in capabilities:
        lines.append(f"- {capability.id}: {capability.name} [{capability.status}]")
        lines.append(f"  family: {capability.family}")
        lines.append(f"  description: {capability.description}")
        if capability.health_check:
            lines.append(f"  health check: {capability.health_check}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect Tokimon tool capabilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    host_parser = subparsers.add_parser("host", help="Show the detected host profile")
    host_parser.add_argument("--json", action="store_true")

    list_parser = subparsers.add_parser(
        "list",
        help="List known tool capabilities and current status",
    )
    list_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser("show", help="Show one tool capability")
    show_parser.add_argument("capability_id")
    show_parser.add_argument("--json", action="store_true")

    require_parser = subparsers.add_parser(
        "require",
        help="Require an available tool capability",
    )
    require_parser.add_argument("capability_id")
    require_parser.add_argument("--json", action="store_true")

    return parser


def _render_inventory_error(args: argparse.Namespace, exc: MCPConnectorInventoryError) -> int:
    if getattr(args, "json", False):
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
    else:
        print(str(exc), file=sys.stderr)
    return exc.exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "host":
            host = detect_host_profile()
            if args.json:
                print(json.dumps(host.to_dict(), indent=2))
            else:
                print(f"Host: {host.host_id}")
                print(f"OS: {host.os_name}")
                print(f"GUI: {host.has_gui}")
                print(f"Active Codex host: {host.active_codex_host}")
                print(f"Local tool families: {', '.join(host.local_tool_families)}")
                print(
                    "Declared available MCP connectors: "
                    f"{', '.join(host.reachable_mcp_endpoints) or 'none'}"
                )
            return 0

        host, capabilities = _resolve_registry(
            workspace_root=WORKSPACE_ROOT,
            env=None,
        )
        if args.command == "list":
            if args.json:
                print(
                    json.dumps(
                        _capability_map_from_resolved(host, capabilities),
                        indent=2,
                    )
                )
            else:
                print(render_capabilities(host, capabilities))
            return 0

        if args.command == "show":
            capability = _find_resolved_capability(args.capability_id, capabilities)
            if capability is None:
                message = f"Tool capability not found: {args.capability_id}"
                if args.json:
                    print(
                        json.dumps(
                            {
                                "ok": False,
                                "error": message,
                                "capabilityId": args.capability_id,
                            },
                            indent=2,
                        )
                    )
                else:
                    print(message)
                return 2
            if args.json:
                print(json.dumps(capability.to_dict(), indent=2))
            else:
                print(render_capabilities(host, (capability,)))
            return 0

        if args.command == "require":
            try:
                capability = _require_resolved_capability(
                    args.capability_id,
                    capabilities,
                )
            except ToolCapabilityRequirementError as exc:
                if args.json:
                    payload: dict[str, object] = {
                        "ok": False,
                        "error": str(exc),
                        "capabilityId": args.capability_id,
                    }
                    if exc.capability is not None:
                        payload["capability"] = exc.capability.to_dict()
                    print(json.dumps(payload, indent=2))
                else:
                    print(str(exc), file=sys.stderr)
                return exc.exit_code
            if args.json:
                print(
                    json.dumps(
                        {"ok": True, "capability": capability.to_dict()},
                        indent=2,
                    )
                )
            else:
                print(render_capabilities(host, (capability,)))
            return 0

    except MCPConnectorInventoryError as exc:
        return _render_inventory_error(args, exc)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
