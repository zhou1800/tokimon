"""MCP manifest generation and route-back diagnostics."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from tokimon.ai_harness_interface import contract, route


def build_mcp_manifest(
    *,
    install_route: route.InstallRoute,
    platform: contract.TargetPlatform = contract.CODEX_PLATFORM,
    entries: tuple[contract.McpEntry, ...] = contract.MCP_ENTRIES,
) -> dict[str, Any]:
    return {
        "mcpServers": {
            entry.server_id: build_mcp_server(entry=entry, install_route=install_route, platform=platform)
            for entry in entries
        }
    }


def build_mcp_server(
    *,
    entry: contract.McpEntry,
    install_route: route.InstallRoute,
    platform: contract.TargetPlatform = contract.CODEX_PLATFORM,
) -> dict[str, Any]:
    python = install_route.python_executable
    server: dict[str, Any] = {
        "command": str(python),
        "args": ["-m", entry.module, *entry.serve_args],
    }
    if platform.platform_id == "claude":
        # Claude Code's MCP schema documents command/args/env but not cwd, so
        # module resolution routes through PYTHONPATH instead.
        server["env"] = {"PYTHONPATH": str(install_route.liveCheckoutPath)}
    else:
        server["cwd"] = str(install_route.liveCheckoutPath)
    return server


def run_health_checks(
    install_route: route.InstallRoute,
    *,
    timeout_sec: int = 30,
    env: dict[str, str] | None = None,
    entries: tuple[contract.McpEntry, ...] = contract.MCP_ENTRIES,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    python = install_route.python_executable
    active_env = os.environ.copy()
    if env:
        active_env.update(env)
    for entry in entries:
        command = [str(python), "-m", entry.module, *entry.health_args]
        try:
            result = subprocess.run(
                command,
                cwd=install_route.liveCheckoutPath,
                env=active_env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            checks.append(
                {
                    "server_id": entry.server_id,
                    "capability_id": entry.capability_id,
                    "command": command,
                    "ok": False,
                    "status": "timeout",
                    "message": f"health check timed out after {timeout_sec}s",
                    "stdout": exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or "",
                    "stderr": exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or "",
                }
            )
            continue
        except OSError as exc:
            checks.append(
                {
                    "server_id": entry.server_id,
                    "capability_id": entry.capability_id,
                    "command": command,
                    "ok": False,
                    "status": "fail",
                    "message": str(exc),
                    "stdout": "",
                    "stderr": str(exc),
                }
            )
            continue
        parsed_stdout = _parse_json(result.stdout)
        route_ok = (
            isinstance(parsed_stdout, dict)
            and isinstance(parsed_stdout.get("ok"), bool)
        )
        checks.append(
            {
                "server_id": entry.server_id,
                "capability_id": entry.capability_id,
                "command": command,
                "ok": route_ok,
                "status": "pass" if route_ok else "fail",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "parsed_stdout": parsed_stdout,
            }
        )
    return checks


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
