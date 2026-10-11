"""Post-install readiness reporting for AI Harness Interface."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Callable, Sequence

from tokimon import settings as settings_helpers
from tokimon.ai_harness_interface import claude_cli, configuration, contract, generator, installation_policy, mcp, route


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], Path, int], CommandResult]
ExecutableResolver = Callable[[str], str | None]


def post_install_readiness(
    *,
    paths: generator.GenerationPaths,
    settings_path: Path | None = None,
    target_platform: str = contract.TARGET_PLATFORM,
    run_health: bool = True,
    check_platform: bool = True,
    command_runner: CommandRunner | None = None,
    executable_resolver: ExecutableResolver | None = None,
    timeout_sec: int = 30,
) -> dict[str, Any]:
    """Report post-install configuration, platform registration, and readiness."""

    install_route = generator.install_route_for(paths)
    config = _configuration_readiness(source_root=paths.source_root, settings_path=settings_path)
    package = _package_readiness(
        paths=paths,
        install_route=install_route,
    )
    try:
        policy = generator.load_installation_policy(paths)
    except generator.PackageValidationError:
        policy = installation_policy.InstallationPolicy("disabled", False)
    platform = _platform_readiness(
        target_platform=target_platform,
        paths=paths,
        check_platform=check_platform,
        command_runner=command_runner or _run_command,
        executable_resolver=executable_resolver or which,
        timeout_sec=timeout_sec,
        entries=policy.mcp_entries,
    )
    health = (
        mcp.run_health_checks(
            install_route,
            timeout_sec=timeout_sec,
            entries=policy.mcp_entries,
        )
        if run_health
        else []
    )
    servers = _mcp_readiness(
        health=health,
        run_health=run_health,
        platform=platform,
        entries=policy.mcp_entries,
    )
    pair = _pair_readiness(
        config=config,
        source_root=paths.source_root,
        target_platform=target_platform,
        executable_resolver=executable_resolver or which,
    )
    remaining_work = _remaining_work(
        config=config,
        package=package,
        platform=platform,
        servers=servers,
        pair=pair,
    )
    capability_setup = _capability_setup(servers=servers)
    next_steps = _next_steps(remaining_work=remaining_work)
    return {
        "ok": not remaining_work,
        "status": "ready" if not remaining_work else "action_required",
        "summary": _summary(
            remaining_work=remaining_work,
            capability_setup=capability_setup,
            platform=platform,
            servers=servers,
            run_health=run_health,
        ),
        "configuration": config,
        "package": package,
        "target_platform": platform,
        "mcp_servers": servers,
        "pair": pair,
        "remaining_work": remaining_work,
        "capability_setup": capability_setup,
        "next_steps": next_steps,
        "installation_policy": policy.to_json(),
    }


def _configuration_readiness(*, source_root: Path, settings_path: Path | None) -> dict[str, Any]:
    try:
        payload = configuration.require_configured_settings(
            source_root=source_root,
            settings_path=settings_path,
        )
    except configuration.ConfigurationError as exc:
        return {
            "ok": False,
            "settings_path": str(
                settings_helpers.resolve_settings_path(settings_path, workspace_root=source_root)
            ),
            "message": str(exc),
        }
    return {
        "ok": True,
        "settings_path": payload["settings_path"],
        "pair": dict(payload["settings"]["pair"]),
    }


def _package_readiness(
    *,
    paths: generator.GenerationPaths,
    install_route: route.InstallRoute,
) -> dict[str, Any]:
    try:
        route.validate_route(install_route)
    except route.RouteError as exc:
        return {
            "ok": False,
            "install_route": install_route.to_json_object(),
            "message": str(exc),
        }
    try:
        validation = generator.validate_generated_package(
            paths,
            install_route=install_route,
        )
    except generator.PackageValidationError as exc:
        return {
            "ok": False,
            "package_root": str(paths.package_root),
            "marketplace_path": str(paths.marketplace_path),
            "install_route": install_route.to_json_object(),
            "message": str(exc),
        }
    manifest_version = None
    try:
        manifest_version = json.loads(paths.manifest_path.read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError):
        manifest_version = None
    return {
        "ok": True,
        "package_root": str(paths.package_root),
        "marketplace_path": str(paths.marketplace_path),
        "marketplace_name": contract.MARKETPLACE_NAME,
        "package_id": contract.PACKAGE_ID,
        "version": manifest_version,
        "install_route": install_route.to_json_object(),
        "validation": {"ok": validation["ok"]},
    }


def _platform_readiness(
    *,
    target_platform: str,
    paths: generator.GenerationPaths,
    check_platform: bool,
    command_runner: CommandRunner,
    executable_resolver: ExecutableResolver,
    timeout_sec: int,
    entries: tuple[contract.McpEntry, ...] = contract.MCP_ENTRIES,
) -> dict[str, Any]:
    if not check_platform:
        return {
            "ok": True,
            "target_platform": target_platform,
            "checked": False,
            "plugin": {"checked": False},
            "mcp_servers": {},
        }
    if target_platform == "codex":
        plugin = _codex_plugin_status(
            paths=paths,
            command_runner=command_runner,
            timeout_sec=timeout_sec,
        )
        mcp_servers = {
            entry.server_id: _codex_mcp_status(
                server_id=entry.server_id,
                paths=paths,
                command_runner=command_runner,
                timeout_sec=timeout_sec,
            )
            for entry in entries
        }
    elif target_platform == "claude":
        plugin = _claude_plugin_status(
            paths=paths,
            command_runner=command_runner,
            executable_resolver=executable_resolver,
            timeout_sec=timeout_sec,
        )
        mcp_servers = {
            entry.server_id: _claude_mcp_status(server_id=entry.server_id, plugin=plugin)
            for entry in entries
        }
    else:
        return {
            "ok": False,
            "target_platform": target_platform,
            "checked": True,
            "message": f"post-install platform checks are not implemented for {target_platform}",
            "plugin": {"checked": False},
            "mcp_servers": {},
        }
    if plugin.get("skipped"):
        platform_ok = bool(plugin.get("ok"))
    else:
        platform_ok = bool(plugin.get("installed_enabled")) and all(
            bool(item.get("registered")) for item in mcp_servers.values()
        )
    return {
        "ok": platform_ok,
        "target_platform": target_platform,
        "checked": True,
        "plugin": plugin,
        "mcp_servers": mcp_servers,
    }


def _codex_plugin_status(
    *,
    paths: generator.GenerationPaths,
    command_runner: CommandRunner,
    timeout_sec: int,
) -> dict[str, Any]:
    command = ["codex", "plugin", "list"]
    result = command_runner(command, paths.source_root, timeout_sec)
    expected = f"{contract.PACKAGE_ID}@{contract.MARKETPLACE_NAME}"
    line = _matching_line(result.stdout, expected)
    installed_enabled = bool(line and "installed, enabled" in line)
    return {
        "checked": True,
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "plugin_id": contract.PACKAGE_ID,
        "marketplace_name": contract.MARKETPLACE_NAME,
        "installed": bool(line),
        "installed_enabled": installed_enabled,
        "status_line": line,
    }


def _codex_mcp_status(
    *,
    server_id: str,
    paths: generator.GenerationPaths,
    command_runner: CommandRunner,
    timeout_sec: int,
) -> dict[str, Any]:
    command = ["codex", "mcp", "get", server_id]
    result = command_runner(command, paths.source_root, timeout_sec)
    enabled = result.returncode == 0 and "enabled: true" in result.stdout.lower()
    return {
        "checked": True,
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "server_id": server_id,
        "registered": result.returncode == 0,
        "enabled": enabled,
        "transport": _line_value(result.stdout, "transport"),
        "status": "enabled" if enabled else "missing_or_disabled",
    }


def _claude_plugin_status(
    *,
    paths: generator.GenerationPaths,
    command_runner: CommandRunner,
    executable_resolver: ExecutableResolver,
    timeout_sec: int,
) -> dict[str, Any]:
    command = claude_cli.plugin_list_command()
    if executable_resolver(command[0]) is None:
        return {
            "checked": True,
            "ok": not claude_cli.require_claude_cli(),
            "skipped": True,
            "command": command,
            "message": f"missing Claude Code CLI executable: {command[0]}",
            "plugin_id": contract.PACKAGE_ID,
            "marketplace_name": contract.MARKETPLACE_NAME,
            "installed": None,
            "installed_enabled": None,
            "entry": None,
        }
    result = command_runner(command, paths.source_root, timeout_sec)
    inventory_inspection = _inspect_claude_plugin_inventory(result)
    entry = inventory_inspection.get("entry") if inventory_inspection.get("ok") else None
    return {
        "checked": True,
        "ok": bool(inventory_inspection.get("ok")),
        "skipped": False,
        "command": command,
        "returncode": result.returncode,
        "plugin_id": contract.PACKAGE_ID,
        "marketplace_name": contract.MARKETPLACE_NAME,
        "installed": entry is not None,
        "installed_enabled": _claude_entry_enabled(entry) if inventory_inspection.get("ok") else None,
        "entry": entry,
        "inventory_inspection": inventory_inspection,
    }


def _inspect_claude_plugin_inventory(result: CommandResult) -> dict[str, Any]:
    command_text = "claude plugin list --json"
    if result.returncode != 0:
        return {
            "checked": True,
            "ok": False,
            "status": "command_failed",
            "message": f"Claude plugin inventory command exited with status {result.returncode}.",
            "action": f"Fix `{command_text}` so it completes successfully, then rerun post-install.",
            "entry": None,
        }
    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "checked": True,
            "ok": False,
            "status": "invalid_json",
            "message": "Claude plugin inventory output was not valid JSON.",
            "action": f"Run `{command_text}`, resolve its invalid JSON output, then rerun post-install.",
            "entry": None,
        }
    if not isinstance(entries, list):
        return {
            "checked": True,
            "ok": False,
            "status": "invalid_shape",
            "message": "Claude plugin inventory output must be a JSON array.",
            "action": f"Run `{command_text}`, resolve its unexpected output shape, then rerun post-install.",
            "entry": None,
        }
    return {
        "checked": True,
        "ok": True,
        "status": "valid",
        "message": "Claude plugin inventory returned a valid JSON array.",
        "entry_count": len(entries),
        "entry": _matching_claude_plugin_entry(entries),
    }


def _matching_claude_plugin_entry(entries: list[Any]) -> dict[str, Any] | None:
    qualified = f"{contract.PACKAGE_ID}@{contract.MARKETPLACE_NAME}"
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        # The platform does not pin the field names, so match tolerantly.
        name = next((entry[key] for key in ("name", "plugin", "id") if isinstance(entry.get(key), str)), None)
        marketplace_name = next(
            (entry[key] for key in ("marketplace", "marketplaceName", "marketplace_name") if isinstance(entry.get(key), str)),
            None,
        )
        if name == qualified:
            return entry
        if name == contract.PACKAGE_ID and marketplace_name == contract.MARKETPLACE_NAME:
            return entry
    return None


def _claude_entry_enabled(entry: dict[str, Any] | None) -> bool:
    if entry is None:
        return False
    for key in ("enabled", "isEnabled"):
        if key in entry:
            return bool(entry[key])
    # Claude Code enables installed plugins by default when no enabled flag is reported.
    return True


def _claude_mcp_status(*, server_id: str, plugin: dict[str, Any]) -> dict[str, Any]:
    if plugin.get("skipped"):
        return {
            "checked": True,
            "ok": bool(plugin.get("ok")),
            "skipped": True,
            "server_id": server_id,
            "registered": None,
            "enabled": None,
            "derived_from": "plugin_status",
            "status": "skipped",
        }
    inventory_inspection = plugin.get("inventory_inspection")
    if isinstance(inventory_inspection, dict) and not inventory_inspection.get("ok"):
        return {
            "checked": True,
            "ok": False,
            "skipped": False,
            "server_id": server_id,
            "registered": None,
            "enabled": None,
            "derived_from": "plugin_status",
            "registration_inspection_ok": False,
            "status": "inspection_failed",
        }
    registered = bool(plugin.get("installed"))
    enabled = bool(plugin.get("installed_enabled"))
    return {
        "checked": True,
        "ok": bool(plugin.get("ok")),
        "skipped": False,
        "server_id": server_id,
        "registered": registered,
        "enabled": enabled,
        "derived_from": "plugin_status",
        "status": "enabled" if enabled else "missing_or_disabled",
    }


def _mcp_readiness(
    *,
    health: list[dict[str, Any]],
    run_health: bool,
    platform: dict[str, Any],
    entries: tuple[contract.McpEntry, ...] = contract.MCP_ENTRIES,
) -> list[dict[str, Any]]:
    health_by_server = {item.get("server_id"): item for item in health}
    platform_servers = platform.get("mcp_servers") if isinstance(platform.get("mcp_servers"), dict) else {}
    items = []
    for entry in entries:
        health_item = health_by_server.get(entry.server_id)
        platform_item = platform_servers.get(entry.server_id, {})
        route_ok = None if not run_health else bool(health_item and health_item.get("ok"))
        operational = _operational_ready(entry=entry, health_item=health_item, route_ok=route_ok)
        remaining, setup_actions = _server_actions(
            entry=entry,
            platform_item=platform_item,
            route_ok=route_ok,
            operational=operational,
            health_item=health_item,
        )
        items.append(
            {
                "capability_id": entry.capability_id,
                "server_id": entry.server_id,
                "registered": platform_item.get("registered") if platform.get("checked") else None,
                "enabled": platform_item.get("enabled") if platform.get("checked") else None,
                "route_back_health": route_ok,
                "operational": operational,
                "health": _sanitized_health(health_item),
                "remaining_work": remaining,
                "setup_actions": setup_actions,
            }
        )
    return items


def _operational_ready(
    *,
    entry: contract.McpEntry,
    health_item: dict[str, Any] | None,
    route_ok: bool | None,
) -> bool | None:
    if route_ok is None:
        return None
    if not route_ok:
        return False
    parsed = health_item.get("parsed_stdout") if isinstance(health_item, dict) else None
    if entry.capability_id == "mcp.telegram-bot" and isinstance(parsed, dict):
        return bool(parsed.get("ready_for_send"))
    if (
        entry.capability_id == "mcp.ai-debate"
        and isinstance(parsed, dict)
        and parsed.get("ok")
        and parsed.get("authentication_checked") is False
    ):
        return None
    if isinstance(parsed, dict) and "ok" in parsed:
        return bool(parsed.get("ok"))
    return True


def _server_actions(
    *,
    entry: contract.McpEntry,
    platform_item: dict[str, Any],
    route_ok: bool | None,
    operational: bool | None,
    health_item: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    work: list[str] = []
    setup: list[str] = []
    registration_inspection_ok = platform_item.get("registration_inspection_ok", True)
    if (
        platform_item
        and not platform_item.get("skipped")
        and registration_inspection_ok
        and not platform_item.get("registered")
    ):
        work.append(f"Refresh the target-platform installation until {entry.server_id} is registered.")
    elif platform_item.get("registered") and platform_item.get("enabled") is False:
        setup.append(f"Enable {entry.server_id} when this packaged capability should be active.")
    if route_ok is False:
        work.append(f"Fix route-back health for {entry.server_id}.")
    parsed = health_item.get("parsed_stdout") if isinstance(health_item, dict) else None
    if entry.capability_id == "mcp.telegram-bot" and isinstance(parsed, dict) and not parsed.get("ready_for_send"):
        if not parsed.get("token_configured"):
            setup.append("Configure the Telegram bot token in TELEGRAM_BOT_TOKEN or .tokimon/secrets.local.json.")
        if parsed.get("token_configured") and not parsed.get("default_chat_configured"):
            setup.append("Configure a Telegram default chat or allowed chat before relying on default sends.")
    elif operational is False:
        setup.append(_operational_setup_action(entry=entry, health_item=health_item))
    return work, setup


def _operational_setup_action(
    *,
    entry: contract.McpEntry,
    health_item: dict[str, Any] | None,
) -> str:
    parsed = health_item.get("parsed_stdout") if isinstance(health_item, dict) else None
    if entry.capability_id == "mcp.ai-debate" and isinstance(parsed, dict):
        if parsed.get("settings_error"):
            return "Configure AI Debate settings before using tokimon-ai-debate."
        providers = parsed.get("providers")
        if isinstance(providers, dict):
            required_providers = parsed.get("required_providers")
            provider_names = (
                list(dict.fromkeys(required_providers))
                if isinstance(required_providers, list)
                and all(isinstance(provider, str) for provider in required_providers)
                else [str(provider) for provider in providers]
            )
            missing = sorted(
                provider
                for provider in provider_names
                if not isinstance(providers.get(provider), dict)
                or not providers[provider].get("available")
            )
            if missing:
                return f"Required AI Debate provider CLI(s) unavailable: {', '.join(missing)}. Configure an existing provider or leave this capability inactive."
    if entry.capability_id == "mcp.remote-browser-broker" and isinstance(parsed, dict):
        docker = parsed.get("docker")
        if isinstance(docker, dict) and not docker.get("path"):
            return "Docker is unavailable. Disable Docker features and regenerate, or configure an existing Docker runtime."
    return f"Configure external prerequisites before using {entry.server_id}."


def _pair_readiness(
    *,
    config: dict[str, Any],
    source_root: Path,
    target_platform: str,
    executable_resolver: ExecutableResolver,
) -> dict[str, Any]:
    if not config.get("ok"):
        return {
            "ok": False,
            "configured": False,
            "remaining_work": ["Configure Pair settings in .tokimon/settings.local.json."],
        }
    pair = config["pair"]
    access = pair["access"]
    peer_b = pair["peer_b"]
    cwd = Path(pair["cwd"]).expanduser().resolve(strict=False)
    provider_commands: dict[str, dict[str, Any]] = {}
    native_subagents: dict[str, Any] | None = None
    if access == "local-cli":
        for agent in sorted({"codex", peer_b}):
            resolved = executable_resolver(agent)
            provider_commands[agent] = {
                "available": bool(resolved),
                "path": resolved,
                "authentication": "not_checked",
            }
    elif access == "native-subagent":
        native_subagents = {
            "target_platform_supported": target_platform == "codex",
            "spawn_check": "not_run",
            "reason": "Post-install readiness cannot create native subagents.",
        }
    missing = [agent for agent, item in provider_commands.items() if not item["available"]]
    remaining = [f"Required {agent} CLI unavailable for Pair local-cli mode; choose an available placement." for agent in missing]
    if native_subagents is not None and not native_subagents["target_platform_supported"]:
        remaining.append(
            "Set pair.access to local-cli for the Claude target platform; "
            "native-subagent is a Codex placement."
        )
    return {
        "ok": not remaining and cwd == source_root.expanduser().resolve(strict=False),
        "configured": True,
        "access": access,
        "peer_b": peer_b,
        "cwd": str(cwd),
        "cwd_matches_live_checkout": cwd == source_root.expanduser().resolve(strict=False),
        "provider_commands": provider_commands,
        "native_subagents": native_subagents,
        "smoke_check": {
            "status": "not_run",
            "reason": "Post-install readiness does not run model calls by default.",
        },
        "remaining_work": remaining,
    }


def _remaining_work(
    *,
    config: dict[str, Any],
    package: dict[str, Any],
    platform: dict[str, Any],
    servers: list[dict[str, Any]],
    pair: dict[str, Any],
) -> list[str]:
    work: list[str] = []
    if not config.get("ok"):
        work.append("Run AI Harness configuration with --yes or explicit AI Debate and Pair choices.")
    if not package.get("ok"):
        work.append("Generate and validate the AI Harness package before installing it.")
    if platform.get("checked") and not platform.get("ok"):
        plugin = platform.get("plugin", {})
        inventory_inspection = plugin.get("inventory_inspection", {})
        if plugin.get("skipped"):
            work.append(
                "Install the Claude Code CLI so post-install can inspect the installed plugin, "
                "or unset TOKIMON_REQUIRE_CLAUDE_CLI."
            )
        elif isinstance(inventory_inspection, dict) and not inventory_inspection.get("ok", True):
            work.append(str(inventory_inspection["action"]))
        else:
            work.append("Refresh the target-platform installation, then start a new target-platform thread.")
    for server in servers:
        work.extend(str(item) for item in server.get("remaining_work", []))
    work.extend(str(item) for item in pair.get("remaining_work", []))
    return _dedupe(work)


def _capability_setup(*, servers: list[dict[str, Any]]) -> list[str]:
    setup: list[str] = []
    for server in servers:
        setup.extend(str(item) for item in server.get("setup_actions", []))
    return _dedupe(setup)


def _next_steps(*, remaining_work: list[str]) -> list[str]:
    if remaining_work:
        return ["Resolve blocking remaining work, then rerun post-install."]
    return [
        "Start a new target-platform thread after reinstall so plugin skills and MCP entries refresh.",
        "Use Tokimon skills and the packaged MCP entries from the installed plugin.",
    ]


def _summary(
    *,
    remaining_work: list[str],
    capability_setup: list[str],
    platform: dict[str, Any],
    servers: list[dict[str, Any]],
    run_health: bool,
) -> dict[str, Any]:
    registered_count = sum(1 for item in servers if item.get("registered") is True)
    enabled_count = sum(1 for item in servers if item.get("enabled") is True)
    health_count = sum(1 for item in servers if item.get("route_back_health") is True)
    operational_count = sum(1 for item in servers if item.get("operational") is True)
    operational_checked = run_health and all(
        item.get("operational") is not None for item in servers
    )
    total = len(servers)
    registration_checked = bool(platform.get("checked")) and bool(servers) and all(
        item.get("registered") is not None for item in servers
    )
    return {
        "status": "ready" if not remaining_work else "action_required",
        "remaining_work_count": len(remaining_work),
        "capability_setup_count": len(capability_setup),
        "plugin_installed_enabled": platform.get("plugin", {}).get("installed_enabled"),
        "mcp_registration": {
            "checked": registration_checked,
            "registered": registered_count,
            "enabled": enabled_count,
            "total": total,
        },
        "mcp_route_back_health": {
            "checked": run_health,
            "pass": health_count,
            "total": total,
        },
        "mcp_operational": {
            "checked": operational_checked,
            "ready": operational_count,
            "total": total,
        },
    }


def _sanitized_health(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not item:
        return None
    return {
        "ok": item.get("ok"),
        "status": item.get("status"),
        "returncode": item.get("returncode"),
        "command": item.get("command"),
        "parsed_stdout": item.get("parsed_stdout"),
    }


def _run_command(command: Sequence[str], cwd: Path, timeout_sec: int) -> CommandResult:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(returncode=127, stdout="", stderr=str(exc))
    return CommandResult(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def _matching_line(text: str, needle: str) -> str | None:
    for line in text.splitlines():
        if needle in line:
            return " ".join(line.split())
    return None


def _line_value(text: str, key: str) -> str | None:
    prefix = f"{key}:"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            return stripped.split(":", 1)[1].strip()
    return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
