"""Diagnostics and handoff facts for AI Harness Interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tokimon.ai_harness_interface import configuration, contract, generator, installation_policy, mcp, route


def handoff_facts(*, paths: generator.GenerationPaths, install_route: route.InstallRoute) -> dict[str, Any]:
    return {
        "target_platform": paths.platform.platform_id,
        "package_id": contract.PACKAGE_ID,
        "marketplace_name": contract.MARKETPLACE_NAME,
        "package_root": str(paths.package_root),
        "marketplace_path": str(paths.marketplace_path),
        "install_route": install_route.to_json_object(),
        "account_level_install_owner": paths.platform.account_level_install_owner,
        "account_level_mutation": "not_performed_by_ai_harness_interface",
    }


def diagnose(
    *,
    paths: generator.GenerationPaths,
    run_health: bool = True,
    settings_path: Path | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    install_route = generator.install_route_for(paths)
    handoff = handoff_facts(paths=paths, install_route=install_route)

    try:
        configuration.require_configured_settings(
            source_root=paths.source_root,
            settings_path=settings_path,
        )
        checks.append(
            {
                "name": "configuration",
                "ok": True,
            }
        )
    except configuration.ConfigurationError as exc:
        checks.append({"name": "configuration", "ok": False, "message": str(exc)})

    try:
        route.validate_route(install_route)
        checks.append(
            {
                "name": "install-route",
                "ok": True,
                "liveCheckoutPath": str(install_route.liveCheckoutPath),
                "runtimeRoot": str(install_route.runtimeRoot),
            }
        )
    except route.RouteError as exc:
        checks.append({"name": "install-route", "ok": False, "message": str(exc)})
        return {"ok": False, "checks": checks, "handoff": handoff}

    try:
        package_validation = generator.validate_generated_package(
            paths,
            install_route=install_route,
        )
        checks.append({"name": "generated-package", "ok": True, "details": package_validation})
    except generator.PackageValidationError as exc:
        checks.append({"name": "generated-package", "ok": False, "message": str(exc)})

    try:
        policy = generator.load_installation_policy(paths)
    except generator.PackageValidationError:
        policy = installation_policy.InstallationPolicy("disabled", False)
    handoff["installation_policy"] = policy.to_json()
    if run_health:
        health = mcp.run_health_checks(install_route, entries=policy.mcp_entries)
        checks.append({"name": "mcp-route-back-health", "ok": all(item["ok"] for item in health), "items": health})

    return {"ok": all(check["ok"] for check in checks), "checks": checks, "handoff": handoff}
