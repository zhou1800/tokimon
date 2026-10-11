"""Operator CLI for AI Harness Interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tokimon import runtime
from tokimon.ai_harness_interface import configuration, contract, diagnostics, generator, readiness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and validate AI Harness Interface packages.")
    parser.add_argument("--workspace-root", default=str(runtime.source_root()), help="Live Tokimon checkout root")
    parser.add_argument(
        "--output-root",
        help="Root that receives generated package and marketplace output; defaults to .tokimon/ai-harness-interface/<target-platform>",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract_parser = subparsers.add_parser("contract", help="Emit AI Harness Interface contract")
    _add_target_platform_argument(contract_parser)
    contract_parser.add_argument("--json", action="store_true")

    bootstrap_parser = subparsers.add_parser("bootstrap-codex", help="Bootstrap Codex scaffold output")
    bootstrap_parser.add_argument("--force", action="store_true", help="Replace existing scaffold files")
    bootstrap_parser.add_argument("--json", action="store_true")

    configure_parser = subparsers.add_parser("configure", help="Configure repository-local AI Harness settings")
    configure_parser.add_argument("--pair-access", choices=configuration.PAIR_ACCESS_CHOICES)
    configure_parser.add_argument("--pair-peer-b", choices=configuration.PAIR_PEER_CHOICES)
    configure_parser.add_argument("--docker-features", choices=("auto", "disabled"))
    configure_parser.add_argument(
        "--ai-debate-access",
        choices=configuration.AI_DEBATE_ACCESS_CHOICES,
    )
    configure_parser.add_argument(
        "--ai-debate-peer-b",
        choices=configuration.PEER_CHOICES,
    )
    configure_parser.add_argument("--settings-path", type=Path)
    configure_parser.add_argument("--yes", action="store_true", help="Use existing or template defaults")
    configure_parser.add_argument("--json", action="store_true")

    generate_parser = subparsers.add_parser("generate", help="Generate package output")
    _add_target_platform_argument(generate_parser)
    generate_parser.add_argument("--cachebuster", help="Optional Codex-only cachebuster token for generated plugin version")
    generate_parser.add_argument("--settings-path", type=Path)
    generate_parser.add_argument("--json", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate generated package output")
    _add_target_platform_argument(validate_parser)
    validate_parser.add_argument("--settings-path", type=Path)
    validate_parser.add_argument("--json", action="store_true")

    diagnose_parser = subparsers.add_parser("diagnose", help="Run package and route-back diagnostics")
    _add_target_platform_argument(diagnose_parser)
    diagnose_parser.add_argument("--settings-path", type=Path)
    diagnose_parser.add_argument("--skip-health", action="store_true")
    diagnose_parser.add_argument("--json", action="store_true")

    post_install_parser = subparsers.add_parser("post-install", help="Report post-install readiness and next steps")
    post_install_parser.add_argument("--settings-path", type=Path)
    _add_target_platform_argument(post_install_parser)
    post_install_parser.add_argument("--skip-platform", action="store_true")
    post_install_parser.add_argument("--skip-health", action="store_true")
    post_install_parser.add_argument("--json", action="store_true")

    return parser


def _add_target_platform_argument(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--target-platform",
        choices=sorted(contract.TARGET_PLATFORMS),
        default=contract.TARGET_PLATFORM,
        help="Target AI platform for generated package output",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace_root = Path(args.workspace_root)
    output_root = Path(args.output_root) if args.output_root else None
    target_platform = getattr(args, "target_platform", contract.TARGET_PLATFORM)
    if args.command == "bootstrap-codex":
        target_platform = "codex"
    paths = generator.paths_for(
        source_root=workspace_root,
        output_root=output_root,
        target_platform=target_platform,
    )

    if args.command == "contract":
        return _emit(
            {
                "ok": True,
                "contract": contract.contract_payload(
                    workspace_root=workspace_root,
                    target_platform=target_platform,
                ),
            },
            as_json=args.json,
            text_renderer=_render_contract_text,
        )

    if args.command == "bootstrap-codex":
        payload = generator.bootstrap_scaffold(paths, force=args.force)
        return _emit(payload, as_json=args.json, text_renderer=_render_bootstrap_text, fail_code=2)

    if args.command == "configure":
        try:
            payload = configuration.configure_settings(
                source_root=workspace_root,
                settings_path=args.settings_path,
                ai_debate_access=args.ai_debate_access,
                ai_debate_peer_b=args.ai_debate_peer_b,
                pair_access=args.pair_access,
                pair_peer_b=args.pair_peer_b,
                docker_features=args.docker_features,
                assume_yes=args.yes,
            )
        except configuration.ConfigurationError as exc:
            payload = {"ok": False, "message": str(exc)}
        return _emit(payload, as_json=args.json, text_renderer=_render_configure_text)

    if args.command == "generate":
        try:
            payload = generator.generate_package(
                paths,
                cachebuster=args.cachebuster,
                settings_path=args.settings_path,
            )
        except (configuration.ConfigurationError, generator.PackageValidationError) as exc:
            payload = {"ok": False, "message": str(exc)}
        return _emit(payload, as_json=args.json, text_renderer=_render_generate_text)

    if args.command == "validate":
        try:
            configuration.require_configured_settings(
                source_root=workspace_root,
                settings_path=args.settings_path,
            )
            payload = generator.validate_generated_package(paths)
        except (configuration.ConfigurationError, generator.PackageValidationError) as exc:
            payload = {"ok": False, "message": str(exc)}
        return _emit(payload, as_json=args.json, text_renderer=_render_validate_text, fail_code=2)

    if args.command == "diagnose":
        payload = diagnostics.diagnose(
            paths=paths,
            run_health=not args.skip_health,
            settings_path=args.settings_path,
        )
        return _emit(payload, as_json=args.json, text_renderer=_render_diagnose_text, fail_code=2)

    if args.command == "post-install":
        payload = readiness.post_install_readiness(
            paths=paths,
            settings_path=args.settings_path,
            target_platform=args.target_platform,
            run_health=not args.skip_health,
            check_platform=not args.skip_platform,
        )
        return _emit(payload, as_json=args.json, text_renderer=_render_post_install_text, fail_code=2)

    return 1


def _emit(
    payload: Mapping[str, Any],
    *,
    as_json: bool,
    text_renderer,
    fail_code: int = 1,
) -> int:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(text_renderer(payload))
    return 0 if payload.get("ok") else fail_code


def _render_contract_text(payload: Mapping[str, Any]) -> str:
    data = payload["contract"]
    return "\n".join(
        (
            f"capability_id: {data['capability_id']}",
            f"target_platform: {data['target_platform']}",
            f"package_id: {data['package_id']}",
            f"marketplace_name: {data['marketplace_name']}",
            f"skills: {', '.join(data['package_skill_ids'])}",
        )
    )


def _render_generate_text(payload: Mapping[str, Any]) -> str:
    if not payload.get("ok"):
        return f"AI Harness package generation failed: {payload.get('message', 'see JSON output')}"
    return "\n".join(
        (
            f"Generated AI Harness package: {payload['package_root']}",
            f"Marketplace: {payload['marketplace_path']}",
            f"Version: {payload['version']}",
        )
    )


def _render_bootstrap_text(payload: Mapping[str, Any]) -> str:
    if not payload.get("ok"):
        return f"AI Harness Codex bootstrap failed: {payload.get('message', 'see JSON output')}"
    return f"AI Harness Codex scaffold ready: {payload.get('stdout', '').strip()}"


def _render_configure_text(payload: Mapping[str, Any]) -> str:
    if not payload.get("ok"):
        return f"AI Harness configuration failed: {payload.get('message', 'see JSON output')}"
    settings = payload["settings"]
    return "\n".join(
        (
            f"AI Harness configuration written: {payload['settings_path']}",
            (
                "AI Debate: "
                f"{settings['ai_debate']['access']} / {settings['ai_debate']['peer_b']}"
            ),
            f"Pair: {settings['pair']['access']} / {settings['pair']['peer_b']}",
        )
    )


def _render_validate_text(payload: Mapping[str, Any]) -> str:
    if not payload.get("ok"):
        return f"AI Harness package validation failed: {payload.get('message', 'see JSON output')}"
    return f"AI Harness package validation passed: {payload['package_root']}"


def _render_diagnose_text(payload: Mapping[str, Any]) -> str:
    status = "passed" if payload.get("ok") else "failed"
    lines = [f"AI Harness diagnostics {status}"]
    for check in payload.get("checks", []):
        lines.append(f"- {'pass' if check.get('ok') else 'fail'} {check.get('name')}")
    return "\n".join(lines)


def _render_post_install_text(payload: Mapping[str, Any]) -> str:
    status = "ready" if payload.get("ok") else "action required"
    summary = payload.get("summary", {})
    registration = summary.get("mcp_registration", {})
    route_health = summary.get("mcp_route_back_health", {})
    operational = summary.get("mcp_operational", {})
    pair = payload.get("pair", {})
    registration_text = "not checked"
    if registration.get("checked"):
        registration_text = (
            f"{registration.get('registered', 0)}/{registration.get('total', 0)} registered; "
            f"{registration.get('enabled', 0)}/{registration.get('total', 0)} enabled"
        )
    route_health_text = "not checked"
    if route_health.get("checked"):
        route_health_text = f"{route_health.get('pass', 0)}/{route_health.get('total', 0)} pass"
    operational_text = "not checked"
    if operational.get("checked"):
        operational_text = f"{operational.get('ready', 0)}/{operational.get('total', 0)} ready"
    lines = [
        f"Tokimon post-install readiness: {status}",
        f"- plugin: {_status_word(summary.get('plugin_installed_enabled'))}",
        f"- package: {_status_word(payload.get('package', {}).get('ok'))}",
        f"- configuration: {_status_word(payload.get('configuration', {}).get('ok'))}",
        f"- MCP registration: {registration_text}",
        f"- MCP route-back health: {route_health_text}",
        f"- operational readiness: {operational_text}",
    ]
    if pair.get("configured"):
        lines.append(f"- Pair: {pair.get('access')} / {pair.get('peer_b')}")
    else:
        lines.append("- Pair: not configured")
    policy = payload.get("installation_policy", {})
    if policy.get("disabled_capabilities"):
        lines.append("- disabled optional capabilities: " + ", ".join(policy["disabled_capabilities"]))
    remaining_work = payload.get("remaining_work") or []
    if remaining_work:
        lines.extend(("", "Remaining work:"))
        for index, step in enumerate(remaining_work, start=1):
            lines.append(f"{index}. {step}")
    next_steps = payload.get("next_steps") or []
    if next_steps:
        lines.extend(("", "Next steps:"))
        for index, step in enumerate(next_steps, start=1):
            lines.append(f"{index}. {step}")
    capability_setup = payload.get("capability_setup") or []
    if capability_setup:
        lines.append("")
        lines.append("Capability setup (non-blocking):")
        for index, step in enumerate(capability_setup, start=1):
            lines.append(f"{index}. {step}")
    return "\n".join(lines)


def _status_word(value: Any) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "not checked"
