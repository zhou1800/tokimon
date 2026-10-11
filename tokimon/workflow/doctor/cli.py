"""Operator CLI for the Tokimon doctor workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tokimon.workflow.doctor import checks, contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokimon doctor repository health checks.")
    parser.add_argument("--workspace-root", default=str(checks.WORKSPACE_ROOT), help="Tokimon checkout root to inspect")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    subparsers = parser.add_subparsers(dest="command")

    check_parser = subparsers.add_parser("check", help="Run read-only repository health checks")
    check_parser.add_argument("--workspace-root", default=argparse.SUPPRESS, help="Tokimon checkout root to inspect")
    check_parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS)

    contract_parser = subparsers.add_parser("contract", help="Emit workflow contract declarations")
    contract_parser.add_argument("--workspace-root", default=argparse.SUPPRESS, help="Tokimon checkout root to inspect")
    contract_parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS)

    return parser


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _contract_payload() -> dict[str, Any]:
    return {
        "workflow_id": contract.WORKFLOW_ID,
        "python_package": contract.PYTHON_PACKAGE,
        "workflow_status": contract.WORKFLOW_STATUS,
        "canonical_entrypoint": contract.CANONICAL_ENTRYPOINT,
        "operator_entrypoint": contract.OPERATOR_ENTRYPOINT,
        "local_state_root": str(contract.LOCAL_STATE_ROOT),
        "memory_scope": contract.MEMORY_SCOPE,
        "memory_room": contract.MEMORY_ROOM,
        "default_memory_db": str(contract.DEFAULT_MEMORY_DB),
        "required_tool_capabilities": list(contract.REQUIRED_TOOL_CAPABILITIES),
        "commands": list(contract.COMMANDS),
        "status_values": list(contract.STATUS_VALUES),
        "readiness_status_values": list(contract.READINESS_STATUS_VALUES),
        "workflow_readiness": contract.WORKFLOW_READINESS,
    }


def _render_contract_text(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            f"workflow_id: {payload['workflow_id']}",
            f"status: {payload['workflow_status']}",
            f"entrypoint: {payload['operator_entrypoint']}",
            "required_tool_capabilities: none",
        )
    )


def _render_item_lines(items: list[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in items:
        lines.append(f"- {item['status']} {item['id']}: {item['summary']}")
        for reason in item.get("reasons", []):
            lines.append(f"  {reason}")
    return lines


def _render_check_text(payload: Mapping[str, Any]) -> str:
    readiness_summary = payload["summary"]["readiness"]
    check_summary = payload["summary"]["checks"]
    memory = payload["memory"]
    memory_data = memory.get("data", {})
    tools = payload["tools"]
    lines = [
        f"Tokimon doctor: {payload['status']}",
        "readiness: "
        f"{readiness_summary['ready']} ready, "
        f"{readiness_summary['partial']} partial, "
        f"{readiness_summary['blocked']} blocked",
        "foundation checks: "
        f"{check_summary['pass']} pass, "
        f"{check_summary['warn']} warn, "
        f"{check_summary['fail']} fail",
        "",
        "Memory:",
        f"- {memory['status']} {memory['id']}: {memory['summary']}",
    ]
    if memory_data:
        lines.append(
            "  "
            f"schema={memory_data.get('schema_version', 'unknown')}, "
            f"entities={memory_data.get('entities', 'unknown')}, "
            f"relationships={memory_data.get('relationships', 'unknown')}, "
            f"db_bytes={memory_data.get('database_size_bytes', 'unknown')}"
        )

    lines.extend(
        (
            "",
            "Tools:",
            f"- host: {tools.get('host', {}).get('host_id', 'unknown')} "
            f"({tools.get('host', {}).get('os_name', 'unknown')})",
        )
    )
    lines.extend(_render_item_lines(tools.get("items", [])))

    lines.extend(("", "Workflows:"))
    lines.extend(_render_item_lines(payload.get("workflows", [])))

    failed_or_warned_checks = [
        check for check in payload["checks"] if check.get("status") in {"warn", "fail"}
    ]
    if failed_or_warned_checks:
        lines.extend(("", "Foundation Diagnostics:"))
        for check in failed_or_warned_checks:
            lines.append(f"- {check['status']} {check['id']}: {check['summary']}")
            for detail in check.get("details", []):
                lines.append(f"  {detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "check"
    workspace_root = Path(args.workspace_root)

    if command == "contract":
        payload = {"ok": True, "contract": _contract_payload()}
        if args.json:
            _print_json(payload)
        else:
            print(_render_contract_text(payload["contract"]))
        return 0

    if command == "check":
        payload = checks.run_doctor(workspace_root=workspace_root)
        if args.json:
            _print_json(payload)
        else:
            print(_render_check_text(payload))
        return 0 if payload["ok"] else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
