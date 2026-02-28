"""CLI entrypoint for Tokimon."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import sys
import time
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from chat_ui.server import ChatUIConfig, run_chat_ui
from gateway import health_client as gateway_health_client
from gateway.health_client import call_gateway_rpc, check_gateway_health
from gateway.server import GatewayConfig, run_gateway
from benchmarks.harness import EvaluationHarness
from llm.client import (
    ClaudeCLIClient,
    ClaudeCLISettings,
    CodexCLIClient,
    CodexCLISettings,
    LLMClient,
    MockLLMClient,
    build_llm_client,
)
from memory.store import MemoryStore
from runners.baseline import BaselineRunner
from runners.hierarchical import HierarchicalRunner
from self_improve.orchestrator import SelfImproveOrchestrator, SelfImproveSettings
from self_improve.provider_mix import mixed_provider_for_session, validate_mixed_sessions_per_batch
from skills.builder import SkillBuilder
from skills.registry import SkillRegistry
from skills.spec import SkillSpec


def build_parser(*, exit_on_error: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokimon", exit_on_error=exit_on_error)
    subparsers = parser.add_subparsers(dest="command")

    auto = subparsers.add_parser("auto")
    auto.add_argument("prompt")

    run_task = subparsers.add_parser("run-task")
    run_task.add_argument("--task-id", required=True)
    run_task.add_argument("--runner", choices=["baseline", "hierarchical"], default="hierarchical", help=argparse.SUPPRESS)

    subparsers.add_parser("run-suite")

    resume = subparsers.add_parser("resume-run")
    resume.add_argument("--run-path", required=True)

    inspect_run = subparsers.add_parser("inspect-run")
    inspect_run.add_argument("--run-path", required=True)

    sessions = subparsers.add_parser("sessions")
    sessions.add_argument("--root", default=None, help="Self-improve runs root directory (default: <workspace>/runs/self-improve).")
    sessions.add_argument("--active", type=int, default=None, help="Only include runs modified within the last N minutes.")
    sessions.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    subparsers.add_parser("list-skills")

    build_skill = subparsers.add_parser("build-skill")
    build_skill.add_argument("--name", required=True)
    build_skill.add_argument("--purpose", required=True)
    build_skill.add_argument("--contract", required=True)
    build_skill.add_argument("--tools", default="")
    build_skill.add_argument("--justification", required=True)

    self_improve = subparsers.add_parser("self-improve")
    self_improve.add_argument("--goal", default="Improve tokimon based on docs and failing tests.")
    self_improve.add_argument("--input", default=None)
    self_improve.add_argument("--sessions", type=int, default=5, help=argparse.SUPPRESS)
    self_improve.add_argument("--batches", type=int, default=1, help=argparse.SUPPRESS)
    self_improve.add_argument("--workers", type=int, default=4, help=argparse.SUPPRESS)
    self_improve.add_argument("--no-merge", action="store_true", help=argparse.SUPPRESS)
    self_improve.add_argument(
        "--llm",
        choices=["mock", "codex", "claude", "mixed"],
        default=os.environ.get("TOKIMON_LLM", "mixed"),
        help=argparse.SUPPRESS,
    )

    chat_ui = subparsers.add_parser("chat-ui")
    chat_ui.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    chat_ui.add_argument("--port", type=int, default=8765)
    chat_ui.add_argument(
        "--llm",
        choices=["mock", "codex", "claude"],
        default=os.environ.get("TOKIMON_LLM", "mock"),
        help=argparse.SUPPRESS,
    )
    chat_ui.add_argument("--workspace", default=None, help=argparse.SUPPRESS)

    health_common = argparse.ArgumentParser(add_help=False)
    health_common.add_argument("--url", default="ws://127.0.0.1:8765/gateway", help="Gateway WebSocket URL.")
    health_common.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    health_common.add_argument("--verbose", action="store_true", help="Print additional diagnostics.")
    health_common.add_argument("--timeout-ms", type=int, default=2_000, help="Overall timeout in milliseconds.")

    gateway_query_common = argparse.ArgumentParser(add_help=False)
    gateway_query_common.add_argument("--url", default="ws://127.0.0.1:8765/gateway", help="Gateway WebSocket URL.")
    gateway_query_common.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    gateway_query_common.add_argument("--timeout-ms", type=int, default=2_000, help="Overall timeout in milliseconds.")

    gateway_run_common = argparse.ArgumentParser(add_help=False)
    gateway_run_common.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    gateway_run_common.add_argument("--port", type=int, default=8765)
    gateway_run_common.add_argument(
        "--llm",
        choices=["mock", "codex", "claude"],
        default=os.environ.get("TOKIMON_LLM", "mock"),
        help=argparse.SUPPRESS,
    )
    gateway_run_common.add_argument("--workspace", default=None, help=argparse.SUPPRESS)

    gateway = subparsers.add_parser("gateway", parents=[gateway_run_common])
    gateway_sub = gateway.add_subparsers(dest="gateway_command")
    gateway_sub.add_parser("run", parents=[gateway_run_common])
    gateway_sub.add_parser("health", parents=[health_common])

    gateway_call = gateway_sub.add_parser("call", parents=[gateway_query_common])
    gateway_call.add_argument("method", help="Gateway WebSocket RPC method.")
    gateway_call.add_argument("--params", default=None, help="RPC params as a JSON object (default: {}).")

    gateway_sub.add_parser("probe", parents=[gateway_query_common])

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    doctor.add_argument("--repair", "--fix", dest="repair", action="store_true", help="Attempt safe, non-destructive repairs.")

    subparsers.add_parser("health", parents=[health_common])

    logs = subparsers.add_parser("logs")
    logs.add_argument("--url", default="ws://127.0.0.1:8765/gateway", help="Gateway WebSocket URL.")
    logs.add_argument("--follow", action="store_true", help="Keep tailing logs until interrupted.")
    logs.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    logs.add_argument("--limit", type=int, default=200, help="Maximum number of log entries to return.")
    logs.add_argument("--local-time", action="store_true", help="Render timestamps in your local timezone.")

    memory = subparsers.add_parser("memory")
    memory_sub = memory.add_subparsers(dest="memory_command")

    memory_common = argparse.ArgumentParser(add_help=False)
    memory_common.add_argument("--root", default=None, help="Memory store root directory (default: <workspace>/memory).")
    memory_common.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    memory_common.add_argument("--verbose", action="store_true", help="Print additional diagnostics.")

    memory_status = memory_sub.add_parser("status", parents=[memory_common])
    memory_status.add_argument("--deep", action="store_true", help="Include additional index/file reconciliation details.")
    memory_status.add_argument("--index", action="store_true", help="Reindex when the store is dirty.")

    memory_sub.add_parser("index", parents=[memory_common])

    memory_search = memory_sub.add_parser("search", parents=[memory_common])
    memory_search.add_argument("query", nargs="?", help="Search query text.")
    memory_search.add_argument("--query", dest="query_flag", default=None, help="Search query text (wins over positional).")
    memory_search.add_argument("--limit", type=int, default=5, help="Maximum number of hits to return.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    repo_root = Path(__file__).resolve().parent
    workspace_root = repo_root.parent
    runs_root = workspace_root / "runs"
    llm_client = MockLLMClient(script=[])

    match args.command:
        case "auto":
            return _cmd_auto(parser, str(args.prompt))
        case "run-task":
            return _cmd_run_task(args, repo_root, runs_root, llm_client)
        case "run-suite":
            return _cmd_run_suite(repo_root, runs_root)
        case "resume-run":
            return _cmd_resume_run(args, repo_root, runs_root, llm_client)
        case "inspect-run":
            return _cmd_inspect_run(args)
        case "list-skills":
            return _cmd_list_skills(repo_root)
        case "build-skill":
            return _cmd_build_skill(args, repo_root)
        case "self-improve":
            return _cmd_self_improve(args)
        case "chat-ui":
            return _cmd_chat_ui(args)
        case "gateway":
            return _cmd_gateway(args)
        case "doctor":
            return _cmd_doctor(args, workspace_root)
        case "health":
            return _cmd_health(args)
        case "logs":
            return _cmd_logs(args)
        case "sessions":
            return _cmd_sessions(args, workspace_root)
        case "memory":
            return _cmd_memory(args, workspace_root)
        case _:
            return 1

    return 1


def _cmd_auto(parser: argparse.ArgumentParser, prompt: str) -> int:
    routed_argv = _auto_decide_argv(prompt)
    if not routed_argv:
        parser.print_help()
        return 0
    return main(routed_argv)


def _cmd_run_task(args: argparse.Namespace, repo_root: Path, runs_root: Path, llm_client: LLMClient) -> int:
    task_dir = _find_task_dir(repo_root, args.task_id)
    if task_dir is None:
        raise SystemExit(f"Unknown task: {args.task_id}")
    spec_path = task_dir / "task.json"
    spec = json.loads(spec_path.read_text())
    workspace = runs_root / "workspaces" / args.task_id / args.runner
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(task_dir / "starter", workspace)
    tests_dst = workspace / "tests"
    shutil.copytree(task_dir / "tests", tests_dst)
    test_args = [str(tests_dst)]
    if args.runner == "baseline":
        runner = BaselineRunner(workspace, llm_client, base_dir=runs_root)
        runner.run(spec.get("description", ""), task_id=args.task_id, test_args=test_args)
    else:
        runner = HierarchicalRunner(workspace, llm_client, base_dir=runs_root)
        runner.run(spec.get("description", ""), task_steps=None, task_id=args.task_id, test_args=test_args)
    return 0


def _cmd_run_suite(repo_root: Path, runs_root: Path) -> int:
    harness = EvaluationHarness(repo_root, runs_dir=runs_root)
    harness.run_suite()
    return 0


def _cmd_resume_run(args: argparse.Namespace, repo_root: Path, runs_root: Path, llm_client: LLMClient) -> int:
    run_path = Path(args.run_path)
    workflow_state = run_path / "workflow_state.json"
    if not workflow_state.exists():
        raise SystemExit("workflow_state.json not found")
    runner = HierarchicalRunner(repo_root, llm_client, base_dir=runs_root)
    runner.resume(run_path)
    return 0


def _cmd_inspect_run(args: argparse.Namespace) -> int:
    run_path = Path(args.run_path)
    run_manifest = run_path / "run.json"
    workflow_state = run_path / "workflow_state.json"
    if run_manifest.exists():
        _write_line(sys.stdout, run_manifest.read_text())
    if workflow_state.exists():
        _write_line(sys.stdout, workflow_state.read_text())
    return 0


def _cmd_list_skills(repo_root: Path) -> int:
    registry = SkillRegistry(repo_root)
    registry.load()
    for spec in registry.list_skills():
        _write_line(sys.stdout, f"{spec.name}: {spec.purpose}")
    return 0


def _cmd_build_skill(args: argparse.Namespace, repo_root: Path) -> int:
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    spec = SkillSpec(
        name=args.name,
        purpose=args.purpose,
        contract=args.contract,
        required_tools=tools,
        retrieval_prefs={"stage1": "tight", "stage2": "broaden", "stage3": "targeted"},
    )
    builder = SkillBuilder(repo_root, MemoryStore(repo_root / "memory"))
    ok = builder.build_skill(spec, args.justification)
    return 0 if ok else 1


def _cmd_self_improve(args: argparse.Namespace) -> int:
    master_root = Path.cwd().resolve()
    settings = SelfImproveSettings(
        sessions_per_batch=args.sessions,
        batches=args.batches,
        max_workers=args.workers,
        merge_on_success=not args.no_merge,
    )
    llm_provider = str(args.llm or "mock").strip().lower()
    if llm_provider == "mixed":
        validate_mixed_sessions_per_batch(int(args.sessions))

    def llm_factory(_session_id: str, workspace_dir: Path):
        session_provider = llm_provider
        if llm_provider == "mixed":
            session_provider = mixed_provider_for_session(_session_id)

        if session_provider in {"codex", "codex-cli"}:
            codex_settings = CodexCLISettings.from_env()
            if "TOKIMON_CODEX_SANDBOX" not in os.environ:
                codex_settings = replace(codex_settings, sandbox="workspace-write")
            if "TOKIMON_CODEX_APPROVAL" not in os.environ:
                codex_settings = replace(codex_settings, ask_for_approval="never")
            if "TOKIMON_CODEX_SEARCH" not in os.environ:
                codex_settings = replace(codex_settings, search=True)
            if "TOKIMON_CODEX_TIMEOUT_S" not in os.environ:
                codex_settings = replace(codex_settings, timeout_s=240)
            return CodexCLIClient(workspace_dir, settings=codex_settings)

        if session_provider in {"claude", "claude-cli"}:
            claude_settings = ClaudeCLISettings.from_env()
            if "TOKIMON_CLAUDE_TIMEOUT_S" not in os.environ:
                claude_settings = replace(claude_settings, timeout_s=240)
            return ClaudeCLIClient(workspace_dir, settings=claude_settings)

        return build_llm_client(session_provider, workspace_dir=workspace_dir)

    orchestrator = SelfImproveOrchestrator(master_root, llm_factory=llm_factory, settings=settings)
    try:
        report = orchestrator.run(args.goal, input_ref=args.input)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    _write_line(sys.stdout, json.dumps({"run_root": report.run_root}, indent=2))
    return 0


def _cmd_chat_ui(args: argparse.Namespace) -> int:
    workspace_dir = Path(args.workspace).resolve() if args.workspace else Path.cwd().resolve()
    config = ChatUIConfig(
        host=str(args.host),
        port=int(args.port),
        llm_provider=str(args.llm),
        workspace_dir=workspace_dir,
    )
    run_chat_ui(config)
    return 0


def _cmd_gateway(args: argparse.Namespace) -> int:
    gateway_command = str(getattr(args, "gateway_command", "") or "").strip().lower()
    if not gateway_command or gateway_command == "run":
        workspace_dir = Path(args.workspace).resolve() if args.workspace else Path.cwd().resolve()
        config = GatewayConfig(
            host=str(args.host),
            port=int(args.port),
            llm_provider=str(args.llm),
            workspace_dir=workspace_dir,
        )
        run_gateway(config)
        return 0

    if gateway_command == "health":
        return _cmd_health(args)

    if gateway_command == "call":
        return _cmd_gateway_call(args)

    if gateway_command == "probe":
        return _cmd_health(args)

    _write_line(sys.stdout, f"error: unknown gateway subcommand: {gateway_command}")
    return 2


def _cmd_gateway_call(args: argparse.Namespace) -> int:
    url = str(getattr(args, "url", "") or "").strip()
    timeout_ms = int(getattr(args, "timeout_ms", 2_000) or 2_000)
    method = str(getattr(args, "method", "") or "").strip()
    params_text = getattr(args, "params", None)

    params: dict[str, object] = {}
    if params_text:
        try:
            parsed = json.loads(str(params_text))
        except Exception as exc:
            return _cmd_gateway_call_error(args, ValueError(f"--params must be valid JSON ({exc.__class__.__name__})"))
        if not isinstance(parsed, dict):
            return _cmd_gateway_call_error(args, ValueError("--params must be a JSON object"))
        params = parsed

    try:
        response = call_gateway_rpc(url, method, params=params, timeout_ms=timeout_ms)
    except Exception as exc:
        return _cmd_gateway_call_error(args, exc)

    _write_line(sys.stdout, json.dumps(response, indent=2, sort_keys=True) if getattr(args, "json", False) else json.dumps(response))
    return 0 if response.get("ok") is True else 1


def _cmd_gateway_call_error(args: argparse.Namespace, exc: BaseException) -> int:
    error = _format_cli_exception(exc)
    if getattr(args, "json", False):
        _write_line(sys.stdout, json.dumps({"ok": False, "error": error}, indent=2, sort_keys=True))
        return 1
    _write_line(sys.stdout, f"error: {error}")
    return 1


def _cmd_doctor(args: argparse.Namespace, workspace_root: Path) -> int:
    from doctor.runner import default_deps, render_human, report_to_json_dict, run_doctor

    deps = default_deps(workspace_root)
    report = run_doctor(deps, repair=bool(getattr(args, "repair", False)))
    if getattr(args, "json", False):
        _write_line(sys.stdout, json.dumps(report_to_json_dict(report), indent=2, sort_keys=True))
        return 0 if report.ok else 1
    sys.stdout.write(render_human(report))
    return 0 if report.ok else 1


def _cmd_health(args: argparse.Namespace) -> int:
    url = str(getattr(args, "url", "") or "").strip()
    timeout_ms = int(getattr(args, "timeout_ms", 2_000) or 2_000)
    verbose = bool(getattr(args, "verbose", False))

    def log(message: str) -> None:
        sys.stderr.write(message)
        sys.stderr.write("\n")

    report = check_gateway_health(url, timeout_ms=timeout_ms, log=log if verbose else None)

    if getattr(args, "json", False):
        _write_line(sys.stdout, json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1

    if report["ok"]:
        _write_line(sys.stdout, f"ok ({report['elapsed_ms']} ms)")
        return 0

    error = report.get("error") or "health check failed"
    _write_line(sys.stdout, f"error: {error}")
    if verbose:
        details = report.get("details")
        if isinstance(details, dict) and details:
            _write_line(sys.stdout, json.dumps(details, indent=2, sort_keys=True))
    return 1


def _cmd_logs(args: argparse.Namespace) -> int:
    url = str(getattr(args, "url", "") or "").strip()
    follow = bool(getattr(args, "follow", False))
    json_output = bool(getattr(args, "json", False))
    limit = int(getattr(args, "limit", 200) or 200)
    local_time = bool(getattr(args, "local_time", False))

    if limit <= 0:
        error = _format_cli_exception(ValueError("--limit must be a positive integer"))
        if json_output:
            _write_line(sys.stdout, json.dumps({"ok": False, "error": error}, indent=2, sort_keys=True))
        else:
            _write_line(sys.stdout, f"error: {error}")
        return 2

    def emit_entry(entry: object) -> None:
        if json_output:
            _write_line(sys.stdout, json.dumps(entry, separators=(",", ":"), ensure_ascii=False))
            return
        _write_line(sys.stdout, _format_log_entry(entry, local_time=local_time))

    if not follow:
        try:
            response = call_gateway_rpc(url, "logs.tail", params={"limit": limit}, timeout_ms=2_000)
            if response.get("ok") is not True:
                raise RuntimeError(str(response.get("error") or "logs.tail failed"))
            payload = response.get("payload")
            if not isinstance(payload, dict):
                raise RuntimeError("invalid logs.tail payload")
            entries = payload.get("entries")
            if not isinstance(entries, list):
                raise RuntimeError("invalid logs.tail entries")

            if json_output:
                _write_line(sys.stdout, json.dumps({"ok": True, "entries": entries}, indent=2, sort_keys=True))
            else:
                for entry in entries:
                    emit_entry(entry)
            return 0
        except Exception as exc:
            error = _format_cli_exception(exc)
            if json_output:
                _write_line(sys.stdout, json.dumps({"ok": False, "error": error}, indent=2, sort_keys=True))
            else:
                _write_line(sys.stdout, f"error: {error}")
            return 1

    after: int | None = None
    try:
        host, port, path = gateway_health_client._parse_gateway_ws_url(url)
        deadline = time.monotonic() + 2.0
        ws = gateway_health_client._ws_connect(host, port, path, deadline=deadline)
    except Exception as exc:
        error = _format_cli_exception(exc)
        if json_output:
            _write_line(sys.stdout, json.dumps({"ok": False, "error": error}, indent=2, sort_keys=True))
        else:
            _write_line(sys.stdout, f"error: {error}")
        return 1

    try:
        gateway_health_client._gateway_ws_handshake(ws, deadline=deadline, log=None)
        req_counter = 2
        while True:
            params: dict[str, object] = {"limit": limit}
            if after is not None:
                params["after"] = after

            call_deadline = time.monotonic() + 2.0
            ws.send_json({"type": "req", "id": str(req_counter), "method": "logs.tail", "params": params}, deadline=call_deadline)
            response = ws.recv_json(deadline=call_deadline)
            if response.get("type") != "res" or response.get("id") != str(req_counter):
                raise RuntimeError("logs.tail rpc call failed")
            if response.get("ok") is not True:
                raise RuntimeError(str(response.get("error") or "logs.tail failed"))

            payload = response.get("payload")
            if not isinstance(payload, dict):
                raise RuntimeError("invalid logs.tail payload")
            entries = payload.get("entries")
            if not isinstance(entries, list):
                raise RuntimeError("invalid logs.tail entries")
            cursor = payload.get("cursor")
            if isinstance(cursor, int):
                after = cursor
            elif entries:
                last = entries[-1]
                if isinstance(last, dict) and isinstance(last.get("id"), int):
                    after = int(last["id"])

            for entry in entries:
                emit_entry(entry)

            req_counter += 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        error = _format_cli_exception(exc)
        if json_output:
            _write_line(sys.stdout, json.dumps({"ok": False, "error": error}, indent=2, sort_keys=True))
        else:
            _write_line(sys.stdout, f"error: {error}")
        return 1
    finally:
        try:
            ws.close()
        except Exception:
            pass


def _format_log_entry(entry: object, *, local_time: bool) -> str:
    if not isinstance(entry, dict):
        return str(entry)
    ts_ms = entry.get("ts_ms")
    event = entry.get("event")
    payload = entry.get("payload")
    stamp = "?"
    if isinstance(ts_ms, int):
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        if local_time:
            dt = dt.astimezone()
        stamp = dt.isoformat()
    event_text = str(event or "").strip() or "event"
    payload_text = ""
    if isinstance(payload, dict) and payload:
        payload_text = f" {json.dumps(payload, sort_keys=True, ensure_ascii=False)}"
    return f"{stamp} {event_text}{payload_text}"


def _cmd_sessions(args: argparse.Namespace, workspace_root: Path) -> int:
    root_value = getattr(args, "root", None)
    root = Path(root_value).expanduser() if root_value else (workspace_root / "runs" / "self-improve")
    active_minutes = getattr(args, "active", None)
    json_output = bool(getattr(args, "json", False))

    if active_minutes is not None and int(active_minutes) <= 0:
        payload = {"ok": False, "error": "--active must be a positive integer"}
        if json_output:
            _write_line(sys.stdout, json.dumps(payload, indent=2, sort_keys=True))
        else:
            _write_line(sys.stdout, json.dumps(payload, sort_keys=True))
        return 2

    if root.exists() and not root.is_dir():
        payload = {"ok": False, "error": f"root is not a directory: {root}"}
        if json_output:
            _write_line(sys.stdout, json.dumps(payload, indent=2, sort_keys=True))
        else:
            _write_line(sys.stdout, json.dumps(payload, sort_keys=True))
        return 2

    sessions = _list_sessions(root, active_minutes=int(active_minutes) if active_minutes is not None else None)
    payload = {
        "ok": True,
        "root": str(root),
        "active_minutes": int(active_minutes) if active_minutes is not None else None,
        "count": len(sessions),
        "sessions": sessions,
    }

    if json_output:
        _write_line(sys.stdout, json.dumps(payload, indent=2, sort_keys=True))
        return 0

    for session in sessions:
        goal = str(session.get("goal") or "").replace("\n", " ").strip()
        _write_line(sys.stdout, f"{session.get('id')} {session.get('status')} {goal} {session.get('path')}")
    return 0


def _list_sessions(root: Path, *, active_minutes: int | None) -> list[dict[str, str]]:
    if not root.exists():
        return []

    threshold: float | None = None
    if active_minutes is not None:
        threshold = time.time() - (active_minutes * 60)

    sessions: list[dict[str, str]] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        run_manifest = run_dir / "run.json"
        run_data = _read_optional_json_dict(run_manifest)
        modified_at = run_manifest.stat().st_mtime if run_manifest.exists() else run_dir.stat().st_mtime

        if threshold is not None and modified_at < threshold:
            continue

        sessions.append(
            {
                "id": run_dir.name,
                "path": str(run_dir),
                "status": _run_status(run_data),
                "goal": _run_goal(run_data),
            }
        )

    return sessions


def _read_optional_json_dict(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_status(run_data: dict[str, object]) -> str:
    status = run_data.get("status")
    if isinstance(status, str) and status.strip():
        return status.strip()

    ok_value = run_data.get("ok")
    if ok_value is True:
        return "ok"
    if ok_value is False:
        return "failed"

    post_change = run_data.get("post_change_evaluation")
    if isinstance(post_change, dict):
        post_ok = post_change.get("ok")
        if post_ok is True:
            return "ok"
        if post_ok is False:
            return "failed"

    return "unknown"


def _run_goal(run_data: dict[str, object]) -> str:
    for key in ("goal", "goal_summary"):
        value = run_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    settings = run_data.get("settings")
    if isinstance(settings, dict):
        goal = settings.get("goal")
        if isinstance(goal, str) and goal.strip():
            return goal.strip()

    return ""


def _cmd_memory(args: argparse.Namespace, workspace_root: Path) -> int:
    root_value = getattr(args, "root", None)
    root = Path(root_value).expanduser() if root_value else (workspace_root / "memory")
    json_output = bool(getattr(args, "json", False))
    store = MemoryStore(root)

    def emit(payload: object) -> None:
        if json_output:
            _write_line(sys.stdout, json.dumps(payload, indent=2, sort_keys=True))
            return
        _write_line(sys.stdout, json.dumps(payload, sort_keys=True))

    memory_command = getattr(args, "memory_command", None)
    match memory_command:
        case "status":
            deep = bool(getattr(args, "deep", False))
            payload = store.cli_status(deep=deep)
            index_requested = bool(getattr(args, "index", False))
            if index_requested and payload.get("dirty") is True:
                index_report = store.cli_reindex()
                payload = store.cli_status(deep=deep)
                payload["reindex"] = {
                    "ok": index_report.get("ok", False),
                    "indexed_lessons": index_report.get("indexed_lessons", 0),
                    "errors": index_report.get("errors", []),
                }
                emit(payload)
                return 0 if index_report.get("ok") else 1
            emit(payload)
            return 0
        case "index":
            payload = store.cli_reindex()
            emit(payload)
            return 0 if payload.get("ok") else 1
        case "search":
            query_flag = getattr(args, "query_flag", None)
            query_positional = getattr(args, "query", None)
            query = str(query_flag).strip() if query_flag else str(query_positional or "").strip()
            if not query:
                error_payload = {"ok": False, "error": "query is required"}
                emit(error_payload)
                return 2
            limit = int(getattr(args, "limit", 5) or 5)
            if limit <= 0:
                error_payload = {"ok": False, "error": "limit must be positive"}
                emit(error_payload)
                return 2
            payload = store.cli_search(query, limit=limit)
            emit(payload)
            return 0
        case _:
            emit({"ok": False, "error": "unknown memory subcommand"})
            return 2


def _write_line(stream: TextIO, text: str) -> None:
    stream.write(text)
    stream.write("\n")


def _format_cli_exception(exc: BaseException) -> str:
    text = str(exc).strip()
    if not text:
        return exc.__class__.__name__
    return f"{exc.__class__.__name__}: {text}"


def _find_task_dir(repo_root: Path, task_id: str) -> Path | None:
    tasks_dir = repo_root / "benchmarks" / "tasks"
    if not tasks_dir.exists():
        return None
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        spec_path = task_dir / "task.json"
        if not spec_path.exists():
            continue
        spec = json.loads(spec_path.read_text())
        if spec.get("id") == task_id:
            return task_dir
    return None


_AUTO_ALLOWED_COMMANDS: frozenset[str] = frozenset({"run-suite", "list-skills", "run-task", "self-improve"})


def _auto_decide_argv(prompt: str, *, llm_client: LLMClient | None = None) -> list[str]:
    """Convert a free-form prompt into a concrete CLI argv list.

    Primary behavior: use an AI router to select an argv list.
    Fallback: deterministic heuristic routing when the router fails.
    """

    normalized = prompt.strip()
    if not normalized:
        return []

    routed = _auto_route_with_llm(normalized, llm_client=llm_client)
    if routed:
        return routed

    return _auto_route_heuristic(normalized)


def _auto_route_with_llm(prompt: str, *, llm_client: LLMClient | None) -> list[str] | None:
    normalized = prompt.strip()
    if not normalized:
        return None

    client = llm_client or _build_auto_router_client(Path.cwd().resolve())
    messages = [
        {
            "role": "system",
            "content": "\n".join(
                [
                    "You are Tokimon's CLI router.",
                    "Given a user prompt, choose exactly one Tokimon subcommand to run.",
                    "",
                    "Allowed commands (argv[0]): run-suite, list-skills, run-task, self-improve.",
                    "Return JSON with: {\"status\":\"SUCCESS\",\"argv\":[...]} and nothing else.",
                    "Rules:",
                    "- argv excludes the leading 'tokimon'.",
                    "- Use minimal argv; prefer self-improve for learning/improving prompts.",
                    "- run-task MUST include: --task-id <id>.",
                    "- Do NOT return 'auto' and do NOT include -h/--help.",
                    "- Do NOT request tool calls.",
                    "",
                    "Examples:",
                    "Prompt: run suite -> {\"status\":\"SUCCESS\",\"argv\":[\"run-suite\"]}",
                    "Prompt: list skills -> {\"status\":\"SUCCESS\",\"argv\":[\"list-skills\"]}",
                    "Prompt: run task demo-1 -> {\"status\":\"SUCCESS\",\"argv\":[\"run-task\",\"--task-id\",\"demo-1\"]}",
                    "Prompt: please learn X -> {\"status\":\"SUCCESS\",\"argv\":[\"self-improve\",\"--goal\",\"please learn X\"]}",
                ]
            ),
        },
        {"role": "user", "content": normalized},
    ]

    try:
        response = client.send(messages, tools=None)
    except Exception:
        return None
    argv = _extract_routed_argv(response)
    if argv is None:
        return None
    if not _validate_routed_argv(argv):
        return None
    return argv


def _extract_routed_argv(response: object) -> list[str] | None:
    if not isinstance(response, dict):
        return None
    raw = response.get("argv")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            argv = shlex.split(raw)
        except ValueError:
            return None
        return argv or None
    elif isinstance(raw, list):
        candidate = raw
    else:
        return None

    argv: list[str] = []
    for item in candidate:
        if not isinstance(item, str):
            return None
        stripped = item.strip()
        if not stripped:
            continue
        argv.append(stripped)
    if not argv:
        return None
    return argv


def _validate_routed_argv(argv: list[str]) -> bool:
    if argv[0] not in _AUTO_ALLOWED_COMMANDS:
        return False
    if any(arg in {"-h", "--help"} for arg in argv):
        return False

    parser = build_parser(exit_on_error=False)
    try:
        parsed = parser.parse_args(argv)
    except (argparse.ArgumentError, ValueError):
        return False
    except SystemExit:
        return False
    if getattr(parsed, "command", None) != argv[0]:
        return False
    return True


def _auto_router_provider() -> str:
    provider = (os.environ.get("TOKIMON_LLM") or "").strip().lower()
    if provider in {"codex", "codex-cli"}:
        return "codex"
    if provider in {"claude", "claude-cli"}:
        return "claude"
    if provider in {"mock"}:
        return "mock"
    if provider in {"mixed"}:
        return "codex"
    return "codex"


def _build_auto_router_client(workspace_dir: Path) -> LLMClient:
    provider = _auto_router_provider()
    if provider == "codex":
        codex_settings = replace(
            CodexCLISettings.from_env(),
            sandbox="read-only",
            ask_for_approval="never",
            search=False,
            timeout_s=60,
        )
        return CodexCLIClient(workspace_dir, settings=codex_settings)
    if provider == "claude":
        claude_settings = replace(ClaudeCLISettings.from_env(), timeout_s=60)
        return ClaudeCLIClient(workspace_dir, settings=claude_settings)
    if provider == "mock":
        return MockLLMClient(script=[])
    try:
        return build_llm_client(provider, workspace_dir=workspace_dir)
    except Exception:
        return MockLLMClient(script=[])


def _auto_route_heuristic(prompt: str) -> list[str]:
    normalized = prompt.strip()
    if not normalized:
        return []

    lower = normalized.lower()

    if lower in {"run-suite", "suite"} or "run suite" in lower:
        return ["run-suite"]

    if lower in {"list-skills", "skills"} or "list skills" in lower:
        return ["list-skills"]

    task_match = re.search(
        r"(?:^|\b)(?:run\s+task\s+|task\s*[:=]\s*|task-id\s*[:=]\s*)([A-Za-z0-9_.-]+)(?:\b|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if task_match:
        return ["run-task", "--task-id", task_match.group(1)]

    return ["self-improve", "--goal", normalized]


if __name__ == "__main__":
    raise SystemExit(main())
