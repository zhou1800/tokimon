"""CLI entrypoint for Tokimon."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

from benchmarks.harness import EvaluationHarness
from llm.client import CodexCLIClient, CodexCLISettings, MockLLMClient, build_llm_client
from memory.store import MemoryStore
from runners.baseline import BaselineRunner
from runners.hierarchical import HierarchicalRunner
from self_improve.orchestrator import SelfImproveOrchestrator, SelfImproveSettings
from skills.builder import SkillBuilder
from skills.registry import SkillRegistry
from skills.spec import SkillSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokimon")
    subparsers = parser.add_subparsers(dest="command")

    run_task = subparsers.add_parser("run-task")
    run_task.add_argument("--task-id", required=True)
    run_task.add_argument("--runner", choices=["baseline", "hierarchical"], default="hierarchical")

    subparsers.add_parser("run-suite")

    resume = subparsers.add_parser("resume-run")
    resume.add_argument("--run-path", required=True)

    inspect_run = subparsers.add_parser("inspect-run")
    inspect_run.add_argument("--run-path", required=True)

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
    self_improve.add_argument("--sessions", type=int, default=4)
    self_improve.add_argument("--batches", type=int, default=1)
    self_improve.add_argument("--workers", type=int, default=4)
    self_improve.add_argument("--no-merge", action="store_true")
    self_improve.add_argument(
        "--llm",
        choices=["mock", "codex"],
        default=os.environ.get("TOKIMON_LLM") or os.environ.get("AGENT_FLOW_LLM", "mock"),
        help="LLM provider to use for self-improve sessions (or set TOKIMON_LLM / AGENT_FLOW_LLM).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    repo_root = Path(__file__).resolve().parent
    llm_client = MockLLMClient(script=[])

    if args.command == "run-task":
        task_dir = _find_task_dir(repo_root, args.task_id)
        if task_dir is None:
            raise SystemExit(f"Unknown task: {args.task_id}")
        spec_path = task_dir / "task.json"
        spec = json.loads(spec_path.read_text())
        workspace = repo_root / "runs" / "workspaces" / args.task_id / args.runner
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(task_dir / "starter", workspace)
        tests_dst = workspace / "tests"
        shutil.copytree(task_dir / "tests", tests_dst)
        test_args = [str(tests_dst)]
        if args.runner == "baseline":
            runner = BaselineRunner(workspace, llm_client, base_dir=repo_root / "runs")
            runner.run(spec.get("description", ""), task_id=args.task_id, test_args=test_args)
        else:
            runner = HierarchicalRunner(workspace, llm_client, base_dir=repo_root / "runs")
            runner.run(spec.get("description", ""), task_steps=None, task_id=args.task_id, test_args=test_args)
        return 0

    if args.command == "run-suite":
        harness = EvaluationHarness(repo_root)
        harness.run_suite()
        return 0

    if args.command == "resume-run":
        run_path = Path(args.run_path)
        workflow_state = run_path / "workflow_state.json"
        if not workflow_state.exists():
            raise SystemExit("workflow_state.json not found")
        runner = HierarchicalRunner(repo_root, llm_client, base_dir=repo_root / "runs")
        runner.resume(run_path)
        return 0

    if args.command == "inspect-run":
        run_path = Path(args.run_path)
        run_manifest = run_path / "run.json"
        workflow_state = run_path / "workflow_state.json"
        if run_manifest.exists():
            print(run_manifest.read_text())
        if workflow_state.exists():
            print(workflow_state.read_text())
        return 0

    if args.command == "list-skills":
        registry = SkillRegistry(repo_root)
        registry.load()
        for spec in registry.list_skills():
            print(f"{spec.name}: {spec.purpose}")
        return 0

    if args.command == "build-skill":
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

    if args.command == "self-improve":
        master_root = Path.cwd().resolve()
        settings = SelfImproveSettings(
            sessions_per_batch=args.sessions,
            batches=args.batches,
            max_workers=args.workers,
            merge_on_success=not args.no_merge,
        )
        llm_provider = str(args.llm or "mock").strip().lower()

        def llm_factory(_session_id: str, workspace_dir: Path):
            if llm_provider in {"codex", "codex-cli"}:
                codex_settings = CodexCLISettings.from_env()
                if "AGENT_FLOW_CODEX_SANDBOX" not in os.environ:
                    codex_settings = replace(codex_settings, sandbox="workspace-write")
                if "AGENT_FLOW_CODEX_APPROVAL" not in os.environ:
                    codex_settings = replace(codex_settings, ask_for_approval="never")
                if "AGENT_FLOW_CODEX_SEARCH" not in os.environ:
                    codex_settings = replace(codex_settings, search=True)
                if "AGENT_FLOW_CODEX_TIMEOUT_S" not in os.environ:
                    codex_settings = replace(codex_settings, timeout_s=240)
                return CodexCLIClient(workspace_dir, settings=codex_settings)
            return build_llm_client(llm_provider, workspace_dir=workspace_dir)

        orchestrator = SelfImproveOrchestrator(master_root, llm_factory=llm_factory, settings=settings)
        report = orchestrator.run(args.goal, input_ref=args.input)
        print(json.dumps({"run_root": report.run_root}, indent=2))
        return 0

    return 1


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


if __name__ == "__main__":
    raise SystemExit(main())
