"""Benchmark harness for Tokimon."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from llm.client import MockLLMClient
from runs import RunContext, create_run_context
from runners.baseline import BaselineRunner
from runners.hierarchical import HierarchicalRunner
from tools.pytest_tool import PytestTool


@dataclass
class BenchmarkTask:
    task_id: str
    description: str
    path: Path
    test_args: list[str]


@dataclass
class TaskResult:
    task_id: str
    runner: str
    passed: int | None
    failed: int | None
    best_passed: int | None
    best_failed: int | None
    wall_time_s: float
    model_calls: int
    tool_calls: int
    lessons: int
    artifacts_path: str


class BenchmarkSuite:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.tasks = self._load_tasks()

    def _load_tasks(self) -> list[BenchmarkTask]:
        tasks = []
        tasks_dir = self.root / "benchmarks" / "tasks"
        if not tasks_dir.exists():
            return tasks
        for task_dir in sorted(tasks_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            spec_path = task_dir / "task.json"
            if not spec_path.exists():
                continue
            spec = json.loads(spec_path.read_text())
            tasks.append(
                BenchmarkTask(
                    task_id=spec["id"],
                    description=spec.get("description", ""),
                    path=task_dir,
                    test_args=[str(task_dir / "tests")],
                )
            )
        return tasks


class EvaluationHarness:
    def __init__(self, repo_root: Path, runs_dir: Path | None = None) -> None:
        self.repo_root = repo_root
        self.suite = BenchmarkSuite(repo_root)
        self.base_runs_dir = runs_dir or (repo_root.parent / "runs")
        self.base_runs_dir.mkdir(parents=True, exist_ok=True)

    def run_suite(self) -> RunContext:
        run_context = create_run_context(self.base_runs_dir)
        run_context.write_manifest({"suite": "default", "runner": "comparison"})
        results: list[TaskResult] = []
        for task in self.suite.tasks:
            results.extend(self._run_task(task, run_context))
        report = {
            "suite": "default",
            "tasks": [result.__dict__ for result in results],
        }
        json_path = run_context.reports_dir / "suite-report.json"
        json_path.write_text(json.dumps(report, indent=2))
        md_path = run_context.reports_dir / "suite-report.md"
        md_path.write_text(_format_markdown_report(results))
        return run_context

    def _run_task(self, task: BenchmarkTask, run_context: RunContext) -> list[TaskResult]:
        results = []
        for runner_name in ("baseline", "hierarchical"):
            workspace = run_context.root / "workspaces" / task.task_id / runner_name
            if workspace.exists():
                shutil.rmtree(workspace)
            shutil.copytree(task.path / "starter", workspace)
            tests_dst = workspace / "tests"
            shutil.copytree(task.path / "tests", tests_dst)
            llm_client = MockLLMClient(script=[])
            start = time.perf_counter()
            runner_result = None
            if runner_name == "baseline":
                runner = BaselineRunner(workspace, llm_client, base_dir=run_context.root / "runs")
                runner_result = runner.run(task.description, task_id=task.task_id, test_args=[str(tests_dst)])
            else:
                runner = HierarchicalRunner(workspace, llm_client, base_dir=run_context.root / "runs")
                runner_result = runner.run(task.description, task_steps=None, task_id=task.task_id, test_args=[str(tests_dst)])
            wall_time = time.perf_counter() - start
            pytest_tool = PytestTool(workspace)
            pytest_result = pytest_tool.run([str(tests_dst)])
            lessons_dir = workspace / "memory" / "lessons"
            lessons_count = len(list(lessons_dir.glob("lesson-*.md"))) if lessons_dir.exists() else 0
            artifacts_path = str(runner_result.run_context.root) if runner_result is not None else str(run_context.root)
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    runner=runner_name,
                    passed=pytest_result.data.get("passed"),
                    failed=pytest_result.data.get("failed"),
                    wall_time_s=wall_time,
                    best_passed=getattr(runner_result, "best_passed", None) if runner_result is not None else None,
                    best_failed=getattr(runner_result, "best_failed", None) if runner_result is not None else None,
                    model_calls=getattr(runner_result, "model_calls", 0) if runner_result is not None else 0,
                    tool_calls=getattr(runner_result, "tool_calls", 0) if runner_result is not None else 0,
                    lessons=lessons_count,
                    artifacts_path=artifacts_path,
                )
            )
        return results


def _format_markdown_report(results: list[TaskResult]) -> str:
    lines = [
        "# Tokimon Benchmark Report",
        "",
        "| Task | Runner | Passed | Failed | Best Passed | Best Failed | Model Calls | Tool Calls | Wall Time (s) | Lessons | Artifacts |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.task_id} | {result.runner} | {result.passed} | {result.failed} | {result.best_passed} | {result.best_failed} | {result.model_calls} | {result.tool_calls} | {result.wall_time_s:.2f} | {result.lessons} | {result.artifacts_path} |"
        )
    return "\n".join(lines)
