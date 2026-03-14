"""Baseline runner with a single-agent loop."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.worker import Worker
from artifacts import ArtifactStore
from flow_types import ProgressMetrics, WorkerStatus
from logging_utils import log_to_file
from memory.store import MemoryStore
from observability.reports import build_run_metrics_payload
from observability.reports import normalize_step_metrics
from observability.reports import write_metrics_and_dashboard
from replay import ReplayRecorder
from runs import RunContext, create_run_context
from tools.file_tool import FileTool
from tools.grep_tool import GrepTool
from tools.patch_tool import PatchTool
from tools.pytest_tool import PytestTool
from tracing import TraceLogger


@dataclass
class BaselineResult:
    run_context: RunContext
    metrics: ProgressMetrics
    model_calls: int
    tool_calls: int
    best_passed: int | None = None
    best_failed: int | None = None


class BaselineRunner:
    def __init__(self, repo_root: Path, llm_client, base_dir: Path | None = None) -> None:
        self.repo_root = repo_root
        self.llm_client = llm_client
        self.base_dir = base_dir or (repo_root / "runs")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def run(self, goal: str, task_id: str | None = None, test_args: list[str] | None = None) -> BaselineResult:
        run_start = time.perf_counter()
        run_context = create_run_context(self.base_dir)
        run_context.write_manifest({"goal": goal, "task_id": task_id, "runner": "baseline"})
        trace = TraceLogger(run_context.trace_path)
        memory_store = MemoryStore(self.repo_root / "memory")
        artifact_store = ArtifactStore(run_context.artifacts_dir, memory_store=memory_store)
        log_to_file(run_context.logs_dir / "baseline.log", f"Run start: {goal}")
        tools = {
            "file": FileTool(self.repo_root),
            "grep": GrepTool(self.repo_root),
            "patch": PatchTool(self.repo_root),
            "pytest": PytestTool(self.repo_root),
        }
        worker = Worker("Implementer", self.llm_client, tools)
        replay = ReplayRecorder(
            step_id="single-step",
            worker_role="Implementer",
            goal=goal,
            inputs={},
            memory=[],
        )
        output = worker.run(
            goal,
            "single-step",
            {},
            [],
            trace=trace,
            trace_context={"task_id": task_id or "baseline", "worker_type": "Implementer"},
            replay_recorder=replay,
        )
        log_to_file(run_context.logs_dir / "baseline.log", f"Output status {output.status} summary {output.summary}")
        pytest_metrics = None
        if test_args:
            pytest_metrics = tools["pytest"].run(test_args).data
        raw_ui_blocks = output.data.get("ui_blocks") if isinstance(output.data, dict) else None
        ui_blocks: list[dict[str, Any]] = []
        if isinstance(raw_ui_blocks, list):
            ui_blocks = [block for block in raw_ui_blocks if isinstance(block, dict)]
        step_result: dict[str, Any] = {
            "status": output.status.value,
            "summary": output.summary,
            "artifacts": output.artifacts,
            "metrics": output.metrics,
            "next_actions": output.next_actions,
            "failure_signature": str(output.failure_signature or ""),
            "ui_blocks": ui_blocks,
        }
        artifact_store.write_step(
            task_id or "baseline",
            "single-step",
            output.artifacts,
            outputs={"summary": output.summary},
            step_result=step_result,
            replay_record=replay.build(),
        )
        progress = ProgressMetrics(
            failing_tests=pytest_metrics.get("failed") if pytest_metrics else None,
            passed_tests=pytest_metrics.get("passed") if pytest_metrics else None,
            new_artifacts=len(output.artifacts),
            artifact_delta_hash=_hash_artifacts(output.artifacts),
        )
        model_calls = int(output.metrics.get("model_calls") or 0)
        tool_calls = int(output.metrics.get("tool_calls") or 0)
        wall_time_s = time.perf_counter() - run_start
        step_metrics = normalize_step_metrics(
            step_id="single-step",
            attempt_id=1,
            status=output.status.value,
            artifacts=output.artifacts,
            raw_metrics=output.metrics,
            failure_signature=output.failure_signature,
        )
        run_metrics_payload = build_run_metrics_payload(
            run_id=run_context.run_id,
            runner="baseline",
            wall_time_s=wall_time_s,
            steps=[step_metrics],
            tests_passed=pytest_metrics.get("passed") if pytest_metrics else None,
            tests_failed=pytest_metrics.get("failed") if pytest_metrics else None,
        )
        write_metrics_and_dashboard(run_context.reports_dir, run_metrics_payload)
        trace.log("baseline_complete", {"status": output.status, "summary": output.summary})
        log_to_file(run_context.logs_dir / "baseline.log", "Run complete")
        return BaselineResult(
            run_context=run_context,
            metrics=progress,
            model_calls=model_calls,
            tool_calls=tool_calls,
            best_passed=pytest_metrics.get("passed") if pytest_metrics else None,
            best_failed=pytest_metrics.get("failed") if pytest_metrics else None,
        )


def _hash_artifacts(artifacts: list[dict[str, Any]]) -> str:
    data = json.dumps(artifacts, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()
