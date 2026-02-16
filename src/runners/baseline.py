"""Baseline runner with a single-agent loop."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.worker import Worker
from artifacts import ArtifactStore
from flow_types import ProgressMetrics, WorkerStatus
from logging_utils import log_to_file
from memory.store import MemoryStore
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
        output = worker.run(
            goal,
            "single-step",
            {},
            [],
            trace=trace,
            trace_context={"task_id": task_id or "baseline", "worker_type": "Implementer"},
        )
        log_to_file(run_context.logs_dir / "baseline.log", f"Output status {output.status} summary {output.summary}")
        pytest_metrics = None
        if test_args:
            pytest_metrics = tools["pytest"].run(test_args).data
        artifact_store.write_step(task_id or "baseline", "single-step", output.artifacts, outputs={"summary": output.summary})
        progress = ProgressMetrics(
            failing_tests=pytest_metrics.get("failed") if pytest_metrics else None,
            passed_tests=pytest_metrics.get("passed") if pytest_metrics else None,
            new_artifacts=len(output.artifacts),
            artifact_delta_hash=_hash_artifacts(output.artifacts),
        )
        model_calls = int(output.metrics.get("model_calls") or 0)
        tool_calls = int(output.metrics.get("tool_calls") or 0)
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
