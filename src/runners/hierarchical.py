"""Hierarchical runner using manager/worker and workflow engine."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.manager import Manager
from agents.worker import Worker
from artifacts import ArtifactStore
from execution.parallel import AsyncExecutor, ConcurrencyConfig
from flow_types import ProgressMetrics, StepStatus, WorkerStatus
from logging_utils import log_to_file
from memory.store import MemoryStore
from runs import RunContext, create_run_context, load_run_context
from tools.file_tool import FileTool
from tools.grep_tool import GrepTool
from tools.patch_tool import PatchTool
from tools.pytest_tool import PytestTool
from tools.web_tool import WebTool
from tracing import TraceLogger
from workflow.engine import WorkflowEngine
from workflow.models import StepAttempt


@dataclass
class HierarchicalResult:
    run_context: RunContext
    workflow_state_path: Path
    model_calls: int
    tool_calls: int
    best_passed: int | None = None
    best_failed: int | None = None


class HierarchicalRunner:
    def __init__(self, repo_root: Path, llm_client, base_dir: Path | None = None) -> None:
        self.repo_root = repo_root
        self.llm_client = llm_client
        self.base_dir = base_dir or (repo_root / "runs")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def run(self, goal: str, task_steps: list[dict[str, Any]] | None = None,
            task_id: str | None = None, test_args: list[str] | None = None,
            concurrency: int = 4) -> HierarchicalResult:
        run_context = create_run_context(self.base_dir)
        run_context.write_manifest({"goal": goal, "task_id": task_id, "runner": "hierarchical"})
        memory_store = MemoryStore(self.repo_root / "memory")
        manager = Manager(memory_store)
        artifact_store = ArtifactStore(run_context.artifacts_dir, memory_store=memory_store)
        trace = TraceLogger(run_context.trace_path)
        tools = self._build_tools()
        if task_steps is None:
            planned = manager.plan_steps(goal, self.llm_client, tools)
            if planned:
                task_steps = planned
        workflow_spec = manager.build_workflow(goal, task_steps)
        engine = WorkflowEngine(workflow_spec)
        executor = AsyncExecutor(ConcurrencyConfig(max_concurrency=concurrency))

        manager_log = run_context.logs_dir / "manager.log"
        log_to_file(manager_log, f"Run start: {goal}")

        async def run_loop() -> None:
            while not engine.state.is_complete():
                ready = engine.ready_steps()
                if not ready:
                    break
                tasks = [
                    lambda step_id=step_id: self._run_step(step_id, engine, manager, tools, trace, run_context,
                                                           task_id or workflow_spec.workflow_id, test_args, artifact_store)
                    for step_id in ready
                ]
                await executor.run(tasks)
                termination = engine.state.metadata.pop("terminate_workflow", None)
                if isinstance(termination, dict):
                    reason = str(termination.get("reason") or "terminated early").strip()
                    triggered_by = str(termination.get("step_id") or "<unknown>")
                    engine.skip_remaining(reason=reason or "terminated early", triggered_by=triggered_by)
                    engine.save(run_context.workflow_state_path)
                    break
                engine.save(run_context.workflow_state_path)

        asyncio.run(run_loop())
        log_to_file(manager_log, "Run complete")
        model_calls, tool_calls, best_passed, best_failed = _summarize_workflow(engine)
        return HierarchicalResult(
            run_context=run_context,
            workflow_state_path=run_context.workflow_state_path,
            model_calls=model_calls,
            tool_calls=tool_calls,
            best_passed=best_passed,
            best_failed=best_failed,
        )

    def resume(self, run_path: Path, test_args: list[str] | None = None,
               concurrency: int = 4) -> HierarchicalResult:
        run_context = load_run_context(run_path)
        engine = WorkflowEngine.load(run_context.workflow_state_path)
        memory_store = MemoryStore(self.repo_root / "memory")
        manager = Manager(memory_store)
        artifact_store = ArtifactStore(run_context.artifacts_dir, memory_store=memory_store)
        trace = TraceLogger(run_context.trace_path)
        tools = self._build_tools()
        executor = AsyncExecutor(ConcurrencyConfig(max_concurrency=concurrency))
        manager_log = run_context.logs_dir / "manager.log"
        log_to_file(manager_log, "Resume run")

        async def run_loop() -> None:
            while not engine.state.is_complete():
                ready = engine.ready_steps()
                if not ready:
                    break
                tasks = [
                    lambda step_id=step_id: self._run_step(step_id, engine, manager, tools, trace, run_context,
                                                           engine.spec.workflow_id, test_args, artifact_store)
                    for step_id in ready
                ]
                await executor.run(tasks)
                termination = engine.state.metadata.pop("terminate_workflow", None)
                if isinstance(termination, dict):
                    reason = str(termination.get("reason") or "terminated early").strip()
                    triggered_by = str(termination.get("step_id") or "<unknown>")
                    engine.skip_remaining(reason=reason or "terminated early", triggered_by=triggered_by)
                    engine.save(run_context.workflow_state_path)
                    break
                engine.save(run_context.workflow_state_path)

        asyncio.run(run_loop())
        log_to_file(manager_log, "Resume complete")
        model_calls, tool_calls, best_passed, best_failed = _summarize_workflow(engine)
        return HierarchicalResult(
            run_context=run_context,
            workflow_state_path=run_context.workflow_state_path,
            model_calls=model_calls,
            tool_calls=tool_calls,
            best_passed=best_passed,
            best_failed=best_failed,
        )

    def _build_tools(self) -> dict[str, Any]:
        return {
            "file": FileTool(self.repo_root),
            "grep": GrepTool(self.repo_root),
            "patch": PatchTool(self.repo_root),
            "pytest": PytestTool(self.repo_root),
            "web": WebTool(),
        }

    async def _run_step(self, step_id: str, engine: WorkflowEngine, manager: Manager, tools: dict[str, Any],
                        trace: TraceLogger, run_context: RunContext, task_id: str, test_args: list[str] | None,
                        artifact_store: ArtifactStore) -> None:
        step_state = engine.state.steps[step_id]
        step_spec = engine.spec.step_map()[step_id]
        worker_log = run_context.logs_dir / f"worker-{step_id}.log"
        log_to_file(worker_log, f"Step start attempt {len(step_state.attempts) + 1}")
        attempt_index = len(step_state.attempts)
        strategy = manager.next_strategy(attempt_index)
        if strategy is None:
            engine.mark_status(step_id, StepStatus.FAILED, error="No remaining strategies")
            trace.log("step_failed", {"step_id": step_id, "reason": "no_strategies"})
            return

        if attempt_index > 0:
            prev = manager.next_strategy(attempt_index - 1)
            if prev:
                manager.write_retry_lesson(task_id, step_id, prev, strategy, step_state.last_attempt.failure_signature if step_state.last_attempt else "", "Changed strategy")

        worker_type = step_spec.worker if attempt_index == 0 else strategy.worker_type
        call_signature = manager.compute_call_signature(engine.spec.goal, step_id, worker_type, step_state.inputs, strategy)
        call_id = manager.next_call_id()
        if not manager.delegation_graph.add_edge(call_id, call_signature):
            engine.mark_status(step_id, StepStatus.FAILED, error="Delegation cycle detected")
            trace.log("step_failed", {"step_id": step_id, "reason": "cycle"})
            return

        engine.mark_running(step_id)
        memory = [lesson.body for lesson in manager.memory_store.retrieve(step_id, strategy.retrieval_stage, limit=3)]
        worker = Worker(worker_type, self.llm_client, tools)
        output = worker.run(engine.spec.goal, step_id, step_state.inputs, memory)
        log_to_file(worker_log, f"Output status {output.status} summary {output.summary}")
        if output.failure_signature:
            log_to_file(worker_log, f"Failure signature: {output.failure_signature}")
        details = output.metrics.get("details")
        if isinstance(details, str) and details.strip():
            log_to_file(worker_log, f"Details: {details.strip()}")

        outputs_payload: dict[str, Any] = {"summary": output.summary, "artifact_count": len(output.artifacts)}
        if output.failure_signature:
            outputs_payload["failure_signature"] = output.failure_signature
        if isinstance(details, str) and details.strip():
            outputs_payload["details"] = details.strip()
        engine.mark_outputs(step_id, outputs_payload)

        pytest_metrics = await self._run_tests(test_args, tools.get("pytest")) if test_args else None
        artifact_hash = artifact_store.write_step(task_id, step_id, output.artifacts, outputs=step_state.outputs)
        touched_hash = None
        touched_files = output.metrics.get("touched_files") if isinstance(output.metrics, dict) else None
        if isinstance(touched_files, list) and touched_files:
            touched_hash = _hash_touched_files(self.repo_root, [str(p) for p in touched_files])
        progress = ProgressMetrics(
            failing_tests=pytest_metrics.get("failed") if pytest_metrics else None,
            passed_tests=pytest_metrics.get("passed") if pytest_metrics else None,
            new_artifacts=len(output.artifacts),
            artifact_delta_hash=touched_hash or artifact_hash,
        )
        prev_attempt = step_state.last_attempt
        attempt = StepAttempt(
            attempt_id=attempt_index + 1,
            status=StepStatus.SUCCEEDED if output.status == WorkerStatus.SUCCESS else StepStatus.BLOCKED if output.status == WorkerStatus.BLOCKED else StepStatus.FAILED if output.status == WorkerStatus.FAILURE else StepStatus.PARTIAL,
            call_signature=call_signature,
            worker_type=worker_type,
            strategy_id=strategy.strategy_id,
            retrieval_stage=strategy.retrieval_stage,
            summary=output.summary,
            failure_signature=output.failure_signature,
            progress_metrics={
                "failing_tests": progress.failing_tests,
                "passed_tests": progress.passed_tests,
                "new_artifacts": progress.new_artifacts,
                "artifact_delta_hash": progress.artifact_delta_hash,
                "model_calls": output.metrics.get("model_calls"),
                "tool_calls": output.metrics.get("tool_calls"),
                "iteration_count": output.metrics.get("iteration_count"),
            },
            artifacts=output.artifacts,
        )
        engine.record_attempt(step_id, attempt)

        terminate_workflow = False
        terminate_reason = ""
        if isinstance(output.metrics, dict):
            terminate_workflow = output.metrics.get("terminate_workflow") is True
            if not terminate_workflow:
                control = output.metrics.get("workflow_control")
                if isinstance(control, dict):
                    terminate_workflow = control.get("terminate_workflow") is True
                    terminate_reason = str(control.get("reason") or control.get("terminate_reason") or "")
            terminate_reason = str(output.metrics.get("terminate_reason") or terminate_reason or "")

        tests_green = True
        if isinstance(pytest_metrics, dict):
            returncode = pytest_metrics.get("returncode")
            if isinstance(returncode, int):
                tests_green = returncode == 0

        if output.status == WorkerStatus.SUCCESS:
            manager.record_progress(task_id, call_signature, output.failure_signature, progress)
            manager.delegation_graph.record_artifacts(call_signature, progress.artifact_delta_hash or "")
            engine.mark_status(step_id, StepStatus.SUCCEEDED)
            if terminate_workflow and tests_green and "terminate_workflow" not in engine.state.metadata:
                reason = terminate_reason.strip() or output.summary or "terminated early"
                engine.state.metadata["terminate_workflow"] = {"step_id": step_id, "reason": reason}
                log_to_file(worker_log, f"Requested workflow termination: {reason}")
            trace.log("step_succeeded", {"step_id": step_id})
            return

        if output.status in {WorkerStatus.PARTIAL, WorkerStatus.FAILURE}:
            prev_metrics = _progress_from_attempt(prev_attempt)
            allowed = manager.check_retry_allowed(task_id, call_signature, output.failure_signature, prev_metrics, progress)
            manager.record_progress(task_id, call_signature, output.failure_signature, progress)
            repeated = manager.delegation_graph.record_artifacts(call_signature, progress.artifact_delta_hash or "")
            if repeated or not allowed:
                engine.mark_status(step_id, StepStatus.FAILED, error="retry blocked")
                trace.log("step_failed", {"step_id": step_id, "reason": "retry_blocked"})
            else:
                engine.mark_status(step_id, StepStatus.RETRY_PENDING)
                trace.log("step_retry", {"step_id": step_id, "reason": "allowed"})
            return

        if output.status == WorkerStatus.BLOCKED:
            manager.record_progress(task_id, call_signature, output.failure_signature, progress)
            manager.delegation_graph.record_artifacts(call_signature, progress.artifact_delta_hash or "")
            engine.mark_status(step_id, StepStatus.BLOCKED, error="worker blocked")
            trace.log("step_blocked", {"step_id": step_id})

    async def _run_tests(self, test_args: list[str], pytest_tool: PytestTool | None) -> dict[str, Any]:
        if pytest_tool is None:
            return {}
        result = pytest_tool.run(test_args)
        return result.data


def _hash_touched_files(repo_root: Path, relpaths: list[str], *, max_files: int = 50, max_bytes: int = 2_000_000) -> str:
    repo_root = repo_root.resolve()
    hasher = hashlib.sha256()
    unique = []
    seen = set()
    for relpath in relpaths:
        relpath = str(relpath).strip()
        if not relpath or relpath in seen:
            continue
        seen.add(relpath)
        unique.append(relpath)

    truncated = unique[:max_files]
    for relpath in sorted(truncated):
        normalized = relpath.lstrip("./")
        if not normalized or normalized.startswith("/"):
            hasher.update(f"invalid:{relpath}".encode())
            continue
        path = (repo_root / normalized).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError:
            hasher.update(f"traversal:{normalized}".encode())
            continue
        if not path.exists() or not path.is_file():
            hasher.update(f"missing:{normalized}".encode())
            continue
        data = path.read_bytes()
        if len(data) > max_bytes:
            data = data[:max_bytes]
        hasher.update(normalized.encode())
        hasher.update(b"\0")
        hasher.update(data)
        hasher.update(b"\0")

    if len(unique) > max_files:
        hasher.update(f"truncated:{len(unique) - max_files}".encode())
    return hasher.hexdigest()


def _progress_from_attempt(attempt: StepAttempt | None) -> ProgressMetrics | None:
    if attempt is None:
        return None
    return ProgressMetrics(
        failing_tests=attempt.progress_metrics.get("failing_tests"),
        passed_tests=attempt.progress_metrics.get("passed_tests"),
        new_artifacts=attempt.progress_metrics.get("new_artifacts"),
        artifact_delta_hash=attempt.progress_metrics.get("artifact_delta_hash"),
    )


def _summarize_workflow(engine: WorkflowEngine) -> tuple[int, int, int | None, int | None]:
    model_calls = 0
    tool_calls = 0
    best_passed: int | None = None
    best_failed: int | None = None
    for step_state in engine.state.steps.values():
        for attempt in step_state.attempts:
            model_calls += int(attempt.progress_metrics.get("model_calls") or 0)
            tool_calls += int(attempt.progress_metrics.get("tool_calls") or 0)
            passed = attempt.progress_metrics.get("passed_tests")
            failed = attempt.progress_metrics.get("failing_tests")
            if isinstance(passed, int):
                best_passed = passed if best_passed is None else max(best_passed, passed)
            if isinstance(failed, int):
                best_failed = failed if best_failed is None else min(best_failed, failed)
    return model_calls, tool_calls, best_passed, best_failed
