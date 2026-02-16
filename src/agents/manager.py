"""Manager agent: planning, delegation, retry gating."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from agents.delegation import DelegationGraph
from agents.retry import RetryGate, compute_call_signature
from agents.worker import Worker
from flow_types import ProgressMetrics, StepStatus
from llm.client import LLMClient
from memory.store import MemoryStore
from tracing import TraceLogger
from workflow.models import StepSpec, WorkflowSpec


@dataclass
class Strategy:
    strategy_id: str
    worker_type: str
    retrieval_stage: int
    strategy_class: str
    tool_sequence: list[str]


DEFAULT_STRATEGIES = [
    Strategy(strategy_id="draft", worker_type="Planner", retrieval_stage=1, strategy_class="write_from_scratch", tool_sequence=["grep", "file"]),
    Strategy(strategy_id="patch", worker_type="Debugger", retrieval_stage=2, strategy_class="patch", tool_sequence=["grep", "pytest", "patch"]),
    Strategy(strategy_id="refactor", worker_type="Reviewer", retrieval_stage=3, strategy_class="refactor", tool_sequence=["grep", "file"]),
]


class Manager:
    def __init__(self, memory_store: MemoryStore, retry_gate: RetryGate | None = None) -> None:
        self.memory_store = memory_store
        self.retry_gate = retry_gate or RetryGate()
        self.delegation_graph = DelegationGraph()
        self._call_counter = 0

    def plan_steps(
        self,
        goal: str,
        llm_client: LLMClient,
        tools: dict[str, Any],
        max_steps: int = 12,
        *,
        trace: TraceLogger | None = None,
        trace_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | None:
        memory = [lesson.body for lesson in self.memory_store.retrieve(goal, stage=1, limit=3)]
        worker = Worker("Planner", llm_client, tools)
        output = worker.run(
            goal,
            "plan-workflow",
            {"max_steps": max_steps},
            memory,
            trace=trace,
            trace_context=trace_context,
        )

        workflow = output.data.get("workflow")
        if isinstance(workflow, dict):
            steps = workflow.get("steps")
            if isinstance(steps, list):
                return _normalize_steps(steps)

        task_steps = output.data.get("task_steps")
        if isinstance(task_steps, list):
            return _normalize_steps(task_steps)

        return None

    def build_workflow(self, goal: str, task_steps: list[dict[str, Any]] | None = None) -> WorkflowSpec:
        steps: list[StepSpec] = []
        if task_steps:
            for step in task_steps:
                steps.append(
                    StepSpec(
                        step_id=step["id"],
                        name=step.get("name", step["id"]),
                        description=step.get("description", ""),
                        worker=step.get("worker", "Implementer"),
                        depends_on=step.get("depends_on", []),
                        inputs_schema=step.get("inputs_schema", {}),
                        outputs_schema=step.get("outputs_schema", {}),
                        default_inputs=step.get("inputs", {}),
                    )
                )
        else:
            steps = [
                StepSpec(
                    step_id="solve",
                    name="Solve task",
                    description="Solve the task and produce artifacts.",
                    worker="Implementer",
                    depends_on=[],
                    inputs_schema={"type": "object"},
                    outputs_schema={"type": "object"},
                    default_inputs={},
                )
            ]
        return WorkflowSpec(workflow_id=str(uuid.uuid4()), goal=goal, steps=steps)

    def next_strategy(self, attempt_index: int) -> Strategy | None:
        if attempt_index < len(DEFAULT_STRATEGIES):
            return DEFAULT_STRATEGIES[attempt_index]
        return None

    def write_retry_lesson(self, task_id: str, step_id: str, prev_strategy: Strategy, next_strategy: Strategy,
                           failure_signature: str, note: str) -> str:
        lesson_id = str(uuid.uuid4())
        metadata = {
            "id": lesson_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "step_id": step_id,
            "failure_signature": failure_signature,
            "strategy_id": next_strategy.strategy_id,
            "tags": ["retry", prev_strategy.strategy_id, next_strategy.strategy_id],
            "component": "manager",
        }
        body = (
            f"Retrying step {step_id} for task {task_id}.\n"
            f"Previous strategy: {prev_strategy.strategy_id} ({prev_strategy.strategy_class}).\n"
            f"New strategy: {next_strategy.strategy_id} ({next_strategy.strategy_class}).\n"
            f"Failure signature: {failure_signature}.\n"
            f"Plan change: {note}\n"
        )
        self.memory_store.write_lesson(metadata, body)
        return lesson_id

    def compute_call_signature(self, goal: str, step_id: str, worker_type: str,
                               inputs: dict[str, Any], strategy: Strategy) -> str:
        key_inputs = {
            "inputs": inputs,
        }
        return compute_call_signature(goal, step_id, worker_type, key_inputs, strategy.strategy_id, strategy.retrieval_stage)

    def record_progress(self, task_id: str, call_signature: str, failure_signature: str, metrics: ProgressMetrics) -> None:
        self.retry_gate.record_signature(call_signature)
        if failure_signature:
            self.retry_gate.record_failure(task_id, call_signature, failure_signature)

    def check_retry_allowed(self, task_id: str, call_signature: str, failure_signature: str,
                             prev_metrics: ProgressMetrics | None, new_metrics: ProgressMetrics | None) -> bool:
        decision = self.retry_gate.can_retry(task_id, call_signature, failure_signature, prev_metrics, new_metrics)
        return decision.allow

    def next_call_id(self) -> str:
        self._call_counter += 1
        return f"call-{self._call_counter}"


def _normalize_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    normalized: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = step.get("id") or step.get("step_id")
        if not step_id:
            continue
        normalized.append({**step, "id": str(step_id)})
    return normalized or None
