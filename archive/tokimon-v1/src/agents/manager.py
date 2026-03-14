"""Manager agent: planning, delegation, retry gating."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agents.delegation import DelegationGraph
from agents.retry import RetryGate, compute_call_signature
from agents.worker import Worker
from flow_types import ProgressMetrics, StepStatus
from llm.client import LLMClient
from memory.store import MemoryStore
from tracing import TraceLogger
from workflow.models import StepSpec, WorkflowSpec

if TYPE_CHECKING:
    from skills.gap_detector import SkillGapDetector


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
        memory = [
            lesson.body
            for lesson in self.memory_store.retrieve(
                goal,
                stage=1,
                limit=3,
                component="planner",
                tags=["plan", "component:planner", f"goal:{goal[:80]}"],
                failure_signature="unknown:plan-workflow",
            )
        ]
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

    def write_failure_lesson(
        self,
        task_id: str,
        step_id: str,
        strategy: Strategy,
        failure_signature: str,
        summary: str,
        *,
        component: str,
        tool_name: str | None = None,
        details: str | None = None,
    ) -> str:
        lesson_id = str(uuid.uuid4())
        normalized_failure = str(failure_signature or "").strip()
        if not normalized_failure:
            normalized_failure = f"unknown:{task_id}:{step_id}"
        tools = [str(tool).strip().lower() for tool in (strategy.tool_sequence or []) if str(tool).strip()]
        if tool_name and tool_name.strip():
            tools.append(tool_name.strip().lower())
        tools = sorted(set(tools))
        retrieval_tags = [
            "failure",
            f"task:{task_id}",
            f"step:{step_id}",
            f"component:{component}",
            f"worker:{strategy.worker_type}",
            f"strategy:{strategy.strategy_id}",
            *[f"tool:{tool}" for tool in tools],
        ]
        metadata = {
            "id": lesson_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lesson_type": "failure",
            "task_id": task_id,
            "step_id": step_id,
            "worker_type": strategy.worker_type,
            "retrieval_stage": strategy.retrieval_stage,
            "strategy_id": strategy.strategy_id,
            "failure_signature": normalized_failure,
            "tool_sequence": tools,
            "root_cause_hypothesis": f"Unknown; failure_signature={normalized_failure}.",
            "strategy_change": "none",
            "evidence_of_novelty": "n/a",
            "retrieval_tags": retrieval_tags,
            "tags": ["failure", step_id, strategy.worker_type, strategy.strategy_id],
            "component": component,
        }
        safe_summary = str(summary or "").strip()
        safe_details = str(details or "").strip()
        body_lines = [
            f"Failure in step {step_id} for task {task_id}.",
            f"Strategy: {strategy.strategy_id} ({strategy.strategy_class}).",
            f"Failure signature: {normalized_failure}.",
        ]
        if tools:
            body_lines.append(f"Tools: {', '.join(tools)}.")
        if safe_summary:
            body_lines.append(f"Summary: {safe_summary}")
        if safe_details:
            body_lines.append(f"Details: {safe_details}")
        body = "\n".join(body_lines) + "\n"
        self.memory_store.write_lesson(metadata, body)
        return lesson_id

    def write_retry_lesson(
        self,
        task_id: str,
        step_id: str,
        prev_strategy: Strategy,
        next_strategy: Strategy,
        failure_signature: str,
        note: str,
        *,
        step_description: str | None = None,
        gap_detector: "SkillGapDetector | None" = None,
    ) -> str:
        lesson_id = str(uuid.uuid4())
        normalized_failure = str(failure_signature or "").strip()
        if not normalized_failure:
            normalized_failure = f"unknown:{task_id}:{step_id}"
        description = str(step_description or "").strip()
        description_hash = hashlib.sha1(description.encode("utf-8")).hexdigest()[:10]
        subtask_signature = f"{step_id}|{next_strategy.worker_type}|{description_hash}"
        tools = [str(tool).strip().lower() for tool in (next_strategy.tool_sequence or []) if str(tool).strip()]
        tool_workflow_signature = f"{next_strategy.strategy_id}|{','.join(tools)}"

        metadata = {
            "id": lesson_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lesson_type": "retry",
            "task_id": task_id,
            "step_id": step_id,
            "worker_type": next_strategy.worker_type,
            "retrieval_stage": next_strategy.retrieval_stage,
            "strategy_id": next_strategy.strategy_id,
            "failure_signature": normalized_failure,
            "subtask_signature": subtask_signature,
            "tool_workflow_signature": tool_workflow_signature,
            "tool_sequence": tools,
            "root_cause_hypothesis": f"Unknown; retry after failure_signature={normalized_failure}.",
            "strategy_change": note,
            "evidence_of_novelty": f"Strategy changed from {prev_strategy.strategy_id} to {next_strategy.strategy_id}.",
            "retrieval_tags": ["retry", step_id, next_strategy.worker_type, next_strategy.strategy_id],
            "tags": ["retry", prev_strategy.strategy_id, next_strategy.strategy_id],
            "component": "manager",
        }
        body = (
            f"Retrying step {step_id} for task {task_id}.\n"
            f"Previous strategy: {prev_strategy.strategy_id} ({prev_strategy.strategy_class}).\n"
            f"New strategy: {next_strategy.strategy_id} ({next_strategy.strategy_class}).\n"
            f"Failure signature: {normalized_failure}.\n"
            f"Plan change: {note}\n"
        )
        self.memory_store.write_lesson(metadata, body)
        if gap_detector is not None:
            gap_detector.observe_retry_lesson(metadata)
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

    def memory_informed_retry_gate(
        self,
        *,
        task_id: str,
        step_id: str,
        strategy: Strategy,
        component: str,
        retrieval_tags: list[str],
        failure_signature: str,
    ) -> tuple[bool, str, list[str]]:
        normalized_failure = str(failure_signature or "").strip()
        if not normalized_failure:
            normalized_failure = f"unknown:{task_id}:{step_id}"
        retrieved: list[str] = []
        matching_failures = 0
        for stage in (1, 2, 3):
            lessons = self.memory_store.retrieve(
                step_id,
                stage=stage,
                limit=8,
                component=component,
                tags=retrieval_tags,
                failure_signature=normalized_failure,
            )
            for lesson in lessons:
                lesson_id = str(lesson.metadata.get("id") or "")
                if lesson_id:
                    retrieved.append(lesson_id)
                if lesson.metadata.get("lesson_type") != "failure":
                    continue
                if str(lesson.metadata.get("failure_signature") or "").strip() != normalized_failure:
                    continue
                matching_failures += 1
                if str(lesson.metadata.get("strategy_id") or "").strip() == strategy.strategy_id:
                    return False, "known failure for strategy; force strategy change", retrieved
        if matching_failures >= 2:
            return False, "repeated failure; stop to avoid loop", retrieved
        return True, "no memory block", retrieved

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
