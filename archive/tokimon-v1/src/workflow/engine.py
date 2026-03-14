"""Workflow engine with DAG validation, persistence, and resume support."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flow_types import StepStatus

from .models import StepAttempt, StepSpec, StepState, WorkflowSpec, WorkflowState
from .schema import validate_schema


class WorkflowEngine:
    def __init__(self, spec: WorkflowSpec) -> None:
        self.spec = spec
        self._step_map = spec.step_map()
        self.state = WorkflowState(
            workflow_id=spec.workflow_id,
            goal=spec.goal,
            steps={step.step_id: StepState(step_id=step.step_id, inputs=dict(step.default_inputs)) for step in spec.steps},
            metadata=dict(spec.metadata),
        )
        self._validate_dag()
        self._refresh_ready()

    def _validate_dag(self) -> None:
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node: str) -> None:
            if node in in_stack:
                raise ValueError(f"Cycle detected in workflow at step '{node}'")
            if node in visited:
                return
            in_stack.add(node)
            visited.add(node)
            for dep in self._step_map[node].depends_on:
                if dep not in self._step_map:
                    raise ValueError(f"Unknown dependency '{dep}' for step '{node}'")
                dfs(dep)
            in_stack.remove(node)

        for step_id in self._step_map:
            dfs(step_id)

    def _refresh_ready(self) -> None:
        for step_id, state in self.state.steps.items():
            if state.status in {
                StepStatus.SUCCEEDED,
                StepStatus.SKIPPED,
                StepStatus.RUNNING,
                StepStatus.FAILED,
                StepStatus.PARTIAL,
                StepStatus.BLOCKED,
            }:
                continue
            deps = self._step_map[step_id].depends_on
            if all(self.state.steps[dep].status in {StepStatus.SUCCEEDED, StepStatus.PARTIAL, StepStatus.SKIPPED} for dep in deps):
                state.status = StepStatus.READY
            else:
                state.status = StepStatus.NEW

    def ready_steps(self) -> list[str]:
        self._refresh_ready()
        return [step_id for step_id, state in self.state.steps.items() if state.status == StepStatus.READY]

    def mark_running(self, step_id: str) -> None:
        self.state.steps[step_id].status = StepStatus.RUNNING

    def record_attempt(self, step_id: str, attempt: StepAttempt) -> None:
        self.state.steps[step_id].attempts.append(attempt)

    def mark_outputs(self, step_id: str, outputs: dict[str, Any]) -> None:
        step_spec = self._step_map[step_id]
        if step_spec.outputs_schema:
            validate_schema(outputs, step_spec.outputs_schema)
        self.state.steps[step_id].outputs = outputs

    def mark_status(self, step_id: str, status: StepStatus, error: str | None = None) -> None:
        state = self.state.steps[step_id]
        state.status = status
        state.error = error

    def skip_remaining(self, *, reason: str, triggered_by: str) -> list[str]:
        """Mark all non-terminal steps as SKIPPED.

        Intended for early-termination flows when the goal is already satisfied.
        Returns the list of step ids that were skipped.
        """

        skipped: list[str] = []
        for step_id, state in self.state.steps.items():
            if state.status in {StepStatus.SUCCEEDED, StepStatus.PARTIAL, StepStatus.SKIPPED}:
                continue
            if state.status in {StepStatus.FAILED, StepStatus.BLOCKED, StepStatus.RUNNING}:
                continue
            state.status = StepStatus.SKIPPED
            if not state.outputs:
                state.outputs = {"summary": f"skipped: {reason}"}
            skipped.append(step_id)

        termination = self.state.metadata.get("termination")
        if not isinstance(termination, dict):
            termination = {}
        termination.update({"triggered_by": triggered_by, "reason": reason, "skipped_steps": skipped})
        self.state.metadata["termination"] = termination
        return skipped

    def set_inputs(self, step_id: str, inputs: dict[str, Any]) -> None:
        step_spec = self._step_map[step_id]
        if step_spec.inputs_schema:
            validate_schema(inputs, step_spec.inputs_schema)
        self.state.steps[step_id].inputs = inputs

    def save(self, path: Path) -> None:
        payload = {
            "spec": _serialize_spec(self.spec),
            "state": _serialize_state(self.state),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "WorkflowEngine":
        payload = json.loads(path.read_text())
        spec = _deserialize_spec(payload["spec"])
        engine = cls(spec)
        engine.state = _deserialize_state(payload["state"])
        return engine


def _serialize_spec(spec: WorkflowSpec) -> dict[str, Any]:
    return {
        "workflow_id": spec.workflow_id,
        "goal": spec.goal,
        "metadata": spec.metadata,
        "steps": [asdict(step) for step in spec.steps],
    }


def _deserialize_spec(data: dict[str, Any]) -> WorkflowSpec:
    steps = [StepSpec(**step) for step in data["steps"]]
    return WorkflowSpec(workflow_id=data["workflow_id"], goal=data["goal"], steps=steps, metadata=data.get("metadata", {}))


def _serialize_state(state: WorkflowState) -> dict[str, Any]:
    return {
        "workflow_id": state.workflow_id,
        "goal": state.goal,
        "metadata": state.metadata,
        "steps": {
            step_id: {
                "step_id": step.step_id,
                "status": step.status.value,
                "inputs": step.inputs,
                "outputs": step.outputs,
                "error": step.error,
                "attempts": [
                    {
                        "attempt_id": attempt.attempt_id,
                        "status": attempt.status.value,
                        "call_signature": attempt.call_signature,
                        "worker_type": attempt.worker_type,
                        "strategy_id": attempt.strategy_id,
                        "retrieval_stage": attempt.retrieval_stage,
                        "summary": attempt.summary,
                        "failure_signature": attempt.failure_signature,
                        "progress_metrics": attempt.progress_metrics,
                        "artifacts": attempt.artifacts,
                    }
                    for attempt in step.attempts
                ],
            }
            for step_id, step in state.steps.items()
        },
    }


def _deserialize_state(data: dict[str, Any]) -> WorkflowState:
    steps: dict[str, StepState] = {}
    for step_id, step_data in data["steps"].items():
        attempts = [
            StepAttempt(
                attempt_id=attempt["attempt_id"],
                status=StepStatus(attempt["status"]),
                call_signature=attempt["call_signature"],
                worker_type=attempt["worker_type"],
                strategy_id=attempt["strategy_id"],
                retrieval_stage=attempt["retrieval_stage"],
                summary=attempt.get("summary"),
                failure_signature=attempt.get("failure_signature"),
                progress_metrics=attempt.get("progress_metrics", {}),
                artifacts=attempt.get("artifacts", []),
            )
            for attempt in step_data.get("attempts", [])
        ]
        steps[step_id] = StepState(
            step_id=step_id,
            status=StepStatus(step_data["status"]),
            attempts=attempts,
            inputs=step_data.get("inputs", {}),
            outputs=step_data.get("outputs", {}),
            error=step_data.get("error"),
        )
    return WorkflowState(
        workflow_id=data["workflow_id"],
        goal=data["goal"],
        steps=steps,
        metadata=data.get("metadata", {}),
    )
