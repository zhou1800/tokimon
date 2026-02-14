"""Workflow models and state definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flow_types import StepStatus


@dataclass
class StepSpec:
    step_id: str
    name: str
    description: str
    worker: str
    depends_on: list[str] = field(default_factory=list)
    inputs_schema: dict[str, Any] = field(default_factory=dict)
    outputs_schema: dict[str, Any] = field(default_factory=dict)
    default_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowSpec:
    workflow_id: str
    goal: str
    steps: list[StepSpec]
    metadata: dict[str, Any] = field(default_factory=dict)

    def step_map(self) -> dict[str, StepSpec]:
        return {step.step_id: step for step in self.steps}


@dataclass
class StepAttempt:
    attempt_id: int
    status: StepStatus
    call_signature: str
    worker_type: str
    strategy_id: str
    retrieval_stage: int
    summary: str | None = None
    failure_signature: str | None = None
    progress_metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepState:
    step_id: str
    status: StepStatus = StepStatus.NEW
    attempts: list[StepAttempt] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def last_attempt(self) -> StepAttempt | None:
        return self.attempts[-1] if self.attempts else None


@dataclass
class WorkflowState:
    workflow_id: str
    goal: str
    steps: dict[str, StepState]
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        terminal = {StepStatus.SUCCEEDED, StepStatus.PARTIAL, StepStatus.SKIPPED}
        return all(state.status in terminal for state in self.steps.values())
