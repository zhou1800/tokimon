"""Agent output structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flow_types import WorkerStatus


@dataclass
class WorkerOutput:
    status: WorkerStatus
    summary: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    next_actions: list[str] = field(default_factory=list)
    failure_signature: str = ""
    data: dict[str, Any] = field(default_factory=dict)
