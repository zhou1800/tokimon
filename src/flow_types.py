"""Shared types and enums for Tokimon."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepStatus(str, Enum):
    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    PARTIAL = "PARTIAL"
    RETRY_PENDING = "RETRY_PENDING"


class WorkerStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    BLOCKED = "BLOCKED"
    PARTIAL = "PARTIAL"


@dataclass
class ArtifactRef:
    path: str
    kind: str
    step_id: str | None = None
    hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressMetrics:
    failing_tests: int | None = None
    passed_tests: int | None = None
    new_artifacts: int | None = None
    artifact_delta_hash: str | None = None
    notes: str | None = None


@dataclass
class ToolCallRecord:
    tool_name: str
    ok: bool
    summary: str
    data: dict[str, Any]
    elapsed_ms: float
    error: str | None = None
