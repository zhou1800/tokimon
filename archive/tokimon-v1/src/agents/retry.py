"""Retry gating and call signature utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from flow_types import ProgressMetrics


def compute_call_signature(
    goal: str,
    step_id: str,
    worker_type: str,
    key_inputs: dict[str, Any],
    strategy_id: str,
    retrieval_stage: int,
) -> str:
    payload = {
        "goal": goal,
        "step_id": step_id,
        "worker_type": worker_type,
        "key_inputs": key_inputs,
        "strategy_id": strategy_id,
        "retrieval_stage": retrieval_stage,
    }
    data = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


@dataclass
class RetryDecision:
    allow: bool
    reason: str


class RetryGate:
    def __init__(self) -> None:
        self.seen_signatures: set[str] = set()
        self.seen_failures: set[str] = set()

    def record_signature(self, signature: str) -> None:
        self.seen_signatures.add(signature)

    def record_failure(self, task_id: str, call_signature: str, failure_signature: str) -> None:
        self.seen_failures.add(_failure_key(task_id, call_signature, failure_signature))

    def can_retry(
        self,
        task_id: str,
        call_signature: str,
        failure_signature: str,
        prev_metrics: ProgressMetrics | None,
        new_metrics: ProgressMetrics | None,
    ) -> RetryDecision:
        has_progress = _has_progress(prev_metrics, new_metrics)
        if prev_metrics is not None and not has_progress:
            return RetryDecision(False, "no measurable progress")

        fail_key = _failure_key(task_id, call_signature, failure_signature)
        if call_signature in self.seen_signatures and not has_progress:
            return RetryDecision(False, "signature repeated without progress")
        if fail_key in self.seen_failures and not has_progress:
            return RetryDecision(False, "failure signature repeated without progress")
        return RetryDecision(True, "retry allowed")


def _failure_key(task_id: str, call_signature: str, failure_signature: str) -> str:
    material = f"{task_id}:{call_signature}:{failure_signature}".encode()
    return hashlib.sha256(material).hexdigest()


def _has_progress(prev: ProgressMetrics | None, new: ProgressMetrics | None) -> bool:
    if not new:
        return False
    if prev is None:
        return True
    if new.failing_tests is not None and prev.failing_tests is not None:
        if new.failing_tests < prev.failing_tests:
            return True
    if new.passed_tests is not None and prev.passed_tests is not None:
        if new.passed_tests > prev.passed_tests:
            return True
    if new.new_artifacts is not None and new.new_artifacts > 0:
        return True
    if new.artifact_delta_hash and new.artifact_delta_hash != prev.artifact_delta_hash:
        return True
    if new.notes and new.notes != prev.notes:
        return True
    return False
