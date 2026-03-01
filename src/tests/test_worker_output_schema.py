from __future__ import annotations

from agents.worker import Worker
from flow_types import WorkerStatus
from llm.client import MockLLMClient


def test_worker_repairs_invalid_final_output_schema() -> None:
    llm = MockLLMClient(
        script=[
            {"status": "SUCCESS", "summary": "ok"},
            {
                "status": "SUCCESS",
                "summary": "done",
                "artifacts": [],
                "metrics": {},
                "next_actions": [],
                "failure_signature": "",
            },
        ]
    )
    worker = Worker("Implementer", llm, tools={})
    output = worker.run("goal", "step", inputs={}, memory=[])
    assert output.status == WorkerStatus.SUCCESS
    assert output.metrics["model_calls"] == 2
    assert output.metrics["schema_repairs"] == 1


def test_worker_fails_after_bounded_schema_repairs() -> None:
    llm = MockLLMClient(
        script=[
            {"status": "SUCCESS", "summary": "ok"},
            {"status": "SUCCESS", "summary": "still missing keys"},
            {"status": "SUCCESS", "summary": "third time missing keys"},
        ]
    )
    worker = Worker("Implementer", llm, tools={})
    output = worker.run("goal", "step", inputs={}, memory=[])
    assert output.status == WorkerStatus.FAILURE
    assert output.metrics["model_calls"] == 3
    assert output.metrics["schema_repairs"] == 2
    assert output.failure_signature == "worker-output-schema-invalid:missing_required:$.artifacts"
