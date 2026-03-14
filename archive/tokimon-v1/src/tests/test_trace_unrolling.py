from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.worker import Worker
from llm.client import MockLLMClient
from tools.base import ToolResult
from tracing import TraceLogger


@dataclass
class DummyTool:
    calls: list[str] = field(default_factory=list)

    def echo(self, text: str) -> ToolResult:
        self.calls.append(text)
        return ToolResult(ok=True, summary="echo", data={"text": text}, elapsed_ms=0.1)


def _read_trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def test_worker_writes_trace_jsonl_unrolled_loop(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace = TraceLogger(trace_path)
    long_text = "x" * 5000

    llm = MockLLMClient(
        script=[
            {"tool_calls": [{"tool": "dummy", "action": "echo", "args": {"text": long_text}}]},
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
    tool = DummyTool()
    worker = Worker("Implementer", llm, tools={"dummy": tool})
    output = worker.run(
        "goal",
        "step-1",
        inputs={},
        memory=[],
        trace=trace,
        trace_context={"task_id": "t1", "call_id": "c1"},
    )
    assert output.summary == "done"
    assert trace_path.exists()

    records = _read_trace(trace_path)
    assert [record.get("event_type") for record in records] == [
        "worker_model_call",
        "worker_model_response",
        "worker_tool_call",
        "worker_tool_result",
        "worker_model_call",
        "worker_model_response",
        "worker_final",
    ]
    for record in records:
        assert "timestamp" in record
        payload = record.get("payload")
        assert isinstance(payload, dict)
        assert payload.get("worker_role") == "Implementer"
        assert payload.get("step_id") == "step-1"
        assert payload.get("task_id") == "t1"
        assert payload.get("call_id") == "c1"

    model_response = records[1]["payload"]
    assert model_response["response_type"] == "tool_calls"
    assert model_response["tool_calls_count"] == 1

    traced_call = records[2]["payload"]["call"]
    assert traced_call["tool"] == "dummy"
    traced_text = traced_call["args"]["text"]
    assert isinstance(traced_text, str)
    assert "truncated" in traced_text
    assert len(traced_text) < len(long_text)
