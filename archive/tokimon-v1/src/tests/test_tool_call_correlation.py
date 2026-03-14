from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.worker import Worker
from llm.client import LLMClient
from tools.base import ToolResult
from tracing import TraceLogger


@dataclass
class DummyTool:
    calls: list[str] = field(default_factory=list)

    def echo(self, text: str) -> ToolResult:
        self.calls.append(text)
        return ToolResult(ok=True, summary="echo", data={"text": text}, elapsed_ms=0.1)


class RecordingLLMClient:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.seen_messages: list[list[dict[str, Any]]] = []

    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.seen_messages.append([dict(msg) for msg in messages])
        if self.script:
            return self.script.pop(0)
        return {"status": "FAILURE", "summary": "script exhausted"}


def _read_trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def test_worker_echoes_tool_call_id_in_records_and_tool_message(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace = TraceLogger(trace_path)

    llm: LLMClient = RecordingLLMClient(
        script=[
            {
                "tool_calls": [
                    {
                        "tool": "dummy",
                        "action": "echo",
                        "args": {"text": "hello"},
                        "call_id": "call-123",
                    }
                ]
            },
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

    records = output.metrics["tool_call_records"]
    assert records[0]["call_id"] == "call-123"

    second_prompt = llm.seen_messages[1]
    tool_messages = [
        msg for msg in second_prompt if msg.get("role") == "tool" and msg.get("name") == "dummy"
    ]
    assert tool_messages
    payload = json.loads(tool_messages[0]["content"])
    assert payload["call_id"] == "call-123"

    trace_records = _read_trace(trace_path)
    tool_result = [record for record in trace_records if record.get("event_type") == "worker_tool_result"][0]
    assert tool_result["payload"]["call_id"] == "c1"
    assert tool_result["payload"]["tool_call_id"] == "call-123"
