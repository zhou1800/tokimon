from __future__ import annotations

from dataclasses import dataclass, field

from agents.worker import Worker
from flow_types import WorkerStatus
from llm.client import MockLLMClient
from tools.base import ToolResult


@dataclass
class DummyTool:
    calls: list[str] = field(default_factory=list)

    def echo(self, text: str) -> ToolResult:
        self.calls.append(text)
        return ToolResult(ok=True, summary="echo", data={"text": text}, elapsed_ms=0.1)


def test_worker_executes_tool_calls_and_counts() -> None:
    dummy = DummyTool()
    llm = MockLLMClient(
        script=[
            {"tool_calls": [{"tool": "dummy", "action": "echo", "args": {"text": "hi"}}]},
            {"status": "SUCCESS", "summary": "done", "artifacts": [], "metrics": {}, "next_actions": [], "failure_signature": ""},
        ]
    )
    worker = Worker("Implementer", llm, tools={"dummy": dummy})
    output = worker.run("goal", "step", inputs={}, memory=[])
    assert output.status == WorkerStatus.SUCCESS
    assert dummy.calls == ["hi"]
    assert output.metrics["model_calls"] == 2
    assert output.metrics["tool_calls"] == 1
    assert output.metrics["iteration_count"] == 2


def test_worker_tool_loop_detection_repeated_signature(monkeypatch) -> None:
    monkeypatch.setenv("TOKIMON_TOOL_LOOP_DETECTION_ENABLED", "true")
    monkeypatch.setenv("TOKIMON_TOOL_LOOP_REPEAT_THRESHOLD", "2")

    dummy = DummyTool()
    llm = MockLLMClient(
        script=[
            {"tool_calls": [{"tool": "dummy", "action": "echo", "args": {"text": "hi"}}]},
            {"tool_calls": [{"tool": "dummy", "action": "echo", "args": {"text": "hi"}}]},
        ]
    )
    worker = Worker("Implementer", llm, tools={"dummy": dummy})
    output = worker.run("goal", "step", inputs={}, memory=[])

    assert output.status == WorkerStatus.PARTIAL
    assert str(output.failure_signature).startswith("worker-tool-loop-detected:")
    assert dummy.calls == ["hi", "hi"]
    assert output.metrics["tool_calls"] == 2
    evidence = output.metrics["tool_loop_detection"]
    assert evidence["trigger"]["reason"] == "repeat_signature"
    assert evidence["trigger"]["count"] == 2


def test_worker_tool_loop_detection_repeated_failures(monkeypatch) -> None:
    monkeypatch.setenv("TOKIMON_TOOL_LOOP_DETECTION_ENABLED", "true")
    monkeypatch.setenv("TOKIMON_TOOL_LOOP_REPEAT_THRESHOLD", "2")

    @dataclass
    class FailingTool:
        calls: list[str] = field(default_factory=list)

        def echo(self, text: str) -> ToolResult:
            self.calls.append(text)
            return ToolResult(ok=False, summary="fail", data={"text": text}, elapsed_ms=0.1, error="nope")

    tool = FailingTool()
    llm = MockLLMClient(
        script=[
            {"tool_calls": [{"tool": "dummy", "action": "echo", "args": {"text": "hi"}}]},
            {"tool_calls": [{"tool": "dummy", "action": "echo", "args": {"text": "hi"}}]},
        ]
    )
    worker = Worker("Implementer", llm, tools={"dummy": tool})
    output = worker.run("goal", "step", inputs={}, memory=[])

    assert output.status == WorkerStatus.PARTIAL
    evidence = output.metrics["tool_loop_detection"]
    assert evidence["trigger"]["reason"] == "repeat_failure"
    assert evidence["trigger"]["failure_count"] == 2


def test_worker_tool_approval_block_mode(monkeypatch) -> None:
    monkeypatch.setenv("TOKIMON_TOOL_APPROVAL_MODE", "block")

    llm = MockLLMClient(
        script=[
            {"tool_calls": [{"tool": "file", "action": "write", "args": {"path": "tmp/blocked.txt", "content": "hi"}}]},
        ]
    )
    worker = Worker("Implementer", llm, tools={})
    output = worker.run("goal", "step", inputs={}, memory=[])

    assert output.status == WorkerStatus.BLOCKED
    assert output.failure_signature == "worker-tool-approval-blocked"
    approval_request = output.metrics["approval_request"]
    assert approval_request["tool"] == "file"
    assert approval_request["action"] == "write"
    assert isinstance(approval_request["approval_id"], str) and approval_request["approval_id"]


def test_worker_tool_approval_deny_mode_records_error(monkeypatch) -> None:
    monkeypatch.setenv("TOKIMON_TOOL_APPROVAL_MODE", "deny")

    llm = MockLLMClient(
        script=[
            {"tool_calls": [{"tool": "file", "action": "write", "args": {"path": "tmp/denied.txt", "content": "hi"}}]},
            {"status": "SUCCESS", "summary": "done", "artifacts": [], "metrics": {}, "next_actions": [], "failure_signature": ""},
        ]
    )
    worker = Worker("Implementer", llm, tools={})
    output = worker.run("goal", "step", inputs={}, memory=[])

    assert output.status == WorkerStatus.SUCCESS
    assert output.metrics["tool_calls"] == 1
    denied = output.metrics["tool_call_records"][0]
    assert denied["tool_name"] == "file"
    assert denied["ok"] is False
    assert denied["summary"] == "denied (approval required)"
