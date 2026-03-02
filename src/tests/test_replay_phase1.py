from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agents.worker import Worker
from cli import main as cli_main
from llm.client import MockLLMClient
from replay import ReplayRecorder
from runners.baseline import BaselineRunner
from tools.base import ToolResult


@dataclass
class DummyTool:
    calls: int = 0

    def ping(self) -> ToolResult:
        self.calls += 1
        return ToolResult(ok=True, summary="pong", data={}, elapsed_ms=0.0)


def test_tool_call_records_include_policy_decision() -> None:
    llm = MockLLMClient(
        script=[
            {"tool_calls": [{"tool": "dummy", "action": "ping", "args": {}}]},
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
    output = worker.run("goal", "step-1", inputs={}, memory=[])

    records = output.metrics["tool_call_records"]
    assert len(records) == 1
    policy = records[0]["policy_decision"]
    assert policy["decision"] == "allow"
    assert policy["policy_id"] == "default-v1"
    assert policy["risk_tier"] in {"low", "medium", "high"}
    assert policy["risk_tier"] == "low"
    assert policy.get("requires_approval") is False
    assert isinstance(policy["reason"], str) and policy["reason"]


@dataclass
class CountingFileTool:
    writes: int = 0

    def write(self, path: str, content: str) -> ToolResult:
        self.writes += 1
        return ToolResult(ok=True, summary="wrote", data={"path": path, "content": content}, elapsed_ms=0.0)


def test_side_effect_tool_calls_are_deduped_with_cached_flag() -> None:
    llm = MockLLMClient(
        script=[
            {
                "tool_calls": [
                    {"tool": "file", "action": "write", "args": {"path": "a.txt", "content": "hello"}},
                    {"tool": "file", "action": "write", "args": {"path": "a.txt", "content": "hello"}},
                ]
            },
            {
                "status": "SUCCESS",
                "summary": "ok",
                "artifacts": [],
                "metrics": {},
                "next_actions": [],
                "failure_signature": "",
            },
        ]
    )
    tool = CountingFileTool()
    worker = Worker("Implementer", llm, tools={"file": tool})
    output = worker.run("goal", "step-1", inputs={}, memory=[])

    assert tool.writes == 1
    records = output.metrics["tool_call_records"]
    assert len(records) == 2
    assert records[0]["cached"] is False
    assert records[1]["cached"] is True
    assert records[0]["policy_decision"]["risk_tier"] == "high"
    assert records[0]["policy_decision"].get("requires_approval") is True


def test_replay_artifact_written_and_cli_replays(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runs_root = tmp_path / "runs"
    llm = MockLLMClient(
        script=[
            {
                "status": "SUCCESS",
                "summary": "baseline ok",
                "artifacts": [],
                "metrics": {},
                "next_actions": [],
                "failure_signature": "",
            }
        ]
    )
    runner = BaselineRunner(workspace, llm, base_dir=runs_root)
    result = runner.run("goal", task_id="t1", test_args=None)

    replay_path = result.run_context.artifacts_dir / "single-step" / "replay.json"
    assert replay_path.exists()
    replay_record = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay_record["schema_version"] == "1.0"
    assert replay_record["step_id"] == "single-step"
    assert isinstance(replay_record["model_script"], list)
    assert isinstance(replay_record["tool_script"], list)

    assert cli_main(["replay", "--run-path", str(result.run_context.root)]) == 0


def test_replay_redacts_bearer_tokens() -> None:
    recorder = ReplayRecorder(
        step_id="s1",
        worker_role="Implementer",
        goal="g",
        inputs={},
        memory=[],
    )
    recorder.record_model_response(
        {
            "status": "SUCCESS",
            "summary": "Authorization: Bearer supersecret",
            "artifacts": [],
            "metrics": {},
            "next_actions": [],
            "failure_signature": "",
        }
    )
    recorder.record_final(
        {
            "status": "SUCCESS",
            "summary": "Authorization: Bearer supersecret",
            "failure_signature": "",
            "metrics": {},
        }
    )
    replay_record = recorder.build()
    payload = json.dumps(replay_record, sort_keys=True)
    assert "<REDACTED>" in payload
    assert "supersecret" not in payload
