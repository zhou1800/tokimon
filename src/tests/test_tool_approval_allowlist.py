"""Tests for Phase 4 approval allowlists."""

from __future__ import annotations

import json
from pathlib import Path

from policy.tool_approval import (
    _load_allowlist_from_env,
    _load_allowlist_from_file,
    approval_id_for,
    check_allowlist,
    load_approval_allowlist,
)
from policy.tool_loop_detection import stable_args_hash

from agents.worker import Worker
from flow_types import WorkerStatus
from llm.client import MockLLMClient
from tools.base import ToolResult


# ---------------------------------------------------------------------------
# Unit: allowlist loading
# ---------------------------------------------------------------------------


def test_load_allowlist_from_env_empty() -> None:
    assert _load_allowlist_from_env({"OTHER": "x"}) == set()
    assert _load_allowlist_from_env({}) == set()


def test_load_allowlist_from_env_csv() -> None:
    ids = _load_allowlist_from_env({"TOKIMON_TOOL_APPROVAL_ALLOWLIST": "abc, def , ghi"})
    assert ids == {"abc", "def", "ghi"}


def test_load_allowlist_from_file_valid(tmp_path: Path) -> None:
    al_dir = tmp_path / ".tokimon-tmp" / "approvals"
    al_dir.mkdir(parents=True)
    (al_dir / "allowlist.json").write_text(json.dumps({"allowlist": ["id1", "id2"]}))
    ids = _load_allowlist_from_file(tmp_path)
    assert ids == {"id1", "id2"}


def test_load_allowlist_from_file_missing(tmp_path: Path) -> None:
    ids = _load_allowlist_from_file(tmp_path)
    assert ids == set()


def test_load_allowlist_from_file_malformed_json(tmp_path: Path) -> None:
    al_dir = tmp_path / ".tokimon-tmp" / "approvals"
    al_dir.mkdir(parents=True)
    (al_dir / "allowlist.json").write_text("NOT JSON")
    ids = _load_allowlist_from_file(tmp_path)
    assert ids == set()


def test_load_allowlist_from_file_wrong_shape(tmp_path: Path) -> None:
    al_dir = tmp_path / ".tokimon-tmp" / "approvals"
    al_dir.mkdir(parents=True)
    (al_dir / "allowlist.json").write_text(json.dumps(["not", "a", "dict"]))
    ids = _load_allowlist_from_file(tmp_path)
    assert ids == set()


def test_load_allowlist_from_file_missing_key(tmp_path: Path) -> None:
    al_dir = tmp_path / ".tokimon-tmp" / "approvals"
    al_dir.mkdir(parents=True)
    (al_dir / "allowlist.json").write_text(json.dumps({"other": []}))
    ids = _load_allowlist_from_file(tmp_path)
    assert ids == set()


# ---------------------------------------------------------------------------
# Unit: check_allowlist
# ---------------------------------------------------------------------------


def test_check_allowlist_env_wins() -> None:
    approved, source = check_allowlist("x", {"x"}, {"x"})
    assert approved is True
    assert source == "env"


def test_check_allowlist_file_fallback() -> None:
    approved, source = check_allowlist("x", set(), {"x"})
    assert approved is True
    assert source == "file"


def test_check_allowlist_not_found() -> None:
    approved, source = check_allowlist("x", {"y"}, {"z"})
    assert approved is False
    assert source == ""


# ---------------------------------------------------------------------------
# Integration: Worker + allowlist in block mode
# ---------------------------------------------------------------------------


_FILE_WRITE_ARGS = {"path": "tmp/blocked.txt", "content": "hi"}
_FILE_WRITE_ARGS_HASH = stable_args_hash(_FILE_WRITE_ARGS)
_FILE_WRITE_APPROVAL_ID = approval_id_for("file", "write", _FILE_WRITE_ARGS_HASH)


def test_worker_block_mode_allowlist_env_bypasses(monkeypatch) -> None:
    """block mode + approval_id in env allowlist -> tool proceeds, SUCCESS."""
    monkeypatch.setenv("TOKIMON_TOOL_APPROVAL_MODE", "block")
    monkeypatch.setenv("TOKIMON_TOOL_APPROVAL_ALLOWLIST", _FILE_WRITE_APPROVAL_ID)

    from dataclasses import dataclass, field

    @dataclass
    class FakeFileTool:
        writes: list[dict] = field(default_factory=list)

        def write(self, path: str, content: str) -> ToolResult:
            self.writes.append({"path": path, "content": content})
            return ToolResult(ok=True, summary="write ok", data={"path": path}, elapsed_ms=0.1)

    tool = FakeFileTool()
    llm = MockLLMClient(
        script=[
            {"tool_calls": [{"tool": "file", "action": "write", "args": _FILE_WRITE_ARGS}]},
            {"status": "SUCCESS", "summary": "done", "artifacts": [], "metrics": {}, "next_actions": [], "failure_signature": ""},
        ]
    )
    worker = Worker("Implementer", llm, tools={"file": tool})
    output = worker.run("goal", "step", inputs={}, memory=[])

    assert output.status == WorkerStatus.SUCCESS
    assert len(tool.writes) == 1
    rec = output.metrics["tool_call_records"][0]
    assert rec["policy_decision"]["pre_approved"] is True
    assert rec["policy_decision"]["allowlist_source"] == "env"


def test_worker_deny_mode_allowlist_file_bypasses(monkeypatch, tmp_path: Path) -> None:
    """deny mode + approval_id in file allowlist -> tool proceeds, SUCCESS."""
    monkeypatch.setenv("TOKIMON_TOOL_APPROVAL_MODE", "deny")
    monkeypatch.delenv("TOKIMON_TOOL_APPROVAL_ALLOWLIST", raising=False)

    al_dir = tmp_path / ".tokimon-tmp" / "approvals"
    al_dir.mkdir(parents=True)
    (al_dir / "allowlist.json").write_text(json.dumps({"allowlist": [_FILE_WRITE_APPROVAL_ID]}))

    monkeypatch.chdir(tmp_path)

    from dataclasses import dataclass, field

    @dataclass
    class FakeFileTool:
        writes: list[dict] = field(default_factory=list)

        def write(self, path: str, content: str) -> ToolResult:
            self.writes.append({"path": path, "content": content})
            return ToolResult(ok=True, summary="write ok", data={"path": path}, elapsed_ms=0.1)

    tool = FakeFileTool()
    llm = MockLLMClient(
        script=[
            {"tool_calls": [{"tool": "file", "action": "write", "args": _FILE_WRITE_ARGS}]},
            {"status": "SUCCESS", "summary": "done", "artifacts": [], "metrics": {}, "next_actions": [], "failure_signature": ""},
        ]
    )
    worker = Worker("Implementer", llm, tools={"file": tool})
    output = worker.run("goal", "step", inputs={}, memory=[])

    assert output.status == WorkerStatus.SUCCESS
    assert len(tool.writes) == 1
    rec = output.metrics["tool_call_records"][0]
    assert rec["policy_decision"]["pre_approved"] is True
    assert rec["policy_decision"]["allowlist_source"] == "file"

