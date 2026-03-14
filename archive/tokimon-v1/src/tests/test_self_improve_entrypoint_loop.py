from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from llm.client import MockLLMClient
from self_improve.orchestrator import SelfImproveOrchestrator, SelfImproveSettings
from self_improve.orchestrator import _path_charter


def test_entrypoint_loop_blocks_ambiguous_goal_with_clarifying_questions(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")

    master = _init_master_repo(tmp_path)
    settings = SelfImproveSettings(
        sessions_per_batch=1,
        batches=1,
        max_workers=1,
        include_paths=["proj"],
        pytest_args=["proj/tests"],
        merge_on_success=False,
    )
    orchestrator = SelfImproveOrchestrator(master, llm_factory=lambda _sid: MockLLMClient(script=[]), settings=settings)
    report = orchestrator.run("fix it", input_ref=None)

    session = report.batches[0].sessions[0]
    assert session.workflow_status == "BLOCKED"
    assert session.workflow_ok is False
    assert session.entrypoint_attempts == 0
    assert session.clarifying_questions
    assert session.verification_ok is False
    assert session.run_root is None


def test_entrypoint_loop_retries_from_prompt_generation_after_failed_verification(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")

    master = _init_master_repo(tmp_path)

    def llm_factory(_sid: str) -> MockLLMClient:
        return MockLLMClient(
            script=[
                {
                    "status": "SUCCESS",
                    "summary": "planned",
                    "artifacts": [],
                    "metrics": {},
                    "next_actions": [],
                    "failure_signature": "",
                    "workflow": {"steps": [{"id": "work", "worker": "Implementer"}]},
                },
                {
                    "tool_calls": [
                        _experiment_summary_write_tool_call(
                            "1-1",
                            1,
                            "No-op first attempt should not satisfy required change; expect retry.",
                        )
                    ]
                },
                {
                    "status": "SUCCESS",
                    "summary": "no-op",
                    "artifacts": [],
                    "metrics": {},
                    "next_actions": [],
                    "failure_signature": "",
                },
                {
                    "status": "SUCCESS",
                    "summary": "planned retry",
                    "artifacts": [],
                    "metrics": {},
                    "next_actions": [],
                    "failure_signature": "",
                    "workflow": {"steps": [{"id": "work", "worker": "Implementer"}]},
                },
                {
                    "tool_calls": [
                        {
                            "tool": "file",
                            "action": "write",
                            "args": {
                                "path": "proj/app.py",
                                "content": "def add(a, b):\n    # retry attempt wrote this line\n    return a + b\n",
                            },
                        }
                    ]
                },
                {
                    "tool_calls": [
                        _experiment_summary_write_tool_call(
                            "1-1",
                            2,
                            "Updating add() to return sum satisfies tests and produces required concrete change.",
                        )
                    ]
                },
                {
                    "status": "SUCCESS",
                    "summary": "updated file",
                    "artifacts": [],
                    "metrics": {},
                    "next_actions": [],
                    "failure_signature": "",
                },
            ]
        )

    settings = SelfImproveSettings(
        sessions_per_batch=1,
        batches=1,
        max_workers=1,
        include_paths=["proj"],
        pytest_args=["proj/tests"],
        merge_on_success=True,
        entrypoint_max_attempts=3,
    )
    orchestrator = SelfImproveOrchestrator(master, llm_factory=llm_factory, settings=settings)
    report = orchestrator.run("Fix proj.add behavior.", input_ref=None)

    batch = report.batches[0]
    session = batch.sessions[0]
    assert session.entrypoint_attempts == 2
    assert len(session.attempts) == 2
    assert session.attempts[0]["verification"]["ok"] is False
    assert session.attempts[1]["verification"]["ok"] is True
    assert session.verification_ok is True
    assert "proj/app.py" in session.changed_files
    assert batch.merged is True
    assert "retry attempt wrote this line" in (master / "proj" / "app.py").read_text()


def _init_master_repo(tmp_path: Path) -> Path:
    master = tmp_path / "master"
    (master / "proj").mkdir(parents=True, exist_ok=True)
    (master / "proj" / "__init__.py").write_text("")
    (master / "proj" / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (master / "proj" / "tests").mkdir(parents=True, exist_ok=True)
    (master / ".gitignore").write_text(".tokimon-tmp/\nruns/\n__pycache__/\n.pytest_cache/\n")
    (master / "proj" / "tests" / "test_app.py").write_text(
        "from proj.app import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    _git(master, ["init"])
    _git(master, ["config", "user.email", "test@example.com"])
    _git(master, ["config", "user.name", "Test User"])
    _git(master, ["add", "."])
    _git(master, ["commit", "-m", "init"])
    return master


def _git(cwd: Path, args: list[str]) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def _experiment_summary_write_tool_call(session_id: str, attempt_index: int, hypothesis: str) -> dict[str, object]:
    payload = {
        "causal_mechanism_hypothesis": hypothesis,
        "pass_condition": "Maintain evaluation ok while keeping session energy within the planned energy budget.",
        "baseline_evaluation": {"ok": True, "passed": 1, "failed": 0, "failing_tests": []},
        "post_change_evaluation": {"ok": True, "passed": 1, "failed": 0, "failing_tests": []},
        "delta": {"passed": 0, "failed": 0},
        "plan": ["Baseline eval", "Smallest change", "Re-run eval", "Report delta"],
        "path_charter": _path_charter(session_id),
        "self_critique": "Low risk for test harness; confidence medium because only entrypoint loop behavior changes.",
        "lessons": ["Ensure experiment summaries include protocol fields for deterministic selection/reporting."],
    }
    return {
        "tool": "file",
        "action": "write",
        "args": {
            "path": f".tokimon-tmp/self-improve/experiment/{session_id}/attempt-{attempt_index}.json",
            "content": json.dumps(payload),
        },
    }
