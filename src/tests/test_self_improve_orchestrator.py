from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from llm.client import MockLLMClient
from self_improve.orchestrator import EvaluationResult
from self_improve.orchestrator import SelfImproveOrchestrator, SelfImproveSessionResult, SelfImproveSettings
from self_improve.workspace import clone_master, compute_changes


def test_self_improve_runs_sessions_and_merges_winner(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")

    master = tmp_path / "master"
    (master / "proj").mkdir(parents=True, exist_ok=True)
    (master / "proj" / "__init__.py").write_text("")
    (master / "proj" / "app.py").write_text("def add(a, b):\n    return a - b\n")
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

    def llm_factory(session_id: str) -> MockLLMClient:
        plan = {
            "status": "SUCCESS",
            "summary": "planned",
            "workflow": {"steps": [{"id": "fix", "worker": "Implementer"}]},
        }
        if session_id.endswith("-1"):
            return MockLLMClient(
                script=[
                    plan,
                    {"tool_calls": [{"tool": "file", "action": "write", "args": {"path": "proj/app.py", "content": "def add(a, b):\n    return a + b\n"}}]},
                    {"status": "SUCCESS", "summary": "fixed"},
                ]
            )
        return MockLLMClient(
            script=[
                plan,
                {"status": "SUCCESS", "summary": "no-op"},
            ]
        )

    settings = SelfImproveSettings(
        sessions_per_batch=2,
        batches=1,
        max_workers=2,
        include_paths=["proj"],
        pytest_args=["proj/tests"],
        merge_on_success=True,
    )
    orchestrator = SelfImproveOrchestrator(master, llm_factory=llm_factory, settings=settings)
    report = orchestrator.run("Fix proj.add", input_ref=None)

    assert report.batches[0].winner_session_id is not None
    assert report.batches[0].merged is True
    assert report.batches[0].master_evaluation is not None
    assert report.batches[0].master_evaluation.ok is True
    assert "return a + b" in (master / "proj" / "app.py").read_text()


def test_self_improve_merge_resolves_conflicts_and_commits(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")

    master = tmp_path / "master"
    (master / "proj").mkdir(parents=True, exist_ok=True)
    (master / "proj" / "__init__.py").write_text("")
    (master / "proj" / "app.py").write_text("def add(a, b):\n    return a - b\n")
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

    workspace = tmp_path / "workspace"
    clone_master(master, workspace, include_paths=["proj"])
    (workspace / "proj" / "app.py").write_text("def add(a, b):\n    return a + b\n")

    # Advance master with a conflicting change so the winner commit must be merged with conflict resolution.
    (master / "proj" / "app.py").write_text("def add(a, b):\n    return a * b\n")
    _git(master, ["add", "proj/app.py"])
    _git(master, ["commit", "-m", "conflicting change"])

    changes = compute_changes(master, workspace, include_paths=["proj"])
    settings = SelfImproveSettings(
        sessions_per_batch=1,
        batches=1,
        max_workers=1,
        include_paths=["proj"],
        pytest_args=["proj/tests"],
        merge_on_success=True,
    )
    orchestrator = SelfImproveOrchestrator(master, llm_factory=lambda _sid: MockLLMClient(script=[]), settings=settings)
    winner = SelfImproveSessionResult(
        session_id="1-1",
        workspace_root=str(workspace),
        run_root=None,
        workflow_ok=True,
        workflow_status="SUCCEEDED",
        workflow_error=None,
        evaluation=EvaluationResult(ok=True, passed=1, failed=0, failing_tests=[], elapsed_s=0.0),
        score=(1, 1, 1, 1, 1),
        model_calls=0,
        tool_calls=0,
        changed_files=[change.relpath for change in changes],
        changes=changes,
        error=None,
    )

    merged, master_eval = orchestrator._merge_winner(winner)

    assert merged is True
    assert master_eval is not None and master_eval.ok is True
    assert "return a + b" in (master / "proj" / "app.py").read_text()


def test_compute_changes_detects_added_files_when_workspace_path_contains_runs(tmp_path: Path) -> None:
    master = tmp_path / "master"
    (master / "proj").mkdir(parents=True, exist_ok=True)
    (master / "proj" / "app.py").write_text("print('hello')\n")

    workspace_root = tmp_path / "runs" / "session-workspace"
    clone_master(master, workspace_root, include_paths=["proj"])
    (workspace_root / "proj" / "new_file.txt").write_text("new\n")

    changes = compute_changes(master, workspace_root, include_paths=["proj"])
    assert any(
        change.kind == "add" and change.relpath == str(Path("proj") / "new_file.txt")
        for change in changes
    )


def test_self_improve_runs_all_batches_when_merge_disabled(tmp_path: Path) -> None:
    master = tmp_path / "master"
    (master / "proj").mkdir(parents=True, exist_ok=True)
    (master / "proj" / "__init__.py").write_text("")
    (master / "proj" / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (master / "proj" / "tests").mkdir(parents=True, exist_ok=True)
    (master / "proj" / "tests" / "test_app.py").write_text(
        "from proj.app import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )

    def llm_factory(_session_id: str) -> MockLLMClient:
        plan = {
            "status": "SUCCESS",
            "summary": "planned",
            "workflow": {"steps": [{"id": "noop", "worker": "Implementer"}]},
        }
        return MockLLMClient(
            script=[
                plan,
                {"status": "SUCCESS", "summary": "done", "artifacts": [], "metrics": {}, "next_actions": [], "failure_signature": ""},
            ]
        )

    settings = SelfImproveSettings(
        sessions_per_batch=1,
        batches=2,
        max_workers=1,
        include_paths=["proj"],
        pytest_args=["proj/tests"],
        merge_on_success=False,
    )
    orchestrator = SelfImproveOrchestrator(master, llm_factory=llm_factory, settings=settings)
    report = orchestrator.run("No-op", input_ref=None)

    assert len(report.batches) == 2
    assert report.batches[0].merged is False
    assert report.batches[1].merged is False


def test_self_improve_continues_after_failed_batch_then_merges(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")

    master = tmp_path / "master"
    (master / "proj").mkdir(parents=True, exist_ok=True)
    (master / "proj" / "__init__.py").write_text("")
    (master / "proj" / "app.py").write_text("def add(a, b):\n    return a - b\n")
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

    def llm_factory(session_id: str) -> MockLLMClient:
        plan = {
            "status": "SUCCESS",
            "summary": "planned",
            "workflow": {"steps": [{"id": "fix", "worker": "Implementer"}]},
        }
        if session_id == "2-1":
            return MockLLMClient(
                script=[
                    plan,
                    {"tool_calls": [{"tool": "file", "action": "write", "args": {"path": "proj/app.py", "content": "def add(a, b):\n    return a + b\n"}}]},
                    {"status": "SUCCESS", "summary": "fixed", "artifacts": [], "metrics": {}, "next_actions": [], "failure_signature": ""},
                ]
            )
        return MockLLMClient(
            script=[
                plan,
                {"status": "SUCCESS", "summary": "no-op", "artifacts": [], "metrics": {}, "next_actions": [], "failure_signature": ""},
            ]
        )

    settings = SelfImproveSettings(
        sessions_per_batch=1,
        batches=2,
        max_workers=1,
        include_paths=["proj"],
        pytest_args=["proj/tests"],
        merge_on_success=True,
    )
    orchestrator = SelfImproveOrchestrator(master, llm_factory=llm_factory, settings=settings)
    report = orchestrator.run("Fix proj.add", input_ref=None)

    assert len(report.batches) == 2
    assert report.batches[0].merged is False
    assert report.batches[1].merged is True
    assert report.batches[1].master_evaluation is not None
    assert report.batches[1].master_evaluation.ok is True
    assert "return a + b" in (master / "proj" / "app.py").read_text()


def test_self_improve_records_per_step_pytest_metrics(tmp_path: Path) -> None:
    master = tmp_path / "master"
    (master / "proj").mkdir(parents=True, exist_ok=True)
    (master / "proj" / "__init__.py").write_text("")
    (master / "proj" / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (master / "proj" / "tests").mkdir(parents=True, exist_ok=True)
    (master / "proj" / "tests" / "test_app.py").write_text(
        "from proj.app import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )

    def llm_factory(_session_id: str) -> MockLLMClient:
        plan = {
            "status": "SUCCESS",
            "summary": "planned",
            "workflow": {"steps": [{"id": "noop", "worker": "Implementer"}]},
        }
        return MockLLMClient(
            script=[
                plan,
                {"status": "SUCCESS", "summary": "done", "artifacts": [], "metrics": {}, "next_actions": [], "failure_signature": ""},
            ]
        )

    settings = SelfImproveSettings(
        sessions_per_batch=1,
        batches=1,
        max_workers=1,
        include_paths=["proj"],
        pytest_args=["proj/tests"],
        merge_on_success=False,
    )
    orchestrator = SelfImproveOrchestrator(master, llm_factory=llm_factory, settings=settings)
    report = orchestrator.run("No-op", input_ref=None)

    run_root = report.batches[0].sessions[0].run_root
    assert run_root is not None
    workflow_state = Path(run_root) / "workflow_state.json"
    payload = json.loads(workflow_state.read_text())
    attempts = payload["state"]["steps"]["noop"]["attempts"]
    assert attempts
    assert isinstance(attempts[0]["progress_metrics"].get("passed_tests"), int)


def test_self_improve_counts_skipped_steps_as_succeeded(tmp_path: Path) -> None:
    master = tmp_path / "master"
    (master / "proj").mkdir(parents=True, exist_ok=True)
    (master / "proj" / "__init__.py").write_text("")
    (master / "proj" / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (master / "proj" / "tests").mkdir(parents=True, exist_ok=True)
    (master / "proj" / "tests" / "test_app.py").write_text(
        "from proj.app import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )

    def llm_factory(_session_id: str) -> MockLLMClient:
        return MockLLMClient(
            script=[
                {
                    "status": "SUCCESS",
                    "summary": "planned",
                    "workflow": {
                        "steps": [
                            {"id": "check", "worker": "Implementer"},
                            {"id": "unneeded", "worker": "Implementer", "depends_on": ["check"]},
                        ]
                    },
                },
                {
                    "status": "SUCCESS",
                    "summary": "tests green; terminate early",
                    "metrics": {"terminate_workflow": True, "terminate_reason": "tests green"},
                    "artifacts": [],
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
        merge_on_success=False,
    )
    orchestrator = SelfImproveOrchestrator(master, llm_factory=llm_factory, settings=settings)
    report = orchestrator.run("Terminate early", input_ref=None)

    session = report.batches[0].sessions[0]
    assert session.workflow_ok is True
    assert session.workflow_status == "SUCCEEDED"


def _git(cwd: Path, args: list[str]) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
