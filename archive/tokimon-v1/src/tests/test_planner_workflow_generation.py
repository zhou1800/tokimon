from __future__ import annotations

import json
from pathlib import Path

from agents.manager import Manager
from llm.client import MockLLMClient
from memory.store import MemoryStore
from runners.hierarchical import HierarchicalRunner


def test_manager_plan_steps_parses_workflow_steps(tmp_path: Path) -> None:
    memory_store = MemoryStore(tmp_path / "memory")
    manager = Manager(memory_store)
    llm = MockLLMClient(
        script=[
            {
                "status": "SUCCESS",
                "summary": "planned",
                "artifacts": [],
                "metrics": {},
                "next_actions": [],
                "failure_signature": "",
                "workflow": {
                    "steps": [
                        {"id": "s1", "worker": "Implementer"},
                        {"id": "s2", "worker": "Reviewer", "depends_on": ["s1"]},
                    ]
                },
            }
        ]
    )
    steps = manager.plan_steps("goal", llm, tools={})
    assert steps is not None
    assert [step["id"] for step in steps] == ["s1", "s2"]


def test_hierarchical_runner_uses_planned_steps_and_step_workers(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    llm = MockLLMClient(
        script=[
            {
                "status": "SUCCESS",
                "summary": "planned",
                "artifacts": [],
                "metrics": {},
                "next_actions": [],
                "failure_signature": "",
                "workflow": {
                    "steps": [
                        {"id": "s1", "worker": "Implementer"},
                        {"id": "s2", "worker": "Reviewer", "depends_on": ["s1"]},
                    ]
                },
            },
            {"status": "SUCCESS", "summary": "s1 done", "artifacts": [], "metrics": {}, "next_actions": [], "failure_signature": ""},
            {"status": "SUCCESS", "summary": "s2 done", "artifacts": [], "metrics": {}, "next_actions": [], "failure_signature": ""},
        ]
    )
    runner = HierarchicalRunner(workspace, llm, base_dir=tmp_path / "runs")
    result = runner.run("goal", task_steps=None, task_id="t", test_args=None, concurrency=1)

    payload = json.loads(result.workflow_state_path.read_text())
    spec_steps = payload["spec"]["steps"]
    assert [step["step_id"] for step in spec_steps] == ["s1", "s2"]

    state_steps = payload["state"]["steps"]
    assert state_steps["s1"]["attempts"][0]["worker_type"] == "Implementer"
    assert state_steps["s2"]["attempts"][0]["worker_type"] == "Reviewer"
