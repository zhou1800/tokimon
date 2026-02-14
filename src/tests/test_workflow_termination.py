from __future__ import annotations

import json
from pathlib import Path

from llm.client import MockLLMClient
from runners.hierarchical import HierarchicalRunner


def test_workflow_terminate_marks_remaining_steps_skipped(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    llm = MockLLMClient(
        script=[
            {
                "status": "SUCCESS",
                "summary": "done early",
                "metrics": {"terminate_workflow": True, "terminate_reason": "goal satisfied"},
                "artifacts": [],
                "next_actions": [],
                "failure_signature": "",
            }
        ]
    )
    runner = HierarchicalRunner(workspace, llm, base_dir=tmp_path / "runs")
    result = runner.run(
        "goal",
        task_steps=[
            {"id": "s1", "worker": "Implementer"},
            {"id": "s2", "worker": "Implementer", "depends_on": ["s1"]},
        ],
        task_id="t",
        test_args=None,
        concurrency=1,
    )

    payload = json.loads(result.workflow_state_path.read_text())
    state = payload["state"]["steps"]
    assert state["s1"]["status"] == "SUCCEEDED"
    assert state["s2"]["status"] == "SKIPPED"
    assert state["s2"]["attempts"] == []
    assert payload["state"]["metadata"]["termination"]["triggered_by"] == "s1"

