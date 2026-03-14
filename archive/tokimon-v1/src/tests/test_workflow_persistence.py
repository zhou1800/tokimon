from pathlib import Path

from flow_types import StepStatus
from workflow.engine import WorkflowEngine
from workflow.models import StepSpec, WorkflowSpec


def test_workflow_save_load(tmp_path: Path) -> None:
    spec = WorkflowSpec(
        workflow_id="wf",
        goal="goal",
        steps=[
            StepSpec(step_id="s1", name="s1", description="", worker="Implementer"),
            StepSpec(step_id="s2", name="s2", description="", worker="Implementer", depends_on=["s1"]),
        ],
    )
    engine = WorkflowEngine(spec)
    engine.mark_status("s1", StepStatus.SUCCEEDED)
    path = tmp_path / "workflow_state.json"
    engine.save(path)

    loaded = WorkflowEngine.load(path)
    assert loaded.state.steps["s1"].status == StepStatus.SUCCEEDED
    assert loaded.state.steps["s2"].status in {StepStatus.NEW, StepStatus.READY}
