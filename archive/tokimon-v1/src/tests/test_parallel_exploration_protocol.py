from __future__ import annotations

import json
from pathlib import Path

from self_improve.orchestrator import (
    DECLARED_SCORING_FUNCTION,
    EvaluationResult,
    SelfImproveBatchResult,
    SelfImproveReport,
    SelfImproveSessionResult,
    _charter_distance,
    _load_experiment_summary,
    _path_charter,
    _report_to_markdown,
)
from self_improve.source import InputPayload


def test_path_charter_deterministic_and_diverse() -> None:
    assert _path_charter("1-1") == _path_charter("1-1")

    charters = [_path_charter(f"1-{idx}") for idx in range(1, 6)]
    for i in range(len(charters)):
        for j in range(i + 1, len(charters)):
            assert _charter_distance(charters[i], charters[j]) >= 2


def test_experiment_summary_requires_parallel_protocol_fields(tmp_path: Path) -> None:
    path = tmp_path / "attempt.json"
    path.write_text(json.dumps({"causal_mechanism_hypothesis": "x"}), encoding="utf-8")
    payload, error = _load_experiment_summary(path)
    assert payload is None
    assert error is not None


def test_report_includes_protocol_section_and_declared_scoring() -> None:
    evaluation = EvaluationResult(ok=True, passed=1, failed=0, failing_tests=[], elapsed_s=0.0)
    sessions = [
        SelfImproveSessionResult(
            session_id="1-1",
            path_charter=_path_charter("1-1"),
            workspace_root="w1",
            run_root="r1",
            workflow_ok=True,
            workflow_status="SUCCEEDED",
            workflow_error=None,
            evaluation=evaluation,
            score=(1, 1, 1, 1, 1),
            model_calls=1,
            tool_calls=1,
            changed_files=[],
            changes=[],
            verification_ok=True,
            verification_reason="ok",
            entrypoint_attempts=1,
            attempts=[
                {
                    "experiment_summary": {
                        "plan": ["a"],
                        "path_charter": _path_charter("1-1"),
                        "self_critique": "c",
                        "lessons": ["l"],
                    }
                }
            ],
        ),
        SelfImproveSessionResult(
            session_id="1-2",
            path_charter=_path_charter("1-2"),
            workspace_root="w2",
            run_root="r2",
            workflow_ok=True,
            workflow_status="SUCCEEDED",
            workflow_error=None,
            evaluation=evaluation,
            score=(0, 1, 1, 1, 1),
            model_calls=1,
            tool_calls=1,
            changed_files=[],
            changes=[],
            verification_ok=False,
            verification_reason="failed",
            entrypoint_attempts=1,
            attempts=[
                {
                    "experiment_summary": {
                        "plan": ["b"],
                        "path_charter": _path_charter("1-2"),
                        "self_critique": "c",
                        "lessons": ["l2"],
                    }
                }
            ],
        ),
    ]
    report = SelfImproveReport(
        goal="Goal",
        input_payload=InputPayload(kind="none", ref=None, content=""),
        run_root="runs/self-improve/1",
        settings={"sessions_per_batch": 2, "batches": 1, "entrypoint_max_attempts": 1},
        batches=[
            SelfImproveBatchResult(
                batch_index=1,
                master_baseline_evaluation=evaluation,
                sessions=sessions,
                winner_session_id="1-1",
                merged=False,
                master_evaluation=None,
            )
        ],
    )
    markdown = _report_to_markdown(report)
    assert "## Parallel Exploration Protocol" in markdown
    assert f"Declared scoring function: {DECLARED_SCORING_FUNCTION}" in markdown
    assert "Diversity check:" in markdown
    assert "Per-path comparison (Parallel Exploration Protocol):" in markdown

