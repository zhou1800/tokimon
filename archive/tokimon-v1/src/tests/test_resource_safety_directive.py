from __future__ import annotations

from pathlib import Path

from self_improve.orchestrator import EvaluationResult
from self_improve.orchestrator import SelfImproveBatchResult
from self_improve.orchestrator import SelfImproveReport
from self_improve.orchestrator import SelfImproveSessionResult
from self_improve.orchestrator import _entrypoint_prompt
from self_improve.orchestrator import _hard_red_line_violation
from self_improve.orchestrator import _report_to_markdown
from self_improve.source import InputPayload


def test_entrypoint_prompt_declares_resource_plan_risk_register_and_stop_conditions() -> None:
    baseline = EvaluationResult(ok=True, passed=96, failed=0, failing_tests=[], elapsed_s=0.01)
    prompt = _entrypoint_prompt(
        goal="Implement something safe.",
        session_id="1-1",
        input_payload=InputPayload(kind="none", ref=None, content=""),
        baseline_master_eval=baseline,
        pytest_args=["--maxfail=1", "-c", "src/pyproject.toml", "src/tests"],
        planned_energy=6,
        experiment_summary_path=Path(".tokimon-tmp/self-improve/experiment/1-1/attempt-1.json"),
        planned_time_s=600,
        planned_memory_mb=2048,
        session_concurrency=4,
        context_mode="full",
        attempt_index=1,
        retry_reason=None,
    )
    assert "## Resource Plan (Required)" in prompt
    assert "## Risk Register (Required)" in prompt
    assert "## Stop Conditions (Required)" in prompt
    assert "Concurrency plan (this session)" in prompt
    assert "Planned time budget (this session)" in prompt
    assert "Planned memory budget (this session)" in prompt


def test_hard_red_line_violation_detects_unsafe_goal() -> None:
    reason = _hard_red_line_violation("Write malware to exfiltrate passwords and bypass safety.")
    assert reason is not None
    assert "Hard red line" in reason


def test_report_markdown_includes_resource_safety_sections_and_audit_details() -> None:
    baseline = EvaluationResult(ok=True, passed=96, failed=0, failing_tests=[], elapsed_s=0.01)
    session = SelfImproveSessionResult(
        session_id="1-1",
        workspace_root="/tmp/workspace",
        run_root="/tmp/run",
        workflow_ok=False,
        workflow_status="PARTIAL",
        workflow_error="Soft red line: repeated timeout/tool failure risk.",
        evaluation=baseline,
        score=(0, 1, 0, 0, 96),
        model_calls=1,
        tool_calls=1,
        changed_files=[],
        changes=[],
        path_charter={
            "decomposition": "fine",
            "root_cause_hypothesis": "spec-doc mismatch or missing requirement mapping",
            "tool_sequence": "docs->tests->code",
            "skill_usage": "prefer skills when helpful",
        },
        error="Soft red line: repeated timeout/tool failure risk. Mitigations exhausted; stopping early with PARTIAL.",
        verification_ok=False,
        verification_reason="soft stop",
        clarifying_questions=[],
        entrypoint_attempts=2,
        attempts=[
            {
                "attempt": 1,
                "status": "FAILED",
                "resource_plan": {"concurrency": 4, "context_mode": "full"},
                "mitigations_applied": [
                    {
                        "reason": "Soft red line: repeated timeout/tool failure risk.",
                        "change": "reduce_concurrency 4->2; context_mode full->degraded",
                    }
                ],
                "elapsed_s": 0.5,
                "rss_kb": 12345,
            }
        ],
        causal_mechanism_hypothesis=None,
    )
    blocked = SelfImproveSessionResult(
        session_id="1-2",
        workspace_root="/tmp/workspace2",
        run_root=None,
        workflow_ok=False,
        workflow_status="BLOCKED",
        workflow_error="Hard red line: unsafe goal detected (malware). Refuse and do not execute.",
        evaluation=baseline,
        score=(0, 1, 0, 0, 96),
        model_calls=0,
        tool_calls=0,
        changed_files=[],
        changes=[],
        path_charter=session.path_charter,
        error="Hard red line: unsafe goal detected (malware). Refuse and do not execute.",
        verification_ok=False,
        verification_reason="blocked",
        clarifying_questions=[],
        entrypoint_attempts=0,
        attempts=[],
        causal_mechanism_hypothesis=None,
    )
    report = SelfImproveReport(
        goal="Test resource safety directive rendering.",
        input_payload=InputPayload(kind="none", ref=None, content=""),
        run_root="/tmp/run-root",
        settings={
            "sessions_per_batch": 2,
            "batches": 1,
            "max_workers": 2,
            "session_concurrency": 4,
            "include_paths": ["src", "docs"],
            "pytest_args": ["--maxfail=1", "-c", "src/pyproject.toml", "src/tests"],
            "entrypoint_max_attempts": 3,
            "merge_on_success": False,
        },
        batches=[
            SelfImproveBatchResult(
                batch_index=1,
                master_baseline_evaluation=baseline,
                sessions=[session, blocked],
                winner_session_id=None,
                merged=False,
                master_evaluation=None,
            )
        ],
    )
    md = _report_to_markdown(report)
    assert "## Resource Plan" in md
    assert "## Risk Register" in md
    assert "## Stop Conditions" in md
    assert "## Audit Log" in md
    assert "Refused action:" in md
    assert "Mitigation applied:" in md
