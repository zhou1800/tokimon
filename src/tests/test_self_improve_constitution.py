from __future__ import annotations

from pathlib import Path

from self_improve.orchestrator import (
    EvaluationResult,
    SelfImproveBatchResult,
    SelfImproveReport,
    SelfImproveSessionResult,
    _entrypoint_prompt,
    _rank_sessions,
    _report_to_markdown,
    _score,
)
from self_improve.source import InputPayload


def test_constitution_doc_exists_with_required_headings() -> None:
    path = Path("docs/tokimon-constitution.md")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "## Immutable Invariants" in content
    assert "## Evaluation Plan (Required)" in content


def test_entrypoint_prompt_includes_constitution_sections() -> None:
    prompt = _entrypoint_prompt(
        goal="Do a thing.",
        session_id="1-1",
        input_payload=InputPayload(kind="none", ref=None, content=""),
        baseline_master_eval=EvaluationResult(ok=True, passed=10, failed=0, failing_tests=[], elapsed_s=0.1),
        pytest_args=["-q"],
        planned_energy=2,
        experiment_summary_path=Path(".tokimon-tmp/self-improve/experiment/1-1/attempt-1.json"),
        attempt_index=1,
        retry_reason=None,
    )
    first_line = prompt.splitlines()[0]
    assert first_line.startswith("Constitution Acknowledgement:")
    assert "Immutable Invariants:" in prompt
    assert "## Evaluation Plan (Required)" in prompt
    assert "Evaluation-first experiment loop" in prompt
    assert "Pass condition (this session):" in prompt
    assert "Required reporting fields (for auditability):" in prompt
    assert "- baseline evaluation summary" in prompt
    assert "- post-change evaluation summary" in prompt
    assert "- delta (improvement or regression)" in prompt
    assert "- causal mechanism hypothesis linking changes to delta" in prompt
    assert "- explicit pass condition for the run" in prompt
    assert "Required attempt artifact (must be written by the agent):" in prompt
    assert ".tokimon-tmp/self-improve/experiment/1-1/attempt-1.json" in prompt


def test_entrypoint_prompt_includes_agent_worktree_rule() -> None:
    prompt = _entrypoint_prompt(
        goal="Learn from this steering prompt.",
        session_id="1-1",
        input_payload=InputPayload(kind="none", ref=None, content=""),
        baseline_master_eval=EvaluationResult(ok=True, passed=10, failed=0, failing_tests=[], elapsed_s=0.1),
        pytest_args=["-q"],
        planned_energy=2,
        experiment_summary_path=Path(".tokimon-tmp/self-improve/experiment/1-1/attempt-1.json"),
        attempt_index=1,
        retry_reason=None,
    )
    assert "## AI Agent Worktree Rule (Required when git/shell access is available)" in prompt
    assert "temp/codex-worktrees/" in prompt
    assert "Worktree: <absolute-path>" in prompt
    assert "do not wait for extra human instructions" in prompt
    assert "does not replace Tokimon's outer batch winner merge" in prompt
    assert "`git worktree add -b \"$BRANCH\" \"$WT_DIR\" HEAD`" in prompt
    assert "`git merge --ff-only <worktree-branch>`" in prompt
    assert "`git worktree remove <worktree-path>`" in prompt


def test_winner_selection_tie_breaker_by_session_id() -> None:
    evaluation = EvaluationResult(ok=True, passed=1, failed=0, failing_tests=[], elapsed_s=0.0)
    shared = dict(
        workspace_root=".",
        run_root=None,
        workflow_ok=True,
        workflow_status="SUCCEEDED",
        workflow_error=None,
        evaluation=evaluation,
        score=(1, 1, 1, 1, 1),
        model_calls=0,
        tool_calls=0,
        changed_files=[],
        changes=[],
        error=None,
        verification_ok=True,
        verification_reason="ok",
        entrypoint_attempts=1,
    )
    session_b = SelfImproveSessionResult(session_id="1-2", **shared)
    session_a = SelfImproveSessionResult(session_id="1-1", **shared)
    ranked = _rank_sessions([session_b, session_a])
    assert ranked[0].session_id == "1-1"


def test_report_markdown_includes_constitution_headings_and_energy() -> None:
    evaluation = EvaluationResult(ok=True, passed=1, failed=0, failing_tests=["a::test_one"], elapsed_s=0.0)
    sessions = [
        SelfImproveSessionResult(
            session_id="1-1",
            workspace_root="w1",
            run_root="r1",
            workflow_ok=True,
            workflow_status="SUCCEEDED",
            workflow_error=None,
            evaluation=evaluation,
            score=(1, 1, 1, 1, 1),
            model_calls=2,
            tool_calls=3,
            changed_files=["a.py"],
            changes=[],
            error=None,
            verification_ok=True,
            verification_reason="ok",
            entrypoint_attempts=1,
            causal_mechanism_hypothesis="Fixing flaky setup removes hidden retries, so stable assertions improve pass rate.",
        ),
        SelfImproveSessionResult(
            session_id="1-2",
            workspace_root="w2",
            run_root="r2",
            workflow_ok=True,
            workflow_status="SUCCEEDED",
            workflow_error=None,
            evaluation=evaluation,
            score=(1, 1, 1, 1, 1),
            model_calls=1,
            tool_calls=4,
            changed_files=["b.py"],
            changes=[],
            error=None,
            verification_ok=True,
            verification_reason="ok",
            entrypoint_attempts=1,
            causal_mechanism_hypothesis="No effect.",
        ),
    ]
    report = SelfImproveReport(
        goal="Goal",
        input_payload=InputPayload(kind="none", ref=None, content=""),
        run_root="runs/self-improve/1",
        settings={
            "sessions_per_batch": 2,
            "batches": 1,
            "entrypoint_max_attempts": 3,
        },
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
    assert "## Constitution Acknowledgement" in markdown
    assert "## Scoring Rubric" in markdown
    assert "## Energy Budget" in markdown
    assert "## Experiment Loop Summary" in markdown
    assert "## Parallel Exploration Protocol" in markdown
    assert "## Audit Log" in markdown
    planned_energy = 2 * 1 * 3 * 50
    actual_energy = (2 + 3) + (1 + 4)
    assert f"Planned energy: {planned_energy}" in markdown
    assert f"Actual energy: {actual_energy}" in markdown
    assert "Declared scoring function:" in markdown
    assert "Baseline eval:" in markdown
    assert "Post-change eval (winner):" in markdown
    assert "failing_tests=['a::test_one']" in markdown
    assert "Delta (winner - baseline):" in markdown
    expected_pass_condition = "Maintain evaluation ok while keeping session energy within the planned energy budget."
    assert f"Pass condition: {expected_pass_condition} Met:" in markdown
    assert "Causal mechanism hypothesis:" in markdown
    assert "Fixing flaky setup removes hidden retries" in markdown


def test_scoring_does_not_optimize_for_energy() -> None:
    evaluation = EvaluationResult(ok=True, passed=10, failed=None, failing_tests=[], elapsed_s=0.0)
    score_low_energy = _score(True, evaluation, True, 1, 1, 1)
    score_high_energy = _score(True, evaluation, True, 1, 100, 200)
    assert score_low_energy == score_high_energy
