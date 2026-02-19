"""Self-improvement runner: parallel sessions, evaluation, and winner merge."""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from llm.client import LLMClient, MockLLMClient
from runs import RunContext, create_run_context
from runners.hierarchical import HierarchicalRunner
from tools.pytest_tool import PytestTool

from .source import InputPayload
from .source import read_goal_input
from .workspace import WorkspaceChange
from .workspace import abort_squash_merge
from .workspace import can_use_git_merge
from .workspace import commit_squash_merge
from .workspace import clone_master
from .workspace import compute_changes
from .workspace import create_workspace_candidate_commit
from .workspace import create_git_merge_candidate
from .workspace import delete_branch
from .workspace import git_toplevel
from .workspace import list_unmerged_paths
from .workspace import merge_lock
from .workspace import purge_bytecode_for_changes
from .workspace import squash_merge_commit
from .workspace import resolve_unmerged_paths

CONSTITUTION_PATH = "docs/tokimon-constitution.md"
CONSTITUTION_ACK = (
    "Constitution Acknowledgement: This session is bound by the Tokimon Constitution "
    f"({CONSTITUTION_PATH}) and must comply with its immutable invariants, governance rules, "
    "and evaluation requirements."
)
IMMUTABLE_INVARIANTS = [
    "Self-improve runs optimize for measurable capability growth under explicit evaluation.",
    "All self-improve decisions must be auditable, reproducible, and deterministic.",
    "Rollback safety is mandatory: do not merge or persist changes when verification fails.",
    "The system must honor stop capability signals and terminate safely without continuing work.",
    "Energy is defined as `energy = model_calls + tool_calls` and must be reported as planned vs actual.",
]
SCORING_RUBRIC = [
    "Verification outcome (pass=1, fail=0)",
    "Evaluation outcome (ok=1, fail=0)",
    "Workflow outcome (success=1, otherwise=0)",
    "Concrete changes produced (yes=1, no=0)",
    "Passed test count (higher is better)",
    "Failed test count (lower is better)",
]


@dataclass(frozen=True)
class SelfImproveSettings:
    sessions_per_batch: int = 4
    batches: int = 1
    max_workers: int = 4
    session_concurrency: int = 4
    include_paths: list[str] = field(
        default_factory=lambda: ["AGENTS.md", "README.md", "requirements.txt", "src", "docs"]
    )
    pytest_args: list[str] = field(
        default_factory=lambda: ["--maxfail=1", "-c", "src/pyproject.toml", "src/tests"]
    )
    entrypoint_max_attempts: int = 3
    merge_on_success: bool = True


@dataclass(frozen=True)
class EvaluationResult:
    ok: bool
    passed: int | None
    failed: int | None
    failing_tests: list[str]
    elapsed_s: float


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    reason: str


@dataclass(frozen=True)
class SelfImproveSessionResult:
    session_id: str
    workspace_root: str
    run_root: str | None
    workflow_ok: bool | None
    workflow_status: str | None
    workflow_error: str | None
    evaluation: EvaluationResult
    score: tuple[int, int, int, int, int, int]
    model_calls: int
    tool_calls: int
    changed_files: list[str]
    changes: list[WorkspaceChange]
    error: str | None = None
    verification_ok: bool = False
    verification_reason: str | None = None
    clarifying_questions: list[str] = field(default_factory=list)
    entrypoint_attempts: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SelfImproveBatchResult:
    batch_index: int
    master_baseline_evaluation: EvaluationResult
    sessions: list[SelfImproveSessionResult]
    winner_session_id: str | None
    merged: bool
    master_evaluation: EvaluationResult | None


@dataclass(frozen=True)
class SelfImproveReport:
    goal: str
    input_payload: InputPayload
    run_root: str
    settings: dict[str, Any]
    batches: list[SelfImproveBatchResult]


class SelfImproveOrchestrator:
    def __init__(
        self,
        master_root: Path,
        llm_factory: Callable[..., LLMClient] | None = None,
        settings: SelfImproveSettings | None = None,
    ) -> None:
        resolved = master_root.resolve()
        self.master_root = git_toplevel(resolved) or resolved
        self.settings = settings or SelfImproveSettings()
        self.llm_factory = _wrap_llm_factory(llm_factory) if llm_factory else (lambda _sid, _ws: MockLLMClient(script=[]))

    def run(self, goal: str, input_ref: str | None = None) -> SelfImproveReport:
        if not can_use_git_merge(self.master_root):
            raise RuntimeError(
                "self-improve requires a clean git checkout so sessions can be created as `git worktree`s "
                "and winners can be merged deterministically (ensure `git status --porcelain` is empty)."
            )
        input_payload = read_goal_input(goal, input_ref)
        run_context = self._create_self_improve_run()
        self._write_status(run_context, goal, input_payload, status="RUNNING", batches=[])
        batches: list[SelfImproveBatchResult] = []
        for batch_index in range(1, self.settings.batches + 1):
            batch = self._run_batch(run_context, batch_index, goal, input_payload)
            batches.append(batch)
        report = SelfImproveReport(
            goal=goal,
            input_payload=input_payload,
            run_root=str(run_context.root),
            settings=self._settings_dict(),
            batches=batches,
        )
        self._write_report(run_context, report)
        return report

    def _run_batch(self, run_context: RunContext, batch_index: int, goal: str,
                   input_payload: InputPayload) -> SelfImproveBatchResult:
        sessions_dir = run_context.root / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        baseline_master_eval = self._evaluate_workspace(self.master_root)
        session_results: list[SelfImproveSessionResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.settings.max_workers) as executor:
            futures = []
            for index in range(1, self.settings.sessions_per_batch + 1):
                session_id = f"{batch_index}-{index}"
                futures.append(
                    executor.submit(
                        self._run_session, sessions_dir, session_id, goal, input_payload, baseline_master_eval
                    )
                )
            for fut in concurrent.futures.as_completed(futures):
                session_results.append(fut.result())
        session_results = _rank_sessions(session_results)
        winner = session_results[0] if session_results else None
        merged = False
        master_eval = None
        if winner and self.settings.merge_on_success:
            merged, master_eval = self._merge_winner(winner)
        return SelfImproveBatchResult(
            batch_index=batch_index,
            master_baseline_evaluation=baseline_master_eval,
            sessions=session_results,
            winner_session_id=winner.session_id if winner else None,
            merged=merged,
            master_evaluation=master_eval,
        )

    def _run_session(
        self,
        sessions_dir: Path,
        session_id: str,
        goal: str,
        input_payload: InputPayload,
        baseline_master_eval: EvaluationResult,
    ) -> SelfImproveSessionResult:
        session_dir = sessions_dir / f"session-{session_id}"
        workspace_dir = session_dir / "workspace"
        session_dir.mkdir(parents=True, exist_ok=True)
        clone_master(self.master_root, workspace_dir, include_paths=self.settings.include_paths)

        llm_client = self.llm_factory(session_id, workspace_dir)
        runner = HierarchicalRunner(workspace_dir, llm_client, base_dir=workspace_dir / "runs")

        run_root: str | None = None
        error: str | None = None
        workflow_ok: bool | None = None
        workflow_status: str | None = None
        workflow_error: str | None = None
        model_calls = 0
        tool_calls = 0
        verification = VerificationResult(ok=False, reason="verification not run")
        clarifying_questions = _clarifying_questions_for_goal(goal)
        attempts: list[dict[str, Any]] = []
        entrypoint_max_attempts = max(1, int(self.settings.entrypoint_max_attempts))
        planned_energy = _planned_energy_budget(1, entrypoint_max_attempts)

        session_report: dict[str, Any] = {
            "session_id": session_id,
            "goal": goal,
            "input": {"kind": input_payload.kind, "ref": input_payload.ref},
            "master_baseline_evaluation": baseline_master_eval.__dict__,
            "pytest_args": list(self.settings.pytest_args),
            "entrypoint_max_attempts": entrypoint_max_attempts,
            "clarifying_questions": clarifying_questions,
            "attempts": attempts,
            "status": "RUNNING",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (session_dir / "session.json").write_text(json.dumps(session_report, indent=2))

        if clarifying_questions:
            workflow_ok = False
            workflow_status = "BLOCKED"
            workflow_error = "request is ambiguous; clarification needed before prompt generation"
            error = workflow_error
            verification = VerificationResult(ok=False, reason=workflow_error)
            evaluation = self._evaluate_workspace(workspace_dir)
            changes = compute_changes(self.master_root, workspace_dir, include_paths=self.settings.include_paths)
            changed_files = [change.relpath for change in changes]
            score = _score(
                verification.ok,
                evaluation,
                workflow_ok,
                len(changed_files),
                model_calls,
                tool_calls,
            )
            session_report.update(
                {
                    "status": "BLOCKED",
                    "workflow_ok": workflow_ok,
                    "workflow_status": workflow_status,
                    "workflow_error": workflow_error,
                    "evaluation": evaluation.__dict__,
                    "model_calls": model_calls,
                    "tool_calls": tool_calls,
                    "changed_files": changed_files,
                    "verification": verification.__dict__,
                    "entrypoint_attempts": 0,
                    "error": error,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            (session_dir / "session.json").write_text(json.dumps(session_report, indent=2))
            return SelfImproveSessionResult(
                session_id=session_id,
                workspace_root=str(workspace_dir),
                run_root=run_root,
                workflow_ok=workflow_ok,
                workflow_status=workflow_status,
                workflow_error=workflow_error,
                evaluation=evaluation,
                score=score,
                model_calls=model_calls,
                tool_calls=tool_calls,
                changed_files=changed_files,
                changes=changes,
                error=error,
                verification_ok=verification.ok,
                verification_reason=verification.reason,
                clarifying_questions=clarifying_questions,
                entrypoint_attempts=0,
                attempts=attempts,
            )

        evaluation = baseline_master_eval
        changes: list[WorkspaceChange] = []
        changed_files: list[str] = []
        entrypoint_attempts = 0

        for attempt_index in range(1, entrypoint_max_attempts + 1):
            entrypoint_attempts = attempt_index
            retry_reason = verification.reason if attempt_index > 1 else None
            session_goal = _entrypoint_prompt(
                goal,
                session_id,
                input_payload,
                baseline_master_eval,
                self.settings.pytest_args,
                planned_energy,
                attempt_index=attempt_index,
                retry_reason=retry_reason,
            )
            attempt_report: dict[str, Any] = {
                "attempt": attempt_index,
                "prompt": session_goal,
                "status": "RUNNING",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            attempts.append(attempt_report)
            session_report.update({"status": "RUNNING", "entrypoint_attempts": attempt_index})
            (session_dir / "session.json").write_text(json.dumps(session_report, indent=2))

            attempt_run_root = None
            attempt_error: str | None = None
            attempt_workflow_ok: bool | None = None
            attempt_workflow_status: str | None = None
            attempt_workflow_error: str | None = None
            attempt_model_calls = 0
            attempt_tool_calls = 0

            try:
                result = runner.run(
                    session_goal,
                    task_steps=None,
                    task_id=f"self-improve-{session_id}-attempt-{attempt_index}",
                    test_args=self.settings.pytest_args,
                    concurrency=self.settings.session_concurrency,
                )
                attempt_run_root = str(result.run_context.root)
                attempt_workflow_ok, attempt_workflow_status, attempt_workflow_error = _summarize_workflow_state(
                    result.workflow_state_path
                )
                if attempt_workflow_ok is False and attempt_error is None:
                    attempt_error = attempt_workflow_error or "workflow failed"
                attempt_model_calls = int(result.model_calls or 0)
                attempt_tool_calls = int(result.tool_calls or 0)
            except Exception as exc:
                attempt_error = str(exc)

            run_root = attempt_run_root or run_root
            workflow_ok = attempt_workflow_ok
            workflow_status = attempt_workflow_status
            workflow_error = attempt_workflow_error
            error = attempt_error
            model_calls += attempt_model_calls
            tool_calls += attempt_tool_calls

            changes = compute_changes(self.master_root, workspace_dir, include_paths=self.settings.include_paths)
            purge_bytecode_for_changes(workspace_dir, changes)
            evaluation = self._evaluate_workspace(workspace_dir)
            changed_files = [change.relpath for change in changes]
            verification = _verify_session_outcome(
                goal,
                baseline_master_eval,
                evaluation,
                workflow_ok,
                changed_files,
                error,
            )
            attempt_report.update(
                {
                    "status": "COMPLETED" if verification.ok else "FAILED",
                    "run_root": run_root,
                    "workflow_ok": workflow_ok,
                    "workflow_status": workflow_status,
                    "workflow_error": workflow_error,
                    "evaluation": evaluation.__dict__,
                    "model_calls": attempt_model_calls,
                    "tool_calls": attempt_tool_calls,
                    "changed_files": changed_files,
                    "error": error,
                    "verification": verification.__dict__,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            session_report.update(
                {
                    "status": "COMPLETED" if verification.ok else "RETRYING",
                    "run_root": run_root,
                    "workflow_ok": workflow_ok,
                    "workflow_status": workflow_status,
                    "workflow_error": workflow_error,
                    "evaluation": evaluation.__dict__,
                    "model_calls": model_calls,
                    "tool_calls": tool_calls,
                    "changed_files": changed_files,
                    "verification": verification.__dict__,
                    "entrypoint_attempts": attempt_index,
                    "error": error,
                }
            )
            (session_dir / "session.json").write_text(json.dumps(session_report, indent=2))
            if verification.ok:
                break

        score = _score(
            verification.ok,
            evaluation,
            workflow_ok,
            len(changed_files),
            model_calls,
            tool_calls,
        )
        session_report.update(
            {
                "status": "COMPLETED" if verification.ok else "FAILED",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        (session_dir / "session.json").write_text(json.dumps(session_report, indent=2))
        return SelfImproveSessionResult(
            session_id=session_id,
            workspace_root=str(workspace_dir),
            run_root=run_root,
            workflow_ok=workflow_ok,
            workflow_status=workflow_status,
            workflow_error=workflow_error,
            evaluation=evaluation,
            score=score,
            model_calls=model_calls,
            tool_calls=tool_calls,
            changed_files=changed_files,
            changes=changes,
            error=error,
            verification_ok=verification.ok,
            verification_reason=verification.reason,
            clarifying_questions=clarifying_questions,
            entrypoint_attempts=entrypoint_attempts,
            attempts=attempts,
        )

    def _evaluate_workspace(self, root: Path) -> EvaluationResult:
        tool = PytestTool(root)
        start = time.perf_counter()
        result = tool.run(self.settings.pytest_args)
        elapsed_s = time.perf_counter() - start
        data = result.data
        return EvaluationResult(
            ok=result.ok,
            passed=data.get("passed"),
            failed=data.get("failed"),
            failing_tests=data.get("failing_tests", []),
            elapsed_s=elapsed_s,
        )

    def _merge_winner(self, winner: SelfImproveSessionResult) -> tuple[bool, EvaluationResult | None]:
        if not winner.verification_ok:
            return False, None
        if not winner.evaluation.ok:
            return False, None
        if not winner.changes:
            return True, self._evaluate_workspace(self.master_root)

        winner_workspace = Path(winner.workspace_root)
        candidate_commit = create_workspace_candidate_commit(
            winner_workspace,
            include_paths=self.settings.include_paths,
            message=f"tokimon: self-improve candidate {winner.session_id}",
        )

        candidate_branch: str | None = None
        with merge_lock(self.master_root):
            if not can_use_git_merge(self.master_root):
                return False, None

            if candidate_commit is None:
                candidate = create_git_merge_candidate(self.master_root, winner_workspace, winner.changes)
                if candidate is None:
                    return True, self._evaluate_workspace(self.master_root)
                candidate_commit = candidate.commit
                candidate_branch = candidate.branch

            try:
                merge_result = squash_merge_commit(self.master_root, candidate_commit)
                if merge_result.returncode != 0:
                    abort_squash_merge(self.master_root)
                    merge_result = squash_merge_commit(self.master_root, candidate_commit, strategy_option="theirs")
                    if merge_result.returncode != 0:
                        conflicted = list_unmerged_paths(self.master_root)
                        if not conflicted or not resolve_unmerged_paths(self.master_root, candidate_commit, conflicted):
                            abort_squash_merge(self.master_root)
                            return False, None

                if list_unmerged_paths(self.master_root):
                    abort_squash_merge(self.master_root)
                    return False, None

                purge_bytecode_for_changes(self.master_root, winner.changes)
                master_eval = self._evaluate_workspace(self.master_root)
                if not master_eval.ok:
                    abort_squash_merge(self.master_root)
                    return False, master_eval

                commit_result = commit_squash_merge(
                    self.master_root,
                    f"tokimon: self-improve winner session {winner.session_id}",
                )
                if commit_result.returncode != 0:
                    abort_squash_merge(self.master_root)
                    return False, master_eval
                return True, master_eval
            finally:
                if candidate_branch:
                    delete_branch(self.master_root, candidate_branch)

    def _create_self_improve_run(self) -> RunContext:
        base_dir = self.master_root / "runs" / "self-improve"
        base_dir.mkdir(parents=True, exist_ok=True)
        return create_run_context(base_dir)

    def _write_report(self, run_context: RunContext, report: SelfImproveReport) -> None:
        json_path = run_context.root / "self_improve.json"
        json_path.write_text(json.dumps(_report_to_dict(report), indent=2))
        md_path = run_context.root / "self_improve.md"
        md_path.write_text(_report_to_markdown(report))

    def _write_status(self, run_context: RunContext, goal: str, input_payload: InputPayload, *, status: str,
                      batches: list[SelfImproveBatchResult]) -> None:
        json_path = run_context.root / "self_improve.json"
        json_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "goal": goal,
                    "input": {"kind": input_payload.kind, "ref": input_payload.ref},
                    "run_root": str(run_context.root),
                    "settings": self._settings_dict(),
                    "batches": [
                        {
                            "batch_index": batch.batch_index,
                            "winner_session_id": batch.winner_session_id,
                            "merged": batch.merged,
                        }
                        for batch in batches
                    ],
                },
                indent=2,
            )
        )
        md_path = run_context.root / "self_improve.md"
        md_path.write_text(
            "\n".join(
                [
                    "# Tokimon Self-Improve Report",
                    "",
                    f"Status: {status}",
                    "",
                    f"Goal: {goal}",
                    "",
                    f"Input: {input_payload.kind} {input_payload.ref or ''}".strip(),
                    "",
                ]
            )
        )

    def _settings_dict(self) -> dict[str, Any]:
        return {
            "sessions_per_batch": self.settings.sessions_per_batch,
            "batches": self.settings.batches,
            "max_workers": self.settings.max_workers,
            "session_concurrency": self.settings.session_concurrency,
            "include_paths": self.settings.include_paths,
            "pytest_args": self.settings.pytest_args,
            "entrypoint_max_attempts": self.settings.entrypoint_max_attempts,
            "merge_on_success": self.settings.merge_on_success,
        }


def _entrypoint_prompt(
    goal: str,
    session_id: str,
    input_payload: InputPayload,
    baseline_master_eval: EvaluationResult,
    pytest_args: list[str],
    planned_energy: int,
    *,
    attempt_index: int,
    retry_reason: str | None,
) -> str:
    strategy = _strategy_hint(session_id)
    parts = [
        CONSTITUTION_ACK,
        "",
        "Immutable Invariants:",
        *[f"- {item}" for item in IMMUTABLE_INVARIANTS],
        "",
        "## Evaluation Plan (Required)",
        f"- Planned energy budget (this session): {planned_energy}",
        "- Actual energy must be reported as model_calls + tool_calls.",
        "",
        "Self-improve entry-point task.",
        "",
        "Execution flow (must be followed in order):",
        "1) Understand the user request; if ambiguous, stop and ask clarifying questions immediately.",
        "2) Once clear, continue.",
        "3) Generate a concrete prompt/plan tied to requested outcomes.",
        "4) Run the agent workflow with the generated prompt.",
        "5) Monitor and report progress.",
        "6) Verify final outcome; if verification fails, return details so the orchestrator retries from step 3.",
        "",
        f"User request: {goal.strip()}",
        "",
        f"Session: {session_id}",
        f"Attempt: {attempt_index}",
        f"Strategy: {strategy}",
        "",
        "Master baseline evaluation (before this session):",
        f"- ok: {baseline_master_eval.ok}",
        f"- passed: {baseline_master_eval.passed}",
        f"- failed: {baseline_master_eval.failed}",
        f"- failing_tests: {baseline_master_eval.failing_tests[:20]}",
        "",
        "Evaluation command (run via pytest tool):",
        f"- args: {pytest_args}",
    ]
    if retry_reason:
        parts.extend(
            [
                "",
                "Previous attempt verification failed:",
                f"- {retry_reason}",
                "Retry from step 3 with a revised concrete prompt and implementation plan.",
            ]
        )
    if input_payload.kind != "none":
        parts.extend(
            [
                "",
                f"Input ({input_payload.kind}): {input_payload.ref or ''}".strip(),
                input_payload.content,
            ]
        )
    return "\n".join(parts).strip() + "\n"


def _wrap_llm_factory(factory: Callable[..., LLMClient]) -> Callable[[str, Path], LLMClient]:
    """Normalize llm_factory to accept (session_id, workspace_dir).

    Backward compatible with older factories that only accept (session_id).
    """

    try:
        params = list(inspect.signature(factory).parameters.values())
    except (TypeError, ValueError):  # pragma: no cover
        params = []
    if len(params) <= 1:
        return lambda session_id, _ws: factory(session_id)
    return lambda session_id, workspace_dir: factory(session_id, workspace_dir)


def _strategy_hint(session_id: str) -> str:
    options = [
        "Minimize diff; prefer small safe changes.",
        "Increase test coverage first; then change code.",
        "Focus on retry/loop safety and observability.",
        "Focus on workflow planning and step contracts.",
    ]
    try:
        idx = int(session_id.split("-")[-1]) - 1
    except Exception:
        idx = 0
    return options[idx % len(options)]


def _score(
    verification_ok: bool,
    evaluation: EvaluationResult,
    workflow_ok: bool | None,
    changed_files: int,
    _model_calls: int,
    _tool_calls: int,
) -> tuple[int, int, int, int, int, int]:
    verified = 1 if verification_ok else 0
    ok = 1 if evaluation.ok else 0
    workflow = 1 if workflow_ok else 0
    has_changes = 1 if changed_files > 0 else 0
    passed = int(evaluation.passed or 0)
    failed = int(evaluation.failed or 0) if evaluation.ok else int(evaluation.failed or 9999)
    # Primary: verification outcome; Secondary: evaluation ok; then workflow, concrete changes, and test counts.
    # Energy is reported for auditability but must not influence scoring/winner selection.
    return (verified, ok, workflow, has_changes, passed, -failed)


def _planned_energy_budget(sessions: int, max_attempts: int) -> int:
    total_sessions = max(0, int(sessions))
    attempts = max(1, int(max_attempts))
    return total_sessions * attempts * 2


def _planned_energy_from_settings(settings: dict[str, Any]) -> int:
    sessions_per_batch = int(settings.get("sessions_per_batch") or 0)
    batches = int(settings.get("batches") or 0)
    entrypoint_max_attempts = int(settings.get("entrypoint_max_attempts") or 1)
    return _planned_energy_budget(sessions_per_batch * batches, entrypoint_max_attempts)


def _actual_energy(report: SelfImproveReport) -> int:
    total = 0
    for batch in report.batches:
        for session in batch.sessions:
            total += int(session.model_calls) + int(session.tool_calls)
    return total


def _rank_sessions(sessions: list[SelfImproveSessionResult]) -> list[SelfImproveSessionResult]:
    return sorted(sessions, key=_session_sort_key)


def _session_sort_key(session: SelfImproveSessionResult) -> tuple[int, int, int, int, int, int, str]:
    return (
        -int(session.score[0]),
        -int(session.score[1]),
        -int(session.score[2]),
        -int(session.score[3]),
        -int(session.score[4]),
        -int(session.score[5]),
        session.session_id,
    )


def _report_to_dict(report: SelfImproveReport) -> dict[str, Any]:
    return {
        "status": "COMPLETED",
        "goal": report.goal,
        "input": {"kind": report.input_payload.kind, "ref": report.input_payload.ref},
        "run_root": report.run_root,
        "settings": report.settings,
        "batches": [
            {
                "batch_index": batch.batch_index,
                "winner_session_id": batch.winner_session_id,
                "merged": batch.merged,
                "master_evaluation": batch.master_evaluation.__dict__ if batch.master_evaluation else None,
                "master_baseline_evaluation": batch.master_baseline_evaluation.__dict__,
                "sessions": [
                    {
                        "session_id": session.session_id,
                        "workspace_root": session.workspace_root,
                        "run_root": session.run_root,
                        "workflow_ok": session.workflow_ok,
                        "workflow_status": session.workflow_status,
                        "workflow_error": session.workflow_error,
                        "evaluation": session.evaluation.__dict__,
                        "model_calls": session.model_calls,
                        "tool_calls": session.tool_calls,
                        "changed_files": session.changed_files,
                        "verification_ok": session.verification_ok,
                        "verification_reason": session.verification_reason,
                        "clarifying_questions": session.clarifying_questions,
                        "entrypoint_attempts": session.entrypoint_attempts,
                        "attempts": session.attempts,
                        "error": session.error,
                        "score": list(session.score),
                    }
                    for session in batch.sessions
                ],
            }
            for batch in report.batches
        ],
    }


def _report_to_markdown(report: SelfImproveReport) -> str:
    planned_energy = _planned_energy_from_settings(report.settings)
    actual_energy = _actual_energy(report)
    lines = [
        "# Tokimon Self-Improve Report",
        "",
        "Status: COMPLETED",
        "",
        f"Goal: {report.goal}",
        "",
        f"Input: {report.input_payload.kind} {report.input_payload.ref or ''}".strip(),
        "",
        "## Constitution Acknowledgement",
        "",
        CONSTITUTION_ACK,
        "",
        "## Scoring Rubric",
        "",
        *[f"- {item}" for item in SCORING_RUBRIC],
        "",
        "Tie-breaker: lowest session_id (lexicographic) when scores tie.",
        "Energy is tracked for auditability and reporting, not winner selection.",
        "",
        "## Energy Budget",
        "",
        f"Planned energy: {planned_energy}",
        f"Actual energy: {actual_energy} (sum of model_calls + tool_calls across all sessions)",
        "",
        "## Audit Log",
        "",
        "Audit entries (per session):",
        "",
        "| Batch | Session | Verified | Workflow | Model Calls | Tool Calls | Energy | Changed Files |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for batch in report.batches:
        for session in batch.sessions:
            energy = int(session.model_calls) + int(session.tool_calls)
            lines.append(
                f"| {batch.batch_index} | {session.session_id} | {session.verification_ok} | "
                f"{session.workflow_status or ''} | {session.model_calls} | {session.tool_calls} | "
                f"{energy} | {len(session.changed_files)} |"
            )
    lines.extend(
        [
            "",
            "## Batches",
            "",
        ]
    )
    for batch in report.batches:
        lines.append(f"### Batch {batch.batch_index}")
        lines.append("")
        lines.append(f"Winner: {batch.winner_session_id} | Merged: {batch.merged}")
        lines.append(
            f"Baseline eval: ok={batch.master_baseline_evaluation.ok} passed={batch.master_baseline_evaluation.passed} failed={batch.master_baseline_evaluation.failed}"
        )
        if batch.master_evaluation:
            lines.append(
                f"Master eval: ok={batch.master_evaluation.ok} passed={batch.master_evaluation.passed} failed={batch.master_evaluation.failed}"
            )
        lines.append("")
        lines.append("| Session | Verified | OK | Passed | Failed | Workflow | Attempts | Model | Tool | Changed Files | Workspace | Run |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for session in batch.sessions:
            lines.append(
                f"| {session.session_id} | {session.verification_ok} | {session.evaluation.ok} | {session.evaluation.passed} | {session.evaluation.failed} | "
                f"{session.workflow_status or ''} | {session.entrypoint_attempts} | {session.model_calls} | {session.tool_calls} | {len(session.changed_files)} | {session.workspace_root} | {session.run_root or ''} |"
            )
        lines.append("")
    return "\n".join(lines)


def _clarifying_questions_for_goal(goal: str) -> list[str]:
    normalized = " ".join((goal or "").strip().split())
    if not normalized:
        return [
            "What exact change do you want Tokimon to make?",
            "Which files, docs, or tests should this update target?",
        ]

    generic = {
        "fix it",
        "improve it",
        "work on it",
        "do it",
        "help",
        "make it better",
        "something",
    }
    if normalized.lower() in generic:
        return [
            "What concrete outcome should be verified at the end?",
            "Which repository area should Tokimon modify first?",
        ]
    return []


def _goal_requires_changes(goal: str) -> bool:
    text = (goal or "").strip().lower()
    if not text:
        return False
    verbs = (
        "fix",
        "improve",
        "implement",
        "add",
        "update",
        "modify",
        "refactor",
        "harden",
        "work on",
    )
    return any(re.search(rf"\b{re.escape(verb)}\b", text) for verb in verbs)


def _verify_session_outcome(
    goal: str,
    baseline_master_eval: EvaluationResult,
    evaluation: EvaluationResult,
    workflow_ok: bool | None,
    changed_files: list[str],
    error: str | None,
) -> VerificationResult:
    if error:
        return VerificationResult(ok=False, reason=f"agent run error: {error}")
    if workflow_ok is not True:
        return VerificationResult(ok=False, reason=f"workflow not successful: {workflow_ok}")
    if not evaluation.ok:
        failed = evaluation.failed if evaluation.failed is not None else "unknown"
        return VerificationResult(ok=False, reason=f"verification tests failed: failed={failed}")

    requires_changes = _goal_requires_changes(goal)
    if requires_changes and not changed_files:
        return VerificationResult(
            ok=False,
            reason="verification failed: goal requires concrete repo changes but none were produced",
        )

    baseline_failed = int(baseline_master_eval.failed or 0)
    current_failed = int(evaluation.failed or 0)
    if baseline_failed > 0 and current_failed > baseline_failed:
        return VerificationResult(
            ok=False,
            reason=(
                "verification failed: regression detected "
                f"(baseline failed={baseline_failed}, current failed={current_failed})"
            ),
        )
    return VerificationResult(ok=True, reason="verification passed")


def _summarize_workflow_state(workflow_state_path: Path) -> tuple[bool | None, str | None, str | None]:
    if not workflow_state_path.exists():
        return None, None, "workflow_state.json missing"
    try:
        data = json.loads(workflow_state_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        return None, None, f"invalid workflow_state.json: {exc}"

    steps = data.get("state", {}).get("steps", {})
    if not isinstance(steps, dict) or not steps:
        return None, None, "workflow has no steps"

    statuses: list[str] = []
    failures: list[str] = []
    for step_id, step in steps.items():
        if not isinstance(step, dict):
            continue
        status = str(step.get("status") or "")
        statuses.append(status)
        if status not in {"SUCCEEDED", "SKIPPED"}:
            outputs = step.get("outputs") if isinstance(step.get("outputs"), dict) else {}
            summary = outputs.get("summary") if isinstance(outputs, dict) else None
            error = step.get("error")
            detail = f"{step_id}:{status or 'UNKNOWN'}"
            if summary:
                detail = f"{detail} summary={summary}"
            if error:
                detail = f"{detail} error={error}"
            failures.append(detail)

    workflow_status = _overall_workflow_status(statuses)
    ok = workflow_status == "SUCCEEDED" and not failures
    if ok:
        return True, workflow_status, None
    return False, workflow_status, f"{workflow_status}: " + "; ".join(failures)


def _overall_workflow_status(statuses: list[str]) -> str:
    if any(status == "FAILED" for status in statuses):
        return "FAILED"
    if any(status == "BLOCKED" for status in statuses):
        return "BLOCKED"
    if any(status == "PARTIAL" for status in statuses):
        return "PARTIAL"
    if statuses and all(status in {"SUCCEEDED", "SKIPPED"} for status in statuses):
        return "SUCCEEDED"
    return "UNKNOWN"
