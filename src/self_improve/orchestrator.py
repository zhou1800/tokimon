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
]

PATH_CHARTER_KEYS: tuple[str, ...] = (
    "decomposition",
    "root_cause_hypothesis",
    "tool_sequence",
    "skill_usage",
)

DECLARED_SCORING_FUNCTION = (
    "score = (verification_ok, evaluation_ok, workflow_ok, has_changes, passed_tests); "
    "higher is better (lexicographic). Tie-breaker: lowest session_id (lexicographic). "
    "Energy is reported for auditability but MUST NOT influence scoring."
)

EVAL_FIRST_EXPERIMENT_REQUIREMENTS = [
    "baseline evaluation summary",
    "post-change evaluation summary",
    "delta (improvement or regression)",
    "causal mechanism hypothesis linking changes to delta",
    "explicit pass condition for the run",
]

DEFAULT_PLANNED_TIME_S_PER_ATTEMPT = 10 * 60
DEFAULT_PLANNED_MEMORY_MB = 2048

DEFAULT_RISK_REGISTER: list[dict[str, str]] = [
    {
        "risk": "OOM / overscan from broad searches or large artifacts",
        "trigger": "very large tool output, repeated grep across generated directories",
        "mitigation": "bound outputs, narrow paths, shorten context, reduce concurrency",
    },
    {
        "risk": "Non-determinism (inconsistent selection or uncontrolled retries)",
        "trigger": "randomness, time-based branching, identical retries without novelty",
        "mitigation": "deterministic policies, novelty-gated retries, explicit audit logs",
    },
    {
        "risk": "Retry loops consume budget without verification",
        "trigger": "repeated timeouts/tool failures or evaluation regressions",
        "mitigation": "auto-degrade (reduce concurrency + shorten context) then stop PARTIAL",
    },
]

DEFAULT_STOP_CONDITIONS: dict[str, list[str]] = {
    "hard": [
        "Hard red line violation (unsafe goal or governance violation) -> BLOCKED (no agent execution).",
        "Explicit stop signal -> stop now; do not initiate new work.",
        "Verification/evaluation failing -> do not merge; preserve audit log.",
    ],
    "soft": [
        "Repeated tool failures/timeouts or stalled progress -> apply mitigations (reduce concurrency + shorten context).",
        "Mitigations exhausted or verification not feasible within remaining budget -> PARTIAL early stop with best artifacts + next-step plan.",
    ],
}


@dataclass(frozen=True)
class SelfImproveSettings:
    sessions_per_batch: int = 5
    batches: int = 1
    max_workers: int = 5
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
    score: tuple[int, int, int, int, int]
    model_calls: int
    tool_calls: int
    changed_files: list[str]
    changes: list[WorkspaceChange]
    path_charter: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    verification_ok: bool = False
    verification_reason: str | None = None
    clarifying_questions: list[str] = field(default_factory=list)
    entrypoint_attempts: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    causal_mechanism_hypothesis: str | None = None


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
        session_input_payload = _materialize_session_input(workspace_dir, input_payload)
        path_charter = _path_charter(session_id)

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
        hard_red_line = _hard_red_line_violation(goal)
        clarifying_questions = [] if hard_red_line else _clarifying_questions_for_goal(goal)
        attempts: list[dict[str, Any]] = []
        entrypoint_max_attempts = max(1, int(self.settings.entrypoint_max_attempts))
        planned_energy = _planned_energy_budget(1, entrypoint_max_attempts)
        planned_time_s = _planned_time_budget_s(entrypoint_max_attempts)
        planned_memory_mb = int(DEFAULT_PLANNED_MEMORY_MB)
        session_concurrency = max(1, int(self.settings.session_concurrency))
        context_mode = "full"
        soft_limit_triggered = False

        session_report: dict[str, Any] = {
            "session_id": session_id,
            "goal": goal,
            "input": {"kind": session_input_payload.kind, "ref": session_input_payload.ref},
            "path_charter": path_charter,
            "master_baseline_evaluation": baseline_master_eval.__dict__,
            "pytest_args": list(self.settings.pytest_args),
            "entrypoint_max_attempts": entrypoint_max_attempts,
            "resource_plan": {
                "planned_time_s": planned_time_s,
                "planned_memory_mb": planned_memory_mb,
                "planned_energy": planned_energy,
                "planned_concurrency": session_concurrency,
            },
            "risk_register": DEFAULT_RISK_REGISTER,
            "stop_conditions": DEFAULT_STOP_CONDITIONS,
            "clarifying_questions": clarifying_questions,
            "attempts": attempts,
            "status": "RUNNING",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (session_dir / "session.json").write_text(json.dumps(session_report, indent=2))

        if hard_red_line:
            workflow_ok = False
            workflow_status = "BLOCKED"
            workflow_error = hard_red_line
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
                path_charter=path_charter,
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
                clarifying_questions=[],
                entrypoint_attempts=0,
                attempts=attempts,
            )

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
                path_charter=path_charter,
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
        causal_mechanism_hypothesis: str | None = None

        for attempt_index in range(1, entrypoint_max_attempts + 1):
            entrypoint_attempts = attempt_index
            retry_reason = verification.reason if attempt_index > 1 else None
            session_input_payload = _materialize_session_input(workspace_dir, session_input_payload)
            session_goal = _entrypoint_prompt(
                goal=goal,
                session_id=session_id,
                input_payload=session_input_payload,
                baseline_master_eval=baseline_master_eval,
                pytest_args=self.settings.pytest_args,
                planned_energy=planned_energy,
                experiment_summary_path=_experiment_summary_relpath(session_id, attempt_index),
                planned_time_s=planned_time_s,
                planned_memory_mb=planned_memory_mb,
                session_concurrency=session_concurrency,
                context_mode=context_mode,
                attempt_index=attempt_index,
                retry_reason=retry_reason,
            )
            attempt_report: dict[str, Any] = {
                "attempt": attempt_index,
                "prompt": session_goal,
                "resource_plan": {
                    "planned_time_s": planned_time_s,
                    "planned_memory_mb": planned_memory_mb,
                    "planned_energy": planned_energy,
                    "concurrency": session_concurrency,
                    "context_mode": context_mode,
                },
                "mitigations_applied": [],
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
            attempt_elapsed_s = 0.0
            attempt_rss_kb: int | None = None

            try:
                experiment_summary_path = workspace_dir / _experiment_summary_relpath(session_id, attempt_index)
                _write_experiment_summary_skeleton(
                    experiment_summary_path,
                    session_id=session_id,
                    goal=goal,
                    baseline_master_eval=baseline_master_eval,
                    pass_condition=_pick_pass_condition(baseline_master_eval),
                    path_charter=path_charter,
                )
                attempt_start = time.perf_counter()
                result = runner.run(
                    session_goal,
                    task_steps=None,
                    task_id=f"self-improve-{session_id}-attempt-{attempt_index}",
                    test_args=self.settings.pytest_args,
                    concurrency=session_concurrency,
                )
                attempt_elapsed_s = time.perf_counter() - attempt_start
                attempt_rss_kb = _best_effort_rss_kb()
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
            experiment_summary_path = workspace_dir / _experiment_summary_relpath(session_id, attempt_index)
            experiment_summary, experiment_error = _repair_experiment_summary(
                experiment_summary_path,
                session_id=session_id,
                goal=goal,
                baseline_master_eval=baseline_master_eval,
                evaluation=evaluation,
                pass_condition=_pick_pass_condition(baseline_master_eval),
                path_charter=path_charter,
            )
            if experiment_error is not None:
                verification = VerificationResult(
                    ok=False,
                    reason=f"experiment summary invalid: {experiment_error}",
                )
            else:
                verification = _verify_session_outcome(
                    goal,
                    baseline_master_eval,
                    evaluation,
                    workflow_ok,
                    changed_files,
                    error,
                )
                if verification.ok and experiment_summary is not None:
                    causal_mechanism_hypothesis = str(
                        experiment_summary["causal_mechanism_hypothesis"]
                    ).strip()
            attempt_report.update(
                {
                    "status": "COMPLETED" if verification.ok else "FAILED",
                    "run_root": run_root,
                    "workflow_ok": workflow_ok,
                    "workflow_status": workflow_status,
                    "workflow_error": workflow_error,
                    "elapsed_s": attempt_elapsed_s,
                    "rss_kb": attempt_rss_kb,
                    "evaluation": evaluation.__dict__,
                    "model_calls": attempt_model_calls,
                    "tool_calls": attempt_tool_calls,
                    "changed_files": changed_files,
                    "error": error,
                    "verification": verification.__dict__,
                    "experiment_summary_path": str(_experiment_summary_relpath(session_id, attempt_index)),
                    "experiment_summary": experiment_summary,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            if not verification.ok:
                trigger = _soft_red_line_trigger(
                    attempt_error=attempt_error,
                    evaluation=evaluation,
                    baseline_master_eval=baseline_master_eval,
                )
                if trigger:
                    soft_limit_triggered = True
                    new_concurrency, new_context_mode, mitigation = _apply_soft_mitigation(
                        concurrency=session_concurrency,
                        context_mode=context_mode,
                    )
                    if mitigation:
                        attempt_report["mitigations_applied"].append({"reason": trigger, "change": mitigation})
                        session_concurrency = new_concurrency
                        context_mode = new_context_mode
                    else:
                        # No further deterministic mitigations available; stop early with PARTIAL.
                        workflow_ok = False
                        workflow_status = "PARTIAL"
                        workflow_error = trigger
                        error = f"{trigger} Mitigations exhausted; stopping early with PARTIAL."
                        verification = VerificationResult(ok=False, reason=error)
                        attempt_report["status"] = "PARTIAL"
                        session_report.update(
                            {
                                "status": "PARTIAL",
                                "workflow_ok": workflow_ok,
                                "workflow_status": workflow_status,
                                "workflow_error": workflow_error,
                                "verification": verification.__dict__,
                                "error": error,
                            }
                        )
                        (session_dir / "session.json").write_text(json.dumps(session_report, indent=2))
                        break
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
                    "causal_mechanism_hypothesis": causal_mechanism_hypothesis,
                    "error": error,
                }
            )
            (session_dir / "session.json").write_text(json.dumps(session_report, indent=2))
            if verification.ok:
                break

        if not verification.ok and soft_limit_triggered and workflow_status != "PARTIAL":
            workflow_status = "PARTIAL"

        score = _score(
            verification.ok,
            evaluation,
            workflow_ok,
            len(changed_files),
            model_calls,
            tool_calls,
        )
        final_status = "COMPLETED" if verification.ok else ("PARTIAL" if workflow_status == "PARTIAL" else "FAILED")
        session_report.update(
            {
                "status": final_status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        (session_dir / "session.json").write_text(json.dumps(session_report, indent=2))
        return SelfImproveSessionResult(
            session_id=session_id,
            path_charter=path_charter,
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
            causal_mechanism_hypothesis=causal_mechanism_hypothesis,
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
    experiment_summary_path: Path,
    planned_time_s: int = DEFAULT_PLANNED_TIME_S_PER_ATTEMPT,
    planned_memory_mb: int = DEFAULT_PLANNED_MEMORY_MB,
    session_concurrency: int = 1,
    context_mode: str = "full",
    *,
    attempt_index: int,
    retry_reason: str | None,
) -> str:
    strategy = _strategy_hint(session_id)
    path_charter = _path_charter(session_id)
    pass_condition = _pick_pass_condition(baseline_master_eval)
    parts = [
        CONSTITUTION_ACK,
        "",
        "Immutable Invariants:",
        *[f"- {item}" for item in IMMUTABLE_INVARIANTS],
        "",
        "## Evaluation Plan (Required)",
        f"- Planned energy budget (this session): {planned_energy}",
        "- Actual energy must be reported as model_calls + tool_calls.",
        "- Evaluation-first experiment loop: run baseline, change, re-measure, and report baseline/post-change/delta/causal/pass condition.",
        f"- Pass condition (this session): {pass_condition}",
        "",
        "## Resource Plan (Required)",
        f"- Planned time budget (this session): {int(planned_time_s)}s",
        f"- Planned memory budget (this session): {int(planned_memory_mb)} MB",
        f"- Planned energy budget (this session): {planned_energy} (energy = model_calls + tool_calls)",
        f"- Concurrency plan (this session): {int(session_concurrency)}",
        f"- Context mode (this session): {context_mode}",
        "",
        "## Risk Register (Required)",
        *[
            f"- Risk: {item['risk']} | Trigger: {item['trigger']} | Mitigation: {item['mitigation']}"
            for item in DEFAULT_RISK_REGISTER
        ],
        "",
        "## Stop Conditions (Required)",
        "Hard stops:",
        *[f"- {item}" for item in DEFAULT_STOP_CONDITIONS["hard"]],
        "",
        "Soft stops:",
        *[f"- {item}" for item in DEFAULT_STOP_CONDITIONS["soft"]],
        "",
        "Self-improve entry-point task.",
        "",
        "Execution flow (must be followed in order):",
        "1) Understand the user request; if ambiguous, stop and ask clarifying questions immediately.",
        "2) Once clear, continue.",
        "3) Run a baseline evaluation and summarize results.",
        "4) Identify the smallest candidate improvements that could move the evaluation signal.",
        "5) Execute improvements in short, verifiable steps.",
        "6) After each step, re-run evaluation signals and log progress.",
        "7) If progress stalls, change strategy (do not repeat identical attempts).",
        "8) If evaluation regresses, undo the change and record a Lesson about why it regressed.",
        "9) Verify final outcome; if verification fails, return details so the orchestrator retries from step 3.",
        "",
        f"User request: {goal.strip()}",
        "",
        f"Session: {session_id}",
        f"Attempt: {attempt_index}",
        f"Strategy: {strategy}",
        f"Path charter (Parallel Exploration Protocol): {json.dumps(path_charter, sort_keys=True)}",
        "",
        "Master baseline evaluation (before this session):",
        f"- ok: {baseline_master_eval.ok}",
        f"- passed: {baseline_master_eval.passed}",
        f"- failed: {baseline_master_eval.failed}",
        f"- failing_tests: {baseline_master_eval.failing_tests[:20]}",
        "",
        "Evaluation command (run via pytest tool):",
        f"- args: {pytest_args}",
        "",
        "Required reporting fields (for auditability):",
        *[f"- {item}" for item in EVAL_FIRST_EXPERIMENT_REQUIREMENTS],
        "",
        "Required attempt artifact (must be written by the agent):",
        f"- path: {experiment_summary_path}",
        "- json fields (minimum): causal_mechanism_hypothesis, pass_condition, baseline_evaluation, post_change_evaluation, delta, plan, path_charter, self_critique, lessons",
        "",
        "Parallel Exploration Protocol (required):",
        "1) Follow the path charter dimensions; do not collapse into other paths.",
        "2) Write a short plan and a 1-paragraph self-critique into the experiment summary artifact.",
        "3) Run the same evaluation harness as other paths and report comparable metrics.",
        "4) Do not modify governance invariants to 'win'.",
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
                input_payload.content if context_mode == "full" else _shorten_context(input_payload.content),
            ]
        )
    return "\n".join(parts).strip() + "\n"


def _shorten_context(text: str, *, max_chars: int = 2000) -> str:
    raw = text or ""
    if len(raw) <= max_chars:
        return raw
    return raw[: max(0, int(max_chars))].rstrip() + "\n\n[truncated for resource safety]\n"


def _planned_time_budget_s(max_attempts: int) -> int:
    return max(1, int(max_attempts)) * int(DEFAULT_PLANNED_TIME_S_PER_ATTEMPT)


def _hard_red_line_violation(goal: str) -> str | None:
    """Return a refusal reason if the goal is unsafe (hard red line), else None."""
    text = " ".join((goal or "").strip().split()).lower()
    if not text:
        return None
    patterns: list[tuple[str, str]] = [
        (r"\b(exploit|0day|zero[- ]day|rce)\b", "cyber exploitation"),
        (r"\b(malware|ransomware|trojan|keylogger|botnet)\b", "malware"),
        (r"\b(phish|phishing)\b", "credential theft"),
        (r"\b(steal|exfiltrate)\b.*\b(password|credential|token|api key|cookie)\b", "credential theft"),
        (r"\b(bypass|evade)\b.*\b(safety|sandbox|permission|guardrail)\b", "safety bypass"),
    ]
    for pattern, category in patterns:
        if re.search(pattern, text):
            return f"Hard red line: unsafe goal detected ({category}). Refuse and do not execute."
    return None


def _soft_red_line_trigger(
    *,
    attempt_error: str | None,
    evaluation: EvaluationResult,
    baseline_master_eval: EvaluationResult,
) -> str | None:
    err = (attempt_error or "").lower()
    if "timeout" in err or "timed out" in err:
        return "Soft red line: repeated timeout/tool failure risk."
    if baseline_master_eval.ok and not evaluation.ok:
        return "Soft red line: evaluation regression (tests failing)."
    return None


def _apply_soft_mitigation(
    *,
    concurrency: int,
    context_mode: str,
) -> tuple[int, str, str | None]:
    """Return (new_concurrency, new_context_mode, mitigation_summary_or_none)."""
    current_concurrency = max(1, int(concurrency))
    new_concurrency = max(1, current_concurrency // 2)
    new_context_mode = "degraded"
    if new_concurrency == current_concurrency and context_mode == new_context_mode:
        return current_concurrency, context_mode, None
    summary = f"reduce_concurrency {current_concurrency}->{new_concurrency}; context_mode {context_mode}->{new_context_mode}"
    return new_concurrency, new_context_mode, summary


def _best_effort_rss_kb() -> int | None:
    try:
        import resource  # noqa: PLC0415

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value) if value is not None else None
    except Exception:
        return None


def _pick_pass_condition(baseline_master_eval: EvaluationResult) -> str:
    baseline_failed = int(baseline_master_eval.failed or 0)
    if baseline_failed > 0:
        return "Reduce failing tests by >= 1 (relative to master baseline)."
    return "Maintain evaluation ok while keeping session energy within the planned energy budget."


def _pass_condition_met(
    baseline_master_eval: EvaluationResult,
    session_eval: EvaluationResult,
    session_energy: int,
    planned_session_energy: int,
) -> bool:
    baseline_failed = int(baseline_master_eval.failed or 0)
    current_failed = int(session_eval.failed or 0)
    if baseline_failed > 0:
        return session_eval.ok and current_failed <= (baseline_failed - 1)
    return session_eval.ok and session_energy <= planned_session_energy


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


def _materialize_session_input(workspace_dir: Path, input_payload: InputPayload) -> InputPayload:
    """Ensure local file inputs referenced in the prompt exist inside the session workspace.

    For safe/deterministic sessions, avoid path traversal and never leak absolute host paths into the agent prompt.
    """

    if input_payload.kind != "file" or not input_payload.ref:
        return input_payload

    raw_ref = str(input_payload.ref)
    ref_path = Path(raw_ref)
    fallback_relpath = Path(".tokimon-tmp") / "self-improve-inputs" / (ref_path.name or "input.txt")
    dest_relpath = ref_path
    if ref_path.is_absolute() or any(part == ".." for part in ref_path.parts):
        dest_relpath = fallback_relpath

    workspace_root = workspace_dir.resolve()
    dest_path = (workspace_dir / dest_relpath).resolve()
    if dest_path != workspace_root and workspace_root not in dest_path.parents:
        dest_relpath = fallback_relpath
        dest_path = (workspace_dir / dest_relpath).resolve()

    if dest_path.exists():
        if dest_path.is_file():
            if str(dest_relpath) == raw_ref:
                return input_payload
            return InputPayload(kind="file", ref=str(dest_relpath), content=input_payload.content)
        dest_relpath = fallback_relpath
        dest_path = (workspace_dir / dest_relpath).resolve()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(input_payload.content, encoding="utf-8")
    if str(dest_relpath) == raw_ref:
        return input_payload
    return InputPayload(kind="file", ref=str(dest_relpath), content=input_payload.content)


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


def _path_charter(session_id: str) -> dict[str, str]:
    """Return a deterministic Path Charter for Parallel Exploration Protocol.

    Charters must be meaningfully different across sessions (at least 2 dimensions).
    """

    charters: list[dict[str, str]] = [
        {
            "decomposition": "fine",
            "root_cause_hypothesis": "spec-doc mismatch or missing requirement mapping",
            "tool_sequence": "docs->tests->code",
            "skill_usage": "prefer skills when helpful",
        },
        {
            "decomposition": "coarse",
            "root_cause_hypothesis": "missing tests or weak assertions are hiding regressions",
            "tool_sequence": "tests->code",
            "skill_usage": "no special skills",
        },
        {
            "decomposition": "fine",
            "root_cause_hypothesis": "artifact schema/report gaps prevent reliable selection",
            "tool_sequence": "code->tests->docs",
            "skill_usage": "use ai-agent-cli style batching if relevant",
        },
        {
            "decomposition": "coarse",
            "root_cause_hypothesis": "determinism or winner-selection implementation bug",
            "tool_sequence": "read->code->tests",
            "skill_usage": "no special skills",
        },
        {
            "decomposition": "medium",
            "root_cause_hypothesis": "resource stress or concurrency corner case",
            "tool_sequence": "measure->cap->run",
            "skill_usage": "no special skills",
        },
    ]
    try:
        idx = int(session_id.split("-")[-1]) - 1
    except Exception:
        idx = 0
    return dict(charters[idx % len(charters)])


def _experiment_summary_relpath(session_id: str, attempt_index: int) -> Path:
    return Path(".tokimon-tmp") / "self-improve" / "experiment" / session_id / f"attempt-{attempt_index}.json"


def _evaluation_snapshot(result: EvaluationResult, *, max_failing_tests: int = 50) -> dict[str, Any]:
    failing_tests = result.failing_tests if isinstance(result.failing_tests, list) else []
    return {
        "ok": bool(result.ok),
        "passed": int(result.passed or 0),
        "failed": int(result.failed or 0),
        "failing_tests": [str(item) for item in failing_tests[: max(0, int(max_failing_tests))]],
    }


def _default_experiment_plan() -> list[str]:
    return [
        "Baseline eval",
        "Smallest change",
        "Re-run eval",
        "Report delta",
    ]


def _default_causal_hypothesis(goal: str) -> str:
    bounded_goal = " ".join((goal or "").strip().split())
    if len(bounded_goal) > 160:
        bounded_goal = bounded_goal[:160].rstrip() + "...(truncated)"
    if not bounded_goal:
        bounded_goal = "(empty goal)"
    return f"(auto-filled placeholder) Work aims to satisfy goal: {bounded_goal}"


def _write_experiment_summary_skeleton(
    path: Path,
    *,
    session_id: str,
    goal: str,
    baseline_master_eval: EvaluationResult,
    pass_condition: str,
    path_charter: dict[str, str],
) -> None:
    """Create a valid experiment summary artifact before the agent runs.

    This ensures the required artifact exists even when an attempt fails early.
    Agents are expected to update the placeholders with real content.
    """
    if path.exists():
        return
    baseline = _evaluation_snapshot(baseline_master_eval)
    payload = {
        "causal_mechanism_hypothesis": _default_causal_hypothesis(goal),
        "pass_condition": str(pass_condition or "").strip() or _pick_pass_condition(baseline_master_eval),
        "baseline_evaluation": baseline,
        "post_change_evaluation": dict(baseline),
        "delta": {"passed": 0, "failed": 0},
        "plan": _default_experiment_plan(),
        "path_charter": dict(path_charter),
        "self_critique": "(auto-filled placeholder) Update with risks, confidence, and likely failure modes.",
        "lessons": ["(auto-filled placeholder) Record at least one Lesson for auditability."],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _repair_experiment_summary(
    path: Path,
    *,
    session_id: str,
    goal: str,
    baseline_master_eval: EvaluationResult,
    evaluation: EvaluationResult,
    pass_condition: str,
    path_charter: dict[str, str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Best-effort repair so experiment summaries are valid and comparable."""
    payload: dict[str, Any] = {}
    try:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        loaded = json.loads(raw) if raw.strip() else {}
        if isinstance(loaded, dict):
            payload = loaded
    except Exception:
        payload = {}

    baseline = _evaluation_snapshot(baseline_master_eval)
    post_change = _evaluation_snapshot(evaluation)

    payload["baseline_evaluation"] = baseline
    payload["post_change_evaluation"] = post_change
    payload["delta"] = {
        "passed": int(post_change.get("passed", 0)) - int(baseline.get("passed", 0)),
        "failed": int(post_change.get("failed", 0)) - int(baseline.get("failed", 0)),
    }

    hypothesis = payload.get("causal_mechanism_hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        payload["causal_mechanism_hypothesis"] = _default_causal_hypothesis(goal)

    raw_pass_condition = payload.get("pass_condition")
    if not isinstance(raw_pass_condition, str) or not raw_pass_condition.strip():
        payload["pass_condition"] = str(pass_condition or "").strip() or _pick_pass_condition(baseline_master_eval)

    plan = payload.get("plan")
    if not isinstance(plan, list) or not plan or not all(isinstance(step, str) and step.strip() for step in plan):
        payload["plan"] = _default_experiment_plan()

    charter = payload.get("path_charter")
    if not isinstance(charter, dict):
        payload["path_charter"] = dict(path_charter)
    else:
        for key in PATH_CHARTER_KEYS:
            value = charter.get(key)
            if not isinstance(value, str) or not value.strip():
                charter[key] = str(path_charter.get(key, "")).strip() or f"(missing {key})"
        payload["path_charter"] = charter

    self_critique = payload.get("self_critique")
    if not isinstance(self_critique, str) or not self_critique.strip():
        payload["self_critique"] = "(auto-filled placeholder) Self-critique missing; update with failure modes and confidence."

    lessons = payload.get("lessons")
    if not isinstance(lessons, list) or not lessons or not all(isinstance(item, str) and item.strip() for item in lessons):
        payload["lessons"] = ["(auto-filled placeholder) Lessons missing; record what worked/failed and what to try next."]

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        return None, f"failed to write experiment summary artifact: {exc}"

    return _load_experiment_summary(path)


def _load_experiment_summary(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing required artifact at {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid JSON at {path}: {exc}"

    if not isinstance(payload, dict):
        return None, "artifact must be a JSON object"

    hypothesis = payload.get("causal_mechanism_hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return None, "causal_mechanism_hypothesis must be a non-empty string"

    pass_condition = payload.get("pass_condition")
    if not isinstance(pass_condition, str) or not pass_condition.strip():
        return None, "pass_condition must be a non-empty string"

    for field_name in ("baseline_evaluation", "post_change_evaluation"):
        value = payload.get(field_name)
        if not isinstance(value, dict):
            return None, f"{field_name} must be an object"
        required = {"ok", "passed", "failed", "failing_tests"}
        if not required.issubset(value.keys()):
            return None, f"{field_name} must include keys: {sorted(required)}"

    delta = payload.get("delta")
    if not isinstance(delta, dict):
        return None, "delta must be an object"
    if "passed" not in delta or "failed" not in delta:
        return None, "delta must include passed and failed"

    plan = payload.get("plan")
    if not isinstance(plan, list) or not plan:
        return None, "plan must be a non-empty list of strings"
    if not all(isinstance(step, str) and step.strip() for step in plan):
        return None, "plan must be a non-empty list of non-empty strings"

    path_charter = payload.get("path_charter")
    if not isinstance(path_charter, dict):
        return None, "path_charter must be an object"
    if not set(PATH_CHARTER_KEYS).issubset(path_charter.keys()):
        return None, f"path_charter must include keys: {list(PATH_CHARTER_KEYS)}"
    for key in PATH_CHARTER_KEYS:
        value = path_charter.get(key)
        if not isinstance(value, str) or not value.strip():
            return None, f"path_charter.{key} must be a non-empty string"

    self_critique = payload.get("self_critique")
    if not isinstance(self_critique, str) or not self_critique.strip():
        return None, "self_critique must be a non-empty string"

    lessons = payload.get("lessons")
    if not isinstance(lessons, list):
        return None, "lessons must be a list of strings"
    if not all(isinstance(item, str) and item.strip() for item in lessons):
        return None, "lessons must be a list of non-empty strings"

    return payload, None


def _score(
    verification_ok: bool,
    evaluation: EvaluationResult,
    workflow_ok: bool | None,
    changed_files: int,
    _model_calls: int,
    _tool_calls: int,
) -> tuple[int, int, int, int, int]:
    verified = 1 if verification_ok else 0
    ok = 1 if evaluation.ok else 0
    workflow = 1 if workflow_ok else 0
    has_changes = 1 if changed_files > 0 else 0
    passed = int(evaluation.passed or 0)
    # Primary: verification outcome; Secondary: evaluation ok; then workflow, concrete changes, and test counts.
    # Energy is reported for auditability but must not influence scoring/winner selection.
    return (verified, ok, workflow, has_changes, passed)


def _planned_energy_budget(sessions: int, max_attempts: int) -> int:
    total_sessions = max(0, int(sessions))
    attempts = max(1, int(max_attempts))
    # A self-improve attempt typically needs multiple model calls plus several tool calls
    # (docs reads, greps, patches, and at least one evaluation run). Use a conservative
    # per-attempt budget so agents do not stop early before making verifiable changes.
    return total_sessions * attempts * 50


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


def _session_sort_key(session: SelfImproveSessionResult) -> tuple[int, int, int, int, int, str]:
    return (
        -int(session.score[0]),
        -int(session.score[1]),
        -int(session.score[2]),
        -int(session.score[3]),
        -int(session.score[4]),
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
                        "path_charter": session.path_charter,
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
                        "causal_mechanism_hypothesis": session.causal_mechanism_hypothesis,
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
    planned_session_energy = _planned_energy_budget(
        1, int(report.settings.get("entrypoint_max_attempts") or 1)
    )
    planned_time_s = (
        int(report.settings.get("sessions_per_batch") or 0)
        * int(report.settings.get("batches") or 0)
        * _planned_time_budget_s(int(report.settings.get("entrypoint_max_attempts") or 1))
    )
    planned_memory_mb = int(DEFAULT_PLANNED_MEMORY_MB)
    planned_concurrency = max(1, int(report.settings.get("session_concurrency") or 1))
    actual_elapsed_s = _actual_elapsed_s(report)
    peak_rss_kb = _peak_rss_kb(report)
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
        "## Resource Plan",
        "",
        f"Planned time budget: {planned_time_s}s",
        f"Actual elapsed (best-effort): {actual_elapsed_s:.3f}s",
        f"Planned memory budget: {planned_memory_mb} MB",
        f"Actual peak RSS (best-effort): {peak_rss_kb if peak_rss_kb is not None else 'unknown'} KB",
        f"Planned concurrency (per session): {planned_concurrency}",
        "",
        "## Risk Register",
        "",
        *[
            f"- Risk: {item['risk']} | Trigger: {item['trigger']} | Mitigation: {item['mitigation']}"
            for item in DEFAULT_RISK_REGISTER
        ],
        "",
        "## Stop Conditions",
        "",
        "Hard stops:",
        *[f"- {item}" for item in DEFAULT_STOP_CONDITIONS["hard"]],
        "",
        "Soft stops:",
        *[f"- {item}" for item in DEFAULT_STOP_CONDITIONS["soft"]],
        "",
        "## Experiment Loop Summary",
        "",
        "This run follows an evaluation-first experiment loop: baseline evaluation, bounded changes, re-evaluation, then acceptance/rejection.",
        "",
        "Required reporting: baseline, post-change, delta, causal mechanism, pass condition.",
        "",
        "## Parallel Exploration Protocol",
        "",
        "This run uses diverse parallel paths with deterministic winner selection.",
        f"Declared scoring function: {DECLARED_SCORING_FUNCTION}",
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
            "Audit entries (details):",
            "",
        ]
    )
    for batch in report.batches:
        for session in batch.sessions:
            if (session.workflow_status or "").upper() == "BLOCKED":
                reason = (session.workflow_error or session.error or "blocked").strip()
                lines.append(f"- Refused action: session {session.session_id} ({reason})")
            if (session.workflow_status or "").upper() == "PARTIAL":
                reason = (session.workflow_error or session.error or "partial").strip()
                lines.append(f"- Soft stop: session {session.session_id} ({reason})")
            for attempt in session.attempts or []:
                attempt_id = attempt.get("attempt")
                plan = attempt.get("resource_plan") if isinstance(attempt.get("resource_plan"), dict) else {}
                conc = plan.get("concurrency", "")
                ctx = plan.get("context_mode", "")
                lines.append(
                    f"- Attempted action: session {session.session_id} attempt {attempt_id} "
                    f"(concurrency={conc} context_mode={ctx}) status={attempt.get('status','')}"
                )
                mitigations = attempt.get("mitigations_applied")
                if isinstance(mitigations, list) and mitigations:
                    for item in mitigations:
                        if isinstance(item, dict):
                            lines.append(
                                f"  - Mitigation applied: reason={item.get('reason','')} change={item.get('change','')}"
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
        diversity_ok, diversity_detail = _diversity_check(batch.sessions)
        lines.append(f"Diversity check: {'PASS' if diversity_ok else 'FAIL'} ({diversity_detail})")
        lines.append(f"Scoring (declared): {DECLARED_SCORING_FUNCTION}")
        lines.append(
            f"Baseline eval: ok={batch.master_baseline_evaluation.ok} passed={batch.master_baseline_evaluation.passed} failed={batch.master_baseline_evaluation.failed} failing_tests={batch.master_baseline_evaluation.failing_tests[:20]}"
        )
        winner = None
        if batch.winner_session_id:
            for session in batch.sessions:
                if session.session_id == batch.winner_session_id:
                    winner = session
                    break
        if winner:
            delta_passed = (winner.evaluation.passed or 0) - (batch.master_baseline_evaluation.passed or 0)
            delta_failed = (winner.evaluation.failed or 0) - (batch.master_baseline_evaluation.failed or 0)
            winner_energy = int(winner.model_calls) + int(winner.tool_calls)
            pass_condition = _pick_pass_condition(batch.master_baseline_evaluation)
            met = _pass_condition_met(
                batch.master_baseline_evaluation,
                winner.evaluation,
                winner_energy,
                planned_session_energy,
            )
            lines.append(
                f"Post-change eval (winner): ok={winner.evaluation.ok} passed={winner.evaluation.passed} failed={winner.evaluation.failed} failing_tests={winner.evaluation.failing_tests[:20]}"
            )
            lines.append(f"Delta (winner - baseline): passed={delta_passed} failed={delta_failed}")
            lines.append(f"Pass condition: {pass_condition} Met: {met}")
            winner_hypothesis = (winner.causal_mechanism_hypothesis or "").strip()
            lines.append(f"Causal mechanism hypothesis: {winner_hypothesis or 'N/A'}")
        if batch.master_evaluation:
            lines.append(
                f"Master eval: ok={batch.master_evaluation.ok} passed={batch.master_evaluation.passed} failed={batch.master_evaluation.failed} failing_tests={batch.master_evaluation.failing_tests[:20]}"
            )
        lines.append("")
        lines.append("Per-path comparison (Parallel Exploration Protocol):")
        lines.append("")
        lines.append(
            "| Session | Score | Verified | Eval OK | Passed | Changes | Decomposition | Hypothesis | Tool Seq | Skill Usage | Loser Reason | Lessons |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for session in batch.sessions:
            experiment = _latest_experiment_summary(session)
            lessons = ", ".join((experiment.get("lessons") or []) if isinstance(experiment, dict) else [])
            if not lessons:
                lessons = "N/A"
            loser_reason = _loser_reason(winner, session) if winner else "N/A"
            charter = session.path_charter or {}
            lines.append(
                f"| {session.session_id} | {session.score} | {session.verification_ok} | {session.evaluation.ok} | "
                f"{session.evaluation.passed} | {len(session.changed_files)} | "
                f"{charter.get('decomposition','')} | {charter.get('root_cause_hypothesis','')} | "
                f"{charter.get('tool_sequence','')} | {charter.get('skill_usage','')} | {loser_reason} | {lessons} |"
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


def _actual_elapsed_s(report: SelfImproveReport) -> float:
    total = 0.0
    for batch in report.batches:
        for session in batch.sessions:
            for attempt in session.attempts or []:
                elapsed = attempt.get("elapsed_s")
                if isinstance(elapsed, (int, float)):
                    total += float(elapsed)
    return total


def _peak_rss_kb(report: SelfImproveReport) -> int | None:
    peak: int | None = None
    for batch in report.batches:
        for session in batch.sessions:
            for attempt in session.attempts or []:
                rss = attempt.get("rss_kb")
                if isinstance(rss, int):
                    peak = rss if peak is None else max(peak, rss)
    return peak


def _latest_experiment_summary(session: SelfImproveSessionResult) -> dict[str, Any] | None:
    for attempt in reversed(session.attempts or []):
        summary = attempt.get("experiment_summary")
        if isinstance(summary, dict):
            return summary
    return None


def _charter_distance(a: dict[str, str] | None, b: dict[str, str] | None) -> int:
    left = a or {}
    right = b or {}
    distance = 0
    for key in PATH_CHARTER_KEYS:
        if str(left.get(key, "")).strip() != str(right.get(key, "")).strip():
            distance += 1
    return distance


def _diversity_check(sessions: list[SelfImproveSessionResult]) -> tuple[bool, str]:
    if not sessions:
        return False, "no sessions"
    min_distance: int | None = None
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            dist = _charter_distance(sessions[i].path_charter, sessions[j].path_charter)
            min_distance = dist if min_distance is None else min(min_distance, dist)
    if min_distance is None:
        return False, "insufficient sessions for diversity check"
    ok = min_distance >= 2
    return ok, f"min_pairwise_distance={min_distance} (require >= 2)"


def _loser_reason(winner: SelfImproveSessionResult | None, session: SelfImproveSessionResult) -> str:
    if winner is None:
        return "N/A"
    if session.session_id == winner.session_id:
        return "winner"
    if not session.verification_ok:
        return f"verification failed: {(session.verification_reason or 'unknown')}"
    if not session.evaluation.ok:
        return "evaluation not ok"
    if session.score == winner.score:
        return "tie lost by session_id"
    # Deterministic explanation: first tuple element where the winner dominates.
    for idx, (w, s) in enumerate(zip(winner.score, session.score), start=1):
        if w != s:
            return f"lower score at position {idx} ({s} < {w})"
    return "lower score"


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
