"""Self-improvement runner: parallel sessions, evaluation, and winner merge."""

from __future__ import annotations

import concurrent.futures
import inspect
import json
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
    merge_on_success: bool = True


@dataclass(frozen=True)
class EvaluationResult:
    ok: bool
    passed: int | None
    failed: int | None
    failing_tests: list[str]
    elapsed_s: float


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
    error: str | None = None


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
        session_results.sort(key=lambda r: r.score, reverse=True)
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
        session_goal = _session_goal(goal, session_id, input_payload, baseline_master_eval, self.settings.pytest_args)
        runner = HierarchicalRunner(workspace_dir, llm_client, base_dir=workspace_dir / "runs")

        run_root = None
        error = None
        workflow_ok: bool | None = None
        workflow_status: str | None = None
        workflow_error: str | None = None
        model_calls = 0
        tool_calls = 0

        session_report: dict[str, Any] = {
            "session_id": session_id,
            "goal": session_goal,
            "input": {"kind": input_payload.kind, "ref": input_payload.ref},
            "master_baseline_evaluation": baseline_master_eval.__dict__,
            "pytest_args": list(self.settings.pytest_args),
            "status": "RUNNING",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (session_dir / "session.json").write_text(json.dumps(session_report, indent=2))
        try:
            result = runner.run(
                session_goal,
                task_steps=None,
                task_id=f"self-improve-{session_id}",
                test_args=self.settings.pytest_args,
                concurrency=self.settings.session_concurrency,
            )
            run_root = str(result.run_context.root)
            workflow_ok, workflow_status, workflow_error = _summarize_workflow_state(result.workflow_state_path)
            if workflow_ok is False and error is None:
                error = workflow_error or "workflow failed"
            model_calls = result.model_calls
            tool_calls = result.tool_calls
        except Exception as exc:
            error = str(exc)

        evaluation = self._evaluate_workspace(workspace_dir)
        changes = compute_changes(self.master_root, workspace_dir, include_paths=self.settings.include_paths)
        changed_files = [change.relpath for change in changes]
        score = _score(evaluation, workflow_ok, len(changed_files), model_calls, tool_calls)
        session_report.update(
            {
                "status": "COMPLETED",
                "run_root": run_root,
                "workflow_ok": workflow_ok,
                "workflow_status": workflow_status,
                "workflow_error": workflow_error,
                "evaluation": evaluation.__dict__,
                "model_calls": model_calls,
                "tool_calls": tool_calls,
                "changed_files": changed_files,
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
            "merge_on_success": self.settings.merge_on_success,
        }


def _session_goal(
    goal: str,
    session_id: str,
    input_payload: InputPayload,
    baseline_master_eval: EvaluationResult,
    pytest_args: list[str],
) -> str:
    strategy = _strategy_hint(session_id)
    parts = [
        goal.strip(),
        "",
        f"Session: {session_id}",
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
    evaluation: EvaluationResult,
    workflow_ok: bool | None,
    changed_files: int,
    model_calls: int,
    tool_calls: int,
) -> tuple[int, int, int, int, int]:
    ok = 1 if evaluation.ok else 0
    workflow = 1 if workflow_ok else 0
    has_changes = 1 if changed_files > 0 else 0
    passed = int(evaluation.passed or 0)
    failed = int(evaluation.failed or 9999)
    # Primary: tests pass; Secondary: workflow completes; Tertiary: prefer actual changes; then maximize passed.
    return (ok, workflow, has_changes, passed, -failed - tool_calls - model_calls)


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
    lines = [
        "# Tokimon Self-Improve Report",
        "",
        "Status: COMPLETED",
        "",
        f"Goal: {report.goal}",
        "",
        f"Input: {report.input_payload.kind} {report.input_payload.ref or ''}".strip(),
        "",
        "## Batches",
        "",
    ]
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
        lines.append("| Session | OK | Passed | Failed | Workflow | Model | Tool | Changed Files | Workspace | Run |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for session in batch.sessions:
            lines.append(
                f"| {session.session_id} | {session.evaluation.ok} | {session.evaluation.passed} | {session.evaluation.failed} | "
                f"{session.workflow_status or ''} | {session.model_calls} | {session.tool_calls} | {len(session.changed_files)} | {session.workspace_root} | {session.run_root or ''} |"
            )
        lines.append("")
    return "\n".join(lines)


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
