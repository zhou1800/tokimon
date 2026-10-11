"""Operator CLI for code-audit rule/subject matrices and ledger updates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tokimon.workflow import audit_project as project
from tokimon.workflow.code_audit import contract, ledger, matrix, recheck_plan, reports, rules, subjects


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Code audit rule, subject, matrix, and ledger support.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract_parser = subparsers.add_parser("contract", help="Emit workflow contract declarations")
    contract_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    rules_parser = subparsers.add_parser("rules", help="Extract live code guideline rule IDs")
    rules_parser.add_argument("--source", action="append", help="Optional guideline source path; may repeat")
    rules_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    subjects_parser = subparsers.add_parser("subjects", help="Validate and normalize audit subjects")
    subjects_parser.add_argument("--subjects-from", required=True, help="JSON file containing audit subjects")
    subjects_parser.add_argument("--subject", action="append", help="Audit subject id filter; may repeat")
    subjects_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    matrix_parser = subparsers.add_parser("matrix", help="Write a full rule-subject checklist matrix")
    matrix_parser.add_argument("--scope-id", required=True, help="Stable audit scope id")
    matrix_parser.add_argument("--run-id", default=None, help="Optional audit run id")
    matrix_parser.add_argument("--run-root", default=None)
    matrix_parser.add_argument("--source", action="append", help="Optional guideline source path; may repeat")
    matrix_parser.add_argument("--subjects-from", required=True, help="JSON file containing audit subjects")
    matrix_parser.add_argument("--subject", action="append", help="Audit subject id filter; may repeat")
    matrix_parser.add_argument("--evaluations", default=None, help="Optional JSON file containing evaluated matrix rows")
    matrix_parser.add_argument("--artifact-prefix", default="checklist-matrix", help="Artifact filename prefix")
    matrix_parser.add_argument("--expect-rule-count", type=int, default=None)
    matrix_parser.add_argument("--expect-subject-count", type=int, default=None)
    matrix_parser.add_argument("--expect-matrix-count", type=int, default=None)
    matrix_parser.add_argument("--require-complete", action="store_true", help="Fail when any row remains unchecked")
    matrix_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    pass_locks_parser = subparsers.add_parser(
        "pass-locks",
        help="Write pass-locks for completed pass or not-applicable reports",
    )
    pass_locks_parser.add_argument("--matrix", required=True, help="Checklist matrix JSON path")
    pass_locks_parser.add_argument("--reports-dir", required=True, help="Directory containing report JSON files")
    pass_locks_parser.add_argument("--output-dir", required=True, help="Output directory for pass-locks.json")
    pass_locks_parser.add_argument("--recheck-plan", default=None, help="Optional validated recheck-plan JSON path")
    pass_locks_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    recheck_parser = subparsers.add_parser("recheck-plan", help="Write a lock-aware target recheck plan")
    recheck_parser.add_argument("--matrix", required=True, help="Current checklist matrix JSON path")
    recheck_parser.add_argument("--prior-pass-locks", required=True, help="Prior pass-locks.json path")
    recheck_parser.add_argument("--output-dir", required=True, help="Output directory for recheck-plan.json")
    recheck_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    verify_reports_parser = subparsers.add_parser(
        "verify-reports",
        help="Verify one structured report exists for every checklist row",
    )
    verify_reports_parser.add_argument("--matrix", required=True, help="Checklist matrix JSON path")
    verify_reports_parser.add_argument("--reports-dir", required=True, help="Directory containing report JSON files")
    verify_reports_parser.add_argument("--recheck-plan", default=None, help="Optional validated recheck-plan JSON path")
    verify_reports_parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero when any report is missing, invalid, duplicate, or extra",
    )
    verify_reports_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    compile_reports_parser = subparsers.add_parser(
        "compile-reports",
        help="Compile structured reports into support-only row evaluations",
    )
    compile_reports_parser.add_argument("--matrix", required=True, help="Checklist matrix JSON path")
    compile_reports_parser.add_argument("--reports-dir", required=True, help="Directory containing report JSON files")
    compile_reports_parser.add_argument("--recheck-plan", default=None, help="Optional validated recheck-plan JSON path")
    compile_reports_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluations-from-reports.json and reports-summary.json",
    )
    compile_reports_parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero without compiling when report coverage is incomplete",
    )
    compile_reports_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    record_parser = subparsers.add_parser(
        "record-findings",
        help="Record auditor-accepted current row dispositions",
    )
    record_parser.add_argument(
        "--accepted-evaluations",
        dest="evaluations",
        required=True,
        help="Auditor-owned JSON file containing accepted evaluated rows; not raw compile-reports support output",
    )
    record_parser.add_argument("--scope-id", required=True, help="Stable audit scope id")
    record_parser.add_argument("--run-id", default=None, help="Optional audit run id")
    record_parser.add_argument("--ledger", default=None)
    record_parser.add_argument("--run-root", default=None)
    record_parser.add_argument("--source", action="append", help="Optional guideline source path; may repeat")
    record_parser.add_argument("--allow-unknown-rules", action="store_true")
    record_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    summary_parser = subparsers.add_parser("ledger-summary", help="Summarize the current code-audit ledger")
    summary_parser.add_argument("--ledger", default=None)
    summary_parser.add_argument("--scope-id", default=None)
    summary_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    for command_parser in (parser, *subparsers.choices.values()):
        command_parser.add_argument(
            "--project-root", default=argparse.SUPPRESS,
            help="Audited project checkout; defaults to the caller or recorded matrix project",
        )
    return parser


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=contract.JSON_ENSURE_ASCII, indent=2, sort_keys=True))


def _contract_payload() -> dict[str, Any]:
    return {
        "workflow_id": contract.WORKFLOW_ID,
        "skill_id": contract.SKILL_ID,
        "python_package": contract.PYTHON_PACKAGE,
        "workflow_status": contract.WORKFLOW_STATUS,
        "canonical_entrypoint": contract.CANONICAL_ENTRYPOINT,
        "operator_entrypoint": contract.OPERATOR_ENTRYPOINT,
        "json_encoding": contract.JSON_ENCODING,
        "json_ensure_ascii": contract.JSON_ENSURE_ASCII,
        "project": project.current().to_dict(),
        "local_state_root": str(project.current().state_root(contract.WORKFLOW_ID)),
        "run_root": str(project.current().state_root(contract.WORKFLOW_ID) / "runs"),
        "ledger_root": str(project.current().state_root(contract.WORKFLOW_ID) / "ledger"),
        "current_ledger_path": str(project.current().state_root(contract.WORKFLOW_ID) / "ledger/current.json"),
        "report_dir_name": contract.REPORT_DIR_NAME,
        "compiled_evaluations_name": contract.COMPILED_EVALUATIONS_NAME,
        "reports_summary_name": contract.REPORTS_SUMMARY_NAME,
        "pass_locks_name": contract.PASS_LOCKS_NAME,
        "recheck_plan_name": contract.RECHECK_PLAN_NAME,
        "guideline_rule_sources": [str(path) for path in contract.GUIDELINE_RULE_SOURCES],
        "commands": list(contract.COMMANDS),
        "audit_statuses": list(contract.AUDIT_STATUSES),
        "durable_record_statuses": list(contract.DURABLE_RECORD_STATUSES),
        "ledger_statuses": list(contract.LEDGER_STATUSES),
        "required_subject_fields": list(contract.REQUIRED_SUBJECT_FIELDS),
        "subject_boundary_fields": list(contract.SUBJECT_BOUNDARY_FIELDS),
        "optional_subject_fields": list(contract.OPTIONAL_SUBJECT_FIELDS),
        "evidence_line_fields": list(contract.EVIDENCE_LINE_FIELDS),
        "evidence_text_fields": list(contract.EVIDENCE_TEXT_FIELDS),
        "current_ledger_schema_version": contract.CURRENT_LEDGER_SCHEMA_VERSION,
        "run_artifact_schema_version": contract.RUN_ARTIFACT_SCHEMA_VERSION,
        "checklist_matrix_schema_version": contract.CHECKLIST_MATRIX_SCHEMA_VERSION,
        "subjects_schema_version": contract.SUBJECTS_SCHEMA_VERSION,
        "evaluations_schema_version": contract.EVALUATIONS_SCHEMA_VERSION,
        "audit_report_statuses": list(contract.AUDIT_REPORT_STATUSES),
        "audit_report_schema_version": contract.AUDIT_REPORT_SCHEMA_VERSION,
        "pass_locks_schema_version": contract.PASS_LOCKS_SCHEMA_VERSION,
        "recheck_plan_schema_version": contract.RECHECK_PLAN_SCHEMA_VERSION,
        "workflow_readiness": contract.WORKFLOW_READINESS,
    }


def _render_contract_text(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            f"workflow_id: {payload['workflow_id']}",
            f"status: {payload['workflow_status']}",
            f"entrypoint: {payload['operator_entrypoint']}",
            f"current_ledger_path: {payload['current_ledger_path']}",
            f"encoding: {payload['json_encoding']}",
            f"ensure_ascii: {str(payload['json_ensure_ascii']).lower()}",
        )
    )


def _render_rules_text(payload: Mapping[str, Any]) -> str:
    lines = [f"rules: {payload['rule_count']}"]
    lines.extend(f"- {rule['rule_id']}: {rule['description']}" for rule in payload["rules"])
    return "\n".join(lines)


def _render_subjects_text(payload: Mapping[str, Any]) -> str:
    lines = [f"subjects: {payload['subject_count']}"]
    lines.extend(f"- {subject['audit_subject_id']}: {subject['title']}" for subject in payload["subjects"])
    return "\n".join(lines)


def _render_matrix_text(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    counts = summary["status_counts"]
    paths = payload["artifact_paths"]
    return "\n".join(
        (
            f"code-audit matrix recorded: {payload['run_id']}",
            f"scope_id: {payload['scope_id']}",
            f"rules: {summary['rule_count']}",
            f"subjects: {summary['audit_subject_count']}",
            f"matrix_count: {summary['matrix_count']}",
            (
                "status_counts: "
                f"pass={counts['pass']} fail={counts['fail']} "
                f"not_applicable={counts['not_applicable']} "
                f"needs_human_review={counts['needs_human_review']} "
                f"unchecked={counts['unchecked']}"
            ),
            f"complete: {str(summary['complete']).lower()}",
            f"matrix_json: {paths['matrix_json']}",
            f"summary_json: {paths['summary_json']}",
        )
    )


def _render_verify_reports_text(payload: Mapping[str, Any]) -> str:
    counts = payload["status_counts"]
    return "\n".join(
        (
            f"code-audit reports verified: {str(payload['complete']).lower()}",
            f"matrix_path: {payload['matrix_path']}",
            f"reports_dir: {payload['reports_dir']}",
            f"recheck_plan: {payload.get('recheck_plan_path') or '*'}",
            f"expected_reports: {payload['expected_report_count']}",
            f"expected_fresh_reports: {payload['expected_fresh_report_count']}",
            f"covered_reports: {payload['covered_report_count']}",
            f"locked_rows: {payload['locked_row_count']}",
            f"coverage_count: {payload['coverage_count']}",
            (
                "gaps: "
                f"missing={payload['missing_count']} invalid={payload['invalid_count']} "
                f"duplicate={payload['duplicate_count']} extra={payload['extra_count']} "
                f"unexpected={payload['unexpected_count']}"
            ),
            (
                "status_counts: "
                f"pass={counts['pass']} fail={counts['fail']} "
                f"not_applicable={counts['not_applicable']} "
                f"needs_human_review={counts['needs_human_review']}"
            ),
        )
    )


def _render_recheck_plan_text(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    paths = payload["artifact_paths"]
    return "\n".join(
        (
            "code-audit recheck plan recorded",
            f"locked_rows: {summary['locked_row_count']}",
            f"fresh_required_rows: {summary['fresh_required_row_count']}",
            f"lock_validation_available: {str(summary['lock_validation_available']).lower()}",
            f"recheck_plan_json: {paths['recheck_plan_json']}",
        )
    )


def _render_pass_locks_text(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "code-audit pass locks recorded",
            f"pass_locks: {payload['pass_lock_count']}",
            f"skipped: {payload['pass_lock_skipped_count']}",
            f"pass_locks_json: {payload['pass_locks_path']}",
        )
    )


def _render_compile_reports_text(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    paths = payload["artifact_paths"]
    return "\n".join(
        (
            f"code-audit reports compiled: {str(summary['complete']).lower()}",
            f"evaluations: {summary['evaluation_count']}",
            f"locked_rows: {summary['locked_row_count']}",
            f"evaluations_json: {paths['evaluations_json']}",
            f"summary_json: {paths['summary_json']}",
            f"pass_locks_json: {paths['pass_locks_json']}",
        )
    )


def _render_record_text(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            f"code-audit findings recorded: {payload['run_id']}",
            f"scope_id: {payload['scope_id']}",
            f"finding_count: {payload['finding_count']}",
            f"resolution_count: {payload['resolution_count']}",
            f"ledger_path: {payload['ledger_path']}",
            f"run_findings_path: {payload['run_findings_path']}",
        )
    )


def _render_summary_text(payload: Mapping[str, Any]) -> str:
    stats = payload["stats"]
    return "\n".join(
        (
            f"ledger_path: {payload['ledger_path']}",
            f"scope_id: {payload['scope_id'] or '*'}",
            f"updated_at: {payload['updated_at']}",
            f"entries: {stats['total']}",
            f"open: {stats['open_total']}",
            f"resolved: {stats['resolved_total']}",
            f"fail: {stats['by_status']['fail']}",
            f"needs_human_review: {stats['by_status']['needs_human_review']}",
            f"healthy: {str(payload['health']['healthy']).lower()}",
            f"open_missing_evidence_paths: {payload['health']['open_missing_evidence_path_count']}",
        )
    )


def _emit(payload: Mapping[str, Any], *, as_json: bool, text: str) -> None:
    if as_json:
        _print_json(payload)
    else:
        print(text)


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"code-audit-{stamp}"


def _run_id_for_evaluations(evaluations: list[dict[str, Any]], requested_run_id: str | None) -> str:
    if requested_run_id:
        return requested_run_id
    run_ids = {str(evaluation.get("run_id", "")).strip() for evaluation in evaluations}
    run_ids.discard("")
    if not evaluations:
        return _default_run_id()
    if len(run_ids) != 1:
        raise ValueError("evaluations require one shared run_id when --run-id is omitted")
    return run_ids.pop()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        explicit = getattr(args, "project_root", None)
        chosen = project.resolve(explicit or Path.cwd())
        for field in (
            "matrix", "reports_dir", "recheck_plan", "prior_pass_locks", "output_dir",
            "ledger", "run_root", "subjects_from", "evaluations",
        ):
            value = getattr(args, field, None)
            if value is not None:
                setattr(args, field, str(Path(value).expanduser().resolve()))
        if getattr(args, "matrix", None):
            payload = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
            recorded = project.from_artifact(payload)
            if recorded:
                if explicit and chosen != recorded:
                    raise ValueError("explicit project does not match the matrix checkout")
                chosen = recorded
        with project.selected(chosen):
            return _dispatch(args)
    except (OSError, ValueError) as exc:
        _emit({"ok": False, "error": str(exc)}, as_json=getattr(args, "json", False), text=str(exc))
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    command = args.command

    try:
        if command == "contract":
            payload = {"ok": True, "contract": _contract_payload()}
            _emit(payload, as_json=getattr(args, "json", False), text=_render_contract_text(payload["contract"]))
            return 0

        if command == "rules":
            sources = [Path(source) for source in args.source] if args.source else None
            payload = {"ok": True, **rules.rules_payload(sources)}
            _emit(payload, as_json=args.json, text=_render_rules_text(payload))
            return 0

        if command == "subjects":
            payload = {"ok": True, **subjects.subjects_payload(args.subjects_from, selected_ids=args.subject)}
            _emit(payload, as_json=args.json, text=_render_subjects_text(payload))
            return 0

        if command == "matrix":
            sources = [Path(source) for source in args.source] if args.source else None
            audit_subjects = subjects.load_subjects(args.subjects_from, selected_ids=args.subject)
            evaluations = ledger.load_evaluations(args.evaluations)
            run_id = _run_id_for_evaluations(evaluations, args.run_id)
            matrix_payload = matrix.build_matrix_payload(
                scope_id=args.scope_id,
                run_id=run_id,
                audit_subjects=audit_subjects,
                rule_sources=sources,
                evaluations=evaluations,
            )
            matrix.validate_matrix_expectations(
                matrix_payload,
                expect_rule_count=args.expect_rule_count,
                expect_subject_count=args.expect_subject_count,
                expect_matrix_count=args.expect_matrix_count,
                require_complete=args.require_complete,
            )
            result = matrix.write_matrix_artifacts(
                matrix_payload,
                run_root=project.runtime_path(args.run_root, workflow_id=contract.WORKFLOW_ID, default="runs"),
                artifact_prefix=args.artifact_prefix,
            )
            payload = {"ok": True, **result}
            _emit(payload, as_json=args.json, text=_render_matrix_text(payload))
            return 0

        if command == "pass-locks":
            snapshot = reports.collect_report_coverage(
                matrix_path=args.matrix,
                reports_dir=args.reports_dir,
                recheck_plan_path=args.recheck_plan,
            )
            verification = snapshot["verification"]
            if not verification["complete"]:
                payload = {"ok": False, **verification}
                _emit(payload, as_json=args.json, text=_render_verify_reports_text(payload))
                return 2
            fresh_reports = snapshot["fresh_reports"]
            locked_reports = [
                dict(
                    row["locked_evaluation"],
                    run_id=row["locked_evaluation"].get("prior_run_id", row["locked_evaluation"]["run_id"]),
                    report_path=row["prior_report_path"],
                )
                for row in verification.get("locked_rows", [])
                if isinstance(row.get("locked_evaluation"), Mapping)
            ]
            payload = {
                "ok": True,
                "workflow_id": contract.WORKFLOW_ID,
                **recheck_plan.write_pass_locks(
                    matrix_path=args.matrix,
                    reports=[*fresh_reports, *locked_reports],
                    output_dir=args.output_dir,
                    reports_dir=args.reports_dir,
                    recheck_plan_path=args.recheck_plan,
                ),
            }
            _emit(payload, as_json=args.json, text=_render_pass_locks_text(payload))
            return 0

        if command == "recheck-plan":
            payload = {
                "ok": True,
                **recheck_plan.write_recheck_plan(
                    matrix_path=args.matrix,
                    prior_pass_locks_path=args.prior_pass_locks,
                    output_dir=args.output_dir,
                ),
            }
            _emit(payload, as_json=args.json, text=_render_recheck_plan_text(payload))
            return 0

        if command == "verify-reports":
            verification = reports.verify_report_coverage(
                matrix_path=args.matrix,
                reports_dir=args.reports_dir,
                recheck_plan_path=args.recheck_plan,
            )
            payload = {"ok": True, **verification}
            if args.require_complete and not verification["complete"]:
                payload["ok"] = False
                _emit(payload, as_json=args.json, text=_render_verify_reports_text(payload))
                return 2
            _emit(payload, as_json=args.json, text=_render_verify_reports_text(payload))
            return 0

        if command == "compile-reports":
            result = reports.compile_reports(
                matrix_path=args.matrix,
                reports_dir=args.reports_dir,
                output_dir=args.output_dir,
                recheck_plan_path=args.recheck_plan,
                require_complete=args.require_complete,
            )
            if not result["compiled"]:
                payload = {"ok": False, **result["verification"]}
                _emit(payload, as_json=args.json, text=_render_verify_reports_text(payload))
                return 2
            payload = {"ok": True, **result}
            _emit(payload, as_json=args.json, text=_render_compile_reports_text(payload))
            return 0

        if command == "record-findings":
            sources = [Path(source) for source in args.source] if args.source else None
            extracted_rules = rules.extract_rules(sources)
            rules_by_id = None if args.allow_unknown_rules else {rule["rule_id"]: rule for rule in extracted_rules}
            evaluations = ledger.load_accepted_evaluations(args.evaluations)
            run_id = _run_id_for_evaluations(evaluations, args.run_id)
            payload = {
                "ok": True,
                **ledger.record_evaluations(
                    evaluations=evaluations,
                    scope_id=args.scope_id,
                    run_id=run_id,
                    ledger_path=project.runtime_path(args.ledger, workflow_id=contract.WORKFLOW_ID, default="ledger/current.json"),
                    run_root=project.runtime_path(args.run_root, workflow_id=contract.WORKFLOW_ID, default="runs"),
                    rules_by_id=rules_by_id,
                ),
            }
            _emit(payload, as_json=args.json, text=_render_record_text(payload))
            return 0

        if command == "ledger-summary":
            payload = {"ok": True, **ledger.summarize_ledger(project.runtime_path(args.ledger, workflow_id=contract.WORKFLOW_ID, default="ledger/current.json"), scope_id=args.scope_id)}
            _emit(payload, as_json=args.json, text=_render_summary_text(payload))
            return 0

    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {"ok": False, "error": str(exc)}
        _emit(payload, as_json=getattr(args, "json", False), text=str(exc))
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
