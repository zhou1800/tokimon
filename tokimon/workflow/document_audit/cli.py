"""Operator CLI for document-audit rule extraction and ledger maintenance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tokimon.workflow import audit_project as project
from tokimon.workflow.document_audit import audit_units, contract, ledger, matrix, recheck_plan, reports, rules


def required_tool_capabilities() -> tuple[str, ...]:
    return ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokimon document audit rule and ledger support.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract_parser = subparsers.add_parser("contract", help="Emit workflow contract declarations")
    contract_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    rules_parser = subparsers.add_parser("rules", help="Extract live guideline rule IDs")
    rules_parser.add_argument("--source", action="append", help="Optional guideline source path; may repeat")
    rules_parser.add_argument("--source-ref", default=None, help="Optional git tree-ish for reading guideline sources")
    rules_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    matrix_parser = subparsers.add_parser("matrix", help="Write a full rule-document checklist matrix")
    matrix_parser.add_argument("--scope-id", required=True, help="Stable audit scope id")
    matrix_parser.add_argument("--run-id", default=None, help="Optional audit run id")
    matrix_parser.add_argument("--run-root", default=None)
    matrix_parser.add_argument("--source", action="append", help="Optional guideline source path; may repeat")
    matrix_parser.add_argument(
        "--source-ref",
        default=None,
        help="Optional git tree-ish for reading guideline sources",
    )
    matrix_parser.add_argument("--document", action="append", help="Document path under audit; may repeat")
    matrix_parser.add_argument("--document-glob", action="append", help="Document glob under audit; may repeat")
    matrix_parser.add_argument("--documents-from", default=None, help="Text or JSON file containing document paths")
    matrix_parser.add_argument(
        "--units-from",
        default=None,
        help="Optional JSON audit-unit definitions; defaults to one unit per document",
    )
    matrix_parser.add_argument(
        "--evaluations",
        default=None,
        help="Optional JSON file containing evaluated matrix rows",
    )
    matrix_parser.add_argument("--artifact-prefix", default="checklist-matrix", help="Artifact filename prefix")
    matrix_parser.add_argument("--expect-rule-count", type=int, default=None)
    matrix_parser.add_argument("--expect-document-count", type=int, default=None)
    matrix_parser.add_argument("--expect-matrix-count", type=int, default=None)
    matrix_parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every matrix row is pass, fail, or not_applicable",
    )
    matrix_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    locks_parser = subparsers.add_parser(
        "clean-unit-locks",
        help="Write support-only clean, settled, and waiting audit-unit state",
    )
    locks_parser.add_argument("--matrix", required=True, help="Full checklist matrix JSON path")
    locks_parser.add_argument("--reports-dir", required=True, help="Directory containing fresh report JSON files")
    locks_parser.add_argument("--output-dir", required=True, help="Output directory for clean-unit-locks.json")
    locks_parser.add_argument("--ledger", default=None)
    locks_parser.add_argument("--recheck-plan", default=None, help="Optional validated recheck-plan JSON path")
    locks_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    recheck_parser = subparsers.add_parser(
        "recheck-plan",
        help="Write a clean-unit lock-aware target recheck plan",
    )
    recheck_parser.add_argument("--matrix", required=True, help="Current full checklist matrix JSON path")
    recheck_parser.add_argument(
        "--prior-clean-unit-locks",
        required=True,
        help="Prior clean-unit-locks.json path",
    )
    recheck_parser.add_argument("--output-dir", required=True, help="Output directory for recheck-plan.json")
    recheck_parser.add_argument("--ledger", default=None)
    recheck_parser.add_argument(
        "--activate-unit",
        action="append",
        help="Explicit audit unit to activate; dependents follow bound-input invalidation; may repeat",
    )
    recheck_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    verify_reports_parser = subparsers.add_parser(
        "verify-reports",
        help="Verify one structured report exists for every checklist row",
    )
    verify_reports_parser.add_argument("--matrix", required=True, help="Checklist matrix JSON path")
    verify_reports_parser.add_argument("--reports-dir", required=True, help="Directory containing report JSON files")
    verify_reports_parser.add_argument(
        "--recheck-plan",
        default=None,
        help="Optional validated recheck-plan JSON path",
    )
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
    compile_reports_parser.add_argument(
        "--recheck-plan",
        default=None,
        help="Optional validated recheck-plan JSON path",
    )
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

    summary_parser = subparsers.add_parser("ledger-summary", help="Summarize the compact current ledger")
    summary_parser.add_argument("--ledger", default=None)
    summary_parser.add_argument("--scope-id", default=None)
    summary_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    record_parser = subparsers.add_parser(
        "record-findings",
        help="Merge auditor-accepted findings into the current ledger",
    )
    record_parser.add_argument(
        "--accepted-findings",
        dest="findings",
        required=True,
        help="Auditor-owned JSON file containing accepted findings; not a compile-reports support artifact",
    )
    record_parser.add_argument("--scope-id", required=True, help="Stable audit scope id")
    record_parser.add_argument("--run-id", default=None, help="Optional audit run id")
    record_parser.add_argument("--ledger", default=None)
    record_parser.add_argument("--run-root", default=None)
    record_parser.add_argument(
        "--source",
        action="append",
        help="Optional guideline source path for rule validation; may repeat",
    )
    record_parser.add_argument(
        "--source-ref",
        default=None,
        help="Optional git tree-ish for reading guideline sources",
    )
    record_parser.add_argument(
        "--allow-unknown-rules",
        action="store_true",
        help="Allow findings whose rule_id is not in the live guideline rules",
    )
    record_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    for command_parser in (parser, *subparsers.choices.values()):
        command_parser.add_argument(
            "--project-root",
            default=argparse.SUPPRESS,
            help="Audited project checkout; defaults to the caller or recorded matrix project",
        )
    return parser


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=contract.JSON_ENSURE_ASCII, indent=2, sort_keys=True))


def _contract_payload() -> dict[str, Any]:
    return {
        "project": project.current().to_dict(),
        "workflow_id": contract.WORKFLOW_ID,
        "skill_id": contract.SKILL_ID,
        "python_package": contract.PYTHON_PACKAGE,
        "workflow_status": contract.WORKFLOW_STATUS,
        "canonical_entrypoint": contract.CANONICAL_ENTRYPOINT,
        "operator_entrypoint": contract.OPERATOR_ENTRYPOINT,
        "json_encoding": contract.JSON_ENCODING,
        "json_ensure_ascii": contract.JSON_ENSURE_ASCII,
        "local_state_root": str(project.current().state_root(contract.WORKFLOW_ID)),
        "run_root": str(project.current().state_root(contract.WORKFLOW_ID) / "runs"),
        "ledger_root": str(project.current().state_root(contract.WORKFLOW_ID) / "ledger"),
        "current_ledger_path": str(project.current().state_root(contract.WORKFLOW_ID) / "ledger/current.json"),
        "report_dir_name": contract.REPORT_DIR_NAME,
        "compiled_evaluations_name": contract.COMPILED_EVALUATIONS_NAME,
        "reports_summary_name": contract.REPORTS_SUMMARY_NAME,
        "audit_units_name": contract.AUDIT_UNITS_NAME,
        "clean_unit_locks_name": contract.CLEAN_UNIT_LOCKS_NAME,
        "recheck_plan_name": contract.RECHECK_PLAN_NAME,
        "guideline_rule_sources": [str(path) for path in contract.GUIDELINE_RULE_SOURCES],
        "commands": list(contract.COMMANDS),
        "finding_statuses": list(contract.FINDING_STATUSES),
        "terminal_finding_statuses": list(contract.TERMINAL_FINDING_STATUSES),
        "finding_severities": list(contract.FINDING_SEVERITIES),
        "required_finding_fields": list(contract.REQUIRED_FINDING_FIELDS),
        "evidence_line_fields": list(contract.EVIDENCE_LINE_FIELDS),
        "evidence_text_fields": list(contract.EVIDENCE_TEXT_FIELDS),
        "current_ledger_schema_version": contract.CURRENT_LEDGER_SCHEMA_VERSION,
        "legacy_ledger_schema_versions": list(contract.LEGACY_LEDGER_SCHEMA_VERSIONS),
        "run_artifact_schema_version": contract.RUN_ARTIFACT_SCHEMA_VERSION,
        "workflow_readiness": contract.WORKFLOW_READINESS,
        "checklist_statuses": list(contract.CHECKLIST_STATUSES),
        "audit_report_statuses": list(contract.AUDIT_REPORT_STATUSES),
        "checklist_matrix_schema_version": contract.CHECKLIST_MATRIX_SCHEMA_VERSION,
        "legacy_checklist_matrix_schema_versions": list(contract.LEGACY_CHECKLIST_MATRIX_SCHEMA_VERSIONS),
        "audit_report_schema_version": contract.AUDIT_REPORT_SCHEMA_VERSION,
        "review_fields": ["reviewer_id", "source_path", "source_sha256"],
        "review_source_fields": ["reviewer_id", "reports"],
        "legacy_audit_report_schema_versions": list(contract.LEGACY_AUDIT_REPORT_SCHEMA_VERSIONS),
        "audit_units_schema_version": contract.AUDIT_UNITS_SCHEMA_VERSION,
        "clean_unit_locks_schema_version": contract.CLEAN_UNIT_LOCKS_SCHEMA_VERSION,
        "recheck_plan_schema_version": contract.RECHECK_PLAN_SCHEMA_VERSION,
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


def _render_summary_text(payload: Mapping[str, Any]) -> str:
    stats = payload["stats"]
    return "\n".join(
        (
            f"ledger_path: {payload['ledger_path']}",
            f"scope_id: {payload['scope_id'] or '*'}",
            f"updated_at: {payload['updated_at']}",
            f"entries: {stats['total']}",
            f"open_must_fix: {stats['open_must_fix']}",
            f"open_should_fix: {stats['open_should_fix']}",
            f"deferred: {stats['by_status']['deferred']}",
            f"rejected: {stats['by_status']['rejected']}",
            f"healthy: {str(payload['health']['healthy']).lower()}",
            f"open_missing_files: {payload['health']['open_missing_file_count']}",
        )
    )


def _render_record_text(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            f"document-audit findings recorded: {payload['run_id']}",
            f"scope_id: {payload['scope_id']}",
            f"finding_count: {payload['finding_count']}",
            f"ledger_path: {payload['ledger_path']}",
            f"run_findings_path: {payload['run_findings_path']}",
        )
    )


def _render_matrix_text(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    counts = summary["status_counts"]
    paths = payload["artifact_paths"]
    return "\n".join(
        (
            f"document-audit matrix recorded: {payload['run_id']}",
            f"scope_id: {payload['scope_id']}",
            f"rules: {summary['rule_count']}",
            f"documents: {summary['document_count']}",
            f"audit_units: {summary['audit_unit_count']}",
            f"matrix_count: {summary['matrix_count']}",
            f"audit_snapshot_id: {summary['audit_snapshot_id']}",
            (
                "status_counts: "
                f"pass={counts['pass']} fail={counts['fail']} "
                f"not_applicable={counts['not_applicable']} unchecked={counts['unchecked']}"
            ),
            f"complete: {str(summary['complete']).lower()}",
            f"matrix_json: {paths['matrix_json']}",
            f"audit_units_json: {paths['audit_units_json']}",
            f"summary_json: {paths['summary_json']}",
        )
    )


def _render_verify_reports_text(payload: Mapping[str, Any]) -> str:
    counts = payload["status_counts"]
    return "\n".join(
        (
            f"document-audit reports verified: {str(payload['complete']).lower()}",
            f"matrix_path: {payload['matrix_path']}",
            f"reports_dir: {payload['reports_dir']}",
            f"expected_reports: {payload['expected_report_count']}",
            f"expected_fresh_reports: {payload['expected_fresh_report_count']}",
            f"covered_reports: {payload['covered_report_count']}",
            f"locked_rows: {payload['locked_row_count']}",
            f"settled_rows: {payload['settled_row_count']}",
            f"waiting_rows: {payload['waiting_row_count']}",
            f"final_coverage_complete: {str(payload['final_coverage_complete']).lower()}",
            (
                "gaps: "
                f"missing={payload['missing_count']} invalid={payload['invalid_count']} "
                f"duplicate={payload['duplicate_count']} extra={payload['extra_count']} "
                f"unexpected={payload['unexpected_count']}"
            ),
            (
                "status_counts: "
                f"pass={counts['pass']} fail={counts['fail']} "
                f"not_applicable={counts['not_applicable']}"
            ),
        )
    )


def _render_compile_reports_text(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    paths = payload["artifact_paths"]
    return "\n".join(
        (
            f"document-audit reports compiled: {str(summary['complete']).lower()}",
            f"evaluations: {summary['evaluation_count']}",
            f"locked_rows: {summary['locked_row_count']}",
            f"settled_rows: {summary['settled_row_count']}",
            f"waiting_rows: {summary['waiting_row_count']}",
            f"final_coverage_complete: {str(summary['final_coverage_complete']).lower()}",
            f"evaluations_json: {paths['evaluations_json']}",
            f"summary_json: {paths['summary_json']}",
        )
    )


def _render_clean_unit_locks_text(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "document-audit clean-unit locks recorded",
            f"clean_units: {payload['clean_unit_lock_count']}",
            f"settled_units: {payload['settled_unit_count']}",
            f"skipped_units: {payload['clean_unit_lock_skipped_count']}",
            f"clean_unit_locks_json: {payload['clean_unit_locks_path']}",
        )
    )


def _render_recheck_plan_text(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join(
        (
            "document-audit recheck plan recorded",
            f"locked_units: {summary['locked_unit_count']}",
            f"settled_units: {summary['settled_unit_count']}",
            f"waiting_units: {summary['waiting_unit_count']}",
            f"active_units: {summary['active_unit_count']}",
            f"locked_rows: {summary['locked_row_count']}",
            f"settled_rows: {summary['settled_row_count']}",
            f"waiting_rows: {summary['waiting_row_count']}",
            f"fresh_required_rows: {summary['fresh_required_row_count']}",
            f"recheck_plan_json: {payload['artifact_paths']['recheck_plan_json']}",
        )
    )


def _emit(payload: Mapping[str, Any], *, as_json: bool, text: str) -> None:
    if as_json:
        _print_json(payload)
    else:
        print(text)


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"document-audit-{stamp}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        explicit = getattr(args, "project_root", None)
        chosen = project.resolve(explicit or Path.cwd())
        # Artifact arguments belong to the invocation, not to the recorded subject.
        for field in (
            "matrix", "reports_dir", "recheck_plan", "prior_clean_unit_locks", "output_dir",
            "ledger", "run_root", "units_from", "documents_from", "evaluations", "findings",
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
            payload = {"ok": True, **rules.rules_payload(sources, source_ref=args.source_ref)}
            _emit(payload, as_json=args.json, text=_render_rules_text(payload))
            return 0

        if command == "matrix":
            sources = [Path(source) for source in args.source] if args.source else None
            unit_definitions = audit_units.load_units(args.units_from) if args.units_from else None
            explicit_documents = bool(args.document or args.document_glob or args.documents_from)
            if unit_definitions is not None and not explicit_documents:
                document_paths = [
                    document_path
                    for unit in unit_definitions
                    for document_path in unit["document_paths"]
                ]
            else:
                document_paths = matrix.collect_document_paths(
                    documents=args.document,
                    document_globs=args.document_glob,
                    documents_from=args.documents_from,
                )
            matrix_payload = matrix.build_matrix_payload(
                scope_id=args.scope_id,
                run_id=args.run_id or _default_run_id(),
                document_paths=document_paths,
                unit_definitions=unit_definitions,
                rule_sources=sources,
                source_ref=args.source_ref,
                evaluations=matrix.load_evaluations(args.evaluations),
            )
            matrix.validate_matrix_expectations(
                matrix_payload,
                expect_rule_count=args.expect_rule_count,
                expect_document_count=args.expect_document_count,
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

        if command == "clean-unit-locks":
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
            locked_reports = [
                dict(
                    row["locked_evaluation"],
                    report_path=row["prior_report_path"],
                    findings=[],
                )
                for row in verification.get("locked_rows", [])
                if isinstance(row.get("locked_evaluation"), Mapping)
            ]
            settled_reports = [
                dict(
                    row["settled_evaluation"],
                    report_path=row["prior_report_path"],
                )
                for row in verification.get("settled_rows", [])
                if isinstance(row.get("settled_evaluation"), Mapping)
            ]
            payload = {
                "ok": True,
                "workflow_id": contract.WORKFLOW_ID,
                **recheck_plan.write_clean_unit_locks(
                    matrix_path=args.matrix,
                    reports=[*snapshot["fresh_reports"], *locked_reports, *settled_reports],
                    output_dir=Path(args.output_dir),
                    reports_dir=args.reports_dir,
                    ledger_path=project.runtime_path(args.ledger, workflow_id=contract.WORKFLOW_ID, default="ledger/current.json"),
                    recheck_plan_path=args.recheck_plan,
                ),
            }
            _emit(payload, as_json=args.json, text=_render_clean_unit_locks_text(payload))
            return 0

        if command == "recheck-plan":
            payload = {
                "ok": True,
                **recheck_plan.write_recheck_plan(
                    matrix_path=args.matrix,
                    prior_clean_unit_locks_path=args.prior_clean_unit_locks,
                    output_dir=Path(args.output_dir),
                    ledger_path=project.runtime_path(args.ledger, workflow_id=contract.WORKFLOW_ID, default="ledger/current.json"),
                    activated_unit_ids=args.activate_unit or (),
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
            compiled = reports.compile_reports(
                matrix_path=args.matrix,
                reports_dir=args.reports_dir,
                output_dir=args.output_dir,
                recheck_plan_path=args.recheck_plan,
                require_complete=args.require_complete,
            )
            if compiled.get("compiled") is False:
                payload = {"ok": False, **compiled["verification"]}
                _emit(payload, as_json=args.json, text=_render_verify_reports_text(payload))
                return 2
            payload = {"ok": True, **compiled}
            _emit(payload, as_json=args.json, text=_render_compile_reports_text(payload))
            return 0

        if command == "ledger-summary":
            payload = {"ok": True, **ledger.summarize_ledger(project.runtime_path(args.ledger, workflow_id=contract.WORKFLOW_ID, default="ledger/current.json"), scope_id=args.scope_id)}
            _emit(payload, as_json=args.json, text=_render_summary_text(payload))
            return 0

        if command == "record-findings":
            sources = [Path(source) for source in args.source] if args.source else None
            extracted_rules = rules.extract_rules(sources, source_ref=args.source_ref)
            rules_by_id = None if args.allow_unknown_rules else {rule["rule_id"]: rule for rule in extracted_rules}
            findings = ledger.load_findings(args.findings)
            payload = {
                "ok": True,
                **ledger.record_findings(
                    findings=findings,
                    scope_id=args.scope_id,
                    run_id=args.run_id or _default_run_id(),
                    ledger_path=project.runtime_path(args.ledger, workflow_id=contract.WORKFLOW_ID, default="ledger/current.json"),
                    run_root=project.runtime_path(args.run_root, workflow_id=contract.WORKFLOW_ID, default="runs"),
                    rules_by_id=rules_by_id,
                ),
            }
            _emit(payload, as_json=args.json, text=_render_record_text(payload))
            return 0

    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {"ok": False, "error": str(exc)}
        _emit(payload, as_json=getattr(args, "json", False), text=str(exc))
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
