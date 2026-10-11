from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.workflow.document_audit.helpers import reviewed_report
from tokimon.workflow import audit_project
from tokimon.workflow.document_audit import cli, matrix


def subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("parser has no subcommands")


class DocumentAuditCliContractTests(unittest.TestCase):
    def test_record_findings_rejects_missing_severity_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            ledger_path.write_text(json.dumps({
                "schema_version": 3, "workflow_id": "document-audit",
                "project_id": audit_project.current().project_id,
                "entries": {"historical": {"status": "fixed"}},
            }))
            findings_path = Path(tmp) / "findings.json"
            findings_path.write_text("[]")
            original_inputs = {path: path.read_bytes() for path in (ledger_path, findings_path)}
            run_root = Path(tmp) / "runs"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main([
                    "record-findings", "--accepted-findings", str(findings_path),
                    "--ledger", str(ledger_path), "--scope-id", "shape-fixture",
                    "--run-id", "shape-fixture", "--run-root", str(run_root), "--json",
                ])
            result = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 2, result)
            self.assertFalse(result["ok"])
            self.assertIn("severity", result["error"])
            self.assertEqual({path: path.read_bytes() for path in original_inputs}, original_inputs)
            self.assertFalse(run_root.exists())
            self.assertEqual(set(Path(tmp).iterdir()), set(original_inputs))

    def test_ledger_summary_rejects_malformed_shapes_without_mutating_input(self) -> None:
        valid_entry = {"status": "fixed", "severity": "optional"}
        template = {"schema_version": 3, "workflow_id": "document-audit",
                    "project_id": audit_project.current().project_id, "entries": {}}
        cases = [(f"root-{index}", value) for index, value in enumerate(([], None, 1, "ledger", True))]
        cases += [(f"entries-{index}", dict(template, entries=value))
                  for index, value in enumerate(([], None, 1, "entries", True))]
        cases += [(f"entry-{index}", dict(template, entries={"bad": value}))
                  for index, value in enumerate((None, [], 1, "entry", {}))]
        cases += [(f"schema-{index}", dict(template, schema_version=value))
                  for index, value in enumerate(([], {}))]
        for field in ("status", "severity"):
            entry = {key: value for key, value in valid_entry.items() if key != field}
            cases.append((f"missing-{field}", dict(template, entries={"bad": entry})))
            for index, value in enumerate((None, [], {}, 1, "unsupported")):
                entry = dict(valid_entry, **{field: value})
                cases.append((f"invalid-{field}-{index}", dict(template, entries={"bad": entry})))
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            for case, payload in cases:
                with self.subTest(case=case):
                    ledger_path.write_text(json.dumps(payload))
                    original = ledger_path.read_bytes()
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        exit_code = cli.main(["ledger-summary", "--ledger", str(ledger_path), "--json"])
                    result = json.loads(stdout.getvalue())
                    self.assertEqual(exit_code, 2, result)
                    self.assertFalse(result["ok"])
                    self.assertIn("ledger", result["error"])
                    self.assertEqual(ledger_path.read_bytes(), original)
                    self.assertEqual(list(Path(tmp).iterdir()), [ledger_path])

    def test_ledger_summary_preserves_legacy_records_and_missing_entries_default(self) -> None:
        entries = {"historical": {"status": "fixed", "severity": "optional",
                                  "migration_sources": ["original.json"]}}
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            for version in (1, 2, 3):
                for include_entries in (False, True):
                    with self.subTest(version=version, include_entries=include_entries):
                        payload = {"schema_version": version, "workflow_id": "document-audit"}
                        if version == 3:
                            payload["project_id"] = audit_project.current().project_id
                        if include_entries:
                            payload["entries"] = entries
                        ledger_path.write_text(json.dumps(payload))
                        original = ledger_path.read_bytes()
                        stdout = io.StringIO()
                        with contextlib.redirect_stdout(stdout):
                            exit_code = cli.main(["ledger-summary", "--ledger", str(ledger_path), "--json"])
                        result = json.loads(stdout.getvalue())
                        self.assertEqual(exit_code, 0, result)
                        self.assertTrue(result["ok"])
                        self.assertEqual(result["stats"]["total"], int(include_entries))
                        self.assertEqual(result["entries"], list(entries.values()) if include_entries else [])
                        self.assertEqual(result["stats"]["by_status"]["fixed"], int(include_entries))
                        self.assertEqual(result["stats"]["by_severity"]["optional"], int(include_entries))
                        self.assertEqual(ledger_path.read_bytes(), original)

            ledger_path.write_text(json.dumps({
                "schema_version": 1, "workflow_id": "document-audit",
                "entries": {"historical": {"status": "rejected"}},
            }))
            findings_path = Path(tmp) / "findings.json"
            findings_path.write_text("[]")
            original_inputs = {path: path.read_bytes() for path in (ledger_path, findings_path)}
            run_root = Path(tmp) / "runs"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main([
                    "record-findings", "--accepted-findings", str(findings_path),
                    "--ledger", str(ledger_path), "--scope-id", "legacy-fixture",
                    "--run-id", "legacy-fixture", "--run-root", str(run_root), "--json",
                ])
            result = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 2, result)
            self.assertIn("legacy unbound", result["error"])
            self.assertEqual({path: path.read_bytes() for path in original_inputs}, original_inputs)
            self.assertFalse(run_root.exists())

    def test_unreadable_prior_locks_require_full_fresh_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rule = root / "rules.md"
            rule.write_text("- **DOC-TEST-001:** Synthetic lock recovery rule.\n")
            documents = [root / "first.md", root / "second.md"]
            for document in documents:
                document.write_text("# Synthetic lock recovery document\n")
            payload = matrix.build_matrix_payload(
                scope_id="lock-recovery", run_id="lock-recovery-run",
                document_paths=[str(document) for document in documents],
                unit_definitions=[
                    {"audit_unit_id": "first", "document_paths": [str(documents[0])]},
                    {"audit_unit_id": "second", "document_paths": [str(documents[1])],
                     "bound_input_paths": [str(documents[0])]},
                ],
                rule_sources=[rule],
            )
            matrix_path = root / "matrix.json"
            matrix_path.write_text(json.dumps(payload))
            prior_locks = root / "prior-locks.json"
            ledger_path = root / "independent-ledger.json"
            original_inputs = {path: path.read_bytes() for path in (rule, matrix_path, *documents)}
            for case, content in (("invalid-json", b"{"), ("invalid-utf8", b"\xff")):
                with self.subTest(case=case):
                    prior_locks.write_bytes(content)
                    output = root / case
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        exit_code = cli.main([
                            "recheck-plan", "--matrix", str(matrix_path),
                            "--prior-clean-unit-locks", str(prior_locks),
                            "--output-dir", str(output), "--ledger", str(ledger_path), "--json",
                        ])
                    result = json.loads(stdout.getvalue())
                    self.assertEqual(exit_code, 0, result)
                    self.assertTrue(result["ok"])
                    plan_path = Path(result["artifact_paths"]["recheck_plan_json"])
                    self.assertEqual(plan_path, output / "recheck-plan.json")
                    plan = json.loads(plan_path.read_text())
                    self.assertFalse(plan["audit_authority"])
                    self.assertFalse(plan["lock_validation_available"])
                    self.assertTrue(plan["complete_fresh_reports_required"])
                    self.assertTrue(plan["lock_validation_error"])
                    self.assertEqual(plan["active_unit_count"], 2)
                    for field in ("locked_unit_count", "settled_unit_count", "waiting_unit_count",
                                  "locked_row_count", "settled_row_count", "waiting_row_count"):
                        self.assertEqual(plan[field], 0)
                    self.assertEqual(plan["fresh_required_row_count"], 2)
                    self.assertEqual(len(plan["rows"]), 2)
                    self.assertEqual({row["action"] for row in plan["rows"]}, {"fresh_report_required"})
                    self.assertEqual(prior_locks.read_bytes(), content)
                    self.assertEqual({path: path.read_bytes() for path in original_inputs}, original_inputs)
                    self.assertFalse(ledger_path.exists())

    def test_generated_outputs_reject_escaping_symlinks_and_loops(self) -> None:
        def invoke(args: list[str]) -> tuple[int, dict]:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main([*args, "--json"])
            return exit_code, json.loads(stdout.getvalue())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rule = root / "rules.md"
            rule.write_text("- **DOC-TEST-001:** Synthetic output boundary rule.\n")
            document = root / "doc.md"
            document.write_text("# Output fixture\n")
            payload = matrix.build_matrix_payload(
                scope_id="output-boundary", run_id="output-boundary-run",
                document_paths=[str(document)], rule_sources=[rule],
            )
            matrix_path = root / "matrix.json"
            matrix_path.write_text(json.dumps(payload))
            report_path = root / "reports" / "row.json"
            report_path.parent.mkdir()
            report = reviewed_report({
                "schema_version": 3, "workflow_id": "document-audit",
                "run_id": payload["run_id"], "scope_id": payload["scope_id"],
                "audit_snapshot_id": payload["audit_snapshot_id"], "file_path": payload["document_paths"][0],
                "rule_id": "DOC-TEST-001", "status": "pass", "findings": [],
                "summary": "Synthetic output boundary fixture.", "reason": "Exercise public output rejection.",
            }, root / "reviews")
            report_path.write_text(json.dumps(report))
            initial = root / "initial"
            ledger_path = root / "independent-ledger.json"
            exit_code, result = invoke([
                "clean-unit-locks", "--matrix", str(matrix_path), "--reports-dir", str(report_path.parent),
                "--output-dir", str(initial), "--ledger", str(ledger_path),
            ])
            self.assertEqual(exit_code, 0, result)
            prior_locks = initial / "clean-unit-locks.json"
            original_inputs = {path: path.read_bytes() for path in (document, matrix_path, report_path, prior_locks)}
            for command, filename in (
                ("clean-unit-locks", "clean-unit-locks.json"), ("recheck-plan", "recheck-plan.json"),
                ("compile-reports", "evaluations-from-reports.json"), ("compile-reports", "reports-summary.json"),
            ):
                for link_kind in ("escape", "loop"):
                    with self.subTest(command=command, filename=filename, link_kind=link_kind):
                        output = root / f"{command}-{filename}-{link_kind}"
                        output.mkdir()
                        victim = root / f"victim-{command}-{filename}-{link_kind}"
                        victim.write_bytes(b"preserve this file")
                        target = output / filename
                        target.symlink_to(victim if link_kind == "escape" else target)
                        args = [command, "--matrix", str(matrix_path), "--output-dir", str(output)]
                        args += (["--prior-clean-unit-locks", str(prior_locks)] if command == "recheck-plan"
                                 else ["--reports-dir", str(report_path.parent)])
                        if command != "compile-reports":
                            args += ["--ledger", str(ledger_path)]
                        exit_code, result = invoke(args)
                        self.assertEqual(exit_code, 2, result)
                        self.assertFalse(result["ok"])
                        self.assertIn("configured", result["error"])
                        self.assertEqual(victim.read_bytes(), b"preserve this file")
                        self.assertTrue(target.is_symlink())
                        self.assertEqual(list(output.iterdir()), [target])
                        self.assertEqual({path: path.read_bytes() for path in original_inputs}, original_inputs)
                        self.assertFalse(ledger_path.exists())

    def test_cli_subcommands(self) -> None:
        parser = cli.build_parser()

        self.assertEqual(
            subcommand_names(parser),
            {
                "contract",
                "rules",
                "matrix",
                "clean-unit-locks",
                "recheck-plan",
                "verify-reports",
                "compile-reports",
                "ledger-summary",
                "record-findings",
            },
        )
        record_args = parser.parse_args(
            [
                "record-findings",
                "--accepted-findings",
                "/tmp/findings.json",
                "--scope-id",
                "docs-example",
                "--run-id",
                "run-1",
                "--ledger",
                "/tmp/current.json",
                "--run-root",
                "/tmp/runs",
                "--source",
                "docs/guidelines/document-guidelines.md",
                "--json",
            ]
        )
        self.assertEqual(record_args.command, "record-findings")
        self.assertEqual(record_args.scope_id, "docs-example")
        self.assertEqual(record_args.source, ["docs/guidelines/document-guidelines.md"])
        self.assertTrue(record_args.json)
        matrix_args = parser.parse_args(
            [
                "matrix",
                "--scope-id",
                "docs-example",
                "--run-id",
                "run-1",
                "--document-glob",
                "docs/c4/level-1-context/*.md",
                "--units-from",
                "/tmp/audit-units.json",
                "--expect-rule-count",
                "50",
                "--expect-document-count",
                "5",
                "--expect-matrix-count",
                "250",
                "--require-complete",
                "--json",
            ]
        )
        self.assertEqual(matrix_args.command, "matrix")
        self.assertEqual(matrix_args.scope_id, "docs-example")
        self.assertEqual(matrix_args.expect_rule_count, 50)
        self.assertEqual(matrix_args.expect_document_count, 5)
        self.assertEqual(matrix_args.expect_matrix_count, 250)
        self.assertEqual(matrix_args.units_from, "/tmp/audit-units.json")
        self.assertTrue(matrix_args.require_complete)
        locks_args = parser.parse_args(
            [
                "clean-unit-locks",
                "--matrix",
                "/tmp/checklist-matrix.json",
                "--reports-dir",
                "/tmp/reports",
                "--output-dir",
                "/tmp/locks",
                "--ledger",
                "/tmp/current.json",
                "--json",
            ]
        )
        self.assertEqual(locks_args.command, "clean-unit-locks")
        self.assertEqual(locks_args.output_dir, "/tmp/locks")
        recheck_args = parser.parse_args(
            [
                "recheck-plan",
                "--matrix",
                "/tmp/checklist-matrix.json",
                "--prior-clean-unit-locks",
                "/tmp/clean-unit-locks.json",
                "--output-dir",
                "/tmp/recheck",
                "--activate-unit",
                "docs-one",
                "--json",
            ]
        )
        self.assertEqual(recheck_args.command, "recheck-plan")
        self.assertEqual(recheck_args.activate_unit, ["docs-one"])
        verify_args = parser.parse_args(
            [
                "verify-reports",
                "--matrix",
                "/tmp/checklist-matrix.json",
                "--reports-dir",
                "/tmp/reports",
                "--recheck-plan",
                "/tmp/recheck-plan.json",
                "--require-complete",
                "--json",
            ]
        )
        self.assertEqual(verify_args.command, "verify-reports")
        self.assertEqual(verify_args.recheck_plan, "/tmp/recheck-plan.json")
        self.assertTrue(verify_args.require_complete)
        compile_args = parser.parse_args(
            [
                "compile-reports",
                "--matrix",
                "/tmp/checklist-matrix.json",
                "--reports-dir",
                "/tmp/reports",
                "--output-dir",
                "/tmp/compiled",
                "--json",
            ]
        )
        self.assertEqual(compile_args.command, "compile-reports")
        self.assertEqual(compile_args.output_dir, "/tmp/compiled")
        self.assertEqual(cli.required_tool_capabilities(), ())

    def test_rules_and_ledger_summary_smoke(self) -> None:
        rules_result = subprocess.run(
            [sys.executable, "-m", "tokimon.workflow.document_audit", "rules", "--json"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        rules_payload = json.loads(rules_result.stdout)
        self.assertTrue(rules_payload["ok"])
        self.assertGreater(rules_payload["rule_count"], 0)

        with tempfile.TemporaryDirectory() as tmp:
            summary_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.document_audit",
                    "ledger-summary",
                    "--ledger",
                    str(Path(tmp) / "missing-current.json"),
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
        summary_payload = json.loads(summary_result.stdout)
        self.assertTrue(summary_payload["ok"])
        self.assertEqual(summary_payload["stats"]["total"], 0)

    def test_record_findings_module_smoke_with_temp_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            findings_path = tmp_path / "findings.json"
            findings_path.write_text(
                json.dumps(
                    [
                        {
                            "rule_id": "DOC-OWN-001",
                            "file_path": "docs/example.md",
                            "summary": "Duplicated durable ownership statement.",
                            "severity": "should_fix",
                            "status": "open",
                            "evidence": [{"line": 12, "text": "duplicate"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.document_audit",
                    "record-findings",
                    "--accepted-findings",
                    str(findings_path),
                    "--scope-id",
                    "docs-example",
                    "--run-id",
                    "run-1",
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--run-root",
                    str(tmp_path / "runs"),
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            payload = json.loads(result.stdout)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["finding_count"], 1)
        self.assertEqual(payload["stats"]["open_should_fix"], 1)

    def test_record_findings_validates_against_selected_rule_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            findings_path = tmp_path / "findings.json"
            findings_path.write_text(
                json.dumps(
                    [
                        {
                            "rule_id": "C4-L1-DEP-001",
                            "file_path": "docs/example.md",
                            "summary": "C4 rule should not validate for a non-C4 audit.",
                            "severity": "should_fix",
                            "status": "open",
                            "evidence": [{"line": 12, "text": "dependency"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.document_audit",
                    "record-findings",
                    "--accepted-findings",
                    str(findings_path),
                    "--scope-id",
                    "docs-example",
                    "--run-id",
                    "run-1",
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--run-root",
                    str(tmp_path / "runs"),
                    "--source",
                    "docs/guidelines/document-guidelines.md",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 2)
        self.assertFalse(payload["ok"])
        self.assertIn("unknown rule_id: C4-L1-DEP-001", payload["error"])

    def test_record_findings_rejects_support_artifact_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            findings_path = tmp_path / "findings.json"
            findings_path.write_text(
                json.dumps({"audit_authority": False, "findings": []}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.document_audit",
                    "record-findings",
                    "--accepted-findings",
                    str(findings_path),
                    "--scope-id",
                    "docs-example",
                    "--run-id",
                    "run-1",
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--run-root",
                    str(tmp_path / "runs"),
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 2)
        self.assertFalse(payload["ok"])
        self.assertIn("accepted findings input must not be a compile-reports support artifact", payload["error"])

    def test_default_output_is_human_text(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tokimon.workflow.document_audit", "contract"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("workflow_id: document-audit", result.stdout)
        self.assertIn("ensure_ascii: false", result.stdout)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(result.stdout)

    def test_missing_subcommand_fails_fast(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tokimon.workflow.document_audit"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)
        self.assertIn("the following arguments are required: command", result.stderr)


if __name__ == "__main__":
    unittest.main()
