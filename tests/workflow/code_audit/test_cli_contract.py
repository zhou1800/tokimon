from __future__ import annotations

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

from tests.workflow.code_audit.helpers import reviewed_report
from tokimon.workflow import audit_project
from tokimon.workflow.code_audit import cli, matrix

class CodeAuditCliContractTests(unittest.TestCase):
    def test_ledger_summary_rejects_malformed_shapes_without_mutating_input(self) -> None:
        valid_entry = {"status": "fixed", "rule_id": "CODE-TEST-001", "audit_subject_id": "fixture"}
        template = {"schema_version": 3, "workflow_id": "code-audit",
                    "project_id": audit_project.current().project_id, "entries": {}}
        cases = [(f"root-{index}", value) for index, value in enumerate(([], None, 1, "ledger", True))]
        cases += [(f"entries-{index}", dict(template, entries=value))
                  for index, value in enumerate(([], None, 1, "entries", True))]
        cases += [(f"entry-{index}", dict(template, entries={"bad": value}))
                  for index, value in enumerate((None, [], 1, "entry", {}))]
        cases += [(f"schema-{index}", dict(template, schema_version=value))
                  for index, value in enumerate(([], {}))]
        cases += [(f"evidence-{index}", dict(template, entries={"bad": dict(valid_entry, evidence=value)}))
                  for index, value in enumerate((None, 1, True))]
        for field in ("status", "rule_id", "audit_subject_id"):
            entry = {key: value for key, value in valid_entry.items() if key != field}
            cases.append((f"missing-{field}", dict(template, entries={"bad": entry})))
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

    def test_ledger_summary_preserves_legacy_records_and_unknown_status(self) -> None:
        entries = {"historical": {"status": "historical-status", "rule_id": "CODE-TEST-001",
                                  "audit_subject_id": "fixture", "migration_sources": ["original.json"]}}
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            for version in (1, 2, 3):
                for include_entries in (False, True):
                    with self.subTest(version=version, include_entries=include_entries):
                        payload = {"schema_version": version, "workflow_id": "code-audit"}
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
                        if include_entries:
                            self.assertEqual(result["stats"]["by_status"]["historical-status"], 1)
                            self.assertEqual(result["stats"]["open_total"], 0)
                            self.assertEqual(result["stats"]["resolved_total"], 0)
                        self.assertEqual(ledger_path.read_bytes(), original)

            for evidence in ([], {"historical": "value"}, "historical"):
                with self.subTest(evidence=evidence):
                    entry = dict(entries["historical"], evidence=evidence)
                    payload = {"schema_version": 2, "workflow_id": "code-audit",
                               "entries": {"historical": entry}}
                    ledger_path.write_text(json.dumps(payload))
                    original = ledger_path.read_bytes()
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        exit_code = cli.main(["ledger-summary", "--ledger", str(ledger_path), "--json"])
                    result = json.loads(stdout.getvalue())
                    self.assertEqual(exit_code, 0, result)
                    self.assertEqual(result["entries"], [entry])
                    self.assertTrue(result["health"]["healthy"])
                    self.assertEqual(ledger_path.read_bytes(), original)

    def test_unreadable_prior_locks_require_full_fresh_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rule = root / "rules.md"
            rule.write_text("- **CODE-TEST-001:** Synthetic lock recovery rule.\n")
            payload = matrix.build_matrix_payload(
                scope_id="lock-recovery", run_id="lock-recovery-run",
                audit_subjects=[
                    {"audit_subject_id": "first", "title": "First fixture"},
                    {"audit_subject_id": "second", "title": "Second fixture"},
                ],
                rule_sources=[rule],
            )
            matrix_path = root / "matrix.json"
            matrix_path.write_text(json.dumps(payload))
            prior_locks = root / "prior-locks.json"
            original_inputs = {path: path.read_bytes() for path in (rule, matrix_path)}
            for case, content in (("invalid-json", b"{"), ("invalid-utf8", b"\xff")):
                with self.subTest(case=case):
                    prior_locks.write_bytes(content)
                    output = root / case
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        exit_code = cli.main([
                            "recheck-plan", "--matrix", str(matrix_path),
                            "--prior-pass-locks", str(prior_locks),
                            "--output-dir", str(output), "--json",
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
                    self.assertEqual(plan["locked_row_count"], 0)
                    self.assertEqual(plan["fresh_required_row_count"], 2)
                    self.assertEqual(len(plan["rows"]), 2)
                    self.assertEqual({row["action"] for row in plan["rows"]}, {"fresh_report_required"})
                    self.assertEqual(prior_locks.read_bytes(), content)
                    self.assertEqual({path: path.read_bytes() for path in original_inputs}, original_inputs)

    def test_generated_outputs_reject_escaping_symlinks_and_loops(self) -> None:
        def invoke(args: list[str]) -> tuple[int, dict]:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main([*args, "--json"])
            return exit_code, json.loads(stdout.getvalue())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rule = root / "rules.md"
            rule.write_text("- **CODE-TEST-001:** Synthetic output boundary rule.\n")
            payload = matrix.build_matrix_payload(
                scope_id="output-boundary", run_id="output-boundary-run",
                audit_subjects=[{"audit_subject_id": "fixture", "title": "Output fixture"}],
                rule_sources=[rule],
            )
            matrix_path = root / "matrix.json"
            matrix_path.write_text(json.dumps(payload))
            report_path = root / "reports" / "row.json"
            report_path.parent.mkdir()
            report = reviewed_report({
                "schema_version": 3, "workflow_id": "code-audit",
                "run_id": payload["run_id"], "scope_id": payload["scope_id"],
                "audit_snapshot_id": payload["audit_snapshot_id"],
                "audit_subject_id": "fixture", "rule_id": "CODE-TEST-001", "status": "pass",
                "summary": "Synthetic output boundary fixture.", "reason": "Exercise public output rejection.",
                "implementation_paths_examined": [], "test_paths_examined": [], "owning_specs_examined": [],
                "verification": ["Synthetic fixture; not an implementation audit."],
            }, report_path, source_dir=root / "reviews")
            report_path.write_text(json.dumps(report))
            initial = root / "initial"
            exit_code, result = invoke([
                "pass-locks", "--matrix", str(matrix_path), "--reports-dir", str(report_path.parent),
                "--output-dir", str(initial),
            ])
            self.assertEqual(exit_code, 0, result)
            prior_locks = initial / "pass-locks.json"
            original_inputs = {path: path.read_bytes() for path in (matrix_path, report_path, prior_locks)}
            for command, filename in (
                ("pass-locks", "pass-locks.json"), ("recheck-plan", "recheck-plan.json"),
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
                        args += (["--prior-pass-locks", str(prior_locks)] if command == "recheck-plan"
                                 else ["--reports-dir", str(report_path.parent)])
                        exit_code, result = invoke(args)
                        self.assertEqual(exit_code, 2, result)
                        self.assertFalse(result["ok"])
                        self.assertIn("configured", result["error"])
                        self.assertEqual(victim.read_bytes(), b"preserve this file")
                        self.assertTrue(target.is_symlink())
                        self.assertEqual(list(output.iterdir()), [target])
                        self.assertEqual({path: path.read_bytes() for path in original_inputs}, original_inputs)

    def test_help_lists_public_subcommands(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tokimon.workflow.code_audit", "--help"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

        for command in (
            "contract",
            "rules",
            "subjects",
            "matrix",
            "pass-locks",
            "recheck-plan",
            "verify-reports",
            "compile-reports",
            "record-findings",
            "ledger-summary",
        ):
            self.assertIn(command, result.stdout)

    def test_contract_command_reports_readiness(self) -> None:
        contract_result = subprocess.run(
            [sys.executable, "-m", "tokimon.workflow.code_audit", "contract", "--json"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        contract_payload = json.loads(contract_result.stdout)
        self.assertTrue(contract_payload["ok"])
        self.assertEqual(contract_payload["contract"]["workflow_id"], "code-audit")
        self.assertEqual(
            contract_payload["contract"]["workflow_readiness"]["modes"][0]["required_tool_capabilities"],
            [],
        )
        self.assertNotIn("candidate_issues_name", contract_payload["contract"])
        self.assertNotIn("compiled_findings_name", contract_payload["contract"])

    def test_rules_command_smoke(self) -> None:
        rules_result = subprocess.run(
            [sys.executable, "-m", "tokimon.workflow.code_audit", "rules", "--json"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        rules_payload = json.loads(rules_result.stdout)
        self.assertTrue(rules_payload["ok"])
        self.assertGreater(rules_payload["rule_count"], 0)

    def test_ledger_summary_command_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.code_audit",
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

    def test_subjects_and_record_findings_smoke_with_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subjects_path = tmp_path / "subjects.json"
            evaluations_path = tmp_path / "evaluations.json"
            subjects_path.write_text(
                json.dumps(
                    {
                        "subjects": [
                            {
                                "audit_subject_id": "tokimon.workflow.document_audit",
                                "title": "Document Audit workflow",
                                "implementation_paths": ["tokimon/workflow/document_audit"],
                                "test_paths": ["tests/workflow/document_audit"],
                                "owning_specs": [
                                    "docs/c4/level-3-components/workflow/document-audit-cli.md"
                                ],
                                "verification_commands": ["unit test command"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            evaluations_path.write_text(
                json.dumps(
                    {
                        "evaluations": [
                            {
                                "run_id": "run-1",
                                "scope_id": "tokimon-workflows",
                                "rule_id": "CODE-CONTRACT-001",
                                "audit_subject_id": "tokimon.workflow.document_audit",
                                "status": "fail",
                                "summary": "Missing CLI contract coverage.",
                                "evidence": [
                                    {
                                        "path": "tokimon/workflow/document_audit/cli.py",
                                        "line_start": 10,
                                        "excerpt": "contract",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            subjects_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.code_audit",
                    "subjects",
                    "--subjects-from",
                    str(subjects_path),
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            record_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.code_audit",
                    "record-findings",
                    "--accepted-evaluations",
                    str(evaluations_path),
                    "--scope-id",
                    "tokimon-workflows",
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

        self.assertTrue(json.loads(subjects_result.stdout)["ok"])
        payload = json.loads(record_result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["finding_count"], 1)

    def test_record_findings_rejects_support_artifact_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            evaluations_path = tmp_path / "evaluations.json"
            evaluations_path.write_text(
                json.dumps({"audit_authority": False, "evaluations": []}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.code_audit",
                    "record-findings",
                    "--accepted-evaluations",
                    str(evaluations_path),
                    "--scope-id",
                    "tokimon-workflows",
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
        self.assertIn("accepted evaluations input must not be a compile-reports support artifact", payload["error"])

    def test_default_output_is_human_text(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tokimon.workflow.code_audit", "contract"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("workflow_id: code-audit", result.stdout)
        self.assertIn("ensure_ascii: false", result.stdout)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(result.stdout)

    def test_missing_subcommand_fails_fast(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tokimon.workflow.code_audit"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)
        self.assertIn("the following arguments are required: command", result.stderr)


if __name__ == "__main__":
    unittest.main()
