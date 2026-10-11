from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.code_audit import reports


from tests.workflow.code_audit.helpers import reviewed_report, write_fixture


def matrix_payload() -> dict[str, object]:
    return {
        "schema_version": 3,
        "workflow_id": "code-audit",
        "run_id": "run-1",
        "scope_id": "scope-1",
        "rows": [
            {
                "run_id": "run-1",
                "scope_id": "scope-1",
                "audit_subject_id": "tokimon.workflow.code-audit",
                "rule_id": "CODE-TEST-001",
                "status": "unchecked",
                "rule_hash": "hash-1",
            },
            {
                "run_id": "run-1",
                "scope_id": "scope-1",
                "audit_subject_id": "tokimon.workflow.code-audit",
                "rule_id": "CODE-TEST-002",
                "status": "unchecked",
                "rule_hash": "hash-2",
            },
            {
                "run_id": "run-1",
                "scope_id": "scope-1",
                "audit_subject_id": "tokimon.workflow.document-audit",
                "rule_id": "CODE-TEST-001",
                "status": "unchecked",
                "rule_hash": "hash-3",
            },
        ]
    }


def base_report(subject_id: str, rule_id: str, status: str = "pass") -> dict[str, object]:
    return {
        "schema_version": 3,
        "workflow_id": "code-audit",
        "run_id": "run-1",
        "scope_id": "scope-1",
        "audit_subject_id": subject_id,
        "rule_id": rule_id,
        "status": status,
        "summary": "The row has been audited.",
        "reason": "The checked surfaces satisfy the rule.",
        "implementation_paths_examined": ["tokimon/workflow/code_audit"],
        "test_paths_examined": ["tests/workflow/code_audit"],
        "owning_specs_examined": ["docs/c4/level-3-components/workflow/code-audit-cli.md"],
        "verification": [".venv/bin/python -m unittest discover -s tests/workflow/code_audit -t ."],
    }


class CodeAuditReportTests(unittest.TestCase):
    def test_report_provenance_must_match_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            payload = matrix_payload()
            payload["rows"] = [payload["rows"][0]]
            write_fixture(matrix_path, payload)
            report = base_report("tokimon.workflow.code-audit", "CODE-TEST-001")
            report["run_id"] = "different-run"
            write_fixture(reports_dir / "wrong-run.json", report)

            verification = reports.verify_report_coverage(
                matrix_path=matrix_path,
                reports_dir=reports_dir,
            )

        self.assertFalse(verification["complete"])
        self.assertEqual(verification["invalid_count"], 1)
        self.assertEqual(verification["covered_report_count"], 0)
        self.assertIn("provenance mismatch", verification["invalid_reports"][0]["error"])

    def test_verify_reports_reports_all_gap_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            write_fixture(matrix_path, matrix_payload())
            write_fixture(reports_dir / "one.json", base_report("tokimon.workflow.code-audit", "CODE-TEST-001"))
            write_fixture(reports_dir / "duplicate.json", base_report("tokimon.workflow.code-audit", "CODE-TEST-001"))
            write_fixture(reports_dir / "extra.json", base_report("tokimon.workflow.missing", "CODE-TEST-001"))
            invalid = base_report("tokimon.workflow.code-audit", "CODE-TEST-002")
            invalid.pop("reason")
            write_fixture(reports_dir / "invalid.json", invalid)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.code_audit",
                    "verify-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--require-complete",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 2)
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["complete"])
        self.assertEqual(payload["expected_report_count"], 3)
        self.assertEqual(payload["covered_report_count"], 1)
        self.assertEqual(payload["missing_count"], 2)
        self.assertEqual(payload["invalid_count"], 1)
        self.assertEqual(payload["duplicate_count"], 1)
        self.assertEqual(payload["extra_count"], 1)

    def test_compile_reports_writes_only_support_evaluations_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            output_dir = tmp_path / "compiled"
            write_fixture(
                matrix_path,
                {
                    "schema_version": 3,
                    "workflow_id": "code-audit",
                    "run_id": "run-1",
                    "scope_id": "scope-1",
                    "rows": [
                        {
                            "run_id": "run-1",
                            "scope_id": "scope-1",
                            "audit_subject_id": "tokimon.workflow.code-audit",
                            "rule_id": "CODE-TEST-001",
                            "status": "unchecked",
                            "rule_hash": "hash-1",
                        },
                        {
                            "run_id": "run-1",
                            "scope_id": "scope-1",
                            "audit_subject_id": "tokimon.workflow.code-audit",
                            "rule_id": "CODE-TEST-002",
                            "status": "unchecked",
                            "rule_hash": "hash-2",
                        },
                    ]
                },
            )
            write_fixture(reports_dir / "pass.json", base_report("tokimon.workflow.code-audit", "CODE-TEST-001"))
            failing = base_report("tokimon.workflow.code-audit", "CODE-TEST-002", status="fail")
            failing["summary"] = "The workflow does not verify report coverage."
            failing["reason"] = "No verifier checks one report per matrix row."
            failing["evidence"] = [
                {
                    "path": "tokimon/workflow/code_audit/cli.py",
                    "line_start": 1,
                    "excerpt": "code audit CLI",
                }
            ]
            write_fixture(reports_dir / "fail.json", failing)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.code_audit",
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(output_dir),
                    "--require-complete",
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            payload = json.loads(result.stdout)
            evaluations = json.loads((output_dir / "evaluations-from-reports.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "reports-summary.json").read_text(encoding="utf-8"))

        self.assertTrue(payload["ok"])
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["audit_authority"])
        self.assertEqual(summary["evaluation_count"], 2)
        self.assertNotIn("candidate_issue_count", summary)
        self.assertEqual(evaluations["evaluations"][1]["rule_hash"], "hash-2")
        self.assertEqual(evaluations["run_id"], "run-1")
        self.assertEqual(evaluations["scope_id"], "scope-1")
        self.assertEqual(evaluations["evaluations"][0]["run_id"], "run-1")
        self.assertEqual(evaluations["evaluations"][0]["scope_id"], "scope-1")
        self.assertFalse(evaluations["audit_authority"])
        self.assertNotIn("candidate_issues_json", payload["artifact_paths"])
        self.assertFalse((output_dir / "candidate-issues-from-reports.json").exists())
        self.assertFalse((output_dir / "findings-from-reports.json").exists())

    def test_report_requires_examined_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            report = base_report("tokimon.workflow.code-audit", "CODE-TEST-001")
            report.pop("implementation_paths_examined")

            with self.assertRaisesRegex(ValueError, "implementation_paths_examined must be a list"):
                reports.validate_report(report, report_path=report_path)

    def test_report_requires_schema_version_and_workflow_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            missing_schema = base_report("tokimon.workflow.code-audit", "CODE-TEST-001")
            missing_schema.pop("schema_version")

            with self.assertRaisesRegex(ValueError, "missing required field: schema_version"):
                reports.validate_report(missing_schema, report_path=report_path)

            missing_workflow = base_report("tokimon.workflow.code-audit", "CODE-TEST-001")
            missing_workflow.pop("workflow_id")

            with self.assertRaisesRegex(ValueError, "missing required field: workflow_id"):
                reports.validate_report(missing_workflow, report_path=report_path)

    def test_fail_report_requires_exact_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            report = base_report("tokimon.workflow.code-audit", "CODE-TEST-001", status="fail")
            report = reviewed_report(report, report_path)

            with self.assertRaisesRegex(ValueError, "missing exact evidence"):
                reports.validate_report(report, report_path=report_path)


if __name__ == "__main__":
    unittest.main()
