from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.document_audit import cli, reports


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def matrix_payload() -> dict[str, object]:
    return {
        "rows": [
            {
                "file_path": "docs/doc-one.md",
                "rule_id": "DOC-TEST-001",
                "status": "unchecked",
                "rule_hash": "hash-1",
            },
            {
                "file_path": "docs/doc-two.md",
                "rule_id": "DOC-TEST-001",
                "status": "unchecked",
                "rule_hash": "hash-2",
            },
            {
                "file_path": "docs/doc-two.md",
                "rule_id": "DOC-TEST-002",
                "status": "unchecked",
                "rule_hash": "hash-3",
            },
        ]
    }


class DocumentAuditReportTests(unittest.TestCase):
    def test_compile_cli_reads_one_snapshot_and_rejects_gaps_before_writing(self) -> None:
        for complete in (False, True):
            with self.subTest(complete=complete), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                matrix_path = root / "matrix.json"
                reports_dir = root / "reports"
                reports_dir.mkdir()
                output_dir = root / "compiled"
                matrix = matrix_payload()
                write_json(matrix_path, matrix)
                if complete:
                    for index, row in enumerate(matrix["rows"]):
                        write_json(reports_dir / f"{index}.json", {
                            **row, "status": "pass", "summary": "Rule is satisfied.",
                            "reason": "The document states its responsibility.",
                        })
                output = io.StringIO()
                with (
                    patch.object(reports.report_coverage, "read_reports", wraps=reports.report_coverage.read_reports) as read,
                    contextlib.redirect_stdout(output),
                ):
                    result = cli.main([
                        "compile-reports", "--matrix", str(matrix_path), "--reports-dir", str(reports_dir),
                        "--output-dir", str(output_dir), "--require-complete", "--json",
                    ])
                payload = json.loads(output.getvalue())
                self.assertEqual(result, 0 if complete else 2)
                self.assertEqual(payload["ok"], complete)
                self.assertEqual(read.call_count, 1)
                if complete:
                    self.assertEqual(payload["summary"]["evaluation_count"], 3)
                    self.assertTrue((output_dir / "evaluations-from-reports.json").is_file())
                else:
                    self.assertEqual(payload["missing_count"], 3)
                    self.assertNotIn("compiled", payload)
                    self.assertFalse(output_dir.exists())

    def test_verify_reports_reports_all_gap_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            write_json(matrix_path, matrix_payload())
            write_json(
                reports_dir / "one.json",
                {
                    "file_path": "docs/doc-one.md",
                    "rule_id": "DOC-TEST-001",
                    "status": "pass",
                    "summary": "Rule is satisfied.",
                    "reason": "The document states its responsibility early.",
                },
            )
            write_json(
                reports_dir / "duplicate.json",
                {
                    "file_path": "docs/doc-one.md",
                    "rule_id": "DOC-TEST-001",
                    "status": "pass",
                    "summary": "Duplicate report.",
                    "reason": "This row was already reported.",
                },
            )
            write_json(
                reports_dir / "extra.json",
                {
                    "file_path": "docs/not-in-matrix.md",
                    "rule_id": "DOC-TEST-001",
                    "status": "pass",
                    "summary": "Extra report.",
                    "reason": "The document is outside the checklist.",
                },
            )
            write_json(
                reports_dir / "invalid.json",
                {
                    "file_path": "docs/doc-two.md",
                    "rule_id": "DOC-TEST-001",
                    "status": "pass",
                    "summary": "Missing reason.",
                },
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.document_audit",
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
            write_json(
                matrix_path,
                {
                    "rows": [
                        {
                            "file_path": "docs/doc-one.md",
                            "rule_id": "DOC-TEST-001",
                            "status": "unchecked",
                            "rule_hash": "hash-1",
                        },
                        {
                            "file_path": "docs/doc-two.md",
                            "rule_id": "DOC-TEST-002",
                            "status": "unchecked",
                            "rule_hash": "hash-2",
                        },
                    ]
                },
            )
            write_json(
                reports_dir / "doc-one_DOC-TEST-001.json",
                {
                    "file_path": "docs/doc-one.md",
                    "rule_id": "DOC-TEST-001",
                    "status": "pass",
                    "summary": "Rule is satisfied.",
                    "reason": "The document has one clear owner.",
                },
            )
            write_json(
                reports_dir / "doc-two_DOC-TEST-002.json",
                {
                    "file_path": "docs/doc-two.md",
                    "rule_id": "DOC-TEST-002",
                    "status": "fail",
                    "summary": "Document duplicates an owning contract.",
                    "reason": "The local text repeats a repository-wide rule.",
                    "findings": [
                        {
                            "rule_id": "DOC-TEST-002",
                            "file_path": "docs/doc-two.md",
                            "summary": "Document duplicates an owning contract.",
                            "severity": "should_fix",
                            "status": "open",
                            "recommended_action": "Replace the duplicate prose with a link.",
                            "evidence": [{"line": 7, "text": "Duplicated contract text."}],
                        }
                    ],
                },
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.document_audit",
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
        self.assertEqual(len(evaluations["evaluations"]), 2)
        self.assertFalse(evaluations["audit_authority"])
        self.assertNotIn("candidate_issues_json", payload["artifact_paths"])
        self.assertFalse((output_dir / "candidate-issues-from-reports.json").exists())
        self.assertFalse((output_dir / "findings-from-reports.json").exists())

    def test_fail_report_requires_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            with self.assertRaisesRegex(ValueError, "fail reports require at least one finding"):
                reports.validate_report(
                    {
                        "file_path": "docs/doc.md",
                        "rule_id": "DOC-TEST-001",
                        "status": "fail",
                        "summary": "Failure without finding.",
                        "reason": "The report does not include finding evidence.",
                    },
                    report_path=report_path,
                )

    def test_non_fail_report_rejects_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            with self.assertRaisesRegex(ValueError, "non-fail reports must not include findings"):
                reports.validate_report(
                    {
                        "file_path": "docs/doc.md",
                        "rule_id": "DOC-TEST-001",
                        "status": "pass",
                        "summary": "Passing row with finding.",
                        "reason": "The report contradicts itself.",
                        "findings": [
                            {
                                "rule_id": "DOC-TEST-001",
                                "file_path": "docs/doc.md",
                                "summary": "Finding on a pass row.",
                                "severity": "optional",
                                "status": "open",
                                "evidence": [{"line": 1, "text": "text"}],
                            }
                        ],
                    },
                    report_path=report_path,
                )

    def test_report_schema_version_and_workflow_id_must_match_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            with self.assertRaisesRegex(ValueError, "unsupported report schema_version"):
                reports.validate_report(
                    {
                        "schema_version": 999,
                        "file_path": "docs/doc.md",
                        "rule_id": "DOC-TEST-001",
                        "status": "pass",
                        "summary": "Wrong schema.",
                        "reason": "The version is unsupported.",
                    },
                    report_path=report_path,
                )
            with self.assertRaisesRegex(ValueError, "unexpected workflow_id"):
                reports.validate_report(
                    {
                        "workflow_id": "other-workflow",
                        "file_path": "docs/doc.md",
                        "rule_id": "DOC-TEST-001",
                        "status": "pass",
                        "summary": "Wrong workflow.",
                        "reason": "The workflow id is unsupported.",
                    },
                    report_path=report_path,
                )


if __name__ == "__main__":
    unittest.main()
