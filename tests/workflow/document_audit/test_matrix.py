from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.document_audit import matrix as matrix_module


class DocumentAuditMatrixTests(unittest.TestCase):
    def test_absolute_document_input_matches_project_relative_unit_and_evaluation(self) -> None:
        from tokimon.workflow import audit_project as project
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            document = base / "doc.md"
            rule = base / "rules.md"
            document.write_text("# Fixture\n")
            rule.write_text("- **DOC-TEST-001:** Test rule.\n")
            with mock.patch.dict(os.environ, {"GIT_CEILING_DIRECTORIES": str(base.parent)}), project.selected(project.resolve(base)):
                root = base
                relative = document.relative_to(root).as_posix()
                payload = matrix_module.build_matrix_payload(
                    scope_id="scope", run_id="run", document_paths=[str(document)],
                    unit_definitions=[{"audit_unit_id": "unit", "document_paths": [relative], "bound_input_paths": []}],
                    rule_sources=[rule],
                    evaluations=[{"file_path": str(document), "rule_id": "DOC-TEST-001", "status": "pass"}],
                )
                self.assertEqual(payload["document_paths"], [relative])
                self.assertEqual(len(payload["rows"]), 1)
                self.assertEqual(payload["rows"][0]["file_path"], relative)
                self.assertEqual(payload["rows"][0]["status"], "pass")

    def test_matrix_cli_rejects_unsafe_run_and_prefix_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            runs = base / "runs"
            runs.mkdir()
            victim = base / "victim"
            victim.mkdir()
            marker = victim / "marker"
            marker.write_text("unchanged")
            (runs / "linked").symlink_to(victim, target_is_directory=True)
            (runs / "loop").symlink_to("loop", target_is_directory=True)
            rule = base / "rules.md"
            rule.write_text("- **RULE-TEST-001:** Test rule.\n")
            document = base / "doc.md"
            document.write_text("# Fixture\n")
            selection = ["--document", str(document)]
            for run_id, prefix in (("../victim", "matrix"), (str(victim), "matrix"), ("nested/run", "matrix"), ("linked", "matrix"), ("loop", "matrix"), ("safe", "../../victim/matrix"), ("safe", "nested/matrix")):
                with self.subTest(run_id=run_id, prefix=prefix):
                    result = subprocess.run(
                        [sys.executable, "-m", "tokimon.workflow.document_audit", "matrix", "--scope-id", "scope", "--run-id", run_id, "--run-root", str(runs), "--source", str(rule), "--artifact-prefix", prefix, *selection, "--json"],
                        cwd=ROOT, capture_output=True, text=True,
                    )
                    payload = json.loads(result.stdout)
                    self.assertEqual(result.returncode, 2, payload)
                    self.assertFalse(payload["ok"])
                    self.assertIn("error", payload)
                    self.assertEqual(marker.read_text(), "unchanged")
                    self.assertEqual(list(victim.iterdir()), [marker])
                    self.assertFalse((runs / "safe").exists())
            result = subprocess.run(
                [sys.executable, "-m", "tokimon.workflow.document_audit", "matrix", "--scope-id", "scope", "--run-id", "legacy run.v1", "--run-root", str(runs), "--source", str(rule), "--artifact-prefix", "review v1", *selection, "--json"],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((runs / "legacy run.v1" / "review v1.json").is_file())

    def test_matrix_command_writes_complete_rule_document_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rule_source = tmp_path / "rules.md"
            doc_one = tmp_path / "doc-one.md"
            doc_two = tmp_path / "doc-two.md"
            evaluations = tmp_path / "evaluations.json"
            run_root = tmp_path / "runs"
            rule_source.write_text(
                "\n".join(
                    (
                        "- **DOC-TEST-001:** First rule.",
                        "- **DOC-TEST-002:** Second rule.",
                    )
                ),
                encoding="utf-8",
            )
            doc_one.write_text("# One\n", encoding="utf-8")
            doc_two.write_text("# Two\n", encoding="utf-8")
            evaluations.write_text(
                json.dumps(
                    [
                        {
                            "file_path": str(doc_one),
                            "rule_id": "DOC-TEST-001",
                            "status": "pass",
                        },
                        {
                            "file_path": str(doc_one),
                            "rule_id": "DOC-TEST-002",
                            "status": "not_applicable",
                        },
                        {
                            "file_path": str(doc_two),
                            "rule_id": "DOC-TEST-001",
                            "status": "fail",
                        },
                        {
                            "file_path": str(doc_two),
                            "rule_id": "DOC-TEST-002",
                            "status": "pass",
                            "summary": "Verified exact coverage.",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.document_audit",
                    "matrix",
                    "--scope-id",
                    "docs-example",
                    "--run-id",
                    "run-1",
                    "--run-root",
                    str(run_root),
                    "--source",
                    str(rule_source),
                    "--document",
                    str(doc_one),
                    "--document",
                    str(doc_two),
                    "--evaluations",
                    str(evaluations),
                    "--expect-rule-count",
                    "2",
                    "--expect-document-count",
                    "2",
                    "--expect-matrix-count",
                    "4",
                    "--require-complete",
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            payload = json.loads(result.stdout)

            self.assertTrue(payload["ok"])
            summary = payload["summary"]
            self.assertEqual(summary["rule_count"], 2)
            self.assertEqual(summary["document_count"], 2)
            self.assertEqual(summary["matrix_count"], 4)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["status_counts"]["pass"], 2)
            self.assertEqual(summary["status_counts"]["fail"], 1)
            self.assertEqual(summary["status_counts"]["not_applicable"], 1)
            self.assertEqual(summary["status_counts"]["unchecked"], 0)
            matrix_path = Path(payload["artifact_paths"]["matrix_json"])
            csv_path = Path(payload["artifact_paths"]["matrix_csv"])
            summary_path = Path(payload["artifact_paths"]["summary_json"])
            units_path = Path(payload["artifact_paths"]["audit_units_json"])
            self.assertTrue(matrix_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(units_path.exists())
            matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
            self.assertEqual(matrix_payload["schema_version"], 3)
            self.assertEqual(matrix_payload["resolved_source_ref"], None)
            self.assertTrue(matrix_payload["audit_snapshot_id"])
            self.assertEqual(len(matrix_payload["rule_source_snapshots"]), 1)
            self.assertEqual(len(matrix_payload["document_snapshots"]), 2)
            self.assertEqual(len(matrix_payload["rows"]), 4)
            self.assertEqual(matrix_payload["audit_unit_count"], 2)
            self.assertEqual(matrix_payload["rows"][0]["audit_unit_id"], doc_one.relative_to(ROOT).as_posix() if doc_one.is_relative_to(ROOT) else str(doc_one))
            units_payload = json.loads(units_path.read_text(encoding="utf-8"))
            self.assertEqual(units_payload["unit_count"], 2)

    def test_require_complete_rejects_unchecked_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rule_source = tmp_path / "rules.md"
            document = tmp_path / "doc.md"
            rule_source.write_text("- **DOC-TEST-001:** First rule.\n", encoding="utf-8")
            document.write_text("# Doc\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.document_audit",
                    "matrix",
                    "--scope-id",
                    "docs-example",
                    "--run-id",
                    "run-1",
                    "--run-root",
                    str(tmp_path / "runs"),
                    "--source",
                    str(rule_source),
                    "--document",
                    str(document),
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
        self.assertIn("matrix has 1 unchecked rows", payload["error"])

    def test_matrix_accepts_explicit_coherent_audit_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rule_source = tmp_path / "rules.md"
            doc_one = tmp_path / "doc-one.md"
            doc_two = tmp_path / "doc-two.md"
            owner = tmp_path / "owner.md"
            units = tmp_path / "units.json"
            rule_source.write_text("- **DOC-TEST-001:** First rule.\n", encoding="utf-8")
            doc_one.write_text("# One\n", encoding="utf-8")
            doc_two.write_text("# Two\n", encoding="utf-8")
            owner.write_text("# Owner\n", encoding="utf-8")
            units.write_text(
                json.dumps(
                    {
                        "units": [
                            {
                                "audit_unit_id": "paired-docs",
                                "document_paths": [str(doc_one), str(doc_two)],
                                "bound_input_paths": [str(owner)],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.document_audit",
                    "matrix",
                    "--scope-id",
                    "docs-example",
                    "--run-id",
                    "run-1",
                    "--run-root",
                    str(tmp_path / "runs"),
                    "--source",
                    str(rule_source),
                    "--units-from",
                    str(units),
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            payload = json.loads(result.stdout)
            matrix = json.loads(Path(payload["artifact_paths"]["matrix_json"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["summary"]["audit_unit_count"], 1)
        self.assertEqual(payload["summary"]["document_count"], 2)
        self.assertEqual({row["audit_unit_id"] for row in matrix["rows"]}, {"paired-docs"})
        self.assertEqual(matrix["audit_units"][0]["bound_input_paths"], [owner.relative_to(ROOT).as_posix() if owner.is_relative_to(ROOT) else str(owner)])

    def test_compiled_evaluation_from_stale_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rule_source = tmp_path / "rules.md"
            document = tmp_path / "doc.md"
            rule_source.write_text("- **DOC-TEST-001:** First rule.\n", encoding="utf-8")
            document.write_text("# Before\n", encoding="utf-8")
            prior = matrix_module.build_matrix_payload(
                scope_id="docs-example",
                run_id="run-1",
                document_paths=[str(document)],
                rule_sources=[rule_source],
            )
            evaluation = {
                "file_path": str(document),
                "rule_id": "DOC-TEST-001",
                "status": "pass",
                "compiled_for_run_id": "run-1",
                "compiled_for_scope_id": "docs-example",
                "compiled_for_audit_snapshot_id": prior["audit_snapshot_id"],
            }
            document.write_text("# After\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match the current matrix snapshot"):
                matrix_module.build_matrix_payload(
                    scope_id="docs-example",
                    run_id="run-1",
                    document_paths=[str(document)],
                    rule_sources=[rule_source],
                    evaluations=[evaluation],
                )

    def test_legacy_compiled_evaluations_without_snapshot_provenance_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluations_path = Path(tmp) / "evaluations-from-reports.json"
            evaluations_path.write_text(
                json.dumps(
                    {
                        "workflow_id": "document-audit",
                        "artifact_role": "row_evaluations_from_reports",
                        "audit_authority": False,
                        "evaluations": [
                            {
                                "file_path": "docs/example.md",
                                "rule_id": "DOC-TEST-001",
                                "status": "pass",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing current snapshot provenance"):
                matrix_module.load_evaluations(evaluations_path)


if __name__ == "__main__":
    unittest.main()
