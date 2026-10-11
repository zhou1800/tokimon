from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.code_audit import matrix as audit_matrix, subjects


class CodeAuditMatrixTests(unittest.TestCase):
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
            manifest = base / "subjects.json"
            manifest.write_text(json.dumps({"subjects": [{"audit_subject_id": "synthetic", "title": "Synthetic boundary", "notes": "Fixture only"}]}))
            selection = ["--subjects-from", str(manifest)]
            for run_id, prefix in (("../victim", "matrix"), (str(victim), "matrix"), ("nested/run", "matrix"), ("linked", "matrix"), ("loop", "matrix"), ("safe", "../../victim/matrix"), ("safe", "nested/matrix")):
                with self.subTest(run_id=run_id, prefix=prefix):
                    result = subprocess.run(
                        [sys.executable, "-m", "tokimon.workflow.code_audit", "matrix", "--scope-id", "scope", "--run-id", run_id, "--run-root", str(runs), "--source", str(rule), "--artifact-prefix", prefix, *selection, "--json"],
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
                [sys.executable, "-m", "tokimon.workflow.code_audit", "matrix", "--scope-id", "scope", "--run-id", "legacy run.v1", "--run-root", str(runs), "--source", str(rule), "--artifact-prefix", "review v1", *selection, "--json"],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((runs / "legacy run.v1" / "review v1.json").is_file())

    def test_matrix_command_writes_complete_rule_subject_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rule_source = tmp_path / "rules.md"
            subject_file = tmp_path / "subjects.json"
            evaluations = tmp_path / "evaluations.json"
            run_root = tmp_path / "runs"
            rule_source.write_text(
                "\n".join(
                    (
                        "- **CODE-TEST-001:** First rule.",
                        "- **CODE-TEST-002:** Second rule.",
                    )
                ),
                encoding="utf-8",
            )
            subject_file.write_text(
                json.dumps(
                    {
                        "subjects": [
                            {
                                "run_id": "run-1",
                                "scope_id": "tokimon-workflows",
                                "audit_subject_id": "tokimon.workflow.document_audit",
                                "title": "Document Audit workflow",
                                "implementation_paths": ["tokimon/workflow/document_audit"],
                                "test_paths": ["tests/workflow/document_audit"],
                                "owning_specs": [
                                    "docs/c4/level-3-components/workflow/document-audit-cli.md"
                                ],
                                "verification_commands": [
                                    ".venv/bin/python -m unittest discover -s tests/workflow/document_audit -t ."
                                ],
                            },
                            {
                                "audit_subject_id": "tokimon.workflow.pair",
                                "title": "Pair workflow",
                                "implementation_paths": ["tokimon/workflow/pair"],
                                "test_paths": ["tests/workflow/pair"],
                                "owning_specs": ["docs/c4/level-3-components/workflow/pair-cli.md"],
                                "verification_commands": [
                                    ".venv/bin/python -m unittest discover -s tests/workflow/pair -t ."
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            evaluations.write_text(
                json.dumps(
                    {
                        "evaluations": [
                            {
                                "run_id": "run-1",
                                "scope_id": "tokimon-workflows",
                                "audit_subject_id": "tokimon.workflow.document_audit",
                                "rule_id": "CODE-TEST-001",
                                "status": "pass",
                                "verification": ["unit test command"],
                            },
                            {
                                "run_id": "run-1",
                                "scope_id": "tokimon-workflows",
                                "audit_subject_id": "tokimon.workflow.document_audit",
                                "rule_id": "CODE-TEST-002",
                                "status": "not_applicable",
                                "reason": "Rule does not apply to this subject.",
                            },
                            {
                                "run_id": "run-1",
                                "scope_id": "tokimon-workflows",
                                "audit_subject_id": "tokimon.workflow.pair",
                                "rule_id": "CODE-TEST-001",
                                "status": "fail",
                                "summary": "Pair subject has a failing contract row.",
                                "evidence": [
                                    {
                                        "path": "tokimon/workflow/pair/contract.py",
                                        "line_start": 10,
                                        "excerpt": "contract",
                                    }
                                ],
                            },
                            {
                                "run_id": "run-1",
                                "scope_id": "tokimon-workflows",
                                "audit_subject_id": "tokimon.workflow.pair",
                                "rule_id": "CODE-TEST-002",
                                "status": "needs_human_review",
                                "summary": "Dependency direction needs review.",
                                "reason": "The contract boundary is semantic.",
                                "evidence": [
                                    {
                                        "path": "tokimon/workflow/pair/contract.py",
                                        "line_start": 12,
                                        "excerpt": "dependency",
                                    }
                                ],
                                "review_prompt": "Confirm whether this dependency direction is intended.",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            unchecked = audit_matrix.build_matrix_payload(
                scope_id="tokimon-workflows", run_id="run-1",
                audit_subjects=subjects.load_subjects(subject_file), rule_sources=[rule_source],
            )
            evaluation_payload = json.loads(evaluations.read_text())
            for row in evaluation_payload["evaluations"]:
                row["audit_snapshot_id"] = unchecked["audit_snapshot_id"]
            evaluations.write_text(json.dumps(evaluation_payload))

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.code_audit",
                    "matrix",
                    "--scope-id",
                    "tokimon-workflows",
                    "--run-id",
                    "run-1",
                    "--run-root",
                    str(run_root),
                    "--source",
                    str(rule_source),
                    "--subjects-from",
                    str(subject_file),
                    "--evaluations",
                    str(evaluations),
                    "--expect-rule-count",
                    "2",
                    "--expect-subject-count",
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
            self.assertEqual(summary["audit_subject_count"], 2)
            self.assertEqual(summary["matrix_count"], 4)
            self.assertEqual(summary["status_counts"]["needs_human_review"], 1)
            self.assertTrue(summary["complete"])
            for artifact in payload["artifact_paths"].values():
                self.assertTrue(Path(artifact).exists())

    def test_require_complete_rejects_unchecked_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rule_source = tmp_path / "rules.md"
            subject_file = tmp_path / "subjects.json"
            rule_source.write_text("- **CODE-TEST-001:** First rule.\n", encoding="utf-8")
            subject_file.write_text(
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

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.code_audit",
                    "matrix",
                    "--scope-id",
                    "tokimon-workflows",
                    "--run-id",
                    "run-1",
                    "--run-root",
                    str(tmp_path / "runs"),
                    "--source",
                    str(rule_source),
                    "--subjects-from",
                    str(subject_file),
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


if __name__ == "__main__":
    unittest.main()
