from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.workflow.document_audit.helpers import reviewed_report
from tokimon.workflow import audit_project as project
from tokimon.workflow.document_audit import ledger, matrix, reports


TOKIMON = Path(__file__).resolve().parents[3]


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "-c", "user.name=Audit Fixture", "-c", "user.email=fixture@example.invalid", *args],
        text=True, stderr=subprocess.PIPE,
    ).strip()


def repository(path: Path, rule: str) -> Path:
    path.mkdir()
    (path / "rules.md").write_text(f"- **DOC-TEST-001:** {rule}\n")
    (path / "doc.md").write_text("# Same document in each repository\n")
    git(path, "init", "--quiet")
    git(path, "add", "rules.md", "doc.md")
    git(path, "commit", "--quiet", "-m", "Fixture")
    return path


def invoke(cwd: Path, *args: str) -> tuple[int, dict]:
    result = subprocess.run(
        [sys.executable, "-m", "tokimon.workflow.document_audit", *args, "--json"],
        cwd=cwd, env={**os.environ, "PYTHONPATH": str(TOKIMON)},
        text=True, capture_output=True, check=False,
    )
    return result.returncode, json.loads(result.stdout)


def inputs(root: Path) -> tuple[dict, dict]:
    source = root / "rules.md"
    document = root / "doc.md"
    source.write_text("- **DOC-TEST-001:** Test rule.\n")
    document.write_text("# Synthetic document\n")
    payload = matrix.build_matrix_payload(
        scope_id="scope", run_id="run", document_paths=[str(document)], rule_sources=[source],
    )
    body = {
        "schema_version": 3, "workflow_id": "document-audit", "run_id": "run", "scope_id": "scope",
        "audit_snapshot_id": payload["audit_snapshot_id"], "file_path": payload["document_paths"][0],
        "rule_id": "DOC-TEST-001", "status": "pass", "summary": "Synthetic test result.",
        "reason": "Fixture only.", "findings": [],
    }
    return payload, body


class DocumentAuditProjectTests(unittest.TestCase):
    def test_malformed_matrix_project_returns_a_structured_cli_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "matrix.json"
            for payload in ([], {"project": "invalid"}):
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload))
                    code, output = invoke(TOKIMON, "verify-reports", "--matrix", str(path), "--reports-dir", str(root))
                    self.assertEqual(code, 2)
                    self.assertFalse(output["ok"])
                    self.assertIn("error", output)

    def test_rules_and_head_belong_to_caller_or_explicit_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = repository(root / "first", "First project rule.")
            second = repository(root / "second", "Second project rule.")
            for cwd, args in [
                (first, ["rules"]),
                (second, ["--project-root", str(first), "rules"]),
                (second, ["rules", "--project-root", str(first)]),
            ]:
                code, output = invoke(cwd, *args, "--source", "rules.md", "--source-ref", "HEAD")
                self.assertEqual(code, 0, output)
                self.assertEqual(output["rules"][0]["description"], "First project rule.")

    def test_matrix_consumers_follow_recorded_checkout_and_reject_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = repository(root / "first", "First project rule.")
            second = repository(root / "second", "Second project rule.")
            code, output = invoke(first, "matrix", "--source", "rules.md", "--source-ref", "HEAD",
                                  "--document", "doc.md", "--scope-id", "scope", "--run-id", "run",
                                  "--run-root", str(root / "runs"))
            self.assertEqual(code, 0, output)
            path = Path(output["artifact_paths"]["matrix_json"])
            payload = json.loads(path.read_text())
            self.assertEqual(payload["project"]["project_root"], str(first))
            self.assertEqual(payload["resolved_source_ref"], git(first, "rev-parse", "HEAD"))
            reports_dir = root / "reports"
            reports_dir.mkdir()
            _, output = invoke(second, "verify-reports", "--matrix", str(path), "--reports-dir", str(reports_dir))
            self.assertEqual(output["missing_count"], 1)
            code, output = invoke(second, "verify-reports", "--matrix", str(path), "--reports-dir", str(reports_dir),
                                  "--project-root", str(second))
            self.assertEqual(code, 2)
            self.assertIn("does not match", output["error"])
            (first / "doc.md").write_text("Changed\n")
            code, output = invoke(second, "verify-reports", "--matrix", str(path), "--reports-dir", str(reports_dir))
            self.assertEqual(code, 2)
            self.assertIn("snapshot is stale", output["error"])

    def test_projects_have_separate_defaults_and_linked_worktrees_share_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = repository(root / "first", "Rule.")
            second = repository(root / "second", "Rule.")
            linked = root / "linked"
            git(first, "worktree", "add", "--quiet", "--detach", str(linked))
            contracts = []
            for checkout in (first, second, linked):
                code, output = invoke(TOKIMON, "contract", "--project-root", str(checkout))
                self.assertEqual(code, 0, output)
                contracts.append(output["contract"])
            self.assertNotEqual(contracts[0]["current_ledger_path"], contracts[1]["current_ledger_path"])
            self.assertEqual(contracts[0]["current_ledger_path"], contracts[2]["current_ledger_path"])
            self.assertNotEqual(contracts[0]["project"]["project_root"], contracts[2]["project"]["project_root"])

    def test_explicit_shared_ledger_rejects_another_project_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = repository(root / "first", "Rule.")
            second = repository(root / "second", "Rule.")
            shared = root / "ledger.json"
            with project.selected(project.resolve(first)):
                ledger.record_findings(findings=[], scope_id="scope", run_id="one", ledger_path=shared, run_root=root / "runs")
            before = shared.read_bytes()
            code, output = invoke(second, "ledger-summary", "--ledger", str(shared))
            self.assertEqual(code, 2)
            self.assertIn("different project", output["error"])
            self.assertEqual(shared.read_bytes(), before)

    def test_legacy_ledger_is_readable_but_cannot_be_silently_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "legacy.json"
            payload = {"schema_version": 1, "workflow_id": "document-audit", "entries": {"old": {"status": "rejected"}}}
            path.write_text(json.dumps(payload))
            before = path.read_bytes()
            self.assertEqual(ledger.load_ledger(path)["schema_version"], 1)
            with self.assertRaisesRegex(ValueError, "legacy unbound"):
                ledger.record_findings(findings=[], scope_id="scope", run_id="new", ledger_path=path, run_root=root / "runs")
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((root / "runs").exists())

    def test_locks_cannot_cross_projects_or_outlive_the_review_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = repository(root / "first", "Same rule.")
            second = repository(root / "second", "Same rule.")
            matrices = []
            for checkout in (first, second):
                code, output = invoke(checkout, "matrix", "--source", "rules.md", "--document", "doc.md",
                                      "--scope-id", "scope", "--run-id", "run", "--run-root", str(root / checkout.name))
                self.assertEqual(code, 0, output)
                matrices.append(Path(output["artifact_paths"]["matrix_json"]))
            bound = json.loads(matrices[0].read_text())
            body = {"schema_version": 3, "workflow_id": "document-audit", "run_id": "run", "scope_id": "scope",
                    "audit_snapshot_id": bound["audit_snapshot_id"], "file_path": "doc.md", "rule_id": "DOC-TEST-001",
                    "status": "pass", "summary": "Fixture.", "reason": "Synthetic review."}
            report = reviewed_report(body, root / "reviews")
            reports_dir = root / "reports"
            reports_dir.mkdir()
            (reports_dir / "row.json").write_text(json.dumps(report))
            code, output = invoke(TOKIMON, "clean-unit-locks", "--matrix", str(matrices[0]),
                                  "--reports-dir", str(reports_dir), "--output-dir", str(root / "locks"))
            self.assertEqual(code, 0, output)
            prior = output["clean_unit_locks_path"]
            code, output = invoke(TOKIMON, "recheck-plan", "--matrix", str(matrices[1]),
                                  "--prior-clean-unit-locks", prior, "--output-dir", str(root / "foreign-plan"))
            self.assertEqual(code, 0, output)
            self.assertEqual(output["summary"]["locked_unit_count"], 0)
            self.assertEqual(output["summary"]["active_unit_count"], 1)
            Path(report["review"]["source_path"]).write_text('{}')
            code, output = invoke(TOKIMON, "recheck-plan", "--matrix", str(matrices[0]),
                                  "--prior-clean-unit-locks", prior, "--output-dir", str(root / "changed-review-plan"))
            self.assertEqual(code, 0, output)
            self.assertEqual(output["summary"]["locked_unit_count"], 0)
            self.assertEqual(output["summary"]["active_unit_count"], 1)

    def test_recording_a_new_disposition_keeps_migrated_source_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "ledger.json"
            finding = {"rule_id": "DOC-TEST-001", "file_path": "doc.md", "summary": "Fixture finding",
                       "status": "open", "severity": "should_fix", "evidence": [{"line": 1, "text": "Fixture"}]}
            arguments = {"scope_id": "scope", "ledger_path": path, "run_root": root / "runs"}
            ledger.record_findings(findings=[finding], run_id="old", **arguments)
            payload = json.loads(path.read_text())
            entry = next(iter(payload["entries"].values()))
            source = [{"source_ledger": str(root / "history.json"), "fingerprint": "original-key"}]
            entry["migration_sources"] = source
            path.write_text(json.dumps(payload))
            ledger.record_findings(findings=[dict(finding, status="fixed")], run_id="new", **arguments)
            updated = next(iter(ledger.load_ledger(path)["entries"].values()))
            self.assertEqual(updated["status"], "fixed")
            self.assertEqual(updated["migration_sources"], source)


class DocumentAuditReviewSourceTests(unittest.TestCase):
    def test_current_reports_require_preserved_matching_review_and_keep_it_in_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, body = inputs(root)
            report_path = root / "report.json"
            with self.assertRaisesRegex(ValueError, "review field"):
                reports.validate_report(body, report_path=report_path)
            report = reviewed_report(body, root / "reviews")
            normalized = reports.validate_report(report, report_path=report_path)
            self.assertEqual(reports.evaluation_from_report(normalized)["review"], report["review"])
            report["status"] = "not_applicable"
            with self.assertRaisesRegex(ValueError, "exact unique row"):
                reports.validate_report(report, report_path=report_path)

    def test_changed_or_wrong_reviewer_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, body = inputs(root)
            report = reviewed_report(body, root / "reviews")
            source = Path(report["review"]["source_path"])
            payload = json.loads(source.read_text())
            payload["reviewer_id"] = "subagent:another-reviewer"
            source.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "source hash"):
                reports.validate_report(report, report_path=root / "report.json")
            report["review"]["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "reviewer does not match"):
                reports.validate_report(report, report_path=root / "report.json")

    def test_one_preserved_group_can_supply_multiple_rows_and_detect_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, first = inputs(root)
            second = dict(first, rule_id="DOC-TEST-002", status="not_applicable")
            source = root / "review.json"
            source.write_text(json.dumps({"reviewer_id": "subagent:group", "reports": [first, second]}))
            reference = {"reviewer_id": "subagent:group", "source_path": "review.json", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            for body in (first, second):
                normalized = reports.validate_report(dict(body, review=reference), report_path=root / "report.json")
                self.assertEqual(normalized["review"]["source_path"], str(source))
            source.write_text(json.dumps({"reviewer_id": "subagent:group", "reports": [first, first]}))
            reference["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "exact unique row"):
                reports.validate_report(dict(first, review=reference), report_path=root / "report.json")

    def test_legacy_report_cannot_supply_current_matrix_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload, body = inputs(root)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(json.dumps(payload))
            reports_dir = root / "reports"
            reports_dir.mkdir()
            (reports_dir / "legacy.json").write_text(json.dumps(dict(body, schema_version=2)))
            result = reports.verify_report_coverage(matrix_path=matrix_path, reports_dir=reports_dir)
            self.assertFalse(result["complete"])
            self.assertEqual(result["invalid_count"], 1)


if __name__ == "__main__":
    unittest.main()
