from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from tests.workflow.code_audit.helpers import reviewed_report
from tokimon.workflow import audit_project as project
from tokimon.workflow.code_audit import cli, contract, ledger, reports


class CodeAuditProjectReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "project"
        self.impl = self.repo / "src/module.py"
        self.test = self.repo / "tests/test_module.py"
        self.spec = self.repo / "docs/spec.md"
        self.rule = self.repo / "docs/guidelines/code-guidelines.md"
        for path, text in (
            (self.impl, "value = 1\n"),
            (self.test, "assert 1 == 1\n"),
            (self.spec, "Expected value: 1\n"),
            (self.rule, "- **CODE-TEST-001:** Verify the value.\n"),
            (
                self.repo / "docs/guidelines/unit-testing-guidelines.md",
                "- **UNIT-TEST-001:** Verify tests.\n",
            ),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.subjects = self.root / "subjects.json"
        self.subjects.write_text(
            json.dumps(
                {
                    "subjects": [
                        {
                            "audit_subject_id": "subject.one",
                            "title": "Synthetic subject",
                            "implementation_paths": ["src"],
                            "test_paths": ["tests/test_module.py"],
                            "owning_specs": ["docs/spec.md"],
                            "verification_commands": ["synthetic fixture command"],
                        }
                    ]
                }
            )
        )
        ceiling = mock.patch.dict(os.environ, {"GIT_CEILING_DIRECTORIES": str(self.root)})
        ceiling.start()
        self.addCleanup(ceiling.stop)
        selected = project.selected(project.resolve(self.repo))
        selected.__enter__()
        self.addCleanup(selected.__exit__, None, None, None)

    def command(self, *args: str, cwd: Path | None = None) -> tuple[int, dict]:
        out = io.StringIO()
        previous = Path.cwd()
        try:
            os.chdir(cwd or self.repo)
            with contextlib.redirect_stdout(out):
                code = cli.main([*args, "--json"])
        finally:
            os.chdir(previous)
        return code, json.loads(out.getvalue())

    def matrix(self, run: str) -> tuple[Path, dict]:
        code, result = self.command(
            "matrix",
            "--scope-id",
            "scope-one",
            "--run-id",
            run,
            "--subjects-from",
            str(self.subjects),
            "--source",
            str(self.rule),
            "--run-root",
            str(self.root / "runs"),
        )
        self.assertEqual(code, 0, result)
        path = Path(result["artifact_paths"]["matrix_json"])
        return path, json.loads(path.read_text())

    def report(self, matrix_path: Path, matrix: dict) -> Path:
        path = matrix_path.parent / "reports/row.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "schema_version": 3,
            "workflow_id": "code-audit",
            "run_id": matrix["run_id"],
            "scope_id": matrix["scope_id"],
            "audit_snapshot_id": matrix["audit_snapshot_id"],
            "audit_subject_id": "subject.one",
            "rule_id": "CODE-TEST-001",
            "status": "pass",
            "summary": "Synthetic fixture, not a real audit.",
            "reason": "Validate the audit contract.",
            "implementation_paths_examined": ["src/module.py"],
            "test_paths_examined": ["tests/test_module.py"],
            "owning_specs_examined": ["docs/spec.md"],
            "evidence": [{"path": "src/module.py", "line": 1, "text": "value = 1"}],
        }
        path.write_text(
            json.dumps(
                reviewed_report(body, path, source_dir=matrix_path.parent / "reviews")
            )
        )
        return path

    def compile(self, matrix_path: Path) -> tuple[int, dict]:
        return self.command(
            "compile-reports",
            "--matrix",
            str(matrix_path),
            "--reports-dir",
            str(matrix_path.parent / "reports"),
            "--output-dir",
            str(matrix_path.parent / "compiled"),
            "--require-complete",
        )

    def test_project_sources_and_caller_manifest_paths(self) -> None:
        code, catalog = self.command("rules")
        self.assertEqual(code, 0)
        self.assertEqual(
            {row["rule_id"] for row in catalog["rules"]},
            {"CODE-TEST-001", "UNIT-TEST-001"},
        )
        code, subjects = self.command(
            "--project-root",
            str(self.repo),
            "subjects",
            "--subjects-from",
            "subjects.json",
            cwd=self.root,
        )
        self.assertEqual(code, 0, subjects)
        self.assertEqual(subjects["subjects"][0]["implementation_paths"], ["src"])
        first = self.command("contract")[1]["contract"]
        second = self.command("contract", "--project-root", str(self.root))[1][
            "contract"
        ]
        self.assertNotEqual(first["current_ledger_path"], second["current_ledger_path"])

    def test_consumers_use_saved_checkout_and_reject_explicit_conflict(self) -> None:
        path, matrix = self.matrix("first")
        report = self.report(path, matrix)
        args = (
            "verify-reports",
            "--matrix",
            str(path),
            "--reports-dir",
            str(report.parent),
            "--require-complete",
        )
        self.assertEqual(self.command(*args, cwd=self.root)[0], 0)
        code, result = self.command(*args, "--project-root", str(self.root))
        self.assertEqual(code, 2)
        self.assertIn("does not match", result["error"])

    def test_report_changes_before_finalization_require_fresh_review(self) -> None:
        for label, target in (
            ("code", self.impl),
            ("test", self.test),
            ("owner", self.spec),
            ("rule", self.rule),
        ):
            with self.subTest(input=label):
                path, matrix = self.matrix(label)
                self.report(path, matrix)
                original = target.read_bytes()
                target.write_bytes(original + b"\n# changed after the report\n")
                code, result = self.compile(path)
                self.assertEqual(code, 2, result)
                self.assertIn("snapshot is stale", result["error"])
                self.assertFalse((path.parent / "compiled").exists())
                target.write_bytes(original)

    def test_unchanged_lock_reuses_source_and_changed_review_unlocks(self) -> None:
        path, matrix = self.matrix("first")
        report_path = self.report(path, matrix)
        code, compiled = self.compile(path)
        self.assertEqual(code, 0, compiled)
        locks = compiled["artifact_paths"]["pass_locks_json"]
        current_path, current = self.matrix("second")
        code, plan = self.command(
            "recheck-plan",
            "--matrix",
            str(current_path),
            "--prior-pass-locks",
            locks,
            "--output-dir",
            str(current_path.parent),
        )
        self.assertEqual(code, 0, plan)
        self.assertEqual(plan["summary"]["locked_row_count"], 1)
        empty = current_path.parent / "reports"
        empty.mkdir()
        code, result = self.command(
            "compile-reports",
            "--matrix",
            str(current_path),
            "--reports-dir",
            str(empty),
            "--recheck-plan",
            plan["artifact_paths"]["recheck_plan_json"],
            "--require-complete",
        )
        self.assertEqual(code, 0, result)
        evaluated = json.loads(
            Path(result["artifact_paths"]["evaluations_json"]).read_text()
        )["evaluations"][0]
        self.assertEqual(evaluated["audit_snapshot_id"], matrix["audit_snapshot_id"])
        self.assertEqual(
            evaluated["compiled_for_audit_snapshot_id"], current["audit_snapshot_id"]
        )
        self.assertEqual(
            evaluated["review"], json.loads(report_path.read_text())["review"]
        )
        source = Path(evaluated["review"]["source_path"])
        source.write_text(source.read_text() + "\n")
        code, changed = self.command(
            "recheck-plan",
            "--matrix",
            str(current_path),
            "--prior-pass-locks",
            locks,
            "--output-dir",
            str(current_path.parent / "changed"),
        )
        self.assertEqual(code, 0, changed)
        self.assertEqual(changed["summary"]["locked_row_count"], 0)

    def test_new_worktree_files_invalidate_locks_but_ignored_files_do_not(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        (self.repo / ".gitignore").write_text("*.cache\n")
        path, matrix = self.matrix("git-first")
        self.report(path, matrix)
        code, compiled = self.compile(path)
        self.assertEqual(code, 0, compiled)
        locks = compiled["artifact_paths"]["pass_locks_json"]
        (self.repo / "src/ignored.cache").write_text("ignored")
        current, _ = self.matrix("git-ignored")
        code, plan = self.command(
            "recheck-plan",
            "--matrix",
            str(current),
            "--prior-pass-locks",
            locks,
            "--output-dir",
            str(current.parent),
        )
        self.assertEqual(code, 0, plan)
        self.assertEqual(plan["summary"]["locked_row_count"], 1)
        (self.repo / "src/new module.py").write_text("added = True\n")
        current, _ = self.matrix("git-added")
        code, plan = self.command(
            "recheck-plan",
            "--matrix",
            str(current),
            "--prior-pass-locks",
            locks,
            "--output-dir",
            str(current.parent),
        )
        self.assertEqual(code, 0, plan)
        self.assertEqual(plan["summary"]["locked_row_count"], 0)

    def test_new_matrix_rejects_old_evaluations_after_input_changes(self) -> None:
        path, matrix = self.matrix("first")
        self.report(path, matrix)
        code, compiled = self.compile(path)
        self.assertEqual(code, 0, compiled)
        original = path.read_bytes()
        self.impl.write_text("value = 2\n")
        code, result = self.command(
            "matrix",
            "--scope-id",
            "scope-one",
            "--run-id",
            "first",
            "--subjects-from",
            str(self.subjects),
            "--source",
            str(self.rule),
            "--run-root",
            str(self.root / "runs"),
            "--evaluations",
            compiled["artifact_paths"]["evaluations_json"],
        )
        self.assertEqual(code, 2, result)
        self.assertIn("does not match", result["error"])
        self.assertEqual(path.read_bytes(), original)

    def test_explicit_run_directory_cannot_overwrite_another_projects_matrix(
        self,
    ) -> None:
        path, _ = self.matrix("collision")
        original = path.read_bytes()
        foreign = self.root / "foreign"
        shutil.copytree(self.repo, foreign)
        code, result = self.command(
            "matrix",
            "--project-root",
            str(foreign),
            "--scope-id",
            "scope-one",
            "--run-id",
            "collision",
            "--subjects-from",
            str(self.subjects),
            "--source",
            str(foreign / "docs/guidelines/code-guidelines.md"),
            "--run-root",
            str(self.root / "runs"),
        )
        self.assertEqual(code, 2, result)
        self.assertIn("another project or input snapshot", result["error"])
        self.assertEqual(path.read_bytes(), original)

    def test_unhashable_input_does_not_hide_changes_to_other_inputs(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        (self.repo / "docs/empty").mkdir()
        subjects = json.loads(self.subjects.read_text())
        subjects["subjects"][0]["owning_specs"].append("docs/empty")
        self.subjects.write_text(json.dumps(subjects))
        path, matrix = self.matrix("unhashable-input")
        self.assertIn("error", matrix["input_snapshots"]["subject.one"])
        self.report(path, matrix)
        code, result = self.compile(path)
        self.assertEqual(code, 0, result)
        self.assertEqual(result["summary"]["pass_lock_count"], 0)
        self.impl.write_text("value = 2\n")
        code, result = self.compile(path)
        self.assertEqual(code, 2, result)
        self.assertIn("snapshot is stale", result["error"])

    def test_matrix_tampering_is_rejected_before_compilation(self) -> None:
        path, matrix = self.matrix("tampered")
        self.report(path, matrix)
        matrix["audit_subjects"][0]["verification_commands"] = ["different command"]
        path.write_text(json.dumps(matrix))
        code, result = self.compile(path)
        self.assertEqual(code, 2, result)
        self.assertIn("snapshot identity is invalid", result["error"])
        self.assertFalse((path.parent / "compiled").exists())

    def test_group_result_preserves_each_unique_reviewed_row(self) -> None:
        self.rule.write_text(
            self.rule.read_text() + "- **CODE-TEST-002:** Verify the contract.\n"
        )
        path, matrix = self.matrix("group")
        first_path = self.report(path, matrix)
        first = json.loads(first_path.read_text())
        review = first.pop("review")
        second = dict(first, rule_id="CODE-TEST-002")
        source = Path(review["source_path"])
        source.write_text(
            json.dumps(
                {"reviewer_id": review["reviewer_id"], "reports": [first, second]}
            )
        )
        review["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        first_path.write_text(json.dumps(dict(first, review=review)))
        second_path = first_path.with_name("second.json")
        second_path.write_text(json.dumps(dict(second, review=review)))
        code, result = self.compile(path)
        self.assertEqual(code, 0, result)
        self.assertEqual(result["summary"]["pass_lock_count"], 2)
        source.write_text(
            json.dumps(
                {
                    "reviewer_id": review["reviewer_id"],
                    "reports": [first, first, second],
                }
            )
        )
        review["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "exact unique row"):
            reports.validate_report(dict(first, review=review), report_path=first_path)

    def test_current_reports_require_preserved_source_and_exact_payload(self) -> None:
        path, matrix = self.matrix("source")
        report_path = self.report(path, matrix)
        original = json.loads(report_path.read_text())
        for label, change, expected in (
            ("missing", lambda row: row.pop("review"), "review field"),
            (
                "body",
                lambda row: row.update(reason="different result"),
                "exact unique row",
            ),
            (
                "reviewer",
                lambda row: row["review"].update(reviewer_id="subagent:other"),
                "reviewer does not match",
            ),
        ):
            with self.subTest(label=label):
                row = json.loads(json.dumps(original))
                change(row)
                with self.assertRaisesRegex(ValueError, expected):
                    reports.validate_report(row, report_path=report_path)

    def test_legacy_reports_are_readable_but_cannot_cover_current_matrix(self) -> None:
        path, matrix = self.matrix("legacy")
        report_path = self.report(path, matrix)
        row = json.loads(report_path.read_text())
        row.pop("review")
        row.pop("audit_snapshot_id")
        row["schema_version"] = 2
        self.assertEqual(
            reports.validate_report(row, report_path=report_path)["schema_version"], 2
        )
        report_path.write_text(json.dumps(row))
        code, result = self.compile(path)
        self.assertEqual(code, 2)
        self.assertEqual(result["invalid_count"], 1)
        self.assertFalse((path.parent / "compiled").exists())

    def test_ledger_project_binding_and_migration_provenance_survive_updates(
        self,
    ) -> None:
        path = self.root / "ledger.json"
        evaluation = {
            "scope_id": "scope",
            "run_id": "first",
            "audit_subject_id": "subject.one",
            "rule_id": "CODE-TEST-001",
            "status": "fail",
            "summary": "Synthetic issue",
            "evidence": [{"path": "src/module.py", "line": 1, "text": "value = 1"}],
        }
        ledger.record_evaluations(
            evaluations=[evaluation],
            scope_id="scope",
            run_id="first",
            ledger_path=path,
            run_root=self.root / "ledger-runs",
        )
        saved = json.loads(path.read_text())
        entry = next(iter(saved["entries"].values()))
        entry["migration_sources"] = [
            {
                "source_ledger": "preserved-legacy.json",
                "fingerprint": entry["fingerprint"],
            }
        ]
        path.write_text(json.dumps(saved))
        for run, status in (("resolved", "pass"), ("fresh-finding", "fail")):
            ledger.record_evaluations(
                evaluations=[dict(evaluation, run_id=run, status=status)],
                scope_id="scope",
                run_id=run,
                ledger_path=path,
                run_root=self.root / "ledger-runs",
            )
            self.assertEqual(
                next(iter(json.loads(path.read_text())["entries"].values()))[
                    "migration_sources"
                ],
                entry["migration_sources"],
            )
        before = path.read_bytes()
        with project.selected(project.resolve(self.root)):
            with self.assertRaisesRegex(ValueError, "different audit project"):
                ledger.record_evaluations(
                    evaluations=[],
                    scope_id="scope",
                    run_id="foreign",
                    ledger_path=path,
                    run_root=self.root / "foreign-runs",
                )
        self.assertEqual(path.read_bytes(), before)
        saved = json.loads(before)
        saved["schema_version"] = 1
        saved.pop("project_id")
        path.write_text(json.dumps(saved))
        self.assertEqual(ledger.load_ledger(path)["schema_version"], 1)
        with self.assertRaisesRegex(ValueError, "must be migrated"):
            ledger.record_evaluations(
                evaluations=[],
                scope_id="scope",
                run_id="legacy",
                ledger_path=path,
                run_root=self.root / "legacy-runs",
            )

    def test_linked_worktrees_share_project_ledger_but_not_checkout_binding(
        self,
    ) -> None:
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "synthetic fixture",
            ],
            check=True,
        )
        linked = self.root / "linked"
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "worktree",
                "add",
                "--detach",
                "-q",
                str(linked),
            ],
            check=True,
        )
        first, second = project.resolve(self.repo), project.resolve(linked)
        self.assertEqual(
            first.state_root(contract.WORKFLOW_ID),
            second.state_root(contract.WORKFLOW_ID),
        )
        self.assertNotEqual(first.to_dict(), second.to_dict())
        path, matrix = self.matrix("primary")
        self.report(path, matrix)
        code, compiled = self.compile(path)
        self.assertEqual(code, 0, compiled)
        code, created = self.command(
            "matrix",
            "--project-root",
            str(linked),
            "--scope-id",
            "scope-one",
            "--run-id",
            "linked",
            "--subjects-from",
            str(self.subjects),
            "--source",
            str(linked / "docs/guidelines/code-guidelines.md"),
            "--run-root",
            str(self.root / "runs"),
        )
        self.assertEqual(code, 0, created)
        code, plan = self.command(
            "recheck-plan",
            "--matrix",
            created["artifact_paths"]["matrix_json"],
            "--prior-pass-locks",
            compiled["artifact_paths"]["pass_locks_json"],
            "--output-dir",
            str(self.root / "linked-plan"),
        )
        self.assertEqual(code, 0, plan)
        self.assertEqual(plan["summary"]["locked_row_count"], 0)
        saved = json.loads(
            Path(plan["artifact_paths"]["recheck_plan_json"]).read_text()
        )
        self.assertIn("project_changed", saved["rows"][0]["reason_codes"])


if __name__ == "__main__":
    unittest.main()
