from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokimon.workflow import audit_project as project

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


from tests.workflow.code_audit.helpers import write_fixture


def create_inputs(root: Path) -> dict[str, Path]:
    paths = {
        "impl": root / "src" / "module.py",
        "test": root / "tests" / "test_module.py",
        "spec": root / "docs" / "spec.md",
        "rules": root / "rules.md",
    }
    paths["impl"].parent.mkdir(parents=True, exist_ok=True)
    paths["test"].parent.mkdir(parents=True, exist_ok=True)
    paths["spec"].parent.mkdir(parents=True, exist_ok=True)
    paths["impl"].write_text("def run():\n    return 1\n", encoding="utf-8")
    paths["test"].write_text("def test_run():\n    assert True\n", encoding="utf-8")
    paths["spec"].write_text("# Spec\n\nExpected behavior.\n", encoding="utf-8")
    paths["rules"].write_text("- **CODE-TEST-001:** Test rule.\n", encoding="utf-8")
    return paths


def matrix_payload(paths: dict[str, Path], *, rule_hash: str = "rule-hash") -> dict[str, object]:
    subject = {
        "audit_subject_id": "subject.one",
        "title": "Subject One",
        "implementation_paths": [str(paths["impl"].parent)],
        "test_paths": [str(paths["test"])],
        "owning_specs": [str(paths["spec"])],
        "verification_commands": ["unit test command"],
    }
    return {
        "schema_version": 3,
        "workflow_id": "code-audit",
        "run_id": "run-current",
        "scope_id": "scope-one",
        "rule_sources": [str(paths["rules"])],
        "audit_subjects": [subject],
        "rows": [
            {
                "run_id": "run-current",
                "scope_id": "scope-one",
                "audit_subject_id": "subject.one",
                "rule_id": "CODE-TEST-001",
                "status": "unchecked",
                "rule_hash": rule_hash,
            }
        ],
    }


def report_payload(
    status: str = "pass",
    *,
    subject_id: str = "subject.one",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 3,
        "workflow_id": "code-audit",
        "run_id": "run-current",
        "scope_id": "scope-one",
        "audit_subject_id": subject_id,
        "rule_id": "CODE-TEST-001",
        "status": status,
        "summary": "The row has been audited.",
        "reason": "The checked surfaces satisfy the rule.",
        "implementation_paths_examined": ["src/module.py"],
        "test_paths_examined": ["tests/test_module.py"],
        "owning_specs_examined": ["docs/spec.md"],
        "verification": ["unit test command"],
    }
    if status == "fail":
        payload["evidence"] = [{"path": "src/module.py", "line_start": 1, "excerpt": "def run"}]
    if status == "needs_human_review":
        payload["evidence"] = [{"path": "src/module.py", "line_start": 1, "excerpt": "def run"}]
        payload["review_prompt"] = "Confirm the intended behavior."
    return payload


def run_code_audit(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tokimon.workflow.code_audit", *args],
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=True,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": tempfile.gettempdir()},
    )


class CodeAuditRecheckPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        # This synthetic project is intentionally non-Git, even when TMPDIR is
        # inside a real checkout. Audit inputs in other temporary directories
        # retain their absolute identities and filesystem hash semantics.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        ceiling = mock.patch.dict(os.environ, {"GIT_CEILING_DIRECTORIES": tempfile.gettempdir()})
        ceiling.start()
        self.addCleanup(ceiling.stop)
        selected = project.selected(project.resolve(directory.name))
        selected.__enter__()
        self.addCleanup(selected.__exit__, None, None, None)

    def test_recheck_plan_counts_validated_pass_lock_as_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            prior_dir = tmp_path / "prior"
            current_dir = tmp_path / "current"
            empty_reports = tmp_path / "empty-reports"
            empty_reports.mkdir()
            write_fixture(matrix_path, matrix_payload(paths))
            write_fixture(reports_dir / "pass.json", report_payload("pass"))

            compile_result = run_code_audit(
                [
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(prior_dir),
                    "--require-complete",
                    "--json",
                ]
            )
            prior_payload = json.loads(compile_result.stdout)
            pass_locks_path = Path(prior_payload["artifact_paths"]["pass_locks_json"])
            self.assertTrue(pass_locks_path.exists())
            pass_locks = json.loads(pass_locks_path.read_text(encoding="utf-8"))
            self.assertEqual(
                pass_locks["locks"][0]["implementation_path_hashes"][0]["hash_basis"],
                "filesystem_tree_v2",
            )

            plan_result = run_code_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-pass-locks",
                    str(pass_locks_path),
                    "--output-dir",
                    str(current_dir),
                    "--json",
                ]
            )
            plan_payload = json.loads(plan_result.stdout)
            plan_path = Path(plan_payload["artifact_paths"]["recheck_plan_json"])
            self.assertEqual(plan_payload["summary"]["locked_row_count"], 1)
            self.assertEqual(plan_payload["summary"]["fresh_required_row_count"], 0)

            verify_result = run_code_audit(
                [
                    "verify-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(empty_reports),
                    "--recheck-plan",
                    str(plan_path),
                    "--require-complete",
                    "--json",
                ]
            )
            verify_payload = json.loads(verify_result.stdout)
            self.assertTrue(verify_payload["complete"])
            self.assertEqual(verify_payload["covered_report_count"], 0)
            self.assertEqual(verify_payload["locked_row_count"], 1)

            compile_locked_result = run_code_audit(
                [
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(empty_reports),
                    "--recheck-plan",
                    str(plan_path),
                    "--output-dir",
                    str(current_dir),
                    "--require-complete",
                    "--json",
                ]
            )
            compiled_payload = json.loads(compile_locked_result.stdout)
            evaluations = json.loads(Path(compiled_payload["artifact_paths"]["evaluations_json"]).read_text())
            self.assertEqual(compiled_payload["summary"]["locked_row_count"], 1)
            self.assertEqual(evaluations["evaluations"][0]["evaluation_source"], "validated_prior_pass_lock")
            self.assertFalse(evaluations["audit_authority"])

            write_fixture(empty_reports / "unexpected.json", report_payload("pass"))
            unexpected_result = run_code_audit(
                [
                    "verify-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(empty_reports),
                    "--recheck-plan",
                    str(plan_path),
                    "--require-complete",
                    "--json",
                ],
                check=False,
            )
            unexpected_payload = json.loads(unexpected_result.stdout)
            self.assertEqual(unexpected_result.returncode, 2)
            self.assertEqual(unexpected_payload["unexpected_count"], 1)
            self.assertEqual(unexpected_payload["covered_report_count"], 0)
            self.assertEqual(unexpected_payload["locked_row_count"], 1)

    def test_not_applicable_row_is_reused_when_inputs_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            prior_dir = tmp_path / "prior"
            current_dir = tmp_path / "current"
            empty_reports = tmp_path / "empty-reports"
            empty_reports.mkdir()
            write_fixture(matrix_path, matrix_payload(paths))
            write_fixture(reports_dir / "not-applicable.json", report_payload("not_applicable"))

            compile_result = run_code_audit(
                [
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(prior_dir),
                    "--require-complete",
                    "--json",
                ]
            )
            pass_locks_path = Path(json.loads(compile_result.stdout)["artifact_paths"]["pass_locks_json"])
            locks = json.loads(pass_locks_path.read_text(encoding="utf-8"))
            self.assertEqual(locks["locks"][0]["status"], "not_applicable")

            plan_result = run_code_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-pass-locks",
                    str(pass_locks_path),
                    "--output-dir",
                    str(current_dir),
                    "--json",
                ]
            )
            plan_payload = json.loads(plan_result.stdout)
            plan_path = Path(plan_payload["artifact_paths"]["recheck_plan_json"])
            self.assertEqual(plan_payload["summary"]["locked_row_count"], 1)
            self.assertEqual(plan_payload["summary"]["fresh_required_row_count"], 0)

            verify_result = run_code_audit(
                [
                    "verify-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(empty_reports),
                    "--recheck-plan",
                    str(plan_path),
                    "--require-complete",
                    "--json",
                ]
            )
            verification = json.loads(verify_result.stdout)
            compile_locked_result = run_code_audit(
                [
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(empty_reports),
                    "--recheck-plan",
                    str(plan_path),
                    "--output-dir",
                    str(current_dir),
                    "--require-complete",
                    "--json",
                ]
            )
            compiled = json.loads(compile_locked_result.stdout)
            refreshed_locks = json.loads(
                Path(compiled["artifact_paths"]["pass_locks_json"]).read_text(encoding="utf-8")
            )
            self.assertTrue(verification["complete"])
            self.assertEqual(verification["status_counts"]["not_applicable"], 1)
            self.assertEqual(verification["status_counts"]["pass"], 0)
            self.assertEqual(compiled["summary"]["pass_lock_count"], 1)
            self.assertEqual(refreshed_locks["locks"][0]["status"], "not_applicable")

    def test_changed_input_unlocks_prior_pass_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            prior_dir = tmp_path / "prior"
            current_dir = tmp_path / "current"
            write_fixture(matrix_path, matrix_payload(paths))
            write_fixture(reports_dir / "pass.json", report_payload("pass"))
            compile_result = run_code_audit(
                [
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(prior_dir),
                    "--require-complete",
                    "--json",
                ]
            )
            pass_locks_path = Path(json.loads(compile_result.stdout)["artifact_paths"]["pass_locks_json"])
            paths["impl"].write_text("def run():\n    return 2\n", encoding="utf-8")

            plan_result = run_code_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-pass-locks",
                    str(pass_locks_path),
                    "--output-dir",
                    str(current_dir),
                    "--json",
                ]
            )
            plan = json.loads((current_dir / "recheck-plan.json").read_text(encoding="utf-8"))
            plan_payload = json.loads(plan_result.stdout)

            self.assertEqual(plan_payload["summary"]["locked_row_count"], 0)
            self.assertEqual(plan_payload["summary"]["fresh_required_row_count"], 1)
            self.assertIn("implementation_path_hashes_changed", plan["rows"][0]["reason_codes"])

    def test_missing_input_unlocks_only_its_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            second_impl = tmp_path / "src-two" / "module.py"
            second_impl.parent.mkdir(parents=True)
            second_impl.write_text("def run():\n    return 2\n", encoding="utf-8")
            payload = matrix_payload(paths)
            payload["audit_subjects"][0]["implementation_paths"] = [str(paths["impl"])]
            payload["audit_subjects"].append(
                {
                    "audit_subject_id": "subject.two",
                    "title": "Subject Two",
                    "implementation_paths": [str(second_impl)],
                    "test_paths": [str(paths["test"])],
                    "owning_specs": [str(paths["spec"])],
                    "verification_commands": ["unit test command"],
                }
            )
            payload["rows"].append(
                {
                    "run_id": "run-current",
                    "scope_id": "scope-one",
                    "audit_subject_id": "subject.two",
                    "rule_id": "CODE-TEST-001",
                    "status": "unchecked",
                    "rule_hash": "rule-hash",
                }
            )
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            prior_dir = tmp_path / "prior"
            current_dir = tmp_path / "current"
            write_fixture(matrix_path, payload)
            write_fixture(reports_dir / "one.json", report_payload(subject_id="subject.one"))
            write_fixture(reports_dir / "two.json", report_payload(subject_id="subject.two"))
            compile_result = run_code_audit(
                [
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(prior_dir),
                    "--require-complete",
                    "--json",
                ]
            )
            pass_locks_path = Path(
                json.loads(compile_result.stdout)["artifact_paths"]["pass_locks_json"]
            )
            paths["impl"].unlink()

            plan_result = run_code_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-pass-locks",
                    str(pass_locks_path),
                    "--output-dir",
                    str(current_dir),
                    "--json",
                ]
            )
            plan = json.loads((current_dir / "recheck-plan.json").read_text(encoding="utf-8"))
            rows = {row["audit_subject_id"]: row for row in plan["rows"]}

        self.assertEqual(json.loads(plan_result.stdout)["summary"]["locked_row_count"], 1)
        self.assertEqual(rows["subject.one"]["action"], "fresh_report_required")
        self.assertIn("lock_input_path_does_not_exist", "_".join(rows["subject.one"]["reason_codes"]))
        self.assertEqual(rows["subject.two"]["action"], "use_validated_pass_lock")

    @unittest.skipIf(sys.platform == "win32", "POSIX executable mode is required")
    def test_executable_bit_change_unlocks_prior_pass_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            prior_dir = tmp_path / "prior"
            current_dir = tmp_path / "current"
            write_fixture(matrix_path, matrix_payload(paths))
            write_fixture(reports_dir / "pass.json", report_payload("pass"))
            compile_result = run_code_audit(
                [
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(prior_dir),
                    "--require-complete",
                    "--json",
                ]
            )
            pass_locks_path = Path(
                json.loads(compile_result.stdout)["artifact_paths"]["pass_locks_json"]
            )
            paths["impl"].chmod(paths["impl"].stat().st_mode | 0o111)

            plan_result = run_code_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-pass-locks",
                    str(pass_locks_path),
                    "--output-dir",
                    str(current_dir),
                    "--json",
                ]
            )
            plan = json.loads((current_dir / "recheck-plan.json").read_text(encoding="utf-8"))
            plan_payload = json.loads(plan_result.stdout)

            self.assertEqual(plan_payload["summary"]["locked_row_count"], 0)
            self.assertEqual(plan_payload["summary"]["fresh_required_row_count"], 1)
            self.assertIn("implementation_path_hashes_changed", plan["rows"][0]["reason_codes"])

    @unittest.skipIf(sys.platform == "win32", "POSIX symlinks are required")
    def test_symlink_type_change_with_same_bytes_unlocks_prior_pass_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            payload = matrix_payload(paths)
            payload["audit_subjects"][0]["implementation_paths"] = [str(paths["impl"])]
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            prior_dir = tmp_path / "prior"
            current_dir = tmp_path / "current"
            write_fixture(matrix_path, payload)
            write_fixture(reports_dir / "pass.json", report_payload("pass"))
            compile_result = run_code_audit(
                [
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(prior_dir),
                    "--require-complete",
                    "--json",
                ]
            )
            pass_locks_path = Path(
                json.loads(compile_result.stdout)["artifact_paths"]["pass_locks_json"]
            )
            same_bytes_target = tmp_path / "same-bytes.py"
            same_bytes_target.write_bytes(paths["impl"].read_bytes())
            paths["impl"].unlink()
            paths["impl"].symlink_to(same_bytes_target)

            plan_result = run_code_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-pass-locks",
                    str(pass_locks_path),
                    "--output-dir",
                    str(current_dir),
                    "--json",
                ]
            )
            plan = json.loads((current_dir / "recheck-plan.json").read_text(encoding="utf-8"))
            plan_payload = json.loads(plan_result.stdout)

            self.assertEqual(plan_payload["summary"]["locked_row_count"], 0)
            self.assertEqual(plan_payload["summary"]["fresh_required_row_count"], 1)
            self.assertIn("implementation_path_hashes_changed", plan["rows"][0]["reason_codes"])

    def test_prior_fail_and_human_review_rows_do_not_lock(self) -> None:
        for status in ("fail", "needs_human_review"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                paths = create_inputs(tmp_path)
                matrix_path = tmp_path / "checklist-matrix.json"
                reports_dir = tmp_path / "reports"
                output_dir = tmp_path / "compiled"
                write_fixture(matrix_path, matrix_payload(paths))
                write_fixture(reports_dir / f"{status}.json", report_payload(status))

                result = run_code_audit(
                    [
                        "compile-reports",
                        "--matrix",
                        str(matrix_path),
                        "--reports-dir",
                        str(reports_dir),
                        "--output-dir",
                        str(output_dir),
                        "--require-complete",
                        "--json",
                    ]
                )
                payload = json.loads(result.stdout)

                self.assertEqual(payload["summary"]["pass_lock_count"], 0)
                locks = json.loads(Path(payload["artifact_paths"]["pass_locks_json"]).read_text(encoding="utf-8"))
                self.assertEqual(locks["lock_count"], 0)

    def test_tampered_prior_report_hash_unlocks_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            prior_dir = tmp_path / "prior"
            current_dir = tmp_path / "current"
            report_path = reports_dir / "pass.json"
            write_fixture(matrix_path, matrix_payload(paths))
            write_fixture(report_path, report_payload("pass"))
            compile_result = run_code_audit(
                [
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(prior_dir),
                    "--require-complete",
                    "--json",
                ]
            )
            pass_locks_path = Path(json.loads(compile_result.stdout)["artifact_paths"]["pass_locks_json"])
            tampered = report_payload("pass")
            tampered["reason"] = "Tampered after lock creation."
            write_fixture(report_path, tampered)

            run_code_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-pass-locks",
                    str(pass_locks_path),
                    "--output-dir",
                    str(current_dir),
                    "--json",
                ]
            )
            plan = json.loads((current_dir / "recheck-plan.json").read_text(encoding="utf-8"))

            self.assertEqual(plan["locked_row_count"], 0)
            self.assertEqual(plan["fresh_required_row_count"], 1)
            self.assertIn("pass_lock_report_hash_changed", plan["rows"][0]["reason_codes"])

    def test_tampered_lock_id_unlocks_only_that_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            prior_dir = tmp_path / "prior"
            current_dir = tmp_path / "current"
            write_fixture(matrix_path, matrix_payload(paths))
            write_fixture(reports_dir / "pass.json", report_payload("pass"))
            compile_result = run_code_audit(
                [
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(prior_dir),
                    "--require-complete",
                    "--json",
                ]
            )
            pass_locks_path = Path(
                json.loads(compile_result.stdout)["artifact_paths"]["pass_locks_json"]
            )
            locks = json.loads(pass_locks_path.read_text(encoding="utf-8"))
            locks["locks"][0]["lock_id"] = "tampered"
            write_fixture(pass_locks_path, locks)

            plan_result = run_code_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-pass-locks",
                    str(pass_locks_path),
                    "--output-dir",
                    str(current_dir),
                    "--json",
                ]
            )
            plan = json.loads((current_dir / "recheck-plan.json").read_text(encoding="utf-8"))

        self.assertEqual(json.loads(plan_result.stdout)["summary"]["locked_row_count"], 0)
        self.assertEqual(plan["rows"][0]["action"], "fresh_report_required")
        self.assertIn("pass_lock_lock_id_is_invalid", plan["rows"][0]["reason_codes"])

    def test_incomplete_compile_does_not_emit_usable_pass_locks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            output_dir = tmp_path / "compiled"
            payload = matrix_payload(paths)
            payload["rows"].append(
                {
                    "run_id": "run-current",
                    "scope_id": "scope-one",
                    "audit_subject_id": "subject.one",
                    "rule_id": "CODE-TEST-002",
                    "status": "unchecked",
                    "rule_hash": "rule-hash-2",
                }
            )
            write_fixture(matrix_path, payload)
            write_fixture(reports_dir / "pass.json", report_payload("pass"))

            result = run_code_audit(
                [
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(output_dir),
                    "--json",
                ]
            )
            compiled = json.loads(result.stdout)
            locks = json.loads(Path(compiled["artifact_paths"]["pass_locks_json"]).read_text(encoding="utf-8"))

            self.assertFalse(compiled["summary"]["complete"])
            self.assertEqual(locks["lock_count"], 0)


if __name__ == "__main__":
    unittest.main()
