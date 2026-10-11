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

from tests.workflow.document_audit.helpers import reviewed_report
from tokimon.workflow.document_audit import ledger as ledger_module
from tokimon.workflow.document_audit import matrix as matrix_module
from tokimon.workflow.document_audit import recheck_plan as recheck_plan_module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def create_inputs(root: Path) -> dict[str, Path]:
    paths = {
        "doc_a": root / "docs" / "a.md",
        "doc_b": root / "docs" / "b.md",
        "doc_c": root / "docs" / "c.md",
        "rules": root / "rules.md",
    }
    paths["doc_a"].parent.mkdir(parents=True, exist_ok=True)
    paths["doc_a"].write_text("# A\n", encoding="utf-8")
    paths["doc_b"].write_text("# B\n", encoding="utf-8")
    paths["doc_c"].write_text("# C\n", encoding="utf-8")
    paths["rules"].write_text("- **DOC-TEST-001:** Test rule.\n", encoding="utf-8")
    return paths


def matrix_payload(paths: dict[str, Path], *, run_id: str = "run-current") -> dict[str, object]:
    units = [
        {
            "audit_unit_id": "unit-a",
            "document_paths": [str(paths["doc_a"])],
            "bound_input_paths": [],
        },
        {
            "audit_unit_id": "unit-b",
            "document_paths": [str(paths["doc_b"])],
            "bound_input_paths": [str(paths["doc_a"])],
        },
        {
            "audit_unit_id": "unit-c",
            "document_paths": [str(paths["doc_c"])],
            "bound_input_paths": [],
        },
    ]
    return matrix_module.build_matrix_payload(
        run_id=run_id,
        scope_id="scope-one",
        document_paths=[str(paths["doc_a"]), str(paths["doc_b"]), str(paths["doc_c"])],
        unit_definitions=units,
        rule_sources=[paths["rules"]],
    )


def report_payload(
    matrix: dict[str, object],
    path: Path,
    *,
    status: str = "pass",
    finding_status: str = "open",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 3,
        "workflow_id": "document-audit",
        "run_id": matrix["run_id"],
        "scope_id": matrix["scope_id"],
        "audit_snapshot_id": matrix["audit_snapshot_id"],
        "file_path": ledger_module.display_path(path),
        "rule_id": "DOC-TEST-001",
        "status": status,
        "summary": "The row has been audited.",
        "reason": "The document satisfies the rule.",
        "findings": [],
    }
    if status == "fail":
        payload["findings"] = [
            {
                "rule_id": "DOC-TEST-001",
                "file_path": ledger_module.display_path(path),
                "summary": "The document does not satisfy the rule.",
                "severity": "should_fix",
                "status": finding_status,
                "evidence": [{"line": 1, "text": "# A"}],
            }
        ]
    return reviewed_report(payload, path.parent.parent / "reviews" / str(matrix["run_id"]))


def run_document_audit(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tokimon.workflow.document_audit", *args, "--project-root", str(project.root())],
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=True,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": tempfile.gettempdir()},
    )


def create_clean_locks(root: Path, paths: dict[str, Path]) -> Path:
    matrix_path = root / "checklist-matrix.json"
    reports_dir = root / "reports"
    locks_dir = root / "locks"
    matrix = matrix_payload(paths)
    write_json(matrix_path, matrix)
    write_json(reports_dir / "a.json", report_payload(matrix, paths["doc_a"]))
    write_json(reports_dir / "b.json", report_payload(matrix, paths["doc_b"], status="not_applicable"))
    write_json(reports_dir / "c.json", report_payload(matrix, paths["doc_c"]))
    result = run_document_audit(
        [
            "clean-unit-locks",
            "--matrix",
            str(matrix_path),
            "--reports-dir",
            str(reports_dir),
            "--output-dir",
            str(locks_dir),
            "--ledger",
            str(root / "current.json"),
            "--json",
        ]
    )
    payload = json.loads(result.stdout)
    if payload["clean_unit_lock_count"] != 3:
        raise AssertionError(payload)
    return Path(payload["clean_unit_locks_path"])


class DocumentAuditRecheckPlanTests(unittest.TestCase):
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

    def test_cycle_members_share_external_active_prerequisites(self) -> None:
        blockers = recheck_plan_module._active_dependency_blockers(
            units_by_id={
                "a": {"bound_input_paths": ["b.md"]},
                "b": {"bound_input_paths": ["a.md", "c.md"]},
                "c": {"bound_input_paths": []},
            },
            unit_by_document={"a.md": "a", "b.md": "b", "c.md": "c"},
            active_unit_ids={"a", "b", "c"},
        )
        self.assertEqual(blockers, {"a": {"c"}, "b": {"c"}, "c": set()})
        cleared = recheck_plan_module._active_dependency_blockers(
            units_by_id={
                "a": {"bound_input_paths": ["b.md"]},
                "b": {"bound_input_paths": ["a.md", "c.md"]},
                "c": {"bound_input_paths": []},
            },
            unit_by_document={"a.md": "a", "b.md": "b", "c.md": "c"},
            active_unit_ids={"a", "b"},
        )
        self.assertEqual(cleared, {"a": set(), "b": set()})

    def test_fresh_terminal_state_requires_acceptance_from_the_source_run(self) -> None:
        report = {
            "status": "fail",
            "findings": [
                {
                    "rule_id": "DOC-TEST-001",
                    "file_path": "docs/a.md",
                    "summary": "Needs an owner decision.",
                    "status": "open",
                }
            ],
        }
        fingerprint = ledger_module.make_fingerprint(
            "scope-one",
            "DOC-TEST-001",
            "docs/a.md",
            "Needs an owner decision.",
        )
        entry = {
            "scope_id": "scope-one",
            "rule_id": "DOC-TEST-001",
            "file_path": "docs/a.md",
            "normalized_subject": "needs-an-owner-decision",
            "status": "blocked",
            "last_seen_run": "run-prior",
        }

        stale = recheck_plan_module._terminal_dispositions(
            reports=[report],
            scope_id="scope-one",
            entries={fingerprint: entry},
            required_last_seen_run="run-current",
        )
        current = recheck_plan_module._terminal_dispositions(
            reports=[report],
            scope_id="scope-one",
            entries={fingerprint: dict(entry, last_seen_run="run-current")},
            required_last_seen_run="run-current",
        )

        self.assertIsNone(stale)
        self.assertEqual(current[0]["status"], "blocked")

    def test_dependency_waves_wait_downstream_but_keep_cycles_active(self) -> None:
        units = {
            "unit-a": {"bound_input_paths": []},
            "unit-b": {"bound_input_paths": ["a.md"]},
            "unit-c": {"bound_input_paths": ["b.md"]},
            "unit-d": {"bound_input_paths": ["e.md"]},
            "unit-e": {"bound_input_paths": ["d.md"]},
        }
        blockers = recheck_plan_module._active_dependency_blockers(
            units_by_id=units,
            unit_by_document={
                "a.md": "unit-a",
                "b.md": "unit-b",
                "c.md": "unit-c",
                "d.md": "unit-d",
                "e.md": "unit-e",
            },
            active_unit_ids=set(units),
        )

        self.assertEqual(blockers["unit-a"], set())
        self.assertEqual(blockers["unit-b"], {"unit-a"})
        self.assertEqual(blockers["unit-c"], {"unit-b"})
        self.assertEqual(blockers["unit-d"], set())
        self.assertEqual(blockers["unit-e"], set())

    def test_valid_clean_unit_locks_supply_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "next-matrix.json"
            empty_reports = tmp_path / "empty-reports"
            empty_reports.mkdir()
            locks_path = create_clean_locks(tmp_path, paths)
            current_matrix = matrix_payload(paths, run_id="run-next")
            write_json(matrix_path, current_matrix)

            plan_result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(locks_path),
                    "--output-dir",
                    str(tmp_path / "recheck"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ]
            )
            plan_payload = json.loads(plan_result.stdout)
            plan_path = Path(plan_payload["artifact_paths"]["recheck_plan_json"])
            verify_result = run_document_audit(
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
            compile_result = run_document_audit(
                [
                    "compile-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(empty_reports),
                    "--recheck-plan",
                    str(plan_path),
                    "--output-dir",
                    str(tmp_path / "compiled"),
                    "--require-complete",
                    "--json",
                ]
            )
            compiled_payload = json.loads(compile_result.stdout)
            evaluations = json.loads(
                Path(compiled_payload["artifact_paths"]["evaluations_json"]).read_text(encoding="utf-8")
            )
            rebuilt_with_evaluations = matrix_module.build_matrix_payload(
                run_id="run-next",
                scope_id="scope-one",
                document_paths=[
                    str(paths["doc_a"]),
                    str(paths["doc_b"]),
                    str(paths["doc_c"]),
                ],
                unit_definitions=current_matrix["audit_units"],
                rule_sources=[paths["rules"]],
                evaluations=matrix_module.load_evaluations(
                    compiled_payload["artifact_paths"]["evaluations_json"]
                ),
            )

        self.assertEqual(plan_payload["summary"]["locked_unit_count"], 3)
        self.assertEqual(plan_payload["summary"]["active_unit_count"], 0)
        self.assertTrue(verify_payload["complete"])
        self.assertEqual(verify_payload["covered_report_count"], 0)
        self.assertEqual(verify_payload["locked_row_count"], 3)
        self.assertEqual(verify_payload["expected_fresh_report_count"], 0)
        self.assertEqual(verify_payload["status_counts"]["not_applicable"], 1)
        self.assertEqual(len(evaluations["evaluations"]), 3)
        self.assertEqual(evaluations["audit_snapshot_id"], current_matrix["audit_snapshot_id"])
        self.assertTrue(
            all(
                evaluation["compiled_for_audit_snapshot_id"] == evaluations["audit_snapshot_id"]
                for evaluation in evaluations["evaluations"]
            )
        )
        self.assertTrue(rebuilt_with_evaluations["complete"])

    def test_changed_document_activates_only_unit_and_bound_input_dependents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            locks_path = create_clean_locks(tmp_path, paths)
            paths["doc_a"].write_text("# A\n\nChanged.\n", encoding="utf-8")
            write_json(matrix_path, matrix_payload(paths, run_id="run-next"))

            result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(locks_path),
                    "--output-dir",
                    str(tmp_path / "recheck"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ]
            )
            payload = json.loads(result.stdout)
            plan = json.loads(Path(payload["artifact_paths"]["recheck_plan_json"]).read_text(encoding="utf-8"))
            units = {unit["audit_unit_id"]: unit for unit in plan["units"]}

        self.assertEqual(payload["summary"]["active_unit_count"], 1)
        self.assertEqual(payload["summary"]["waiting_unit_count"], 1)
        self.assertEqual(payload["summary"]["locked_unit_count"], 1)
        self.assertIn("document_path_hashes_changed", units["unit-a"]["reason_codes"])
        self.assertIn("bound_input_path_hashes_changed", units["unit-b"]["reason_codes"])
        self.assertEqual(units["unit-b"]["action"], "waiting_on_dependency")
        self.assertEqual(units["unit-c"]["action"], "use_validated_clean_unit_lock")

    def test_explicit_activation_does_not_speculatively_activate_dependents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            locks_path = create_clean_locks(tmp_path, paths)

            result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(locks_path),
                    "--output-dir",
                    str(tmp_path / "recheck"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--activate-unit",
                    "unit-a",
                    "--json",
                ]
            )
            payload = json.loads(result.stdout)
            plan = json.loads(Path(payload["artifact_paths"]["recheck_plan_json"]).read_text(encoding="utf-8"))
            units = {unit["audit_unit_id"]: unit for unit in plan["units"]}

        self.assertEqual(payload["summary"]["active_unit_count"], 1)
        self.assertIn("explicitly_activated", units["unit-a"]["reason_codes"])
        self.assertEqual(units["unit-b"]["action"], "use_validated_clean_unit_lock")
        self.assertEqual(units["unit-c"]["action"], "use_validated_clean_unit_lock")

    def test_lock_refresh_preserves_valid_locks_and_adds_newly_clean_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            prior_locks = create_clean_locks(tmp_path, paths)
            paths["doc_a"].write_text("# A\n\nChanged.\n", encoding="utf-8")
            current_matrix = matrix_payload(paths, run_id="run-next")
            write_json(matrix_path, current_matrix)
            first_plan_result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(prior_locks),
                    "--output-dir",
                    str(tmp_path / "first-recheck"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ]
            )
            first_plan = Path(json.loads(first_plan_result.stdout)["artifact_paths"]["recheck_plan_json"])
            fresh_reports = tmp_path / "fresh-reports"
            write_json(fresh_reports / "a.json", report_payload(current_matrix, paths["doc_a"]))
            refresh_result = run_document_audit(
                [
                    "clean-unit-locks",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(fresh_reports),
                    "--recheck-plan",
                    str(first_plan),
                    "--output-dir",
                    str(tmp_path / "refreshed-locks"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ]
            )
            refresh_payload = json.loads(refresh_result.stdout)
            refreshed_locks = Path(refresh_payload["clean_unit_locks_path"])
            second_plan_result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(refreshed_locks),
                    "--output-dir",
                    str(tmp_path / "second-recheck"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ]
            )
            second_payload = json.loads(second_plan_result.stdout)
            second_plan = Path(second_payload["artifact_paths"]["recheck_plan_json"])
            dependent_reports = tmp_path / "dependent-reports"
            write_json(
                dependent_reports / "b.json",
                report_payload(current_matrix, paths["doc_b"]),
            )
            final_lock_result = run_document_audit(
                [
                    "clean-unit-locks",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(dependent_reports),
                    "--recheck-plan",
                    str(second_plan),
                    "--output-dir",
                    str(tmp_path / "final-locks"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ]
            )
            final_lock_payload = json.loads(final_lock_result.stdout)
            final_plan_result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(final_lock_payload["clean_unit_locks_path"]),
                    "--output-dir",
                    str(tmp_path / "final-recheck"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ]
            )
            final_plan_payload = json.loads(final_plan_result.stdout)

        self.assertEqual(refresh_payload["clean_unit_lock_count"], 2)
        self.assertEqual(second_payload["summary"]["locked_unit_count"], 2)
        self.assertEqual(second_payload["summary"]["active_unit_count"], 1)
        self.assertEqual(second_payload["summary"]["waiting_unit_count"], 0)
        self.assertEqual(final_lock_payload["clean_unit_lock_count"], 3)
        self.assertEqual(final_plan_payload["summary"]["locked_unit_count"], 3)
        self.assertEqual(final_plan_payload["summary"]["active_unit_count"], 0)
        self.assertEqual(final_plan_payload["summary"]["waiting_unit_count"], 0)

    def test_dependent_unit_waits_until_bound_unit_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            prior_locks = create_clean_locks(tmp_path, paths)
            paths["doc_a"].write_text("# A\n\nStill needs work.\n", encoding="utf-8")
            current_matrix = matrix_payload(paths, run_id="run-next")
            write_json(matrix_path, current_matrix)
            plan_result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(prior_locks),
                    "--output-dir",
                    str(tmp_path / "recheck"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ]
            )
            plan_path = Path(json.loads(plan_result.stdout)["artifact_paths"]["recheck_plan_json"])
            fresh_reports = tmp_path / "fresh-reports"
            write_json(
                fresh_reports / "a.json",
                report_payload(current_matrix, paths["doc_a"], status="fail"),
            )
            ledger_path = tmp_path / "current.json"
            write_json(
                ledger_path,
                {
                    "schema_version": 1,
                    "workflow_id": "document-audit",
                    "entries": {
                        "open-a": {
                            "scope_id": "scope-one",
                            "status": "open",
                            "file_path": str(paths["doc_a"]),
                        }
                    },
                },
            )
            refresh_result = run_document_audit(
                [
                    "clean-unit-locks",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(fresh_reports),
                    "--recheck-plan",
                    str(plan_path),
                    "--output-dir",
                    str(tmp_path / "refreshed-locks"),
                    "--ledger",
                    str(ledger_path),
                    "--json",
                ]
            )
            refresh_payload = json.loads(refresh_result.stdout)
            locks = json.loads(Path(refresh_payload["clean_unit_locks_path"]).read_text(encoding="utf-8"))
            skipped = {entry["audit_unit_id"]: entry for entry in locks["skipped"]}
            next_plan_result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(refresh_payload["clean_unit_locks_path"]),
                    "--output-dir",
                    str(tmp_path / "next-recheck"),
                    "--ledger",
                    str(ledger_path),
                    "--json",
                ]
            )
            next_plan = json.loads(
                Path(json.loads(next_plan_result.stdout)["artifact_paths"]["recheck_plan_json"]).read_text()
            )
            next_units = {unit["audit_unit_id"]: unit for unit in next_plan["units"]}

        self.assertEqual(refresh_payload["clean_unit_lock_count"], 1)
        self.assertIn("unit_has_open_findings", skipped["unit-a"]["reason_codes"])
        self.assertIn("waiting_on_dependency", skipped["unit-b"]["reason_codes"])
        self.assertEqual(skipped["unit-b"]["blocked_by_unit_ids"], ["unit-a"])
        self.assertEqual(next_units["unit-b"]["action"], "waiting_on_dependency")

    def test_open_finding_prevents_clean_unit_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            ledger_path = tmp_path / "current.json"
            matrix = matrix_payload(paths)
            write_json(matrix_path, matrix)
            for name in ("doc_a", "doc_b", "doc_c"):
                write_json(reports_dir / f"{name}.json", report_payload(matrix, paths[name]))
            write_json(
                ledger_path,
                {
                    "schema_version": 1,
                    "workflow_id": "document-audit",
                    "entries": {
                        "open-a": {
                            "scope_id": "scope-one",
                            "status": "open",
                            "file_path": str(paths["doc_a"]),
                        }
                    },
                },
            )

            result = run_document_audit(
                [
                    "clean-unit-locks",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(tmp_path / "locks"),
                    "--ledger",
                    str(ledger_path),
                    "--json",
                ]
            )
            payload = json.loads(result.stdout)
            locks = json.loads(Path(payload["clean_unit_locks_path"]).read_text(encoding="utf-8"))

        skipped = {entry["audit_unit_id"]: entry for entry in locks["skipped"]}
        self.assertEqual(payload["clean_unit_lock_count"], 1)
        self.assertIn("unit_has_open_findings", skipped["unit-a"]["reason_codes"])
        self.assertIn("waiting_on_dependency", skipped["unit-b"]["reason_codes"])

    def test_tampered_source_report_unlocks_only_its_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            locks_path = create_clean_locks(tmp_path, paths)
            report_path = tmp_path / "reports" / "a.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["reason"] = "Tampered after lock creation."
            write_json(report_path, report)

            result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(locks_path),
                    "--output-dir",
                    str(tmp_path / "recheck"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ]
            )
            payload = json.loads(result.stdout)
            plan = json.loads(Path(payload["artifact_paths"]["recheck_plan_json"]).read_text(encoding="utf-8"))
            units = {unit["audit_unit_id"]: unit for unit in plan["units"]}

        self.assertEqual(payload["summary"]["active_unit_count"], 1)
        self.assertIn("source_report_hash_changed", "_".join(units["unit-a"]["reason_codes"]))
        self.assertEqual(units["unit-b"]["action"], "use_validated_clean_unit_lock")
        self.assertEqual(units["unit-c"]["action"], "use_validated_clean_unit_lock")

    def test_report_from_another_snapshot_cannot_create_a_clean_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            old_matrix = matrix_payload(paths, run_id="run-old")
            reports_dir = tmp_path / "reports"
            write_json(reports_dir / "a.json", report_payload(old_matrix, paths["doc_a"]))
            paths["doc_a"].write_text("# A\n\nChanged after the old audit.\n", encoding="utf-8")
            current_matrix_path = tmp_path / "current-matrix.json"
            write_json(current_matrix_path, matrix_payload(paths, run_id="run-current"))

            result = run_document_audit(
                [
                    "clean-unit-locks",
                    "--matrix",
                    str(current_matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(tmp_path / "locks"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ],
                check=False,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 2)
        self.assertFalse(payload["complete"])
        self.assertEqual(payload["invalid_count"], 1)
        self.assertIn("report provenance mismatch", payload["invalid_reports"][0]["error"])

    def test_waiting_rows_need_no_report_and_reject_one_if_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            prior_locks = create_clean_locks(tmp_path, paths)
            paths["doc_a"].write_text("# A\n\nChanged.\n", encoding="utf-8")
            current_matrix = matrix_payload(paths, run_id="run-next")
            matrix_path = tmp_path / "current-matrix.json"
            write_json(matrix_path, current_matrix)
            plan_result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(prior_locks),
                    "--output-dir",
                    str(tmp_path / "recheck"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ]
            )
            plan_path = Path(json.loads(plan_result.stdout)["artifact_paths"]["recheck_plan_json"])
            reports_dir = tmp_path / "fresh-reports"
            write_json(reports_dir / "a.json", report_payload(current_matrix, paths["doc_a"]))
            first_verify = run_document_audit(
                [
                    "verify-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--recheck-plan",
                    str(plan_path),
                    "--require-complete",
                    "--json",
                ]
            )
            first_payload = json.loads(first_verify.stdout)
            write_json(reports_dir / "b.json", report_payload(current_matrix, paths["doc_b"]))
            unexpected_verify = run_document_audit(
                [
                    "verify-reports",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--recheck-plan",
                    str(plan_path),
                    "--require-complete",
                    "--json",
                ],
                check=False,
            )
            unexpected_payload = json.loads(unexpected_verify.stdout)

        self.assertTrue(first_payload["complete"])
        self.assertFalse(first_payload["final_coverage_complete"])
        self.assertEqual(first_payload["waiting_row_count"], 1)
        self.assertEqual(unexpected_verify.returncode, 2)
        self.assertEqual(unexpected_payload["unexpected_count"], 1)

    def test_stale_matrix_snapshot_fails_before_lock_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            locks_path = create_clean_locks(tmp_path, paths)
            paths["rules"].write_text(
                "- **DOC-TEST-001:** Changed rule meaning.\n",
                encoding="utf-8",
            )

            result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(locks_path),
                    "--output-dir",
                    str(tmp_path / "recheck"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ],
                check=False,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 2)
        self.assertIn("guideline source snapshot is stale", payload["error"])

    def test_open_report_accepted_as_blocked_remains_settled_after_rerecord(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            reports_dir = tmp_path / "reports"
            ledger_path = tmp_path / "current.json"
            matrix = matrix_payload(paths, run_id="run-one")
            write_json(matrix_path, matrix)
            open_report = report_payload(
                matrix,
                paths["doc_a"],
                status="fail",
            )
            write_json(reports_dir / "a.json", open_report)
            write_json(reports_dir / "b.json", report_payload(matrix, paths["doc_b"]))
            write_json(reports_dir / "c.json", report_payload(matrix, paths["doc_c"]))
            accepted = tmp_path / "accepted.json"
            accepted_finding = dict(open_report["findings"][0], status="blocked")
            write_json(accepted, {"findings": [accepted_finding]})
            run_document_audit(
                [
                    "record-findings",
                    "--accepted-findings",
                    str(accepted),
                    "--scope-id",
                    "scope-one",
                    "--run-id",
                    "run-one",
                    "--ledger",
                    str(ledger_path),
                    "--run-root",
                    str(tmp_path / "runs"),
                    "--source",
                    str(paths["rules"]),
                    "--json",
                ]
            )
            lock_result = run_document_audit(
                [
                    "clean-unit-locks",
                    "--matrix",
                    str(matrix_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--output-dir",
                    str(tmp_path / "locks"),
                    "--ledger",
                    str(ledger_path),
                    "--json",
                ]
            )
            locks_payload = json.loads(lock_result.stdout)
            run_document_audit(
                [
                    "record-findings",
                    "--accepted-findings",
                    str(accepted),
                    "--scope-id",
                    "scope-one",
                    "--run-id",
                    "run-two",
                    "--ledger",
                    str(ledger_path),
                    "--run-root",
                    str(tmp_path / "runs"),
                    "--source",
                    str(paths["rules"]),
                    "--json",
                ]
            )
            current_matrix = matrix_payload(paths, run_id="run-three")
            write_json(matrix_path, current_matrix)
            plan_result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(locks_payload["clean_unit_locks_path"]),
                    "--output-dir",
                    str(tmp_path / "recheck"),
                    "--ledger",
                    str(ledger_path),
                    "--json",
                ]
            )
            plan_payload = json.loads(plan_result.stdout)
            plan_path = Path(plan_payload["artifact_paths"]["recheck_plan_json"])
            empty_reports = tmp_path / "empty-reports"
            empty_reports.mkdir()
            verify_result = run_document_audit(
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

        self.assertEqual(locks_payload["settled_unit_count"], 1)
        self.assertEqual(plan_payload["summary"]["settled_unit_count"], 1)
        self.assertEqual(plan_payload["summary"]["active_unit_count"], 0)
        self.assertEqual(plan_payload["summary"]["waiting_unit_count"], 0)
        self.assertTrue(verification["complete"])
        self.assertTrue(verification["final_coverage_complete"])
        self.assertEqual(verification["settled_row_count"], 1)
        self.assertEqual(verification["status_counts"]["fail"], 1)

    def test_invalid_lock_artifact_requires_full_fresh_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = create_inputs(tmp_path)
            matrix_path = tmp_path / "checklist-matrix.json"
            invalid_locks = tmp_path / "clean-unit-locks.json"
            write_json(matrix_path, matrix_payload(paths))
            write_json(invalid_locks, {"schema_version": 999, "workflow_id": "document-audit"})

            result = run_document_audit(
                [
                    "recheck-plan",
                    "--matrix",
                    str(matrix_path),
                    "--prior-clean-unit-locks",
                    str(invalid_locks),
                    "--output-dir",
                    str(tmp_path / "recheck"),
                    "--ledger",
                    str(tmp_path / "current.json"),
                    "--json",
                ]
            )
            payload = json.loads(result.stdout)
            plan = json.loads(Path(payload["artifact_paths"]["recheck_plan_json"]).read_text(encoding="utf-8"))

        self.assertFalse(payload["summary"]["lock_validation_available"])
        self.assertEqual(payload["summary"]["active_unit_count"], 3)
        self.assertTrue(plan["complete_fresh_reports_required"])
        self.assertTrue(
            all("clean_unit_lock_validation_unavailable" in unit["reason_codes"] for unit in plan["units"])
        )


if __name__ == "__main__":
    unittest.main()
