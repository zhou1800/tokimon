from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.code_audit import ledger


def evidence() -> list[dict[str, object]]:
    return [{"path": "tokimon/workflow/document_audit/cli.py", "line_start": 10, "excerpt": "contract"}]


class CodeAuditLedgerTests(unittest.TestCase):
    def test_recording_rejects_unsafe_run_or_artifact_without_mutation(self) -> None:
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
            safe = runs / "safe"
            safe.mkdir()
            (safe / "findings.json").symlink_to(marker)
            ledger_path = base / "ledger.json"
            for run_id in ("../victim", str(victim), "linked", "loop", "safe"):
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    ledger.record_evaluations(
                        evaluations=[], scope_id="scope", run_id=run_id,
                        ledger_path=ledger_path, run_root=runs,
                    )
                self.assertEqual(marker.read_text(), "unchanged")
                self.assertFalse(ledger_path.exists())
                self.assertEqual(list(victim.iterdir()), [marker])

    def test_later_pass_closes_prior_fail_for_the_same_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "current.json"
            run_root = tmp_path / "runs"
            rules = {"CODE-CONTRACT-001": {"rule_id": "CODE-CONTRACT-001", "rule_hash": "abc"}}
            ledger.record_evaluations(
                evaluations=[
                    {
                        "run_id": "run-1",
                        "scope_id": "tokimon-workflows",
                        "rule_id": "CODE-CONTRACT-001",
                        "audit_subject_id": "tokimon.workflow.document_audit",
                        "status": "fail",
                        "summary": "Missing contract coverage.",
                        "evidence": evidence(),
                    }
                ],
                scope_id="tokimon-workflows",
                run_id="run-1",
                ledger_path=ledger_path,
                run_root=run_root,
                rules_by_id=rules,
            )

            result = ledger.record_evaluations(
                evaluations=[
                    {
                        "run_id": "run-2",
                        "scope_id": "tokimon-workflows",
                        "rule_id": "CODE-CONTRACT-001",
                        "audit_subject_id": "tokimon.workflow.document_audit",
                        "status": "pass",
                        "summary": "Contract coverage is present.",
                        "reason": "The owner contract and tests now match.",
                        "verification": ["unit tests"],
                    }
                ],
                scope_id="tokimon-workflows",
                run_id="run-2",
                ledger_path=ledger_path,
                run_root=run_root,
                rules_by_id=rules,
            )

            current = json.loads(ledger_path.read_text(encoding="utf-8"))
            entry = next(iter(current["entries"].values()))
            run_payload = json.loads((run_root / "run-2" / "findings.json").read_text(encoding="utf-8"))

        self.assertEqual(entry["status"], "fixed")
        self.assertEqual(entry["resolved_by_run"], "run-2")
        self.assertEqual(result["finding_count"], 0)
        self.assertEqual(result["resolution_count"], 1)
        self.assertEqual(result["stats"]["open_total"], 0)
        self.assertEqual(result["stats"]["resolved_total"], 1)
        self.assertEqual(run_payload["disposition_count"], 1)

    def test_evaluation_provenance_must_match_recording_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "run_id mismatch"):
                ledger.record_evaluations(
                    evaluations=[
                        {
                            "run_id": "wrong-run",
                            "scope_id": "tokimon-workflows",
                            "rule_id": "CODE-CONTRACT-001",
                            "audit_subject_id": "tokimon.workflow.document_audit",
                            "status": "pass",
                            "summary": "Verified.",
                            "verification": ["unit tests"],
                        }
                    ],
                    scope_id="tokimon-workflows",
                    run_id="run-1",
                    ledger_path=Path(tmp) / "current.json",
                    run_root=Path(tmp) / "runs",
                    rules_by_id={"CODE-CONTRACT-001": {"rule_id": "CODE-CONTRACT-001"}},
                )

    def test_duplicate_accepted_disposition_rows_are_rejected_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "current.json"
            with self.assertRaisesRegex(ValueError, "duplicate accepted disposition row"):
                ledger.record_evaluations(
                    evaluations=[
                        {
                            "run_id": "run-1",
                            "scope_id": "tokimon-workflows",
                            "rule_id": "CODE-CONTRACT-001",
                            "audit_subject_id": "tokimon.workflow.document_audit",
                            "status": "pass",
                            "summary": "Verified once.",
                            "verification": ["unit tests"],
                        },
                        {
                            "run_id": "run-1",
                            "scope_id": "tokimon-workflows",
                            "rule_id": "CODE-CONTRACT-001",
                            "audit_subject_id": "tokimon.workflow.document_audit",
                            "status": "not_applicable",
                            "summary": "Conflicting second disposition.",
                            "reason": "The same row cannot be accepted twice.",
                        },
                    ],
                    scope_id="tokimon-workflows",
                    run_id="run-1",
                    ledger_path=ledger_path,
                    run_root=tmp_path / "runs",
                    rules_by_id={
                        "CODE-CONTRACT-001": {"rule_id": "CODE-CONTRACT-001"}
                    },
                )

            self.assertFalse(ledger_path.exists())
            self.assertFalse((tmp_path / "runs" / "run-1").exists())

    def test_supplied_fingerprint_cannot_cross_ledger_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "current.json"
            run_root = tmp_path / "runs"
            rules = {
                "CODE-CONTRACT-001": {
                    "rule_id": "CODE-CONTRACT-001",
                    "rule_hash": "abc",
                }
            }
            first = ledger.record_evaluations(
                evaluations=[
                    {
                        "run_id": "run-1",
                        "scope_id": "tokimon-workflows",
                        "rule_id": "CODE-CONTRACT-001",
                        "audit_subject_id": "tokimon.workflow.first",
                        "status": "fail",
                        "summary": "First row finding.",
                        "evidence": evidence(),
                    }
                ],
                scope_id="tokimon-workflows",
                run_id="run-1",
                ledger_path=ledger_path,
                run_root=run_root,
                rules_by_id=rules,
            )
            fingerprint = first["updated_fingerprints"][0]
            before = ledger_path.read_text(encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
                ledger.record_evaluations(
                    evaluations=[
                        {
                            "run_id": "run-2",
                            "scope_id": "tokimon-workflows",
                            "rule_id": "CODE-CONTRACT-001",
                            "audit_subject_id": "tokimon.workflow.second",
                            "status": "fail",
                            "summary": "Second row finding.",
                            "evidence": evidence(),
                            "fingerprint": fingerprint,
                        }
                    ],
                    scope_id="tokimon-workflows",
                    run_id="run-2",
                    ledger_path=ledger_path,
                    run_root=run_root,
                    rules_by_id=rules,
                )

            self.assertEqual(ledger_path.read_text(encoding="utf-8"), before)
            self.assertFalse((run_root / "run-2").exists())

    def test_record_evaluations_records_fail_and_needs_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "current.json"
            run_root = tmp_path / "runs"
            result = ledger.record_evaluations(
                evaluations=[
                    {
                        "run_id": "run-1",
                        "scope_id": "tokimon-workflows",
                        "rule_id": "CODE-CONTRACT-001",
                        "audit_subject_id": "tokimon.workflow.document_audit",
                        "status": "pass",
                        "summary": "Verified.",
                        "verification": ["unit tests"],
                    },
                    {
                        "run_id": "run-1",
                        "scope_id": "tokimon-workflows",
                        "rule_id": "CODE-CONTRACT-001",
                        "audit_subject_id": "tokimon.workflow.pair",
                        "status": "fail",
                        "summary": "Missing contract coverage.",
                        "evidence": evidence(),
                    },
                    {
                        "run_id": "run-1",
                        "scope_id": "tokimon-workflows",
                        "rule_id": "CODE-STRUCT-001",
                        "audit_subject_id": "tokimon.workflow.pair",
                        "status": "needs_human_review",
                        "summary": "Boundary needs review.",
                        "reason": "Semantic dependency direction.",
                        "evidence": evidence(),
                        "review_prompt": "Confirm dependency direction.",
                    },
                ],
                scope_id="tokimon-workflows",
                run_id="run-1",
                ledger_path=ledger_path,
                run_root=run_root,
                rules_by_id={
                    "CODE-CONTRACT-001": {"rule_id": "CODE-CONTRACT-001", "rule_hash": "abc"},
                    "CODE-STRUCT-001": {"rule_id": "CODE-STRUCT-001", "rule_hash": "def"},
                },
            )

            self.assertEqual(result["finding_count"], 2)
            current = json.loads(ledger_path.read_text(encoding="utf-8"))
            entries = list(current["entries"].values())
            self.assertEqual(len(entries), 2)
            self.assertEqual(current["stats"]["by_status"]["fail"], 1)
            self.assertEqual(current["stats"]["by_status"]["needs_human_review"], 1)
            self.assertIn("evidence", entries[0])
            run_payload = json.loads((run_root / "run-1" / "findings.json").read_text(encoding="utf-8"))
            self.assertEqual(run_payload["finding_count"], 2)

    def test_unknown_rule_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unknown rule_id"):
                ledger.record_evaluations(
                    evaluations=[
                        {
                            "run_id": "run-1",
                            "scope_id": "tokimon-workflows",
                            "rule_id": "CODE-MISSING-001",
                            "audit_subject_id": "tokimon.workflow.document_audit",
                            "status": "fail",
                            "summary": "Missing rule.",
                            "evidence": evidence(),
                        }
                    ],
                    scope_id="tokimon-workflows",
                    run_id="run-1",
                    ledger_path=Path(tmp) / "current.json",
                    run_root=Path(tmp) / "runs",
                    rules_by_id={},
                )

    def test_needs_human_review_requires_prompt_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "review_prompt"):
                ledger.record_evaluations(
                    evaluations=[
                        {
                            "run_id": "run-1",
                            "scope_id": "tokimon-workflows",
                            "rule_id": "CODE-STRUCT-001",
                            "audit_subject_id": "tokimon.workflow.document_audit",
                            "status": "needs_human_review",
                            "summary": "Needs review.",
                            "evidence": evidence(),
                        }
                    ],
                    scope_id="tokimon-workflows",
                    run_id="run-1",
                    ledger_path=Path(tmp) / "current.json",
                    run_root=Path(tmp) / "runs",
                    rules_by_id={"CODE-STRUCT-001": {"rule_id": "CODE-STRUCT-001"}},
                )

    def test_unchecked_is_rejected_in_evaluations(self) -> None:
        with self.assertRaisesRegex(ValueError, "unchecked is not allowed"):
            ledger.validate_evaluation(
                {
                    "run_id": "run-1",
                    "scope_id": "tokimon-workflows",
                    "rule_id": "CODE-CONTRACT-001",
                    "audit_subject_id": "tokimon.workflow.document_audit",
                    "status": "unchecked",
                }
            )

    def test_run_artifact_write_failure_does_not_update_current_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "current.json"
            run_root = tmp_path / "runs-as-file"
            run_root.write_text("not a directory", encoding="utf-8")

            with self.assertRaises(OSError):
                ledger.record_evaluations(
                    evaluations=[
                        {
                            "run_id": "run-1",
                            "scope_id": "tokimon-workflows",
                            "rule_id": "CODE-CONTRACT-001",
                            "audit_subject_id": "tokimon.workflow.document_audit",
                            "status": "fail",
                            "summary": "Missing contract coverage.",
                            "evidence": evidence(),
                        }
                    ],
                    scope_id="tokimon-workflows",
                    run_id="run-1",
                    ledger_path=ledger_path,
                    run_root=run_root,
                    rules_by_id={"CODE-CONTRACT-001": {"rule_id": "CODE-CONTRACT-001"}},
                )

            self.assertFalse(ledger_path.exists())


if __name__ == "__main__":
    unittest.main()
