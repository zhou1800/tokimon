from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.document_audit import ledger


class DocumentAuditLedgerTests(unittest.TestCase):
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
                    ledger.record_findings(
                        findings=[], scope_id="scope", run_id=run_id,
                        ledger_path=ledger_path, run_root=runs, rules_by_id={},
                    )
                self.assertEqual(marker.read_text(), "unchanged")
                self.assertFalse(ledger_path.exists())
                self.assertEqual(list(victim.iterdir()), [marker])

    def test_summary_reports_open_findings_for_missing_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "current.json"
            run_root = tmp_path / "runs"
            missing_path = tmp_path / "deleted.md"
            ledger.record_findings(
                findings=[
                    {
                        "rule_id": "DOC-OWN-001",
                        "file_path": str(missing_path),
                        "summary": "The deleted document still has an open finding.",
                        "severity": "must_fix",
                        "status": "open",
                        "evidence": [{"line": 1, "text": "deleted"}],
                    }
                ],
                scope_id="docs-example",
                run_id="run-1",
                ledger_path=ledger_path,
                run_root=run_root,
                rules_by_id={"DOC-OWN-001": {"rule_id": "DOC-OWN-001"}},
            )

            summary = ledger.summarize_ledger(ledger_path, scope_id="docs-example")

        self.assertFalse(summary["health"]["healthy"])
        self.assertEqual(summary["health"]["missing_file_count"], 1)
        self.assertEqual(summary["health"]["open_missing_file_count"], 1)
        self.assertEqual(summary["health"]["affected_entries"][0]["file_path"], str(missing_path))

    def test_record_findings_updates_compact_current_ledger_and_run_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "current.json"
            run_root = tmp_path / "runs"
            rules_by_id = {
                "DOC-OWN-001": {
                    "rule_id": "DOC-OWN-001",
                    "rule_hash": "abc123",
                }
            }

            result = ledger.record_findings(
                findings=[
                    {
                        "rule_id": "DOC-OWN-001",
                        "file_path": "docs/example.md",
                        "summary": "Duplicated durable ownership statement.",
                        "severity": "must_fix",
                        "status": "open",
                        "recommended_action": "Move the repeated fact to the owner.",
                        "evidence": [{"line": 12, "text": "duplicate"}],
                    }
                ],
                scope_id="docs-example",
                run_id="run-1",
                ledger_path=ledger_path,
                run_root=run_root,
                rules_by_id=rules_by_id,
            )

            self.assertEqual(result["finding_count"], 1)
            self.assertEqual(result["stats"]["open_must_fix"], 1)
            current = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(current["schema_version"], 3)
            self.assertEqual(current["stats"]["open_must_fix"], 1)
            entries = list(current["entries"].values())
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["rule_id"], "DOC-OWN-001")
            self.assertEqual(entries[0]["status"], "open")
            self.assertNotIn("evidence", entries[0])

            run_payload = json.loads((run_root / "run-1" / "findings.json").read_text(encoding="utf-8"))
            self.assertEqual(run_payload["schema_version"], 3)
            self.assertEqual(run_payload["finding_count"], 1)
            self.assertIn("evidence", run_payload["findings"][0])

            updated = ledger.record_findings(
                findings=[
                    {
                        "rule_id": "DOC-OWN-001",
                        "file_path": "docs/example.md",
                        "summary": "Duplicated durable ownership statement.",
                        "severity": "must_fix",
                        "status": "fixed",
                        "resolution": "Removed duplicate.",
                        "evidence": [{"line": 12, "text": "duplicate removed"}],
                    }
                ],
                scope_id="docs-example",
                run_id="run-2",
                ledger_path=ledger_path,
                run_root=run_root,
                rules_by_id=rules_by_id,
            )

            self.assertEqual(updated["stats"]["open_must_fix"], 0)
            summary = ledger.summarize_ledger(ledger_path, scope_id="docs-example")
            self.assertEqual(summary["stats"]["by_status"]["fixed"], 1)
            self.assertEqual(summary["stats"]["open_must_fix"], 0)

    def test_unknown_rule_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unknown rule_id"):
                ledger.record_findings(
                    findings=[
                        {
                            "rule_id": "DOC-MISSING-001",
                            "file_path": "docs/example.md",
                            "summary": "Missing rule.",
                            "severity": "optional",
                            "status": "open",
                            "evidence": [{"line": 12, "text": "missing rule"}],
                        }
                    ],
                    scope_id="docs-example",
                    run_id="run-1",
                    ledger_path=Path(tmp) / "current.json",
                    run_root=Path(tmp) / "runs",
                    rules_by_id={},
                )

    def test_duplicate_accepted_fingerprints_are_rejected_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "current.json"
            finding = {
                "rule_id": "DOC-OWN-001",
                "file_path": "docs/example.md",
                "summary": "One finding cannot have two accepted dispositions.",
                "severity": "should_fix",
                "status": "open",
                "evidence": [{"line": 12, "text": "duplicate"}],
            }

            with self.assertRaisesRegex(ValueError, "duplicate accepted finding fingerprint"):
                ledger.record_findings(
                    findings=[finding, dict(finding, status="fixed", resolution="Fixed.")],
                    scope_id="docs-example",
                    run_id="run-1",
                    ledger_path=ledger_path,
                    run_root=tmp_path / "runs",
                    rules_by_id={"DOC-OWN-001": {"rule_id": "DOC-OWN-001"}},
                )

            self.assertFalse(ledger_path.exists())
            self.assertFalse((tmp_path / "runs" / "run-1").exists())

    def test_supplied_fingerprint_cannot_cross_finding_owners(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "current.json"
            run_root = tmp_path / "runs"
            rules = {"DOC-OWN-001": {"rule_id": "DOC-OWN-001"}}
            first = ledger.record_findings(
                findings=[
                    {
                        "rule_id": "DOC-OWN-001",
                        "file_path": "docs/first.md",
                        "summary": "First finding.",
                        "severity": "should_fix",
                        "status": "open",
                        "evidence": [{"line": 1, "text": "first"}],
                    }
                ],
                scope_id="docs-example",
                run_id="run-1",
                ledger_path=ledger_path,
                run_root=run_root,
                rules_by_id=rules,
            )
            fingerprint = first["updated_fingerprints"][0]
            before = ledger_path.read_text(encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
                ledger.record_findings(
                    findings=[
                        {
                            "rule_id": "DOC-OWN-001",
                            "file_path": "docs/second.md",
                            "summary": "Second finding.",
                            "severity": "should_fix",
                            "status": "open",
                            "evidence": [{"line": 1, "text": "second"}],
                            "fingerprint": fingerprint,
                        }
                    ],
                    scope_id="docs-example",
                    run_id="run-2",
                    ledger_path=ledger_path,
                    run_root=run_root,
                    rules_by_id=rules,
                )

            self.assertEqual(ledger_path.read_text(encoding="utf-8"), before)
            self.assertFalse((run_root / "run-2").exists())

    def test_missing_exact_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "finding missing exact evidence"):
                ledger.record_findings(
                    findings=[
                        {
                            "rule_id": "DOC-OWN-001",
                            "file_path": "docs/example.md",
                            "summary": "No evidence.",
                            "severity": "optional",
                            "status": "open",
                            "evidence": [],
                        }
                    ],
                    scope_id="docs-example",
                    run_id="run-1",
                    ledger_path=Path(tmp) / "current.json",
                    run_root=Path(tmp) / "runs",
                    rules_by_id={"DOC-OWN-001": {"rule_id": "DOC-OWN-001"}},
                )

    def test_evidence_must_include_line_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for evidence, message in (
                ([{"text": "has text but no line"}], "missing positive line or start_line"),
                ([{"line": 12}], "missing text, excerpt, quote, or details"),
            ):
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        ledger.record_findings(
                            findings=[
                                {
                                    "rule_id": "DOC-OWN-001",
                                    "file_path": "docs/example.md",
                                    "summary": "Bad evidence.",
                                    "severity": "optional",
                                    "status": "open",
                                    "evidence": evidence,
                                }
                            ],
                            scope_id="docs-example",
                            run_id="run-1",
                            ledger_path=Path(tmp) / f"{message}.json",
                            run_root=Path(tmp) / "runs",
                            rules_by_id={"DOC-OWN-001": {"rule_id": "DOC-OWN-001"}},
                        )

    def test_run_artifact_write_failure_does_not_update_current_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "current.json"
            run_root = tmp_path / "runs-as-file"
            run_root.write_text("not a directory", encoding="utf-8")

            with self.assertRaises(OSError):
                ledger.record_findings(
                    findings=[
                        {
                            "rule_id": "DOC-OWN-001",
                            "file_path": "docs/example.md",
                            "summary": "Duplicated durable ownership statement.",
                            "severity": "should_fix",
                            "status": "open",
                            "evidence": [{"line": 12, "text": "duplicate"}],
                        }
                    ],
                    scope_id="docs-example",
                    run_id="run-1",
                    ledger_path=ledger_path,
                    run_root=run_root,
                    rules_by_id={"DOC-OWN-001": {"rule_id": "DOC-OWN-001"}},
                )

            self.assertFalse(ledger_path.exists())


if __name__ == "__main__":
    unittest.main()
