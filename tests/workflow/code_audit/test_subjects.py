from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.code_audit import subjects


def write_subjects(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"subjects": entries}), encoding="utf-8")


def subject_entry(subject_id: str = "tokimon.workflow.document_audit") -> dict[str, object]:
    return {
        "audit_subject_id": subject_id,
        "title": "Document Audit workflow",
        "implementation_paths": ["tokimon/workflow/document_audit"],
        "test_paths": ["tests/workflow/document_audit"],
        "owning_specs": ["docs/c4/level-3-components/workflow/document-audit-cli.md"],
        "verification_commands": [
            ".venv/bin/python -m unittest discover -s tests/workflow/document_audit -t ."
        ],
        "dependency_edges": [{"kind": "reads", "target": "docs/guidelines/document-guidelines.md"}],
    }


class CodeAuditSubjectsTests(unittest.TestCase):
    def test_load_subjects_validates_and_normalizes_caller_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_file = Path(tmp) / "subjects.json"
            write_subjects(subject_file, [subject_entry()])

            loaded = subjects.load_subjects(subject_file)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["audit_subject_id"], "tokimon.workflow.document_audit")
        self.assertEqual(loaded[0]["implementation_paths"], ["tokimon/workflow/document_audit"])
        self.assertEqual(loaded[0]["dependency_edges"][0]["kind"], "reads")

    def test_duplicate_subject_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_file = Path(tmp) / "subjects.json"
            write_subjects(subject_file, [subject_entry(), subject_entry()])

            with self.assertRaisesRegex(ValueError, "duplicate audit_subject_id"):
                subjects.load_subjects(subject_file)

    def test_subject_filter_rejects_unknown_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_file = Path(tmp) / "subjects.json"
            write_subjects(subject_file, [subject_entry()])

            with self.assertRaisesRegex(ValueError, "unknown audit_subject_id"):
                subjects.load_subjects(subject_file, selected_ids=["tokimon.workflow.missing"])

    def test_subject_can_use_owner_doc_as_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_file = Path(tmp) / "subjects.json"
            write_subjects(
                subject_file,
                [
                    {
                        "audit_subject_id": "tokimon.workflow.document_audit",
                        "title": "Document Audit workflow",
                        "subject_source": "owner_doc_default",
                        "owning_specs": ["docs/c4/level-3-components/workflow/document-audit-cli.md"],
                    }
                ],
            )

            loaded = subjects.load_subjects(subject_file)

        self.assertEqual(loaded[0]["implementation_paths"], [])
        self.assertEqual(loaded[0]["owning_specs"], ["docs/c4/level-3-components/workflow/document-audit-cli.md"])
        self.assertEqual(loaded[0]["subject_source"], "owner_doc_default")

    def test_missing_boundary_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_file = Path(tmp) / "subjects.json"
            entry = subject_entry()
            for field in (
                "implementation_paths",
                "test_paths",
                "owning_specs",
                "verification_commands",
                "dependency_edges",
            ):
                entry.pop(field, None)
            write_subjects(subject_file, [entry])

            with self.assertRaisesRegex(ValueError, "needs at least one boundary field"):
                subjects.load_subjects(subject_file)

    def test_missing_subject_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_file = Path(tmp) / "subjects.json"
            entry = subject_entry()
            entry["implementation_paths"] = ["tokimon/workflow/missing_code_audit_subject"]
            write_subjects(subject_file, [entry])

            with self.assertRaisesRegex(ValueError, "path does not exist"):
                subjects.load_subjects(subject_file)


if __name__ == "__main__":
    unittest.main()
