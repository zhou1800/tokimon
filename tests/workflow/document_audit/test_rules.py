from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.document_audit import rules


class DocumentAuditRulesTests(unittest.TestCase):
    def test_extracts_live_guideline_rule_ids(self) -> None:
        payload = rules.rules_payload()
        rule_ids = {rule["rule_id"] for rule in payload["rules"]}

        self.assertGreater(payload["rule_count"], 40)
        self.assertIn("DOC-OWN-001", rule_ids)
        self.assertIn("DOC-CLEAR-006", rule_ids)
        self.assertIn("DOC-REF-001", rule_ids)
        self.assertIn("DOC-MACHINE-001", rule_ids)
        self.assertIn("C4-L1-DEP-001", rule_ids)
        self.assertIn("C4-L2-INTERFACE-001", rule_ids)
        self.assertIn("C4-L3-VERIFY-002", rule_ids)
        self.assertEqual(payload["rule_count"], len(rule_ids))
        for rule in payload["rules"]:
            self.assertRegex(rule["rule_hash"], r"^[a-f0-9]{64}$")
            self.assertGreater(rule["line"], 0)

    def test_can_extract_guideline_rules_from_git_ref(self) -> None:
        payload = rules.rules_payload([Path("docs/guidelines/document-guidelines.md")], source_ref="HEAD")

        self.assertEqual(payload["source_ref"], "HEAD")
        self.assertEqual(payload["rule_sources"], ["docs/guidelines/document-guidelines.md"])

    def test_guideline_top_level_bullets_have_rule_ids(self) -> None:
        guideline_paths = (
            ROOT / "docs" / "guidelines" / "document-guidelines.md",
            ROOT / "docs" / "guidelines" / "c4-document-guidelines.md",
        )
        unlabeled: list[str] = []
        for path in guideline_paths:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if line.startswith("- ") and not rules.RULE_RE.match(line):
                    unlabeled.append(f"{path.relative_to(ROOT)}:{line_number}:{line}")

        self.assertEqual(unlabeled, [])

    def test_duplicate_rule_ids_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.md"
            path.write_text(
                "\n".join(
                    (
                        "- **DOC-TEST-001:** first.",
                        "- **DOC-TEST-001:** duplicate.",
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate rule ids"):
                rules.extract_rules([path])

    def test_continuation_lines_are_part_of_rule_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.md"
            path.write_text(
                "\n".join(
                    (
                        "- **DOC-TEST-001:** Keep the original rule text.",
                        "  Supporting explanation belongs to the same rule.",
                        "  Do not rewrite it to make extraction easier.",
                        "",
                        "**Next section**",
                        "- **DOC-TEST-002:** Next rule.",
                    )
                ),
                encoding="utf-8",
            )

            extracted = rules.extract_rules([path])

        self.assertEqual(
            extracted[0]["description"],
            "Keep the original rule text. Supporting explanation belongs to the same rule. Do not rewrite it to make extraction easier.",
        )
        self.assertEqual(extracted[1]["description"], "Next rule.")


if __name__ == "__main__":
    unittest.main()
