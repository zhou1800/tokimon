from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.code_audit import rules


class CodeAuditRulesTests(unittest.TestCase):
    def test_live_guideline_rule_ids_extract(self) -> None:
        payload = rules.rules_payload()

        self.assertGreater(payload["rule_count"], 0)
        rule_ids = {rule["rule_id"] for rule in payload["rules"]}
        self.assertIn("CODE-CONTRACT-001", rule_ids)
        self.assertIn("CODE-VERIFY-005", rule_ids)
        self.assertIn("UNIT-FAST-001", rule_ids)
        self.assertIn("UNIT-GUARD-006", rule_ids)
        self.assertEqual(
            payload["rule_sources"],
            [
                "docs/guidelines/code-guidelines.md",
                "docs/guidelines/unit-testing-guidelines.md",
            ],
        )

    def test_guideline_rule_bullets_have_rule_ids(self) -> None:
        guideline_ranges = (
            (
                ROOT / "docs" / "guidelines" / "code-guidelines.md",
                (("## Engineering Principles", "## References"),),
            ),
            (
                ROOT / "docs" / "guidelines" / "unit-testing-guidelines.md",
                (
                    ("## Purpose & Scope", "## Fast Feedback Workflow"),
                    ("## Guardrails", "## Acceptance Checklist"),
                ),
            ),
        )
        unlabeled: list[str] = []
        for guideline_path, ranges in guideline_ranges:
            lines = guideline_path.read_text(encoding="utf-8").splitlines()
            for start_heading, end_heading in ranges:
                in_rules = False
                for line_number, line in enumerate(lines, start=1):
                    if line == start_heading:
                        in_rules = True
                        continue
                    if line == end_heading:
                        in_rules = False
                    if in_rules and line.startswith("- ") and not rules.RULE_RE.match(line):
                        unlabeled.append(f"{guideline_path.relative_to(ROOT)}:{line_number}:{line}")

        self.assertEqual(unlabeled, [])

    def test_duplicate_rule_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "rules.md"
            source.write_text(
                "\n".join(
                    (
                        "- **CODE-TEST-001:** First rule.",
                        "- **CODE-TEST-001:** Duplicate rule.",
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate rule ids: CODE-TEST-001"):
                rules.extract_rules([source])

    def test_rule_hash_is_stable(self) -> None:
        self.assertEqual(rules.rule_hash("CODE-TEST-001", "A rule."), rules.rule_hash("CODE-TEST-001", "A rule."))


if __name__ == "__main__":
    unittest.main()
