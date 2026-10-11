from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
C4_ROOT = ROOT / "docs" / "c4"


class C4ContractLanguageTests(unittest.TestCase):
    def test_c4_guidelines_name_the_canonical_plural_roots(self) -> None:
        guideline = (ROOT / "docs" / "guidelines" / "c4-document-guidelines.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("docs/c4/level-2-containers/<capability>.md", guideline)
        self.assertIn("docs/c4/level-3-components/<capability>/*.md", guideline)
        self.assertNotIn("docs/c4/level-2-container/<capability>", guideline)
        self.assertNotIn("docs/c4/level-3-component/<capability>", guideline)

    def test_c4_docs_do_not_use_migration_progress_language(self) -> None:
        prohibited_phrases = (
            "Implementation status:",
            "planned target contract",
            "post-implementation",
            "unless the referenced implementation root exists",
            "unless the referenced adapter package root exists",
            "unless the referenced `skills/` root exists",
            "planned adapter",
            "planned package source",
            "planned local account package",
            "planned in-checkout source root",
            "planned `skills/` root",
            "planned in this repository",
            "planned Codex adapter",
        )
        violations: list[str] = []

        for path in sorted(C4_ROOT.rglob("*.md")):
            relative_path = path.relative_to(ROOT)
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                for phrase in prohibited_phrases:
                    if phrase in line:
                        violations.append(f"{relative_path}:{line_number}: {phrase}")

        self.assertEqual(violations, [])

    def test_c4_docs_use_tokimon_for_product_identity(self) -> None:
        violations: list[str] = []

        for path in sorted(C4_ROOT.rglob("*.md")):
            relative_path = path.relative_to(ROOT)
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                remainder = line.replace("AI Harness Interface", "")
                if "AI Harness" in remainder:
                    violations.append(f"{relative_path}:{line_number}: {line.strip()}")

        self.assertEqual(violations, [])

if __name__ == "__main__":
    unittest.main()
