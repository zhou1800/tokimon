from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SkillContractLanguageTests(unittest.TestCase):
    def test_product_development_keeps_pair_explicit_opt_in(self) -> None:
        product_development = (ROOT / "skills" / "product-development" / "skill.md").read_text(
            encoding="utf-8"
        )
        pair = (ROOT / "skills" / "pair" / "skill.md").read_text(encoding="utf-8")

        self.assertIn("only when the human explicitly opted into Pair collaboration", product_development)
        self.assertIn("Use this skill only when the human explicitly invokes Pair.", pair)
        self.assertNotIn("Use: ChatGPT + `pair` skill", product_development)

    def test_code_audit_skills_require_complete_accepted_dispositions(self) -> None:
        audit_text = (ROOT / "skills" / "code-audit" / "skill.md").read_text(
            encoding="utf-8"
        )
        fix_text = (ROOT / "skills" / "code-audit-fix" / "skill.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("accepted disposition for every freshly audited row", audit_text)
        self.assertIn("accepted `pass` and `not_applicable` rows", audit_text)
        self.assertIn("- `run_id`", audit_text)
        self.assertIn("- `scope_id`", audit_text)
        self.assertIn("accepted-dispositions artifact for every current row", fix_text)
        self.assertNotIn("records durable `fail` and `needs_human_review` rows only", fix_text)

    def test_code_audit_scope_includes_untracked_repository_additions(self) -> None:
        audit_text = (ROOT / "skills" / "code-audit" / "skill.md").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "including untracked additions in the requested worktree",
            audit_text,
        )
        self.assertNotIn("Audit tracked implementation code and tests only", audit_text)


if __name__ == "__main__":
    unittest.main()
