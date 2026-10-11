from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.code_audit import contract


class CodeAuditContractTests(unittest.TestCase):
    def test_contract_declarations(self) -> None:
        self.assertEqual(contract.WORKFLOW_ID, "code-audit")
        self.assertEqual(contract.SKILL_ID, "code-audit")
        self.assertEqual(contract.OPERATOR_ENTRYPOINT, ".venv/bin/python -m tokimon.workflow.code_audit")
        self.assertEqual(contract.CURRENT_LEDGER_PATH.as_posix(), ".tokimon/workflow/code-audit/ledger/current.json")
        self.assertEqual(
            contract.COMMANDS,
            (
                "contract",
                "rules",
                "subjects",
                "matrix",
                "pass-locks",
                "recheck-plan",
                "verify-reports",
                "compile-reports",
                "record-findings",
                "ledger-summary",
            ),
        )
        self.assertEqual(
            contract.AUDIT_STATUSES,
            ("pass", "fail", "not_applicable", "needs_human_review", "unchecked"),
        )
        self.assertEqual(contract.AUDIT_REPORT_STATUSES, ("pass", "fail", "not_applicable", "needs_human_review"))
        self.assertEqual(contract.DURABLE_RECORD_STATUSES, ("fail", "needs_human_review"))
        self.assertEqual(contract.RESOLVED_LEDGER_STATUSES, ("fixed", "stale"))
        self.assertEqual(contract.REQUIRED_SUBJECT_FIELDS, ("audit_subject_id", "title"))
        self.assertIn("implementation_paths", contract.SUBJECT_BOUNDARY_FIELDS)
        self.assertNotIn("implementation_paths", contract.OPTIONAL_SUBJECT_FIELDS)
        self.assertEqual(contract.REPORT_DIR_NAME, "reports")
        self.assertEqual(contract.PASS_LOCKS_NAME, "pass-locks.json")
        self.assertEqual(contract.RECHECK_PLAN_NAME, "recheck-plan.json")
        self.assertFalse(hasattr(contract, "CANDIDATE_ISSUES_NAME"))
        self.assertFalse(hasattr(contract, "COMPILED_FINDINGS_NAME"))
        self.assertEqual(contract.AUDIT_REPORT_SCHEMA_VERSION, 3)
        self.assertIn(Path("docs/guidelines/code-guidelines.md"), contract.GUIDELINE_RULE_SOURCES)
        self.assertIn(Path("docs/guidelines/unit-testing-guidelines.md"), contract.GUIDELINE_RULE_SOURCES)
        self.assertEqual(
            contract.GUIDELINE_RULE_SOURCES,
            (
                Path("docs/guidelines/code-guidelines.md"),
                Path("docs/guidelines/unit-testing-guidelines.md"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
