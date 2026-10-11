from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.document_audit import cli, contract


L2_DOC = ROOT / "docs" / "c4" / "level-2-containers" / "workflow" / "document-audit.md"
L3_DOC = ROOT / "docs" / "c4" / "level-3-components" / "workflow" / "document-audit-cli.md"
SKILL_DOC = ROOT / "skills" / "document-audit" / "skill.md"
FIX_SKILL_DOC = ROOT / "skills" / "document-audit-fix" / "skill.md"


class DocumentAuditWorkflowContractTests(unittest.TestCase):
    def test_contract_declarations_match_tokimon_package(self) -> None:
        self.assertEqual(contract.WORKFLOW_ID, "document-audit")
        self.assertEqual(contract.SKILL_ID, "document-audit")
        self.assertEqual(contract.PYTHON_PACKAGE, "tokimon.workflow.document_audit")
        self.assertEqual(contract.WORKFLOW_STATUS, "implemented")
        self.assertEqual(contract.CANONICAL_ENTRYPOINT, "tokimon.workflow.document_audit.cli:main")
        self.assertEqual(contract.OPERATOR_ENTRYPOINT, ".venv/bin/python -m tokimon.workflow.document_audit")
        self.assertEqual(contract.LOCAL_STATE_ROOT, Path(".tokimon") / "workflow" / "document-audit")
        self.assertEqual(contract.RUN_ROOT, contract.LOCAL_STATE_ROOT / "runs")
        self.assertEqual(contract.CURRENT_LEDGER_PATH, contract.LOCAL_STATE_ROOT / "ledger" / "current.json")
        self.assertEqual(
            contract.COMMANDS,
            (
                "contract",
                "rules",
                "matrix",
                "clean-unit-locks",
                "recheck-plan",
                "verify-reports",
                "compile-reports",
                "ledger-summary",
                "record-findings",
            ),
        )
        self.assertEqual(
            contract.FINDING_STATUSES,
            ("open", "fixed", "deferred", "rejected", "blocked", "stale"),
        )
        self.assertEqual(contract.TERMINAL_FINDING_STATUSES, ("deferred", "rejected", "blocked"))
        self.assertEqual(contract.FINDING_SEVERITIES, ("must_fix", "should_fix", "optional"))
        self.assertEqual(
            contract.REQUIRED_FINDING_FIELDS,
            ("rule_id", "file_path", "summary", "severity", "status", "evidence"),
        )
        self.assertEqual(contract.EVIDENCE_LINE_FIELDS, ("line", "start_line"))
        self.assertEqual(contract.EVIDENCE_TEXT_FIELDS, ("text", "excerpt", "quote", "details"))
        self.assertEqual(contract.CHECKLIST_STATUSES, ("pass", "fail", "not_applicable", "unchecked"))
        self.assertEqual(contract.AUDIT_REPORT_STATUSES, ("pass", "fail", "not_applicable"))
        self.assertEqual(contract.CURRENT_LEDGER_SCHEMA_VERSION, 3)
        self.assertEqual(contract.LEGACY_LEDGER_SCHEMA_VERSIONS, (1, 2))
        self.assertEqual(contract.RUN_ARTIFACT_SCHEMA_VERSION, 3)
        self.assertEqual(contract.REPORT_DIR_NAME, "reports")
        self.assertEqual(contract.AUDIT_UNITS_NAME, "audit-units.json")
        self.assertEqual(contract.CLEAN_UNIT_LOCKS_NAME, "clean-unit-locks.json")
        self.assertEqual(contract.RECHECK_PLAN_NAME, "recheck-plan.json")
        self.assertFalse(hasattr(contract, "CANDIDATE_ISSUES_NAME"))
        self.assertFalse(hasattr(contract, "COMPILED_FINDINGS_NAME"))
        self.assertEqual(contract.CHECKLIST_MATRIX_SCHEMA_VERSION, 3)
        self.assertEqual(contract.LEGACY_CHECKLIST_MATRIX_SCHEMA_VERSIONS, (1, 2))
        self.assertEqual(contract.AUDIT_REPORT_SCHEMA_VERSION, 3)
        self.assertEqual(contract.LEGACY_AUDIT_REPORT_SCHEMA_VERSIONS, (1, 2))
        self.assertEqual(contract.AUDIT_UNITS_SCHEMA_VERSION, 1)
        self.assertEqual(contract.CLEAN_UNIT_LOCKS_SCHEMA_VERSION, 1)
        self.assertEqual(contract.RECHECK_PLAN_SCHEMA_VERSION, 1)
        self.assertEqual(contract.WORKFLOW_READINESS["modes"][0]["id"], "matrix-reports-ledger")

    def test_docs_and_skill_name_required_contract_surfaces(self) -> None:
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (L2_DOC, L3_DOC, SKILL_DOC, FIX_SKILL_DOC)
        )

        for required_text in (
            "Workflow id:** `document-audit`",
            "Consumer skills:** `document-audit`, `document-audit-fix`",
            "Python package:** `tokimon/workflow/document_audit/`",
            "tokimon.workflow.document_audit.cli:main",
            ".venv/bin/python -m tokimon.workflow.document_audit",
            "tokimon/workflow/document_audit/contract.py",
            ".tokimon/workflow/document-audit/projects/<project-id>/ledger/current.json",
            "matrix",
            "checklist-matrix.json",
            "checklist-summary.json",
            "verify-reports",
            "compile-reports",
            "evaluations-from-reports.json",
            "audit-units.json",
            "clean-unit-locks.json",
            "recheck-plan.json",
            "record-findings",
            "rule_id",
            "fingerprint",
            "unittest discover -s tests/workflow/document_audit",
            "skills/document-audit-fix/skill.md",
            "fresh re-audit",
        ):
            self.assertIn(required_text, combined)

    def test_cli_contract_payload(self) -> None:
        payload = cli._contract_payload()

        self.assertEqual(payload["workflow_id"], "document-audit")
        self.assertEqual(payload["operator_entrypoint"], ".venv/bin/python -m tokimon.workflow.document_audit")
        self.assertEqual(payload["json_encoding"], "UTF-8")
        self.assertFalse(payload["json_ensure_ascii"])
        self.assertEqual(
            payload["commands"],
            [
                "contract",
                "rules",
                "matrix",
                "clean-unit-locks",
                "recheck-plan",
                "verify-reports",
                "compile-reports",
                "ledger-summary",
                "record-findings",
            ],
        )
        self.assertEqual(
            payload["required_finding_fields"],
            ["rule_id", "file_path", "summary", "severity", "status", "evidence"],
        )
        self.assertEqual(payload["evidence_line_fields"], ["line", "start_line"])
        self.assertEqual(payload["evidence_text_fields"], ["text", "excerpt", "quote", "details"])
        self.assertEqual(payload["checklist_statuses"], ["pass", "fail", "not_applicable", "unchecked"])
        self.assertEqual(payload["audit_report_statuses"], ["pass", "fail", "not_applicable"])
        self.assertEqual(payload["terminal_finding_statuses"], ["deferred", "rejected", "blocked"])
        self.assertEqual(payload["current_ledger_schema_version"], 3)
        self.assertEqual(payload["legacy_ledger_schema_versions"], [1, 2])
        self.assertEqual(payload["run_artifact_schema_version"], 3)
        self.assertEqual(payload["report_dir_name"], "reports")
        self.assertEqual(payload["audit_units_name"], "audit-units.json")
        self.assertEqual(payload["clean_unit_locks_name"], "clean-unit-locks.json")
        self.assertEqual(payload["recheck_plan_name"], "recheck-plan.json")
        self.assertEqual(payload["audit_units_schema_version"], 1)
        self.assertEqual(payload["checklist_matrix_schema_version"], 3)
        self.assertEqual(payload["legacy_checklist_matrix_schema_versions"], [1, 2])
        self.assertEqual(payload["audit_report_schema_version"], 3)
        self.assertEqual(payload["legacy_audit_report_schema_versions"], [1, 2])
        self.assertEqual(payload["clean_unit_locks_schema_version"], 1)
        self.assertEqual(payload["recheck_plan_schema_version"], 1)
        self.assertNotIn("candidate_issues_name", payload)
        self.assertNotIn("compiled_findings_name", payload)
        self.assertEqual(payload["workflow_readiness"]["id"], "document-audit")
        self.assertEqual(payload["workflow_readiness"]["modes"][0]["id"], "matrix-reports-ledger")


if __name__ == "__main__":
    unittest.main()
