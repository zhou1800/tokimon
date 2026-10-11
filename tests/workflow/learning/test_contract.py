from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.learning import cli, contract


L2_DOC = ROOT / "docs" / "c4" / "level-2-containers" / "workflow" / "learning.md"
L3_DOC = ROOT / "docs" / "c4" / "level-3-components" / "workflow" / "learning-cli.md"
SKILL_DOC = ROOT / "skills" / "learning" / "skill.md"


class LearningWorkflowContractTests(unittest.TestCase):
    def test_contract_declarations_match_learning_package(self) -> None:
        self.assertEqual(contract.PRODUCT_ID, "learning")
        self.assertEqual(contract.SKILL_ID, "learning")
        self.assertEqual(contract.WORKFLOW_ID, "learning")
        self.assertEqual(contract.WORKFLOW_STATUS, "implemented")
        self.assertEqual(contract.CANONICAL_ENTRYPOINT, "tokimon.workflow.learning.cli:main")
        self.assertEqual(contract.OPERATOR_ENTRYPOINT, ".venv/bin/python -m tokimon.workflow.learning")
        self.assertEqual(contract.LOCAL_STATE_ROOT, Path(".tokimon") / "workflow" / "learning")
        self.assertEqual(contract.RUN_ROOT, Path(".tokimon") / "workflow" / "learning" / "runs")
        self.assertEqual(contract.INTAKE_PACKAGE_ROOT, Path(".tokimon") / "workflow" / "learning" / "intake-packages")
        self.assertEqual(contract.REQUIRED_TOOL_CAPABILITIES, ())
        self.assertIn("create-intake-package", contract.COMMANDS)
        self.assertIn("record-activation", contract.COMMANDS)
        self.assertEqual(
            contract.VALID_TARGET_SURFACES,
            ("skill", "mcp", "skill_and_mcp", "workflow", "existing_surface_update"),
        )
        self.assertEqual(
            contract.REQUIRED_PRODUCT_DEVELOPMENT_EVIDENCE_FIELDS,
            ("acceptance_status", "acceptance_summary", "accepted_artifacts", "verification_evidence"),
        )
        self.assertEqual(contract.REQUIRED_ACCEPTED_ARTIFACT_FIELDS, ("path", "role"))
        self.assertEqual(contract.REQUIRED_VERIFICATION_EVIDENCE_FIELDS, ("kind", "reference", "outcome"))
        self.assertEqual(contract.WORKFLOW_READINESS["id"], "learning")

    def test_runtime_defaults_use_contract_declarations(self) -> None:
        payload = cli._contract_payload()

        self.assertEqual(cli.required_tool_capabilities(), contract.REQUIRED_TOOL_CAPABILITIES)
        self.assertEqual(payload["workflow_id"], contract.WORKFLOW_ID)
        self.assertEqual(payload["workflow_readiness"], contract.WORKFLOW_READINESS)
        self.assertIn("required_active_evidence_fields", payload)
        self.assertEqual(
            payload["required_product_development_evidence_fields"],
            list(contract.REQUIRED_PRODUCT_DEVELOPMENT_EVIDENCE_FIELDS),
        )
        self.assertEqual(
            payload["required_accepted_artifact_fields"],
            list(contract.REQUIRED_ACCEPTED_ARTIFACT_FIELDS),
        )
        self.assertEqual(
            payload["required_verification_evidence_fields"],
            list(contract.REQUIRED_VERIFICATION_EVIDENCE_FIELDS),
        )

    def test_docs_and_skill_name_required_surfaces(self) -> None:
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (L2_DOC, L3_DOC, SKILL_DOC)
        )

        for required_text in (
            "Workflow id:** `learning`",
            "Python package:** `tokimon/workflow/learning/`",
            "tokimon.workflow.learning.cli:main",
            ".venv/bin/python -m tokimon.workflow.learning",
            "tokimon/workflow/learning/contract.py",
            "WORKFLOW_READINESS",
            "agent-driven",
            "CLI-backed",
            ".tokimon/workflow/learning/intake-packages/",
            "product-development intake package",
            "Skill, MCP, workflow",
            "Candidate packets are reviewable evidence",
            "Mem Palace may",
        ):
            self.assertIn(required_text, combined)
        self.assertNotIn("docs/learning", combined)
        for old_text in (
            "promotion_draft",
            "promotion-draft",
            "promotion draft",
            "PROMOTION_DRAFT",
            "create-promotion-draft",
        ):
            self.assertNotIn(old_text, combined)


if __name__ == "__main__":
    unittest.main()
