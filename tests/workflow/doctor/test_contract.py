from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.doctor import contract


L2_DOC = ROOT / "docs" / "c4" / "level-2-containers" / "workflow" / "doctor.md"
L3_DOC = ROOT / "docs" / "c4" / "level-3-components" / "workflow" / "doctor-cli.md"


class DoctorWorkflowContractTests(unittest.TestCase):
    def test_contract_declarations_match_doctor_package(self) -> None:
        self.assertEqual(contract.WORKFLOW_ID, "doctor")
        self.assertEqual(contract.PYTHON_PACKAGE, "tokimon.workflow.doctor")
        self.assertEqual(contract.WORKFLOW_STATUS, "implemented")
        self.assertEqual(contract.CANONICAL_ENTRYPOINT, "tokimon.workflow.doctor.cli:main")
        self.assertEqual(contract.OPERATOR_ENTRYPOINT, ".venv/bin/python -m tokimon.workflow.doctor")
        self.assertEqual(contract.LOCAL_STATE_ROOT, Path(".tokimon") / "workflow" / "doctor")
        self.assertEqual(contract.MEMORY_SCOPE, "workflow")
        self.assertEqual(contract.MEMORY_ROOM, "doctor")
        self.assertEqual(contract.DEFAULT_MEMORY_DB, Path(".tokimon") / "memory" / "memory.db")
        self.assertEqual(contract.REQUIRED_TOOL_CAPABILITIES, ())
        self.assertEqual(contract.COMMANDS, ("check", "contract"))
        self.assertEqual(contract.STATUS_VALUES, ("pass", "warn", "fail"))
        self.assertEqual(contract.READINESS_STATUS_VALUES, ("ready", "partial", "blocked"))
        self.assertEqual(contract.WORKFLOW_READINESS["id"], "doctor")
        self.assertEqual(contract.WORKFLOW_READINESS["entrypoint"], contract.OPERATOR_ENTRYPOINT)

    def test_c4_docs_name_required_workflow_contract_surfaces(self) -> None:
        l2 = L2_DOC.read_text(encoding="utf-8")
        l3 = L3_DOC.read_text(encoding="utf-8")
        combined = f"{l2}\n{l3}"

        for required_text in (
            "## Workflow Package Declarations",
            "Workflow id:** `doctor`",
            "Python package:** `tokimon/workflow/doctor/`",
            "tokimon.workflow.doctor.cli:main",
            ".venv/bin/python -m tokimon.workflow.doctor",
            "tokimon/workflow/doctor/contract.py",
            "WORKFLOW_READINESS",
            "Required tool capability:** none",
            "scope=workflow",
            "room=doctor",
            ".tokimon/workflow/doctor/",
            "check",
            "contract",
            "pass",
            "warn",
            "fail",
            "ready",
            "partial",
            "blocked",
            "Security & Trust",
            "tests/workflow/doctor/",
        ):
            self.assertIn(required_text, combined)


if __name__ == "__main__":
    unittest.main()
