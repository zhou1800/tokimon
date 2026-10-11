from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import tokimon.runtime as runtime_helpers
from tokimon.workflow.memory_maintenance import cli, contract, operations


L2_DOC = ROOT / "docs" / "c4" / "level-2-containers" / "workflow" / "memory-maintenance.md"
L3_DOC = ROOT / "docs" / "c4" / "level-3-components" / "workflow" / "memory-maintenance-cli.md"


class MemoryMaintenanceWorkflowContractTests(unittest.TestCase):
    def test_contract_declarations_match_memory_maintenance_package(self) -> None:
        self.assertEqual(contract.WORKFLOW_ID, "memory-maintenance")
        self.assertEqual(contract.PYTHON_PACKAGE, "tokimon.workflow.memory_maintenance")
        self.assertEqual(contract.WORKFLOW_STATUS, "implemented")
        self.assertEqual(contract.CANONICAL_ENTRYPOINT, "tokimon.workflow.memory_maintenance.cli:main")
        self.assertEqual(contract.OPERATOR_ENTRYPOINT, ".venv/bin/python -m tokimon.workflow.memory_maintenance")
        self.assertEqual(contract.LOCAL_STATE_ROOT, Path(".tokimon") / "workflow" / "memory-maintenance")
        self.assertEqual(contract.RUN_LOG_ROOT, contract.LOCAL_STATE_ROOT / "runs")
        self.assertEqual(contract.BACKUP_ROOT, contract.LOCAL_STATE_ROOT / "backups")
        self.assertEqual(contract.MEMORY_SCOPE, "memory")
        self.assertEqual(contract.MEMORY_ROOM, "maintenance")
        self.assertEqual(contract.DEFAULT_MEMORY_DB, Path(".tokimon") / "memory" / "memory.db")
        self.assertEqual(contract.SNAPSHOT_COMMAND, "snapshot")
        self.assertEqual(contract.MUTATION_COMMANDS, ("delete", "update-weight", "repair"))

    def test_runtime_defaults_use_contract_declarations(self) -> None:
        self.assertEqual(
            operations.DEFAULT_RUN_ROOT,
            runtime_helpers.runtime_root(ROOT) / "workflow" / contract.WORKFLOW_ID,
        )
        self.assertEqual(cli.required_tool_capabilities(), ())

    def test_c4_docs_name_required_workflow_contract_surfaces(self) -> None:
        l2 = L2_DOC.read_text(encoding="utf-8")
        l3 = L3_DOC.read_text(encoding="utf-8")
        combined = f"{l2}\n{l3}"

        for required_text in (
            "## Workflow Package Declarations",
            "Workflow id:** `memory-maintenance`",
            "Python package:** `tokimon/workflow/memory_maintenance/`",
            "tokimon.workflow.memory_maintenance.cli:main",
            ".venv/bin/python -m tokimon.workflow.memory_maintenance",
            "tokimon/workflow/memory_maintenance/contract.py",
            "WORKFLOW_READINESS",
            "Required tool capability:** none",
            "scope=memory",
            "room=maintenance",
            ".tokimon/workflow/memory-maintenance/",
            "snapshot",
            "delete",
            "update-weight",
            "repair",
            "Security & Trust",
            "tests/workflow/memory_maintenance/",
        ):
            self.assertIn(required_text, combined)


if __name__ == "__main__":
    unittest.main()
