from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WORKFLOW_ROOT = ROOT / "tokimon" / "workflow"
L2_WORKFLOW_ROOT = ROOT / "docs" / "c4" / "level-2-containers" / "workflow"
L3_WORKFLOW_ROOT = ROOT / "docs" / "c4" / "level-3-components" / "workflow"
DELETED_SHARED_WORKFLOW_DOC = L2_WORKFLOW_ROOT / ("over" + "view.md")
VALID_WORKFLOW_STATUSES = ("implemented", "support", "planned", "deprecated")
FORBIDDEN_REFERENCE_TOKENS = (
    "docs/c4/level-2-containers/workflow/" + ("over" + "view.md"),
    "WORKFLOW_" + "OVERVIEW",
    "Workflow Package" + " Standard",
    "workflow-package" + "-standard",
)


class WorkflowPackageContractTests(unittest.TestCase):
    def test_invalid_shared_workflow_overview_is_absent(self) -> None:
        self.assertFalse(DELETED_SHARED_WORKFLOW_DOC.exists())

    def test_docs_and_workflow_tests_do_not_reference_deleted_overview(self) -> None:
        offenders: list[str] = []
        for path in self._reference_scan_paths():
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN_REFERENCE_TOKENS:
                if token in text:
                    offenders.append(f"{path.relative_to(ROOT)}: {token}")

        self.assertEqual(offenders, [])

    def test_active_workflow_contracts_have_real_c4_owners(self) -> None:
        for package_name, module, contract_path in self._active_workflow_contracts():
            workflow_id = getattr(module, "WORKFLOW_ID", package_name)
            canonical_entrypoint = getattr(module, "CANONICAL_ENTRYPOINT", None)
            operator_entrypoint = getattr(module, "OPERATOR_ENTRYPOINT", None)
            package_path = contract_path.parent.relative_to(ROOT).as_posix() + "/"
            contract_reference = contract_path.relative_to(ROOT).as_posix()
            local_state_root = str(getattr(module, "LOCAL_STATE_ROOT", ""))
            l2_path = L2_WORKFLOW_ROOT / f"{workflow_id}.md"
            l3_path = L3_WORKFLOW_ROOT / f"{workflow_id}-cli.md"

            with self.subTest(workflow_id=workflow_id):
                self.assertTrue(l2_path.exists(), f"missing {l2_path.relative_to(ROOT)}")
                self.assertTrue(l3_path.exists(), f"missing {l3_path.relative_to(ROOT)}")

                l2_text = l2_path.read_text(encoding="utf-8")
                l3_text = l3_path.read_text(encoding="utf-8")

                for required_text in (
                    "Operator entrypoint:",
                    f"`{operator_entrypoint}`",
                    "## Containers",
                    "## Key Interactions",
                    "## Security & Trust",
                ):
                    self.assertIn(required_text, l2_text)

                for required_text in (
                    "## Workflow Package Declarations",
                    f"Workflow id:** `{workflow_id}`",
                    f"Python package:** `{package_path}`",
                    f"Canonical code entrypoint:** `{canonical_entrypoint}`",
                    f"Operator entrypoint:** `{operator_entrypoint}`",
                    f"Contract declarations:** `{contract_reference}`",
                    f"Workflow readiness declaration:** `{contract_reference}` `WORKFLOW_READINESS`",
                    l2_path.relative_to(ROOT).as_posix(),
                ):
                    self.assertIn(required_text, l3_text)

                if local_state_root:
                    self.assertIn(local_state_root, f"{l2_text}\n{l3_text}")

    def test_doctor_owns_workflow_readiness_declaration_shape(self) -> None:
        text = (L3_WORKFLOW_ROOT / "doctor-cli.md").read_text(encoding="utf-8")

        for required_text in (
            "WORKFLOW_READINESS",
            "Declaration fields: `id`, `workflow_status`, `entrypoint`, and `modes`.",
            "Workflow status values: `implemented`, `support`, `planned`, and `deprecated`.",
            "Mode fields: `id`, `role`, `summary`, `required_tool_capabilities`, `any_tool_capability_groups`, `agent_connector_capabilities`, and `memory_requirement`.",
            "Mode roles: `primary` and `fallback`.",
            "Readiness statuses: `ready`, `partial`, and `blocked`.",
            "A workflow is `ready` when a primary mode is ready, `partial` when a primary mode is partial or only a fallback mode can run, and `blocked` when no declared mode can run.",
        ):
            self.assertIn(required_text, text)

    def test_active_workflow_contracts_expose_doctor_readiness_declarations(self) -> None:
        offenders: list[str] = []
        for package_name, module, _contract_path in self._active_workflow_contracts():
            workflow_id = getattr(module, "WORKFLOW_ID", package_name)
            workflow_status = getattr(module, "WORKFLOW_STATUS", None)

            declaration = getattr(module, "WORKFLOW_READINESS", None)
            if not isinstance(declaration, dict):
                offenders.append(f"{workflow_id}: missing WORKFLOW_READINESS")
                continue
            if workflow_status not in VALID_WORKFLOW_STATUSES:
                offenders.append(f"{workflow_id}: WORKFLOW_STATUS must be a valid workflow status")
            if declaration.get("workflow_status") != workflow_status:
                offenders.append(f"{workflow_id}: readiness workflow_status does not match WORKFLOW_STATUS")
            if declaration.get("id") != workflow_id:
                offenders.append(f"{workflow_id}: readiness id does not match WORKFLOW_ID")
            if not str(declaration.get("entrypoint") or "").strip():
                offenders.append(f"{workflow_id}: readiness entrypoint is empty")
            modes = declaration.get("modes")
            if not isinstance(modes, tuple) or not modes:
                offenders.append(f"{workflow_id}: readiness modes must be a non-empty tuple")
                continue
            for mode in modes:
                if not isinstance(mode, dict):
                    offenders.append(f"{workflow_id}: readiness mode must be a dict")
                    continue
                if not str(mode.get("id") or "").strip():
                    offenders.append(f"{workflow_id}: readiness mode id is empty")
                if mode.get("role") not in {"primary", "fallback"}:
                    offenders.append(f"{workflow_id}: readiness mode role must be primary or fallback")
                if not str(mode.get("summary") or "").strip():
                    offenders.append(f"{workflow_id}: readiness mode summary is empty")
                self._assert_string_tuple_field(offenders, workflow_id, mode, "required_tool_capabilities")
                self._assert_string_tuple_field(offenders, workflow_id, mode, "agent_connector_capabilities")
                any_groups = mode.get("any_tool_capability_groups", ())
                if not isinstance(any_groups, tuple):
                    offenders.append(f"{workflow_id}: any_tool_capability_groups must be a tuple")
                else:
                    for group in any_groups:
                        if not isinstance(group, dict):
                            offenders.append(f"{workflow_id}: any tool capability group must be a dict")
                            continue
                        if not str(group.get("id") or "").strip():
                            offenders.append(f"{workflow_id}: any tool capability group id is empty")
                        capability_ids = group.get("capability_ids")
                        if not isinstance(capability_ids, tuple) or not all(
                            isinstance(item, str) and item for item in capability_ids
                        ):
                            offenders.append(f"{workflow_id}: any tool capability group ids must be a string tuple")

        self.assertEqual(offenders, [])

    def _assert_string_tuple_field(
        self,
        offenders: list[str],
        workflow_id: str,
        mode: dict,
        field_name: str,
    ) -> None:
        value = mode.get(field_name, ())
        if not isinstance(value, tuple) or not all(isinstance(item, str) and item for item in value):
            offenders.append(f"{workflow_id}: {field_name} must be a string tuple")

    def _active_workflow_contracts(self) -> tuple[tuple[str, ModuleType, Path], ...]:
        contracts: list[tuple[str, ModuleType, Path]] = []
        for contract_path in sorted(WORKFLOW_ROOT.glob("*/contract.py")):
            package_name = contract_path.parent.name
            module = importlib.import_module(f"tokimon.workflow.{package_name}.contract")
            workflow_status = getattr(module, "WORKFLOW_STATUS", None)
            has_operator_entrypoint = bool(
                getattr(module, "OPERATOR_ENTRYPOINT", None)
                or getattr(module, "CLI_ENTRYPOINT", None)
                or getattr(module, "CANONICAL_ENTRYPOINT", None)
                or getattr(module, "CANONICAL_OPERATOR_ENTRYPOINT", None)
            )
            if workflow_status in {"implemented", "support"} or has_operator_entrypoint:
                contracts.append((package_name, module, contract_path))
        return tuple(contracts)

    def _reference_scan_paths(self) -> tuple[Path, ...]:
        roots = (ROOT / "docs", ROOT / "tests", ROOT / "tokimon", ROOT / "skills")
        paths: list[Path] = []
        for root in roots:
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix in {".md", ".py"} and "__pycache__" not in path.parts
            )
        paths.append(ROOT / "AGENTS.md")
        return tuple(sorted(paths))


if __name__ == "__main__":
    unittest.main()
