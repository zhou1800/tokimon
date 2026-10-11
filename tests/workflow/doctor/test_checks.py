from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.memory.mem_palace import mem_palace
from tokimon.workflow.code_audit import contract as code_audit_contract
from tokimon.workflow.document_audit import contract as document_audit_contract
from tokimon.workflow.doctor import contract as doctor_contract
from tokimon.workflow.doctor import checks
from tokimon.workflow.learning import contract as learning_contract
from tokimon.workflow.memory_maintenance import contract as memory_maintenance_contract
from tokimon.workflow.pair import contract as pair_contract


def create_minimal_tokimon_root(root: Path, *, with_memory_db: bool = True) -> Path:
    for relative_dir in (
        "docs/c4",
        "tokimon/memory",
        "tokimon/tools",
        "tokimon/workflow",
        "tests",
        ".tokimon",
        ".venv/bin",
    ):
        (root / relative_dir).mkdir(parents=True, exist_ok=True)

    for relative_file in (
        "AGENTS.md",
        "docs/guidelines/repository-guidelines.md",
        "docs/guidelines/helix.md",
        "docs/guidelines/code-guidelines.md",
        "docs/guidelines/document-guidelines.md",
        "docs/guidelines/c4-document-guidelines.md",
        "docs/guidelines/language-and-communication-guidelines.md",
        "requirements.txt",
    ):
        path = root / relative_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative_file}\n", encoding="utf-8")

    (root / ".gitignore").write_text(".tokimon/\n", encoding="utf-8")
    python_path = root / ".venv" / "bin" / "python"
    python_path.write_text("#!/bin/sh\n", encoding="utf-8")
    python_path.chmod(0o755)

    if with_memory_db:
        conn = mem_palace.connect(root / ".tokimon" / "memory" / "memory.db")
        conn.close()

    for package_name, workflow_id in (
        ("doctor", "doctor"),
        ("memory_maintenance", "memory-maintenance"),
        ("document_audit", "document-audit"),
        ("code_audit", "code-audit"),
        ("pair", "pair"),
        ("learning", "learning"),
    ):
        workflow_dir = root / "tokimon" / "workflow" / package_name
        workflow_dir.mkdir(parents=True, exist_ok=True)
        (workflow_dir / "contract.py").write_text(
            "\n".join(
                (
                    f"WORKFLOW_ID = {workflow_id!r}",
                    "WORKFLOW_READINESS = {",
                    "    'id': WORKFLOW_ID,",
                    "    'workflow_status': 'implemented',",
                    f"    'entrypoint': '.venv/bin/python -m tokimon.workflow.{package_name}',",
                    "    'modes': (",
                    "        {",
                    "            'id': 'default',",
                    "            'role': 'primary',",
                    "            'summary': 'Synthetic workflow declaration for Doctor tests.',",
                    "            'required_tool_capabilities': (),",
                    "        },",
                    "    ),",
                    "}",
                    "",
                )
            ),
            encoding="utf-8",
        )

    return python_path


class DoctorCheckTests(unittest.TestCase):
    def test_memory_check_reads_committed_wal_without_changing_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_minimal_tokimon_root(root)
            db_path = root / ".tokimon" / "memory" / "memory.db"
            writer = mem_palace.connect(db_path)
            try:
                mem_palace.add_memory(writer, kind="fact", summary="Checkpointed record")
                writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                mem_palace.add_memory(writer, kind="fact", summary="Committed WAL record", pinned=True)
                before = [tuple(row) for row in writer.execute("SELECT * FROM memories ORDER BY id")]
                self.assertGreater(db_path.with_name("memory.db-wal").stat().st_size, 0)

                report = checks.check_memory_database(root)

                self.assertEqual(report.status, "pass")
                self.assertEqual(report.data["memories"], 2)
                self.assertEqual(report.data["pinned_memories"], 1)
                self.assertEqual([tuple(row) for row in writer.execute("SELECT * FROM memories ORDER BY id")], before)
            finally:
                writer.close()

    def test_workflow_contracts_expose_readiness_declarations(self) -> None:
        declarations = (
            doctor_contract.WORKFLOW_READINESS,
            memory_maintenance_contract.WORKFLOW_READINESS,
            document_audit_contract.WORKFLOW_READINESS,
            code_audit_contract.WORKFLOW_READINESS,
            pair_contract.WORKFLOW_READINESS,
            learning_contract.WORKFLOW_READINESS,
        )

        self.assertEqual(checks.load_workflow_readiness_declarations(), declarations)
        for declaration in declarations:
            self.assertIn("id", declaration)
            self.assertIn("entrypoint", declaration)
            self.assertIn("modes", declaration)
            self.assertGreater(len(declaration["modes"]), 0)

    def test_declared_workflow_readiness_selects_fallback_mode(self) -> None:
        memory_item = checks.ReadinessItem(
            id="memory.mem_palace",
            layer="memory",
            kind="capability",
            status="ready",
            summary="ready",
        )
        item = checks.evaluate_workflow_readiness_declaration(
            {
                "id": "synthetic-workflow",
                "workflow_status": "implemented",
                "entrypoint": ".venv/bin/python -m tokimon.workflow.synthetic",
                "modes": (
                    {
                        "id": "primary-tool-mode",
                        "role": "primary",
                        "summary": "Primary mode requires a missing tool.",
                        "required_tool_capabilities": ("missing.tool",),
                    },
                    {
                        "id": "fallback-mode",
                        "role": "fallback",
                        "summary": "Fallback mode can run without tools.",
                        "required_tool_capabilities": (),
                    },
                ),
            },
            capability_by_id={},
            memory_item=memory_item,
        )

        self.assertEqual(item.id, "synthetic-workflow")
        self.assertEqual(item.status, "partial")
        self.assertEqual(item.data["selected_mode"], "fallback-mode")

    def test_declared_workflow_readiness_reports_unverified_agent_connectors(self) -> None:
        memory_item = checks.ReadinessItem(
            id="memory.mem_palace",
            layer="memory",
            kind="capability",
            status="ready",
            summary="ready",
        )
        item = checks.evaluate_workflow_readiness_declaration(
            {
                "id": "synthetic-agent-workflow",
                "workflow_status": "implemented",
                "entrypoint": ".venv/bin/python -m tokimon.workflow.synthetic_agent",
                "modes": (
                    {
                        "id": "agent-connector-mode",
                        "role": "primary",
                        "summary": "Mode needs an external connector.",
                        "required_tool_capabilities": (),
                        "agent_connector_capabilities": ("google-drive.sheets",),
                    },
                ),
            },
            capability_by_id={
                "mcp.telegram-bot": {"status": "available"},
                "mcp.remote-browser-broker": {"status": "available"},
            },
            memory_item=memory_item,
        )

        self.assertEqual(item.id, "synthetic-agent-workflow")
        self.assertEqual(item.status, "partial")
        self.assertIn("google-drive.sheets", item.reasons[0])

    def test_run_doctor_reports_memory_tools_and_workflow_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_path = create_minimal_tokimon_root(root)
            conn = mem_palace.connect(root / ".tokimon" / "memory" / "memory.db")
            try:
                mem_palace.add_memory(
                    conn,
                    kind="fact",
                    summary="Doctor test memory",
                    details="Used to verify memory usage reporting.",
                    scope="workflow",
                    room="doctor",
                    tags=["doctor"],
                )
            finally:
                conn.close()

            report = checks.run_doctor(
                workspace_root=root,
                env={"TOKIMON_MCP_CONNECTORS_JSON": "[]", "PATH": ""},
                python_executable=str(python_path),
                platform_name="Linux",
                host_id="test-host",
            )

        self.assertEqual(report["workflow_id"], "doctor")
        self.assertTrue(report["ok"])
        self.assertEqual(report["status"], "warn")
        self.assertEqual(report["summary"]["checks"]["fail"], 0)
        self.assertGreater(report["summary"]["readiness"]["ready"], 0)
        self.assertGreater(report["summary"]["readiness"]["blocked"], 0)
        self.assertEqual(report["memory"]["id"], "memory.mem_palace")
        self.assertEqual(report["memory"]["data"]["memories"], 1)
        self.assertIn("tools", report)
        self.assertIn("workflows", report)
        self.assertEqual(
            set(report["tools"]["status_counts"]),
            {"available", "unavailable"},
        )
        tool_ids = {item["id"] for item in report["tools"]["items"]}
        self.assertEqual(
            tool_ids,
            {
                "mcp.telegram-bot",
                "mcp.ai-debate",
                "mcp.remote-browser-broker",
                "mcp.remote-agent-broker",
                "mcp.container-executor",
            },
        )
        workflow_ids = {item["id"] for item in report["workflows"]}
        self.assertIn("memory-maintenance", workflow_ids)
        self.assertIn("document-audit", workflow_ids)
        self.assertIn("code-audit", workflow_ids)
        check_ids = {check["id"] for check in report["checks"]}
        self.assertEqual(
            check_ids,
            {
                "repository.required_paths",
                "runtime.gitignore",
                "runtime.root",
                "python.venv",
                "memory.mem_palace",
                "tools.registry",
            },
        )

    def test_tool_registry_rejects_unowned_placeholder_status(self) -> None:
        with patch.object(
            checks.registry,
            "capability_map",
            return_value={
                "host": {"host_id": "test-host"},
                "capabilities": [
                    {
                        "id": "mcp.ai-debate",
                        "name": "AI Debate MCP",
                        "family": "mcp",
                        "status": "placeholder",
                    }
                ],
            },
        ):
            result = checks.check_tool_registry(Path("."))

        self.assertEqual(result.status, "fail")
        self.assertEqual(
            result.summary,
            "Tools registry returned malformed capability metadata.",
        )
        self.assertEqual(result.details, ("mcp.ai-debate",))

    def test_missing_memory_database_is_warning_not_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_path = create_minimal_tokimon_root(root, with_memory_db=False)

            report = checks.run_doctor(
                workspace_root=root,
                env={"TOKIMON_MCP_CONNECTORS_JSON": "[]", "PATH": ""},
                python_executable=str(python_path),
                platform_name="Linux",
                host_id="test-host",
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["status"], "warn")
        memory_check = next(check for check in report["checks"] if check["id"] == "memory.mem_palace")
        self.assertEqual(memory_check["status"], "warn")
        self.assertIn("does not exist", memory_check["summary"])
        self.assertEqual(report["memory"]["status"], "partial")

    def test_malformed_connector_inventory_is_an_actionable_foundation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_path = create_minimal_tokimon_root(root)
            report = checks.run_doctor(
                workspace_root=root,
                env={
                    "TOKIMON_MCP_CONNECTORS_JSON": (
                        '{"connectors":[{"id":"connector","capability_id":'
                        '"mcp.telegram-bot","source":"codex","status":"availble",'
                        '"transport":"codex-mcp","tools":["health"]}]}'
                    ),
                    "PATH": "",
                },
                python_executable=str(python_path),
                platform_name="Linux",
                host_id="test-host",
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "fail")
        tool_check = next(
            check for check in report["checks"] if check["id"] == "tools.registry"
        )
        self.assertEqual(tool_check["status"], "fail")
        self.assertEqual(tool_check["summary"], "Tools registry metadata could not be loaded.")
        self.assertEqual(
            tool_check["details"],
            [
                "MCPConnectorInventoryError: Malformed MCP connector inventory in "
                "TOKIMON_MCP_CONNECTORS_JSON: connector 1 field 'status' must be one of: "
                "available, unavailable."
            ],
        )

    def test_linked_worktree_can_reuse_primary_worktree_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "linked"
            primary = Path(tmpdir) / "primary"
            root.mkdir()
            git_common = primary / ".git"
            worktree_git_dir = git_common / "worktrees" / "linked"
            worktree_git_dir.mkdir(parents=True)
            git_common.mkdir(exist_ok=True)
            (root / ".git").write_text(f"gitdir: {worktree_git_dir}\n", encoding="utf-8")
            python_path = primary / ".venv" / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_text("#!/bin/sh\n", encoding="utf-8")

            check = checks.check_python_environment(root, python_executable=str(python_path))

        self.assertEqual(check.status, "pass")
        self.assertIn("primary worktree .venv", check.summary)

    def test_wheel_workspace_uses_its_owning_python_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = Path(tmpdir)
            workspace = environment / "share/tokimon"
            workspace.mkdir(parents=True)
            python = environment / "bin/python"
            python.parent.mkdir()
            python.write_text("#!/bin/sh\n")
            with patch.object(sys, "prefix", str(environment)), patch.object(sys, "executable", str(python)):
                result = checks.check_python_environment(workspace)
                wrong = checks.check_python_environment(workspace, python_executable="/another-environment/bin/python")

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.data["expected_python"], str(python))
            self.assertFalse((workspace / ".venv").exists())
            self.assertEqual(wrong.status, "fail")

    def test_runtime_policy_uses_primary_worktree_tokimon_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "linked"
            primary = Path(tmpdir) / "primary"
            root.mkdir()
            runtime_root = primary / ".tokimon"
            runtime_root.mkdir(parents=True)
            worktree_git_dir = primary / ".git" / "worktrees" / "linked"
            worktree_git_dir.mkdir(parents=True)
            (root / ".git").write_text(f"gitdir: {worktree_git_dir}\n", encoding="utf-8")
            (root / ".gitignore").write_text(".tokimon/\n", encoding="utf-8")

            _, runtime_check = checks.check_runtime_policy(root)

        self.assertEqual(runtime_check.status, "pass")
        self.assertEqual(runtime_check.data["path"], str(runtime_root))

    def test_workflow_readiness_declarations_are_loaded_from_requested_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            contract_path = root / "tokimon" / "workflow" / "synthetic" / "contract.py"
            contract_path.parent.mkdir(parents=True)
            contract_path.write_text(
                "\n".join(
                    (
                        "WORKFLOW_READINESS = {",
                        "    'id': 'synthetic-from-requested-root',",
                        "    'workflow_status': 'implemented',",
                        "    'entrypoint': '.venv/bin/python -m tokimon.workflow.synthetic',",
                        "    'modes': ({'id': 'default', 'role': 'primary', 'summary': 'ready'},),",
                        "}",
                    )
                ),
                encoding="utf-8",
            )

            declarations = checks.load_workflow_readiness_declarations(workspace_root=root)

        self.assertEqual(len(declarations), 1)
        self.assertEqual(declarations[0]["id"], "synthetic-from-requested-root")
        self.assertNotIn("_load_error", declarations[0])

    def test_missing_required_repository_path_is_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_path = create_minimal_tokimon_root(root)
            (root / "docs" / "guidelines" / "helix.md").unlink()

            report = checks.run_doctor(
                workspace_root=root,
                env={"TOKIMON_MCP_CONNECTORS_JSON": "[]", "PATH": ""},
                python_executable=str(python_path),
                platform_name="Linux",
                host_id="test-host",
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "fail")
        required_paths_check = next(check for check in report["checks"] if check["id"] == "repository.required_paths")
        self.assertEqual(required_paths_check["status"], "fail")
        self.assertIn("docs/guidelines/helix.md (file)", required_paths_check["details"])


if __name__ == "__main__":
    unittest.main()
