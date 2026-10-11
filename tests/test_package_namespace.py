from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class PackageNamespaceTests(unittest.TestCase):
    def test_pyproject_declares_tokimon_package_discovery(self) -> None:
        payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(payload["project"]["name"], "tokimon")
        self.assertEqual(payload["tool"]["setuptools"]["packages"]["find"]["include"], ["tokimon*"])

    def test_core_modules_import_from_tokimon_namespace(self) -> None:
        modules = (
            "tokimon.runtime",
            "tokimon.memory.mem_palace.mem_palace",
            "tokimon.tools.catalog",
            "tokimon.tools.registry",
            "tokimon.tools.mcp.telegram_bot",
            "tokimon.tools.mcp.ai_debate",
            "tokimon.tools.mcp.remote_browser_broker",
            "tokimon.tools.mcp.remote_agent_broker",
            "tokimon.ai_harness_interface.contract",
            "tokimon.ai_harness_interface.cli",
            "tokimon.ai_harness_interface.readiness",
            "tokimon.ai_harness_interface.plugin_creator",
            "tokimon.workflow.doctor.contract",
            "tokimon.workflow.memory_maintenance.contract",
            "tokimon.workflow.document_audit.contract",
            "tokimon.workflow.code_audit.contract",
            "tokimon.workflow.pair.contract",
        )

        for module_name in modules:
            with self.subTest(module_name=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_canonical_workflow_module_commands_emit_json_contracts(self) -> None:
        commands = (
            ("tokimon.workflow.doctor", "doctor"),
            ("tokimon.workflow.memory_maintenance", "memory-maintenance"),
            ("tokimon.workflow.document_audit", "document-audit"),
            ("tokimon.workflow.code_audit", "code-audit"),
            ("tokimon.workflow.pair", "pair"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            # Snapshot is intentionally read-only and requires an existing store.
            subprocess.run(
                [
                    str(PYTHON),
                    "-m",
                    "tokimon.memory.mem_palace.mem_palace",
                    "--db",
                    str(db_path),
                    "stats",
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            for module_name, workflow_id in commands:
                with self.subTest(module_name=module_name):
                    if module_name == "tokimon.workflow.memory_maintenance":
                        args = [
                            str(PYTHON),
                            "-m",
                            module_name,
                            "--db",
                            str(db_path),
                            "snapshot",
                            "--json",
                        ]
                    else:
                        args = [str(PYTHON), "-m", module_name, "contract", "--json"]
                    result = subprocess.run(
                        args,
                        cwd=ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    payload = json.loads(result.stdout)
                    if module_name == "tokimon.workflow.memory_maintenance":
                        self.assertEqual(payload["workflow_id"], workflow_id)
                        self.assertEqual(payload["database"], str(db_path))
                    else:
                        self.assertTrue(payload["ok"])
                        self.assertEqual(payload["contract"]["workflow_id"], workflow_id)
                        self.assertEqual(payload["contract"]["python_package"], module_name)
                        self.assertIn(module_name, payload["contract"]["operator_entrypoint"])

    def test_tools_registry_canonical_module_command_lists_capabilities(self) -> None:
        env = dict(os.environ)
        env["TOKIMON_MCP_CONNECTORS_JSON"] = "[]"
        result = subprocess.run(
            [str(PYTHON), "-m", "tokimon.tools.registry", "list", "--json"],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(result.stdout)
        capability_ids = tuple(capability["id"] for capability in payload["capabilities"])
        self.assertEqual(
            capability_ids,
            (
                "mcp.telegram-bot",
                "mcp.ai-debate",
                "mcp.remote-browser-broker",
                "mcp.remote-agent-broker",
                "mcp.container-executor",
            ),
        )

    def test_mem_palace_canonical_module_command_supports_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    str(PYTHON),
                    "-m",
                    "tokimon.memory.mem_palace.mem_palace",
                    "--db",
                    str(Path(tmpdir) / "memory.db"),
                    "stats",
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema_version"], 3)


if __name__ == "__main__":
    unittest.main()
