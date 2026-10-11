from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tokimon.ai_harness_interface import contract
from tokimon.tools import catalog
from tests.ai_harness_interface.test_generator import create_linked_checkout


class AiHarnessContractTests(unittest.TestCase):
    def test_contract_inventory_comes_from_current_repo_skills(self) -> None:
        repo_skill_ids = contract.repo_skill_ids(source_root=ROOT)
        package_skill_ids = contract.package_skill_ids(source_root=ROOT)

        self.assertIn("using-tokimon", repo_skill_ids)
        self.assertIn("memory", repo_skill_ids)
        self.assertIn("code-audit-fix", repo_skill_ids)
        self.assertIn("document-audit-fix", repo_skill_ids)
        self.assertNotIn("tokimon", repo_skill_ids)
        self.assertEqual(package_skill_ids[0], "using-tokimon")
        self.assertEqual(package_skill_ids[1], "memory")
        self.assertEqual(set(package_skill_ids), set(repo_skill_ids))

    def test_contract_payload_derives_skill_inventory_from_root_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "skills" / "demo"
            skill_dir.mkdir(parents=True)
            (skill_dir / "skill.md").write_text(
                "---\nname: demo\ndescription: Demo skill.\n---\n\n# Demo\n",
                encoding="utf-8",
            )

            payload = contract.contract_payload(workspace_root=root)

        self.assertEqual(payload["repo_skill_ids"], ["demo"])
        self.assertEqual(payload["package_skill_ids"], ["demo"])

    def test_mcp_mapping_uses_accepted_server_ids_and_live_modules(self) -> None:
        entries = {entry.capability_id: entry for entry in contract.MCP_ENTRIES}

        self.assertEqual(entries["mcp.telegram-bot"].server_id, "tokimon-telegram-bot")
        self.assertEqual(entries["mcp.ai-debate"].server_id, "tokimon-ai-debate")
        self.assertEqual(
            entries["mcp.remote-browser-broker"].server_id,
            "tokimon-remote-browser-broker",
        )
        self.assertNotIn("mcp.remote-agent-broker", entries)
        self.assertEqual(contract.MCP_ENTRIES, catalog.package_entries())
        for entry in entries.values():
            self.assertTrue(entry.module.startswith("tokimon.tools.mcp."))
            self.assertIsNotNone(importlib.import_module(entry.module))

    def test_target_platform_registry_declares_codex_and_claude(self) -> None:
        self.assertEqual(set(contract.TARGET_PLATFORMS), {"codex", "claude"})
        self.assertEqual(contract.TARGET_PLATFORM, "codex")

        codex = contract.target_platform_for("codex")
        claude = contract.target_platform_for("claude")

        self.assertEqual(codex.manifest_dir_name, ".codex-plugin")
        self.assertEqual(codex.marketplace_relative_path.as_posix(), ".agents/plugins/marketplace.json")
        self.assertEqual(claude.manifest_dir_name, ".claude-plugin")
        self.assertEqual(claude.marketplace_relative_path.as_posix(), ".claude-plugin/marketplace.json")
        self.assertEqual(claude.generated_output_relative_path.as_posix(), ".tokimon/ai-harness-interface/claude")
        self.assertIn("plugin-creator", codex.account_level_install_owner)
        self.assertIn("Claude Code plugin CLI", claude.account_level_install_owner)
        with self.assertRaisesRegex(ValueError, "unsupported target platform"):
            contract.target_platform_for("gemini")

    def test_contract_payload_supports_claude_target_platform(self) -> None:
        payload = contract.contract_payload(workspace_root=ROOT, target_platform="claude")

        self.assertEqual(payload["target_platform"], "claude")
        self.assertEqual(payload["supported_target_platforms"], ["claude", "codex"])
        self.assertEqual(payload["generated_output_relative_path"], ".tokimon/ai-harness-interface/claude")
        self.assertTrue(payload["marketplace_path"].endswith(".claude-plugin/marketplace.json"))
        self.assertEqual(payload["package_id"], "tokimon")
        self.assertEqual(payload["marketplace_name"], "tokimon-local")

    def test_linked_contract_paths_use_primary_runtime_and_keep_source_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary, linked = create_linked_checkout(Path(tmpdir))
            for platform in ("codex", "claude"):
                with self.subTest(platform=platform):
                    payload = contract.contract_payload(workspace_root=linked, target_platform=platform)
                    output = primary / ".tokimon" / "ai-harness-interface" / platform
                    marketplace = output / (".agents/plugins/marketplace.json" if platform == "codex" else ".claude-plugin/marketplace.json")
                    self.assertEqual(payload["workspace_root"], str(linked))
                    self.assertEqual(payload["output_root"], str(output))
                    self.assertEqual(payload["package_root"], str(output / "plugins" / "tokimon"))
                    self.assertEqual(payload["marketplace_path"], str(marketplace))
                    self.assertFalse((linked / ".tokimon").exists())

    def test_contract_payload_omits_account_config_fields(self) -> None:
        payload = contract.contract_payload(workspace_root=ROOT)

        self.assertEqual(payload["package_id"], "tokimon")
        self.assertEqual(payload["marketplace_name"], "tokimon-local")
        self.assertEqual(payload["startup_bootstrap_skill_id"], "using-tokimon")
        self.assertEqual(payload["memory_activation_skill_id"], "memory")
        self.assertIn("using-tokimon", payload["package_skill_ids"])
        self.assertNotIn("chatgpt", payload["package_skill_ids"])
        self.assertIn("code-audit-fix", payload["package_skill_ids"])
        self.assertIn("document-audit-fix", payload["package_skill_ids"])
        self.assertIn("memory", payload["package_skill_ids"])
        self.assertNotIn("package_skill_agent_manifests", payload)
        self.assertNotIn("aggregate_skill_id", payload)
        self.assertNotIn("startup_bootstrap_content_path", payload)
        self.assertNotIn("session_start_hook_path", payload)
        rendered = str(payload)
        self.assertNotIn("installedPackagePath", rendered)
        self.assertNotIn("targetPlatform", rendered)
        self.assertNotIn("TOKIMON_AI_HARNESS_CONFIG", rendered)
        self.assertNotIn("ai-harness-interface.json", rendered)

    def test_contract_payload_exposes_only_package_route_fields_for_mcp_entries(self) -> None:
        payload = contract.contract_payload(workspace_root=ROOT)

        expected_keys = {
            "capability_id",
            "server_id",
            "module",
            "serve_args",
            "health_args",
        }
        self.assertEqual(len(payload["mcp_entries"]), len(contract.MCP_ENTRIES))
        for entry, rendered in zip(contract.MCP_ENTRIES, payload["mcp_entries"], strict=True):
            self.assertEqual(set(rendered), expected_keys)
            self.assertEqual(rendered["capability_id"], entry.capability_id)
            self.assertEqual(rendered["server_id"], entry.server_id)
            self.assertEqual(rendered["module"], entry.module)
            self.assertEqual(rendered["serve_args"], list(entry.serve_args))
            self.assertEqual(rendered["health_args"], list(entry.health_args))


if __name__ == "__main__":
    unittest.main()
