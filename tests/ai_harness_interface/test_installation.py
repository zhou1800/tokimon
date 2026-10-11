from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokimon.ai_harness_interface import contract, generator, installation_policy, plugin_creator
from tests.ai_harness_interface.support import without_local_claude_cli, without_local_plugin_creator
from tests.ai_harness_interface.test_cli_contract import ROOT, run_main_json


class LocalInstallationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(without_local_plugin_creator())
        self.enterContext(without_local_claude_cli())

    def generate(self, root: Path, platform: str, mode: str = "auto") -> tuple[generator.GenerationPaths, dict]:
        output = root / platform
        settings = root / "settings.json"
        common = ["--workspace-root", str(ROOT), "--output-root", str(output)]
        if platform == "codex":
            status, bootstrap = run_main_json([*common, "bootstrap-codex", "--json"])
            self.assertEqual(status, 0, bootstrap)
            self.assertEqual(bootstrap["method"], "built-in")
        status, configured = run_main_json([
            *common, "configure", "--settings-path", str(settings), "--docker-features", mode, "--yes", "--json",
        ])
        self.assertEqual(status, 0, configured)
        status, generated = run_main_json([
            *common, "generate", "--target-platform", platform, "--settings-path", str(settings), "--json",
        ])
        self.assertEqual(status, 0, generated)
        status, validated = run_main_json([
            *common, "validate", "--target-platform", platform, "--settings-path", str(settings), "--json",
        ])
        self.assertEqual(status, 0, validated)
        return generator.paths_for(source_root=ROOT, output_root=output, target_platform=platform), generated

    def test_fresh_install_without_helpers_or_docker_needs_no_external_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(installation_policy, "which", return_value=None), mock.patch(
            "subprocess.run", side_effect=AssertionError("installation must not invoke dependency installers"),
        ):
            for platform in ("codex", "claude"):
                with self.subTest(platform=platform):
                    paths, generated = self.generate(Path(temp), platform)
                    self.assertEqual(generated["mcp_server_ids"], ["tokimon-telegram-bot", "tokimon-ai-debate"])
                    self.assertNotIn("container-executor", generated["skill_ids"])
                    self.assertFalse((paths.package_root / "skills/container-executor").exists())
                    mcp = json.loads((paths.package_root / ".mcp.json").read_text())
                    self.assertNotIn("tokimon-remote-browser-broker", mcp["mcpServers"])
                    self.assertEqual(generated["installation_policy"]["dependency_installation"], "never")

    def test_available_docker_preserves_supported_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(installation_policy, "which", return_value="/test/docker"):
            paths, generated = self.generate(Path(temp), "codex")
        self.assertIn("tokimon-remote-browser-broker", generated["mcp_server_ids"])
        self.assertIn("container-executor", generated["skill_ids"])

    def test_explicit_disable_wins_over_available_docker(self) -> None:
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(installation_policy, "which", return_value="/test/docker"):
            for platform in ("codex", "claude"):
                paths, generated = self.generate(Path(temp), platform, "disabled")
                self.assertFalse(generated["installation_policy"]["docker_enabled"])
                self.assertNotIn("container-executor", generated["skill_ids"])

    def test_validation_uses_snapshot_after_host_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(installation_policy, "which", return_value=None):
                paths, generated = self.generate(Path(temp), "codex")
            with mock.patch.object(installation_policy, "which", side_effect=AssertionError("validation must not probe the host")):
                self.assertTrue(generator.validate_generated_package(paths)["ok"])

    def test_policy_tampering_and_missing_snapshot_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(installation_policy, "which", return_value=None):
            paths, generated = self.generate(Path(temp), "codex")
            path = paths.output_root / installation_policy.POLICY_FILENAME
            payload = json.loads(path.read_text())
            payload["dependency_installation"] = "automatic"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(generator.PackageValidationError, "snapshot"):
                generator.validate_generated_package(paths)
            path.unlink()
            with self.assertRaisesRegex(generator.PackageValidationError, "snapshot is missing"):
                generator.validate_generated_package(paths)

    def test_native_bootstrap_preserves_existing_files_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = generator.paths_for(source_root=ROOT, output_root=root / "output")
            self.assertTrue(generator.bootstrap_scaffold(paths)["ok"])
            mcp = paths.package_root / ".mcp.json"
            mcp.write_text('{"operator":"preserve"}\n')
            self.assertTrue(generator.bootstrap_scaffold(paths)["ok"])
            self.assertEqual(json.loads(mcp.read_text()), {"operator": "preserve"})
            self.assertTrue(generator.bootstrap_scaffold(paths, force=True)["ok"])
            self.assertEqual(json.loads(mcp.read_text()), {"mcpServers": {}})
            mcp.unlink()
            canary = root / "canary"
            canary.write_text("unchanged")
            mcp.symlink_to(canary)
            self.assertFalse(generator.bootstrap_scaffold(paths, force=True)["ok"])
            self.assertEqual(canary.read_text(), "unchanged")

    def test_required_legacy_helper_and_failing_available_helper_do_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(temp) / "output")
            with mock.patch.dict("os.environ", {plugin_creator.REQUIRE_ENV: "true"}):
                self.assertFalse(generator.bootstrap_scaffold(paths)["ok"])
                self.assertFalse(paths.package_root.exists())
            scripts = Path(temp) / "scripts"
            scripts.mkdir()
            (scripts / "create_basic_plugin.py").write_text("raise SystemExit(1)\n")
            with mock.patch.dict("os.environ", {plugin_creator.SCRIPTS_DIR_ENV: str(scripts)}):
                self.assertFalse(generator.bootstrap_scaffold(paths)["ok"])
                self.assertFalse(paths.package_root.exists())

    def test_cachebuster_works_without_helper_and_invalid_metadata_preserves_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(temp) / "output")
            self.assertTrue(generator.bootstrap_scaffold(paths)["ok"])
            result = plugin_creator.update_cachebuster(package_root=paths.package_root, cachebuster="native-test")
            self.assertTrue(result["ok"])
            self.assertEqual(json.loads(paths.manifest_path.read_text())["version"], contract.PACKAGE_VERSION + "+codex.native-test")
            saved = paths.manifest_path.read_bytes()
            self.assertFalse(plugin_creator.update_cachebuster(package_root=paths.package_root, cachebuster="bad token")["ok"])
            self.assertEqual(paths.manifest_path.read_bytes(), saved)

    def test_custom_docker_command_and_invalid_configuration(self) -> None:
        with mock.patch.dict("os.environ", {"TOKIMON_RB_DOCKER": "custom-docker"}), mock.patch.object(installation_policy, "which", return_value="/test/custom") as resolver:
            self.assertTrue(installation_policy.resolve({}).docker_enabled)
            resolver.assert_called_once_with("custom-docker")
        for value in (None, "enabled", [], True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                installation_policy.validate_settings({"installation": {"docker_features": value}})


if __name__ == "__main__":
    unittest.main()
