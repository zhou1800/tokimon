from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tokimon.ai_harness_interface import plugin_creator


class PluginCreatorAdapterTests(unittest.TestCase):
    def test_missing_optional_validator_skips_unless_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts = Path(tmpdir)
            with mock.patch.dict(
                "os.environ",
                {plugin_creator.SCRIPTS_DIR_ENV: str(scripts)},
                clear=False,
            ):
                optional = plugin_creator.validate_plugin(package_root=scripts / "plugin")
                required = plugin_creator.validate_plugin(package_root=scripts / "plugin", required=True)

        self.assertTrue(optional["ok"])
        self.assertTrue(optional["skipped"])
        self.assertFalse(required["ok"])
        self.assertTrue(required["skipped"])

    def test_ci_fail_loud_env_requires_optional_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts = Path(tmpdir)
            with mock.patch.dict(
                "os.environ",
                {
                    plugin_creator.SCRIPTS_DIR_ENV: str(scripts),
                    plugin_creator.REQUIRE_ENV: "true",
                },
                clear=False,
            ):
                result = plugin_creator.validate_plugin(package_root=scripts / "plugin")

        self.assertFalse(result["ok"])
        self.assertTrue(result["skipped"])

    def test_cachebuster_adapter_invokes_script_and_updates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scripts = root / "scripts"
            plugin_root = root / "plugins" / "tokimon"
            manifest_path = plugin_root / ".codex-plugin" / "plugin.json"
            scripts.mkdir()
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text('{"name":"tokimon","version":"0.1.0"}\n', encoding="utf-8")
            (scripts / "update_plugin_cachebuster.py").write_text(
                "\n".join(
                    (
                        "import argparse, json",
                        "from pathlib import Path",
                        "parser = argparse.ArgumentParser()",
                        "parser.add_argument('plugin_path')",
                        "parser.add_argument('--cachebuster')",
                        "args = parser.parse_args()",
                        "path = Path(args.plugin_path) / '.codex-plugin' / 'plugin.json'",
                        "payload = json.loads(path.read_text())",
                        "payload['version'] = payload['version'].split('+', 1)[0] + '+codex.' + args.cachebuster",
                        "path.write_text(json.dumps(payload) + '\\n')",
                        "print(payload['version'])",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch.dict(
                "os.environ",
                {plugin_creator.SCRIPTS_DIR_ENV: str(scripts), plugin_creator.PYTHON_ENV: sys.executable},
                clear=False,
            ):
                result = plugin_creator.update_cachebuster(
                    package_root=plugin_root,
                    cachebuster="unit-test",
                )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertTrue(result["ok"], result)
        self.assertEqual(payload["version"], "0.1.0+codex.unit-test")

    def test_bootstrap_command_targets_ai_harness_output_layout(self) -> None:
        output_root = Path("/repo/.tokimon/ai-harness-interface/codex")
        marketplace_path = output_root / ".agents" / "plugins" / "marketplace.json"

        command = plugin_creator.bootstrap_command(
            output_root=output_root,
            marketplace_path=marketplace_path,
        )

        self.assertIn("create_basic_plugin.py", command[1])
        self.assertIn("--with-skills", command)
        self.assertIn("--with-mcp", command)
        self.assertIn("--with-marketplace", command)
        self.assertEqual(command[command.index("--path") + 1], str(output_root / "plugins"))
        self.assertEqual(command[command.index("--marketplace-path") + 1], str(marketplace_path))
        self.assertEqual(command[command.index("--marketplace-name") + 1], "tokimon-local")


if __name__ == "__main__":
    unittest.main()
