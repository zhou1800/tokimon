from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.ai_harness_interface.support import without_local_claude_cli, without_local_plugin_creator
from tokimon.ai_harness_interface import cli, configuration, contract, mcp
from tests.ai_harness_interface.test_generator import create_linked_checkout, generate_codex_fixture


def run_main_json(argv: list[str]) -> tuple[int, dict[str, object]]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = cli.main(argv)
    return status, json.loads(output.getvalue())


class AiHarnessCliContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(without_local_plugin_creator())
        self.enterContext(mock.patch("tokimon.ai_harness_interface.installation_policy.which", return_value="/test/docker"))

    def test_cli_exposes_required_commands(self) -> None:
        parser = cli.build_parser()
        commands = {
            "contract",
            "bootstrap-codex",
            "configure",
            "generate",
            "validate",
            "diagnose",
            "post-install",
        }

        help_text = parser.format_help()
        for command in commands:
            with self.subTest(command=command):
                self.assertIn(command, help_text)
                self.assertEqual(parser.parse_args([command]).command, command)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["unknown-command"])

        args = parser.parse_args(["contract", "--json"])
        self.assertEqual(args.command, "contract")
        self.assertTrue(args.json)

        diagnose_args = parser.parse_args(["diagnose", "--skip-health", "--json"])
        self.assertTrue(diagnose_args.skip_health)
        self.assertTrue(diagnose_args.json)

        validate_args = parser.parse_args(["validate", "--settings-path", "settings.json", "--json"])
        self.assertEqual(validate_args.settings_path, Path("settings.json"))
        self.assertTrue(validate_args.json)

        post_install_args = parser.parse_args(["post-install", "--skip-platform", "--skip-health", "--json"])
        self.assertEqual(post_install_args.command, "post-install")
        self.assertTrue(post_install_args.skip_platform)
        self.assertTrue(post_install_args.skip_health)
        self.assertTrue(post_install_args.json)

        configure_args = parser.parse_args(["configure", "--yes", "--json"])
        self.assertEqual(configure_args.command, "configure")
        self.assertTrue(configure_args.yes)
        self.assertTrue(configure_args.json)
        native_pair_args = parser.parse_args(
            ["configure", "--pair-access", "native-subagent", "--yes"]
        )
        self.assertEqual(native_pair_args.pair_access, "native-subagent")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    ["configure", "--ai-debate-access", "native-subagent", "--yes"]
                )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["configure", "--mcp-capability", "mcp.ai-debate"])

    def test_cli_target_platform_flag_defaults_to_codex_and_accepts_claude(self) -> None:
        parser = cli.build_parser()

        for command in ("contract", "generate", "validate", "diagnose", "post-install"):
            with self.subTest(command=command):
                default_args = parser.parse_args([command])
                claude_args = parser.parse_args([command, "--target-platform", "claude"])
                self.assertEqual(default_args.target_platform, "codex")
                self.assertEqual(claude_args.target_platform, "claude")

        with self.assertRaises(SystemExit):
            parser.parse_args(["generate", "--target-platform", "gemini"])

    def test_contract_json_command_for_claude_target_platform(self) -> None:
        status, emitted = run_main_json(["contract", "--target-platform", "claude", "--json"])

        self.assertEqual(status, 0)
        payload = emitted["contract"]
        self.assertEqual(payload["target_platform"], "claude")
        self.assertEqual(payload["supported_target_platforms"], ["claude", "codex"])
        self.assertEqual(payload["generated_output_relative_path"], ".tokimon/ai-harness-interface/claude")
        self.assertTrue(payload["marketplace_path"].endswith(".claude-plugin/marketplace.json"))
        self.assertTrue(payload["package_root"].endswith("claude/plugins/tokimon"))

    def test_contract_json_command(self) -> None:
        status, emitted = run_main_json(["contract", "--json"])

        self.assertEqual(status, 0)
        self.assertTrue(emitted["ok"])
        payload = emitted["contract"]
        self.assertEqual(payload["package_id"], "tokimon")
        self.assertIn("configure", payload["commands"])
        self.assertIn("bootstrap-codex", payload["commands"])
        self.assertIn("post-install", payload["commands"])
        self.assertNotIn("install-codex", payload["commands"])
        self.assertNotIn("refresh-codex", payload["commands"])

    def test_linked_contract_json_routes_both_platform_outputs_to_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary, linked = create_linked_checkout(Path(tmpdir))
            for platform in ("codex", "claude"):
                with self.subTest(platform=platform):
                    status, emitted = run_main_json([
                        "--workspace-root", str(linked), "contract",
                        "--target-platform", platform, "--json",
                    ])
                    self.assertEqual(status, 0)
                    self.assertTrue(emitted["ok"])
                    payload = emitted["contract"]
                    output = primary / ".tokimon" / "ai-harness-interface" / platform
                    self.assertEqual(payload["workspace_root"], str(linked))
                    self.assertEqual(payload["output_root"], str(output))
                    self.assertEqual(payload["package_root"], str(output / "plugins" / "tokimon"))
                    marketplace = output / (".agents/plugins/marketplace.json" if platform == "codex" else ".claude-plugin/marketplace.json")
                    self.assertEqual(payload["marketplace_path"], str(marketplace))
                    self.assertFalse(output.exists())

    def test_post_install_text_marks_skipped_checks_as_not_checked(self) -> None:
        rendered = cli._render_post_install_text(
            {
                "ok": True,
                "summary": {
                    "plugin_installed_enabled": None,
                    "mcp_registration": {"checked": False, "total": 3},
                    "mcp_route_back_health": {"checked": False, "total": 3},
                    "mcp_operational": {"checked": False, "total": 3},
                },
                "package": {"ok": True},
                "configuration": {"ok": True},
                "pair": {"configured": False},
                "remaining_work": [],
                "next_steps": [],
            }
        )

        self.assertIn("MCP registration: not checked", rendered)
        self.assertIn("MCP route-back health: not checked", rendered)
        self.assertIn("operational readiness: not checked", rendered)

    def test_post_install_text_separates_remaining_work_from_next_steps(self) -> None:
        rendered = cli._render_post_install_text(
            {
                "ok": False,
                "summary": {},
                "package": {"ok": False},
                "configuration": {"ok": True},
                "pair": {"configured": False},
                "remaining_work": ["Regenerate the package."],
                "next_steps": ["Resolve blocking remaining work, then rerun post-install."],
            }
        )

        self.assertIn("Remaining work:\n1. Regenerate the package.", rendered)
        self.assertIn(
            "Next steps:\n1. Resolve blocking remaining work, then rerun post-install.",
            rendered,
        )

    def test_configure_yes_writes_repository_local_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.local.json"
            status, emitted = run_main_json(
                [
                    "configure",
                    "--yes",
                    "--settings-path",
                    str(settings_path),
                    "--json",
                ]
            )

            written = json.loads(settings_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertTrue(emitted["ok"])
        self.assertEqual(emitted["settings_path"], str(settings_path.resolve()))
        self.assertEqual(written["pair"]["cwd"], str(ROOT))
        self.assertNotIn("ai_harness_interface", written)
        self.assertEqual(
            written["ai_debate"],
            {"access": "local-cli", "peer_b": "codex"},
        )
        self.assertEqual(written["pair"]["access"], "native-subagent")
        self.assertEqual(written["pair"]["peer_b"], "codex")
        self.assertNotIn("agents", written)

    def test_configure_explicit_flags_override_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.local.json"
            status, emitted = run_main_json(
                [
                    "configure",
                    "--settings-path",
                    str(settings_path),
                    "--ai-debate-access",
                    "local-cli",
                    "--ai-debate-peer-b",
                    "codex",
                    "--pair-access",
                    "local-cli",
                    "--pair-peer-b",
                    "codex",
                    "--json",
                ]
            )

        self.assertEqual(status, 0)
        self.assertTrue(emitted["ok"])
        settings = emitted["settings"]
        self.assertNotIn("ai_harness_interface", settings)
        self.assertEqual(settings["ai_debate"]["access"], "local-cli")
        self.assertEqual(settings["ai_debate"]["peer_b"], "codex")
        self.assertEqual(settings["pair"]["access"], "local-cli")
        self.assertEqual(settings["pair"]["peer_b"], "codex")
        self.assertEqual(settings["pair"]["cwd"], str(ROOT))

    def test_configure_preserves_unrelated_settings_and_normalizes_ai_debate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.local.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "ai_debate": {"provider": "retired"},
                        "ai_harness_interface": {
                            "mcp_capabilities": ["mcp.telegram-bot"],
                        },
                        "unrelated_tool": {"enabled": True, "label": "preserve me"},
                        "agents": {
                            "codex": {"model": "old-codex", "effort": "high"},
                            "claude": {"model": "old-claude", "effort": "max"},
                            "unrelated": {"label": "preserve me"},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            status, emitted = run_main_json(
                [
                    "configure",
                    "--yes",
                    "--settings-path",
                    str(settings_path),
                    "--json",
                ]
            )
            written = json.loads(settings_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertTrue(emitted["ok"])
        self.assertEqual(written["agents"], {"unrelated": {"label": "preserve me"}})
        self.assertEqual(
            written["unrelated_tool"],
            {"enabled": True, "label": "preserve me"},
        )
        self.assertEqual(
            written["ai_debate"],
            {"access": "local-cli", "peer_b": "codex"},
        )
        self.assertNotIn("ai_harness_interface", written)

    def test_configure_migrates_retired_access_values_to_local_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.local.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "codex": {"model": "codex-model", "effort": "high"},
                            "claude": {"model": "claude-model", "effort": "max"},
                        },
                        "ai_debate": {
                            "access": "mcp.remote-agent-broker",
                            "peer_b": "claude",
                        },
                        "pair": {
                            "access": "mcp.remote-agent-broker",
                            "peer_b": "claude",
                            "cwd": "/retired/checkout",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            status, emitted = run_main_json(
                [
                    "configure",
                    "--yes",
                    "--settings-path",
                    str(settings_path),
                    "--json",
                ]
            )
            written = json.loads(settings_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertTrue(emitted["ok"])
        self.assertNotIn("agents", written)
        self.assertEqual(written["ai_debate"]["access"], "local-cli")
        self.assertEqual(written["pair"]["access"], "local-cli")
        self.assertEqual(written["ai_debate"]["peer_b"], "claude")
        self.assertEqual(written["pair"]["peer_b"], "claude")
        self.assertEqual(written["pair"]["cwd"], str(ROOT))

    def test_configure_rejects_unknown_existing_access_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.local.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "ai_debate": {"access": "local-cli", "peer_b": "codex"},
                        "pair": {
                            "access": "unknown-access",
                            "peer_b": "codex",
                            "cwd": str(ROOT),
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            status, emitted = run_main_json(
                [
                    "configure",
                    "--yes",
                    "--settings-path",
                    str(settings_path),
                    "--json",
                ]
            )

        self.assertEqual(status, 1)
        self.assertFalse(emitted["ok"])
        self.assertIn("pair.access must be one of", emitted["message"])

    def test_configure_requires_explicit_ai_debate_peer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status, emitted = run_main_json(
                [
                    "configure",
                    "--settings-path",
                    str(Path(tmpdir) / "settings.local.json"),
                    "--pair-peer-b",
                    "codex",
                    "--json",
                ]
            )

        self.assertEqual(status, 1)
        self.assertFalse(emitted["ok"])
        self.assertIn("ai_debate.peer_b", emitted["message"])

    def test_ai_debate_settings_are_required_for_the_fixed_package(self) -> None:
        payload = {
            "agents": {
                "codex": {"model": "codex-model", "effort": "high"},
            },
            "pair": {
                "access": "native-subagent",
                "peer_b": "codex",
                "cwd": str(ROOT),
            },
        }

        with self.assertRaisesRegex(configuration.ConfigurationError, "ai_debate"):
            configuration.validate_settings_payload(payload, source_root=ROOT)

        payload["ai_debate"] = {"access": "local-cli", "peer_b": "codex"}
        configuration.validate_settings_payload(payload, source_root=ROOT)

        payload["ai_debate"]["peer_b"] = "claude"
        payload.pop("agents", None)
        configuration.validate_settings_payload(payload, source_root=ROOT)


    def test_native_pair_rejects_claude_peer_b(self) -> None:
        payload = {
            "agents": {
                "codex": {"model": "codex-model", "effort": "high"},
                "claude": {"model": "claude-model", "effort": "max"},
            },
            "ai_debate": {"access": "local-cli", "peer_b": "codex"},
            "pair": {
                "access": "native-subagent",
                "peer_b": "claude",
                "cwd": str(ROOT),
            },
        }

        with self.assertRaisesRegex(
            configuration.ConfigurationError,
            "pair.peer_b must be codex for native-subagent access",
        ):
            configuration.validate_settings_payload(payload, source_root=ROOT)

    def test_local_cli_pair_needs_no_provider_model_settings(self) -> None:
        payload = {
            "agents": {
                "codex": {"model": "codex-model", "effort": "high"},
            },
            "ai_debate": {"access": "local-cli", "peer_b": "codex"},
            "pair": {
                "access": "local-cli",
                "peer_b": "claude",
                "cwd": str(ROOT),
            },
        }

        payload.pop("agents", None)
        configuration.validate_settings_payload(payload, source_root=ROOT)

    def test_configure_requires_yes_or_explicit_choices_without_prompting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status, emitted = run_main_json(
                [
                    "configure",
                    "--settings-path",
                    str(Path(tmpdir) / "settings.local.json"),
                    "--json",
                ]
            )

        self.assertEqual(status, 1)
        self.assertFalse(emitted["ok"])
        self.assertIn("does not prompt during install", emitted["message"])
        self.assertIn("post-install", emitted["message"])

    def test_diagnose_reports_handoff_facts_without_codex_plugin_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            status, emitted = run_main_json(
                [
                    "--output-root",
                    str(output_root),
                    "diagnose",
                    "--skip-health",
                    "--json",
                ]
            )

            self.assertFalse((output_root / "plugins" / "tokimon").exists())

        self.assertEqual(status, 2)
        self.assertFalse(emitted["ok"])
        self.assertEqual(emitted["handoff"]["install_route"]["liveCheckoutPath"], str(ROOT))
        self.assertNotIn("account_level_commands", emitted["handoff"])
        self.assertEqual(
            emitted["handoff"]["account_level_mutation"],
            "not_performed_by_ai_harness_interface",
        )
        self.assertIn("plugin-creator", emitted["handoff"]["account_level_install_owner"])
        self.assertNotIn("codex plugin", json.dumps(emitted))

    def test_diagnose_json_reports_timeout_partial_output_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            expired = subprocess.TimeoutExpired("health", 30, output=b"partial \xff", stderr=b"error \xfe")
            with mock.patch.object(mcp.subprocess, "run", side_effect=expired):
                status, emitted = run_main_json([
                    "--output-root", str(Path(tmpdir) / "out"), "diagnose", "--json",
                ])

        self.assertEqual(status, 2)
        self.assertFalse(emitted["ok"])
        health = next(check for check in emitted["checks"] if check["name"] == "mcp-route-back-health")
        self.assertFalse(health["ok"])
        self.assertEqual({item["server_id"] for item in health["items"]}, {"tokimon-telegram-bot", "tokimon-ai-debate"})
        for item in health["items"]:
            self.assertEqual(item["status"], "timeout")
            self.assertEqual(item["stdout"], "partial \ufffd")
            self.assertEqual(item["stderr"], "error \ufffd")

    def test_validate_json_rejects_nonobject_mcp_server_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _ = generate_codex_fixture(Path(tmpdir))
            manifest_path = paths.package_root / ".mcp.json"
            original = json.loads(manifest_path.read_text(encoding="utf-8"))
            for value in (None, [], "invalid"):
                with self.subTest(value=value):
                    payload = json.loads(json.dumps(original))
                    server_id = contract.MCP_ENTRIES[0].server_id
                    payload["mcpServers"][server_id] = value
                    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                    status, emitted = run_main_json([
                        "--output-root", str(paths.output_root), "validate",
                        "--settings-path", str(Path(tmpdir) / "settings.local.json"), "--json",
                    ])
                    self.assertEqual(status, 2)
                    self.assertFalse(emitted["ok"])
                    self.assertIn("mcp-manifest", emitted["message"])

    def test_generate_command_writes_claude_package_through_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            output_root = Path(tmpdir) / "out"
            settings_path = Path(tmpdir) / "settings.local.json"
            configure_status, _ = run_main_json(
                ["configure", "--yes", "--settings-path", str(settings_path), "--json"]
            )
            status, emitted = run_main_json(
                [
                    "--output-root",
                    str(output_root),
                    "generate",
                    "--target-platform",
                    "claude",
                    "--settings-path",
                    str(settings_path),
                    "--json",
                ]
            )
            manifest_exists = (output_root / "plugins" / "tokimon" / ".claude-plugin" / "plugin.json").exists()

        self.assertEqual(configure_status, 0)
        self.assertEqual(status, 0)
        self.assertTrue(emitted["ok"])
        self.assertEqual(emitted["target_platform"], "claude")
        self.assertTrue(manifest_exists)

    def test_diagnose_reports_claude_handoff_facts_without_install_command_plans(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            status, emitted = run_main_json(
                [
                    "--output-root",
                    str(Path(tmpdir)),
                    "diagnose",
                    "--target-platform",
                    "claude",
                    "--skip-health",
                    "--json",
                ]
            )

        self.assertEqual(status, 2)
        self.assertFalse(emitted["ok"])
        handoff = emitted["handoff"]
        self.assertEqual(handoff["target_platform"], "claude")
        self.assertIn("Claude Code plugin CLI", handoff["account_level_install_owner"])
        self.assertNotIn("account_level_commands", handoff)
        self.assertEqual(handoff["account_level_mutation"], "not_performed_by_ai_harness_interface")
        rendered = json.dumps(emitted)
        self.assertNotIn("plugin install", rendered)
        self.assertNotIn("marketplace add", rendered)

    def test_generate_command_fails_when_scaffold_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status, emitted = run_main_json(
                [
                    "--output-root",
                    tmpdir,
                    "generate",
                    "--json",
                ]
            )

        self.assertEqual(status, 1)
        self.assertFalse(emitted["ok"])
        self.assertIn("missing plugin-creator scaffold files", emitted["message"])

    def test_module_entrypoint_contract_json(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in (str(ROOT), env.get("PYTHONPATH", "")) if path
        )

        completed = subprocess.run(
            [sys.executable, "-m", "tokimon.ai_harness_interface", "contract", "--json"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        emitted = json.loads(completed.stdout)
        self.assertTrue(emitted["ok"])
        self.assertEqual(emitted["contract"]["package_id"], "tokimon")


if __name__ == "__main__":
    unittest.main()
