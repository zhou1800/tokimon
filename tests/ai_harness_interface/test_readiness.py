from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.ai_harness_interface.support import without_local_claude_cli, without_local_plugin_creator
from tokimon.ai_harness_interface import claude_cli, contract, generator, readiness


def seed_scaffold(paths: generator.GenerationPaths) -> None:
    (paths.package_root / ".codex-plugin").mkdir(parents=True, exist_ok=True)
    (paths.package_root / "skills").mkdir(parents=True, exist_ok=True)
    paths.marketplace_path.parent.mkdir(parents=True, exist_ok=True)
    (paths.package_root / ".codex-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "tokimon",
                "version": "0.1.0",
                "description": "Seeded plugin",
                "author": {"name": "Local developer"},
                "skills": "./skills/",
                "mcpServers": "./.mcp.json",
                "interface": {
                    "displayName": "Tokimon",
                    "shortDescription": "Use Tokimon in Codex.",
                    "longDescription": "Tokimon adds a local Codex plugin scaffold.",
                    "developerName": "Local developer",
                    "category": "Productivity",
                    "capabilities": [],
                    "defaultPrompt": "Help me use Tokimon.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (paths.package_root / ".mcp.json").write_text('{"mcpServers": {}}\n', encoding="utf-8")
    paths.marketplace_path.write_text(
        json.dumps(
            {
                "name": "tokimon-local",
                "interface": {"displayName": "Tokimon Local"},
                "plugins": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_settings(
    path: Path,
    *,
    source_root: Path = ROOT,
) -> None:
    path.write_text(
        json.dumps(
            {
                "agents": {
                    "codex": {"model": "gpt-test", "effort": "xhigh"},
                    "claude": {"model": "claude-test", "effort": "max"},
                },
                "ai_debate": {
                    "access": "local-cli",
                    "peer_b": "claude",
                },
                "pair": {
                    "access": "local-cli",
                    "peer_b": "claude",
                    "cwd": str(source_root),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_native_settings(path: Path, *, source_root: Path = ROOT) -> None:
    path.write_text(
        json.dumps(
            {
                "agents": {
                    "codex": {"model": "gpt-test", "effort": "xhigh"},
                },
                "ai_debate": {
                    "access": "local-cli",
                    "peer_b": "codex",
                },
                "pair": {
                    "access": "native-subagent",
                    "peer_b": "codex",
                    "cwd": str(source_root),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def all_registered_runner(command: list[str] | tuple[str, ...], cwd: Path, timeout_sec: int) -> readiness.CommandResult:
    if command == ["codex", "plugin", "list"]:
        return readiness.CommandResult(
            0,
            "tokimon@tokimon-local  installed, enabled  0.1.0  /tmp/tokimon\n",
            "",
        )
    if list(command[:3]) == ["codex", "mcp", "get"]:
        server_id = command[3]
        return readiness.CommandResult(
            0,
            f"{server_id}\n  enabled: true\n  transport: stdio\n",
            "",
        )
    raise AssertionError(f"unexpected command: {command}")


def missing_telegram_runner(
    command: list[str] | tuple[str, ...],
    cwd: Path,
    timeout_sec: int,
) -> readiness.CommandResult:
    if command == ["codex", "plugin", "list"]:
        return all_registered_runner(command, cwd, timeout_sec)
    if list(command) == ["codex", "mcp", "get", "tokimon-telegram-bot"]:
        return readiness.CommandResult(1, "", "not found")
    return all_registered_runner(command, cwd, timeout_sec)


def disabled_telegram_runner(
    command: list[str] | tuple[str, ...],
    cwd: Path,
    timeout_sec: int,
) -> readiness.CommandResult:
    if list(command) == ["codex", "mcp", "get", "tokimon-telegram-bot"]:
        return readiness.CommandResult(
            0,
            "tokimon-telegram-bot\n  enabled: false\n  transport: stdio\n",
            "",
        )
    return all_registered_runner(command, cwd, timeout_sec)


def claude_plugin_list_raw_runner(result: readiness.CommandResult):
    def runner(
        command: list[str] | tuple[str, ...],
        cwd: Path,
        timeout_sec: int,
    ) -> readiness.CommandResult:
        if list(command) == claude_cli.plugin_list_command():
            return result
        raise AssertionError(f"unexpected command: {command}")

    return runner


def claude_plugin_list_runner(payload: object):
    return claude_plugin_list_raw_runner(readiness.CommandResult(0, json.dumps(payload), ""))


def rejecting_runner(
    command: list[str] | tuple[str, ...],
    cwd: Path,
    timeout_sec: int,
) -> readiness.CommandResult:
    raise AssertionError(f"platform command must not run: {command}")


def fake_health(
    *,
    telegram_ready: bool = False,
    ai_debate_authentication_checked: bool | None = None,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for entry in contract.MCP_ENTRIES:
        parsed: dict[str, object] = {"ok": True}
        if entry.capability_id == "mcp.telegram-bot":
            parsed = {
                "ok": True,
                "token_configured": telegram_ready,
                "default_chat_configured": telegram_ready,
                "ready_for_send": telegram_ready,
                "secrets_path": str(ROOT / ".tokimon" / "secrets.local.json"),
            }
        elif (
            entry.capability_id == "mcp.ai-debate"
            and ai_debate_authentication_checked is not None
        ):
            parsed["authentication_checked"] = ai_debate_authentication_checked
        items.append(
            {
                "server_id": entry.server_id,
                "capability_id": entry.capability_id,
                "command": [".venv/bin/python", "-m", entry.module, "health", "--json"],
                "ok": True,
                "status": "pass",
                "returncode": 0,
                "parsed_stdout": parsed,
            }
        )
    return items


def unavailable_external_prerequisite_health() -> list[dict[str, object]]:
    items = fake_health(telegram_ready=True)
    for item in items:
        if item["capability_id"] == "mcp.ai-debate":
            item["returncode"] = 1
            item["parsed_stdout"] = {
                "ok": False,
                "providers": {
                    "codex": {"available": True},
                    "claude": {"available": False},
                },
            }
        elif item["capability_id"] == "mcp.remote-browser-broker":
            item["returncode"] = 1
            item["parsed_stdout"] = {
                "ok": False,
                "docker": {"path": None},
            }
    return items


class AiHarnessReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(without_local_plugin_creator())
        self.enterContext(mock.patch("tokimon.ai_harness_interface.installation_policy.which", return_value="/test/docker"))

    def test_docker_free_install_checks_only_selected_servers_and_never_requests_docker_install(self) -> None:
        calls = []

        def runner(command, cwd, timeout_sec):
            calls.append(list(command))
            return all_registered_runner(command, cwd, timeout_sec)

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "tokimon.ai_harness_interface.installation_policy.which", return_value=None,
        ):
            paths, settings_path = self.build_generated_paths(tmpdir)
            with mock.patch.object(readiness.mcp, "run_health_checks", return_value=fake_health(telegram_ready=True)) as health:
                report = readiness.post_install_readiness(paths=paths, settings_path=settings_path, command_runner=runner, executable_resolver=lambda name: f"/test/{name}")
        self.assertTrue(report["ok"], report["remaining_work"])
        self.assertEqual(report["summary"]["mcp_registration"]["total"], 2)
        self.assertIn("mcp.remote-browser-broker", report["installation_policy"]["disabled_capabilities"])
        self.assertFalse(any("tokimon-remote-browser-broker" in command for command in calls))
        self.assertEqual(len(health.call_args.kwargs["entries"]), 2)
        self.assertFalse(any("Docker" in action for action in report["capability_setup"]))

    def build_generated_paths(self, tmpdir: str) -> tuple[generator.GenerationPaths, Path]:
        paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir) / "out")
        settings_path = Path(tmpdir) / "settings.local.json"
        write_settings(settings_path)
        seed_scaffold(paths)
        generator.generate_package(paths, settings_path=settings_path)
        return paths, settings_path

    def build_claude_generated_paths(self, tmpdir: str) -> tuple[generator.GenerationPaths, Path]:
        paths = generator.paths_for(
            source_root=ROOT,
            output_root=Path(tmpdir) / "out",
            target_platform="claude",
        )
        settings_path = Path(tmpdir) / "settings.local.json"
        write_settings(settings_path)
        generator.generate_package(paths, settings_path=settings_path)
        return paths, settings_path

    def test_post_install_claude_reports_ready_when_plugin_installed_and_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths, settings_path = self.build_claude_generated_paths(tmpdir)
            runner = claude_plugin_list_runner(
                [{"name": "tokimon", "marketplace": "tokimon-local", "version": "0.1.0", "enabled": True}]
            )
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    target_platform="claude",
                    command_runner=runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        self.assertTrue(report["ok"], report["remaining_work"])
        self.assertEqual(report["target_platform"]["target_platform"], "claude")
        self.assertTrue(report["target_platform"]["plugin"]["installed_enabled"])
        self.assertEqual(report["summary"]["plugin_installed_enabled"], True)
        self.assertEqual(
            report["summary"]["mcp_registration"],
            {
                "checked": True,
                "registered": len(contract.MCP_ENTRIES),
                "enabled": len(contract.MCP_ENTRIES),
                "total": len(contract.MCP_ENTRIES),
            },
        )
        self.assertEqual(
            report["summary"]["mcp_route_back_health"],
            {
                "checked": True,
                "pass": len(contract.MCP_ENTRIES),
                "total": len(contract.MCP_ENTRIES),
            },
        )
        for server in report["target_platform"]["mcp_servers"].values():
            self.assertEqual(server["derived_from"], "plugin_status")
            self.assertTrue(server["enabled"])

    def test_post_install_claude_accepts_qualified_plugin_name_and_is_enabled_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths, settings_path = self.build_claude_generated_paths(tmpdir)
            runner = claude_plugin_list_runner([{"plugin": "tokimon@tokimon-local", "isEnabled": True}])
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    target_platform="claude",
                    command_runner=runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        self.assertTrue(report["target_platform"]["plugin"]["installed_enabled"])

    def test_post_install_claude_handles_plugin_list_command_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths, settings_path = self.build_claude_generated_paths(tmpdir)
            runner = claude_plugin_list_raw_runner(readiness.CommandResult(1, "", "boom"))
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    target_platform="claude",
                    command_runner=runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        self.assertFalse(report["ok"])
        plugin = report["target_platform"]["plugin"]
        self.assertFalse(plugin["ok"])
        self.assertIsNone(plugin["installed_enabled"])
        self.assertEqual(plugin["inventory_inspection"]["status"], "command_failed")
        self.assertIn(
            "Fix `claude plugin list --json`",
            " ".join(report["remaining_work"]),
        )
        self.assertNotIn("Refresh the target-platform installation", " ".join(report["remaining_work"]))

    def test_post_install_claude_handles_non_json_plugin_list_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths, settings_path = self.build_claude_generated_paths(tmpdir)
            runner = claude_plugin_list_raw_runner(
                readiness.CommandResult(0, "Installed plugins:\n tokimon\n", "")
            )
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    target_platform="claude",
                    command_runner=runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        self.assertFalse(report["ok"])
        plugin = report["target_platform"]["plugin"]
        self.assertFalse(plugin["ok"])
        self.assertIsNone(plugin["installed_enabled"])
        self.assertEqual(plugin["inventory_inspection"]["status"], "invalid_json")
        self.assertEqual(
            plugin["inventory_inspection"]["message"],
            "Claude plugin inventory output was not valid JSON.",
        )
        self.assertIn(
            "resolve its invalid JSON output",
            " ".join(report["remaining_work"]),
        )
        self.assertNotIn("Refresh the target-platform installation", " ".join(report["remaining_work"]))

    def test_post_install_claude_handles_non_list_plugin_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths, settings_path = self.build_claude_generated_paths(tmpdir)
            runner = claude_plugin_list_runner({"plugins": []})
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    target_platform="claude",
                    command_runner=runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        plugin = report["target_platform"]["plugin"]
        self.assertFalse(report["ok"])
        self.assertFalse(plugin["ok"])
        self.assertIsNone(plugin["installed_enabled"])
        self.assertEqual(plugin["inventory_inspection"]["status"], "invalid_shape")
        self.assertEqual(
            plugin["inventory_inspection"]["message"],
            "Claude plugin inventory output must be a JSON array.",
        )
        self.assertIn("unexpected output shape", " ".join(report["remaining_work"]))
        self.assertNotIn("Refresh the target-platform installation", " ".join(report["remaining_work"]))

    def test_post_install_claude_treats_explicit_disabled_flag_as_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths, settings_path = self.build_claude_generated_paths(tmpdir)
            runner = claude_plugin_list_runner(
                [{"name": "tokimon", "marketplace": "tokimon-local", "enabled": False}]
            )
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    target_platform="claude",
                    command_runner=runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        self.assertFalse(report["ok"])
        self.assertFalse(report["target_platform"]["plugin"]["installed_enabled"])
        for server in report["target_platform"]["mcp_servers"].values():
            self.assertTrue(server["registered"])
            self.assertFalse(server["enabled"])

    def test_post_install_claude_defaults_to_enabled_when_no_flag_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths, settings_path = self.build_claude_generated_paths(tmpdir)
            runner = claude_plugin_list_runner([{"plugin": "tokimon@tokimon-local"}])
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    target_platform="claude",
                    command_runner=runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        self.assertTrue(report["target_platform"]["plugin"]["installed_enabled"])

    def test_post_install_claude_skips_platform_inspection_when_cli_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths, settings_path = self.build_claude_generated_paths(tmpdir)
            claude_name = claude_cli.claude_executable()
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    target_platform="claude",
                    command_runner=rejecting_runner,
                    executable_resolver=lambda name: None if name == claude_name else f"/usr/bin/{name}",
                )

        platform = report["target_platform"]
        self.assertTrue(platform["ok"])
        self.assertTrue(platform["plugin"]["skipped"])
        self.assertIsNone(platform["plugin"]["installed_enabled"])
        for server in platform["mcp_servers"].values():
            self.assertEqual(server["status"], "skipped")
            self.assertIsNone(server["enabled"])
        self.assertFalse(report["summary"]["mcp_registration"]["checked"])
        self.assertNotIn(
            "Refresh the target-platform installation",
            " ".join(report["remaining_work"]),
        )

    def test_post_install_claude_fails_closed_when_cli_required_but_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths, settings_path = self.build_claude_generated_paths(tmpdir)
            claude_name = claude_cli.claude_executable()
            with mock.patch.dict("os.environ", {claude_cli.REQUIRE_ENV: "1"}), mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    target_platform="claude",
                    command_runner=rejecting_runner,
                    executable_resolver=lambda name: None if name == claude_name else f"/usr/bin/{name}",
                )

        platform = report["target_platform"]
        self.assertFalse(platform["ok"])
        self.assertTrue(platform["plugin"]["skipped"])
        self.assertIn(
            "Install the Claude Code CLI",
            " ".join(report["remaining_work"]),
        )
        self.assertNotIn(
            "Refresh the target-platform installation",
            " ".join(report["remaining_work"]),
        )

    def test_post_install_claude_reports_missing_plugin_installation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths, settings_path = self.build_claude_generated_paths(tmpdir)
            runner = claude_plugin_list_runner([])
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    target_platform="claude",
                    command_runner=runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        self.assertFalse(report["ok"])
        self.assertFalse(report["target_platform"]["plugin"]["installed_enabled"])
        self.assertFalse(report["summary"]["plugin_installed_enabled"])
        self.assertEqual(
            report["summary"]["mcp_registration"],
            {
                "checked": True,
                "registered": 0,
                "enabled": 0,
                "total": len(contract.MCP_ENTRIES),
            },
        )
        self.assertIn(
            "Refresh the target-platform installation",
            " ".join(report["remaining_work"]),
        )

    def test_post_install_reports_telegram_capability_setup_when_route_health_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, settings_path = self.build_generated_paths(tmpdir)
            with mock.patch.object(readiness.mcp, "run_health_checks", return_value=fake_health()):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    command_runner=all_registered_runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        telegram = next(item for item in report["mcp_servers"] if item["server_id"] == "tokimon-telegram-bot")
        rendered = json.dumps(report)
        self.assertTrue(report["ok"], report["remaining_work"])
        self.assertEqual(
            report["summary"]["mcp_registration"],
            {
                "checked": True,
                "registered": len(contract.MCP_ENTRIES),
                "enabled": len(contract.MCP_ENTRIES),
                "total": len(contract.MCP_ENTRIES),
            },
        )
        self.assertEqual(
            report["summary"]["mcp_route_back_health"],
            {
                "checked": True,
                "pass": len(contract.MCP_ENTRIES),
                "total": len(contract.MCP_ENTRIES),
            },
        )
        self.assertFalse(telegram["operational"])
        self.assertEqual(report["remaining_work"], [])
        self.assertIn("Configure the Telegram bot token", " ".join(report["capability_setup"]))
        self.assertNotIn("bot_token", rendered)
        self.assertNotIn("chat_id", rendered)

    def test_post_install_reports_pair_configuration_and_provider_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, settings_path = self.build_generated_paths(tmpdir)
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    command_runner=all_registered_runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        self.assertTrue(report["ok"], report["remaining_work"])
        self.assertEqual(report["pair"]["access"], "local-cli")
        self.assertEqual(report["pair"]["peer_b"], "claude")
        self.assertTrue(report["pair"]["cwd_matches_live_checkout"])
        self.assertTrue(report["pair"]["provider_commands"]["codex"]["available"])
        self.assertTrue(report["pair"]["provider_commands"]["claude"]["available"])
        self.assertEqual(
            report["pair"]["provider_commands"]["codex"]["authentication"],
            "not_checked",
        )


    def test_post_install_reports_native_pair_without_provider_cli_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir) / "out")
            settings_path = Path(tmpdir) / "settings.local.json"
            write_native_settings(settings_path)
            seed_scaffold(paths)
            generator.generate_package(paths, settings_path=settings_path)
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(
                    telegram_ready=True,
                    ai_debate_authentication_checked=False,
                ),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    command_runner=all_registered_runner,
                    executable_resolver=lambda name: None,
                )

        self.assertTrue(report["ok"], report["remaining_work"])
        self.assertEqual(report["pair"]["access"], "native-subagent")
        self.assertEqual(report["pair"]["peer_b"], "codex")
        self.assertEqual(report["pair"]["provider_commands"], {})
        self.assertTrue(
            report["pair"]["native_subagents"]["target_platform_supported"]
        )
        self.assertEqual(report["pair"]["native_subagents"]["spawn_check"], "not_run")
        ai_debate = next(
            item
            for item in report["mcp_servers"]
            if item["capability_id"] == "mcp.ai-debate"
        )
        self.assertIsNone(ai_debate["operational"])
        self.assertFalse(report["summary"]["mcp_operational"]["checked"])

    def test_post_install_treats_missing_external_prerequisites_as_non_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, settings_path = self.build_generated_paths(tmpdir)
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=unavailable_external_prerequisite_health(),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    command_runner=all_registered_runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        servers = {item["capability_id"]: item for item in report["mcp_servers"]}
        self.assertTrue(report["ok"], report["remaining_work"])
        self.assertEqual(report["remaining_work"], [])
        self.assertTrue(servers["mcp.ai-debate"]["route_back_health"])
        self.assertFalse(servers["mcp.ai-debate"]["operational"])
        self.assertTrue(servers["mcp.remote-browser-broker"]["route_back_health"])
        self.assertFalse(servers["mcp.remote-browser-broker"]["operational"])
        setup = " ".join(report["capability_setup"])
        self.assertIn("AI Debate provider CLI(s) unavailable: claude", setup)
        self.assertIn("Docker is unavailable", setup)

    def test_post_install_only_reports_required_ai_debate_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(
                source_root=ROOT,
                output_root=Path(tmpdir) / "out",
            )
            settings_path = Path(tmpdir) / "settings.local.json"
            write_native_settings(settings_path)
            seed_scaffold(paths)
            generator.generate_package(paths, settings_path=settings_path)
            health = fake_health(telegram_ready=True)
            ai_debate_health = next(
                item
                for item in health
                if item["capability_id"] == "mcp.ai-debate"
            )
            ai_debate_health["parsed_stdout"] = {
                "ok": False,
                "required_providers": ["codex"],
                "providers": {
                    "codex": {"available": False},
                    "claude": {"available": False},
                },
            }
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=health,
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    command_runner=all_registered_runner,
                    executable_resolver=lambda name: None,
                )

        ai_debate_action = next(
            action
            for action in report["capability_setup"]
            if "AI Debate provider CLI" in action
        )
        self.assertEqual(
            ai_debate_action,
            "Required AI Debate provider CLI(s) unavailable: codex. Configure an existing provider or leave this capability inactive.",
        )

    def test_post_install_treats_registered_but_disabled_codex_mcp_as_non_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, settings_path = self.build_generated_paths(tmpdir)
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    command_runner=disabled_telegram_runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        telegram = next(
            item for item in report["mcp_servers"] if item["server_id"] == "tokimon-telegram-bot"
        )
        self.assertTrue(report["ok"], report["remaining_work"])
        self.assertTrue(telegram["registered"])
        self.assertFalse(telegram["enabled"])
        self.assertEqual(report["remaining_work"], [])
        self.assertIn("Enable tokimon-telegram-bot", " ".join(report["capability_setup"]))

    def test_post_install_reports_missing_codex_mcp_registration_by_server_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, settings_path = self.build_generated_paths(tmpdir)
            with mock.patch.object(
                readiness.mcp,
                "run_health_checks",
                return_value=fake_health(telegram_ready=True),
            ):
                report = readiness.post_install_readiness(
                    paths=paths,
                    settings_path=settings_path,
                    command_runner=missing_telegram_runner,
                    executable_resolver=lambda name: f"/usr/bin/{name}",
                )

        telegram = next(item for item in report["mcp_servers"] if item["server_id"] == "tokimon-telegram-bot")
        self.assertFalse(report["ok"])
        self.assertFalse(telegram["registered"])
        self.assertFalse(telegram["enabled"])
        self.assertIn("tokimon-telegram-bot is registered", " ".join(report["remaining_work"]))
        self.assertNotEqual(report["next_steps"], report["remaining_work"])
        self.assertEqual(
            report["next_steps"],
            ["Resolve blocking remaining work, then rerun post-install."],
        )

    def test_post_install_can_skip_platform_and_health_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, settings_path = self.build_generated_paths(tmpdir)
            report = readiness.post_install_readiness(
                paths=paths,
                settings_path=settings_path,
                check_platform=False,
                run_health=False,
                command_runner=missing_telegram_runner,
                executable_resolver=lambda name: f"/usr/bin/{name}",
            )

        self.assertTrue(report["target_platform"]["ok"])
        self.assertFalse(report["target_platform"]["checked"])
        self.assertIsNone(report["mcp_servers"][0]["route_back_health"])
        self.assertIsNone(report["mcp_servers"][0]["operational"])
        self.assertFalse(report["summary"]["mcp_registration"]["checked"])
        self.assertFalse(report["summary"]["mcp_route_back_health"]["checked"])
        self.assertFalse(report["summary"]["mcp_operational"]["checked"])


if __name__ == "__main__":
    unittest.main()
