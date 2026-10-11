from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import tokimon.runtime as runtime_helpers
from tokimon.tools.mcp import remote_browser_broker


def subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("parser has no subcommands")


class RemoteBrowserBrokerCliContractTests(unittest.TestCase):
    def test_remote_browser_broker_cli_contract(self) -> None:
        parser = remote_browser_broker.build_parser()

        self.assertEqual(
            subcommand_names(parser),
            {"serve", "serve-http", "health", "build-image"},
        )

        stdio_args = parser.parse_args(["serve"])
        self.assertEqual(stdio_args.command, "serve")

        serve_args = parser.parse_args(["serve-http", "--host", "127.0.0.1", "--port", "8937"])
        self.assertEqual(serve_args.command, "serve-http")
        self.assertEqual(serve_args.host, "127.0.0.1")
        self.assertEqual(serve_args.port, 8937)

        health_args = parser.parse_args(["health", "--json"])
        self.assertEqual(health_args.command, "health")
        self.assertTrue(health_args.json)

    def test_broker_config_derives_installation_scoped_image_from_env(self) -> None:
        config = remote_browser_broker.BrokerConfig.from_env(
            env={"TOKIMON_RB_INSTALLATION_ID": "Joey-Test"}
        )

        self.assertEqual(config.installation_id, "Joey-Test")
        self.assertEqual(config.image_name, "tokimon-remote-browser-session:joey-test")
        self.assertEqual(config.broker_port, remote_browser_broker.default_broker_port("Joey-Test"))
        self.assertGreaterEqual(config.broker_port, remote_browser_broker.DEFAULT_PORT_BASE)
        self.assertLess(
            config.broker_port,
            remote_browser_broker.DEFAULT_PORT_BASE + remote_browser_broker.DEFAULT_PORT_SPAN,
        )
        self.assertLess(config.broker_port, 32768)

    def test_broker_config_accepts_explicit_port_override(self) -> None:
        config = remote_browser_broker.BrokerConfig.from_env(
            env={
                "TOKIMON_RB_INSTALLATION_ID": "Joey-Test",
                "TOKIMON_RB_PORT": "8937",
            }
        )

        self.assertEqual(config.broker_port, 8937)

    def test_capacity_and_idle_policy_survive_http_config_overrides(self) -> None:
        config = remote_browser_broker.BrokerConfig.from_env(env={
            "TOKIMON_RB_MAX_ACTIVE_SESSIONS": "2",
            "TOKIMON_RB_MAX_TOTAL_SESSIONS": "5",
            "TOKIMON_RB_IDLE_TIMEOUT_SECONDS": "60",
        })
        overridden = remote_browser_broker.replace_config(config, bind_host="127.0.0.1", broker_port=8937)
        self.assertEqual(overridden.max_active_sessions, 2)
        self.assertEqual(overridden.max_total_sessions, 5)
        self.assertEqual(overridden.idle_timeout_seconds, 60)
        with self.assertRaises(remote_browser_broker.BrowserBrokerError):
            remote_browser_broker.BrokerConfig.from_env(env={"TOKIMON_RB_IDLE_TIMEOUT_SECONDS": "-1"})

    def test_broker_config_rejects_invalid_installation_id(self) -> None:
        with self.assertRaisesRegex(
            remote_browser_broker.BrowserBrokerError,
            "TOKIMON_RB_INSTALLATION_ID",
        ):
            remote_browser_broker.BrokerConfig.from_env(
                env={"TOKIMON_RB_INSTALLATION_ID": "../outside"}
            )

    def test_broker_config_rejects_linked_worktree_runtime_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primary = root / "primary"
            linked = root / "linked"
            linked.mkdir()
            worktree_git_dir = primary / ".git" / "worktrees" / "linked"
            worktree_git_dir.mkdir(parents=True)
            (linked / ".git").write_text(f"gitdir: {worktree_git_dir}\n", encoding="utf-8")
            previous_root = remote_browser_broker.WORKSPACE_ROOT
            try:
                remote_browser_broker.WORKSPACE_ROOT = linked
                with self.assertRaisesRegex(runtime_helpers.RuntimePathError, "linked worktree"):
                    remote_browser_broker.BrokerConfig.from_env(
                        env={
                            "TOKIMON_RB_RUNTIME_ROOT": str(
                                linked / ".tokimon" / "tools" / "mcp" / "remote-browser"
                            )
                        }
                    )
            finally:
                remote_browser_broker.WORKSPACE_ROOT = previous_root


if __name__ == "__main__":
    unittest.main()
