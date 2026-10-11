from __future__ import annotations

import argparse
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import tokimon.runtime as runtime_helpers
from tokimon.tools.mcp import remote_agent_broker


def subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("parser has no subcommands")


class RemoteAgentBrokerCliContractTests(unittest.TestCase):
    def test_remote_agent_broker_cli_contract(self) -> None:
        parser = remote_agent_broker.build_parser()

        self.assertEqual(subcommand_names(parser), {"serve", "serve-http", "health", "cleanup"})

        stdio_args = parser.parse_args(["serve"])
        self.assertEqual(stdio_args.command, "serve")

        serve_args = parser.parse_args(["serve-http", "--host", "127.0.0.1", "--port", "8938"])
        self.assertEqual(serve_args.command, "serve-http")
        self.assertEqual(serve_args.host, "127.0.0.1")
        self.assertEqual(serve_args.port, 8938)

        health_args = parser.parse_args(["health", "--json"])
        self.assertEqual(health_args.command, "health")
        self.assertTrue(health_args.json)

        cleanup_args = parser.parse_args(["cleanup", "--ttl-days", "7", "--json"])
        self.assertEqual(cleanup_args.command, "cleanup")
        self.assertEqual(cleanup_args.ttl_days, 7)
        self.assertTrue(cleanup_args.json)

        for invalid_port in ("0", "65536", "not-a-port"):
            with self.subTest(port=invalid_port):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(["serve-http", "--port", invalid_port])

    def test_broker_config_rejects_invalid_port_values(self) -> None:
        for invalid_port in ("0", "65536", "not-a-port"):
            with self.subTest(port=invalid_port):
                with self.assertRaisesRegex(
                    remote_agent_broker.RemoteAgentBrokerError,
                    "broker_port|TOKIMON_RA_PORT",
                ):
                    remote_agent_broker.BrokerConfig.from_env(
                        env={"TOKIMON_RA_PORT": invalid_port}
                    )

    def test_broker_config_rejects_nonpositive_runtime_limits(self) -> None:
        cases = (
            ("TOKIMON_RA_DEFAULT_TTL_DAYS", "0", "default_ttl_days"),
            ("TOKIMON_RA_MAX_READ_BYTES", "0", "max_read_bytes"),
            ("TOKIMON_RA_MAX_READ_BYTES", "-5", "max_read_bytes"),
            (
                "TOKIMON_RA_SHELL_SYNC_TIMEOUT_SEC",
                "0",
                "shell_sync_timeout_sec",
            ),
        )
        for variable, value, expected_field in cases:
            with self.subTest(variable=variable, value=value):
                with self.assertRaisesRegex(
                    remote_agent_broker.RemoteAgentBrokerError,
                    expected_field,
                ):
                    remote_agent_broker.BrokerConfig.from_env(
                        env={variable: value}
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
            previous_root = remote_agent_broker.WORKSPACE_ROOT
            try:
                remote_agent_broker.WORKSPACE_ROOT = linked
                with self.assertRaisesRegex(runtime_helpers.RuntimePathError, "linked worktree"):
                    remote_agent_broker.BrokerConfig.from_env(
                        env={
                            "TOKIMON_RA_RUNTIME_ROOT": str(
                                linked / ".tokimon" / "tools" / "mcp" / "remote-agent"
                            )
                        }
                    )
            finally:
                remote_agent_broker.WORKSPACE_ROOT = previous_root


if __name__ == "__main__":
    unittest.main()
