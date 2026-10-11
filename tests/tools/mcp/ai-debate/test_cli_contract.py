from __future__ import annotations

import argparse
import contextlib
import io
import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

debate = importlib.import_module("tokimon.tools.mcp.ai_debate")


def subcommand_help(parser: argparse.ArgumentParser, command: str) -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            parser.parse_args([command, "--help"])
        except SystemExit as exc:
            if exc.code != 0:
                raise AssertionError(f"{command} --help exited with {exc.code}") from exc
        else:
            raise AssertionError(f"{command} --help did not exit")
    return " ".join(output.getvalue().split())


class AiDebateCliContractTests(unittest.TestCase):
    def test_ai_debate_cli_contract(self) -> None:
        parser = debate.build_parser()
        commands = {
            "serve",
            "health",
            "run",
            "resume",
            "status",
            "turn-start",
            "turn-status",
            "turn-finalize",
            "read-log",
        }
        main_help = " ".join(parser.format_help().split())
        for command in commands:
            self.assertIn(command, main_help)
            subcommand_help(parser, command)

        for command in ("turn-start", "turn-status", "turn-finalize"):
            self.assertIn(
                f"{command} Compatibility endpoint that fails fast in local-only mode",
                main_help,
            )
        lowered_help = main_help.lower()
        self.assertNotIn("asynchronously", lowered_help)
        self.assertNotIn("active async", lowered_help)
        self.assertNotIn("terminal async", lowered_help)
        self.assertNotIn("broker", lowered_help)
        self.assertNotIn("job handle", lowered_help)
        self.assertNotIn("advance debate state", lowered_help)

        run_args = parser.parse_args(
            [
                "run",
                "--topic",
                "T",
                "--side-a",
                "A",
                "--side-b",
                "B",
                "--side-b-provider",
                "codex",
                "--access",
                "local-cli",
                "--max-rounds",
                "1",
                "--json",
            ]
        )
        self.assertEqual(run_args.command, "run")
        self.assertEqual(run_args.side_b_provider, "codex")
        self.assertEqual(run_args.access, "local-cli")
        self.assertEqual(run_args.max_rounds, 1)
        self.assertTrue(run_args.json)

        status_args = parser.parse_args(["status", ".tokimon/tools/mcp/ai-debate/runs/demo", "--json"])
        self.assertEqual(status_args.command, "status")
        self.assertTrue(status_args.json)

        turn_start_args = parser.parse_args(
            ["turn-start", ".tokimon/tools/mcp/ai-debate/runs/demo", "--json"]
        )
        self.assertEqual(turn_start_args.command, "turn-start")
        self.assertTrue(turn_start_args.json)

        turn_status_args = parser.parse_args(
            ["turn-status", ".tokimon/tools/mcp/ai-debate/runs/demo", "--json"]
        )
        self.assertEqual(turn_status_args.command, "turn-status")
        self.assertTrue(turn_status_args.json)

        turn_finalize_args = parser.parse_args(
            ["turn-finalize", ".tokimon/tools/mcp/ai-debate/runs/demo", "--json"]
        )
        self.assertEqual(turn_finalize_args.command, "turn-finalize")
        self.assertTrue(turn_finalize_args.json)

    def test_health_cli_exits_nonzero_when_unhealthy(self) -> None:
        stdout = io.StringIO()

        with (
            patch.object(debate, "load_settings", return_value={}),
            contextlib.redirect_stdout(stdout),
        ):
            status = debate.main(["health", "--json"])

        self.assertEqual(status, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertIn("settings_error", payload)


if __name__ == "__main__":
    unittest.main()
