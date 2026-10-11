from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tokimon.tools.mcp import telegram_bot


def subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("parser has no subcommands")


class TelegramBotMcpCliContractTests(unittest.TestCase):
    def test_telegram_bot_mcp_cli_contract(self) -> None:
        parser = telegram_bot.build_parser()

        self.assertEqual(subcommand_names(parser), {"serve", "health"})

        serve_args = parser.parse_args(["serve"])
        self.assertEqual(serve_args.command, "serve")

        health_args = parser.parse_args(["health", "--live", "--json"])
        self.assertEqual(health_args.command, "health")
        self.assertTrue(health_args.live)
        self.assertTrue(health_args.json)


if __name__ == "__main__":
    unittest.main()
