from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.tools import registry


class ToolRegistryCliContractTests(unittest.TestCase):
    def test_tool_registry_cli_contract(self) -> None:
        parser = registry.build_parser()

        host_args = parser.parse_args(["host", "--json"])
        self.assertEqual(host_args.command, "host")
        self.assertTrue(host_args.json)

        list_args = parser.parse_args(["list", "--json"])
        self.assertEqual(list_args.command, "list")
        self.assertTrue(list_args.json)

        show_args = parser.parse_args(["show", "mcp.ai-debate", "--json"])
        self.assertEqual(show_args.command, "show")
        self.assertEqual(show_args.capability_id, "mcp.ai-debate")
        self.assertTrue(show_args.json)

        require_args = parser.parse_args(["require", "mcp.ai-debate", "--json"])
        self.assertEqual(require_args.command, "require")
        self.assertEqual(require_args.capability_id, "mcp.ai-debate")
        self.assertTrue(require_args.json)


if __name__ == "__main__":
    unittest.main()
