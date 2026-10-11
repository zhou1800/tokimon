from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import tokimon.memory.mem_palace as mem_palace_package
from tokimon.memory.mem_palace import mem_palace as wm


def subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("parser has no subcommands")


class MemPalaceCliContractTests(unittest.TestCase):
    def test_package_facade_exposes_only_mem_palace(self) -> None:
        self.assertIs(mem_palace_package.MemPalace, wm.MemPalace)
        with self.assertRaises(AttributeError):
            getattr(mem_palace_package, "connect")

    def test_module_entrypoint_runs_without_runtime_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-W",
                    "error::RuntimeWarning",
                    "-m",
                    "tokimon.memory.mem_palace.mem_palace",
                    "--db",
                    str(Path(tmpdir) / "memory.db"),
                    "stats",
                    "--json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_command_set_is_stable(self) -> None:
        parser = wm.build_parser()
        self.assertEqual(
            subcommand_names(parser),
            {
                "add",
                "search",
                "recent",
                "show",
                "briefing",
                "link",
                "entity",
                "invalidate",
                "timeline",
                "stats",
                "backup",
                "export",
                "import",
                "ingest",
                "eval",
                "migrate",
            },
        )

    def test_retrieval_command_arguments(self) -> None:
        parser = wm.build_parser()
        search_args = parser.parse_args(
            ["search", "memory", "--limit", "3", "--scope", "workflow", "--room", "example-workflow", "--json"]
        )
        self.assertEqual(search_args.command, "search")
        self.assertEqual(search_args.query, "memory")
        self.assertEqual(search_args.limit, 3)
        self.assertEqual(search_args.scope, "workflow")
        self.assertEqual(search_args.room, "example-workflow")
        self.assertTrue(search_args.json)

        recent_args = parser.parse_args(["recent", "--scope", "workflow", "--include-scope", "global"])
        self.assertEqual(recent_args.command, "recent")
        self.assertEqual(recent_args.scope, "workflow")
        self.assertEqual(recent_args.include_scope, ["global"])

        briefing_args = parser.parse_args(["briefing", "--limit", "8"])
        self.assertEqual(briefing_args.command, "briefing")
        self.assertEqual(briefing_args.limit, 8)

    def test_add_command_arguments(self) -> None:
        parser = wm.build_parser()
        avoid_args = parser.parse_args(
            [
                "add",
                "--kind",
                "avoid",
                "--summary",
                "Avoid deleting durable memory from use count alone",
            ]
        )
        self.assertEqual(avoid_args.command, "add")
        self.assertEqual(avoid_args.kind, "avoid")

    def test_temporal_command_arguments(self) -> None:
        parser = wm.build_parser()
        link_args = parser.parse_args(
            [
                "link",
                "Mem Palace",
                "stores",
                "local durable memory",
                "--valid-from",
                "2026-04-24",
                "--source-memory-id",
                "12",
                "--json",
            ]
        )
        self.assertEqual(link_args.command, "link")
        self.assertEqual(link_args.subject, "Mem Palace")
        self.assertEqual(link_args.predicate, "stores")
        self.assertEqual(link_args.object, "local durable memory")
        self.assertEqual(link_args.source_memory_id, 12)
        self.assertTrue(link_args.json)

        entity_args = parser.parse_args(["entity", "Mem Palace", "--direction", "both", "--json"])
        self.assertEqual(entity_args.command, "entity")
        self.assertEqual(entity_args.name, "Mem Palace")
        self.assertEqual(entity_args.direction, "both")
        self.assertTrue(entity_args.json)

    def test_exchange_command_arguments(self) -> None:
        parser = wm.build_parser()
        export_args = parser.parse_args(["export", "--output", "/tmp/memory.jsonl", "--json"])
        self.assertEqual(export_args.command, "export")
        self.assertEqual(export_args.output, "/tmp/memory.jsonl")
        self.assertTrue(export_args.json)

        import_args = parser.parse_args(["import", "/tmp/memory.jsonl", "--replace", "--json"])
        self.assertEqual(import_args.command, "import")
        self.assertEqual(import_args.path, "/tmp/memory.jsonl")
        self.assertTrue(import_args.replace)
        self.assertFalse(import_args.no_manifest)
        self.assertTrue(import_args.json)

        legacy_import_args = parser.parse_args(["import", "/tmp/memory.jsonl", "--no-manifest"])
        self.assertEqual(legacy_import_args.command, "import")
        self.assertTrue(legacy_import_args.no_manifest)

    def test_ingest_and_eval_command_arguments(self) -> None:
        parser = wm.build_parser()
        ingest_args = parser.parse_args(
            ["ingest", "docs/c4", "--scope", "memory", "--room", "docs", "--include-code", "--json"]
        )
        self.assertEqual(ingest_args.command, "ingest")
        self.assertEqual(ingest_args.paths, ["docs/c4"])
        self.assertEqual(ingest_args.scope, "memory")
        self.assertEqual(ingest_args.room, "docs")
        self.assertTrue(ingest_args.include_code)
        self.assertTrue(ingest_args.json)

        eval_args = parser.parse_args(["eval", "/tmp/recall.jsonl", "--json"])
        self.assertEqual(eval_args.command, "eval")
        self.assertEqual(eval_args.cases_path, "/tmp/recall.jsonl")
        self.assertTrue(eval_args.json)


if __name__ == "__main__":
    unittest.main()
