from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.memory.mem_palace import mem_palace as wm
from tokimon.workflow.memory_maintenance import cli as memory_maintenance_cli


def run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = memory_maintenance_cli.main(argv)
    return status, stdout.getvalue(), stderr.getvalue()


def run_main_json(argv: list[str]) -> tuple[int, dict[str, object]]:
    status, stdout, _ = run_main(argv)
    return status, json.loads(stdout)


def mutation_payload(action: str, *, memory_id: int | None = None, reason: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "action": action,
        "backup": {"path": "/tmp/backup.db"},
        "run_log_path": "/tmp/run.jsonl",
    }
    if memory_id is not None:
        payload["memory_id"] = memory_id
    if reason is not None:
        payload["reason"] = reason
    return payload


def subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("parser has no subcommands")


class MemoryMaintenanceCliContractTests(unittest.TestCase):
    def test_memory_maintenance_cli_contract(self) -> None:
        parser = memory_maintenance_cli.build_parser()

        self.assertEqual(subcommand_names(parser), {"snapshot", "delete", "update-weight", "repair"})

        snapshot_args = parser.parse_args(["--db", "/tmp/memory.db", "snapshot", "--scope", "workflow", "--json"])
        self.assertEqual(snapshot_args.command, "snapshot")
        self.assertEqual(snapshot_args.db, "/tmp/memory.db")
        self.assertEqual(snapshot_args.scope, "workflow")
        self.assertTrue(snapshot_args.json)

        delete_args = parser.parse_args(
            [
                "--state-dir",
                "/tmp/state",
                "delete",
                "--id",
                "12",
                "--reason",
                "duplicate",
                "--run-id",
                "run-1",
                "--json",
            ]
        )
        self.assertEqual(delete_args.command, "delete")
        self.assertEqual(delete_args.state_dir, "/tmp/state")
        self.assertEqual(delete_args.memory_id, 12)
        self.assertEqual(delete_args.reason, "duplicate")
        self.assertEqual(delete_args.run_id, "run-1")
        self.assertTrue(delete_args.json)

        weight_args = parser.parse_args(
            [
                "update-weight",
                "--id",
                "13",
                "--importance",
                "1.5",
                "--confidence",
                "0.6",
                "--reason",
                "stale",
            ]
        )
        self.assertEqual(weight_args.command, "update-weight")
        self.assertEqual(weight_args.memory_id, 13)
        self.assertEqual(weight_args.importance, 1.5)
        self.assertEqual(weight_args.confidence, 0.6)
        self.assertEqual(weight_args.reason, "stale")

        repair_args = parser.parse_args(["repair", "--json"])
        self.assertEqual(repair_args.command, "repair")
        self.assertTrue(repair_args.json)

        self.assertEqual(memory_maintenance_cli.required_tool_capabilities(), ())

    def test_main_snapshot_json_dispatches_public_cli(self) -> None:
        payload = {
            "workflow_id": "memory-maintenance",
            "stats": {"total_count": 0, "pinned_count": 0},
            "exact_duplicate_groups": [],
            "database": "/tmp/memory.db",
            "records": [],
        }

        with mock.patch.object(memory_maintenance_cli.operations, "snapshot", return_value=payload) as snapshot:
            status, emitted = run_main_json(["--db", "/tmp/memory.db", "snapshot", "--scope", "workflow", "--json"])

        self.assertEqual(status, 0)
        self.assertEqual(emitted["workflow_id"], "memory-maintenance")
        snapshot.assert_called_once_with(Path("/tmp/memory.db"), scope="workflow", room=None)

    def test_main_mutation_commands_emit_json(self) -> None:
        with mock.patch.object(
            memory_maintenance_cli.operations,
            "delete_memory",
            return_value=mutation_payload("delete", memory_id=12, reason="duplicate"),
        ) as delete_memory:
            status, emitted = run_main_json(
                [
                    "--db",
                    "/tmp/memory.db",
                    "--state-dir",
                    "/tmp/state",
                    "delete",
                    "--id",
                    "12",
                    "--reason",
                    "duplicate",
                    "--run-id",
                    "run-1",
                    "--json",
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(emitted["action"], "delete")
        delete_memory.assert_called_once_with(
            Path("/tmp/memory.db"),
            memory_id=12,
            reason="duplicate",
            run_id="run-1",
            run_root="/tmp/state",
        )

        with mock.patch.object(
            memory_maintenance_cli.operations,
            "update_weight",
            return_value=mutation_payload("update-weight", memory_id=13, reason="stale"),
        ) as update_weight:
            status, emitted = run_main_json(
                [
                    "--db",
                    "/tmp/memory.db",
                    "--state-dir",
                    "/tmp/state",
                    "update-weight",
                    "--id",
                    "13",
                    "--importance",
                    "1.5",
                    "--confidence",
                    "0.6",
                    "--reason",
                    "stale",
                    "--run-id",
                    "run-2",
                    "--json",
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(emitted["action"], "update-weight")
        update_weight.assert_called_once_with(
            Path("/tmp/memory.db"),
            memory_id=13,
            importance=1.5,
            confidence=0.6,
            reason="stale",
            run_id="run-2",
            run_root="/tmp/state",
        )

        with mock.patch.object(
            memory_maintenance_cli.operations,
            "repair",
            return_value=mutation_payload("repair"),
        ) as repair:
            status, emitted = run_main_json(
                ["--db", "/tmp/memory.db", "--state-dir", "/tmp/state", "repair", "--run-id", "run-3", "--json"]
            )
        self.assertEqual(status, 0)
        self.assertEqual(emitted["action"], "repair")
        repair.assert_called_once_with(Path("/tmp/memory.db"), run_id="run-3", run_root="/tmp/state")

    def test_main_no_command_and_value_error_exit_codes(self) -> None:
        status, stdout, stderr = run_main([])
        self.assertEqual(status, 1)
        self.assertIn("usage:", stdout)
        self.assertEqual(stderr, "")

        with mock.patch.object(
            memory_maintenance_cli.operations,
            "delete_memory",
            side_effect=ValueError("Memory #99 not found"),
        ):
            status, stdout, stderr = run_main(["delete", "--id", "99", "--reason", "missing"])

        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("Memory #99 not found", stderr)

    def test_delete_reports_relationship_references_before_creating_artifacts(self) -> None:
        for valid_to in (None, "2025-12-31"):
            with self.subTest(valid_to=valid_to), tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "memory.db"
                state_root = Path(tmpdir) / "state"
                conn = wm.connect(db_path)
                try:
                    memory_id = wm.add_memory(conn, kind="fact", summary="Relationship provenance")
                    wm.add_relationship(
                        conn, "Audit Run", "uses_evidence", "Memory Record",
                        source_memory_id=memory_id, valid_from="2025-01-01", valid_to=valid_to,
                    )
                    before = wm.get_memory(conn, memory_id)
                finally:
                    conn.close()
                status, stdout, stderr = run_main([
                    "--db", str(db_path), "--state-dir", str(state_root),
                    "delete", "--id", str(memory_id), "--reason", "Duplicate candidate", "--json",
                ])
                self.assertEqual(status, 2)
                self.assertEqual(stdout, "")
                self.assertIn(f"Memory #{memory_id} is referenced by 1 relationship", stderr)
                self.assertNotIn("Traceback", stderr)
                self.assertFalse(state_root.exists())
                conn = wm.connect(db_path)
                try:
                    self.assertEqual(wm.get_memory(conn, memory_id), before)
                    self.assertEqual(conn.execute(
                        "SELECT COUNT(*) FROM relationships WHERE source_memory_id = ?", [memory_id],
                    ).fetchone()[0], 1)
                finally:
                    conn.close()

    def test_module_entrypoint_snapshot_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            conn = wm.connect(db_path)
            conn.close()
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                path for path in (str(ROOT), env.get("PYTHONPATH", "")) if path
            )

            completed = subprocess.run(
                [sys.executable, "-m", "tokimon.workflow.memory_maintenance", "--db", str(db_path), "snapshot", "--json"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        emitted = json.loads(completed.stdout)
        self.assertEqual(emitted["workflow_id"], "memory-maintenance")
        self.assertEqual(emitted["stats"]["total_count"], 0)


if __name__ == "__main__":
    unittest.main()
