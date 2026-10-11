from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.memory.mem_palace import mem_palace as wm
from tokimon.workflow.memory_maintenance import operations


class MemoryMaintenanceOperationTests(unittest.TestCase):
    def _seed_db(self, db_path: Path) -> tuple[int, int, int]:
        conn = wm.connect(db_path)
        try:
            first_id = wm.add_memory(
                conn,
                kind="preference",
                summary="Use slow browser automation actions",
                details="One slow live action is safer for browser-backed workflow writes.",
                tags=["browser", "automation"],
                source="test",
                scope="workflow",
                room="browser-automation",
            )
            duplicate_id = wm.add_memory(
                conn,
                kind="preference",
                summary="Use slow browser automation actions",
                details="One slow live action is safer for browser-backed workflow writes.",
                tags=["browser", "automation"],
                source="test",
                scope="workflow",
                room="browser-automation",
                allow_duplicate=True,
            )
            pinned_id = wm.add_memory(
                conn,
                kind="avoid",
                summary="Avoid deleting pinned memory",
                details="Pinned memory is protected during maintenance.",
                tags=["memory", "maintenance"],
                source="test",
                scope="memory",
                room="maintenance",
                pinned=True,
            )
        finally:
            conn.close()
        return first_id, duplicate_id, pinned_id

    def test_snapshot_returns_raw_records_counts_and_exact_duplicate_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            first_id, duplicate_id, _ = self._seed_db(db_path)

            report = operations.snapshot(db_path)

        self.assertEqual(report["workflow_id"], "memory-maintenance")
        self.assertEqual(report["stats"]["total_count"], 3)
        self.assertEqual(report["stats"]["pinned_count"], 1)
        self.assertEqual(report["stats"]["unpinned_count"], 2)
        self.assertEqual(len(report["exact_duplicate_groups"]), 1)
        self.assertEqual(report["exact_duplicate_groups"][0]["record_ids"], [first_id, duplicate_id])
        self.assertIn("use_count", report["records"][0])
        self.assertIn("last_used_at", report["records"][0])
        self.assertIn("summary_chars", report["records"][0])
        self.assertIn("details_chars", report["records"][0])

    def test_snapshot_missing_database_fails_without_creating_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "missing-memory.db"

            with self.assertRaisesRegex(ValueError, "does not exist"):
                operations.snapshot(db_path)

            self.assertFalse(db_path.exists())

    def test_delete_refuses_pinned_and_logs_unpinned_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "memory.db"
            state_root = tmp / "state"
            first_id, _, pinned_id = self._seed_db(db_path)

            with self.assertRaisesRegex(ValueError, "pinned"):
                operations.delete_memory(
                    db_path,
                    memory_id=pinned_id,
                    reason="Pinned should not be deleted",
                    run_root=state_root,
                )

            report = operations.delete_memory(
                db_path,
                memory_id=first_id,
                reason="Duplicate after full-record review",
                run_root=state_root,
                run_id="delete-test",
            )
            conn = wm.connect(db_path)
            try:
                deleted = wm.get_memory(conn, first_id)
            finally:
                conn.close()

            run_log = Path(report["run_log_path"])
            backup = Path(report["backup"]["path"])

            self.assertIsNone(deleted)
            self.assertTrue(report["deleted"])
            self.assertTrue(run_log.exists())
            self.assertTrue(backup.exists())
            self.assertEqual(json.loads(run_log.read_text(encoding="utf-8").splitlines()[0])["action"], "delete")

    def test_delete_backup_includes_committed_wal_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            state_root = Path(tmpdir) / "state"
            first_id, _, _ = self._seed_db(db_path)
            writer = wm.connect(db_path)
            try:
                retained_id = wm.add_memory(writer, kind="fact", summary="Retained committed memory")
                before = [dict(row) for row in writer.execute("SELECT * FROM memories ORDER BY id")]
                report = operations.delete_memory(
                    db_path, memory_id=first_id, reason="Duplicate record", run_root=state_root,
                )
                with sqlite3.connect(report["backup"]["path"]) as backup:
                    backup.row_factory = sqlite3.Row
                    backed_up = [dict(row) for row in backup.execute("SELECT * FROM memories ORDER BY id")]
                self.assertEqual(backed_up, before)
                self.assertTrue(report["deleted"])
                self.assertIsNone(wm.get_memory(writer, first_id))
                self.assertIsNotNone(wm.get_memory(writer, retained_id))
            finally:
                writer.close()

    def test_mutations_refuse_a_pin_committed_before_writer_lock(self) -> None:
        for action in ("delete", "update-weight"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "memory.db"
                state_root = Path(tmpdir) / "state"
                memory_id, _, _ = self._seed_db(db_path)
                real_connect = operations.connect

                def pin_before_lock(path):
                    conn = real_connect(path)
                    with sqlite3.connect(db_path) as writer:
                        writer.execute("UPDATE memories SET pinned = 1 WHERE id = ?", [memory_id])
                    return conn

                with patch.object(operations, "connect", side_effect=pin_before_lock):
                    with self.assertRaisesRegex(ValueError, "pinned"):
                        if action == "delete":
                            operations.delete_memory(db_path, memory_id=memory_id, reason="Concurrent pin", run_root=state_root)
                        else:
                            operations.update_weight(db_path, memory_id=memory_id, importance=1, confidence=0.5, reason="Concurrent pin", run_root=state_root)
                with sqlite3.connect(db_path) as check:
                    row = check.execute("SELECT pinned, importance, confidence FROM memories WHERE id = ?", [memory_id]).fetchone()
                self.assertEqual(row, (1, 3.0, 1.0))
                self.assertFalse(state_root.exists())

    def test_mutations_hold_writer_lock_through_consistent_backup(self) -> None:
        for action in ("delete", "update-weight"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "memory.db"
                memory_id, _, _ = self._seed_db(db_path)
                writer = wm.connect(db_path)
                try:
                    wm.add_memory(writer, kind="fact", summary="Committed WAL record")
                    before = [dict(row) for row in writer.execute("SELECT * FROM memories ORDER BY id")]
                    memory_before = wm.get_memory(writer, memory_id)
                    real_backup = operations._pre_mutation_backup

                    def backup_while_competing_write(path, **kwargs):
                        with sqlite3.connect(db_path, timeout=0) as competing:
                            with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                                competing.execute("UPDATE memories SET pinned = 1 WHERE id = ?", [memory_id])
                        return real_backup(path, **kwargs)

                    with patch.object(operations, "_pre_mutation_backup", side_effect=backup_while_competing_write) as backup_call:
                        kwargs = dict(memory_id=memory_id, reason="Locked mutation", run_root=Path(tmpdir) / "state")
                        if action == "delete":
                            report = operations.delete_memory(db_path, **kwargs)
                        else:
                            report = operations.update_weight(db_path, importance=1, confidence=0.5, **kwargs)
                    self.assertEqual(backup_call.call_count, 1)
                    self.assertEqual(report["before"], memory_before)
                    with sqlite3.connect(report["backup"]["path"]) as backup:
                        backup.row_factory = sqlite3.Row
                        self.assertEqual([dict(row) for row in backup.execute("SELECT * FROM memories ORDER BY id")], before)
                    if action == "delete":
                        self.assertIsNone(wm.get_memory(writer, memory_id))
                    else:
                        row = wm.get_memory(writer, memory_id)
                        self.assertEqual((row["pinned"], row["importance"], row["confidence"]), (0, 1.0, 0.5))
                finally:
                    writer.close()

    def test_delete_refuses_provenance_committed_before_writer_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            state_root = Path(tmpdir) / "state"
            memory_id, _, _ = self._seed_db(db_path)
            real_connect = operations.connect

            def reference_before_lock(path):
                conn = real_connect(path)
                with wm.MemPalace(db_path) as palace:
                    palace.add_fact("Subject", "uses", "Object", source_memory_id=memory_id)
                return conn

            with patch.object(operations, "connect", side_effect=reference_before_lock):
                with self.assertRaisesRegex(ValueError, "referenced by 1 relationship"):
                    operations.delete_memory(db_path, memory_id=memory_id, reason="Referenced", run_root=state_root)
            self.assertFalse(state_root.exists())
            with wm.MemPalace(db_path) as palace:
                self.assertIsNotNone(palace.get(memory_id))
                self.assertEqual(palace.stats()["relationships"], 1)

    def test_update_weight_refuses_pinned_and_writes_before_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "memory.db"
            state_root = tmp / "state"
            first_id, _, pinned_id = self._seed_db(db_path)

            with self.assertRaisesRegex(ValueError, "pinned"):
                operations.update_weight(
                    db_path,
                    memory_id=pinned_id,
                    importance=1,
                    confidence=0.5,
                    reason="Pinned should not be decayed",
                    run_root=state_root,
                )

            report = operations.update_weight(
                db_path,
                memory_id=first_id,
                importance=1,
                confidence=0.5,
                reason="Still maybe useful but lower signal",
                run_root=state_root,
                run_id="weight-test",
            )

            self.assertEqual(report["before"]["importance"], 3.0)
            self.assertEqual(report["after"]["importance"], 1.0)
            self.assertEqual(report["after"]["confidence"], 0.5)
            self.assertTrue(Path(report["backup"]["path"]).exists())
            self.assertTrue(Path(report["run_log_path"]).exists())

    def test_repair_backfills_empty_hashes_and_reports_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "memory.db"
            state_root = tmp / "state"
            first_id, _, _ = self._seed_db(db_path)

            raw = sqlite3.connect(db_path)
            try:
                raw.execute("UPDATE memories SET content_hash = '' WHERE id = ?", [first_id])
                raw.commit()
            finally:
                raw.close()

            report = operations.repair(db_path, run_root=state_root, run_id="repair-test")

            self.assertEqual(report["after"]["empty_content_hash_count"], 0)
            self.assertTrue(report["after"]["fts_matches_memory_count"])
            self.assertTrue(Path(report["backup"]["path"]).exists())
            self.assertTrue(Path(report["run_log_path"]).exists())

            backup = sqlite3.connect(report["backup"]["path"])
            try:
                backed_up_hash = backup.execute(
                    "SELECT content_hash FROM memories WHERE id = ?",
                    [first_id],
                ).fetchone()[0]
            finally:
                backup.close()
            self.assertEqual(backed_up_hash, "")

    def test_repair_does_not_mutate_pinned_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "memory.db"
            state_root = tmp / "state"
            _, _, pinned_id = self._seed_db(db_path)

            raw = sqlite3.connect(db_path)
            try:
                raw.execute("UPDATE memories SET content_hash = '' WHERE id = ?", [pinned_id])
                raw.commit()
            finally:
                raw.close()

            report = operations.repair(db_path, run_root=state_root, run_id="pinned-repair-test")

            raw = sqlite3.connect(db_path)
            try:
                row = raw.execute(
                    "SELECT content_hash FROM memories WHERE id = ?",
                    [pinned_id],
                ).fetchone()
            finally:
                raw.close()
            self.assertEqual(row[0], "")
            self.assertEqual(report["hashes_backfilled"], 0)
            self.assertEqual(report["after"]["empty_content_hash_count"], 1)

    def test_repair_preserves_a_pin_committed_after_backfill_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            memory_id, _, _ = self._seed_db(db_path)
            with sqlite3.connect(db_path) as writer:
                writer.execute("UPDATE memories SET content_hash = '' WHERE id = ?", [memory_id])
            real_hash = wm.memory_content_hash
            pinned_snapshot = None

            def pin_before_hash(**kwargs):
                nonlocal pinned_snapshot
                with sqlite3.connect(db_path) as writer:
                    writer.row_factory = sqlite3.Row
                    writer.execute("UPDATE memories SET pinned = 1 WHERE id = ?", [memory_id])
                    pinned_snapshot = dict(writer.execute("SELECT * FROM memories WHERE id = ?", [memory_id]).fetchone())
                return real_hash(**kwargs)

            with patch.object(wm, "memory_content_hash", side_effect=pin_before_hash):
                report = operations.repair(db_path, run_root=Path(tmpdir) / "state")

            with sqlite3.connect(db_path) as check:
                check.row_factory = sqlite3.Row
                self.assertEqual(dict(check.execute("SELECT * FROM memories WHERE id = ?", [memory_id]).fetchone()), pinned_snapshot)
            self.assertIsNotNone(pinned_snapshot)
            self.assertEqual(report["hashes_backfilled"], 0)
            self.assertEqual(report["after"]["empty_content_hash_count"], 1)

    def test_repair_preserves_a_hash_committed_after_backfill_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            memory_id, _, _ = self._seed_db(db_path)
            with sqlite3.connect(db_path) as writer:
                writer.execute("UPDATE memories SET content_hash = '' WHERE id = ?", [memory_id])
            real_hash = wm.memory_content_hash

            def update_hash_before_backfill(**kwargs):
                with sqlite3.connect(db_path) as writer:
                    writer.execute("UPDATE memories SET content_hash = 'concurrent-value' WHERE id = ?", [memory_id])
                return real_hash(**kwargs)

            with patch.object(wm, "memory_content_hash", side_effect=update_hash_before_backfill):
                report = operations.repair(db_path, run_root=Path(tmpdir) / "state")

            with sqlite3.connect(db_path) as check:
                self.assertEqual(check.execute("SELECT content_hash FROM memories WHERE id = ?", [memory_id]).fetchone()[0], "concurrent-value")
            self.assertEqual(report["hashes_backfilled"], 0)
            self.assertEqual(report["after"]["empty_content_hash_count"], 0)


if __name__ == "__main__":
    unittest.main()
