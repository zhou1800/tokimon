"""Deterministic helpers for the memory maintenance workflow."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from tokimon.memory.mem_palace import mem_palace as mem
from tokimon.memory.mem_palace.exchange import backup_database
from tokimon.runtime import resolve_runtime_path, runtime_root as repository_runtime_root, source_root
from tokimon.workflow.memory_maintenance import contract


WORKSPACE_ROOT = source_root()
RUNTIME_ROOT = repository_runtime_root(WORKSPACE_ROOT)
DEFAULT_DB_PATH = mem.DEFAULT_DB_PATH
DEFAULT_RUN_ROOT = RUNTIME_ROOT / "workflow" / contract.WORKFLOW_ID
DEFAULT_RUN_LOG_ROOT = DEFAULT_RUN_ROOT / "runs"
DEFAULT_BACKUP_ROOT = DEFAULT_RUN_ROOT / "backups"


EXPECTED_MEMORY_COLUMNS = {
    "id",
    "kind",
    "summary",
    "details",
    "tags",
    "source",
    "scope",
    "room",
    "entities",
    "content_hash",
    "importance",
    "confidence",
    "metadata",
    "pinned",
    "created_at",
    "updated_at",
    "use_count",
    "last_used_at",
}


def run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return cleaned or run_timestamp()


def make_run_id(action: str, run_id: str | None = None) -> str:
    if run_id:
        return sanitize_run_id(run_id)
    return f"{run_timestamp()}-{action}"


def resolve_db_path(db_path: Path | str = DEFAULT_DB_PATH) -> Path:
    return Path(db_path).expanduser().resolve()


def _require_existing_db_path(db_path: Path | str = DEFAULT_DB_PATH) -> Path:
    resolved = resolve_db_path(db_path)
    if not resolved.is_file():
        raise ValueError(f"Mem Palace database does not exist: {resolved}")
    return resolved


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    resolved = _require_existing_db_path(db_path)
    conn = sqlite3.connect(str(resolved), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def connect_readonly(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    resolved = _require_existing_db_path(db_path)
    uri = f"{resolved.as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _filters(scope: str | None = None, room: str | None = None) -> tuple[str, list[str]]:
    clauses: list[str] = []
    params: list[str] = []
    if scope:
        clauses.append("scope = ?")
        params.append(scope.strip().lower())
    if room:
        clauses.append("room = ?")
        params.append(room.strip().lower())
    if not clauses:
        return "", params
    return "WHERE " + " AND ".join(clauses), params


def _snapshot_record(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    record = mem.row_to_dict(row)
    record["summary_chars"] = len(record.get("summary") or "")
    record["details_chars"] = len(record.get("details") or "")
    return record


def _memory_rows(conn: sqlite3.Connection, *, scope: str | None = None, room: str | None = None) -> list[dict[str, Any]]:
    where_sql, params = _filters(scope=scope, room=room)
    rows = conn.execute(
        f"""
        SELECT *
        FROM memories
        {where_sql}
        ORDER BY scope, room, pinned DESC, importance DESC, confidence DESC, created_at DESC, id
        """,
        params,
    ).fetchall()
    return [_snapshot_record(row) for row in rows]


def _count_records(records: Iterable[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    counter = Counter(str(record.get(key) or "") for record in records)
    return [{key: value, "record_count": count} for value, count in sorted(counter.items())]


def _room_counts(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("scope") or ""), str(record.get("room") or ""))
        item = grouped.setdefault(
            key,
            {
                "scope": key[0],
                "room": key[1],
                "record_count": 0,
                "pinned_count": 0,
                "unpinned_count": 0,
                "total_summary_chars": 0,
                "total_details_chars": 0,
            },
        )
        item["record_count"] += 1
        if record.get("pinned"):
            item["pinned_count"] += 1
        else:
            item["unpinned_count"] += 1
        item["total_summary_chars"] += int(record.get("summary_chars") or 0)
        item["total_details_chars"] += int(record.get("details_chars") or 0)
    return sorted(grouped.values(), key=lambda item: (-item["record_count"], item["scope"], item["room"]))


def _exact_duplicate_groups(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for record in records:
        content_hash = str(record.get("content_hash") or "")
        if content_hash:
            groups[content_hash].append(int(record["id"]))
    return [
        {
            "content_hash": content_hash,
            "record_ids": sorted(record_ids),
            "record_count": len(record_ids),
        }
        for content_hash, record_ids in sorted(groups.items())
        if len(record_ids) > 1
    ]


def _snapshot_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    pinned_count = sum(1 for record in records if record.get("pinned"))
    return {
        "total_count": len(records),
        "pinned_count": pinned_count,
        "unpinned_count": len(records) - pinned_count,
        "kind_counts": _count_records(records, "kind"),
        "scope_counts": _count_records(records, "scope"),
        "room_counts": _room_counts(records),
        "total_summary_chars": sum(int(record.get("summary_chars") or 0) for record in records),
        "total_details_chars": sum(int(record.get("details_chars") or 0) for record in records),
    }


def snapshot(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    scope: str | None = None,
    room: str | None = None,
) -> dict[str, Any]:
    resolved_db_path = _require_existing_db_path(db_path)
    conn = connect_readonly(resolved_db_path)
    try:
        records = _memory_rows(conn, scope=scope, room=room)
        return {
            "workflow_id": contract.WORKFLOW_ID,
            "generated_at": utc_now(),
            "database": str(resolved_db_path),
            "filters": {
                "scope": scope,
                "room": room,
            },
            "stats": _snapshot_stats(records),
            "exact_duplicate_groups": _exact_duplicate_groups(records),
            "records": records,
        }
    finally:
        conn.close()


def resolve_run_root(run_root: Path | str = DEFAULT_RUN_ROOT) -> Path:
    path = Path(run_root).expanduser()
    if not path.is_absolute():
        path = resolve_runtime_path(path, workspace_root=WORKSPACE_ROOT)
    return path.resolve()


def _write_run_log(action: str, payload: Mapping[str, Any], *, run_id: str, run_log_root: Path) -> Path:
    run_log_root.mkdir(parents=True, exist_ok=True)
    path = run_log_root / f"{run_id}.jsonl"
    row = {
        "type": "memory_maintenance_run",
        "workflow_id": contract.WORKFLOW_ID,
        "action": action,
        "recorded_at": utc_now(),
        "payload": dict(payload),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _backup(conn: sqlite3.Connection, *, action: str, run_id: str, backup_root: Path) -> dict[str, Any]:
    backup_root.mkdir(parents=True, exist_ok=True)
    return backup_database(conn, backup_root / f"{run_id}-{action}.db")


def _pre_mutation_backup(
    db_path: Path | str,
    *,
    action: str,
    run_id: str,
    backup_root: Path,
) -> dict[str, Any]:
    conn = connect_readonly(db_path)
    try:
        return _backup(conn, action=action, run_id=run_id, backup_root=backup_root)
    finally:
        conn.close()


def _rebuild_fts(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
        conn.commit()
        return True
    except sqlite3.OperationalError:
        return False


def _load_memory(conn: sqlite3.Connection, memory_id: int) -> dict[str, Any]:
    memory = mem.get_memory(conn, memory_id)
    if memory is None:
        raise ValueError(f"Memory #{memory_id} not found")
    return memory


def _require_unpinned(memory: Mapping[str, Any]) -> None:
    if memory.get("pinned"):
        raise ValueError(f"Memory #{memory['id']} is pinned and cannot be mutated by maintenance commands")


def delete_memory(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    memory_id: int,
    reason: str,
    run_id: str | None = None,
    run_root: Path | str = DEFAULT_RUN_ROOT,
) -> dict[str, Any]:
    if not reason.strip():
        raise ValueError("reason is required")
    resolved_run_id = make_run_id("delete", run_id)
    resolved_run_root = resolve_run_root(run_root)
    conn = connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        before = _load_memory(conn, memory_id)
        _require_unpinned(before)
        reference_count = conn.execute(
            "SELECT COUNT(*) FROM relationships WHERE source_memory_id = ?",
            [int(memory_id)],
        ).fetchone()[0]
        if reference_count:
            raise ValueError(
                f"Memory #{memory_id} is referenced by {reference_count} relationship(s) "
                "and cannot be deleted while those provenance references remain"
            )
        backup = _pre_mutation_backup(
            db_path,
            action="delete",
            run_id=resolved_run_id,
            backup_root=resolved_run_root / "backups",
        )
        cursor = conn.execute("DELETE FROM memories WHERE id = ?", [int(memory_id)])
        conn.commit()
        fts_rebuilt = _rebuild_fts(conn)
        payload = {
            "action": "delete",
            "memory_id": int(memory_id),
            "reason": reason.strip(),
            "deleted": cursor.rowcount == 1,
            "before": before,
            "backup": backup,
            "fts_rebuilt": fts_rebuilt,
        }
        run_log_path = _write_run_log("delete", payload, run_id=resolved_run_id, run_log_root=resolved_run_root / "runs")
        payload["run_log_path"] = str(run_log_path)
        return payload
    finally:
        conn.close()


def update_weight(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    memory_id: int,
    importance: float,
    confidence: float,
    reason: str,
    run_id: str | None = None,
    run_root: Path | str = DEFAULT_RUN_ROOT,
) -> dict[str, Any]:
    if not reason.strip():
        raise ValueError("reason is required")
    if not 0.0 <= float(importance) <= 5.0:
        raise ValueError("importance must be between 0 and 5")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    importance_value = mem.clamp_float(importance, minimum=0.0, maximum=5.0, default=mem.DEFAULT_IMPORTANCE)
    confidence_value = mem.clamp_float(confidence, minimum=0.0, maximum=1.0, default=mem.DEFAULT_CONFIDENCE)
    resolved_run_id = make_run_id("update-weight", run_id)
    resolved_run_root = resolve_run_root(run_root)
    conn = connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        before = _load_memory(conn, memory_id)
        _require_unpinned(before)
        backup = _pre_mutation_backup(
            db_path,
            action="update-weight",
            run_id=resolved_run_id,
            backup_root=resolved_run_root / "backups",
        )
        conn.execute(
            """
            UPDATE memories
            SET importance = ?,
                confidence = ?,
                updated_at = ?
            WHERE id = ?
            """,
            [importance_value, confidence_value, utc_now(), int(memory_id)],
        )
        conn.commit()
        fts_rebuilt = _rebuild_fts(conn)
        after = _load_memory(conn, memory_id)
        payload = {
            "action": "update-weight",
            "memory_id": int(memory_id),
            "reason": reason.strip(),
            "before": before,
            "after": after,
            "backup": backup,
            "fts_rebuilt": fts_rebuilt,
        }
        run_log_path = _write_run_log(
            "update-weight",
            payload,
            run_id=resolved_run_id,
            run_log_root=resolved_run_root / "runs",
        )
        payload["run_log_path"] = str(run_log_path)
        return payload
    finally:
        conn.close()


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _fts_count(conn: sqlite3.Connection) -> int | None:
    try:
        return int(conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0])
    except sqlite3.OperationalError:
        return None


def repair_status(conn: sqlite3.Connection) -> dict[str, Any]:
    columns = _table_columns(conn, "memories")
    memory_count = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
    fts_count = _fts_count(conn)
    empty_hash_count = int(
        conn.execute("SELECT COUNT(*) FROM memories WHERE content_hash IS NULL OR content_hash = ''").fetchone()[0]
    )
    duplicate_group_count = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT content_hash
                FROM memories
                WHERE content_hash IS NOT NULL AND content_hash != ''
                GROUP BY content_hash
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )
    return {
        "schema_version": mem.SCHEMA_VERSION,
        "memory_count": memory_count,
        "memory_columns_missing": sorted(EXPECTED_MEMORY_COLUMNS - columns),
        "empty_content_hash_count": empty_hash_count,
        "exact_duplicate_group_count": duplicate_group_count,
        "fts_enabled": mem.has_fts(conn),
        "fts_count": fts_count,
        "fts_matches_memory_count": fts_count == memory_count if fts_count is not None else False,
    }


def _backfill_hashes(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT id, kind, summary, details, tags, source, scope, room
        FROM memories
        WHERE (content_hash IS NULL OR content_hash = '')
          AND COALESCE(pinned, 0) = 0
        """
    ).fetchall()
    backfilled = 0
    for row in rows:
        content_hash = mem.memory_content_hash(
            kind=row["kind"],
            summary=row["summary"],
            details=row["details"],
            tags=mem.normalize_tags([row["tags"]]),
            source=row["source"],
            scope=row["scope"],
            room=row["room"],
        )
        cursor = conn.execute(
            """
            UPDATE memories SET content_hash = ?
            WHERE id = ? AND COALESCE(pinned, 0) = 0
              AND (content_hash IS NULL OR content_hash = '')
            """,
            [content_hash, row["id"]],
        )
        backfilled += cursor.rowcount
    conn.commit()
    return backfilled


def repair(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    run_id: str | None = None,
    run_root: Path | str = DEFAULT_RUN_ROOT,
) -> dict[str, Any]:
    resolved_run_id = make_run_id("repair", run_id)
    resolved_run_root = resolve_run_root(run_root)
    backup = _pre_mutation_backup(
        db_path,
        action="repair",
        run_id=resolved_run_id,
        backup_root=resolved_run_root / "backups",
    )
    conn = connect(db_path)
    try:
        before = repair_status(conn)
        hashes_backfilled = _backfill_hashes(conn)
        conn.execute("UPDATE memories SET use_count = 0 WHERE use_count IS NULL AND COALESCE(pinned, 0) = 0")
        conn.commit()
        fts_rebuilt = _rebuild_fts(conn)
        after = repair_status(conn)
        payload = {
            "action": "repair",
            "before": before,
            "after": after,
            "hashes_backfilled": hashes_backfilled,
            "backup": backup,
            "fts_rebuilt": fts_rebuilt,
        }
        run_log_path = _write_run_log("repair", payload, run_id=resolved_run_id, run_log_root=resolved_run_root / "runs")
        payload["run_log_path"] = str(run_log_path)
        return payload
    finally:
        conn.close()
