"""Backup and JSONL exchange helpers for Mem Palace stores."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tokimon.memory.mem_palace.identity import (
    canonical_metadata,
    canonicalize_import_memory,
    remap_supersession_metadata,
    resolve_memory_duplicate,
)
from tokimon.memory.mem_palace.secret_guard import reject_secret_material
from tokimon.memory.mem_palace.sqlite_support import is_missing_fts_error


EXPORT_FORMAT = "mem-palace-jsonl"
EXPORT_VERSION = 1
EXPORT_MANIFEST_FORMAT = "mem-palace-export-manifest"
EXPORT_MANIFEST_SCHEMA_VERSION = 1
EXPORT_TABLES = ("memories", "entities", "relationships")
INSTANCE_ID_KEY = "instance_id"

ALLOWED_RECORD_FIELDS = {
    "memories": {
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
    },
    "entities": {
        "id",
        "name",
        "kind",
        "aliases",
        "metadata",
        "created_at",
        "updated_at",
    },
    "relationships": {
        "id",
        "subject_id",
        "predicate",
        "object_id",
        "valid_from",
        "valid_to",
        "confidence",
        "source_memory_id",
        "source",
        "created_at",
        "updated_at",
    },
}
REQUIRED_RECORD_FIELDS = {
    "memories": {"id", "kind", "summary", "created_at", "updated_at"},
    "entities": {"id", "name", "created_at", "updated_at"},
    "relationships": {"id", "subject_id", "predicate", "object_id", "created_at", "updated_at"},
}


@dataclass(frozen=True)
class ExportRecord:
    """One validated JSONL stream row."""

    line_number: int
    stream: str
    record_offset: int
    data: dict[str, Any]


class MemPalaceExchangeError(ValueError):
    """User-facing exchange error with import stage context."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        stream: str | None = None,
        record_offset: int | None = None,
        line_number: int | None = None,
    ) -> None:
        self.stage = stage
        self.stream = stream
        self.record_offset = record_offset
        self.line_number = line_number
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        parts = [f"stage={self.stage}"]
        if self.stream is not None:
            parts.append(f"stream={self.stream}")
        if self.record_offset is not None:
            parts.append(f"record_offset={self.record_offset}")
        if self.line_number is not None:
            parts.append(f"line={self.line_number}")
        return f"{' '.join(parts)}: {self.message}"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_json(data: Mapping[str, Any]) -> str:
    return json.dumps(dict(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def export_manifest_path(jsonl_path: Path | str) -> Path:
    """Return the fixed sidecar manifest path for a JSONL exchange file."""

    return Path(jsonl_path).expanduser().resolve().with_name("manifest.json")


def ensure_instance_id(conn: sqlite3.Connection, *, commit: bool = True) -> str:
    """Return a stable local store id, creating one when absent."""

    row = conn.execute("SELECT value FROM schema_meta WHERE key = ? LIMIT 1", [INSTANCE_ID_KEY]).fetchone()
    if row is not None:
        return str(row["value"] if isinstance(row, sqlite3.Row) else row[0])

    started_in_transaction = conn.in_transaction
    instance_id = uuid.uuid4().hex
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
        [INSTANCE_ID_KEY, instance_id],
    )
    if commit and not started_in_transaction:
        conn.commit()
    return instance_id


def backup_database(conn: sqlite3.Connection, output_path: Path | str) -> dict[str, Any]:
    """Write a consistent SQLite backup from an open Mem Palace connection."""

    destination_path = Path(output_path).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    backup_conn = sqlite3.connect(str(destination_path))
    try:
        conn.backup(backup_conn)
    finally:
        backup_conn.close()
    return {
        "path": str(destination_path),
        "bytes": destination_path.stat().st_size,
        "created_at": utc_timestamp(),
    }


def export_jsonl(conn: sqlite3.Connection, output_path: Path | str) -> dict[str, Any]:
    """Export memories, entities, and relationships as portable JSONL rows."""

    destination_path = Path(output_path).expanduser().resolve()
    manifest_path = export_manifest_path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {table: 0 for table in EXPORT_TABLES}
    exported_at = utc_timestamp()
    source_instance_id = ensure_instance_id(conn)

    with destination_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "header",
                    "format": EXPORT_FORMAT,
                    "version": EXPORT_VERSION,
                    "exported_at": exported_at,
                    "tables": list(EXPORT_TABLES),
                },
                sort_keys=True,
            )
            + "\n"
        )
        for table in EXPORT_TABLES:
            for row in _select_rows(conn, table):
                counts[table] += 1
                handle.write(
                    json.dumps(
                        {
                            "type": table[:-1],
                            "table": table,
                            "data": row,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

    jsonl_sha256 = _file_sha256(destination_path)
    manifest = {
        "format": EXPORT_MANIFEST_FORMAT,
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "source_instance_id": source_instance_id,
        "exported_at": exported_at,
        "jsonl_format": EXPORT_FORMAT,
        "jsonl_version": EXPORT_VERSION,
        "streams": list(EXPORT_TABLES),
        "counts": counts,
        "jsonl_sha256": jsonl_sha256,
    }
    manifest_path.write_text(stable_json(manifest), encoding="utf-8")

    return {
        "path": str(destination_path),
        "manifest_path": str(manifest_path),
        "format": EXPORT_FORMAT,
        "version": EXPORT_VERSION,
        "counts": counts,
        "source_instance_id": source_instance_id,
        "jsonl_sha256": jsonl_sha256,
    }


def import_jsonl(
    conn: sqlite3.Connection,
    input_path: Path | str,
    *,
    replace: bool = False,
    skip_duplicate_memories: bool = True,
    require_manifest: bool = True,
) -> dict[str, Any]:
    """Import a Mem Palace JSONL export into the current store."""

    source_path = Path(input_path).expanduser().resolve()
    manifest_path = export_manifest_path(source_path)
    jsonl_bytes = _read_jsonl_bytes(source_path)
    jsonl_sha256 = hashlib.sha256(jsonl_bytes).hexdigest()
    manifest = None
    if require_manifest:
        manifest = _read_manifest(manifest_path)
        _validate_manifest_header(manifest, manifest_path, jsonl_sha256=jsonl_sha256)

    records, export_counts = _load_export_records(source_path, jsonl_bytes)
    if manifest is not None:
        _validate_manifest_counts(manifest, manifest_path, export_counts)

    counts = {table: 0 for table in EXPORT_TABLES}
    skipped = 0
    memory_id_map: dict[int, int] = {}

    savepoint = f"mem_palace_import_{uuid.uuid4().hex}"
    started_in_transaction = conn.in_transaction
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        records_by_stream = {
            table: [record for record in records if record.stream == table]
            for table in EXPORT_TABLES
        }
        records_by_stream["memories"].sort(key=lambda record: int(record.data["id"]))
        for table in EXPORT_TABLES:
            for record in records_by_stream[table]:
                data: Mapping[str, Any] = record.data

                if table == "memories":
                    source_memory_id = int(data["id"])
                    data, superseded_ids = _rewrite_memory_supersession(
                        data,
                        memory_id_map,
                    )
                    if skip_duplicate_memories:
                        duplicate_id = _memory_duplicate_id(conn, data)
                        can_skip_duplicate = (
                            duplicate_id is not None
                            and duplicate_id > max(superseded_ids, default=0)
                            and (not replace or duplicate_id == source_memory_id)
                        )
                        if can_skip_duplicate:
                            assert duplicate_id is not None
                            memory_id_map[source_memory_id] = duplicate_id
                            skipped += 1
                            continue
                    memory_id_map[source_memory_id] = _insert_memory_row(
                        conn,
                        data,
                        replace=replace,
                        minimum_destination_id=max(superseded_ids, default=0),
                    )
                    counts[table] += 1
                    continue

                if table == "relationships":
                    data = _rewrite_relationship_source(conn, data, memory_id_map)

                if _insert_row(conn, table, data, replace=replace):
                    counts[table] += 1
                else:
                    skipped += 1
        _rebuild_fts(conn)
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        if not started_in_transaction:
            conn.commit()
    except MemPalaceExchangeError:
        _rollback_savepoint(conn, savepoint)
        raise
    except (sqlite3.Error, ValueError, TypeError) as exc:
        failed = locals().get("record")
        _rollback_savepoint(conn, savepoint)
        if isinstance(failed, ExportRecord):
            raise MemPalaceExchangeError(
                "apply",
                str(exc),
                stream=failed.stream,
                record_offset=failed.record_offset,
                line_number=failed.line_number,
            ) from exc
        raise MemPalaceExchangeError("apply", str(exc)) from exc

    return {
        "path": str(source_path),
        "manifest_path": str(manifest_path) if require_manifest else None,
        "counts": counts,
        "skipped": skipped,
        "validated_counts": export_counts,
        "jsonl_sha256": jsonl_sha256,
    }


def _read_jsonl_bytes(source_path: Path) -> bytes:
    try:
        return source_path.read_bytes()
    except OSError as exc:
        raise MemPalaceExchangeError("read", f"{source_path}: {exc}") from exc


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MemPalaceExchangeError("manifest", f"manifest not found or unreadable: {manifest_path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MemPalaceExchangeError("manifest", f"{manifest_path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MemPalaceExchangeError("manifest", f"{manifest_path}: manifest must be a JSON object")
    return payload


def _validate_manifest_header(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    jsonl_sha256: str,
) -> None:
    if manifest.get("format") != EXPORT_MANIFEST_FORMAT:
        raise MemPalaceExchangeError("manifest", f"{manifest_path}: unsupported manifest format")
    if manifest.get("schema_version") != EXPORT_MANIFEST_SCHEMA_VERSION:
        raise MemPalaceExchangeError("manifest", f"{manifest_path}: unsupported schema version")
    if manifest.get("jsonl_format") != EXPORT_FORMAT:
        raise MemPalaceExchangeError("manifest", f"{manifest_path}: unsupported JSONL format")
    if manifest.get("jsonl_version") != EXPORT_VERSION:
        raise MemPalaceExchangeError("manifest", f"{manifest_path}: unsupported JSONL version")
    if manifest.get("streams") != list(EXPORT_TABLES):
        raise MemPalaceExchangeError("manifest", f"{manifest_path}: stream order mismatch")
    if not isinstance(manifest.get("source_instance_id"), str) or not manifest.get("source_instance_id"):
        raise MemPalaceExchangeError("manifest", f"{manifest_path}: source_instance_id is required")
    if not isinstance(manifest.get("exported_at"), str) or not manifest.get("exported_at"):
        raise MemPalaceExchangeError("manifest", f"{manifest_path}: exported_at is required")
    if manifest.get("jsonl_sha256") != jsonl_sha256:
        raise MemPalaceExchangeError("manifest", f"{manifest_path}: JSONL SHA-256 mismatch")


def _validate_manifest_counts(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    export_counts: Mapping[str, int],
) -> None:
    raw_counts = manifest.get("counts")
    if not isinstance(raw_counts, Mapping):
        raise MemPalaceExchangeError("manifest", f"{manifest_path}: counts must be an object")
    try:
        manifest_counts = {table: int(raw_counts.get(table, -1)) for table in EXPORT_TABLES}
    except (TypeError, ValueError) as exc:
        raise MemPalaceExchangeError("manifest", f"{manifest_path}: counts must be integers") from exc
    if manifest_counts != dict(export_counts):
        raise MemPalaceExchangeError(
            "manifest",
            f"{manifest_path}: stream count mismatch manifest={manifest_counts} jsonl={dict(export_counts)}",
        )


def _load_export_records(source_path: Path, jsonl_bytes: bytes) -> tuple[list[ExportRecord], dict[str, int]]:
    try:
        text = jsonl_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MemPalaceExchangeError("jsonl", f"{source_path}: invalid UTF-8: {exc}") from exc

    counts = {table: 0 for table in EXPORT_TABLES}
    records: list[ExportRecord] = []
    seen_header = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MemPalaceExchangeError("jsonl", f"{source_path}: invalid JSONL row: {exc}", line_number=line_number) from exc
        if not isinstance(payload, dict):
            raise MemPalaceExchangeError("jsonl", f"{source_path}: JSONL row must be an object", line_number=line_number)

        if payload.get("type") == "header":
            _validate_jsonl_header(payload, source_path, line_number=line_number)
            seen_header = True
            continue

        table = str(payload.get("table") or "")
        if table not in EXPORT_TABLES:
            raise MemPalaceExchangeError(
                "record_schema",
                f"{source_path}: unsupported stream {table!r}",
                stream=table or None,
                line_number=line_number,
            )
        expected_type = table[:-1]
        if payload.get("type") != expected_type:
            raise MemPalaceExchangeError(
                "record_schema",
                f"{source_path}: row type must be {expected_type!r}",
                stream=table,
                record_offset=counts[table] + 1,
                line_number=line_number,
            )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise MemPalaceExchangeError(
                "record_schema",
                f"{source_path}: row data must be an object",
                stream=table,
                record_offset=counts[table] + 1,
                line_number=line_number,
            )
        data = _validate_record_schema(
            table,
            data,
            source_path,
            line_number=line_number,
            record_offset=counts[table] + 1,
        )
        counts[table] += 1
        records.append(ExportRecord(line_number=line_number, stream=table, record_offset=counts[table], data=data))

    if not seen_header:
        raise MemPalaceExchangeError("jsonl", f"{source_path}: missing Mem Palace JSONL header")
    return records, counts


def _validate_jsonl_header(payload: Mapping[str, Any], source_path: Path, *, line_number: int) -> None:
    if payload.get("format") != EXPORT_FORMAT:
        raise MemPalaceExchangeError("jsonl", f"{source_path}: unsupported export format", line_number=line_number)
    if payload.get("version") != EXPORT_VERSION:
        raise MemPalaceExchangeError("jsonl", f"{source_path}: unsupported export version", line_number=line_number)
    if payload.get("tables") != list(EXPORT_TABLES):
        raise MemPalaceExchangeError("jsonl", f"{source_path}: stream order mismatch", line_number=line_number)


def _validate_record_schema(
    table: str,
    data: Mapping[str, Any],
    source_path: Path,
    *,
    line_number: int,
    record_offset: int,
) -> dict[str, Any]:
    allowed_fields = ALLOWED_RECORD_FIELDS[table]
    required_fields = REQUIRED_RECORD_FIELDS[table]
    fields = set(data)
    unknown = sorted(fields - allowed_fields)
    if unknown:
        raise MemPalaceExchangeError(
            "record_schema",
            f"{source_path}: unsupported fields: {', '.join(unknown)}",
            stream=table,
            record_offset=record_offset,
            line_number=line_number,
        )
    missing = sorted(field for field in required_fields if field not in fields)
    if missing:
        raise MemPalaceExchangeError(
            "record_schema",
            f"{source_path}: missing required fields: {', '.join(missing)}",
            stream=table,
            record_offset=record_offset,
            line_number=line_number,
        )
    _validate_id_field(data.get("id"), table, source_path, line_number=line_number, record_offset=record_offset)
    try:
        reject_secret_material(data, context=f"{table} import record")
    except ValueError as exc:
        raise MemPalaceExchangeError(
            "record_schema",
            str(exc),
            stream=table,
            record_offset=record_offset,
            line_number=line_number,
        ) from exc
    if table == "memories":
        try:
            return canonicalize_import_memory(data)
        except ValueError as exc:
            raise MemPalaceExchangeError(
                "record_schema",
                str(exc),
                stream=table,
                record_offset=record_offset,
                line_number=line_number,
            ) from exc
    return dict(data)


def _validate_id_field(value: Any, table: str, source_path: Path, *, line_number: int, record_offset: int) -> None:
    if table == "memories":
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MemPalaceExchangeError(
                "record_schema",
                f"{source_path}: memory id must be a positive integer",
                stream=table,
                record_offset=record_offset,
                line_number=line_number,
            )
        return
    if not isinstance(value, str) or not value.strip():
        raise MemPalaceExchangeError(
            "record_schema",
            f"{source_path}: id must be a non-empty string",
            stream=table,
            record_offset=record_offset,
            line_number=line_number,
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rollback_savepoint(conn: sqlite3.Connection, savepoint: str) -> None:
    try:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
    finally:
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")


def _select_rows(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    order_by = {
        "memories": "id",
        "entities": "id",
        "relationships": "created_at, id",
    }[table]
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}").fetchall()
    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    return dict(row)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _memory_duplicate_id(conn: sqlite3.Connection, data: Mapping[str, Any]) -> int | None:
    content_hash = str(data.get("content_hash") or "")
    if not content_hash:
        return None
    rows = conn.execute(
        """
        SELECT id, pinned, entities, importance, confidence, metadata
        FROM memories
        WHERE content_hash = ?
        ORDER BY id
        """,
        [content_hash],
    ).fetchall()
    return resolve_memory_duplicate(rows, data)


def _insert_memory_row(
    conn: sqlite3.Connection,
    data: Mapping[str, Any],
    *,
    replace: bool,
    minimum_destination_id: int = 0,
) -> int:
    row = _coerce_table_row(conn, "memories", data)
    source_memory_id = int(row["id"])
    if replace:
        if source_memory_id <= minimum_destination_id:
            raise ValueError(
                "replacement memory id must be greater than every superseded memory id"
            )
        _execute_insert(conn, "memories", row, verb="INSERT OR REPLACE")
        return source_memory_id

    occupied = conn.execute("SELECT 1 FROM memories WHERE id = ?", [source_memory_id]).fetchone()
    if occupied is not None or source_memory_id <= minimum_destination_id:
        row.pop("id")

    try:
        cursor = _execute_insert(conn, "memories", row, verb="INSERT")
    except sqlite3.IntegrityError:
        if "id" not in row:
            raise
        occupied = conn.execute("SELECT 1 FROM memories WHERE id = ?", [source_memory_id]).fetchone()
        if occupied is None:
            raise
        row.pop("id")
        cursor = _execute_insert(conn, "memories", row, verb="INSERT")

    destination_memory_id = source_memory_id if "id" in row else cursor.lastrowid
    if destination_memory_id is None:
        raise ValueError("imported memory did not receive a destination id")
    return int(destination_memory_id)


def _rewrite_memory_supersession(
    data: Mapping[str, Any],
    memory_id_map: Mapping[int, int],
) -> tuple[dict[str, Any], tuple[int, ...]]:
    row = dict(data)
    metadata, superseded_ids = remap_supersession_metadata(
        row.get("metadata"),
        memory_id_map,
    )
    row["metadata"] = canonical_metadata(metadata)
    return row, superseded_ids


def _rewrite_relationship_source(
    conn: sqlite3.Connection,
    data: Mapping[str, Any],
    memory_id_map: Mapping[int, int],
) -> dict[str, Any]:
    row = dict(data)
    source_memory_id = row.get("source_memory_id")
    if source_memory_id is None:
        return row
    try:
        source_memory_id = int(source_memory_id)
    except (TypeError, ValueError):
        row["source_memory_id"] = None
        return row
    if source_memory_id in memory_id_map:
        row["source_memory_id"] = memory_id_map[source_memory_id]
        return row
    exists = conn.execute("SELECT id FROM memories WHERE id = ? LIMIT 1", [source_memory_id]).fetchone()
    if exists is None:
        row["source_memory_id"] = None
    return row


def _insert_row(conn: sqlite3.Connection, table: str, data: Mapping[str, Any], *, replace: bool) -> bool:
    row = _coerce_table_row(conn, table, data)
    if not row:
        return False
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    cursor = _execute_insert(conn, table, row, verb=verb)
    return cursor.rowcount > 0


def _coerce_table_row(
    conn: sqlite3.Connection,
    table: str,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    columns = _table_columns(conn, table)
    return {key: _coerce_value(key, value) for key, value in data.items() if key in columns}


def _execute_insert(
    conn: sqlite3.Connection,
    table: str,
    row: Mapping[str, Any],
    *,
    verb: str,
) -> sqlite3.Cursor:
    placeholders = ", ".join("?" for _ in row)
    column_sql = ", ".join(row)
    return conn.execute(
        f"{verb} INTO {table} ({column_sql}) VALUES ({placeholders})",
        list(row.values()),
    )


def _coerce_value(key: str, value: Any) -> Any:
    if key == "metadata" and isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    if key == "pinned":
        return int(value)
    return value


def _rebuild_fts(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
    except sqlite3.OperationalError as exc:
        if not is_missing_fts_error(exc):
            raise
