#!/usr/bin/env python3
"""Standalone Mem Palace local memory for Tokimon.

The public API is the :class:`MemPalace` facade. It wraps a local SQLite
store, keeps the command-line interface small, and hides indexing, migrations,
ranking, and temporal fact storage behind a stable module boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tokimon.runtime import runtime_root as repository_runtime_root, source_root


WORKSPACE_ROOT = source_root()
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tokimon.memory.mem_palace.identity import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMPORTANCE,
    DEFAULT_ROOM,
    DEFAULT_SCOPE,
    SUPERSEDES_MEMORY_IDS_FIELD,
    canonical_metadata,
    canonical_pinned,
    memory_content_hash,
    normalize_tags,
    remap_supersession_metadata,
    resolve_memory_duplicate,
    supersession_memory_ids,
)
from tokimon.memory.mem_palace.evaluation import (
    evaluate_recall,
    load_recall_cases,
    render_recall_evaluation,
)
from tokimon.memory.mem_palace.exchange import (
    backup_database,
    ensure_instance_id,
    export_jsonl,
    import_jsonl,
)
from tokimon.memory.mem_palace.ingest import collect_documents
from tokimon.memory.mem_palace.project_context import (
    ProjectMemoryScope,
    resolve_project_memory_scope,
)
from tokimon.memory.mem_palace.secret_guard import find_secret_indicators, reject_secret_material
from tokimon.memory.mem_palace.sqlite_support import is_missing_fts_error

RUNTIME_ROOT = repository_runtime_root(WORKSPACE_ROOT)
DEFAULT_DB_PATH = RUNTIME_ROOT / "memory" / "memory.db"
DEFAULT_C4_OVERVIEW_PATH = WORKSPACE_ROOT / "docs" / "guidelines" / "repository-guidelines.md"
DEFAULT_BACKUP_DIR = RUNTIME_ROOT / "memory" / "backups"

SCHEMA_VERSION = 3
ALLOWED_KINDS = (
    "decision",
    "preference",
    "context",
    "task",
    "fact",
    "note",
    "event",
    "milestone",
    "problem",
    "instruction",
    "avoid",
    "artifact",
)
RELATION_DIRECTIONS = ("outgoing", "incoming", "both")


def strip_markdown_frontmatter(content: str) -> str:
    """Remove supported C4-style YAML frontmatter from Markdown content."""

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if not lines or lines[0].strip() != "---":
        return content
    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing_index is None:
        return content
    block = "\n".join(lines[1:closing_index])
    if not re.search(r"^(?:title|summary|read_when):", block, flags=re.MULTILINE):
        return content
    return "\n".join(lines[closing_index + 1 :]).lstrip()

TOKEN_RE = re.compile(r"[\w:-]{2,}", re.UNICODE)
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_.-]*(?:\s+[A-Z][A-Za-z0-9_.-]*){0,3}\b")
ENTITY_STOPLIST = {
    "A",
    "An",
    "And",
    "At",
    "But",
    "By",
    "For",
    "From",
    "If",
    "In",
    "No",
    "Not",
    "Of",
    "On",
    "Or",
    "So",
    "The",
    "Then",
    "This",
    "To",
    "When",
    "Where",
    "Why",
    "With",
}
FTS_COLUMNS = ("summary", "details", "tags", "source", "scope", "room", "entities")


@dataclass(frozen=True)
class MemoryRecord:
    """A durable local memory row."""

    id: int
    kind: str
    summary: str
    details: str
    tags: tuple[str, ...]
    source: str
    scope: str
    room: str
    entities: tuple[str, ...]
    importance: float
    confidence: float
    metadata: Mapping[str, Any]
    pinned: bool
    created_at: str
    updated_at: str
    content_hash: str
    use_count: int
    last_used_at: str | None

    @property
    def search_text(self) -> str:
        return "\n".join(
            part
            for part in (
                self.summary,
                self.details,
                " ".join(self.tags),
                self.source,
                self.scope,
                self.room,
                " ".join(self.entities),
            )
            if part
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "summary": self.summary,
            "details": self.details,
            "tags": ", ".join(self.tags),
            "source": self.source,
            "scope": self.scope,
            "room": self.room,
            "entities": ", ".join(self.entities),
            "importance": self.importance,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
            "pinned": self.pinned,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "content_hash": self.content_hash,
            "use_count": self.use_count,
            "last_used_at": self.last_used_at,
        }


@dataclass(frozen=True)
class SearchResult:
    """Ranked memory recall result."""

    rank: int
    memory: MemoryRecord
    score: float
    matched_fields: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = self.memory.to_dict()
        data["rank"] = self.rank
        data["score"] = round(self.score, 4)
        data["matched_fields"] = list(self.matched_fields)
        data["reasons"] = list(self.reasons)
        return data


@dataclass(frozen=True)
class RelationshipFact:
    """Temporal entity relationship stored in the local Mem Palace graph."""

    id: str
    direction: str
    subject: str
    predicate: str
    object: str
    valid_from: str | None
    valid_to: str | None
    confidence: float
    source_memory_id: int | None
    source: str
    created_at: str
    updated_at: str

    @property
    def current(self) -> bool:
        return self.valid_to is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "direction": self.direction,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "confidence": self.confidence,
            "source_memory_id": self.source_memory_id,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current": self.current,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def bounded_float(
    value: float | int | str | None,
    *,
    minimum: float,
    maximum: float,
    default: float,
    name: str,
) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return number


def clamp_float(
    value: float | int | str | None,
    *,
    minimum: float,
    maximum: float,
    default: float,
) -> float:
    return bounded_float(value, minimum=minimum, maximum=maximum, default=default, name="value")


def query_tokens(query: str | None) -> tuple[str, ...]:
    if not query:
        return ()
    return tuple(dict.fromkeys(token.lower() for token in TOKEN_RE.findall(query)))


def normalize_entities(entities: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for raw in entities:
        for part in str(raw).split(","):
            entity = " ".join(part.strip().split())
            if entity and entity not in ENTITY_STOPLIST and entity not in normalized:
                normalized.append(entity)
    return normalized


def extract_entities(text: str, explicit_entities: Iterable[str] = ()) -> list[str]:
    """Return conservative retrieval hints without asserting facts."""

    explicit = normalize_entities(explicit_entities)
    counts: dict[str, int] = {}
    for candidate in ENTITY_RE.findall(text):
        entity = " ".join(candidate.strip().split())
        if entity in ENTITY_STOPLIST:
            continue
        counts[entity] = counts.get(entity, 0) + 1

    inferred = [
        entity
        for entity, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
        if count >= 2 or " " in entity
    ]
    return normalize_entities([*explicit, *inferred])[:20]


def json_object(value: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("metadata must be a JSON object")
    return parsed


def parse_json_object(value: str | None) -> dict[str, Any]:
    return json_object(value)


def safe_json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def serialize_json_object(value: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(value or {}), sort_keys=True, separators=(",", ":"))


def normalize_entity_id(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().lower()).strip("_")
    if normalized:
        return normalized
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def normalize_predicate(predicate: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", predicate.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("predicate is required")
    return normalized


def resolve_c4_overview_path(path: Path | str | None = None) -> Path:
    if path is None:
        return DEFAULT_C4_OVERVIEW_PATH
    return Path(path).expanduser().resolve()


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    initialize(conn)
    return conn


def initialize(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            summary TEXT NOT NULL,
            details TEXT NOT NULL DEFAULT '',
            tags TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL DEFAULT 'tokimon',
            room TEXT NOT NULL DEFAULT 'general',
            entities TEXT NOT NULL DEFAULT '',
            content_hash TEXT NOT NULL DEFAULT '',
            importance REAL NOT NULL DEFAULT 3.0,
            confidence REAL NOT NULL DEFAULT 1.0,
            metadata TEXT NOT NULL DEFAULT '{}',
            pinned INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            use_count INTEGER NOT NULL DEFAULT 0,
            last_used_at TEXT
        )
        """
    )
    _migrate_memory_columns(conn)
    _initialize_relationship_schema(conn)
    ensure_instance_id(conn, commit=False)
    _backfill_memory_hashes(conn)
    _initialize_fts(conn)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        [str(SCHEMA_VERSION)],
    )
    conn.commit()


def _migrate_memory_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    migrations = {
        "scope": "ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'tokimon'",
        "room": "ALTER TABLE memories ADD COLUMN room TEXT NOT NULL DEFAULT 'general'",
        "entities": "ALTER TABLE memories ADD COLUMN entities TEXT NOT NULL DEFAULT ''",
        "content_hash": "ALTER TABLE memories ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''",
        "importance": "ALTER TABLE memories ADD COLUMN importance REAL NOT NULL DEFAULT 3.0",
        "confidence": "ALTER TABLE memories ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0",
        "metadata": "ALTER TABLE memories ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'",
        "use_count": "ALTER TABLE memories ADD COLUMN use_count INTEGER NOT NULL DEFAULT 0",
        "last_used_at": "ALTER TABLE memories ADD COLUMN last_used_at TEXT",
    }
    for column, sql in migrations.items():
        if column not in columns:
            conn.execute(sql)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_scope_room ON memories(scope, room)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_entities ON memories(entities)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_used_at ON memories(last_used_at)")


def _initialize_relationship_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'unknown',
            aliases TEXT NOT NULL DEFAULT '',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_id TEXT NOT NULL,
            valid_from TEXT,
            valid_to TEXT,
            confidence REAL NOT NULL DEFAULT 1.0,
            source_memory_id INTEGER,
            source TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (subject_id) REFERENCES entities(id),
            FOREIGN KEY (object_id) REFERENCES entities(id),
            FOREIGN KEY (source_memory_id) REFERENCES memories(id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relationship_subject ON relationships(subject_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relationship_object ON relationships(object_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relationship_predicate ON relationships(predicate)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relationship_valid ON relationships(valid_from, valid_to)")


def _backfill_memory_hashes(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT id, kind, summary, details, tags, source, scope, room
        FROM memories
        WHERE content_hash IS NULL OR content_hash = ''
        """
    ).fetchall()
    for row in rows:
        content_hash = memory_content_hash(
            kind=row["kind"],
            summary=row["summary"],
            details=row["details"],
            tags=normalize_tags([row["tags"]]),
            source=row["source"],
            scope=row["scope"],
            room=row["room"],
        )
        conn.execute("UPDATE memories SET content_hash = ? WHERE id = ?", [content_hash, row["id"]])


def _drop_fts_objects(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TRIGGER IF EXISTS memories_ai")
    conn.execute("DROP TRIGGER IF EXISTS memories_ad")
    conn.execute("DROP TRIGGER IF EXISTS memories_au")
    conn.execute("DROP TABLE IF EXISTS memory_fts")


def _initialize_fts(conn: sqlite3.Connection) -> None:
    try:
        existing_columns = [row["name"] for row in conn.execute("PRAGMA table_info(memory_fts)").fetchall()]
        if existing_columns and tuple(existing_columns[: len(FTS_COLUMNS)]) != FTS_COLUMNS:
            _drop_fts_objects(conn)

        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                summary,
                details,
                tags,
                source,
                scope,
                room,
                entities,
                content='memories',
                content_rowid='id'
            )
            """
        )
        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memory_fts(rowid, summary, details, tags, source, scope, room, entities)
                VALUES (new.id, new.summary, new.details, new.tags, new.source, new.scope, new.room, new.entities);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, summary, details, tags, source, scope, room, entities)
                VALUES ('delete', old.id, old.summary, old.details, old.tags, old.source, old.scope, old.room, old.entities);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, summary, details, tags, source, scope, room, entities)
                VALUES ('delete', old.id, old.summary, old.details, old.tags, old.source, old.scope, old.room, old.entities);
                INSERT INTO memory_fts(rowid, summary, details, tags, source, scope, room, entities)
                VALUES (new.id, new.summary, new.details, new.tags, new.source, new.scope, new.room, new.entities);
            END;
            """
        )

        row_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        fts_count = conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]
        if row_count != fts_count:
            conn.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
    except sqlite3.OperationalError as exc:
        if not is_missing_fts_error(exc):
            raise


def has_fts(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1 FROM memory_fts LIMIT 1").fetchone()
        return True
    except sqlite3.OperationalError as exc:
        if is_missing_fts_error(exc):
            return False
        raise


def build_filters(
    kind: str | None = None,
    tag: str | None = None,
    scope: str | None = None,
    scopes: Sequence[str] | None = None,
    room: str | None = None,
    entity: str | None = None,
    *,
    table_alias: str | None = None,
) -> tuple[str, list[str]]:
    clauses: list[str] = []
    params: list[str] = []
    prefix = f"{table_alias}." if table_alias else ""

    if kind:
        clauses.append(f"{prefix}kind = ?")
        params.append(kind.strip().lower())

    if tag:
        clauses.append(f"LOWER({prefix}tags) LIKE ?")
        params.append(f"%{tag.strip().lower()}%")

    selected_scopes = []
    for value in [scope, *(scopes or ())]:
        normalized = (value or "").strip().lower()
        if normalized and normalized not in selected_scopes:
            selected_scopes.append(normalized)
    if len(selected_scopes) == 1:
        clauses.append(f"{prefix}scope = ?")
        params.append(selected_scopes[0])
    elif selected_scopes:
        placeholders = ", ".join("?" for _ in selected_scopes)
        clauses.append(f"{prefix}scope IN ({placeholders})")
        params.extend(selected_scopes)

    if room:
        clauses.append(f"{prefix}room = ?")
        params.append(room.strip().lower())

    if entity:
        clauses.append(f"LOWER({prefix}entities) LIKE ?")
        params.append(f"%{entity.strip().lower()}%")

    if not clauses:
        return "", params
    return " AND " + " AND ".join(clauses), params


def build_match_query(query: str) -> str | None:
    tokens = query_tokens(query)
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)


def row_to_record(row: sqlite3.Row | Mapping[str, Any]) -> MemoryRecord:
    data = dict(row)
    return MemoryRecord(
        id=int(data["id"]),
        kind=str(data["kind"]),
        summary=str(data["summary"]),
        details=str(data.get("details") or ""),
        tags=tuple(normalize_tags([data.get("tags") or ""])),
        source=str(data.get("source") or ""),
        scope=str(data.get("scope") or DEFAULT_SCOPE),
        room=str(data.get("room") or DEFAULT_ROOM),
        entities=tuple(normalize_entities([data.get("entities") or ""])),
        importance=float(data.get("importance") if data.get("importance") is not None else DEFAULT_IMPORTANCE),
        confidence=float(data.get("confidence") if data.get("confidence") is not None else DEFAULT_CONFIDENCE),
        metadata=safe_json_loads(data.get("metadata")),
        pinned=bool(data.get("pinned")),
        created_at=str(data["created_at"]),
        updated_at=str(data["updated_at"]),
        content_hash=str(data.get("content_hash") or ""),
        use_count=int(data.get("use_count") or 0),
        last_used_at=data.get("last_used_at"),
    )


def row_to_dict(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    return row_to_record(row).to_dict()


def mark_memories_used(conn: sqlite3.Connection, memory_ids: Iterable[int]) -> int:
    ids = sorted({int(memory_id) for memory_id in memory_ids if int(memory_id) > 0})
    if not ids:
        return 0
    placeholders = ", ".join("?" for _ in ids)
    cursor = conn.execute(
        f"""
        UPDATE memories
        SET use_count = COALESCE(use_count, 0) + 1,
            last_used_at = ?
        WHERE id IN ({placeholders})
        """,
        [utc_now(), *ids],
    )
    conn.commit()
    return int(cursor.rowcount)


def _superseded_memory_filter_sql(
    conn: sqlite3.Connection,
    *,
    table_alias: str | None = None,
) -> str:
    rows = conn.execute("SELECT id, metadata FROM memories").fetchall()
    existing_ids = {int(row["id"]) for row in rows}
    superseded_ids = sorted(
        {
            superseded_id
            for row in rows
            for superseded_id in supersession_memory_ids(row["metadata"])
            if superseded_id in existing_ids and superseded_id < int(row["id"])
        }
    )
    if not superseded_ids:
        return ""
    prefix = f"{table_alias}." if table_alias else ""
    id_list = ", ".join(str(memory_id) for memory_id in superseded_ids)
    return f" AND {prefix}id NOT IN ({id_list})"


def read_c4_overview(path: Path | str = DEFAULT_C4_OVERVIEW_PATH) -> str:
    overview_path = resolve_c4_overview_path(path)
    if not overview_path.exists():
        return (
            f"No repository guidelines found at {overview_path}\n"
            "Create the repository routing document at docs/guidelines/repository-guidelines.md"
        )
    return strip_markdown_frontmatter(overview_path.read_text(encoding="utf-8")).strip()


def add_memory(
    conn: sqlite3.Connection,
    *,
    kind: str,
    summary: str,
    details: str = "",
    tags: Iterable[str] = (),
    source: str = "",
    pinned: bool = False,
    scope: str = DEFAULT_SCOPE,
    room: str = DEFAULT_ROOM,
    entities: Iterable[str] = (),
    importance: float = DEFAULT_IMPORTANCE,
    confidence: float = DEFAULT_CONFIDENCE,
    metadata: Mapping[str, Any] | str | None = None,
    allow_duplicate: bool = False,
    auto_extract_entities: bool = True,
) -> int:
    summary = summary.strip()
    details = details.strip()
    source = source.strip()
    kind = kind.strip().lower()
    scope = (scope or DEFAULT_SCOPE).strip().lower()
    room = (room or DEFAULT_ROOM).strip().lower()

    if not summary:
        raise ValueError("summary is required")
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"kind must be one of: {', '.join(ALLOWED_KINDS)}")

    tags_list = normalize_tags(tags)
    entities_list = normalize_entities(entities)
    if auto_extract_entities:
        entities_list = extract_entities(f"{summary}\n{details}", entities_list)
    metadata_dict = json_object(metadata)
    existing_ids = (
        {int(row["id"]): int(row["id"]) for row in conn.execute("SELECT id FROM memories")}
        if SUPERSEDES_MEMORY_IDS_FIELD in metadata_dict else {}
    )
    metadata_dict, _ = remap_supersession_metadata(metadata_dict, existing_ids)
    metadata_text = canonical_metadata(metadata_dict)
    pinned_value = canonical_pinned(pinned)
    reject_secret_material(
        {
            "kind": kind,
            "summary": summary,
            "details": details,
            "tags": tags_list,
            "source": source,
            "scope": scope,
            "room": room,
            "entities": entities_list,
            "metadata": metadata_dict,
        },
        context="memory",
    )
    importance_value = bounded_float(
        importance,
        minimum=0.0,
        maximum=5.0,
        default=DEFAULT_IMPORTANCE,
        name="importance",
    )
    confidence_value = bounded_float(
        confidence,
        minimum=0.0,
        maximum=1.0,
        default=DEFAULT_CONFIDENCE,
        name="confidence",
    )
    content_hash = memory_content_hash(
        kind=kind,
        summary=summary,
        details=details,
        tags=tags_list,
        source=source,
        scope=scope,
        room=room,
    )

    if not allow_duplicate:
        existing = conn.execute(
            """
            SELECT id, pinned, entities, importance, confidence, metadata
            FROM memories
            WHERE content_hash = ?
            ORDER BY id
            """,
            [content_hash],
        ).fetchall()
        duplicate_id = resolve_memory_duplicate(
            existing,
            {
                "pinned": pinned_value,
                "entities": entities_list,
                "importance": importance_value,
                "confidence": confidence_value,
                "metadata": metadata_dict,
            },
        )
        if duplicate_id is not None:
            return duplicate_id

    now = utc_now()
    cursor = conn.execute(
        """
        INSERT INTO memories(
            kind, summary, details, tags, source, scope, room, entities,
            content_hash, importance, confidence, metadata, pinned, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            kind,
            summary,
            details,
            ", ".join(tags_list),
            source,
            scope,
            room,
            ", ".join(entities_list),
            content_hash,
            importance_value,
            confidence_value,
            metadata_text,
            pinned_value,
            now,
            now,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def get_memory(conn: sqlite3.Connection, memory_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM memories WHERE id = ?", [int(memory_id)]).fetchone()
    if row is None:
        return None
    return row_to_dict(row)


def _memory_records(
    conn: sqlite3.Connection,
    filter_sql: str,
    filter_params: list[Any],
    *,
    order_by: str,
    limit: int,
) -> list[MemoryRecord]:
    superseded_sql = _superseded_memory_filter_sql(conn)
    rows = conn.execute(
        f"""
        SELECT *
        FROM memories
        WHERE 1 = 1 {filter_sql} {superseded_sql}
        ORDER BY {order_by}
        LIMIT ?
        """,
        [*filter_params, max(0, int(limit))],
    ).fetchall()
    return [row_to_record(row) for row in rows]




def _recent_records(
    conn: sqlite3.Connection,
    *,
    limit: int = 10,
    kind: str | None = None,
    tag: str | None = None,
    scope: str | None = None,
    scopes: Sequence[str] | None = None,
    room: str | None = None,
    entity: str | None = None,
) -> list[MemoryRecord]:
    filter_sql, filter_params = build_filters(
        kind=kind,
        tag=tag,
        scope=scope,
        scopes=scopes,
        room=room,
        entity=entity,
    )
    return _memory_records(conn, filter_sql, filter_params, order_by="pinned DESC, created_at DESC", limit=limit)

def recent_memories(
    conn: sqlite3.Connection,
    *,
    limit: int = 10,
    kind: str | None = None,
    tag: str | None = None,
    scope: str | None = None,
    scopes: Sequence[str] | None = None,
    room: str | None = None,
    entity: str | None = None,
) -> list[dict[str, Any]]:
    return [record.to_dict() for record in _recent_records(
        conn, limit=limit, kind=kind, tag=tag, scope=scope, scopes=scopes, room=room, entity=entity,
    )]


def _essential_records(
    conn: sqlite3.Connection,
    *,
    limit: int = 8,
    scope: str | None = None,
    scopes: Sequence[str] | None = None,
    room: str | None = None,
) -> list[MemoryRecord]:
    filter_sql, filter_params = build_filters(scope=scope, scopes=scopes, room=room)
    return _memory_records(
        conn, filter_sql, filter_params,
        order_by="pinned DESC, importance DESC, confidence DESC, created_at DESC", limit=limit,
    )

def essential_memories(
    conn: sqlite3.Connection,
    *,
    limit: int = 8,
    scope: str | None = None,
    scopes: Sequence[str] | None = None,
    room: str | None = None,
) -> list[dict[str, Any]]:
    return [record.to_dict() for record in _essential_records(
        conn, limit=limit, scope=scope, scopes=scopes, room=room,
    )]


def _like_candidate_sql(tokens: Sequence[str], table_alias: str | None = None) -> tuple[str, list[str]]:
    if not tokens:
        return "", []
    prefix = f"{table_alias}." if table_alias else ""
    fields = ("summary", "details", "tags", "source", "scope", "room", "entities")
    clauses: list[str] = []
    params: list[str] = []
    for token in tokens:
        token_clauses = [f"LOWER({prefix}{field}) LIKE ?" for field in fields]
        clauses.append("(" + " OR ".join(token_clauses) + ")")
        params.extend([f"%{token.lower()}%"] * len(fields))
    return " AND (" + " OR ".join(clauses) + ")", params


def _bm25_scores(query: str, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    terms = set(query_tokens(query))
    if not terms or not documents:
        return [0.0] * len(documents)

    tokenized = [query_tokens(document) for document in documents]
    lengths = [len(tokens) for tokens in tokenized]
    if not any(lengths):
        return [0.0] * len(documents)

    avgdl = sum(lengths) / len(lengths)
    df = {term: sum(1 for tokens in tokenized if term in tokens) for term in terms}
    idf = {
        term: math.log((len(documents) - count + 0.5) / (count + 0.5) + 1.0)
        for term, count in df.items()
    }

    scores: list[float] = []
    for tokens, length in zip(tokenized, lengths):
        if length == 0:
            scores.append(0.0)
            continue
        freqs: dict[str, int] = {}
        for token in tokens:
            if token in terms:
                freqs[token] = freqs.get(token, 0) + 1
        score = 0.0
        for term, freq in freqs.items():
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * length / avgdl)
            score += idf[term] * numerator / denominator
        scores.append(score)
    return scores


def _matched_fields(record: MemoryRecord, tokens: Sequence[str], query: str) -> tuple[str, ...]:
    if not tokens:
        return ()
    fields = {
        "summary": record.summary,
        "details": record.details,
        "tags": " ".join(record.tags),
        "source": record.source,
        "scope": record.scope,
        "room": record.room,
        "entities": " ".join(record.entities),
    }
    query_lower = query.lower()
    matched: list[str] = []
    for name, value in fields.items():
        lowered = value.lower()
        if query_lower and query_lower in lowered:
            matched.append(name)
        elif any(token in lowered for token in tokens):
            matched.append(name)
    return tuple(matched)


def _score_records(records: Sequence[MemoryRecord], query: str) -> list[SearchResult]:
    if not records:
        return []

    tokens = query_tokens(query)
    bm25 = _bm25_scores(query, [record.search_text for record in records])
    max_bm25 = max(bm25) if bm25 else 0.0
    normalized_bm25 = [score / max_bm25 if max_bm25 > 0 else 0.0 for score in bm25]
    query_lower = query.lower().strip()

    scored: list[tuple[float, MemoryRecord, tuple[str, ...], tuple[str, ...]]] = []
    for record, lexical_score in zip(records, normalized_bm25):
        matched = _matched_fields(record, tokens, query)
        text_lower = record.search_text.lower()
        score = lexical_score
        reasons: list[str] = []

        if matched:
            reasons.append("lexical match")
        if query_lower and query_lower in text_lower:
            score += 0.35
            reasons.append("exact phrase")
        if "summary" in matched:
            score += 0.25
            reasons.append("summary match")
        if "tags" in matched or "entities" in matched:
            score += 0.18
            reasons.append("index hint match")
        if "scope" in matched or "room" in matched:
            score += 0.12
            reasons.append("scope match")
        if record.pinned:
            score += 0.30
            reasons.append("pinned")
        if record.importance:
            score += min(record.importance, 5.0) * 0.06
        if record.confidence:
            score += min(record.confidence, 1.0) * 0.04

        scored.append((score, record, matched, tuple(dict.fromkeys(reasons))))

    scored.sort(key=lambda item: (item[0], item[1].pinned, item[1].importance, item[1].created_at), reverse=True)
    return [
        SearchResult(rank=rank, memory=record, score=score, matched_fields=matched, reasons=reasons)
        for rank, (score, record, matched, reasons) in enumerate(scored, start=1)
    ]


def search_memory_results(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 5,
    kind: str | None = None,
    tag: str | None = None,
    scope: str | None = None,
    scopes: Sequence[str] | None = None,
    room: str | None = None,
    entity: str | None = None,
) -> list[SearchResult]:
    results = _search_memory_results(
        conn, query, limit=limit, kind=kind, tag=tag, scope=scope,
        scopes=scopes, room=room, entity=entity,
    )
    mark_memories_used(conn, [result.memory.id for result in results])
    return results


def _search_memory_results(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 5,
    kind: str | None = None,
    tag: str | None = None,
    scope: str | None = None,
    scopes: Sequence[str] | None = None,
    room: str | None = None,
    entity: str | None = None,
) -> list[SearchResult]:
    query = query.strip()
    if not query:
        raise ValueError("query is required")
    tokens = query_tokens(query)
    if not tokens:
        raise ValueError("query must contain at least one searchable token")

    overfetch = max(int(limit) * 6, 30)
    filter_sql, filter_params = build_filters(
        kind=kind,
        tag=tag,
        scope=scope,
        scopes=scopes,
        room=room,
        entity=entity,
        table_alias="m",
    )
    candidates: dict[int, MemoryRecord] = {}
    superseded_sql = _superseded_memory_filter_sql(
        conn,
        table_alias="m",
    )

    if has_fts(conn):
        match_query = build_match_query(query)
        if match_query:
            try:
                rows = conn.execute(
                    f"""
                    SELECT m.*
                    FROM memory_fts
                    JOIN memories AS m ON m.id = memory_fts.rowid
                    WHERE memory_fts MATCH ? {filter_sql} {superseded_sql}
                    ORDER BY bm25(memory_fts) ASC
                    LIMIT ?
                    """,
                    [match_query, *filter_params, overfetch],
                ).fetchall()
                for row in rows:
                    record = row_to_record(row)
                    candidates[record.id] = record
            except sqlite3.OperationalError as exc:
                if not is_missing_fts_error(exc):
                    raise

    like_sql, like_params = _like_candidate_sql(tokens, table_alias="m")
    rows = conn.execute(
        f"""
        SELECT m.*
        FROM memories AS m
        WHERE 1 = 1 {filter_sql} {like_sql} {superseded_sql}
        ORDER BY pinned DESC, importance DESC, created_at DESC
        LIMIT ?
        """,
        [*filter_params, *like_params, overfetch],
    ).fetchall()
    for row in rows:
        record = row_to_record(row)
        candidates[record.id] = record

    return _score_records(list(candidates.values()), query)[: max(0, int(limit))]


def search_memories(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 5,
    kind: str | None = None,
    tag: str | None = None,
    scope: str | None = None,
    room: str | None = None,
    entity: str | None = None,
) -> list[dict[str, Any]]:
    return [
        result.to_dict()
        for result in search_memory_results(
            conn,
            query,
            limit=limit,
            kind=kind,
            tag=tag,
            scope=scope,
            room=room,
            entity=entity,
        )
    ]


def add_entity(
    conn: sqlite3.Connection,
    name: str,
    *,
    entity_type: str = "unknown",
    aliases: Iterable[str] = (),
    metadata: Mapping[str, Any] | str | None = None,
) -> str:
    name = " ".join(name.strip().split())
    if not name:
        raise ValueError("entity name is required")
    entity_id = normalize_entity_id(name)
    aliases_text = ", ".join(normalize_entities(aliases))
    metadata_value = json_object(metadata)
    reject_secret_material(
        {"name": name, "kind": entity_type, "aliases": aliases_text, "metadata": metadata_value},
        context="entity",
    )
    metadata_text = serialize_json_object(metadata_value)
    now = utc_now()
    conn.execute(
        """
        INSERT OR IGNORE INTO entities(id, name, kind, aliases, metadata, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [entity_id, name, entity_type, aliases_text, metadata_text, now, now],
    )
    conn.execute(
        """
        UPDATE entities
        SET name = ?, kind = ?, aliases = ?, metadata = ?, updated_at = ?
        WHERE id = ?
        """,
        [name, entity_type, aliases_text, metadata_text, now, entity_id],
    )
    conn.commit()
    return entity_id


def add_relationship(
    conn: sqlite3.Connection,
    subject: str,
    predicate: str,
    obj: str,
    *,
    valid_from: str | None = None,
    valid_to: str | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    source_memory_id: int | None = None,
    source: str = "",
) -> str:
    reject_secret_material(
        {
            "subject": subject, "predicate": predicate, "object": obj,
            "valid_from": valid_from, "valid_to": valid_to, "source": source,
        },
        context="relationship",
    )
    subject_id = add_entity(conn, subject)
    object_id = add_entity(conn, obj)
    normalized_predicate = normalize_predicate(predicate)
    confidence_value = bounded_float(
        confidence,
        minimum=0.0,
        maximum=1.0,
        default=DEFAULT_CONFIDENCE,
        name="confidence",
    )

    existing = conn.execute(
        """
        SELECT id
        FROM relationships
        WHERE subject_id = ? AND predicate = ? AND object_id = ? AND valid_to IS NULL
        LIMIT 1
        """,
        [subject_id, normalized_predicate, object_id],
    ).fetchone()
    if existing is not None:
        return str(existing["id"])

    now = utc_now()
    digest = hashlib.sha256(
        f"{subject_id}|{normalized_predicate}|{object_id}|{valid_from or ''}|{now}".encode("utf-8")
    ).hexdigest()[:12]
    relationship_id = f"rel_{subject_id}_{normalized_predicate}_{object_id}_{digest}"
    conn.execute(
        """
        INSERT INTO relationships(
            id, subject_id, predicate, object_id, valid_from, valid_to,
            confidence, source_memory_id, source, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            relationship_id,
            subject_id,
            normalized_predicate,
            object_id,
            valid_from,
            valid_to,
            confidence_value,
            source_memory_id,
            source.strip(),
            now,
            now,
        ],
    )
    conn.commit()
    return relationship_id


def invalidate_relationship(
    conn: sqlite3.Connection,
    subject: str,
    predicate: str,
    obj: str,
    *,
    ended: str | None = None,
) -> int:
    subject_id = normalize_entity_id(subject)
    object_id = normalize_entity_id(obj)
    normalized_predicate = normalize_predicate(predicate)
    ended = ended or utc_now().split("T", 1)[0]
    reject_secret_material({"valid_to": ended}, context="relationship invalidation")
    cursor = conn.execute(
        """
        UPDATE relationships
        SET valid_to = ?, updated_at = ?
        WHERE subject_id = ? AND predicate = ? AND object_id = ? AND valid_to IS NULL
        """,
        [ended, utc_now(), subject_id, normalized_predicate, object_id],
    )
    conn.commit()
    return int(cursor.rowcount)


def _as_of_sql(as_of: str | None) -> tuple[str, list[str]]:
    if not as_of:
        return "", []
    return " AND (r.valid_from IS NULL OR r.valid_from <= ?) AND (r.valid_to IS NULL OR r.valid_to >= ?)", [as_of, as_of]


def _relationship_from_row(row: sqlite3.Row, *, direction: str) -> RelationshipFact:
    return RelationshipFact(
        id=str(row["id"]),
        direction=direction,
        subject=str(row["subject_name"]),
        predicate=str(row["predicate"]),
        object=str(row["object_name"]),
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        confidence=float(row["confidence"]),
        source_memory_id=row["source_memory_id"],
        source=str(row["source"] or ""),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _entity_facts(
    conn: sqlite3.Connection,
    name: str,
    *,
    as_of: str | None = None,
    direction: str = "both",
    limit: int = 50,
) -> list[RelationshipFact]:
    if direction not in RELATION_DIRECTIONS:
        raise ValueError(f"direction must be one of: {', '.join(RELATION_DIRECTIONS)}")
    entity_id = normalize_entity_id(name)
    as_of_sql, as_of_params = _as_of_sql(as_of)
    facts: list[RelationshipFact] = []

    for fact_direction, column in (("outgoing", "subject_id"), ("incoming", "object_id")):
        if direction not in (fact_direction, "both"):
            continue
        rows = conn.execute(
            f"""
            SELECT r.*, s.name AS subject_name, o.name AS object_name
            FROM relationships AS r
            JOIN entities AS s ON s.id = r.subject_id
            JOIN entities AS o ON o.id = r.object_id
            WHERE r.{column} = ? {as_of_sql}
            ORDER BY COALESCE(r.valid_from, r.created_at) DESC, r.created_at DESC
            LIMIT ?
            """,
            [entity_id, *as_of_params, max(0, int(limit))],
        ).fetchall()
        facts.extend(_relationship_from_row(row, direction=fact_direction) for row in rows)

    facts.sort(key=lambda fact: (fact.valid_from or fact.created_at, fact.created_at), reverse=True)
    return facts[: max(0, int(limit))]



def query_entity(
    conn: sqlite3.Connection,
    name: str,
    *,
    as_of: str | None = None,
    direction: str = "both",
    limit: int = 50,
) -> list[dict[str, Any]]:
    return [fact.to_dict() for fact in _entity_facts(conn, name, as_of=as_of, direction=direction, limit=limit)]


def _timeline_facts(conn: sqlite3.Connection, entity: str | None = None, *, limit: int = 100) -> list[RelationshipFact]:
    params: list[Any] = []
    where_sql = ""
    if entity:
        entity_id = normalize_entity_id(entity)
        where_sql = "WHERE r.subject_id = ? OR r.object_id = ?"
        params.extend([entity_id, entity_id])

    rows = conn.execute(
        f"""
        SELECT r.*, s.name AS subject_name, o.name AS object_name
        FROM relationships AS r
        JOIN entities AS s ON s.id = r.subject_id
        JOIN entities AS o ON o.id = r.object_id
        {where_sql}
        ORDER BY COALESCE(r.valid_from, r.created_at) ASC, r.created_at ASC
        LIMIT ?
        """,
        [*params, max(0, int(limit))],
    ).fetchall()
    return [_relationship_from_row(row, direction="timeline") for row in rows]


def timeline(conn: sqlite3.Connection, entity: str | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
    return [fact.to_dict() for fact in _timeline_facts(conn, entity, limit=limit)]


def memory_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    pinned_count = conn.execute("SELECT COUNT(*) FROM memories WHERE pinned = 1").fetchone()[0]
    entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    relationship_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
    current_relationships = conn.execute(
        "SELECT COUNT(*) FROM relationships WHERE valid_to IS NULL"
    ).fetchone()[0]
    scopes = [
        {"scope": row["scope"], "count": row["count"]}
        for row in conn.execute(
            "SELECT scope, COUNT(*) AS count FROM memories GROUP BY scope ORDER BY count DESC, scope"
        ).fetchall()
    ]
    rooms = [
        {"room": row["room"], "count": row["count"]}
        for row in conn.execute(
            "SELECT room, COUNT(*) AS count FROM memories GROUP BY room ORDER BY count DESC, room LIMIT 20"
        ).fetchall()
    ]
    relationship_types = [
        row["predicate"]
        for row in conn.execute("SELECT DISTINCT predicate FROM relationships ORDER BY predicate").fetchall()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "memories": memory_count,
        "pinned_memories": pinned_count,
        "entities": entity_count,
        "relationships": relationship_count,
        "current_relationships": current_relationships,
        "expired_relationships": relationship_count - current_relationships,
        "scopes": scopes,
        "rooms": rooms,
        "relationship_types": relationship_types,
        "fts_enabled": has_fts(conn),
    }


def default_backup_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_BACKUP_DIR / f"memory-{timestamp}.db"


def ingest_paths(
    conn: sqlite3.Connection,
    paths: Iterable[Path | str],
    *,
    workspace_root: Path | str = WORKSPACE_ROOT,
    kind: str = "artifact",
    scope: str = DEFAULT_SCOPE,
    room: str = DEFAULT_ROOM,
    tags: Iterable[str] = (),
    include_code: bool = False,
    max_file_bytes: int = 256_000,
    max_detail_chars: int = 4_000,
    importance: float = 2.5,
    confidence: float = 0.8,
    allow_duplicate: bool = False,
) -> dict[str, Any]:
    documents = collect_documents(
        paths,
        workspace_root=workspace_root,
        include_code=include_code,
        max_file_bytes=max_file_bytes,
    )
    memories: list[dict[str, Any]] = []
    skipped_secret_documents = 0
    for document in documents:
        if find_secret_indicators({"path": document.relative_path, "text": document.text}, path="ingest document"):
            skipped_secret_documents += 1
            continue
        kwargs = document.to_memory_kwargs(
            kind=kind,
            scope=scope,
            room=room,
            tags=normalize_tags(tags),
            importance=importance,
            confidence=confidence,
            max_detail_chars=max_detail_chars,
        )
        memory_id = add_memory(
            conn,
            **kwargs,
            allow_duplicate=allow_duplicate,
            auto_extract_entities=True,
        )
        memory = get_memory(conn, memory_id)
        if memory is not None:
            memories.append(memory)
    return {
        "documents": len(documents),
        "memories": memories,
        "skipped_secret_documents": skipped_secret_documents,
    }


def evaluate_recall_file(conn: sqlite3.Connection, cases_path: Path | str) -> dict[str, Any]:
    cases = load_recall_cases(cases_path)
    return evaluate_recall(
        lambda query, **kwargs: _search_memory_results(conn, query, **kwargs),
        cases,
    )


class MemPalace:
    """High-level standalone memory facade for other modules."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        *,
        c4_overview_path: Path | str = DEFAULT_C4_OVERVIEW_PATH,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.c4_overview_path = resolve_c4_overview_path(c4_overview_path)
        self._owns_conn = conn is None
        self.conn = conn if conn is not None else connect(self.db_path)
        if conn is not None:
            initialize(self.conn)

    def close(self) -> None:
        if self._owns_conn:
            self.conn.close()

    def __enter__(self) -> "MemPalace":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def remember(self, **kwargs: Any) -> MemoryRecord:
        memory_id = add_memory(self.conn, **kwargs)
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", [memory_id]).fetchone()
        if row is None:
            raise RuntimeError(f"stored memory #{memory_id} could not be reloaded")
        return row_to_record(row)

    def get(self, memory_id: int) -> MemoryRecord | None:
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", [int(memory_id)]).fetchone()
        return row_to_record(row) if row is not None else None

    def recent(self, **kwargs: Any) -> list[MemoryRecord]:
        return _recent_records(self.conn, **kwargs)

    def essential(self, **kwargs: Any) -> list[MemoryRecord]:
        return _essential_records(self.conn, **kwargs)

    def recall(self, query: str, **kwargs: Any) -> list[SearchResult]:
        return search_memory_results(self.conn, query, **kwargs)

    def add_fact(self, subject: str, predicate: str, obj: str, **kwargs: Any) -> str:
        return add_relationship(self.conn, subject, predicate, obj, **kwargs)

    def invalidate_fact(self, subject: str, predicate: str, obj: str, **kwargs: Any) -> int:
        return invalidate_relationship(self.conn, subject, predicate, obj, **kwargs)

    def entity_facts(self, name: str, **kwargs: Any) -> list[RelationshipFact]:
        return _entity_facts(self.conn, name, **kwargs)

    def timeline(self, entity: str | None = None, **kwargs: Any) -> list[RelationshipFact]:
        return _timeline_facts(self.conn, entity, **kwargs)

    def stats(self) -> dict[str, Any]:
        return memory_stats(self.conn)

    def backup(self, output_path: Path | str | None = None) -> dict[str, Any]:
        return backup_database(self.conn, output_path or default_backup_path())

    def export_jsonl(self, output_path: Path | str) -> dict[str, Any]:
        return export_jsonl(self.conn, output_path)

    def import_jsonl(self, input_path: Path | str, **kwargs: Any) -> dict[str, Any]:
        return import_jsonl(self.conn, input_path, **kwargs)

    def ingest_paths(self, paths: Iterable[Path | str], **kwargs: Any) -> dict[str, Any]:
        return ingest_paths(self.conn, paths, **kwargs)

    def evaluate_recall(self, cases_path: Path | str) -> dict[str, Any]:
        return evaluate_recall_file(self.conn, cases_path)

    def briefing(
        self,
        *,
        limit: int = 8,
        kind: str | None = None,
        tag: str | None = None,
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
        room: str | None = None,
        entity: str | None = None,
    ) -> str:
        return format_briefing(
            self.conn,
            c4_overview_path=self.c4_overview_path,
            limit=limit,
            kind=kind,
            tag=tag,
            scope=scope,
            scopes=scopes,
            room=room,
            entity=entity,
        )

def record_from_dict(data: Mapping[str, Any]) -> MemoryRecord:
    return row_to_record({
        **data,
        "id": data["id"],
        "kind": data["kind"],
        "summary": data["summary"],
        "metadata": serialize_json_object(data.get("metadata") or {}),
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
    })


def relationship_from_dict(data: Mapping[str, Any]) -> RelationshipFact:
    return RelationshipFact(
        id=str(data["id"]),
        direction=str(data.get("direction", "")),
        subject=str(data["subject"]),
        predicate=str(data["predicate"]),
        object=str(data["object"]),
        valid_from=data.get("valid_from"),
        valid_to=data.get("valid_to"),
        confidence=float(data.get("confidence", DEFAULT_CONFIDENCE)),
        source_memory_id=data.get("source_memory_id"),
        source=str(data.get("source") or ""),
        created_at=str(data.get("created_at") or ""),
        updated_at=str(data.get("updated_at") or ""),
    )


def format_memory(memory: Mapping[str, Any]) -> str:
    lines = [
        f"[{memory['id']}] {memory['created_at']}  {memory['kind']}"
        + ("  pinned" if memory.get("pinned") else "")
    ]
    lines.append(f"summary: {memory['summary']}")
    if memory.get("details"):
        lines.append(f"details: {memory['details']}")
    if memory.get("tags"):
        lines.append(f"tags: {memory['tags']}")
    if memory.get("scope") or memory.get("room"):
        lines.append(f"scope: {memory.get('scope', DEFAULT_SCOPE)} / {memory.get('room', DEFAULT_ROOM)}")
    if memory.get("entities"):
        lines.append(f"entities: {memory['entities']}")
    if memory.get("importance") is not None or memory.get("confidence") is not None:
        lines.append(
            f"weight: importance={float(memory.get('importance', DEFAULT_IMPORTANCE)):.1f}, "
            f"confidence={float(memory.get('confidence', DEFAULT_CONFIDENCE)):.2f}"
        )
    if memory.get("source"):
        lines.append(f"source: {memory['source']}")
    metadata = memory.get("metadata")
    if metadata:
        lines.append(f"metadata: {json.dumps(metadata, sort_keys=True)}")
    return "\n".join(lines)


def render_memories(memories: Sequence[Mapping[str, Any]]) -> str:
    if not memories:
        return "No memories found."
    return "\n\n".join(format_memory(memory) for memory in memories)


def print_memories(memories: Sequence[Mapping[str, Any]]) -> None:
    print(render_memories(memories))


def render_search_results(results: Sequence[Mapping[str, Any]]) -> str:
    if not results:
        return "No memories found."
    sections: list[str] = []
    for result in results:
        heading = f"Rank {result['rank']}  score={result['score']}"
        if result.get("matched_fields"):
            heading += f"  fields={', '.join(result['matched_fields'])}"
        sections.append(heading + "\n" + format_memory(result))
    return "\n\n".join(sections)


def render_relationships(facts: Sequence[Mapping[str, Any]]) -> str:
    if not facts:
        return "No facts found."
    lines: list[str] = []
    for fact in facts:
        status = "current" if fact.get("current") else f"ended {fact.get('valid_to')}"
        window = ""
        if fact.get("valid_from") or fact.get("valid_to"):
            window = f" [{fact.get('valid_from') or '?'} -> {fact.get('valid_to') or 'present'}]"
        lines.append(
            f"[{fact['id']}] {fact['subject']} --{fact['predicate']}--> {fact['object']}{window} "
            f"({status}, confidence={float(fact.get('confidence', DEFAULT_CONFIDENCE)):.2f})"
        )
        if fact.get("source_memory_id") or fact.get("source"):
            source_bits = []
            if fact.get("source_memory_id"):
                source_bits.append(f"memory #{fact['source_memory_id']}")
            if fact.get("source"):
                source_bits.append(str(fact["source"]))
            lines.append(f"source: {', '.join(source_bits)}")
    return "\n".join(lines)


def render_stats(stats: Mapping[str, Any]) -> str:
    lines = [
        f"schema_version: {stats['schema_version']}",
        f"memories: {stats['memories']} ({stats['pinned_memories']} pinned)",
        f"entities: {stats['entities']}",
        f"relationships: {stats['relationships']} ({stats['current_relationships']} current)",
        f"fts_enabled: {stats['fts_enabled']}",
    ]
    if stats.get("scopes"):
        lines.append("scopes: " + ", ".join(f"{item['scope']}={item['count']}" for item in stats["scopes"]))
    if stats.get("relationship_types"):
        lines.append("relationship_types: " + ", ".join(stats["relationship_types"]))
    return "\n".join(lines)


def render_ingest_report(report: Mapping[str, Any]) -> str:
    memories = list(report.get("memories", []))
    lines = [f"ingested_documents: {report.get('documents', 0)}", f"memory_rows: {len(memories)}"]
    for memory in memories:
        lines.append(f"[{memory['id']}] {memory['summary']}")
    return "\n".join(lines)


def format_briefing(
    conn: sqlite3.Connection,
    *,
    c4_overview_path: Path | str = DEFAULT_C4_OVERVIEW_PATH,
    limit: int = 8,
    kind: str | None = None,
    tag: str | None = None,
    scope: str | None = None,
    scopes: Sequence[str] | None = None,
    room: str | None = None,
    entity: str | None = None,
) -> str:
    overview_path = resolve_c4_overview_path(c4_overview_path)
    c4_overview = read_c4_overview(overview_path)
    essential = essential_memories(
        conn,
        limit=max(1, min(int(limit), 5)),
        scope=scope,
        scopes=scopes,
        room=room,
    )
    memories = recent_memories(
        conn,
        limit=limit,
        kind=kind,
        tag=tag,
        scope=scope,
        scopes=scopes,
        room=room,
        entity=entity,
    )
    mark_memories_used(conn, [memory["id"] for memory in [*essential, *memories]])

    sections = [
        f"== Repository Guidelines ({overview_path}) ==",
        c4_overview,
        "== Essential Memory ==",
        render_memories(essential),
        "== Recent Memory ==",
        render_memories(memories),
    ]
    return "\n\n".join(sections)


def _add_scope_arguments(
    command_parser: argparse.ArgumentParser,
    *,
    write_default: bool = False,
    allow_includes: bool = False,
) -> None:
    selection = command_parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--scope",
        default=DEFAULT_SCOPE if write_default else None,
        help="Explicit broad retrieval scope",
    )
    selection.add_argument(
        "--project-root",
        default=None,
        help="Adopting project directory used to derive a stable local scope",
    )
    if allow_includes:
        command_parser.add_argument(
            "--include-scope",
            action="append",
            default=[],
            help="Additional exact retrieval scope; may repeat",
        )


def _project_scope_from_args(args: argparse.Namespace) -> ProjectMemoryScope | None:
    if hasattr(args, "_resolved_project_scope"):
        return args._resolved_project_scope
    project_root = getattr(args, "project_root", None)
    if not project_root:
        args._resolved_project_scope = None
        return None
    project_scope = resolve_project_memory_scope(project_root, tokimon_root=WORKSPACE_ROOT)
    args._resolved_project_scope = project_scope
    return project_scope


def _scope_selection_from_args(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...] | None, ProjectMemoryScope | None]:
    project_scope = _project_scope_from_args(args)
    values = [
        project_scope.scope if project_scope is not None else getattr(args, "scope", None),
        *getattr(args, "include_scope", []),
    ]
    selected: list[str] = []
    for value in values:
        normalized = (value or "").strip().lower()
        if normalized and normalized not in selected:
            selected.append(normalized)
    return (tuple(selected) if selected else None), project_scope


def _write_scope_from_args(args: argparse.Namespace) -> str:
    project_scope = _project_scope_from_args(args)
    if project_scope is not None:
        return project_scope.scope
    return str(getattr(args, "scope", None) or DEFAULT_SCOPE).strip().lower()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mem Palace standalone local memory for Tokimon.",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--c4-overview-path",
        default=str(DEFAULT_C4_OVERVIEW_PATH),
        help=argparse.SUPPRESS,
    )

    subparsers = parser.add_subparsers(dest="command")

    add_parser = subparsers.add_parser("add", help="Store a durable memory")
    add_parser.add_argument("--kind", required=True, choices=ALLOWED_KINDS)
    add_parser.add_argument("--summary", required=True, help="Short durable summary")
    add_parser.add_argument("--details", default="", help="Optional extra detail")
    add_parser.add_argument("--tags", default="", help="Comma-separated tags")
    add_parser.add_argument("--source", default="", help="Source or context")
    _add_scope_arguments(add_parser, write_default=True)
    add_parser.add_argument("--room", default=DEFAULT_ROOM, help="Topic room within the scope")
    add_parser.add_argument("--entities", default="", help="Comma-separated explicit entities")
    add_parser.add_argument("--importance", type=float, default=DEFAULT_IMPORTANCE, help="0-5 retrieval weight")
    add_parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="0-1 confidence")
    add_parser.add_argument("--metadata-json", default="", help="Optional JSON object metadata")
    add_parser.add_argument("--pinned", action="store_true", help="Pin important memories")
    add_parser.add_argument("--allow-duplicate", action="store_true", help="Store even if the content hash exists")
    add_parser.add_argument("--no-auto-entities", action="store_true", help="Disable local entity hint extraction")

    search_parser = subparsers.add_parser("search", help="Search saved memories")
    search_parser.add_argument("query", help="Keywords to search for")
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--kind", choices=ALLOWED_KINDS)
    search_parser.add_argument("--tag", default=None)
    _add_scope_arguments(search_parser, allow_includes=True)
    search_parser.add_argument("--room", default=None)
    search_parser.add_argument("--entity", default=None)
    search_parser.add_argument("--json", action="store_true")

    recent_parser = subparsers.add_parser("recent", help="Show recent memories")
    recent_parser.add_argument("--limit", type=int, default=8)
    recent_parser.add_argument("--kind", choices=ALLOWED_KINDS)
    recent_parser.add_argument("--tag", default=None)
    _add_scope_arguments(recent_parser, allow_includes=True)
    recent_parser.add_argument("--room", default=None)
    recent_parser.add_argument("--entity", default=None)
    recent_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser("show", help="Show one memory by id")
    show_parser.add_argument("id", type=int)
    show_parser.add_argument("--json", action="store_true")

    briefing_parser = subparsers.add_parser("briefing", help="Show repository guidelines plus durable memory")
    briefing_parser.add_argument("--limit", type=int, default=8)
    briefing_parser.add_argument("--kind", choices=ALLOWED_KINDS)
    briefing_parser.add_argument("--tag", default=None)
    _add_scope_arguments(briefing_parser, allow_includes=True)
    briefing_parser.add_argument("--room", default=None)
    briefing_parser.add_argument("--entity", default=None)
    briefing_parser.add_argument("--json", action="store_true")

    link_parser = subparsers.add_parser("link", help="Store a temporal entity relationship")
    link_parser.add_argument("subject")
    link_parser.add_argument("predicate")
    link_parser.add_argument("object")
    link_parser.add_argument("--valid-from", default=None)
    link_parser.add_argument("--valid-to", default=None)
    link_parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    link_parser.add_argument("--source-memory-id", type=int, default=None)
    link_parser.add_argument("--source", default="")
    link_parser.add_argument("--json", action="store_true")

    entity_parser = subparsers.add_parser("entity", help="Query temporal facts for an entity")
    entity_parser.add_argument("name")
    entity_parser.add_argument("--as-of", default=None)
    entity_parser.add_argument("--direction", choices=RELATION_DIRECTIONS, default="both")
    entity_parser.add_argument("--limit", type=int, default=50)
    entity_parser.add_argument("--json", action="store_true")

    invalidate_parser = subparsers.add_parser("invalidate", help="End a current entity relationship")
    invalidate_parser.add_argument("subject")
    invalidate_parser.add_argument("predicate")
    invalidate_parser.add_argument("object")
    invalidate_parser.add_argument("--ended", default=None)
    invalidate_parser.add_argument("--json", action="store_true")

    timeline_parser = subparsers.add_parser("timeline", help="Show temporal relationship history")
    timeline_parser.add_argument("entity", nargs="?")
    timeline_parser.add_argument("--limit", type=int, default=100)
    timeline_parser.add_argument("--json", action="store_true")

    stats_parser = subparsers.add_parser("stats", help="Show memory store stats")
    stats_parser.add_argument("--json", action="store_true")

    backup_parser = subparsers.add_parser("backup", help="Write a SQLite backup of the memory store")
    backup_parser.add_argument("--output", help="Backup database path")
    backup_parser.add_argument("--json", action="store_true")

    export_parser = subparsers.add_parser("export", help="Export memory store rows as JSONL")
    export_parser.add_argument("--output", required=True, help="Destination JSONL path")
    export_parser.add_argument("--json", action="store_true")

    import_parser = subparsers.add_parser("import", help="Import a Mem Palace JSONL export")
    import_parser.add_argument("path", help="Source JSONL path")
    import_parser.add_argument("--replace", action="store_true", help="Replace rows with matching primary keys")
    import_parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Import a legacy JSONL export without sidecar manifest validation",
    )
    import_parser.add_argument(
        "--keep-duplicate-memories",
        action="store_true",
        help="Import memories even when an existing content hash matches",
    )
    import_parser.add_argument("--json", action="store_true")

    ingest_parser = subparsers.add_parser("ingest", help="Conservatively ingest explicit text files or directories")
    ingest_parser.add_argument("paths", nargs="+")
    ingest_parser.add_argument("--kind", choices=ALLOWED_KINDS, default="artifact")
    _add_scope_arguments(ingest_parser, write_default=True)
    ingest_parser.add_argument("--room", default=DEFAULT_ROOM)
    ingest_parser.add_argument("--tags", default="")
    ingest_parser.add_argument("--include-code", action="store_true")
    ingest_parser.add_argument("--max-file-bytes", type=int, default=256_000)
    ingest_parser.add_argument("--max-detail-chars", type=int, default=4_000)
    ingest_parser.add_argument("--importance", type=float, default=2.5)
    ingest_parser.add_argument("--confidence", type=float, default=0.8)
    ingest_parser.add_argument("--allow-duplicate", action="store_true")
    ingest_parser.add_argument("--json", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate recall against JSON/JSONL fixture cases")
    eval_parser.add_argument("cases_path")
    eval_parser.add_argument("--json", action="store_true")

    migrate_parser = subparsers.add_parser("migrate", help="Run schema migrations")
    migrate_parser.add_argument("--json", action="store_true")

    return parser


def _open_palace_from_args(args: argparse.Namespace) -> MemPalace:
    return MemPalace(Path(args.db), c4_overview_path=resolve_c4_overview_path(args.c4_overview_path))


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2))


def _print_json_or_text(args: argparse.Namespace, payload: Any, text: str) -> None:
    if args.json:
        _print_json(payload)
    else:
        print(text)


def _handle_add(palace: MemPalace, args: argparse.Namespace) -> int:
    memory = palace.remember(
        kind=args.kind,
        summary=args.summary,
        details=args.details,
        tags=normalize_tags([args.tags]),
        source=args.source,
        pinned=args.pinned,
        scope=_write_scope_from_args(args),
        room=args.room,
        entities=normalize_entities([args.entities]),
        importance=args.importance,
        confidence=args.confidence,
        metadata=parse_json_object(args.metadata_json),
        allow_duplicate=args.allow_duplicate,
        auto_extract_entities=not args.no_auto_entities,
    )
    print(f"Stored memory #{memory.id}")
    return 0


def _handle_search(palace: MemPalace, args: argparse.Namespace) -> int:
    scopes, _ = _scope_selection_from_args(args)
    results = [
        result.to_dict()
        for result in palace.recall(
            args.query,
            limit=args.limit,
            kind=args.kind,
            tag=args.tag,
            scopes=scopes,
            room=args.room,
            entity=args.entity,
        )
    ]
    _print_json_or_text(args, results, render_search_results(results))
    return 0


def _handle_recent(palace: MemPalace, args: argparse.Namespace) -> int:
    scopes, _ = _scope_selection_from_args(args)
    memories = recent_memories(
        palace.conn,
        limit=args.limit,
        kind=args.kind,
        tag=args.tag,
        scopes=scopes,
        room=args.room,
        entity=args.entity,
    )
    _print_json_or_text(args, memories, render_memories(memories))
    return 0


def _handle_show(palace: MemPalace, args: argparse.Namespace) -> int:
    memory = get_memory(palace.conn, args.id)
    if memory is None:
        print(f"Memory #{args.id} not found.", file=sys.stderr)
        return 2
    _print_json_or_text(args, memory, format_memory(memory))
    return 0


def _briefing_payload(palace: MemPalace, args: argparse.Namespace) -> dict[str, Any]:
    scopes, project_scope = _scope_selection_from_args(args)
    memories = recent_memories(
        palace.conn,
        limit=args.limit,
        kind=args.kind,
        tag=args.tag,
        scopes=scopes,
        room=args.room,
        entity=args.entity,
    )
    essential = essential_memories(
        palace.conn,
        limit=max(1, min(int(args.limit), 5)),
        scopes=scopes,
        room=args.room,
    )
    mark_memories_used(palace.conn, [memory["id"] for memory in [*essential, *memories]])
    return {
        "c4_overview_path": str(palace.c4_overview_path),
        "c4_overview": read_c4_overview(palace.c4_overview_path),
        "memory_scope_selection": {
            "scopes": list(scopes or ()),
            "project": project_scope.to_dict() if project_scope is not None else None,
        },
        "essential_memories": essential,
        "memories": memories,
        "stats": palace.stats(),
    }


def _handle_briefing(palace: MemPalace, args: argparse.Namespace) -> int:
    scopes, _ = _scope_selection_from_args(args)
    if args.json:
        _print_json(_briefing_payload(palace, args))
    else:
        print(
            format_briefing(
                palace.conn,
                c4_overview_path=palace.c4_overview_path,
                limit=args.limit,
                kind=args.kind,
                tag=args.tag,
                scopes=scopes,
                room=args.room,
                entity=args.entity,
            )
        )
    return 0


def _handle_link(palace: MemPalace, args: argparse.Namespace) -> int:
    relationship_id = palace.add_fact(
        args.subject,
        args.predicate,
        args.object,
        valid_from=args.valid_from,
        valid_to=args.valid_to,
        confidence=args.confidence,
        source_memory_id=args.source_memory_id,
        source=args.source,
    )
    _print_json_or_text(args, {"id": relationship_id}, f"Stored relationship {relationship_id}")
    return 0


def _handle_entity(palace: MemPalace, args: argparse.Namespace) -> int:
    facts = query_entity(
        palace.conn,
        args.name,
        as_of=args.as_of,
        direction=args.direction,
        limit=args.limit,
    )
    _print_json_or_text(args, facts, render_relationships(facts))
    return 0


def _handle_invalidate(palace: MemPalace, args: argparse.Namespace) -> int:
    count = palace.invalidate_fact(args.subject, args.predicate, args.object, ended=args.ended)
    _print_json_or_text(args, {"relationships_invalidated": count}, f"Invalidated {count} relationship(s)")
    return 0


def _handle_timeline(palace: MemPalace, args: argparse.Namespace) -> int:
    facts = timeline(palace.conn, args.entity, limit=args.limit)
    _print_json_or_text(args, facts, render_relationships(facts))
    return 0


def _handle_stats(palace: MemPalace, args: argparse.Namespace) -> int:
    stats = palace.stats()
    _print_json_or_text(args, stats, render_stats(stats))
    return 0


def _handle_backup(palace: MemPalace, args: argparse.Namespace) -> int:
    report = palace.backup(args.output or None)
    _print_json_or_text(args, report, f"Backed up memory store to {report['path']}")
    return 0


def _handle_export(palace: MemPalace, args: argparse.Namespace) -> int:
    report = palace.export_jsonl(args.output)
    text = f"Exported memory store to {report['path']}\n{json.dumps(report['counts'], sort_keys=True)}"
    _print_json_or_text(args, report, text)
    return 0


def _handle_import(palace: MemPalace, args: argparse.Namespace) -> int:
    report = palace.import_jsonl(
        args.path,
        replace=args.replace,
        skip_duplicate_memories=not args.keep_duplicate_memories,
        require_manifest=not args.no_manifest,
    )
    text = (
        f"Imported memory export from {report['path']}\n"
        f"{json.dumps({'counts': report['counts'], 'skipped': report['skipped']}, sort_keys=True)}"
    )
    _print_json_or_text(args, report, text)
    return 0


def _handle_ingest(palace: MemPalace, args: argparse.Namespace) -> int:
    report = palace.ingest_paths(
        args.paths,
        workspace_root=WORKSPACE_ROOT,
        kind=args.kind,
        scope=_write_scope_from_args(args),
        room=args.room,
        tags=normalize_tags([args.tags]),
        include_code=args.include_code,
        max_file_bytes=args.max_file_bytes,
        max_detail_chars=args.max_detail_chars,
        importance=args.importance,
        confidence=args.confidence,
        allow_duplicate=args.allow_duplicate,
    )
    _print_json_or_text(args, report, render_ingest_report(report))
    return 0


def _handle_eval(palace: MemPalace, args: argparse.Namespace) -> int:
    report = palace.evaluate_recall(args.cases_path)
    _print_json_or_text(args, report, render_recall_evaluation(report))
    return 0


def _handle_migrate(palace: MemPalace, args: argparse.Namespace) -> int:
    stats = palace.stats()
    _print_json_or_text(args, stats, "Migrations complete.\n" + render_stats(stats))
    return 0


PALACE_COMMAND_HANDLERS = {
    "add": _handle_add,
    "search": _handle_search,
    "recent": _handle_recent,
    "show": _handle_show,
    "briefing": _handle_briefing,
    "link": _handle_link,
    "entity": _handle_entity,
    "invalidate": _handle_invalidate,
    "timeline": _handle_timeline,
    "stats": _handle_stats,
    "backup": _handle_backup,
    "export": _handle_export,
    "import": _handle_import,
    "ingest": _handle_ingest,
    "eval": _handle_eval,
    "migrate": _handle_migrate,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    handler = PALACE_COMMAND_HANDLERS.get(args.command)
    if handler is None:
        return 1

    try:
        _project_scope_from_args(args)
        with _open_palace_from_args(args) as palace:
            return handler(palace, args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except sqlite3.OperationalError as exc:
        db_path = Path(args.db).expanduser()
        print(
            f"Mem Palace command {args.command!r} failed for database {str(db_path)!r}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
