"""Memory store for Lessons and artifact index."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_CHARTER_REQUIRED_FIELDS = (
    "failure_signature",
    "root_cause_hypothesis",
    "strategy_change",
    "evidence_of_novelty",
    "retrieval_tags",
)

_SECRET_METADATA_KEY_PATTERN = re.compile(r"(api[_-]?key|secret|token|password)", re.IGNORECASE)
_BEARER_TOKEN_PATTERN = re.compile(r"(Authorization\s*:\s*Bearer\s+)(\S+)", re.IGNORECASE)


def _is_empty_field(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _validate_lesson_charter(metadata: dict[str, Any]) -> None:
    lesson_type = metadata.get("lesson_type")
    if lesson_type not in {"failure", "retry"}:
        return

    missing = [field for field in _CHARTER_REQUIRED_FIELDS if field not in metadata or _is_empty_field(metadata.get(field))]
    if missing:
        raise ValueError(f"Lesson metadata missing required charter fields: {', '.join(missing)}")


def _deny_secret_metadata(metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        if not _SECRET_METADATA_KEY_PATTERN.search(key):
            continue
        if _is_empty_field(value):
            continue
        raise ValueError(f"Secret metadata is not allowed in Lesson: {key}")


def _redact_secrets_in_body(body: str) -> str:
    return _BEARER_TOKEN_PATTERN.sub(r"\1<REDACTED>", body)


def _require_retrieval_context(*, component: str | None, tags: list[str] | None, failure_signature: str | None) -> None:
    if not isinstance(component, str) or not component.strip():
        raise ValueError("Memory retrieval requires non-empty component")
    if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and tag.strip() for tag in tags):
        raise ValueError("Memory retrieval requires non-empty retrieval tags")
    if not isinstance(failure_signature, str) or not failure_signature.strip():
        raise ValueError("Memory retrieval requires non-empty failure_signature")


@dataclass
class Lesson:
    metadata: dict[str, Any]
    body: str
    path: Path


class MemoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lessons_dir = root / "lessons"
        self.artifacts_dir = root / "artifacts"
        self.index_path = root / "index.sqlite"
        self.lessons_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.index_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS lessons (id TEXT PRIMARY KEY, metadata TEXT, body TEXT, tags TEXT, component TEXT, failure_signature TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, step_id TEXT, task_id TEXT, path TEXT, hash TEXT, metadata TEXT)"
            )
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5(id, body, tags, component, failure_signature)"
                )
            except sqlite3.OperationalError:
                # FTS5 not available; fallback to table-only search.
                pass
            conn.commit()
        finally:
            conn.close()

    def write_lesson(self, metadata: dict[str, Any], body: str) -> Lesson:
        lesson_id = metadata.get("id") or metadata.get("lesson_id")
        if not lesson_id:
            raise ValueError("Lesson metadata must include an 'id'")
        _validate_lesson_charter(metadata)
        _deny_secret_metadata(metadata)
        body = _redact_secrets_in_body(body)
        path = self.lessons_dir / f"lesson-{lesson_id}.md"
        header = json.dumps(metadata, sort_keys=True)
        content = f"{header}\n---\n{body}\n"
        path.write_text(content)
        self._index_lesson(lesson_id, metadata, body)
        return Lesson(metadata=metadata, body=body, path=path)

    def _index_lesson(self, lesson_id: str, metadata: dict[str, Any], body: str) -> None:
        tag_values: list[str] = []
        for field in ("tags", "retrieval_tags"):
            value = metadata.get(field)
            if isinstance(value, list):
                tag_values.extend([str(item).strip() for item in value if str(item).strip()])
            elif isinstance(value, str) and value.strip():
                tag_values.append(value.strip())
        tags = ",".join(tag_values)
        component = metadata.get("component", "")
        failure_signature = metadata.get("failure_signature", "")
        conn = sqlite3.connect(self.index_path)
        try:
            conn.execute(
                "REPLACE INTO lessons (id, metadata, body, tags, component, failure_signature) VALUES (?, ?, ?, ?, ?, ?)",
                (lesson_id, json.dumps(metadata), body, tags, component, failure_signature),
            )
            try:
                conn.execute(
                    "REPLACE INTO lessons_fts (id, body, tags, component, failure_signature) VALUES (?, ?, ?, ?, ?)",
                    (lesson_id, body, tags, component, failure_signature),
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()
        finally:
            conn.close()

    def load_lesson(self, lesson_id: str) -> Lesson:
        path = self.lessons_dir / f"lesson-{lesson_id}.md"
        content = path.read_text()
        header, body = content.split("---", 1)
        metadata = json.loads(header.strip())
        return Lesson(metadata=metadata, body=body.strip(), path=path)

    def index_artifact(self, artifact_id: str, step_id: str, task_id: str, path: Path, digest: str, metadata: dict[str, Any]) -> None:
        conn = sqlite3.connect(self.index_path)
        try:
            conn.execute(
                "REPLACE INTO artifacts (id, step_id, task_id, path, hash, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (artifact_id, step_id, task_id, str(path), digest, json.dumps(metadata)),
            )
            conn.commit()
        finally:
            conn.close()

    def list_artifacts(self, task_id: str | None = None, step_id: str | None = None) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.index_path)
        try:
            clauses = []
            params: list[Any] = []
            if task_id:
                clauses.append("task_id = ?")
                params.append(task_id)
            if step_id:
                clauses.append("step_id = ?")
                params.append(step_id)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            cursor = conn.execute(f"SELECT id, step_id, task_id, path, hash, metadata FROM artifacts {where}", params)
            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "id": row[0],
                        "step_id": row[1],
                        "task_id": row[2],
                        "path": row[3],
                        "hash": row[4],
                        "metadata": json.loads(row[5]) if row[5] else {},
                    }
                )
            return results
        finally:
            conn.close()

    def retrieve(
        self,
        query: str,
        stage: int,
        limit: int = 5,
        *,
        tags: list[str] | None,
        failure_signature: str | None,
        component: str | None,
    ) -> list[Lesson]:
        _require_retrieval_context(component=component, tags=tags, failure_signature=failure_signature)
        tags = tags or []
        conn = sqlite3.connect(self.index_path)
        try:
            if stage == 1:
                rows = _select_lessons(
                    conn,
                    query=query,
                    tags=tags,
                    components=[component] if component else None,
                    failure_signature=failure_signature,
                    limit=limit,
                )
            elif stage == 2:
                stage2_components: list[str] = []
                if component:
                    stage2_components.append(component)
                    stage2_components.extend(
                        _adjacent_components(conn, component=component, current_failure_signature=failure_signature)
                    )
                rows = _select_lessons(
                    conn,
                    query=None,
                    tags=tags,
                    components=stage2_components or None,
                    failure_signature=failure_signature,
                    limit=limit,
                )
            else:
                stage2_components = []
                if component:
                    stage2_components.append(component)
                    stage2_components.extend(
                        _adjacent_components(conn, component=component, current_failure_signature=failure_signature)
                    )
                rows = _select_lessons(
                    conn,
                    query=None,
                    tags=tags,
                    components=stage2_components or None,
                    failure_signature=failure_signature,
                    limit=limit,
                )
                if len(rows) < limit and failure_signature:
                    rows += _select_lessons(
                        conn,
                        query=None,
                        tags=None,
                        components=None,
                        failure_signature=failure_signature,
                        limit=limit - len(rows),
                    )
                if len(rows) < limit and failure_signature:
                    similar = _similar_failure_signatures(conn, failure_signature=failure_signature, limit=20)
                    for candidate in similar:
                        if len(rows) >= limit:
                            break
                        rows += _select_lessons(
                            conn,
                            query=None,
                            tags=None,
                            components=None,
                            failure_signature=candidate,
                            limit=limit - len(rows),
                        )
            lessons = []
            seen = set()
            for row in rows:
                lesson_id = row[0]
                if lesson_id in seen:
                    continue
                seen.add(lesson_id)
                lessons.append(self.load_lesson(lesson_id))
            return lessons
        finally:
            conn.close()


def _adjacent_components(
    conn: sqlite3.Connection, *, component: str, current_failure_signature: str | None
) -> list[str]:
    if not current_failure_signature:
        return []
    cursor = conn.execute(
        "SELECT DISTINCT failure_signature FROM lessons WHERE component = ? AND failure_signature <> ?",
        (component, current_failure_signature),
    )
    other_failure_signatures = [row[0] for row in cursor.fetchall() if row[0]]
    if not other_failure_signatures:
        return []

    placeholders = ",".join("?" for _ in other_failure_signatures)
    cursor = conn.execute(
        f"SELECT DISTINCT component FROM lessons WHERE failure_signature IN ({placeholders}) AND component <> ?",
        [*other_failure_signatures, component],
    )
    return [row[0] for row in cursor.fetchall() if row[0]]


def _similar_failure_signatures(conn: sqlite3.Connection, *, failure_signature: str, limit: int) -> list[str]:
    normalized = str(failure_signature).strip()
    if not normalized:
        return []
    family = normalized.split(":", 1)[0].strip()
    if not family:
        return []
    cursor = conn.execute(
        "SELECT DISTINCT failure_signature FROM lessons WHERE failure_signature LIKE ? AND failure_signature <> ? LIMIT ?",
        (family + ":%", normalized, limit),
    )
    return [row[0] for row in cursor.fetchall() if row[0]]


def _select_lessons(
    conn: sqlite3.Connection,
    *,
    query: str | None,
    tags: list[str] | None,
    components: list[str] | None,
    failure_signature: str | None,
    limit: int,
) -> list[tuple]:
    clauses: list[str] = []
    params: list[Any] = []

    if components is not None:
        if not components:
            return []
        placeholders = ",".join("?" for _ in components)
        clauses.append(f"component IN ({placeholders})")
        params.extend(components)

    if failure_signature:
        clauses.append("failure_signature = ?")
        params.append(failure_signature)

    if tags:
        for tag in tags:
            clauses.append("tags LIKE ?")
            params.append(f"%{tag}%")

    if query:
        clauses.append("body LIKE ?")
        params.append(f"%{query}%")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    cursor = conn.execute(f"SELECT id FROM lessons {where} LIMIT ?", [*params, limit])
    return cursor.fetchall()


def _search(conn: sqlite3.Connection, query: str, tags: list[str], component: str | None,
            failure_signature: str | None, limit: int) -> list[tuple]:
    params: list[Any] = []
    clauses: list[str] = []
    if component:
        clauses.append("component = ?")
        params.append(component)
    if failure_signature:
        clauses.append("failure_signature = ?")
        params.append(failure_signature)
    if tags:
        clauses.append("tags LIKE ?")
        params.append("%" + "%".join(tags) + "%")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    try:
        cursor = conn.execute(
            f"SELECT id FROM lessons_fts WHERE lessons_fts MATCH ? {where} LIMIT ?",
            [query, *params, limit],
        )
        return cursor.fetchall()
    except sqlite3.OperationalError:
        cursor = conn.execute(
            f"SELECT id FROM lessons {where} LIMIT ?",
            [*params, limit],
        )
        return cursor.fetchall()
