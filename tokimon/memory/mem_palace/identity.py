"""Stable content identity and semantic duplicate handling for memories."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


DEFAULT_SCOPE = "tokimon"
DEFAULT_ROOM = "general"
DEFAULT_IMPORTANCE = 3.0
DEFAULT_CONFIDENCE = 1.0
SUPERSEDES_MEMORY_IDS_FIELD = "supersedes_memory_ids"
SEMANTIC_ATTRIBUTE_FIELDS = (
    "pinned",
    "entities",
    "importance",
    "confidence",
    "metadata",
)


class MemoryDuplicateConflictError(ValueError):
    """A same-content memory would otherwise lose requested attributes."""

    def __init__(self, existing_ids: Iterable[int], conflicting_fields: Iterable[str]) -> None:
        self.existing_ids = tuple(sorted({int(memory_id) for memory_id in existing_ids}))
        self.conflicting_fields = tuple(sorted(set(conflicting_fields)))
        ids = ", ".join(str(memory_id) for memory_id in self.existing_ids)
        fields = ", ".join(self.conflicting_fields)
        super().__init__(
            f"same-content memory conflicts with existing memory ids [{ids}] "
            f"in semantic attributes: {fields}"
        )


def normalize_tags(tags: Iterable[str]) -> list[str]:
    """Normalize tags exactly as the established content-hash contract requires."""

    normalized: list[str] = []
    for raw in tags:
        for part in str(raw).split(","):
            tag = part.strip().lower()
            if tag and tag not in normalized:
                normalized.append(tag)
    return normalized


def memory_content_hash(
    *,
    kind: str,
    summary: str,
    details: str,
    tags: Sequence[str],
    source: str,
    scope: str,
    room: str,
) -> str:
    """Return the existing stable content identity for one memory."""

    payload = json.dumps(
        {
            "kind": kind,
            "summary": " ".join(summary.split()),
            "details": " ".join(details.split()),
            "tags": list(tags),
            "source": source.strip(),
            "scope": scope.strip().lower(),
            "room": room.strip().lower(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_value(data: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    try:
        value = data[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if value is None else value


def _field_values(value: Any, *, field: str) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (bytes, bytearray, Mapping)):
        raise ValueError(f"{field} must be a string or sequence of strings")
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError(f"{field} must be a string or sequence of strings") from exc


def normalize_semantic_entities(value: Any) -> tuple[str, ...]:
    """Normalize entity comparison values without making entity order semantic."""

    return tuple(sorted(_normalize_entity_sequence(value)))


def _normalize_entity_sequence(value: Any) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in _field_values(value, field="entities"):
        for part in str(raw).split(","):
            entity = " ".join(part.strip().split())
            if entity and entity not in normalized:
                normalized.append(entity)
    return tuple(normalized)


def canonical_metadata(value: Any) -> str:
    """Return canonical JSON-object text used for semantic equality."""

    if value is None or value == "":
        parsed: Any = {}
    elif isinstance(value, Mapping):
        parsed = dict(value)
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("metadata must be a JSON object") from exc
    else:
        raise ValueError("metadata must be a JSON object")
    if not isinstance(parsed, dict):
        raise ValueError("metadata must be a JSON object")
    try:
        return json.dumps(
            parsed,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be a JSON object with JSON-compatible values") from exc


def supersession_memory_ids(metadata: Any) -> tuple[int, ...]:
    """Return normalized positive integer ids from reserved supersession metadata."""

    try:
        parsed = json.loads(canonical_metadata(metadata))
    except ValueError:
        return ()
    raw_ids = parsed.get(SUPERSEDES_MEMORY_IDS_FIELD)
    if not isinstance(raw_ids, list):
        return ()
    return tuple(
        sorted(
            {
                memory_id
                for memory_id in raw_ids
                if isinstance(memory_id, int)
                and not isinstance(memory_id, bool)
                and memory_id > 0
            }
        )
    )


def remap_supersession_metadata(
    metadata: Any,
    memory_id_map: Mapping[int, int],
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Normalize the reserved field and remap only ids already known to exist."""

    parsed = json.loads(canonical_metadata(metadata))
    if SUPERSEDES_MEMORY_IDS_FIELD not in parsed:
        return parsed, ()

    remapped_ids = tuple(
        sorted(
            {
                int(memory_id_map[memory_id])
                for memory_id in supersession_memory_ids(parsed)
                if memory_id in memory_id_map
            }
        )
    )
    if remapped_ids:
        parsed[SUPERSEDES_MEMORY_IDS_FIELD] = list(remapped_ids)
    else:
        parsed.pop(SUPERSEDES_MEMORY_IDS_FIELD, None)
    return parsed, remapped_ids


def canonical_pinned(value: Any) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false"}:
            return 0
        if normalized in {"1", "true"}:
            return 1
    raise ValueError("pinned must be a boolean")


def canonical_number(
    value: Any,
    *,
    field: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"{field} must be between {minimum:g} and {maximum:g}")
    return number


def memory_semantic_attributes(data: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Canonicalize fields intentionally excluded from ``content_hash``."""

    return {
        "pinned": canonical_pinned(_record_value(data, "pinned", False)),
        "entities": normalize_semantic_entities(_record_value(data, "entities", "")),
        "importance": canonical_number(
            _record_value(data, "importance", DEFAULT_IMPORTANCE),
            field="importance",
            default=DEFAULT_IMPORTANCE,
            minimum=0.0,
            maximum=5.0,
        ),
        "confidence": canonical_number(
            _record_value(data, "confidence", DEFAULT_CONFIDENCE),
            field="confidence",
            default=DEFAULT_CONFIDENCE,
            minimum=0.0,
            maximum=1.0,
        ),
        "metadata": canonical_metadata(_record_value(data, "metadata", "{}")),
    }


def memory_content_fields(data: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Normalize one record into the established content-identity inputs."""

    return {
        "kind": str(_record_value(data, "kind", "")),
        "summary": str(_record_value(data, "summary", "")).strip(),
        "details": str(_record_value(data, "details", "")).strip(),
        "tags": normalize_tags(_field_values(_record_value(data, "tags", ""), field="tags")),
        "source": str(_record_value(data, "source", "")).strip(),
        "scope": str(_record_value(data, "scope", DEFAULT_SCOPE) or DEFAULT_SCOPE).strip().lower(),
        "room": str(_record_value(data, "room", DEFAULT_ROOM) or DEFAULT_ROOM).strip().lower(),
    }


def canonicalize_import_memory(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and populate canonical identity and semantic fields for import."""

    row = dict(data)
    content = memory_content_fields(row)
    canonical_hash = memory_content_hash(**content)
    provided_hash = str(row.get("content_hash") or "").strip()
    if provided_hash and provided_hash != canonical_hash:
        raise ValueError("content_hash does not match canonical memory content")

    semantic = memory_semantic_attributes(row)
    row.update(content)
    row["tags"] = ", ".join(content["tags"])
    row["entities"] = ", ".join(_normalize_entity_sequence(row.get("entities", "")))
    row["importance"] = semantic["importance"]
    row["confidence"] = semantic["confidence"]
    row["metadata"] = semantic["metadata"]
    row["pinned"] = semantic["pinned"]
    row["content_hash"] = canonical_hash
    return row


def semantic_conflicts(existing: Mapping[str, Any] | Any, incoming: Mapping[str, Any] | Any) -> tuple[str, ...]:
    existing_attributes = memory_semantic_attributes(existing)
    incoming_attributes = memory_semantic_attributes(incoming)
    return tuple(
        field
        for field in SEMANTIC_ATTRIBUTE_FIELDS
        if existing_attributes[field] != incoming_attributes[field]
    )


def resolve_memory_duplicate(
    existing_rows: Iterable[Mapping[str, Any] | Any],
    incoming: Mapping[str, Any] | Any,
) -> int | None:
    """Reuse an exact duplicate or reject same-content semantic conflicts."""

    rows = list(existing_rows)
    existing_ids: list[int] = []
    conflicting_fields: set[str] = set()
    for row in rows:
        memory_id = int(_record_value(row, "id", 0))
        existing_ids.append(memory_id)
        conflicts = semantic_conflicts(row, incoming)
        if not conflicts:
            return memory_id
        conflicting_fields.update(conflicts)
    if rows:
        raise MemoryDuplicateConflictError(existing_ids, conflicting_fields)
    return None
