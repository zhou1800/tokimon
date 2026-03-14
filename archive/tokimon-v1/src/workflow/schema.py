"""Minimal JSON-schema-like validation utilities.

Tokimon intentionally uses a small, deterministic subset of JSON Schema for
runtime validation (no third-party dependencies).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


@dataclass(frozen=True)
class SchemaViolation:
    code: str
    path: str
    message: str


class SchemaValidationError(ValueError):
    def __init__(self, violations: Iterable[SchemaViolation]) -> None:
        self.violations = tuple(violations)
        message = "; ".join(f"{v.code} at {v.path}: {v.message}" for v in self.violations)
        super().__init__(message or "schema validation failed")


WORKER_FINAL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["status", "summary", "artifacts", "metrics", "next_actions", "failure_signature"],
    "properties": {
        "status": {"type": "string", "enum": ["SUCCESS", "FAILURE", "BLOCKED", "PARTIAL"]},
        "summary": {"type": "string"},
        "artifacts": {"type": "array", "items": {"type": "object"}},
        "metrics": {"type": "object"},
        "next_actions": {"type": "array", "items": {"type": "string"}},
        "failure_signature": {"type": "string"},
    },
}


def validate_schema(data: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate `data` against a minimal JSON-schema-like subset.

    Raises:
        SchemaValidationError: on the first encountered set of violations.
    """
    violations: list[SchemaViolation] = []
    _validate_schema_inner(data, schema, path=path, violations=violations)
    if violations:
        raise SchemaValidationError(violations)


def _validate_schema_inner(
    data: Any,
    schema: dict[str, Any],
    *,
    path: str,
    violations: list[SchemaViolation],
) -> None:
    schema_type = schema.get("type")
    if schema_type:
        py_type = _TYPE_MAP.get(schema_type)
        if py_type and not isinstance(data, py_type):
            violations.append(
                SchemaViolation(
                    code="type_mismatch",
                    path=path,
                    message=f"expected {schema_type}",
                )
            )
            return

    enum_values = schema.get("enum")
    if enum_values is not None:
        if data not in set(enum_values):
            violations.append(
                SchemaViolation(
                    code="enum_mismatch",
                    path=path,
                    message="value not in enum",
                )
            )
            return

    if schema_type == "object":
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        if not isinstance(data, dict):
            violations.append(
                SchemaViolation(
                    code="type_mismatch",
                    path=path,
                    message="expected object",
                )
            )
            return
        for key in required:
            if key not in data:
                violations.append(
                    SchemaViolation(
                        code="missing_required",
                        path=f"{path}.{key}",
                        message="missing required key",
                    )
                )
        for key, value in data.items():
            if key in properties:
                _validate_schema_inner(
                    value,
                    properties[key],
                    path=f"{path}.{key}",
                    violations=violations,
                )
    elif schema_type == "array":
        items_schema = schema.get("items")
        if items_schema and isinstance(data, list):
            for idx, item in enumerate(data):
                _validate_schema_inner(
                    item,
                    items_schema,
                    path=f"{path}[{idx}]",
                    violations=violations,
                )
