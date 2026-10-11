"""Pure Markdown rule scanning shared by the audit workflows."""

from __future__ import annotations

import hashlib
from typing import Any, Pattern


def rule_hash(rule_id: str, description: str) -> str:
    return hashlib.sha256(f"{rule_id}\n{description}".encode("utf-8")).hexdigest()


def rules_from_lines(
    source_label: str,
    lines: list[str],
    *,
    rule_pattern: Pattern[str],
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        match = rule_pattern.match(line)
        if not match:
            continue
        rule_id = match.group("rule_id")
        description_parts = [match.group("description")]
        for continuation in lines[line_number:]:
            if (
                rule_pattern.match(continuation)
                or continuation.startswith(("#", "- "))
                or (continuation.strip() and not continuation.startswith((" ", "\t")))
            ):
                break
            if stripped := continuation.strip():
                description_parts.append(stripped)
        description = " ".join(description_parts)
        rules.append({
            "rule_id": rule_id,
            "description": description,
            "source_path": source_label,
            "line": line_number,
            "rule_hash": rule_hash(rule_id, description),
        })
    return rules
