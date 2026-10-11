"""Recall evaluation helpers for Mem Palace."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RecallCase:
    query: str
    expected: tuple[str, ...]
    absent: tuple[str, ...]
    limit: int
    scope: str | None = None
    room: str | None = None
    kind: str | None = None
    tag: str | None = None
    entity: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RecallCase":
        query = str(data.get("query") or "").strip()
        if not query:
            raise ValueError("recall case requires query")
        expected = _string_tuple(data.get("expected") or data.get("expected_contains") or ())
        absent = _string_tuple(data.get("absent") or data.get("absent_contains") or ())
        if not expected and not absent:
            raise ValueError(f"recall case {query!r} requires expected or absent assertions")
        return cls(
            query=query,
            expected=expected,
            absent=absent,
            limit=int(data.get("limit") or 5),
            scope=_optional_string(data.get("scope")),
            room=_optional_string(data.get("room")),
            kind=_optional_string(data.get("kind")),
            tag=_optional_string(data.get("tag")),
            entity=_optional_string(data.get("entity")),
        )


RecallFunction = Callable[..., Sequence[Any]]


def load_recall_cases(path: Path | str) -> list[RecallCase]:
    source_path = Path(path).expanduser().resolve()
    text = source_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if source_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        rows = parsed.get("cases", parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(rows, list):
        raise ValueError("recall evaluation file must contain a list of cases")
    return [RecallCase.from_mapping(row) for row in rows]


def evaluate_recall(recall: RecallFunction, cases: Iterable[RecallCase]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            raw_results = recall(
                case.query,
                limit=case.limit,
                kind=case.kind,
                tag=case.tag,
                scope=case.scope,
                room=case.room,
                entity=case.entity,
            )
            serialized = [_result_to_dict(result) for result in raw_results]
            haystacks = [_result_text(result) for result in serialized]
            missing = [expected for expected in case.expected if not _contains_any(haystacks, expected)]
            unexpected = [absent for absent in case.absent if _contains_any(haystacks, absent)]
            passed = not missing and not unexpected
            results.append(
                {
                    "query": case.query,
                    "passed": passed,
                    "missing": missing,
                    "unexpected": unexpected,
                    "result_count": len(serialized),
                    "top_result": serialized[0].get("summary") if serialized else None,
                    "results": serialized,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive report path
            results.append(
                {
                    "query": case.query,
                    "passed": False,
                    "error": str(exc),
                    "missing": list(case.expected),
                    "unexpected": [],
                    "result_count": 0,
                    "top_result": None,
                    "results": [],
                }
            )

    passed_count = sum(1 for item in results if item["passed"])
    total = len(results)
    return {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": round(passed_count / total, 4) if total else 1.0,
        "cases": results,
    }


def render_recall_evaluation(report: Mapping[str, Any]) -> str:
    lines = [
        f"recall_eval: {report['passed']}/{report['total']} passed "
        f"(pass_rate={float(report['pass_rate']):.2f})"
    ]
    for case in report.get("cases", []):
        status = "PASS" if case.get("passed") else "FAIL"
        lines.append(f"{status}: {case.get('query')}")
        if case.get("top_result"):
            lines.append(f"  top: {case['top_result']}")
        if case.get("missing"):
            lines.append("  missing: " + ", ".join(case["missing"]))
        if case.get("unexpected"):
            lines.append("  unexpected: " + ", ".join(case["unexpected"]))
        if case.get("error"):
            lines.append(f"  error: {case['error']}")
    return "\n".join(lines)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return dict(result.to_dict())
    if isinstance(result, Mapping):
        return dict(result)
    return {"summary": str(result)}


def _result_text(result: Mapping[str, Any]) -> str:
    fields = ("summary", "details", "tags", "source", "scope", "room", "entities")
    return "\n".join(str(result.get(field) or "") for field in fields).lower()


def _contains_any(haystacks: Sequence[str], needle: str) -> bool:
    lowered = needle.lower()
    return any(lowered in haystack for haystack in haystacks)
