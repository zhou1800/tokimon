"""Skill gap detection and candidate draft creation.

This module detects repeated patterns across runs based on persisted signals in
Manager retry Lessons, and proposes candidate Prompt Skill drafts (not
registered) when thresholds are met.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from memory.store import MemoryStore
from skills.spec import SkillSpec


_SIGNAL_FIELDS: dict[str, str] = {
    "retry-failure": "failure_signature",
    "subtask-pattern": "subtask_signature",
    "tool-workflow": "tool_workflow_signature",
}

_SIGNAL_ORDER: tuple[str, ...] = ("retry-failure", "subtask-pattern", "tool-workflow")


@dataclass(frozen=True)
class SkillGapCandidate:
    name: str
    signal_type: str
    signature: str
    count: int
    candidate_dir: Path


class SkillGapDetector:
    def __init__(self, repo_root: Path, memory_store: MemoryStore, *, threshold: int = 3) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.repo_root = repo_root
        self.memory_store = memory_store
        self.threshold = threshold
        self.generated_dir = repo_root / "skills_generated"
        self.candidates_dir = self.generated_dir / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

    def observe_retry_lesson(self, retry_lesson_metadata: dict[str, Any]) -> SkillGapCandidate | None:
        """Check whether a newly written retry Lesson triggers a skill-gap proposal."""

        if not isinstance(retry_lesson_metadata, dict):
            return None

        context = {
            "task_id": retry_lesson_metadata.get("task_id"),
            "step_id": retry_lesson_metadata.get("step_id"),
            "worker_type": retry_lesson_metadata.get("worker_type"),
            "strategy_id": retry_lesson_metadata.get("strategy_id"),
            "example_retry_lesson_id": retry_lesson_metadata.get("id"),
        }

        for signal_type in _SIGNAL_ORDER:
            field_name = _SIGNAL_FIELDS[signal_type]
            signature = str(retry_lesson_metadata.get(field_name) or "").strip()
            if not signature:
                continue
            count = self._count_retry_lessons(field_name, signature)
            if count < self.threshold:
                continue
            candidate = self._create_candidate(signal_type, signature, count, context=context)
            if candidate is not None:
                return candidate

        return None

    def _count_retry_lessons(self, field_name: str, signature: str) -> int:
        count = 0
        for metadata in self._iter_retry_lesson_metadata():
            if str(metadata.get(field_name) or "").strip() == signature:
                count += 1
        return count

    def _iter_retry_lesson_metadata(self) -> Iterable[dict[str, Any]]:
        lessons_dir = self.memory_store.lessons_dir
        if not lessons_dir.exists():
            return
        for path in sorted(lessons_dir.glob("lesson-*.md")):
            metadata = _read_lesson_metadata(path)
            if not metadata:
                continue
            if _is_retry_lesson(metadata):
                yield metadata

    def _create_candidate(
        self,
        signal_type: str,
        signature: str,
        count: int,
        *,
        context: dict[str, Any],
    ) -> SkillGapCandidate | None:
        candidate_name = _candidate_name(signal_type, signature)
        if self._candidate_exists(candidate_name):
            return None
        candidate_dir = self.candidates_dir / f"{_timestamp_slug()}-{candidate_name}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        spec = _candidate_spec(candidate_name, signal_type, signature)
        (candidate_dir / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
        (candidate_dir / "gap.json").write_text(
            json.dumps(
                {"signal_type": signal_type, "signature": signature, "count": count, "threshold": self.threshold, "context": context},
                indent=2,
                sort_keys=True,
            )
        )
        (candidate_dir / "prompt.md").write_text(_candidate_prompt(candidate_name, signal_type, signature))
        (candidate_dir / "validation.md").write_text(_candidate_validation_checklist(candidate_name))
        self._write_gap_lesson(candidate_name, signal_type, signature, count, context=context)
        return SkillGapCandidate(
            name=candidate_name,
            signal_type=signal_type,
            signature=signature,
            count=count,
            candidate_dir=candidate_dir,
        )

    def _candidate_exists(self, candidate_name: str) -> bool:
        pattern = f"*-{candidate_name}"
        for path in self.candidates_dir.glob(pattern):
            if path.is_dir():
                return True
        return False

    def _write_gap_lesson(
        self,
        candidate_name: str,
        signal_type: str,
        signature: str,
        count: int,
        *,
        context: dict[str, Any],
    ) -> None:
        lesson_id = str(uuid.uuid4())
        metadata = {
            "id": lesson_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tags": ["skill-gap", signal_type, candidate_name],
            "component": "skills",
            "signal_type": signal_type,
            "signal_signature": signature,
        }
        body = (
            "Skill gap detected.\n"
            f"Signal type: {signal_type}\n"
            f"Signature: {signature}\n"
            f"Occurrences: {count} (threshold={self.threshold})\n"
            f"Candidate draft created: {candidate_name}\n"
            f"Context: {context}\n"
            "Promotion: candidate is not registered; validate before promoting.\n"
        )
        self.memory_store.write_lesson(metadata, body)


def _read_lesson_metadata(path: Path) -> dict[str, Any] | None:
    try:
        header = path.read_text().split("\n", 1)[0].strip()
        if not header:
            return None
        parsed = json.loads(header)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _is_retry_lesson(metadata: dict[str, Any]) -> bool:
    lesson_type = metadata.get("lesson_type")
    if lesson_type == "retry":
        return True
    tags = metadata.get("tags")
    if isinstance(tags, list):
        return "retry" in tags
    if isinstance(tags, str):
        return "retry" in {t.strip() for t in tags.split(",") if t.strip()}
    return False


def _candidate_name(signal_type: str, signature: str) -> str:
    digest = hashlib.sha1(f"{signal_type}:{signature}".encode("utf-8")).hexdigest()[:10]
    return f"gap-{signal_type}-{digest}"


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _candidate_spec(candidate_name: str, signal_type: str, signature: str) -> SkillSpec:
    return SkillSpec(
        name=candidate_name,
        kind="prompt",
        purpose=f"Standardize a reusable response for repeated signal: {signal_type}.",
        contract={
            "inputs": {"signal_type": signal_type, "signature": signature},
            "outputs": {"summary": "string", "checklist": "markdown"},
            "side_effects": "none",
        },
        preconditions=["Candidate must be reviewed and validated before promotion."],
        required_tools=[],
        retrieval_prefs={"stage1": "recent similar failures", "stage2": "component lessons", "stage3": "cross-task patterns"},
        failure_modes=["Overfitting to a narrow signature; missing context; unsafe suggestions."],
        safety_notes={
            "hard": ["MUST NOT violate Tokimon Non-goals (no hacking/credential theft/malware)."],
            "soft": ["SHOULD ask for clarification when signature/context is ambiguous."],
        },
        cost_energy_notes="Low: prompt-only; avoid if a deterministic code skill is required.",
        validation_method={"type": "checklist", "artifact": "validation.md"},
        version="0.1.0",
        deprecation_policy="Deprecate when promoted skill exists or signature no longer recurs.",
        module=None,
        prompt_template=None,
        prompt_path=None,
    )


def _candidate_prompt(candidate_name: str, signal_type: str, signature: str) -> str:
    return (
        f"# {candidate_name}\n\n"
        "This is an auto-generated candidate Prompt Skill draft.\n\n"
        f"## Trigger\n- Signal type: `{signal_type}`\n- Signature: `{signature}`\n\n"
        "## Instructions\n"
        "- Use Lessons and artifacts to avoid repeating known failures.\n"
        "- Produce a concise checklist and a concrete next-step plan.\n"
        "- Do not propose unsafe actions.\n"
    )


def _candidate_validation_checklist(candidate_name: str) -> str:
    return (
        f"# Validation Checklist: {candidate_name}\n\n"
        "- [ ] Metadata reviewed and complete\n"
        "- [ ] Prompt template is specific and reusable\n"
        "- [ ] Safety notes cover Non-goals\n"
    )

