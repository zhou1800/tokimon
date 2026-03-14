from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agents.manager import Manager, Strategy
from memory.store import MemoryStore


_REQUIRED_CHARTER_FIELDS = (
    "failure_signature",
    "root_cause_hypothesis",
    "strategy_change",
    "evidence_of_novelty",
    "retrieval_tags",
)


def _charter_metadata(lesson_id: str, lesson_type: str) -> dict[str, Any]:
    return {
        "id": lesson_id,
        "lesson_type": lesson_type,
        "failure_signature": "fs1",
        "root_cause_hypothesis": "hypothesis",
        "strategy_change": "change",
        "evidence_of_novelty": "novelty",
        "retrieval_tags": ["memory"],
        "component": "tests",
        "tags": [lesson_type],
    }


@pytest.mark.parametrize("lesson_type", ["failure", "retry"])
def test_write_lesson_requires_charter_fields(tmp_path: Path, lesson_type: str) -> None:
    store = MemoryStore(tmp_path)

    store.write_lesson(_charter_metadata("ok-" + lesson_type, lesson_type), "ok")

    for field in _REQUIRED_CHARTER_FIELDS:
        metadata = _charter_metadata("missing-" + lesson_type + "-" + field, lesson_type)
        metadata.pop(field)
        with pytest.raises(ValueError, match=r"(?i)lesson"):
            store.write_lesson(metadata, "body")


def test_write_lesson_denies_secret_metadata(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    metadata = _charter_metadata("secret-metadata", "failure")
    metadata["api_key"] = "supersecret"
    with pytest.raises(ValueError, match=r"(?i)secret"):
        store.write_lesson(metadata, "body")


def test_write_lesson_redacts_bearer_tokens(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    lesson = store.write_lesson(_charter_metadata("redact-body", "failure"), "Authorization: Bearer supersecret")
    raw = lesson.path.read_text()
    assert "supersecret" not in raw

    loaded = store.load_lesson("redact-body")
    assert "supersecret" not in loaded.body
    assert "<REDACTED>" in loaded.body


def test_retrieval_tags_are_indexed_for_search(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    metadata = _charter_metadata("retrieval-tags", "failure")
    metadata["tags"] = ["not-this"]
    metadata["retrieval_tags"] = ["tag:from_retrieval_tags"]
    store.write_lesson(metadata, "body")
    lessons = store.retrieve(
        "q",
        stage=2,
        limit=5,
        component="tests",
        tags=["tag:from_retrieval_tags"],
        failure_signature="fs1",
    )
    assert [lesson.metadata["id"] for lesson in lessons] == ["retrieval-tags"]


def test_manager_writes_failure_lesson_with_required_fields(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    manager = Manager(store)
    strategy = Strategy(
        strategy_id="patch",
        worker_type="Debugger",
        retrieval_stage=2,
        strategy_class="patch",
        tool_sequence=["grep", "pytest", "patch"],
    )
    lesson_id = manager.write_failure_lesson(
        "task1",
        "step1",
        strategy,
        "E42:boom",
        "it failed",
        component="hierarchical_runner",
    )
    lesson = store.load_lesson(lesson_id)
    assert lesson.metadata["lesson_type"] == "failure"
    for field in _REQUIRED_CHARTER_FIELDS:
        assert field in lesson.metadata
    tags = set(lesson.metadata["retrieval_tags"])
    assert "task:task1" in tags
    assert "component:hierarchical_runner" in tags
    assert "tool:pytest" in tags
