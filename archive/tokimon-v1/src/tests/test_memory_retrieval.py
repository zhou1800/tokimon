from pathlib import Path

from memory.store import MemoryStore


def _lesson_ids(lessons) -> set[str]:
    return {lesson.metadata["id"] for lesson in lessons}


def test_retrieve_requires_context(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.write_lesson({"id": "l1", "tags": ["t"], "component": "c", "failure_signature": "fs1"}, "body")
    try:
        store.retrieve("q", stage=1, limit=1, component=None, tags=["t"], failure_signature="fs1")
    except ValueError:
        return
    raise AssertionError("expected retrieval to require component")


def test_staged_retrieval(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.write_lesson(
        {"id": "l1", "tags": ["alpha", "beta"], "component": "core", "failure_signature": "fs1"},
        "alpha lesson",
    )
    store.write_lesson(
        {"id": "l2", "tags": ["beta"], "component": "core", "failure_signature": "fs1"},
        "beta lesson",
    )
    store.write_lesson(
        {"id": "l3", "tags": ["beta"], "component": "adjacent", "failure_signature": "fs1"},
        "adjacent lesson",
    )
    store.write_lesson(
        {"id": "l4", "tags": ["beta"], "component": "core", "failure_signature": "fs2"},
        "different failure signature",
    )
    store.write_lesson(
        {"id": "l5", "tags": ["gamma"], "component": "other", "failure_signature": "fs1"},
        "fs1, but different tags and component",
    )
    store.write_lesson(
        {"id": "l6", "tags": ["beta"], "component": "other", "failure_signature": "fs1"},
        "beta lesson in other component",
    )
    store.write_lesson(
        {"id": "l7", "tags": ["alpha"], "component": "core", "failure_signature": "fs2"},
        "alpha but wrong failure signature",
    )
    store.write_lesson(
        {"id": "l8", "tags": ["delta"], "component": "core", "failure_signature": "fs1"},
        "alpha but wrong tags",
    )
    store.write_lesson(
        {"id": "l10", "tags": ["alpha"], "component": "other", "failure_signature": "fs1"},
        "alpha but wrong component",
    )
    store.write_lesson(
        {"id": "l9", "tags": ["beta"], "component": "adjacent", "failure_signature": "fs2"},
        "beta adjacent but wrong failure signature",
    )

    stage1 = store.retrieve("alpha", stage=1, limit=5, component="core", tags=["alpha"], failure_signature="fs1")
    assert _lesson_ids(stage1) == {"l1"}

    stage2 = store.retrieve("alpha", stage=2, limit=20, component="core", tags=["beta"], failure_signature="fs1")
    stage2_ids = _lesson_ids(stage2)
    assert {"l1", "l2"} <= stage2_ids
    assert "l3" in stage2_ids
    assert "l4" not in stage2_ids
    assert "l5" not in stage2_ids
    assert "l6" not in stage2_ids
    assert "l7" not in stage2_ids
    assert "l8" not in stage2_ids
    assert "l9" not in stage2_ids

    stage3 = store.retrieve("alpha", stage=3, limit=20, component="core", tags=["beta"], failure_signature="fs1")
    stage3_ids = _lesson_ids(stage3)
    assert stage2_ids <= stage3_ids
    assert "l4" not in stage3_ids
    assert "l5" in stage3_ids
    assert "l6" in stage3_ids
    assert "l7" not in stage3_ids
    assert "l9" not in stage3_ids


def test_stage3_can_expand_by_similar_failure_signature(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.write_lesson(
        {"id": "f1", "tags": ["beta"], "component": "core", "failure_signature": "E123:alpha"},
        "family lesson one",
    )
    store.write_lesson(
        {"id": "f2", "tags": ["beta"], "component": "core", "failure_signature": "E123:bravo"},
        "family lesson two",
    )
    lessons = store.retrieve("q", stage=3, limit=10, component="core", tags=["beta"], failure_signature="E123:charlie")
    assert {"f1", "f2"} <= _lesson_ids(lessons)
