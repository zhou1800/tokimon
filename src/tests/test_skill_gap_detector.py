from __future__ import annotations

import json
from pathlib import Path

from agents.manager import Manager, Strategy
from memory.store import MemoryStore
from skills.gap_detector import SkillGapDetector


def _lesson_metadata(path: Path) -> dict[str, object]:
    header = path.read_text().split("\n", 1)[0].strip()
    return json.loads(header) if header else {}


def test_gap_detector_triggers_only_after_threshold(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    memory_store = MemoryStore(tmp_path / "memory")
    manager = Manager(memory_store)
    gap_detector = SkillGapDetector(repo_root, memory_store, threshold=3)

    prev = Strategy(
        strategy_id="draft",
        worker_type="Planner",
        retrieval_stage=1,
        strategy_class="write_from_scratch",
        tool_sequence=["grep", "file"],
    )
    nxt = Strategy(
        strategy_id="patch",
        worker_type="Debugger",
        retrieval_stage=2,
        strategy_class="patch",
        tool_sequence=["grep", "pytest", "patch"],
    )

    candidates_dir = repo_root / "skills_generated" / "candidates"
    manifest_path = repo_root / "skills_generated" / "manifest.json"

    manager.write_retry_lesson(
        "task-1",
        "step-1",
        prev,
        nxt,
        "failure-sig",
        "Changed strategy",
        step_description="Example step description.",
        gap_detector=gap_detector,
    )
    assert not candidates_dir.exists() or not any(candidates_dir.iterdir())

    manager.write_retry_lesson(
        "task-2",
        "step-1",
        prev,
        nxt,
        "failure-sig",
        "Changed strategy",
        step_description="Example step description.",
        gap_detector=gap_detector,
    )
    assert not candidates_dir.exists() or not any(candidates_dir.iterdir())

    manager.write_retry_lesson(
        "task-3",
        "step-1",
        prev,
        nxt,
        "failure-sig",
        "Changed strategy",
        step_description="Example step description.",
        gap_detector=gap_detector,
    )
    assert candidates_dir.exists()
    candidates = [p for p in candidates_dir.iterdir() if p.is_dir()]
    assert len(candidates) == 1
    candidate_dir = candidates[0]
    assert (candidate_dir / "spec.json").exists()
    assert (candidate_dir / "prompt.md").exists()
    assert (candidate_dir / "validation.md").exists()
    assert (candidate_dir / "gap.json").exists()

    assert not manifest_path.exists()

    lessons = list((tmp_path / "memory" / "lessons").glob("lesson-*.md"))
    assert len(lessons) == 4
    tags = [_lesson_metadata(path).get("tags") for path in lessons]
    assert any(isinstance(tag, list) and "skill-gap" in tag for tag in tags)

