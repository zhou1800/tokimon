from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokimon.engine import add_direction, create_state, feed_tokens, prepare_for_task, run_idle_cycle
from tokimon.persistence import (
    StateLoadError,
    load_state,
    save_state,
)


def test_state_round_trip_preserves_durable_snapshot(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = create_state()
    feed_tokens(state, 5)
    add_direction(state, "python", priority=9, note="backend")
    prepare_for_task(
        state,
        summary="Ship a small API with tests",
        requested_skills=["python", "testing"],
        prep_budget=2,
    )

    save_state(state, state_path)
    reloaded = _load_from(path=state_path, tmp_path=tmp_path)

    assert reloaded.available_tokens == state.available_tokens
    assert reloaded.total_tokens_eaten == state.total_tokens_eaten
    assert reloaded.improvement_cycles == state.improvement_cycles
    assert reloaded.task_runs == state.task_runs
    assert reloaded.skills["python"].level == state.skills["python"].level
    assert reloaded.directions[0].skill == "python"
    assert reloaded.task_history[-1].summary == "Ship a small API with tests"
    assert "quality_score" not in json.loads(state_path.read_text(encoding="utf-8"))


def test_load_state_migrates_v2_snapshot_and_recomputes_runtime_data(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    legacy_payload = {
        "version": 2,
        "available_tokens": 1,
        "total_tokens_eaten": 4,
        "improvement_cycles": 4,
        "idle_cycles": 3,
        "task_runs": 1,
        "quality_score": 999.0,
        "skills": {
            "python": {
                "name": "python",
                "level": 4,
                "practice_runs": 4,
                "directed_cycles": 2,
                "last_improved_at": "2026-04-10T10:00:00+00:00",
            }
        },
        "directions": [
            {
                "skill": "python",
                "priority": 9,
                "note": "backend",
                "created_at": "2026-04-09T10:00:00+00:00",
            }
        ],
        "improvement_history": [
            {
                "skill": "python",
                "reason": "idle",
                "tokens_spent": 1,
                "before_level": 3,
                "after_level": 4,
                "timestamp": "2026-04-10T10:00:00+00:00",
            }
        ],
        "task_history": [
            {
                "summary": "Build API tests",
                "requested_skills": ["python"],
                "focus_skills": ["python"],
                "auto_training_spent": 1,
                "confidence": "medium",
                "timestamp": "2026-04-10T11:00:00+00:00",
            }
        ],
    }
    state_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    loaded = _load_from(path=state_path, tmp_path=tmp_path)

    assert loaded.version == 3
    assert loaded.created_at == "2026-04-09T10:00:00+00:00"
    assert loaded.updated_at == "2026-04-10T11:00:00+00:00"
    assert loaded.quality_score != 999.0
    assert loaded.session.derived_rankings["skills_by_level"] == ["python"]

    save_state(loaded, state_path)
    rewritten = json.loads(state_path.read_text(encoding="utf-8"))
    assert rewritten["version"] == 3
    assert "quality_score" not in rewritten


def test_load_state_fails_without_overwriting_malformed_file(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    malformed = '{"version": 3, "available_tokens": "oops"'
    state_path.write_text(malformed, encoding="utf-8")

    with pytest.raises(StateLoadError):
        _load_from(path=state_path, tmp_path=tmp_path)

    assert state_path.read_text(encoding="utf-8") == malformed


def test_runtime_only_session_state_is_never_serialized(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = create_state()
    state.session.current_command_context = "idle"
    state.session.caches["ephemeral"] = {"key": "value"}
    state.session.derived_rankings["skills_by_level"] = ["secret"]
    state.session.temporary_approvals.add("allow-once")
    state.session.daemon_state["worker"] = "daemon-1"

    save_state(state, state_path)
    serialized = state_path.read_text(encoding="utf-8")
    payload = json.loads(serialized)

    assert "session" not in payload
    assert "current_command_context" not in serialized
    assert "allow-once" not in serialized
    assert "daemon-1" not in serialized


def test_derived_values_are_recomputed_after_load(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = create_state()
    feed_tokens(state, 3)
    add_direction(state, "python", priority=10, note="backend")
    run_idle_cycle(state, max_cycles=2)
    expected_score = state.quality_score

    save_state(state, state_path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["quality_score"] = 12345.0
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = _load_from(path=state_path, tmp_path=tmp_path)

    assert reloaded.quality_score == expected_score
    assert reloaded.session.derived_rankings["skills_by_level"][0] == "python"


def test_project_settings_can_shift_defaults_without_enabling_risky_runtime_flags(tmp_path: Path) -> None:
    project_settings_path = tmp_path / "tokimon.settings.json"
    user_settings_path = tmp_path / "user-settings.json"
    project_settings_path.write_text(
        json.dumps(
            {
                "version": 1,
                "default_task_prep_budget": 5,
                "allow_background_runtime": True,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_state(
        tmp_path / "missing-state.json",
        project_settings_path=project_settings_path,
        user_settings_path=user_settings_path,
    )

    assert loaded.settings.default_task_prep_budget == 5
    assert not loaded.settings.allow_background_runtime


def test_backup_rotation_keeps_correct_generations(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"

    # Write four successive versions so we can verify rotation order.
    for generation in range(1, 5):
        state = create_state()
        feed_tokens(state, generation)
        save_state(state, state_path)

    # Default state_backup_count is 3, so we should have .bak.1 through .bak.3.
    for index in range(1, 4):
        bak = state_path.with_name(f"{state_path.name}.bak.{index}")
        assert bak.exists(), f"expected {bak.name} to exist"
        content = json.loads(bak.read_text(encoding="utf-8"))
        # .bak.1 is the most recent backup (generation before last write).
        # available_tokens for generation N = N after feeding, minus 0 spent.
        expected_generation = 4 - index
        assert content["available_tokens"] == expected_generation

    # .bak.4 must not exist — only 3 backups are kept.
    assert not state_path.with_name(f"{state_path.name}.bak.4").exists()


def _load_from(*, path: Path, tmp_path: Path):
    return load_state(
        path,
        project_settings_path=tmp_path / "tokimon.settings.json",
        user_settings_path=tmp_path / "user-settings.json",
    )
