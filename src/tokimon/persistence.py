from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from tokimon.engine import HISTORY_LIMIT, refresh_runtime_state
from tokimon.models import (
    CURRENT_SETTINGS_VERSION,
    CURRENT_SNAPSHOT_VERSION,
    Direction,
    ImprovementRecord,
    SkillRecord,
    TaskRecord,
    TokimonSettings,
    TokimonSnapshot,
    TokimonState,
    normalize_skill_name,
    utc_now,
)


DEFAULT_STATE_PATH = Path(".tokimon/state.json")
DEFAULT_PROJECT_SETTINGS_PATH = Path("tokimon.settings.json")
DEFAULT_USER_SETTINGS_PATH = Path(".tokimon/user-settings.json")
RISKY_USER_ONLY_FIELDS = {
    "allow_background_runtime",
    "allow_cached_approvals",
}


class TokimonPersistenceError(Exception):
    """Raised when Tokimon cannot safely load or save durable data."""


class StateLoadError(TokimonPersistenceError):
    """Raised when the durable state file is missing required structure."""


class SettingsLoadError(TokimonPersistenceError):
    """Raised when a settings file is malformed."""


def load_state(
    path: Path = DEFAULT_STATE_PATH,
    *,
    project_settings_path: Path = DEFAULT_PROJECT_SETTINGS_PATH,
    user_settings_path: Path = DEFAULT_USER_SETTINGS_PATH,
) -> TokimonState:
    settings = load_settings(
        project_path=project_settings_path,
        user_path=user_settings_path,
    )
    if not path.exists():
        state = TokimonState(settings=settings)
        refresh_runtime_state(state)
        return state

    raw_data = _read_json_file(path, StateLoadError)
    migrated = migrate_snapshot(raw_data)
    snapshot = _parse_snapshot(migrated)
    state = TokimonState(snapshot=snapshot, settings=settings)
    refresh_runtime_state(state)
    return state


def save_state(state: TokimonState, path: Path = DEFAULT_STATE_PATH) -> None:
    snapshot = _prepare_snapshot_for_write(state)
    payload = json.dumps(snapshot.to_dict(), indent=2) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    backup_count = max(0, state.settings.state_backup_count)
    if path.exists() and backup_count > 0:
        _rotate_backups(path, backup_count)
        shutil.copy2(path, _backup_path(path, 1))

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def load_settings(
    *,
    project_path: Path = DEFAULT_PROJECT_SETTINGS_PATH,
    user_path: Path = DEFAULT_USER_SETTINGS_PATH,
) -> TokimonSettings:
    settings = TokimonSettings()

    project_overlay = _load_settings_overlay(project_path, source_name="project")
    if project_overlay:
        settings = _merge_settings(settings, project_overlay, allow_risky=False)

    user_overlay = _load_settings_overlay(user_path, source_name="user")
    if user_overlay:
        settings = _merge_settings(settings, user_overlay, allow_risky=True)

    return settings


def migrate_snapshot(data: object) -> dict[str, object]:
    snapshot = _require_mapping(data, "state file", StateLoadError)
    version = _require_int(snapshot.get("version"), "version", StateLoadError, minimum=1)
    migrated = dict(snapshot)

    while version != CURRENT_SNAPSHOT_VERSION:
        if version == 2:
            migrated = _migrate_v2_snapshot(migrated)
        else:
            raise StateLoadError(f"unsupported Tokimon snapshot version: {version}")
        version = _require_int(migrated.get("version"), "version", StateLoadError, minimum=1)

    return migrated


def _prepare_snapshot_for_write(state: TokimonState) -> TokimonSnapshot:
    state.snapshot.version = CURRENT_SNAPSHOT_VERSION
    state.updated_at = utc_now()
    state.improvement_history = state.improvement_history[-HISTORY_LIMIT:]
    state.task_history = state.task_history[-HISTORY_LIMIT:]

    snapshot = _parse_snapshot(state.snapshot.to_dict())
    state.snapshot = snapshot
    refresh_runtime_state(state)
    return snapshot


def _migrate_v2_snapshot(data: dict[str, object]) -> dict[str, object]:
    improvement_history = _migration_list(data, "improvement_history")
    task_history = _migration_list(data, "task_history")
    directions = _migration_list(data, "directions")
    skills = _migration_mapping(data, "skills")
    created_at, updated_at = _derive_snapshot_window(skills, directions, improvement_history, task_history)

    idle_from_history = sum(
        1
        for entry in improvement_history
        if isinstance(entry, dict) and entry.get("reason") == "idle"
    )
    spent_from_history = sum(
        int(entry.get("tokens_spent", 0))
        for entry in improvement_history
        if isinstance(entry, dict) and not isinstance(entry.get("tokens_spent", 0), bool)
    )

    return {
        "version": CURRENT_SNAPSHOT_VERSION,
        "created_at": created_at,
        "updated_at": updated_at,
        "available_tokens": data.get("available_tokens", 0),
        "total_tokens_eaten": max(
            _coerce_int(data.get("total_tokens_eaten", 0), "total_tokens_eaten", StateLoadError),
            spent_from_history,
        ),
        "improvement_cycles": max(
            _coerce_int(data.get("improvement_cycles", 0), "improvement_cycles", StateLoadError),
            len(improvement_history),
        ),
        "idle_cycles": max(
            _coerce_int(data.get("idle_cycles", 0), "idle_cycles", StateLoadError),
            idle_from_history,
        ),
        "task_runs": max(
            _coerce_int(data.get("task_runs", 0), "task_runs", StateLoadError),
            len(task_history),
        ),
        "skills": skills,
        "directions": directions,
        "improvement_history": improvement_history[-HISTORY_LIMIT:],
        "task_history": task_history[-HISTORY_LIMIT:],
    }


def _derive_snapshot_window(
    skills: dict[str, object],
    directions: list[object],
    improvement_history: list[object],
    task_history: list[object],
) -> tuple[str, str]:
    timestamps: list[str] = []

    for record in directions:
        if isinstance(record, dict) and isinstance(record.get("created_at"), str):
            timestamps.append(record["created_at"])

    for record in improvement_history:
        if isinstance(record, dict) and isinstance(record.get("timestamp"), str):
            timestamps.append(record["timestamp"])

    for record in task_history:
        if isinstance(record, dict) and isinstance(record.get("timestamp"), str):
            timestamps.append(record["timestamp"])

    for record in skills.values():
        if isinstance(record, dict) and isinstance(record.get("last_improved_at"), str):
            timestamps.append(record["last_improved_at"])

    if not timestamps:
        current = utc_now()
        return current, current
    return min(timestamps), max(timestamps)


def _load_settings_overlay(path: Path, *, source_name: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    raw_data = _read_json_file(path, SettingsLoadError)
    return _parse_settings_overlay(raw_data, source_name=source_name)


def _merge_settings(
    base: TokimonSettings,
    overlay: dict[str, object],
    *,
    allow_risky: bool,
) -> TokimonSettings:
    merged = TokimonSettings.from_dict(base.to_dict())

    if "default_task_prep_budget" in overlay:
        merged.default_task_prep_budget = int(overlay["default_task_prep_budget"])
    if "default_idle_max_cycles" in overlay:
        merged.default_idle_max_cycles = overlay["default_idle_max_cycles"]  # type: ignore[assignment]
    if "state_backup_count" in overlay:
        merged.state_backup_count = int(overlay["state_backup_count"])

    if allow_risky:
        if "allow_background_runtime" in overlay:
            merged.allow_background_runtime = bool(overlay["allow_background_runtime"])
        if "allow_cached_approvals" in overlay:
            merged.allow_cached_approvals = bool(overlay["allow_cached_approvals"])

    return merged


def _parse_snapshot(data: object) -> TokimonSnapshot:
    snapshot = _require_mapping(data, "state file", StateLoadError)
    version = _require_int(snapshot.get("version"), "version", StateLoadError, minimum=1)
    if version != CURRENT_SNAPSHOT_VERSION:
        raise StateLoadError(f"expected Tokimon snapshot version {CURRENT_SNAPSHOT_VERSION}, got {version}")

    skills = {
        skill_name: _parse_skill_record(record, key_name=skill_name)
        for skill_name, record in _require_mapping(snapshot.get("skills"), "skills", StateLoadError).items()
    }

    directions = [
        _parse_direction(record, index=index)
        for index, record in enumerate(_require_list(snapshot.get("directions"), "directions", StateLoadError))
    ]
    improvement_history = [
        _parse_improvement_record(record, index=index)
        for index, record in enumerate(
            _require_list(snapshot.get("improvement_history"), "improvement_history", StateLoadError)
        )
    ][-HISTORY_LIMIT:]
    task_history = [
        _parse_task_record(record, index=index)
        for index, record in enumerate(_require_list(snapshot.get("task_history"), "task_history", StateLoadError))
    ][-HISTORY_LIMIT:]

    return TokimonSnapshot(
        version=version,
        created_at=_require_str(snapshot.get("created_at"), "created_at", StateLoadError),
        updated_at=_require_str(snapshot.get("updated_at"), "updated_at", StateLoadError),
        available_tokens=_require_int(
            snapshot.get("available_tokens"),
            "available_tokens",
            StateLoadError,
            minimum=0,
        ),
        total_tokens_eaten=_require_int(
            snapshot.get("total_tokens_eaten"),
            "total_tokens_eaten",
            StateLoadError,
            minimum=0,
        ),
        improvement_cycles=_require_int(
            snapshot.get("improvement_cycles"),
            "improvement_cycles",
            StateLoadError,
            minimum=0,
        ),
        idle_cycles=_require_int(snapshot.get("idle_cycles"), "idle_cycles", StateLoadError, minimum=0),
        task_runs=_require_int(snapshot.get("task_runs"), "task_runs", StateLoadError, minimum=0),
        skills=skills,
        directions=directions,
        improvement_history=improvement_history,
        task_history=task_history,
    )


def _parse_settings_overlay(data: object, *, source_name: str) -> dict[str, object]:
    settings = _require_mapping(data, f"{source_name} settings", SettingsLoadError)
    version = settings.get("version", CURRENT_SETTINGS_VERSION)
    parsed_version = _require_int(version, "version", SettingsLoadError, minimum=1)
    if parsed_version != CURRENT_SETTINGS_VERSION:
        raise SettingsLoadError(
            f"unsupported Tokimon {source_name} settings version: {parsed_version}"
        )

    overlay: dict[str, object] = {}
    if "default_task_prep_budget" in settings:
        overlay["default_task_prep_budget"] = _require_int(
            settings["default_task_prep_budget"],
            "default_task_prep_budget",
            SettingsLoadError,
            minimum=0,
        )
    if "default_idle_max_cycles" in settings:
        idle_value = settings["default_idle_max_cycles"]
        if idle_value is None:
            overlay["default_idle_max_cycles"] = None
        else:
            overlay["default_idle_max_cycles"] = _require_int(
                idle_value,
                "default_idle_max_cycles",
                SettingsLoadError,
                minimum=1,
            )
    if "state_backup_count" in settings:
        overlay["state_backup_count"] = _require_int(
            settings["state_backup_count"],
            "state_backup_count",
            SettingsLoadError,
            minimum=0,
        )

    for field_name in RISKY_USER_ONLY_FIELDS:
        if field_name in settings:
            overlay[field_name] = _require_bool(settings[field_name], field_name, SettingsLoadError)

    return overlay


def _parse_skill_record(data: object, *, key_name: object) -> SkillRecord:
    record = _require_mapping(data, f"skills[{key_name!r}]", StateLoadError)
    record_name = normalize_skill_name(
        _require_str(record.get("name", str(key_name)), f"skills[{key_name!r}].name", StateLoadError)
    )
    return SkillRecord(
        name=record_name,
        level=_require_int(record.get("level", 0), f"skills[{key_name!r}].level", StateLoadError, minimum=0),
        practice_runs=_require_int(
            record.get("practice_runs", 0),
            f"skills[{key_name!r}].practice_runs",
            StateLoadError,
            minimum=0,
        ),
        directed_cycles=_require_int(
            record.get("directed_cycles", 0),
            f"skills[{key_name!r}].directed_cycles",
            StateLoadError,
            minimum=0,
        ),
        last_improved_at=_optional_str(
            record.get("last_improved_at"),
            f"skills[{key_name!r}].last_improved_at",
            StateLoadError,
        ),
    )


def _parse_direction(data: object, *, index: int) -> Direction:
    record = _require_mapping(data, f"directions[{index}]", StateLoadError)
    return Direction(
        skill=normalize_skill_name(_require_str(record.get("skill"), f"directions[{index}].skill", StateLoadError)),
        priority=_require_int(
            record.get("priority", 5),
            f"directions[{index}].priority",
            StateLoadError,
            minimum=1,
        ),
        note=_require_str(record.get("note", ""), f"directions[{index}].note", StateLoadError),
        created_at=_require_str(
            record.get("created_at", utc_now()),
            f"directions[{index}].created_at",
            StateLoadError,
        ),
    )


def _parse_improvement_record(data: object, *, index: int) -> ImprovementRecord:
    record = _require_mapping(data, f"improvement_history[{index}]", StateLoadError)
    return ImprovementRecord(
        skill=normalize_skill_name(
            _require_str(record.get("skill"), f"improvement_history[{index}].skill", StateLoadError)
        ),
        reason=_require_str(record.get("reason"), f"improvement_history[{index}].reason", StateLoadError),
        tokens_spent=_require_int(
            record.get("tokens_spent"),
            f"improvement_history[{index}].tokens_spent",
            StateLoadError,
            minimum=0,
        ),
        before_level=_require_int(
            record.get("before_level"),
            f"improvement_history[{index}].before_level",
            StateLoadError,
            minimum=0,
        ),
        after_level=_require_int(
            record.get("after_level"),
            f"improvement_history[{index}].after_level",
            StateLoadError,
            minimum=0,
        ),
        timestamp=_require_str(
            record.get("timestamp", utc_now()),
            f"improvement_history[{index}].timestamp",
            StateLoadError,
        ),
    )


def _parse_task_record(data: object, *, index: int) -> TaskRecord:
    record = _require_mapping(data, f"task_history[{index}]", StateLoadError)
    return TaskRecord(
        summary=_require_str(record.get("summary"), f"task_history[{index}].summary", StateLoadError),
        requested_skills=[
            normalize_skill_name(_require_str(item, f"task_history[{index}].requested_skills", StateLoadError))
            for item in _require_list(
                record.get("requested_skills", []),
                f"task_history[{index}].requested_skills",
                StateLoadError,
            )
        ],
        focus_skills=[
            normalize_skill_name(_require_str(item, f"task_history[{index}].focus_skills", StateLoadError))
            for item in _require_list(
                record.get("focus_skills", []),
                f"task_history[{index}].focus_skills",
                StateLoadError,
            )
        ],
        auto_training_spent=_require_int(
            record.get("auto_training_spent", 0),
            f"task_history[{index}].auto_training_spent",
            StateLoadError,
            minimum=0,
        ),
        confidence=_require_str(record.get("confidence", "low"), f"task_history[{index}].confidence", StateLoadError),
        timestamp=_require_str(
            record.get("timestamp", utc_now()),
            f"task_history[{index}].timestamp",
            StateLoadError,
        ),
    )


def _read_json_file(path: Path, error_type: type[TokimonPersistenceError]) -> object:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise error_type(f"unable to read {path}: {exc}") from exc

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise error_type(f"malformed JSON in {path}: {exc.msg}") from exc


def _rotate_backups(path: Path, backup_count: int) -> None:
    for index in range(backup_count, 0, -1):
        source = _backup_path(path, index)
        destination = _backup_path(path, index + 1)
        if not source.exists():
            continue
        if index == backup_count:
            source.unlink()
        else:
            source.replace(destination)


def _backup_path(path: Path, index: int) -> Path:
    return path.with_name(f"{path.name}.bak.{index}")


def _migration_list(data: dict[str, object], field_name: str) -> list[object]:
    value = data.get(field_name, [])
    return _require_list(value, field_name, StateLoadError)


def _migration_mapping(data: dict[str, object], field_name: str) -> dict[str, object]:
    value = data.get(field_name, {})
    return _require_mapping(value, field_name, StateLoadError)


def _require_mapping(
    value: object,
    field_name: str,
    error_type: type[TokimonPersistenceError],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise error_type(f"{field_name} must be an object")
    return value


def _require_list(
    value: object,
    field_name: str,
    error_type: type[TokimonPersistenceError],
) -> list[object]:
    if not isinstance(value, list):
        raise error_type(f"{field_name} must be a list")
    return value


def _require_str(
    value: object,
    field_name: str,
    error_type: type[TokimonPersistenceError],
) -> str:
    if not isinstance(value, str):
        raise error_type(f"{field_name} must be a string")
    return value


def _optional_str(
    value: object,
    field_name: str,
    error_type: type[TokimonPersistenceError],
) -> str | None:
    if value is None:
        return None
    return _require_str(value, field_name, error_type)


def _require_bool(
    value: object,
    field_name: str,
    error_type: type[TokimonPersistenceError],
) -> bool:
    if not isinstance(value, bool):
        raise error_type(f"{field_name} must be true or false")
    return value


def _require_int(
    value: object,
    field_name: str,
    error_type: type[TokimonPersistenceError],
    *,
    minimum: int | None = None,
) -> int:
    parsed = _coerce_int(value, field_name, error_type)
    if minimum is not None and parsed < minimum:
        raise error_type(f"{field_name} must be >= {minimum}")
    return parsed


def _coerce_int(
    value: object,
    field_name: str,
    error_type: type[TokimonPersistenceError],
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{field_name} must be an integer")
    return value
