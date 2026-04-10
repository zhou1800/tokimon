from __future__ import annotations

from dataclasses import dataclass, field


CURRENT_SNAPSHOT_VERSION = 3
CURRENT_SETTINGS_VERSION = 1


def utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def normalize_skill_name(name: str) -> str:
    cleaned = " ".join(name.strip().lower().split())
    if not cleaned:
        raise ValueError("skill name cannot be empty")
    return cleaned


@dataclass(slots=True)
class Direction:
    skill: str
    priority: int = 5
    note: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, object]:
        return {
            "skill": self.skill,
            "priority": self.priority,
            "note": self.note,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Direction":
        return cls(
            skill=normalize_skill_name(str(data["skill"])),
            priority=int(data.get("priority", 5)),
            note=str(data.get("note", "")),
            created_at=str(data.get("created_at", utc_now())),
        )


@dataclass(slots=True)
class SkillRecord:
    name: str
    level: int = 0
    practice_runs: int = 0
    directed_cycles: int = 0
    last_improved_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "level": self.level,
            "practice_runs": self.practice_runs,
            "directed_cycles": self.directed_cycles,
            "last_improved_at": self.last_improved_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SkillRecord":
        return cls(
            name=normalize_skill_name(str(data["name"])),
            level=int(data.get("level", 0)),
            practice_runs=int(data.get("practice_runs", 0)),
            directed_cycles=int(data.get("directed_cycles", 0)),
            last_improved_at=str(data["last_improved_at"]) if data.get("last_improved_at") else None,
        )


@dataclass(slots=True)
class ImprovementRecord:
    skill: str
    reason: str
    tokens_spent: int
    before_level: int
    after_level: int
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, object]:
        return {
            "skill": self.skill,
            "reason": self.reason,
            "tokens_spent": self.tokens_spent,
            "before_level": self.before_level,
            "after_level": self.after_level,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ImprovementRecord":
        return cls(
            skill=normalize_skill_name(str(data["skill"])),
            reason=str(data["reason"]),
            tokens_spent=int(data["tokens_spent"]),
            before_level=int(data["before_level"]),
            after_level=int(data["after_level"]),
            timestamp=str(data.get("timestamp", utc_now())),
        )


@dataclass(slots=True)
class TaskRecord:
    summary: str
    requested_skills: list[str]
    focus_skills: list[str]
    auto_training_spent: int
    confidence: str
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "requested_skills": self.requested_skills,
            "focus_skills": self.focus_skills,
            "auto_training_spent": self.auto_training_spent,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TaskRecord":
        return cls(
            summary=str(data["summary"]),
            requested_skills=[normalize_skill_name(str(item)) for item in data.get("requested_skills", [])],
            focus_skills=[normalize_skill_name(str(item)) for item in data.get("focus_skills", [])],
            auto_training_spent=int(data.get("auto_training_spent", 0)),
            confidence=str(data.get("confidence", "low")),
            timestamp=str(data.get("timestamp", utc_now())),
        )


@dataclass(slots=True)
class TokimonSnapshot:
    version: int = CURRENT_SNAPSHOT_VERSION
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    available_tokens: int = 0
    total_tokens_eaten: int = 0
    improvement_cycles: int = 0
    idle_cycles: int = 0
    task_runs: int = 0
    skills: dict[str, SkillRecord] = field(default_factory=dict)
    directions: list[Direction] = field(default_factory=list)
    improvement_history: list[ImprovementRecord] = field(default_factory=list)
    task_history: list[TaskRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "available_tokens": self.available_tokens,
            "total_tokens_eaten": self.total_tokens_eaten,
            "improvement_cycles": self.improvement_cycles,
            "idle_cycles": self.idle_cycles,
            "task_runs": self.task_runs,
            "skills": {name: skill.to_dict() for name, skill in self.skills.items()},
            "directions": [direction.to_dict() for direction in self.directions],
            "improvement_history": [record.to_dict() for record in self.improvement_history],
            "task_history": [record.to_dict() for record in self.task_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TokimonSnapshot":
        skills = {
            normalize_skill_name(str(name)): SkillRecord.from_dict(record)
            for name, record in dict(data.get("skills", {})).items()
        }
        directions = [Direction.from_dict(item) for item in data.get("directions", [])]
        improvement_history = [
            ImprovementRecord.from_dict(item) for item in data.get("improvement_history", [])
        ]
        task_history = [TaskRecord.from_dict(item) for item in data.get("task_history", [])]
        return cls(
            version=int(data.get("version", CURRENT_SNAPSHOT_VERSION)),
            created_at=str(data.get("created_at", utc_now())),
            updated_at=str(data.get("updated_at", utc_now())),
            available_tokens=int(data.get("available_tokens", 0)),
            total_tokens_eaten=int(data.get("total_tokens_eaten", 0)),
            improvement_cycles=int(data.get("improvement_cycles", 0)),
            idle_cycles=int(data.get("idle_cycles", 0)),
            task_runs=int(data.get("task_runs", 0)),
            skills=skills,
            directions=directions,
            improvement_history=improvement_history,
            task_history=task_history,
        )


@dataclass(slots=True)
class TokimonSettings:
    version: int = CURRENT_SETTINGS_VERSION
    default_task_prep_budget: int = 3
    default_idle_max_cycles: int | None = None
    state_backup_count: int = 3
    allow_background_runtime: bool = False
    allow_cached_approvals: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "default_task_prep_budget": self.default_task_prep_budget,
            "default_idle_max_cycles": self.default_idle_max_cycles,
            "state_backup_count": self.state_backup_count,
            "allow_background_runtime": self.allow_background_runtime,
            "allow_cached_approvals": self.allow_cached_approvals,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TokimonSettings":
        idle_value = data.get("default_idle_max_cycles")
        return cls(
            version=int(data.get("version", CURRENT_SETTINGS_VERSION)),
            default_task_prep_budget=int(data.get("default_task_prep_budget", 3)),
            default_idle_max_cycles=None if idle_value is None else int(idle_value),
            state_backup_count=int(data.get("state_backup_count", 3)),
            allow_background_runtime=bool(data.get("allow_background_runtime", False)),
            allow_cached_approvals=bool(data.get("allow_cached_approvals", False)),
        )


@dataclass(slots=True)
class TokimonSessionState:
    current_command_context: str | None = None
    caches: dict[str, object] = field(default_factory=dict)
    derived_rankings: dict[str, list[str]] = field(default_factory=dict)
    temporary_approvals: set[str] = field(default_factory=set)
    daemon_state: dict[str, object] = field(default_factory=dict)
    quality_score: float = 0.0


@dataclass(slots=True)
class TokimonState:
    snapshot: TokimonSnapshot = field(default_factory=TokimonSnapshot)
    settings: TokimonSettings = field(default_factory=TokimonSettings)
    session: TokimonSessionState = field(default_factory=TokimonSessionState)

    @property
    def version(self) -> int:
        return self.snapshot.version

    @property
    def created_at(self) -> str:
        return self.snapshot.created_at

    @created_at.setter
    def created_at(self, value: str) -> None:
        self.snapshot.created_at = value

    @property
    def updated_at(self) -> str:
        return self.snapshot.updated_at

    @updated_at.setter
    def updated_at(self, value: str) -> None:
        self.snapshot.updated_at = value

    @property
    def available_tokens(self) -> int:
        return self.snapshot.available_tokens

    @available_tokens.setter
    def available_tokens(self, value: int) -> None:
        self.snapshot.available_tokens = value

    @property
    def total_tokens_eaten(self) -> int:
        return self.snapshot.total_tokens_eaten

    @total_tokens_eaten.setter
    def total_tokens_eaten(self, value: int) -> None:
        self.snapshot.total_tokens_eaten = value

    @property
    def improvement_cycles(self) -> int:
        return self.snapshot.improvement_cycles

    @improvement_cycles.setter
    def improvement_cycles(self, value: int) -> None:
        self.snapshot.improvement_cycles = value

    @property
    def idle_cycles(self) -> int:
        return self.snapshot.idle_cycles

    @idle_cycles.setter
    def idle_cycles(self, value: int) -> None:
        self.snapshot.idle_cycles = value

    @property
    def task_runs(self) -> int:
        return self.snapshot.task_runs

    @task_runs.setter
    def task_runs(self, value: int) -> None:
        self.snapshot.task_runs = value

    @property
    def skills(self) -> dict[str, SkillRecord]:
        return self.snapshot.skills

    @property
    def directions(self) -> list[Direction]:
        return self.snapshot.directions

    @directions.setter
    def directions(self, value: list[Direction]) -> None:
        self.snapshot.directions = value

    @property
    def improvement_history(self) -> list[ImprovementRecord]:
        return self.snapshot.improvement_history

    @improvement_history.setter
    def improvement_history(self, value: list[ImprovementRecord]) -> None:
        self.snapshot.improvement_history = value

    @property
    def task_history(self) -> list[TaskRecord]:
        return self.snapshot.task_history

    @task_history.setter
    def task_history(self, value: list[TaskRecord]) -> None:
        self.snapshot.task_history = value

    @property
    def quality_score(self) -> float:
        return self.session.quality_score

    @quality_score.setter
    def quality_score(self, value: float) -> None:
        self.session.quality_score = value


@dataclass(slots=True)
class TaskAdvice:
    summary: str
    focus_skills: list[str]
    auto_training_spent: int
    approach: list[str]
    gaps: list[str]
    confidence: str
