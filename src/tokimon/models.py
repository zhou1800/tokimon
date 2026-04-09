from __future__ import annotations

from dataclasses import dataclass, field


def utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat()


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
            skill=str(data["skill"]),
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
            name=str(data["name"]),
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
            skill=str(data["skill"]),
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
            requested_skills=[str(item) for item in data.get("requested_skills", [])],
            focus_skills=[str(item) for item in data.get("focus_skills", [])],
            auto_training_spent=int(data.get("auto_training_spent", 0)),
            confidence=str(data.get("confidence", "low")),
            timestamp=str(data.get("timestamp", utc_now())),
        )


@dataclass(slots=True)
class TokimonState:
    version: int = 2
    available_tokens: int = 0
    total_tokens_eaten: int = 0
    improvement_cycles: int = 0
    idle_cycles: int = 0
    task_runs: int = 0
    quality_score: float = 0.0
    skills: dict[str, SkillRecord] = field(default_factory=dict)
    directions: list[Direction] = field(default_factory=list)
    improvement_history: list[ImprovementRecord] = field(default_factory=list)
    task_history: list[TaskRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "available_tokens": self.available_tokens,
            "total_tokens_eaten": self.total_tokens_eaten,
            "improvement_cycles": self.improvement_cycles,
            "idle_cycles": self.idle_cycles,
            "task_runs": self.task_runs,
            "quality_score": self.quality_score,
            "skills": {name: skill.to_dict() for name, skill in self.skills.items()},
            "directions": [direction.to_dict() for direction in self.directions],
            "improvement_history": [record.to_dict() for record in self.improvement_history],
            "task_history": [record.to_dict() for record in self.task_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TokimonState":
        skills = {
            str(name): SkillRecord.from_dict(record)
            for name, record in dict(data.get("skills", {})).items()
        }
        directions = [Direction.from_dict(item) for item in data.get("directions", [])]
        improvement_history = [
            ImprovementRecord.from_dict(item) for item in data.get("improvement_history", [])
        ]
        task_history = [TaskRecord.from_dict(item) for item in data.get("task_history", [])]
        return cls(
            version=int(data.get("version", 2)),
            available_tokens=int(data.get("available_tokens", 0)),
            total_tokens_eaten=int(data.get("total_tokens_eaten", 0)),
            improvement_cycles=int(data.get("improvement_cycles", 0)),
            idle_cycles=int(data.get("idle_cycles", 0)),
            task_runs=int(data.get("task_runs", 0)),
            quality_score=float(data.get("quality_score", 0.0)),
            skills=skills,
            directions=directions,
            improvement_history=improvement_history,
            task_history=task_history,
        )


@dataclass(slots=True)
class TaskAdvice:
    summary: str
    focus_skills: list[str]
    auto_training_spent: int
    approach: list[str]
    gaps: list[str]
    confidence: str
