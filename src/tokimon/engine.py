from __future__ import annotations

import math
import re

from tokimon.models import (
    Direction,
    ImprovementRecord,
    SkillRecord,
    TaskAdvice,
    TaskRecord,
    TokimonSettings,
    TokimonState,
    normalize_skill_name,
)


DEFAULT_SKILL = "general reasoning"
HISTORY_LIMIT = 50


def create_state(settings: TokimonSettings | None = None) -> TokimonState:
    state = TokimonState(settings=settings or TokimonSettings())
    refresh_runtime_state(state)
    return state


def refresh_quality_score(state: TokimonState) -> float:
    skill_values = list(state.skills.values())
    if not skill_values:
        state.quality_score = round(math.log1p(state.total_tokens_eaten), 2)
        return state.quality_score

    weighted_sum = sum(
        skill.level + skill.directed_cycles * 0.25 + skill.practice_runs * 0.1 for skill in skill_values
    )
    breadth_bonus = len(skill_values) * 0.35
    token_bonus = math.log1p(state.total_tokens_eaten)
    state.quality_score = round(weighted_sum / len(skill_values) + breadth_bonus + token_bonus, 2)
    return state.quality_score


def refresh_runtime_state(state: TokimonState) -> TokimonState:
    state.session.derived_rankings["skills_by_level"] = [
        skill.name for skill in sorted(state.skills.values(), key=lambda item: (-item.level, item.name))
    ]
    state.session.derived_rankings["directions_by_priority"] = [
        direction.skill
        for direction in sorted(state.directions, key=lambda item: (-item.priority, item.skill))
    ]
    state.session.caches["skill_levels"] = {skill.name: skill.level for skill in state.skills.values()}
    refresh_quality_score(state)
    return state


def get_skill(state: TokimonState, skill_name: str) -> SkillRecord:
    normalized = normalize_skill_name(skill_name)
    if normalized not in state.skills:
        state.skills[normalized] = SkillRecord(name=normalized)
    return state.skills[normalized]


def feed_tokens(state: TokimonState, tokens: int) -> int:
    if tokens <= 0:
        raise ValueError("tokens must be greater than zero")
    state.available_tokens += tokens
    refresh_runtime_state(state)
    return state.available_tokens


def add_direction(state: TokimonState, skill_name: str, priority: int = 5, note: str = "") -> Direction:
    skill = normalize_skill_name(skill_name)
    bounded_priority = max(1, min(priority, 10))
    for direction in state.directions:
        if direction.skill == skill:
            direction.priority = bounded_priority
            direction.note = note or direction.note
            get_skill(state, skill)
            refresh_runtime_state(state)
            return direction
    direction = Direction(skill=skill, priority=bounded_priority, note=note)
    state.directions.append(direction)
    get_skill(state, skill)
    refresh_runtime_state(state)
    return direction


def choose_improvement_target(
    state: TokimonState,
    preferred_skills: list[str] | None = None,
) -> tuple[str, str]:
    if preferred_skills:
        ranked_preferred = sorted(
            {normalize_skill_name(skill) for skill in preferred_skills},
            key=lambda skill: (skill_priority(state, skill), -get_skill(state, skill).level, skill),
            reverse=True,
        )
        if ranked_preferred:
            return ranked_preferred[0], "task-preparation"

    if state.directions:
        ranked_directions = sorted(
            state.directions,
            key=lambda direction: (direction.priority, -get_skill(state, direction.skill).level, direction.skill),
            reverse=True,
        )
        direction = ranked_directions[0]
        return direction.skill, "directed-learning"

    if state.skills:
        weakest_skill = min(state.skills.values(), key=lambda skill: (skill.level, skill.name))
        return weakest_skill.name, "balance-existing-skills"

    return DEFAULT_SKILL, "bootstrap"


def skill_priority(state: TokimonState, skill_name: str) -> int:
    normalized = normalize_skill_name(skill_name)
    for direction in state.directions:
        if direction.skill == normalized:
            return direction.priority
    return 0


def spend_token_on_improvement(state: TokimonState, skill_name: str, reason: str) -> ImprovementRecord:
    if state.available_tokens <= 0:
        raise ValueError("no tokens available")

    skill = get_skill(state, skill_name)
    before_level = skill.level
    skill.level += 1
    skill.practice_runs += 1
    if skill_priority(state, skill_name) > 0:
        skill.directed_cycles += 1

    record = ImprovementRecord(
        skill=skill.name,
        reason=reason,
        tokens_spent=1,
        before_level=before_level,
        after_level=skill.level,
    )
    skill.last_improved_at = record.timestamp

    state.available_tokens -= 1
    state.total_tokens_eaten += 1
    state.improvement_cycles += 1
    if reason == "idle":
        state.idle_cycles += 1

    state.improvement_history.append(record)
    if len(state.improvement_history) > HISTORY_LIMIT:
        state.improvement_history = state.improvement_history[-HISTORY_LIMIT:]
    refresh_runtime_state(state)
    return record


def run_idle_cycle(state: TokimonState, max_cycles: int | None = None) -> list[ImprovementRecord]:
    if max_cycles is not None and max_cycles <= 0:
        raise ValueError("max_cycles must be greater than zero")

    records: list[ImprovementRecord] = []
    cycles_remaining = max_cycles
    while state.available_tokens > 0 and (cycles_remaining is None or cycles_remaining > 0):
        target_skill, target_reason = choose_improvement_target(state)
        reason = "idle" if target_reason in {"directed-learning", "balance-existing-skills", "bootstrap"} else target_reason
        records.append(spend_token_on_improvement(state, target_skill, reason))
        if cycles_remaining is not None:
            cycles_remaining -= 1
    return records


def extract_task_skills(state: TokimonState, summary: str, requested_skills: list[str] | None = None) -> list[str]:
    relevant = {normalize_skill_name(skill) for skill in requested_skills or []}
    summary_terms = {term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", summary.lower()) if len(term) > 2}

    for direction in state.directions:
        direction_terms = set(direction.skill.split())
        if direction.skill in summary.lower() or direction_terms & summary_terms:
            relevant.add(direction.skill)

    for skill_name in state.skills:
        skill_terms = set(skill_name.split())
        if skill_name in summary.lower() or skill_terms & summary_terms:
            relevant.add(skill_name)

    if not relevant and state.directions:
        relevant.add(sorted(state.directions, key=lambda direction: direction.priority, reverse=True)[0].skill)
    if not relevant:
        relevant.add(DEFAULT_SKILL)

    return sorted(relevant)


def prepare_for_task(
    state: TokimonState,
    summary: str,
    requested_skills: list[str] | None = None,
    prep_budget: int = 3,
) -> TaskAdvice:
    if prep_budget < 0:
        raise ValueError("prep_budget cannot be negative")

    relevant_skills = extract_task_skills(state, summary=summary, requested_skills=requested_skills)
    auto_training_spent = 0
    while state.available_tokens > 0 and auto_training_spent < prep_budget:
        target_skill, _ = choose_improvement_target(state, preferred_skills=relevant_skills)
        spend_token_on_improvement(state, target_skill, "task-preparation")
        auto_training_spent += 1

    ranked_focus = sorted(
        relevant_skills,
        key=lambda skill: (
            skill_priority(state, skill),
            get_skill(state, skill).level,
            skill,
        ),
        reverse=True,
    )
    focus_skills = ranked_focus[:3]
    average_level = sum(get_skill(state, skill).level for skill in focus_skills) / max(1, len(focus_skills))
    if average_level >= 5:
        confidence = "high"
    elif average_level >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    gaps = [skill for skill in focus_skills if get_skill(state, skill).level < 2]
    approach = [
        f"Clarify the acceptance criteria for: {summary}",
        f"Prioritize execution through: {', '.join(focus_skills)}",
        "Validate quality with tests, benchmarks, or explicit review criteria before calling the task done.",
    ]

    state.task_runs += 1
    task_record = TaskRecord(
        summary=summary,
        requested_skills=sorted({normalize_skill_name(skill) for skill in requested_skills or []}),
        focus_skills=focus_skills,
        auto_training_spent=auto_training_spent,
        confidence=confidence,
    )
    state.task_history.append(task_record)
    if len(state.task_history) > HISTORY_LIMIT:
        state.task_history = state.task_history[-HISTORY_LIMIT:]
    refresh_runtime_state(state)

    return TaskAdvice(
        summary=summary,
        focus_skills=focus_skills,
        auto_training_spent=auto_training_spent,
        approach=approach,
        gaps=gaps,
        confidence=confidence,
    )


def format_status(state: TokimonState) -> str:
    lines = [
        "Tokimon v2",
        f"available_tokens: {state.available_tokens}",
        f"total_tokens_eaten: {state.total_tokens_eaten}",
        f"improvement_cycles: {state.improvement_cycles}",
        f"idle_cycles: {state.idle_cycles}",
        f"task_runs: {state.task_runs}",
        f"quality_score: {state.quality_score:.2f}",
    ]

    if state.directions:
        lines.append("directions:")
        for direction in sorted(state.directions, key=lambda item: item.priority, reverse=True):
            note_suffix = f" ({direction.note})" if direction.note else ""
            lines.append(f"- {direction.skill} [p{direction.priority}]{note_suffix}")
    else:
        lines.append("directions: none")

    if state.skills:
        lines.append("skills:")
        for skill in sorted(state.skills.values(), key=lambda item: (-item.level, item.name)):
            lines.append(
                f"- {skill.name}: level={skill.level} practice_runs={skill.practice_runs} directed_cycles={skill.directed_cycles}"
            )
    else:
        lines.append("skills: none")

    return "\n".join(lines)
