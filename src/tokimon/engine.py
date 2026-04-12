from __future__ import annotations

import math

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
from tokimon.policy import (
    CapabilityProfile,
    PolicyInputs,
    RuntimeModeFlags,
    TaskIntent,
    build_policy_inputs,
    choose_improvement_target as decide_improvement_target,
    directed_priority_for_skill,
    extract_task_skills as policy_extract_task_skills,
    plan_task_preparation,
)


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
    decision = decide_improvement_target(
        _policy_inputs(state),
        preferred_skills=tuple(preferred_skills or ()),
    )
    return decision.selected_improvement_target, decision.decision_reason


def skill_priority(state: TokimonState, skill_name: str) -> int:
    return directed_priority_for_skill(_policy_inputs(state).directed_priorities, skill_name)


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
        decision = decide_improvement_target(_policy_inputs(state))
        reason = (
            "idle"
            if decision.decision_reason in {"directed-learning", "balance-existing-skills", "bootstrap"}
            else decision.decision_reason
        )
        records.append(spend_token_on_improvement(state, decision.selected_improvement_target, reason))
        if cycles_remaining is not None:
            cycles_remaining -= 1
    return records


def extract_task_skills(state: TokimonState, summary: str, requested_skills: list[str] | None = None) -> list[str]:
    return list(
        policy_extract_task_skills(
            _policy_inputs(
                state,
                task_intent=TaskIntent(
                    summary=summary,
                    requested_skills=tuple(normalize_skill_name(skill) for skill in requested_skills or []),
                ),
            )
        )
    )


def prepare_for_task(
    state: TokimonState,
    summary: str,
    requested_skills: list[str] | None = None,
    prep_budget: int = 3,
) -> TaskAdvice:
    normalized_requested = sorted({normalize_skill_name(skill) for skill in requested_skills or []})
    decision = plan_task_preparation(
        _policy_inputs(
            state,
            task_intent=TaskIntent(
                summary=summary,
                requested_skills=tuple(normalized_requested),
            ),
            available_token_budget=prep_budget,
        )
    )
    _ensure_skills(state, decision.focus_context_plan.relevant_skills)

    for target_skill in decision.preparation_spend_plan.planned_targets:
        spend_token_on_improvement(state, target_skill, "task-preparation")

    state.task_runs += 1
    task_record = TaskRecord(
        summary=summary,
        requested_skills=normalized_requested,
        focus_skills=list(decision.focus_context_plan.focus_skills),
        auto_training_spent=decision.preparation_spend_plan.granted_budget,
        confidence=decision.confidence_assessment.level,
    )
    state.task_history.append(task_record)
    if len(state.task_history) > HISTORY_LIMIT:
        state.task_history = state.task_history[-HISTORY_LIMIT:]
    refresh_runtime_state(state)

    return TaskAdvice(
        summary=summary,
        focus_skills=list(decision.focus_context_plan.focus_skills),
        auto_training_spent=decision.preparation_spend_plan.granted_budget,
        approach=list(decision.focus_context_plan.approach),
        gaps=list(decision.confidence_assessment.gaps),
        confidence=decision.confidence_assessment.level,
    )


def _ensure_skills(state: TokimonState, skills: tuple[str, ...]) -> None:
    for skill in skills:
        get_skill(state, skill)


def _policy_inputs(
    state: TokimonState,
    *,
    task_intent: TaskIntent | None = None,
    available_token_budget: int = 0,
) -> PolicyInputs:
    return build_policy_inputs(
        state,
        task_intent=task_intent,
        available_token_budget=available_token_budget,
        capability_profile=_capability_profile(state),
        runtime_mode=_runtime_mode(state),
    )


def _capability_profile(state: TokimonState) -> CapabilityProfile:
    available = ["idle-learning", "task-preparation"]
    if state.settings.allow_background_runtime:
        available.append("background-runtime")
    if state.settings.allow_cached_approvals:
        available.append("cached-approvals")
    return CapabilityProfile(available=tuple(available))


def _runtime_mode(state: TokimonState) -> RuntimeModeFlags:
    return RuntimeModeFlags(
        allow_background_runtime=state.settings.allow_background_runtime,
        allow_cached_approvals=state.settings.allow_cached_approvals,
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
