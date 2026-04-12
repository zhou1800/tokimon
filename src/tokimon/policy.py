from __future__ import annotations

import re
from dataclasses import dataclass, field

from tokimon.models import Direction, TokimonState, normalize_skill_name


DEFAULT_SKILL = "general reasoning"
DEFAULT_ALLOWED_CAPABILITIES = (
    "idle-learning",
    "task-preparation",
)


@dataclass(frozen=True, slots=True)
class SkillSnapshot:
    name: str
    level: int
    practice_runs: int = 0
    directed_cycles: int = 0


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    available_tokens: int
    total_tokens_eaten: int
    skills: tuple[SkillSnapshot, ...] = ()


@dataclass(frozen=True, slots=True)
class DirectedPriority:
    skill: str
    priority: int
    note: str = ""


@dataclass(frozen=True, slots=True)
class TaskIntent:
    summary: str
    requested_skills: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    available: tuple[str, ...] = DEFAULT_ALLOWED_CAPABILITIES
    blocked: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeModeFlags:
    allow_background_runtime: bool = False
    allow_cached_approvals: bool = False


@dataclass(frozen=True, slots=True)
class PolicyInputs:
    state: StateSnapshot
    task_intent: TaskIntent | None
    available_token_budget: int
    directed_priorities: tuple[DirectedPriority, ...] = ()
    capability_profile: CapabilityProfile = field(default_factory=CapabilityProfile)
    runtime_mode: RuntimeModeFlags = field(default_factory=RuntimeModeFlags)


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    skill: str
    priority: int
    current_level: int
    source: str


@dataclass(frozen=True, slots=True)
class ImprovementDecision:
    selected_improvement_target: str
    ranked_candidates: tuple[RankedCandidate, ...]
    decision_reason: str
    allowed_capabilities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparationSpendPlan:
    requested_budget: int
    granted_budget: int
    planned_targets: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class FocusContextPlan:
    relevant_skills: tuple[str, ...]
    focus_skills: tuple[str, ...]
    approach: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConfidenceAssessment:
    level: str
    score: float
    reason: str
    gaps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskPreparationDecision:
    selected_improvement_target: str | None
    ranked_candidates: tuple[RankedCandidate, ...]
    decision_reason: str
    preparation_spend_plan: PreparationSpendPlan
    focus_context_plan: FocusContextPlan
    allowed_capabilities: tuple[str, ...]
    confidence_assessment: ConfidenceAssessment


def snapshot_state(state: TokimonState) -> StateSnapshot:
    return StateSnapshot(
        available_tokens=state.available_tokens,
        total_tokens_eaten=state.total_tokens_eaten,
        skills=tuple(
            SkillSnapshot(
                name=skill.name,
                level=skill.level,
                practice_runs=skill.practice_runs,
                directed_cycles=skill.directed_cycles,
            )
            for skill in sorted(state.skills.values(), key=lambda item: item.name)
        ),
    )


def snapshot_directions(directions: list[Direction]) -> tuple[DirectedPriority, ...]:
    return tuple(
        DirectedPriority(skill=direction.skill, priority=direction.priority, note=direction.note)
        for direction in sorted(directions, key=lambda item: (item.skill, item.priority, item.note))
    )


def build_policy_inputs(
    state: TokimonState,
    *,
    task_intent: TaskIntent | None = None,
    available_token_budget: int = 0,
    capability_profile: CapabilityProfile | None = None,
    runtime_mode: RuntimeModeFlags | None = None,
) -> PolicyInputs:
    return PolicyInputs(
        state=snapshot_state(state),
        task_intent=task_intent,
        available_token_budget=available_token_budget,
        directed_priorities=snapshot_directions(state.directions),
        capability_profile=capability_profile or CapabilityProfile(),
        runtime_mode=runtime_mode or RuntimeModeFlags(),
    )


def resolve_allowed_capabilities(
    capability_profile: CapabilityProfile,
    runtime_mode: RuntimeModeFlags,
) -> tuple[str, ...]:
    blocked = set(capability_profile.blocked)
    allowed = [
        capability
        for capability in capability_profile.available
        if capability not in blocked
    ]
    if not runtime_mode.allow_background_runtime:
        allowed = [capability for capability in allowed if capability != "background-runtime"]
    if not runtime_mode.allow_cached_approvals:
        allowed = [capability for capability in allowed if capability != "cached-approvals"]
    return tuple(dict.fromkeys(allowed))


def directed_priority_for_skill(
    directed_priorities: tuple[DirectedPriority, ...],
    skill_name: str,
) -> int:
    normalized = normalize_skill_name(skill_name)
    for direction in directed_priorities:
        if direction.skill == normalized:
            return direction.priority
    return 0


def extract_task_skills(inputs: PolicyInputs) -> tuple[str, ...]:
    task_intent = inputs.task_intent
    if task_intent is None:
        raise ValueError("task_intent is required to extract task skills")

    relevant = {normalize_skill_name(skill) for skill in task_intent.requested_skills}
    summary_lower = task_intent.summary.lower()
    summary_terms = {
        term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", summary_lower) if len(term) > 2
    }

    for direction in inputs.directed_priorities:
        direction_terms = set(direction.skill.split())
        if direction.skill in summary_lower or direction_terms & summary_terms:
            relevant.add(direction.skill)

    for skill in inputs.state.skills:
        skill_terms = set(skill.name.split())
        if skill.name in summary_lower or skill_terms & summary_terms:
            relevant.add(skill.name)

    if not relevant and inputs.directed_priorities:
        ranked_directions = sorted(
            inputs.directed_priorities,
            key=lambda direction: (-direction.priority, direction.skill),
        )
        relevant.add(ranked_directions[0].skill)
    if not relevant:
        relevant.add(DEFAULT_SKILL)

    return tuple(sorted(relevant))


def choose_improvement_target(
    inputs: PolicyInputs,
    *,
    preferred_skills: tuple[str, ...] = (),
) -> ImprovementDecision:
    allowed_capabilities = resolve_allowed_capabilities(inputs.capability_profile, inputs.runtime_mode)
    skill_levels = _skill_levels(inputs.state.skills)

    normalized_preferred = tuple(
        sorted({normalize_skill_name(skill) for skill in preferred_skills})
    )
    if normalized_preferred:
        ranked_candidates = tuple(
            RankedCandidate(
                skill=skill,
                priority=directed_priority_for_skill(inputs.directed_priorities, skill),
                current_level=skill_levels.get(skill, 0),
                source="preferred-skill",
            )
            for skill in sorted(
                normalized_preferred,
                key=lambda skill: (
                    -directed_priority_for_skill(inputs.directed_priorities, skill),
                    skill_levels.get(skill, 0),
                    skill,
                ),
            )
        )
        return ImprovementDecision(
            selected_improvement_target=ranked_candidates[0].skill,
            ranked_candidates=ranked_candidates,
            decision_reason="task-preparation",
            allowed_capabilities=allowed_capabilities,
        )

    if inputs.directed_priorities:
        ranked_directions = tuple(
            RankedCandidate(
                skill=direction.skill,
                priority=direction.priority,
                current_level=skill_levels.get(direction.skill, 0),
                source="directed-priority",
            )
            for direction in sorted(
                inputs.directed_priorities,
                key=lambda direction: (-direction.priority, skill_levels.get(direction.skill, 0), direction.skill),
            )
        )
        return ImprovementDecision(
            selected_improvement_target=ranked_directions[0].skill,
            ranked_candidates=ranked_directions,
            decision_reason="directed-learning",
            allowed_capabilities=allowed_capabilities,
        )

    if inputs.state.skills:
        ranked_skills = tuple(
            RankedCandidate(
                skill=skill.name,
                priority=0,
                current_level=skill.level,
                source="existing-skill-balance",
            )
            for skill in sorted(inputs.state.skills, key=lambda item: (item.level, item.name))
        )
        return ImprovementDecision(
            selected_improvement_target=ranked_skills[0].skill,
            ranked_candidates=ranked_skills,
            decision_reason="balance-existing-skills",
            allowed_capabilities=allowed_capabilities,
        )

    bootstrap = RankedCandidate(
        skill=DEFAULT_SKILL,
        priority=0,
        current_level=0,
        source="bootstrap",
    )
    return ImprovementDecision(
        selected_improvement_target=bootstrap.skill,
        ranked_candidates=(bootstrap,),
        decision_reason="bootstrap",
        allowed_capabilities=allowed_capabilities,
    )


def plan_task_preparation(inputs: PolicyInputs) -> TaskPreparationDecision:
    if inputs.task_intent is None:
        raise ValueError("task_intent is required for task preparation")
    if inputs.available_token_budget < 0:
        raise ValueError("available_token_budget cannot be negative")

    allowed_capabilities = resolve_allowed_capabilities(inputs.capability_profile, inputs.runtime_mode)
    relevant_skills = extract_task_skills(inputs)
    initial_decision = choose_improvement_target(inputs, preferred_skills=relevant_skills)
    granted_budget = min(inputs.available_token_budget, inputs.state.available_tokens)
    simulated_levels = _skill_levels(inputs.state.skills)
    planned_targets: list[str] = []

    for skill in relevant_skills:
        simulated_levels.setdefault(skill, 0)

    for _ in range(granted_budget):
        simulated_inputs = _simulate_inputs(
            inputs=inputs,
            skill_levels=simulated_levels,
        )
        simulated_decision = choose_improvement_target(simulated_inputs, preferred_skills=relevant_skills)
        planned_targets.append(simulated_decision.selected_improvement_target)
        simulated_levels[simulated_decision.selected_improvement_target] += 1

    ranked_focus = sorted(
        relevant_skills,
        key=lambda skill: (
            -directed_priority_for_skill(inputs.directed_priorities, skill),
            -simulated_levels.get(skill, 0),
            skill,
        ),
    )
    focus_skills = tuple(ranked_focus[:3])
    average_level = (
        sum(simulated_levels.get(skill, 0) for skill in focus_skills) / max(1, len(focus_skills))
    )
    gaps = tuple(skill for skill in focus_skills if simulated_levels.get(skill, 0) < 2)
    confidence = assess_confidence(
        focus_skills=focus_skills,
        simulated_levels=simulated_levels,
        average_level=average_level,
        gaps=gaps,
    )

    approach = (
        f"Clarify the acceptance criteria for: {inputs.task_intent.summary}",
        f"Prioritize execution through: {', '.join(focus_skills)}",
        "Validate quality with tests, benchmarks, or explicit review criteria before calling the task done.",
    )
    spend_plan = PreparationSpendPlan(
        requested_budget=inputs.available_token_budget,
        granted_budget=granted_budget,
        planned_targets=tuple(planned_targets),
        reason=_spend_plan_reason(inputs.available_token_budget, granted_budget),
    )

    return TaskPreparationDecision(
        selected_improvement_target=planned_targets[0] if planned_targets else initial_decision.selected_improvement_target,
        ranked_candidates=initial_decision.ranked_candidates,
        decision_reason=initial_decision.decision_reason,
        preparation_spend_plan=spend_plan,
        focus_context_plan=FocusContextPlan(
            relevant_skills=relevant_skills,
            focus_skills=focus_skills,
            approach=approach,
        ),
        allowed_capabilities=allowed_capabilities,
        confidence_assessment=confidence,
    )


def assess_confidence(
    *,
    focus_skills: tuple[str, ...],
    simulated_levels: dict[str, int],
    average_level: float | None = None,
    gaps: tuple[str, ...] | None = None,
) -> ConfidenceAssessment:
    resolved_average = average_level
    if resolved_average is None:
        resolved_average = sum(simulated_levels.get(skill, 0) for skill in focus_skills) / max(1, len(focus_skills))

    resolved_gaps = gaps
    if resolved_gaps is None:
        resolved_gaps = tuple(skill for skill in focus_skills if simulated_levels.get(skill, 0) < 2)

    if resolved_average >= 5:
        level = "high"
        reason = f"average focus level {resolved_average:.2f} meets the high-confidence threshold"
    elif resolved_average >= 2:
        level = "medium"
        if resolved_gaps:
            reason = (
                f"average focus level {resolved_average:.2f} meets the medium threshold, "
                f"but gaps remain in {', '.join(resolved_gaps)}"
            )
        else:
            reason = f"average focus level {resolved_average:.2f} meets the medium-confidence threshold"
    else:
        level = "low"
        if resolved_gaps:
            reason = (
                f"average focus level {resolved_average:.2f} is below the medium threshold; "
                f"missing depth in {', '.join(resolved_gaps)}"
            )
        else:
            reason = f"average focus level {resolved_average:.2f} is below the medium-confidence threshold"

    return ConfidenceAssessment(
        level=level,
        score=round(resolved_average, 2),
        reason=reason,
        gaps=resolved_gaps,
    )


def _simulate_inputs(
    *,
    inputs: PolicyInputs,
    skill_levels: dict[str, int],
) -> PolicyInputs:
    return PolicyInputs(
        state=StateSnapshot(
            available_tokens=inputs.state.available_tokens,
            total_tokens_eaten=inputs.state.total_tokens_eaten,
            skills=tuple(
                SkillSnapshot(
                    name=skill,
                    level=level,
                )
                for skill, level in sorted(skill_levels.items())
            ),
        ),
        task_intent=inputs.task_intent,
        available_token_budget=inputs.available_token_budget,
        directed_priorities=inputs.directed_priorities,
        capability_profile=inputs.capability_profile,
        runtime_mode=inputs.runtime_mode,
    )


def _skill_levels(skills: tuple[SkillSnapshot, ...]) -> dict[str, int]:
    return {skill.name: skill.level for skill in skills}


def _spend_plan_reason(requested_budget: int, granted_budget: int) -> str:
    if requested_budget == 0:
        return "no-prep-budget-requested"
    if granted_budget == 0:
        return "no-prep-tokens-available"
    if granted_budget < requested_budget:
        return "bounded-by-available-tokens"
    return "full-prep-budget-applied"
