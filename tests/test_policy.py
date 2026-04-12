from __future__ import annotations

import copy

import pytest

from tokimon.engine import add_direction, create_state, feed_tokens
from tokimon.models import SkillRecord
from tokimon.policy import (
    TaskIntent,
    assess_confidence,
    build_policy_inputs,
    choose_improvement_target,
    extract_task_skills,
    plan_task_preparation,
)


def test_improvement_target_precedence_prefers_requested_then_directions_then_existing_then_bootstrap() -> None:
    empty_state = create_state()
    bootstrap_decision = choose_improvement_target(build_policy_inputs(empty_state))
    assert bootstrap_decision.selected_improvement_target == "general reasoning"
    assert bootstrap_decision.decision_reason == "bootstrap"

    balanced_state = create_state()
    balanced_state.skills["zeta"] = SkillRecord(name="zeta", level=3)
    balanced_state.skills["alpha"] = SkillRecord(name="alpha", level=1)
    existing_decision = choose_improvement_target(build_policy_inputs(balanced_state))
    assert existing_decision.selected_improvement_target == "alpha"
    assert existing_decision.decision_reason == "balance-existing-skills"

    directed_state = create_state()
    directed_state.skills["alpha"] = SkillRecord(name="alpha", level=0)
    add_direction(directed_state, "shell", priority=9, note="ops")
    directed_decision = choose_improvement_target(build_policy_inputs(directed_state))
    assert directed_decision.selected_improvement_target == "shell"
    assert directed_decision.decision_reason == "directed-learning"

    preferred_decision = choose_improvement_target(
        build_policy_inputs(directed_state),
        preferred_skills=("testing",),
    )
    assert preferred_decision.selected_improvement_target == "testing"
    assert preferred_decision.decision_reason == "task-preparation"


def test_improvement_target_tie_breaking_is_deterministic() -> None:
    state = create_state()
    add_direction(state, "zeta", priority=7)
    add_direction(state, "alpha", priority=7)

    decision = choose_improvement_target(build_policy_inputs(state))

    assert decision.selected_improvement_target == "alpha"
    assert [candidate.skill for candidate in decision.ranked_candidates] == ["alpha", "zeta"]

    preferred_decision = choose_improvement_target(
        build_policy_inputs(state),
        preferred_skills=("zeta", "alpha"),
    )
    assert preferred_decision.selected_improvement_target == "alpha"
    assert [candidate.skill for candidate in preferred_decision.ranked_candidates] == ["alpha", "zeta"]


def test_task_preparation_budget_handles_zero_and_available_bounds() -> None:
    state = create_state()
    feed_tokens(state, 2)
    add_direction(state, "testing", priority=9, note="quality")

    zero_budget = plan_task_preparation(
        build_policy_inputs(
            state,
            task_intent=TaskIntent(summary="Ship tests", requested_skills=("testing",)),
            available_token_budget=0,
        )
    )
    assert zero_budget.preparation_spend_plan.requested_budget == 0
    assert zero_budget.preparation_spend_plan.granted_budget == 0
    assert zero_budget.preparation_spend_plan.planned_targets == ()
    assert zero_budget.preparation_spend_plan.reason == "no-prep-budget-requested"

    bounded_budget = plan_task_preparation(
        build_policy_inputs(
            state,
            task_intent=TaskIntent(summary="Ship tests", requested_skills=("testing",)),
            available_token_budget=5,
        )
    )
    assert bounded_budget.preparation_spend_plan.requested_budget == 5
    assert bounded_budget.preparation_spend_plan.granted_budget == 2
    assert bounded_budget.preparation_spend_plan.planned_targets == ("testing", "testing")
    assert bounded_budget.preparation_spend_plan.reason == "bounded-by-available-tokens"
    assert bounded_budget.allowed_capabilities == ("idle-learning", "task-preparation")


@pytest.mark.parametrize(
    ("levels", "expected_level", "expected_reason_fragment", "expected_gaps"),
    [
        ({"python": 5, "testing": 5}, "high", "high-confidence threshold", ()),
        ({"python": 3, "testing": 1}, "medium", "gaps remain in testing", ("testing",)),
        ({"python": 1, "testing": 0}, "low", "below the medium threshold", ("python", "testing")),
    ],
)
def test_confidence_thresholds_return_reasons_and_gaps(
    levels: dict[str, int],
    expected_level: str,
    expected_reason_fragment: str,
    expected_gaps: tuple[str, ...],
) -> None:
    assessment = assess_confidence(
        focus_skills=("python", "testing"),
        simulated_levels=levels,
    )

    assert assessment.level == expected_level
    assert expected_reason_fragment in assessment.reason
    assert assessment.gaps == expected_gaps


def test_policy_functions_do_not_mutate_state() -> None:
    state = create_state()
    feed_tokens(state, 3)
    add_direction(state, "python", priority=9, note="backend")
    add_direction(state, "testing", priority=7, note="quality")

    snapshot_before = copy.deepcopy(state.snapshot.to_dict())
    session_before = {
        "current_command_context": state.session.current_command_context,
        "caches": copy.deepcopy(state.session.caches),
        "derived_rankings": copy.deepcopy(state.session.derived_rankings),
        "temporary_approvals": copy.deepcopy(state.session.temporary_approvals),
        "daemon_state": copy.deepcopy(state.session.daemon_state),
        "quality_score": state.session.quality_score,
    }

    inputs = build_policy_inputs(
        state,
        task_intent=TaskIntent(
            summary="Build a tested Python API",
            requested_skills=("python", "testing"),
        ),
        available_token_budget=2,
    )
    choose_improvement_target(inputs, preferred_skills=("python", "testing"))
    extract_task_skills(inputs)
    plan_task_preparation(inputs)

    assert state.snapshot.to_dict() == snapshot_before
    assert state.session.current_command_context == session_before["current_command_context"]
    assert state.session.caches == session_before["caches"]
    assert state.session.derived_rankings == session_before["derived_rankings"]
    assert state.session.temporary_approvals == session_before["temporary_approvals"]
    assert state.session.daemon_state == session_before["daemon_state"]
    assert state.session.quality_score == session_before["quality_score"]
