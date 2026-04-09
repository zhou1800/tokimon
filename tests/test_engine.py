from tokimon.engine import add_direction, create_state, feed_tokens, prepare_for_task, run_idle_cycle


def test_idle_self_improvement_spends_available_tokens() -> None:
    state = create_state()
    feed_tokens(state, 3)

    records = run_idle_cycle(state, max_cycles=2)

    assert len(records) == 2
    assert state.available_tokens == 1
    assert state.total_tokens_eaten == 2
    assert state.skills["general reasoning"].level == 2


def test_directions_are_prioritized_during_idle_learning() -> None:
    state = create_state()
    feed_tokens(state, 2)
    add_direction(state, "python", priority=10, note="backend work")
    add_direction(state, "shell", priority=2, note="ops work")

    records = run_idle_cycle(state, max_cycles=1)

    assert len(records) == 1
    assert records[0].skill == "python"
    assert state.skills["python"].level == 1
    assert state.skills["shell"].level == 0


def test_task_preparation_uses_tokens_on_relevant_skills() -> None:
    state = create_state()
    feed_tokens(state, 3)
    add_direction(state, "testing", priority=9, note="quality")

    advice = prepare_for_task(
        state,
        summary="Build a small API and test plan",
        requested_skills=["python", "testing"],
        prep_budget=2,
    )

    assert advice.auto_training_spent == 2
    assert "testing" in advice.focus_skills
    assert "python" in advice.focus_skills
    assert state.available_tokens == 1
    assert state.task_runs == 1
    assert any("acceptance criteria" in step for step in advice.approach)
