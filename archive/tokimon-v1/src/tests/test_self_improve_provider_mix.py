from __future__ import annotations

import pytest

from self_improve.provider_mix import mixed_provider_for_session, validate_mixed_sessions_per_batch


def test_mixed_provider_for_session_cycle() -> None:
    assert mixed_provider_for_session("1-1") == "claude"
    assert mixed_provider_for_session("1-2") == "codex"
    assert mixed_provider_for_session("1-5") == "codex"
    assert mixed_provider_for_session("1-6") == "claude"


def test_mixed_provider_for_session_invalid_falls_back_to_codex() -> None:
    assert mixed_provider_for_session("not-a-session") == "codex"
    assert mixed_provider_for_session("1-0") == "codex"
    assert mixed_provider_for_session("0-0") == "codex"


def test_validate_mixed_sessions_per_batch_requires_multiple_of_five() -> None:
    validate_mixed_sessions_per_batch(5)
    validate_mixed_sessions_per_batch(10)

    with pytest.raises(ValueError):
        validate_mixed_sessions_per_batch(0)

    with pytest.raises(ValueError):
        validate_mixed_sessions_per_batch(1)

    with pytest.raises(ValueError):
        validate_mixed_sessions_per_batch(6)

