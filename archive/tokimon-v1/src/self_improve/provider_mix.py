"""Deterministic provider assignment helpers for self-improve sessions."""

from __future__ import annotations


_MIXED_PROVIDER_CYCLE: tuple[str, ...] = ("claude", "codex", "codex", "codex", "codex")


def validate_mixed_sessions_per_batch(sessions_per_batch: int) -> None:
    """Validate that mixed mode yields an exact `claude:codex=1:4` ratio."""

    if sessions_per_batch < 1:
        raise ValueError("sessions_per_batch must be >= 1")
    if sessions_per_batch % len(_MIXED_PROVIDER_CYCLE) != 0:
        raise ValueError(
            "mixed LLM mode requires --sessions to be a multiple of 5 to enforce an exact claude:codex=1:4 ratio"
        )


def mixed_provider_for_session(session_id: str) -> str:
    """Return `claude` or `codex` for the given self-improve session id.

    Deterministic schedule:
    - For session ids like `<batch>-<index>`, use numeric `<index>` (1-based).
    - Assign Claude when `(index - 1) % 5 == 0`, otherwise Codex.
    """

    try:
        index = int(str(session_id).split("-")[-1])
    except Exception:
        return "codex"
    if index < 1:
        return "codex"
    return _MIXED_PROVIDER_CYCLE[(index - 1) % len(_MIXED_PROVIDER_CYCLE)]

