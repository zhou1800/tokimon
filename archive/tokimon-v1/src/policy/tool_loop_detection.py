"""Tool-loop detection guardrails (opt-in).

Inspired by OpenClaw: provide deterministic, opt-in guardrails that stop runaway
repeated tool-call loops that do not make progress.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolLoopSettings:
    enabled: bool
    history_size: int
    repeat_threshold: int
    critical_threshold: int

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> "ToolLoopSettings":
        if env is None:
            env = os.environ  # pragma: no cover

        enabled = _parse_bool(env.get("TOKIMON_TOOL_LOOP_DETECTION_ENABLED", ""))
        history_size = _parse_int(env.get("TOKIMON_TOOL_LOOP_HISTORY_SIZE", ""), default=20, min_value=1, max_value=200)
        repeat_threshold = _parse_int(env.get("TOKIMON_TOOL_LOOP_REPEAT_THRESHOLD", ""), default=3, min_value=1, max_value=50)
        critical_threshold = _parse_int(
            env.get("TOKIMON_TOOL_LOOP_CRITICAL_THRESHOLD", ""), default=6, min_value=repeat_threshold, max_value=200
        )

        if critical_threshold < repeat_threshold:
            critical_threshold = repeat_threshold

        return ToolLoopSettings(
            enabled=enabled,
            history_size=history_size,
            repeat_threshold=repeat_threshold,
            critical_threshold=critical_threshold,
        )


@dataclass(frozen=True)
class ToolCallSignature:
    tool: str
    action: str
    args_hash: str

    def key(self) -> str:
        return f"{self.tool}:{self.action}:{self.args_hash}"


@dataclass(frozen=True)
class ToolLoopTrigger:
    reason: str  # "repeat_signature" | "repeat_failure"
    signature: ToolCallSignature
    count: int
    failure_count: int
    critical: bool


def stable_args_hash(args: dict[str, Any]) -> str:
    payload = _make_json_safe(args)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class ToolLoopDetector:
    def __init__(self, settings: ToolLoopSettings) -> None:
        self.settings = settings
        self._history: deque[ToolCallSignature] = deque(maxlen=max(1, int(settings.history_size)))
        self._counts: dict[str, int] = {}
        self._failure_counts: dict[str, int] = {}

    def record(self, signature: ToolCallSignature, *, ok: bool | None = None) -> ToolLoopTrigger | None:
        """Record a tool call occurrence and return a trigger when thresholds are hit.

        The detector is intentionally minimal: it triggers on repeated identical
        signatures, or repeated failures for the same signature.
        """

        key = signature.key()
        self._history.append(signature)

        count = self._counts.get(key, 0) + 1
        self._counts[key] = count

        failure_count = self._failure_counts.get(key, 0)
        if ok is False:
            failure_count += 1
            self._failure_counts[key] = failure_count

        repeat_threshold = max(1, int(self.settings.repeat_threshold))
        critical_threshold = max(repeat_threshold, int(self.settings.critical_threshold))

        if failure_count >= repeat_threshold:
            return ToolLoopTrigger(
                reason="repeat_failure",
                signature=signature,
                count=count,
                failure_count=failure_count,
                critical=failure_count >= critical_threshold,
            )

        if count >= repeat_threshold:
            return ToolLoopTrigger(
                reason="repeat_signature",
                signature=signature,
                count=count,
                failure_count=failure_count,
                critical=count >= critical_threshold,
            )

        return None

    def evidence(self, trigger: ToolLoopTrigger) -> dict[str, Any]:
        recent = list(self._history)
        return {
            "enabled": bool(self.settings.enabled),
            "history_size": int(self.settings.history_size),
            "repeat_threshold": int(self.settings.repeat_threshold),
            "critical_threshold": int(self.settings.critical_threshold),
            "trigger": {
                "reason": trigger.reason,
                "tool": trigger.signature.tool,
                "action": trigger.signature.action,
                "args_hash": trigger.signature.args_hash,
                "count": int(trigger.count),
                "failure_count": int(trigger.failure_count),
                "critical": bool(trigger.critical),
            },
            "recent_signatures": [
                {"tool": sig.tool, "action": sig.action, "args_hash": sig.args_hash} for sig in recent
            ],
        }


def normalize_signature(tool_name: str, action: str, args_hash: str) -> ToolCallSignature:
    return ToolCallSignature(
        tool=str(tool_name or "").strip().lower() or "<missing>",
        action=str(action or "").strip().lower() or "<missing>",
        args_hash=str(args_hash or "").strip().lower() or "<missing>",
    )


def _parse_bool(raw: str) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(raw: str, *, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(str(raw).strip())
    except ValueError:
        value = default
    value = max(min_value, value)
    value = min(max_value, value)
    return value


def _make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    return str(value)

