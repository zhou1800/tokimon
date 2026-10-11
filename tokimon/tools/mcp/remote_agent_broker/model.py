"""Shared Remote Agent Broker domain values and persisted request context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SHELL_RUNNER_NAME = "shell"
COPILOT_RUNNER_NAME = "copilot"
CLAUDE_RUNNER_NAME = "claude"
AGENT_RUNNER_NAMES = (COPILOT_RUNNER_NAME, CLAUDE_RUNNER_NAME)
RUNNER_NAMES = (SHELL_RUNNER_NAME, *AGENT_RUNNER_NAMES)

ACTIVE_STATUSES = frozenset({"queued", "running", "waiting_input"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "lost", "expired"})
JOB_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES
JOB_ID_RE = re.compile(r"^job_[0-9]{8}T[0-9]{6}Z_[a-f0-9]{12}$")

SECRET_KEY_RE = re.compile(
    r"(token|api[_-]?key|password|passwd|secret|cookie|authorization|auth[_-]?code)",
    re.IGNORECASE,
)
TOKEN_TEXT_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
)
KEY_VALUE_SECRET_RE = re.compile(
    r"(?i)\b(token|api[_-]?key|password|passwd|secret|cookie|authorization|auth[_-]?code)"
    r"(\s*[:=]\s*)['\"]?[^'\"\s]+"
)


class RemoteAgentBrokerError(RuntimeError):
    """Raised when the remote agent broker cannot complete an operation."""


@dataclass(frozen=True)
class RequestContext:
    """Correlation metadata propagated from one transport request into a job."""

    correlation_id: str
    request_id: str | None
    transport: str

    @classmethod
    def from_request(
        cls,
        *,
        request_id: object,
        correlation_id: str | None,
        transport: str,
    ) -> "RequestContext":
        normalized_request_id = None if request_id is None else str(request_id)
        return cls(
            correlation_id=(
                str(correlation_id).strip()
                if correlation_id is not None and str(correlation_id).strip()
                else normalized_request_id or "notification"
            ),
            request_id=normalized_request_id,
            transport=transport,
        )

    @classmethod
    def direct_job(cls, job_id: str) -> "RequestContext":
        return cls(correlation_id=job_id, request_id=None, transport="broker")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "transport": self.transport,
        }


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def cleanup_boundary_failure(
    *,
    ttl_days: int,
    path: Path,
    runtime_root: Path,
) -> dict[str, object]:
    return {
        "ok": False,
        "ttl_days": ttl_days,
        "removed": [],
        "removed_count": 0,
        "error": (
            "cleanup boundary failure: "
            f"{path} is outside the remote-agent runtime root {runtime_root}"
        ),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_for_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def validate_job_id(job_id: str) -> None:
    if not JOB_ID_RE.fullmatch(job_id):
        raise RemoteAgentBrokerError("Invalid job_id")


def require_positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RemoteAgentBrokerError(f"{name} must be a positive integer")
    return value


def redact_text(value: str) -> str:
    redacted = value
    for pattern in TOKEN_TEXT_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return KEY_VALUE_SECRET_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>",
        redacted,
    )


def redact_value(key: str, value: Any) -> Any:
    if SECRET_KEY_RE.search(key):
        return "<redacted>"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_value(str(item_key), item_value)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_value(key, item) for item in value]
    return value


def safe_options(options: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): redact_value(str(key), value) for key, value in options.items()}


def summarize_prompt(prompt: str, *, limit: int = 240) -> str:
    cleaned = " ".join(prompt.split())
    if len(cleaned) <= limit:
        return redact_text(cleaned)
    return redact_text(cleaned[: limit - 1] + "...")


def read_file_tail(path: Path, *, limit: int = 20000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")
