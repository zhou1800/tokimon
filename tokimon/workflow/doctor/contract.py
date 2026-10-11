"""Public workflow-package declarations for Tokimon doctor."""

from __future__ import annotations

from pathlib import Path


WORKFLOW_ID = "doctor"
PYTHON_PACKAGE = "tokimon.workflow.doctor"
WORKFLOW_STATUS = "implemented"

CANONICAL_ENTRYPOINT = "tokimon.workflow.doctor.cli:main"
OPERATOR_ENTRYPOINT = ".venv/bin/python -m tokimon.workflow.doctor"

LOCAL_STATE_ROOT = Path(".tokimon") / "workflow" / WORKFLOW_ID

MEMORY_SCOPE = "workflow"
MEMORY_ROOM = WORKFLOW_ID
DEFAULT_MEMORY_DB = Path(".tokimon") / "memory" / "memory.db"

REQUIRED_TOOL_CAPABILITIES: tuple[str, ...] = ()
COMMANDS = ("check", "contract")
STATUS_VALUES = ("pass", "warn", "fail")
READINESS_STATUS_VALUES = ("ready", "partial", "blocked")

WORKFLOW_READINESS = {
    "id": WORKFLOW_ID,
    "workflow_status": WORKFLOW_STATUS,
    "entrypoint": OPERATOR_ENTRYPOINT,
    "modes": (
        {
            "id": "readiness-report",
            "role": "primary",
            "summary": "Doctor readiness reporting can run.",
            "required_tool_capabilities": (),
        },
    ),
}
