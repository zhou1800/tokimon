"""Public workflow-package declarations for memory maintenance."""

from __future__ import annotations

from pathlib import Path


WORKFLOW_ID = "memory-maintenance"
PYTHON_PACKAGE = "tokimon.workflow.memory_maintenance"
WORKFLOW_STATUS = "implemented"

CANONICAL_ENTRYPOINT = "tokimon.workflow.memory_maintenance.cli:main"
OPERATOR_ENTRYPOINT = ".venv/bin/python -m tokimon.workflow.memory_maintenance"

LOCAL_STATE_ROOT = Path(".tokimon") / "workflow" / WORKFLOW_ID
RUN_LOG_ROOT = LOCAL_STATE_ROOT / "runs"
BACKUP_ROOT = LOCAL_STATE_ROOT / "backups"

MEMORY_SCOPE = "memory"
MEMORY_ROOM = "maintenance"
DEFAULT_MEMORY_DB = Path(".tokimon") / "memory" / "memory.db"

MUTATION_COMMANDS = ("delete", "update-weight", "repair")
SNAPSHOT_COMMAND = "snapshot"

WORKFLOW_READINESS = {
    "id": WORKFLOW_ID,
    "workflow_status": WORKFLOW_STATUS,
    "entrypoint": OPERATOR_ENTRYPOINT,
    "modes": (
        {
            "id": "maintain-existing-memory",
            "role": "primary",
            "summary": "Memory maintenance can inspect existing Mem Palace state.",
            "memory_requirement": "readable",
            "required_tool_capabilities": (),
        },
    ),
}
