"""Adapter for read-only Claude Code CLI plugin tooling."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from shutil import which
from typing import Any


EXECUTABLE_ENV = "TOKIMON_CLAUDE_CLI"
REQUIRE_ENV = "TOKIMON_REQUIRE_CLAUDE_CLI"


def claude_executable() -> str:
    return os.environ.get(EXECUTABLE_ENV, "claude")


def require_claude_cli() -> bool:
    return os.environ.get(REQUIRE_ENV, "").strip().lower() in {"1", "true", "yes", "required"}


def validate_plugin(*, validation_root: Path, required: bool | None = None) -> dict[str, Any]:
    """Run read-only `claude plugin validate` against a plugin or marketplace root."""
    command = [claude_executable(), "plugin", "validate", str(validation_root)]
    return _run_command(name="claude-plugin-validator", command=command, required=required)


def plugin_list_command() -> list[str]:
    """Read-only installed-plugin inspection command for post-install readiness."""
    return [claude_executable(), "plugin", "list", "--json"]


def _run_command(
    *,
    name: str,
    command: list[str],
    required: bool | None,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    must_run = require_claude_cli() if required is None else required
    if which(command[0]) is None:
        return {
            "name": name,
            "ok": not must_run,
            "available": False,
            "skipped": True,
            "command": command,
            "message": f"missing Claude Code CLI executable: {command[0]}",
        }
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "ok": False,
            "available": True,
            "skipped": False,
            "command": command,
            "message": f"Claude Code CLI timed out after {timeout_sec}s",
            "stdout": exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or "",
            "stderr": exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or "",
        }
    return {
        "name": name,
        "ok": result.returncode == 0,
        "available": True,
        "skipped": False,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
