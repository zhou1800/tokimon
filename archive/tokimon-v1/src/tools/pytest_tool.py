"""PytestTool runs pytest and parses results."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .base import ToolResult, elapsed_ms


class PytestTool:
    name = "pytest"

    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, args: list[str] | None = None, pytest_args: list[str] | None = None) -> ToolResult:
        start = time.perf_counter()
        normalized_args = args if args is not None else pytest_args
        if normalized_args is None:
            return ToolResult(
                ok=False,
                summary="pytest args missing",
                data={},
                elapsed_ms=elapsed_ms(start),
                error="expected args: list[str]",
            )
        cmd = [sys.executable, "-m", "pytest", *normalized_args]
        env = os.environ.copy()
        tmp_root = _ensure_tmp_root(self.root)
        if tmp_root is not None:
            env.update({"TMPDIR": str(tmp_root), "TEMP": str(tmp_root), "TMP": str(tmp_root)})
            if "--basetemp" not in normalized_args:
                cmd.extend(["--basetemp", str(tmp_root / f"pytest-{int(time.time() * 1000)}")])
        try:
            result = subprocess.run(
                cmd,
                cwd=_safe_cwd(self.root),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            output = result.stdout + "\n" + result.stderr
            passed, failed = _parse_counts(output)
            failing_tests = _parse_failures(output)
            return ToolResult(
                ok=result.returncode == 0,
                summary="pytest run",
                data={
                    "returncode": result.returncode,
                    "passed": passed,
                    "failed": failed,
                    "failing_tests": failing_tests,
                    "output": output,
                },
                elapsed_ms=elapsed_ms(start),
                error=None if result.returncode == 0 else "pytest failed",
            )
        except Exception as exc:
            return ToolResult(ok=False, summary="pytest error", data={}, elapsed_ms=elapsed_ms(start), error=str(exc))


def _safe_cwd(root: Path) -> Path:
    """Avoid stdlib shadowing when running from directories containing stdlib-like module names."""
    if (root / "types.py").exists():
        return root.parent
    return root


def _parse_counts(output: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d+)\s+passed", output)
    passed = int(match.group(1)) if match else None
    match = re.search(r"(\d+)\s+failed", output)
    failed = int(match.group(1)) if match else None
    return passed, failed


def _parse_failures(output: str) -> list[str]:
    failures = []
    for line in output.splitlines():
        if line.startswith("FAILED "):
            failures.append(line.split("FAILED ", 1)[1].strip())
    return failures


def _ensure_tmp_root(root: Path) -> Path | None:
    try:
        tmp_root = (root / ".tokimon-tmp").resolve()
        tmp_root.mkdir(parents=True, exist_ok=True)
        return tmp_root
    except Exception:
        return None
