"""GrepTool uses ripgrep if available."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from .base import ToolResult, elapsed_ms


_DEFAULT_EXCLUDED_DIR_NAMES: tuple[str, ...] = (
    "runs",
    ".tokimon-tmp",
    ".venv",
    "node_modules",
    "dist",
    "build",
)

_DEFAULT_EXCLUDED_SUFFIXES: tuple[str, ...] = (".jsonl", ".ndjson")

_DEFAULT_EXCLUDE_GLOBS: tuple[str, ...] = (
    *(f"--glob=!**/{dir_name}/**" for dir_name in _DEFAULT_EXCLUDED_DIR_NAMES),
    *(f"--glob=!**/*{suffix}" for suffix in _DEFAULT_EXCLUDED_SUFFIXES),
)

_DEFAULT_EXCLUDED_DIR_PARTS: frozenset[str] = frozenset(_DEFAULT_EXCLUDED_DIR_NAMES)

_DEFAULT_MAX_BYTES = 200_000


class GrepTool:
    name = "grep"

    def __init__(self, root: Path) -> None:
        self.root = root

    def search(self, pattern: str, path: str | None = None) -> ToolResult:
        start = time.perf_counter()
        max_bytes = _read_env_int("TOKIMON_GREP_MAX_BYTES", _DEFAULT_MAX_BYTES)
        apply_default_excludes = path is None
        target = self.root if path is None else (self.root / path)
        if shutil.which("rg"):
            cmd = ["rg"]
            if apply_default_excludes:
                cmd.extend(_DEFAULT_EXCLUDE_GLOBS)
            cmd.extend(["--", pattern, str(target)])
            returncode, stdout, truncated = _run_bounded(cmd, cwd=self.root, max_bytes=max_bytes)
            output = stdout.decode(errors="replace")
            ok = truncated or returncode in (0, 1)
            return ToolResult(
                ok=ok,
                summary="rg search (truncated)" if truncated else "rg search",
                data={"output": output, "truncated": truncated},
                elapsed_ms=elapsed_ms(start),
                error=None if ok else "rg error",
            )
        try:
            matches: list[str] = []
            truncated = False
            budget = max_bytes
            remaining = budget
            regex = re.compile(pattern)
            for file_path in target.rglob("*"):
                if truncated:
                    break
                if not file_path.is_file():
                    continue

                if apply_default_excludes:
                    if file_path.suffix in _DEFAULT_EXCLUDED_SUFFIXES:
                        continue
                    rel_parts = file_path.relative_to(target).parts
                    if any(part in _DEFAULT_EXCLUDED_DIR_PARTS for part in rel_parts):
                        continue

                content = file_path.read_text(errors="ignore")
                for line_no, line in enumerate(content.splitlines(), start=1):
                    if not regex.search(line):
                        continue

                    entry = f"{file_path}:{line_no}:{line}"
                    if budget > 0:
                        cost = len((entry + "\n").encode())
                        if cost > remaining:
                            truncated = True
                            break
                        remaining -= cost
                    matches.append(entry)
            output = "\n".join(matches)
            if budget > 0 and len(output.encode()) > budget:
                output = output.encode()[:budget].decode(errors="replace")
                truncated = True
            return ToolResult(
                ok=True,
                summary="fallback grep (truncated)" if truncated else "fallback grep",
                data={"output": output, "truncated": truncated},
                elapsed_ms=elapsed_ms(start),
            )
        except Exception as exc:
            return ToolResult(ok=False, summary="grep error", data={}, elapsed_ms=elapsed_ms(start), error=str(exc))


def _read_env_int(var_name: str, default: int) -> int:
    raw = os.environ.get(var_name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < 0:
        return default
    return value


def _run_bounded(cmd: list[str], *, cwd: Path, max_bytes: int) -> tuple[int, bytes, bool]:
    if max_bytes <= 0:
        completed = subprocess.run(cmd, cwd=cwd, capture_output=True, text=False, check=False)
        return completed.returncode, completed.stdout, False

    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = proc.stdout
    if stdout is None:
        proc.kill()
        proc.wait(timeout=5)
        return proc.returncode if proc.returncode is not None else 1, b"", False

    chunks: list[bytes] = []
    collected = 0
    truncated = False
    try:
        while True:
            chunk = stdout.read(8192)
            if not chunk:
                break
            remaining = max_bytes - collected
            if remaining <= 0:
                truncated = True
                proc.kill()
                break
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                collected += remaining
                truncated = True
                proc.kill()
                break
            chunks.append(chunk)
            collected += len(chunk)
    finally:
        try:
            stdout.close()
        except Exception:
            pass

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    returncode = proc.returncode if proc.returncode is not None else 1
    return returncode, b"".join(chunks), truncated
