"""PatchTool applies unified diffs with validation."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

from .base import ToolResult, elapsed_ms


_HUNK_HEADER_RE = re.compile(r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<suffix>.*)$")


def _coerce_hunk_count(raw: str | None) -> int:
    if raw is None:
        return 1
    try:
        return int(raw)
    except ValueError:
        return 1


def _normalize_unified_diff_hunk_headers(patch_text: str) -> tuple[str, bool]:
    """Repair unified diff hunk headers where the line counts don't match.

    Large-language models sometimes emit syntactically invalid unified diffs by
    providing incorrect hunk line counts (e.g., `@@ -1,5 +1,7 @@` for a 1-line edit).
    `git apply` rejects these diffs as "corrupt patch". We deterministically
    recompute the counts from the hunk body and rewrite the header when needed.
    """

    if not patch_text:
        return patch_text, False

    lines = patch_text.splitlines()
    changed = False
    out: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = _HUNK_HEADER_RE.match(line)
        if not match:
            out.append(line)
            idx += 1
            continue

        old_start = match.group("old_start")
        new_start = match.group("new_start")
        expected_old = _coerce_hunk_count(match.group("old_count"))
        expected_new = _coerce_hunk_count(match.group("new_count"))
        suffix = match.group("suffix") or ""

        idx += 1
        hunk_lines: list[str] = []
        while idx < len(lines):
            candidate = lines[idx]
            if candidate.startswith("diff --git "):
                break
            if _HUNK_HEADER_RE.match(candidate):
                break
            hunk_lines.append(candidate)
            idx += 1

        actual_old = sum(1 for raw in hunk_lines if raw.startswith((" ", "-")))
        actual_new = sum(1 for raw in hunk_lines if raw.startswith((" ", "+")))

        if actual_old != expected_old or actual_new != expected_new:
            changed = True
            out.append(f"@@ -{old_start},{actual_old} +{new_start},{actual_new} @@{suffix}")
        else:
            out.append(line)
        out.extend(hunk_lines)

    normalized = "\n".join(out)
    if patch_text.endswith("\n"):
        normalized += "\n"
    else:
        normalized = normalized.rstrip("\n")
    return normalized, changed


class PatchTool:
    name = "patch"

    def __init__(self, root: Path) -> None:
        self.root = root

    def apply(self, patch_text: str) -> ToolResult:
        start = time.perf_counter()
        if not shutil.which("git"):
            return ToolResult(ok=False, summary="git not available", data={}, elapsed_ms=elapsed_ms(start), error="git is required")
        normalized_patch, normalized = _normalize_unified_diff_hunk_headers(patch_text)
        try:
            check = subprocess.run(
                ["git", "apply", "--check", "-"],
                input=normalized_patch.encode(),
                cwd=self.root,
                capture_output=True,
                check=False,
            )
            if check.returncode != 0:
                return ToolResult(
                    ok=False,
                    summary="patch validation failed",
                    data={"stdout": check.stdout.decode(), "stderr": check.stderr.decode()},
                    elapsed_ms=elapsed_ms(start),
                    error="patch check failed",
                )
            apply = subprocess.run(
                ["git", "apply", "-"],
                input=normalized_patch.encode(),
                cwd=self.root,
                capture_output=True,
                check=False,
            )
            if apply.returncode != 0:
                return ToolResult(
                    ok=False,
                    summary="patch apply failed",
                    data={"stdout": apply.stdout.decode(), "stderr": apply.stderr.decode()},
                    elapsed_ms=elapsed_ms(start),
                    error="patch apply failed",
                )
            data: dict[str, object] = {}
            if normalized:
                data["normalized_hunk_headers"] = True
            return ToolResult(ok=True, summary="patch applied", data=data, elapsed_ms=elapsed_ms(start))
        except Exception as exc:
            return ToolResult(ok=False, summary="patch error", data={}, elapsed_ms=elapsed_ms(start), error=str(exc))
