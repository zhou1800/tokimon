"""FileTool for safe workspace read/write."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import ToolResult, elapsed_ms


class FileTool:
    name = "file"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Path traversal detected") from exc
        return candidate

    def read(self, path: str) -> ToolResult:
        start = time.perf_counter()
        try:
            file_path = self._resolve(path)
            content = file_path.read_text()
            return ToolResult(ok=True, summary="read ok", data={"path": str(file_path), "content": content}, elapsed_ms=elapsed_ms(start))
        except Exception as exc:
            return ToolResult(ok=False, summary="read failed", data={"path": path}, elapsed_ms=elapsed_ms(start), error=str(exc))

    def write(self, path: str, content: str) -> ToolResult:
        start = time.perf_counter()
        try:
            file_path = self._resolve(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return ToolResult(ok=True, summary="write ok", data={"path": str(file_path)}, elapsed_ms=elapsed_ms(start))
        except Exception as exc:
            return ToolResult(ok=False, summary="write failed", data={"path": path}, elapsed_ms=elapsed_ms(start), error=str(exc))
