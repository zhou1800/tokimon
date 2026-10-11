"""Conservative file ingestion helpers for Mem Palace."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TEXT_EXTENSIONS = (
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
)
CODE_EXTENSIONS = (".py", ".sh", ".js", ".ts", ".tsx", ".jsx")
IGNORED_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".tokimon",
    "__pycache__",
    "node_modules",
}


@dataclass(frozen=True)
class IngestDocument:
    path: Path
    relative_path: str
    text: str
    sha256: str
    byte_count: int

    def to_memory_kwargs(
        self,
        *,
        kind: str,
        scope: str,
        room: str,
        tags: Iterable[str],
        importance: float,
        confidence: float,
        max_detail_chars: int,
    ) -> dict[str, Any]:
        title = _title_from_text(self.text) or self.relative_path
        digest = self.sha256[:12]
        return {
            "kind": kind,
            "summary": f"Ingested {self.relative_path}: {_collapse(title, limit=96)}",
            "details": _collapse(self.text, preserve_lines=True, limit=max_detail_chars),
            "tags": [*tags, "ingested"],
            "source": f"ingest:{self.relative_path}#{digest}",
            "scope": scope,
            "room": room,
            "importance": importance,
            "confidence": confidence,
            "metadata": {
                "ingested_path": self.relative_path,
                "sha256": self.sha256,
                "bytes": self.byte_count,
                "chars": len(self.text),
            },
        }


def collect_documents(
    paths: Iterable[Path | str],
    *,
    workspace_root: Path | str,
    include_code: bool = False,
    max_file_bytes: int = 256_000,
) -> list[IngestDocument]:
    """Collect explicit readable text files without crawling ignored runtime trees."""

    root = Path(workspace_root).expanduser().resolve()
    documents: list[IngestDocument] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.is_dir():
            candidates = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            candidates = [path]
        for candidate in candidates:
            if _should_skip(candidate, root=root, include_code=include_code):
                continue
            document = _read_document(candidate, root=root, max_file_bytes=max_file_bytes)
            if document is not None:
                documents.append(document)
    return documents


def _should_skip(path: Path, *, root: Path, include_code: bool) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    if any(part in IGNORED_NAMES for part in relative.parts):
        return True
    suffix = path.suffix.lower()
    allowed = set(DEFAULT_TEXT_EXTENSIONS)
    if include_code:
        allowed.update(CODE_EXTENSIONS)
    return suffix not in allowed


def _read_document(path: Path, *, root: Path, max_file_bytes: int) -> IngestDocument | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    if stat.st_size <= 0 or stat.st_size > max_file_bytes:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    text = text.strip()
    if not text:
        return None
    try:
        relative_path = str(path.relative_to(root))
    except ValueError:
        relative_path = str(path)
    return IngestDocument(
        path=path,
        relative_path=relative_path,
        text=text,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
    )


def _title_from_text(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        return line
    return ""


def _collapse(text: str, *, limit: int, preserve_lines: bool = False) -> str:
    if preserve_lines:
        collapsed = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    else:
        collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 3)].rstrip() + "..."
