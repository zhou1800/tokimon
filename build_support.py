"""Source-workspace inputs for Tokimon's Python distributions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import tomllib


WORKSPACE_DIRECTORY = "share/tokimon"
ROOT_FILES = (
    ".gitignore", "AGENTS.md", "README.md", "LICENSE", "THIRD_PARTY_NOTICES",
    "pyproject.toml", "setup.py", "build_support.py", "MANIFEST.in",
)
EXCLUDED_PARTS = frozenset({".git", ".venv", ".tokimon", "__pycache__"})


def workspace_data_files(source_root: Path) -> list[tuple[str, list[str]]]:
    """Build an install-prefix workspace from public source and declared assets."""
    config = tomllib.loads((source_root / "pyproject.toml").read_text(encoding="utf-8"))
    assets = config["tool"]["setuptools"]["package-data"]["tokimon"]
    selected = [source_root / name for name in ROOT_FILES]
    for directory in ("tokimon", "skills", "docs", "licenses", "tests", ".github"):
        parent = source_root / directory
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(f"distribution source directory must be regular: {parent}")
        for path in sorted(parent.rglob("*")):
            relative = path.relative_to(source_root)
            if EXCLUDED_PARTS.intersection(relative.parts) or path.suffix in {".pyc", ".pyo", ".pyd"}:
                continue
            if path.is_symlink():
                raise ValueError(f"distribution source must not contain symbolic links: {path}")
            if path.is_dir():
                continue
            if directory == "tokimon" and path.suffix != ".py" and not any(path.relative_to(parent).match(pattern) for pattern in assets):
                continue
            selected.append(path)
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in selected:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"distribution source must be a regular file: {path}")
        relative = path.relative_to(source_root)
        destination = (Path(WORKSPACE_DIRECTORY) / relative.parent).as_posix()
        grouped[destination].append(relative.as_posix())
    return [(directory, sorted(files)) for directory, files in sorted(grouped.items())]
