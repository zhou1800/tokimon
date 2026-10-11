"""Stable local project identity for project-scoped durable memory."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class ProjectMemoryScope:
    """Resolved project identity without embedding an absolute path in the scope."""

    scope: str
    repository_name: str
    identity_kind: str

    def to_dict(self) -> dict[str, str]:
        return {
            "scope": self.scope,
            "repository_name": self.repository_name,
            "identity_kind": self.identity_kind,
        }


@dataclass(frozen=True)
class _ProjectIdentity:
    repository_root: Path
    identity_path: Path
    identity_kind: str


def _resolved_directory(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"project root does not exist: {candidate}") from exc
    if not resolved.is_dir():
        raise ValueError(f"project root must be a directory: {candidate}")
    return resolved


def _git_identity(root: Path) -> _ProjectIdentity | None:
    try:
        process = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "rev-parse",
                "--path-format=absolute",
                "--show-toplevel",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Git timed out while resolving project identity") from exc
    except OSError as exc:
        raise ValueError("unable to execute Git while resolving project identity") from exc
    if process.returncode != 0:
        stderr = process.stderr.strip().lower()
        if stderr.startswith("fatal: not a git repository"):
            return None
        raise ValueError("Git could not resolve project identity")
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError("Git returned an invalid project identity")
    repository_root = Path(lines[0]).resolve(strict=False)
    common_dir = Path(lines[1]).resolve(strict=False)
    return _ProjectIdentity(
        repository_root=repository_root,
        identity_path=common_dir,
        identity_kind="git-common-dir",
    )


def _project_identity(path: Path | str) -> _ProjectIdentity:
    root = _resolved_directory(path)
    git_identity = _git_identity(root)
    if git_identity is not None:
        return git_identity
    return _ProjectIdentity(
        repository_root=root,
        identity_path=root,
        identity_kind="directory",
    )


def _project_slug(name: str) -> str:
    slug = PROJECT_SLUG_RE.sub("-", name.strip().lower()).strip("-._")
    return (slug or "project")[:48]


def resolve_project_memory_scope(
    project_root: Path | str,
    *,
    tokimon_root: Path | str,
) -> ProjectMemoryScope:
    """Resolve one stable scope, sharing identity across linked Git worktrees."""

    project = _project_identity(project_root)
    tokimon = _project_identity(tokimon_root)
    repository_name = (
        project.identity_path.parent.name
        if project.identity_kind == "git-common-dir"
        else project.repository_root.name
    )
    if (
        project.identity_kind == tokimon.identity_kind
        and project.identity_path == tokimon.identity_path
    ):
        scope = "tokimon"
    else:
        identity = f"{project.identity_kind}:{project.identity_path}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        scope = f"project:{_project_slug(repository_name)}:{digest}"
    return ProjectMemoryScope(
        scope=scope,
        repository_name=repository_name,
        identity_kind=project.identity_kind,
    )
