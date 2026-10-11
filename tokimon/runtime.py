"""Runtime-root helpers for Tokimon checkouts and linked worktrees."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
import sysconfig


class RuntimePathError(ValueError):
    """Raised when a runtime path would violate repository runtime placement rules."""


@dataclass(frozen=True)
class RuntimePathContext:
    workspace_root: Path
    primary_workspace_root: Path
    linked_worktree: bool

    def tokimon_path(self, relative_path: Path) -> Path:
        if relative_path.is_absolute():
            raise RuntimePathError(f"Tokimon runtime path must be relative: {relative_path}")
        return self.primary_workspace_root / ".tokimon" / relative_path


def source_root() -> Path:
    """Return the source checkout or the installed distribution's workspace."""
    root = Path(__file__).resolve().parent.parent
    site_roots = {Path(sysconfig.get_path(kind)).resolve() for kind in ("purelib", "platlib")}
    if root not in site_roots:
        return root
    workspace = distribution_workspace_root()
    required = ("tokimon/__init__.py", "tokimon/settings.example.json", "skills/using-tokimon/skill.md", "docs/guidelines/repository-guidelines.md", "LICENSE", "THIRD_PARTY_NOTICES")
    missing = [name for name in required if not (workspace / name).is_file()]
    if missing:
        raise RuntimePathError(
            f"installed Tokimon workspace is incomplete at {workspace}; missing: {', '.join(missing)}. "
            "Use a complete Tokimon wheel in a writable virtual environment or run from a source checkout."
        )
    return workspace


def distribution_workspace_root() -> Path:
    return (Path(sys.prefix) / "share" / "tokimon").resolve(strict=False)


def python_executable(workspace_root: Path | str) -> Path:
    """Keep generated commands in the environment that owns their workspace."""
    owner = runtime_workspace_root(workspace_root)
    if owner == distribution_workspace_root():
        return Path(sys.executable).absolute()
    return owner / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def runtime_path_context(*, workspace_root: Path | str) -> RuntimePathContext:
    """Return resolved workspace/runtime ownership context."""
    workspace = Path(workspace_root).expanduser().resolve(strict=False)
    primary = _resolve_primary_workspace_root(workspace)
    return RuntimePathContext(
        workspace_root=workspace,
        primary_workspace_root=primary,
        linked_worktree=workspace != primary,
    )


def primary_worktree_root(workspace_root: Path | str) -> Path | None:
    """Return the primary checkout for a linked worktree, when detectable."""
    root = Path(workspace_root).expanduser().resolve(strict=False)
    primary = _resolve_primary_workspace_root(root)
    return primary if primary != root else None


def _resolve_primary_workspace_root(workspace_root: Path) -> Path:
    git_marker = workspace_root / ".git"
    if git_marker.is_dir():
        return workspace_root
    if not git_marker.is_file():
        return workspace_root

    git_dir = _read_gitdir(git_marker)
    common_dir = _read_common_dir(git_dir)
    if common_dir.name == ".git":
        return common_dir.parent.resolve(strict=False)
    parts = git_dir.resolve(strict=False).parts
    if len(parts) >= 3 and parts[-2] == "worktrees":
        common_git_dir = Path(*parts[:-2])
        if common_git_dir.name == ".git":
            return common_git_dir.parent.resolve(strict=False)
    return workspace_root


def _read_gitdir(git_marker: Path) -> Path:
    try:
        marker = git_marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimePathError(f"Unable to read gitdir marker {git_marker}: {exc}") from exc
    if not marker.startswith("gitdir:"):
        raise RuntimePathError(f"Invalid gitdir marker {git_marker}: expected 'gitdir:'")
    git_dir = Path(marker.split(":", 1)[1].strip()).expanduser()
    if not git_dir.is_absolute():
        git_dir = git_marker.parent / git_dir
    return git_dir.resolve(strict=False)


def _read_common_dir(git_dir: Path) -> Path:
    common_dir_file = git_dir / "commondir"
    if not common_dir_file.exists():
        return git_dir.resolve(strict=False)
    try:
        common_dir = Path(common_dir_file.read_text(encoding="utf-8").strip()).expanduser()
    except OSError as exc:
        raise RuntimePathError(f"Unable to read git commondir {common_dir_file}: {exc}") from exc
    if not common_dir.is_absolute():
        common_dir = git_dir / common_dir
    return common_dir.resolve(strict=False)


def _read_worktree_git_marker(gitdir_file: Path) -> Path:
    git_marker = Path(gitdir_file.read_text(encoding="utf-8").strip()).expanduser()
    if not git_marker.is_absolute():
        git_marker = gitdir_file.parent / git_marker
    return git_marker.resolve(strict=False)


def linked_worktree_roots(context: RuntimePathContext) -> tuple[Path, ...]:
    """Return linked worktree roots that must not own persistent runtime state."""
    roots = {context.workspace_root} if context.linked_worktree else set()
    worktrees_dir = context.primary_workspace_root / ".git" / "worktrees"
    if not worktrees_dir.is_dir():
        return tuple(sorted(roots, key=lambda path: path.as_posix()))

    for gitdir_file in worktrees_dir.glob("*/gitdir"):
        git_marker = _read_worktree_git_marker(gitdir_file)
        if git_marker.name != ".git":
            continue
        root = git_marker.parent.resolve(strict=False)
        if root != context.primary_workspace_root:
            roots.add(root)
    return tuple(sorted(roots, key=lambda path: path.as_posix()))


def reject_linked_worktree_runtime_override(
    runtime_path: Path,
    *,
    context: RuntimePathContext,
    env_var: str,
) -> None:
    """Reject env overrides that place persistent runtime state in linked worktrees."""
    for linked_root in linked_worktree_roots(context):
        if runtime_path.is_relative_to(linked_root):
            raise RuntimePathError(
                f"{env_var} points inside linked worktree {linked_root}: {runtime_path}. "
                f"Persistent Tokimon runtime state must live under the primary worktree "
                f"{context.primary_workspace_root} or another non-worktree path."
            )


def resolve_env_runtime_path(
    *,
    default_relative_path: Path,
    env_var: str,
    env: Mapping[str, str],
    workspace_root: Path | str,
) -> Path:
    """Resolve runtime env overrides while preserving linked-worktree runtime policy."""
    context = runtime_path_context(workspace_root=workspace_root)
    raw_override = env.get(env_var)
    if not raw_override:
        return context.tokimon_path(default_relative_path)

    override_path = Path(raw_override).expanduser()
    if not override_path.is_absolute():
        override_path = context.primary_workspace_root / override_path
    resolved = override_path.resolve(strict=False)
    reject_linked_worktree_runtime_override(resolved, context=context, env_var=env_var)
    return resolved


def runtime_workspace_root(workspace_root: Path | str) -> Path:
    """Return the checkout that owns persistent local runtime state."""
    root = Path(workspace_root).expanduser().resolve(strict=False)
    return primary_worktree_root(root) or root


def runtime_root(workspace_root: Path | str) -> Path:
    """Return Tokimon's ignored runtime root for this checkout."""
    return runtime_workspace_root(workspace_root) / ".tokimon"


def resolve_runtime_path(path: Path | str, *, workspace_root: Path | str) -> Path:
    """Resolve `.tokimon/...` paths against the runtime owner, others against the source checkout."""
    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return raw_path
    parts = raw_path.parts
    if parts and parts[0] == ".tokimon":
        return runtime_workspace_root(workspace_root) / raw_path
    return Path(workspace_root).expanduser() / raw_path
