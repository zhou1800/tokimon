"""Workspace cloning, diffing, and merging utilities for self-improvement sessions."""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


@dataclass(frozen=True)
class WorkspaceChange:
    relpath: str
    kind: str  # "add" | "modify" | "delete"


@dataclass(frozen=True)
class GitMergeCandidate:
    branch: str
    commit: str


_WORKTREE_LOCK = threading.Lock()


def clone_master(master_root: Path, workspace_root: Path, include_paths: list[str]) -> None:
    if _can_use_git_worktree(master_root):
        try:
            _create_git_worktree(master_root, workspace_root)
            return
        except Exception:
            # Fall back to file copying when worktrees are unavailable.
            pass

    workspace_root.mkdir(parents=True, exist_ok=True)
    for rel in include_paths:
        src = master_root / rel
        dst = workspace_root / rel
        if not src.exists():
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=_ignore_cache_dirs)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def compute_changes(master_root: Path, workspace_root: Path, include_paths: list[str]) -> list[WorkspaceChange]:
    master_files = _collect_files(master_root, include_paths)
    workspace_files = _collect_files(workspace_root, include_paths)
    all_paths = sorted(set(master_files) | set(workspace_files))
    changes: list[WorkspaceChange] = []
    for relpath in all_paths:
        master_path = master_root / relpath
        session_path = workspace_root / relpath
        if master_path.exists() and not session_path.exists():
            changes.append(WorkspaceChange(relpath=relpath, kind="delete"))
            continue
        if session_path.exists() and not master_path.exists():
            changes.append(WorkspaceChange(relpath=relpath, kind="add"))
            continue
        if session_path.exists() and master_path.exists():
            if _file_digest(master_path) != _file_digest(session_path):
                changes.append(WorkspaceChange(relpath=relpath, kind="modify"))
    return changes


def can_use_git_merge(master_root: Path) -> bool:
    """Return True when master_root is a clean git toplevel suitable for conflict-aware merges."""
    return _can_use_git_worktree(master_root)


@contextlib.contextmanager
def merge_lock(master_root: Path, *, timeout_s: float = 600.0) -> contextlib.AbstractContextManager[None]:
    """Serialise merge operations across processes for a single git checkout."""
    lock_path = _merge_lock_path(master_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        if fcntl is None:
            yield
            return
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start >= timeout_s:
                    raise TimeoutError(f"Timed out waiting for merge lock: {lock_path}")
                time.sleep(0.1)
        try:
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def create_workspace_candidate_commit(workspace_root: Path, include_paths: list[str], message: str) -> str | None:
    """Commit winner changes inside the session workspace and return the commit hash."""
    try:
        inside = _git(workspace_root, ["rev-parse", "--is-inside-work-tree"], check=False)
    except Exception:
        return None
    if (inside.stdout or "").strip().lower() != "true":
        return None

    paths = sorted({path for path in include_paths if str(path).strip()})
    if not paths:
        return None
    _git(workspace_root, ["add", "-A", "--", *paths], check=False)
    diff = _git(workspace_root, ["diff", "--cached", "--name-only"], check=False)
    if not (diff.stdout or "").strip():
        return None

    commit_result = _git(
        workspace_root,
        [
            "-c",
            "user.email=tokimon@local",
            "-c",
            "user.name=Tokimon",
            "commit",
            "-m",
            message,
            "--no-gpg-sign",
        ],
        check=False,
    )
    if commit_result.returncode != 0:
        return None
    commit = (_git(workspace_root, ["rev-parse", "HEAD"], check=False).stdout or "").strip()
    return commit or None


def create_git_merge_candidate(
    master_root: Path,
    workspace_root: Path,
    changes: list[WorkspaceChange],
) -> GitMergeCandidate | None:
    """Create a temporary branch+commit capturing the winner changes.

    Returns None when there is no staged diff after applying changes.
    """
    token = uuid.uuid4().hex[:12]
    branch = f"tokimon/self-improve/candidate-{token}"
    worktree_path = Path(tempfile.mkdtemp(prefix="tokimon-merge-worktree-"))

    delete_branch_after = False
    with _WORKTREE_LOCK:
        _git(master_root, ["worktree", "prune"], check=False)
        _git(master_root, ["worktree", "add", "-b", branch, str(worktree_path), "HEAD"])

    try:
        _apply_changes(workspace_root, worktree_path, changes)
        paths = sorted({change.relpath for change in changes})
        if paths:
            _git(worktree_path, ["add", "-A", "--", *paths])

        diff_check = _git(worktree_path, ["diff", "--cached", "--name-only"])
        if not (diff_check.stdout or "").strip():
            delete_branch_after = True
            return None

        _git(
            worktree_path,
            [
                "-c",
                "user.email=tokimon@local",
                "-c",
                "user.name=Tokimon",
                "commit",
                "-m",
                "tokimon: self-improve candidate",
                "--no-gpg-sign",
            ],
        )
        commit = (_git(worktree_path, ["rev-parse", "HEAD"]).stdout or "").strip()
        if not commit:
            raise RuntimeError("candidate commit hash missing")
        return GitMergeCandidate(branch=branch, commit=commit)
    except Exception:
        delete_branch_after = True
        raise
    finally:
        with _WORKTREE_LOCK:
            _git(master_root, ["worktree", "remove", "-f", str(worktree_path)], check=False)
            _git(master_root, ["worktree", "prune"], check=False)
        _remove_path(worktree_path)
        if delete_branch_after:
            delete_branch(master_root, branch)


def squash_merge_candidate(
    master_root: Path, candidate: GitMergeCandidate, *, strategy_option: str | None = None
) -> subprocess.CompletedProcess[str]:
    return squash_merge_commit(master_root, candidate.commit, strategy_option=strategy_option)


def squash_merge_commit(
    master_root: Path, commit: str, *, strategy_option: str | None = None
) -> subprocess.CompletedProcess[str]:
    args = ["merge", "--squash"]
    if strategy_option:
        args.extend(["-X", strategy_option])
    args.append(commit)
    return _git(master_root, args, check=False)


def abort_squash_merge(master_root: Path) -> None:
    # `git merge --squash` does not create MERGE_HEAD; reset is the safe abort.
    _git(master_root, ["reset", "--hard", "HEAD"], check=False)


def commit_squash_merge(master_root: Path, message: str) -> subprocess.CompletedProcess[str]:
    return _git(
        master_root,
        [
            "-c",
            "user.email=tokimon@local",
            "-c",
            "user.name=Tokimon",
            "commit",
            "-m",
            message,
            "--no-gpg-sign",
        ],
        check=False,
    )


def delete_branch(master_root: Path, branch: str) -> None:
    _git(master_root, ["branch", "-D", branch], check=False)


def purge_bytecode_for_changes(root: Path, changes: list[WorkspaceChange]) -> None:
    for change in changes:
        _purge_bytecode_for_file(root, change.relpath)


def list_unmerged_paths(master_root: Path) -> list[str]:
    result = _git(master_root, ["diff", "--name-only", "--diff-filter=U"], check=False)
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def resolve_unmerged_paths(master_root: Path, commit: str, paths: list[str]) -> bool:
    """Resolve unmerged paths by preferring the given commit's version."""
    for relpath in paths:
        exists = _git(master_root, ["cat-file", "-e", f"{commit}:{relpath}"], check=False).returncode == 0
        if exists:
            _git(master_root, ["checkout", commit, "--", relpath], check=False)
            _git(master_root, ["add", "--", relpath], check=False)
        else:
            _git(master_root, ["rm", "-f", "--ignore-unmatch", "--", relpath], check=False)
    return not list_unmerged_paths(master_root)


def _collect_files(root: Path, include_paths: list[str]) -> set[str]:
    files: set[str] = set()
    for rel in include_paths:
        base = root / rel
        if not base.exists():
            continue
        if base.is_file():
            files.add(rel)
            continue
        for path in base.rglob("*"):
            if path.is_file() and not _is_ignored(root, path):
                files.add(str(path.relative_to(root)))
    return files


def _is_ignored(root: Path, path: Path) -> bool:
    try:
        relpath = path.relative_to(root)
    except ValueError:
        relpath = path
    parts = set(relpath.parts)
    if "__pycache__" in parts:
        return True
    if ".pytest_cache" in parts:
        return True
    if "runs" in parts:
        return True
    if "build" in parts or "dist" in parts:
        return True
    if any(part.endswith(".egg-info") for part in parts):
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    return False


def _ignore_cache_dirs(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in {"__pycache__", ".pytest_cache", "runs", "build", "dist"}:
            ignored.add(name)
        if name.endswith((".pyc", ".pyo")):
            ignored.add(name)
        if name.endswith(".egg-info"):
            ignored.add(name)
    return ignored


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _purge_bytecode_for_file(root: Path, relpath: str) -> None:
    if not relpath.endswith(".py"):
        return
    rel = Path(relpath)
    pycache_dir = (root / rel.parent / "__pycache__").resolve()
    try:
        pycache_dir.relative_to(root.resolve())
    except ValueError:
        return
    if not pycache_dir.exists() or not pycache_dir.is_dir():
        return
    stem = rel.stem
    for suffix in (".pyc", ".pyo"):
        for candidate in pycache_dir.glob(f"{stem}.*{suffix}"):
            candidate.unlink(missing_ok=True)


def _can_use_git_worktree(master_root: Path) -> bool:
    try:
        result = _git(master_root, ["rev-parse", "--show-toplevel"])
    except Exception:
        return False

    toplevel_raw = (result.stdout or "").strip()
    if not toplevel_raw:
        return False
    toplevel = Path(toplevel_raw).resolve()
    if toplevel != master_root.resolve():
        return False

    try:
        status = _git(master_root, ["status", "--porcelain"])
    except Exception:
        return False
    return not (status.stdout or "").strip()


def _create_git_worktree(master_root: Path, workspace_root: Path) -> None:
    workspace_root.parent.mkdir(parents=True, exist_ok=True)
    _remove_path(workspace_root)

    try:
        with _WORKTREE_LOCK:
            _git(master_root, ["worktree", "prune"], check=False)
            _git(master_root, ["worktree", "add", "--detach", "--no-checkout", str(workspace_root), "HEAD"])
        _git(workspace_root, ["reset", "--hard", "HEAD"])
    except Exception:
        with _WORKTREE_LOCK:
            _git(master_root, ["worktree", "remove", "-f", str(workspace_root)], check=False)
            _git(master_root, ["worktree", "prune"], check=False)
        _remove_path(workspace_root)
        raise


def _git(root: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=check,
    )


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _apply_changes(workspace_root: Path, target_root: Path, changes: list[WorkspaceChange]) -> None:
    for change in changes:
        dst = target_root / change.relpath
        if change.kind == "delete":
            if dst.exists():
                dst.unlink()
            continue
        src = workspace_root / change.relpath
        if not src.exists():
            raise FileNotFoundError(f"missing winner file: {change.relpath}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _merge_lock_path(master_root: Path) -> Path:
    git_dir = _git_dir(master_root)
    if git_dir is None:
        return (master_root / ".tokimon-tmp" / "self-improve-merge.lock").resolve()
    return (git_dir / "tokimon-self-improve-merge.lock").resolve()


def _git_dir(master_root: Path) -> Path | None:
    try:
        result = _git(master_root, ["rev-parse", "--git-dir"], check=False)
    except Exception:
        return None
    raw = (result.stdout or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (master_root / path).resolve()
    return path
