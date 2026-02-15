from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from self_improve.workspace import clone_master


def test_clone_master_uses_git_worktree_when_clean_repo(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")

    master = tmp_path / "master"
    master.mkdir(parents=True, exist_ok=True)
    _git(master, ["init"])
    _git(master, ["config", "user.email", "test@example.com"])
    _git(master, ["config", "user.name", "Test User"])
    (master / "file.txt").write_text("hello\n")
    _git(master, ["add", "file.txt"])
    _git(master, ["commit", "-m", "init"])

    workspace = tmp_path / "workspace"
    clone_master(master, workspace, include_paths=["file.txt"])

    assert (workspace / "file.txt").read_text() == "hello\n"
    assert (workspace / ".git").exists() and (workspace / ".git").is_file()


def test_clone_master_falls_back_to_copy_when_repo_dirty(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")

    master = tmp_path / "master"
    master.mkdir(parents=True, exist_ok=True)
    _git(master, ["init"])
    _git(master, ["config", "user.email", "test@example.com"])
    _git(master, ["config", "user.name", "Test User"])
    (master / "file.txt").write_text("hello\n")
    _git(master, ["add", "file.txt"])
    _git(master, ["commit", "-m", "init"])

    (master / "file.txt").write_text("modified\n")
    workspace = tmp_path / "workspace"
    clone_master(master, workspace, include_paths=["file.txt"])

    assert (workspace / "file.txt").read_text() == "modified\n"
    assert not (workspace / ".git").exists()


def _git(cwd: Path, args: list[str]) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )

