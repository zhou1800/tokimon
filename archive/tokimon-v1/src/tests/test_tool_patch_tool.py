from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.patch_tool import PatchTool


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)


def test_patch_tool_errors_when_git_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)

    tool = PatchTool(tmp_path)
    result = tool.apply("diff --git a/a b/a\n")
    assert result.ok is False
    assert result.summary == "git not available"
    assert result.error == "git is required"


def test_patch_tool_rejects_invalid_patch(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is required for PatchTool")

    _init_git_repo(tmp_path)
    patch_text = (
        "diff --git a/missing.txt b/missing.txt\n"
        "--- a/missing.txt\n"
        "+++ b/missing.txt\n"
        "@@ -1 +1 @@\n"
        "-nope\n"
        "+yep\n"
    )
    result = PatchTool(tmp_path).apply(patch_text)
    assert result.ok is False
    assert result.summary == "patch validation failed"
    assert result.error == "patch check failed"


def test_patch_tool_applies_valid_patch(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is required for PatchTool")

    _init_git_repo(tmp_path)
    (tmp_path / "hello.txt").write_text("hello\n")
    patch_text = (
        "diff --git a/hello.txt b/hello.txt\n"
        "--- a/hello.txt\n"
        "+++ b/hello.txt\n"
        "@@ -1 +1 @@\n"
        "-hello\n"
        "+hi\n"
    )
    result = PatchTool(tmp_path).apply(patch_text)
    assert result.ok is True
    assert (tmp_path / "hello.txt").read_text() == "hi\n"


def test_patch_tool_repairs_mismatched_hunk_header_counts(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is required for PatchTool")

    _init_git_repo(tmp_path)
    (tmp_path / "hello.txt").write_text("hello\n")
    patch_text = (
        "diff --git a/hello.txt b/hello.txt\n"
        "--- a/hello.txt\n"
        "+++ b/hello.txt\n"
        "@@ -1,2 +1,3 @@\n"
        "-hello\n"
        "+hi\n"
    )
    result = PatchTool(tmp_path).apply(patch_text)
    assert result.ok is True
    assert result.data.get("normalized_hunk_headers") is True
    assert (tmp_path / "hello.txt").read_text() == "hi\n"
