from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.grep_tool import GrepTool


class _FakePopen:
    def __init__(self, cmd, *, output: bytes, returncode: int) -> None:
        self.cmd = cmd
        self.stdout = io.BytesIO(output)
        self.returncode = returncode
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_grep_tool_fallback_search_finds_matches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)

    file_path = tmp_path / "hay.txt"
    file_path.write_text("hay\nneedle here\n")

    result = GrepTool(tmp_path).search("needle")
    assert result.ok is True
    assert result.data["truncated"] is False
    assert str(file_path) in result.data["output"]
    assert ":2:needle here" in result.data["output"]


def test_grep_tool_uses_rg_when_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rg" if name == "rg" else None)

    def fake_popen(cmd: list[str], cwd: Path, stdout: Any, stderr: Any) -> _FakePopen:
        assert cmd[0] == "rg"
        assert str(tmp_path) in cmd
        return _FakePopen(cmd, output=b"match\n", returncode=0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = GrepTool(tmp_path).search("pattern")
    assert result.ok is True
    assert result.data["output"] == "match\n"
    assert result.data["truncated"] is False


def test_grep_tool_rg_no_matches_is_ok(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rg" if name == "rg" else None)

    def fake_popen(cmd: list[str], cwd: Path, stdout: Any, stderr: Any) -> _FakePopen:
        return _FakePopen(cmd, output=b"", returncode=1)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = GrepTool(tmp_path).search("pattern")
    assert result.ok is True
    assert result.data["truncated"] is False


def test_grep_tool_repo_wide_search_applies_default_excludes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rg" if name == "rg" else None)

    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], cwd: Path, stdout: Any, stderr: Any) -> _FakePopen:
        calls.append(cmd)
        return _FakePopen(cmd, output=b"", returncode=1)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    GrepTool(tmp_path).search("pattern")
    GrepTool(tmp_path).search("pattern", path="runs")

    assert len(calls) == 2
    repo_wide, targeted = calls
    assert "--glob=!**/runs/**" in repo_wide
    assert "--glob=!**/runs/**" not in targeted


def test_grep_tool_rg_respects_max_bytes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rg" if name == "rg" else None)
    monkeypatch.setenv("TOKIMON_GREP_MAX_BYTES", "5")

    captured: dict[str, object] = {}

    def fake_popen(cmd: list[str], cwd: Path, stdout: Any, stderr: Any) -> _FakePopen:
        proc = _FakePopen(cmd, output=b"0123456789\n", returncode=0)
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = GrepTool(tmp_path).search("pattern")
    assert result.ok is True
    assert result.data["truncated"] is True
    assert len(result.data["output"].encode()) <= 5

    proc = captured.get("proc")
    assert isinstance(proc, _FakePopen)
    assert proc.killed is True


def test_grep_tool_fallback_respects_max_bytes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("root")
    root.mkdir()
    (root / "hay.txt").write_text("needle\nneedle\nneedle\n", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda _: None)
    monkeypatch.setenv("TOKIMON_GREP_MAX_BYTES", "30")

    result = GrepTool(root).search("needle")
    assert result.ok is True
    assert result.data["truncated"] is True
    assert len(result.data["output"].encode()) <= 30
