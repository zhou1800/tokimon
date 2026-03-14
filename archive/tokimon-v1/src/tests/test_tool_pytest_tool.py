from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from tools.pytest_tool import PytestTool, _parse_counts, _parse_failures, _safe_cwd


def test_parse_counts_handles_passed_and_failed() -> None:
    output = "=== 2 failed, 5 passed in 0.12s ==="
    passed, failed = _parse_counts(output)
    assert passed == 5
    assert failed == 2


def test_parse_failures_extracts_failed_lines() -> None:
    output = "\n".join(
        [
            "FAILED test_a.py::test_one - AssertionError: boom",
            "some other line",
            "FAILED test_b.py::test_two - ValueError: nope",
        ]
    )
    assert _parse_failures(output) == [
        "test_a.py::test_one - AssertionError: boom",
        "test_b.py::test_two - ValueError: nope",
    ]


def test_safe_cwd_switches_to_parent_when_types_py_present(tmp_path: Path) -> None:
    assert _safe_cwd(tmp_path) == tmp_path
    (tmp_path / "types.py").write_text("# shadow stdlib types\n")
    assert _safe_cwd(tmp_path) == tmp_path.parent


def test_pytest_tool_run_parses_counts_and_failures(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, cwd, env, capture_output, text, check):
        assert "--basetemp" in cmd
        assert env.get("TMPDIR")
        return SimpleNamespace(
            returncode=1,
            stdout="2 passed, 1 failed\nFAILED tests/test_a.py::test_one - AssertionError: boom\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = PytestTool(tmp_path).run(["-q"])
    assert result.ok is False
    assert result.data["passed"] == 2
    assert result.data["failed"] == 1
    assert result.data["failing_tests"] == ["tests/test_a.py::test_one - AssertionError: boom"]
