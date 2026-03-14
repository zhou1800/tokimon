from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import cli


def _write_run(root: Path, run_id: str, *, goal: str, ok: bool) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"goal": goal, "ok": ok}, sort_keys=True))
    return run_dir


def test_sessions_list_json_includes_expected_fields(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "self-improve"
    root.mkdir()
    run_dir = _write_run(root, "run-1", goal="Improve tokimon", ok=True)

    exit_code = cli.main(["sessions", "--root", str(root), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["root"] == str(root)
    assert payload["active_minutes"] is None
    assert payload["count"] == 1
    assert isinstance(payload["sessions"], list)

    session = payload["sessions"][0]
    assert session["id"] == "run-1"
    assert Path(session["path"]) == run_dir
    assert session["status"] == "ok"
    assert session["goal"] == "Improve tokimon"


def test_sessions_active_filters_by_recent_mtime(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "self-improve"
    root.mkdir()
    recent_dir = _write_run(root, "run-recent", goal="Recent", ok=True)
    old_dir = _write_run(root, "run-old", goal="Old", ok=False)

    now = time.time()
    recent_mtime = now - 60
    old_mtime = now - (60 * 60 * 5)
    os.utime(recent_dir / "run.json", (recent_mtime, recent_mtime))
    os.utime(old_dir / "run.json", (old_mtime, old_mtime))

    exit_code = cli.main(["sessions", "--root", str(root), "--active", "120", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["count"] == 1
    assert [session["id"] for session in payload["sessions"]] == ["run-recent"]
