from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import cli


def _write_lesson(root: Path, lesson_id: str, *, body: str) -> None:
    lessons_dir = root / "lessons"
    lessons_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"id": lesson_id}, sort_keys=True)
    (lessons_dir / f"lesson-{lesson_id}.md").write_text(f"{payload}\n---\n{body}\n")


def _lesson_ids_in_db(index_path: Path) -> set[str]:
    conn = sqlite3.connect(index_path)
    try:
        cursor = conn.execute("SELECT id FROM lessons ORDER BY id")
        return {row[0] for row in cursor.fetchall() if row and row[0]}
    finally:
        conn.close()


def test_memory_status_json_is_stable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "mem"
    root.mkdir()
    _write_lesson(root, "alpha", body="hello world")

    exit_code = cli.main(["memory", "status", "--root", str(root), "--json"])
    assert exit_code == 0
    first = capsys.readouterr().out

    exit_code = cli.main(["memory", "status", "--root", str(root), "--json"])
    assert exit_code == 0
    second = capsys.readouterr().out

    assert first == second
    payload = json.loads(first)
    assert payload["ok"] is True
    assert payload["lesson_files"] == 1
    assert isinstance(payload["indexed_lessons"], int)
    assert isinstance(payload["dirty"], bool)


def test_memory_search_query_flag_wins(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "mem"
    root.mkdir()
    _write_lesson(root, "positional", body="positional only")
    _write_lesson(root, "override", body="override only")

    exit_code = cli.main(["memory", "index", "--root", str(root), "--json"])
    assert exit_code == 0
    capsys.readouterr()

    exit_code = cli.main(["memory", "search", "positional", "--query", "override", "--root", str(root), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "override"
    assert payload["hits"] == ["override"]


def test_memory_search_missing_query_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "mem"
    root.mkdir()

    exit_code = cli.main(["memory", "search", "--root", str(root)])
    assert exit_code == 2
    capsys.readouterr()


def test_memory_index_creates_and_updates_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "mem"
    root.mkdir()
    _write_lesson(root, "alpha", body="alpha")

    index_path = root / "index.sqlite"
    assert index_path.exists() is False

    exit_code = cli.main(["memory", "index", "--root", str(root), "--json"])
    assert exit_code == 0
    capsys.readouterr()
    assert index_path.exists() is True
    assert _lesson_ids_in_db(index_path) == {"alpha"}

    _write_lesson(root, "beta", body="beta")
    exit_code = cli.main(["memory", "index", "--root", str(root), "--json"])
    assert exit_code == 0
    capsys.readouterr()
    assert _lesson_ids_in_db(index_path) == {"alpha", "beta"}


def test_memory_search_returns_expected_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "mem"
    root.mkdir()
    _write_lesson(root, "alpha", body="release checklist")
    _write_lesson(root, "beta", body="unrelated")
    _write_lesson(root, "gamma", body="checklist with extras")

    exit_code = cli.main(["memory", "index", "--root", str(root), "--json"])
    assert exit_code == 0
    capsys.readouterr()

    exit_code = cli.main(["memory", "search", "checklist", "--root", str(root), "--limit", "1", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hits"] == ["alpha"]


def test_memory_without_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["memory"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "tokimon memory" in out
    assert "Subcommands:" in out
    assert "status" in out
    assert "search" in out
