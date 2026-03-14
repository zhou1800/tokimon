from __future__ import annotations

import json
from pathlib import Path

import pytest

import cli


def test_approvals_without_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["approvals"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "tokimon approvals" in out
    assert "Subcommands:" in out
    assert "list" in out
    assert "add" in out
    assert "remove" in out
    assert "clear" in out


def test_approvals_list_merges_env_and_file_with_source_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TOKIMON_TOOL_APPROVAL_ALLOWLIST", "env1, shared")

    al_dir = tmp_path / ".tokimon-tmp" / "approvals"
    al_dir.mkdir(parents=True)
    (al_dir / "allowlist.json").write_text(json.dumps({"allowlist": ["file1", "shared"]}, sort_keys=True), encoding="utf-8")

    exit_code = cli.main(["approvals", "list", "--json"])
    assert exit_code == 0
    first = capsys.readouterr().out
    payload = json.loads(first)

    assert payload["ok"] is True
    assert payload["workspace_root"] == str(tmp_path.resolve())
    assert payload["path"] == str(al_dir / "allowlist.json")
    assert payload["allowlist"] == [
        {"approval_id": "env1", "source": "env"},
        {"approval_id": "file1", "source": "file"},
        {"approval_id": "shared", "source": "env"},
    ]

    exit_code = cli.main(["approvals", "list", "--json"])
    assert exit_code == 0
    second = capsys.readouterr().out
    assert second == first


def test_approvals_add_is_idempotent_and_writes_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TOKIMON_TOOL_APPROVAL_ALLOWLIST", raising=False)

    exit_code = cli.main(["approvals", "add", "id1", "--json"])
    assert exit_code == 0
    first = capsys.readouterr().out

    file_path = tmp_path / ".tokimon-tmp" / "approvals" / "allowlist.json"
    assert json.loads(file_path.read_text(encoding="utf-8")) == {"allowlist": ["id1"]}

    exit_code = cli.main(["approvals", "add", "id1", "--json"])
    assert exit_code == 0
    second = capsys.readouterr().out
    assert second == first
    assert json.loads(file_path.read_text(encoding="utf-8")) == {"allowlist": ["id1"]}


def test_approvals_remove_noop_does_not_create_file_and_clear_empties(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TOKIMON_TOOL_APPROVAL_ALLOWLIST", raising=False)

    file_path = tmp_path / ".tokimon-tmp" / "approvals" / "allowlist.json"
    exit_code = cli.main(["approvals", "remove", "missing", "--json"])
    assert exit_code == 0
    capsys.readouterr()
    assert file_path.exists() is False

    exit_code = cli.main(["approvals", "add", "b", "--json"])
    assert exit_code == 0
    capsys.readouterr()

    exit_code = cli.main(["approvals", "add", "a", "--json"])
    assert exit_code == 0
    capsys.readouterr()
    assert json.loads(file_path.read_text(encoding="utf-8")) == {"allowlist": ["a", "b"]}

    exit_code = cli.main(["approvals", "clear", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["allowlist"] == []
    assert json.loads(file_path.read_text(encoding="utf-8")) == {"allowlist": []}


def test_approvals_remove_existing_updates_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TOKIMON_TOOL_APPROVAL_ALLOWLIST", raising=False)

    file_path = tmp_path / ".tokimon-tmp" / "approvals" / "allowlist.json"
    assert file_path.exists() is False

    exit_code = cli.main(["approvals", "add", "b", "--json"])
    assert exit_code == 0
    capsys.readouterr()

    exit_code = cli.main(["approvals", "add", "a", "--json"])
    assert exit_code == 0
    capsys.readouterr()
    assert json.loads(file_path.read_text(encoding="utf-8")) == {"allowlist": ["a", "b"]}

    exit_code = cli.main(["approvals", "remove", "a", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["allowlist"] == [{"approval_id": "b", "source": "file"}]
    assert json.loads(file_path.read_text(encoding="utf-8")) == {"allowlist": ["b"]}
