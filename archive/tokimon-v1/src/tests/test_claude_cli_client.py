from __future__ import annotations

import json
import subprocess
from pathlib import Path

import llm.client as client


def _install_fake_subprocess_run(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        captured["cwd"] = kwargs.get("cwd")
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "status": "SUCCESS",
                    "summary": "stub",
                    "artifacts": [],
                    "metrics": {},
                    "next_actions": [],
                    "failure_signature": "stub",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(client.subprocess, "run", fake_run)
    return captured


def test_claude_settings_prefers_claude_code_cli_env(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_CLI", "/path/to/claude")
    monkeypatch.setenv("TOKIMON_CLAUDE_CLI", "/path/to/other")
    monkeypatch.setenv("TOKIMON_CLAUDE_MODEL", "opus")
    monkeypatch.setenv("TOKIMON_CLAUDE_TIMEOUT_S", "123")
    monkeypatch.setenv("TOKIMON_CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS", "true")
    monkeypatch.setenv("TOKIMON_CLAUDE_SETTINGS_PATH", "/path/to/settings.json")
    monkeypatch.setenv("TOKIMON_CLAUDE_SETTINGS_JSON", '{"a": 1}')
    monkeypatch.setenv("TOKIMON_CLAUDE_ARGS", "--foo bar")

    settings = client.ClaudeCLISettings.from_env()
    assert settings.cli_command == "/path/to/claude"
    assert settings.model == "opus"
    assert settings.timeout_s == 123
    assert settings.dangerously_skip_permissions is True
    assert settings.settings_path == "/path/to/settings.json"
    assert settings.settings_json == {"a": 1}
    assert settings.extra_args == ["--foo", "bar"]


def test_claude_cli_preamble_includes_permissions_and_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SHELL", "/bin/bash")
    settings = client.ClaudeCLISettings(dangerously_skip_permissions=True)

    preamble = client._claude_cli_preamble(settings, tmp_path, delegation_depth=1)
    assert "<permissions instructions>" in preamble
    assert "provider: claude" in preamble
    assert "dangerously_skip_permissions: True" in preamble
    assert "<environment_context>" in preamble
    assert f"<cwd>{tmp_path}</cwd>" in preamble
    assert "<shell>bash</shell>" in preamble
    assert "<tokimon_context>" in preamble
    assert "<delegated>true</delegated>" in preamble
    assert "<delegation_depth>1</delegation_depth>" in preamble


def test_claude_client_sets_delegation_markers(monkeypatch, tmp_path):
    monkeypatch.delenv("TOKIMON_DELEGATED", raising=False)
    monkeypatch.delenv("TOKIMON_DELEGATION_DEPTH", raising=False)

    captured = _install_fake_subprocess_run(monkeypatch)

    cli = client.ClaudeCLIClient(
        workspace_dir=tmp_path,
        settings=client.ClaudeCLISettings(cli_command="claude"),
    )
    payload = cli.send(messages=[{"role": "user", "content": "hi"}])
    assert payload.get("status") == "SUCCESS"

    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("TOKIMON_DELEGATED") == "1"
    assert env.get("TOKIMON_DELEGATION_DEPTH") == "1"

    prompt = captured.get("input")
    assert isinstance(prompt, str)
    assert "provider: claude" in prompt
    assert "<delegation_depth>1</delegation_depth>" in prompt

    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    assert "--print" in cmd
    assert "--input-format" in cmd
    assert "text" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd

    assert captured.get("cwd") == str(tmp_path)


def test_claude_client_increments_delegation_depth(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKIMON_DELEGATION_DEPTH", "2")

    captured = _install_fake_subprocess_run(monkeypatch)

    cli = client.ClaudeCLIClient(
        workspace_dir=tmp_path,
        settings=client.ClaudeCLISettings(cli_command="claude"),
    )
    cli.send(messages=[{"role": "user", "content": "hi"}])

    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("TOKIMON_DELEGATION_DEPTH") == "3"

    prompt = captured.get("input")
    assert isinstance(prompt, str)
    assert "<delegation_depth>3</delegation_depth>" in prompt


def test_claude_client_writes_settings_json_to_tmp_root(monkeypatch, tmp_path):
    captured = _install_fake_subprocess_run(monkeypatch)

    cli = client.ClaudeCLIClient(
        workspace_dir=tmp_path,
        settings=client.ClaudeCLISettings(
            cli_command="claude",
            settings_json={"permission_mode": "default"},
        ),
    )
    cli.send(messages=[{"role": "user", "content": "hi"}])

    settings_file = tmp_path / ".tokimon-tmp" / "tokimon-claude-settings.json"
    assert settings_file.exists()
    assert json.loads(settings_file.read_text(encoding="utf-8")) == {"permission_mode": "default"}

    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    assert "--settings" in cmd
    settings_index = cmd.index("--settings") + 1
    assert cmd[settings_index] == str(settings_file)

