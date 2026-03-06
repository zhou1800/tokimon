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

        try:
            idx = cmd.index("--output-last-message")
        except ValueError as exc:  # pragma: no cover
            raise AssertionError("Codex CLI args missing --output-last-message") from exc

        out_path = Path(cmd[idx + 1])
        out_path.write_text(
            json.dumps(
                {
                    "status": "SUCCESS",
                    "summary": "stub",
                    "artifacts": [],
                    "metrics": {},
                    "next_actions": [],
                    "failure_signature": "stub",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(client.subprocess, "run", fake_run)
    return captured


def test_codex_client_sets_delegation_markers(monkeypatch, tmp_path):
    monkeypatch.delenv("TOKIMON_DELEGATED", raising=False)
    monkeypatch.delenv("TOKIMON_DELEGATION_DEPTH", raising=False)

    captured = _install_fake_subprocess_run(monkeypatch)

    cli = client.CodexCLIClient(
        workspace_dir=tmp_path,
        settings=client.CodexCLISettings(cli_command="codex"),
    )
    cli.send(messages=[{"role": "user", "content": "hi"}])

    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("TOKIMON_DELEGATED") == "1"
    assert env.get("TOKIMON_DELEGATION_DEPTH") == "1"


def test_codex_client_increments_delegation_depth(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKIMON_DELEGATION_DEPTH", "2")

    captured = _install_fake_subprocess_run(monkeypatch)

    cli = client.CodexCLIClient(
        workspace_dir=tmp_path,
        settings=client.CodexCLISettings(cli_command="codex"),
    )
    cli.send(messages=[{"role": "user", "content": "hi"}])

    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("TOKIMON_DELEGATED") == "1"
    assert env.get("TOKIMON_DELEGATION_DEPTH") == "3"


def test_guard_disabled_does_not_override_ripgrep_config(monkeypatch, tmp_path):
    base_config = tmp_path / "base.ripgreprc"
    base_config.write_text("# base-config\n", encoding="utf-8")

    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(base_config))
    monkeypatch.setenv("TOKIMON_CODEX_RIPGREP_GUARD", "0")
    monkeypatch.delenv("TOKIMON_CODEX_RIPGREP_MAX_COLUMNS", raising=False)

    captured = _install_fake_subprocess_run(monkeypatch)

    cli = client.CodexCLIClient(
        workspace_dir=tmp_path,
        settings=client.CodexCLISettings(cli_command="codex"),
    )
    cli.send(messages=[{"role": "user", "content": "hi"}])

    guard_path = tmp_path / ".tokimon-tmp" / "tokimon-codex.ripgreprc"
    assert not guard_path.exists()

    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("RIPGREP_CONFIG_PATH") == str(base_config)


def test_guard_enabled_by_default_creates_guard_and_preserves_base(monkeypatch, tmp_path):
    base_config = tmp_path / "base.ripgreprc"
    base_marker = "BASE-RIPGREP-CONFIG"
    base_config.write_text(f"{base_marker}\n--hidden\n", encoding="utf-8")

    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(base_config))
    monkeypatch.delenv("TOKIMON_CODEX_RIPGREP_GUARD", raising=False)
    monkeypatch.delenv("TOKIMON_CODEX_RIPGREP_MAX_COLUMNS", raising=False)

    captured = _install_fake_subprocess_run(monkeypatch)

    cli = client.CodexCLIClient(
        workspace_dir=tmp_path,
        settings=client.CodexCLISettings(cli_command="codex"),
    )
    cli.send(messages=[{"role": "user", "content": "hi"}])

    guard_path = tmp_path / ".tokimon-tmp" / "tokimon-codex.ripgreprc"
    assert guard_path.exists()

    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("RIPGREP_CONFIG_PATH") == str(guard_path)

    content = guard_path.read_text(encoding="utf-8")

    expected_globs = [
        "--glob=!**/runs/**",
        "--glob=!**/.tokimon-tmp/**",
        "--glob=!**/.venv/**",
        "--glob=!**/node_modules/**",
        "--glob=!**/dist/**",
        "--glob=!**/build/**",
        "--glob=!**/*.jsonl",
        "--glob=!**/*.ndjson",
    ]
    for line in expected_globs:
        assert line in content

    assert "--max-columns=4096" in content
    assert "--max-columns-preview" in content

    first_glob_pos = content.index("--glob=")
    assert content.index(base_marker) < first_glob_pos


def test_max_columns_disable_omits_flags(monkeypatch, tmp_path):
    base_config = tmp_path / "base.ripgreprc"
    base_config.write_text("BASE\n", encoding="utf-8")

    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(base_config))
    monkeypatch.setenv("TOKIMON_CODEX_RIPGREP_MAX_COLUMNS", "0")
    monkeypatch.delenv("TOKIMON_CODEX_RIPGREP_GUARD", raising=False)

    captured = _install_fake_subprocess_run(monkeypatch)

    cli = client.CodexCLIClient(
        workspace_dir=tmp_path,
        settings=client.CodexCLISettings(cli_command="codex"),
    )
    cli.send(messages=[{"role": "user", "content": "hi"}])

    guard_path = tmp_path / ".tokimon-tmp" / "tokimon-codex.ripgreprc"
    assert guard_path.exists()

    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("RIPGREP_CONFIG_PATH") == str(guard_path)

    content = guard_path.read_text(encoding="utf-8")
    assert "--max-columns=" not in content
    assert "--max-columns-preview" not in content


def test_codex_client_retries_unsupported_requested_model_with_default(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        try:
            idx = cmd.index("--output-last-message")
            model_idx = cmd.index("--model")
        except ValueError as exc:  # pragma: no cover
            raise AssertionError("Codex CLI args missing required flags") from exc

        out_path = Path(cmd[idx + 1])
        model = cmd[model_idx + 1]
        if model == "gpt-5.3-codex-spark":
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr=(
                    'ERROR: {"type":"error","status":400,"error":{"type":"invalid_request_error",'
                    '"message":"The \'gpt-5.3-codex-spark\' model is not supported when using Codex '
                    'with a ChatGPT account."}}'
                ),
            )

        out_path.write_text(
            json.dumps(
                {
                    "status": "SUCCESS",
                    "summary": "fallback worked",
                    "artifacts": [],
                    "metrics": {},
                    "next_actions": [],
                    "failure_signature": "",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(client.subprocess, "run", fake_run)

    cli = client.CodexCLIClient(
        workspace_dir=tmp_path,
        settings=client.CodexCLISettings(cli_command="codex", model="gpt-5.3-codex-spark"),
    )
    payload = cli.send(messages=[{"role": "user", "content": "hi"}])

    assert payload["status"] == "SUCCESS"
    assert payload["summary"] == "fallback worked"
    assert [cmd[cmd.index("--model") + 1] for cmd in calls] == ["gpt-5.3-codex-spark", "gpt-5.4"]


def test_codex_client_does_not_retry_default_model_on_nonzero_exit(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="ERROR: upstream unavailable",
        )

    monkeypatch.setattr(client.subprocess, "run", fake_run)

    cli = client.CodexCLIClient(
        workspace_dir=tmp_path,
        settings=client.CodexCLISettings(cli_command="codex", model="gpt-5.4"),
    )
    payload = cli.send(messages=[{"role": "user", "content": "hi"}])

    assert payload["status"] == "FAILURE"
    assert payload["failure_signature"] == "llm-codex-nonzero-exit"
    assert len(calls) == 1
