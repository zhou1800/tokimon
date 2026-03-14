from __future__ import annotations

from llm.client import CodexCLISettings, interactive_codex_settings_from_env


def test_codex_settings_prefers_tokimon_env(monkeypatch) -> None:
    monkeypatch.setenv("TOKIMON_CODEX_MODEL", "tok-model")
    monkeypatch.setenv("TOKIMON_CODEX_SEARCH", "true")
    monkeypatch.setenv("TOKIMON_CODEX_TIMEOUT_S", "123")
    monkeypatch.setenv("TOKIMON_CODEX_CONFIG_JSON", '{"a": 1}')

    settings = CodexCLISettings.from_env()
    assert settings.model == "tok-model"
    assert settings.search is True
    assert settings.timeout_s == 123
    assert settings.config == {"a": 1}


def test_codex_settings_uses_defaults_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("TOKIMON_CODEX_MODEL", raising=False)
    monkeypatch.delenv("TOKIMON_CODEX_SEARCH", raising=False)
    monkeypatch.delenv("TOKIMON_CODEX_TIMEOUT_S", raising=False)
    monkeypatch.delenv("TOKIMON_CODEX_CONFIG_JSON", raising=False)

    settings = CodexCLISettings.from_env()
    assert settings.model == "gpt-5.4"
    assert settings.search is False
    assert settings.timeout_s == 900
    assert settings.config == {}


def test_interactive_codex_settings_default_to_writable(monkeypatch) -> None:
    monkeypatch.delenv("TOKIMON_CODEX_SANDBOX", raising=False)
    monkeypatch.delenv("TOKIMON_CODEX_APPROVAL", raising=False)

    settings = interactive_codex_settings_from_env()

    assert settings.sandbox == "workspace-write"
    assert settings.ask_for_approval == "never"


def test_interactive_codex_settings_respect_explicit_env(monkeypatch) -> None:
    monkeypatch.setenv("TOKIMON_CODEX_SANDBOX", "read-only")
    monkeypatch.setenv("TOKIMON_CODEX_APPROVAL", "on-failure")

    settings = interactive_codex_settings_from_env(model="gpt-5.4")

    assert settings.model == "gpt-5.4"
    assert settings.sandbox == "read-only"
    assert settings.ask_for_approval == "on-failure"
