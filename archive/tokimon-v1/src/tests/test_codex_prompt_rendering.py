from __future__ import annotations

from pathlib import Path

from llm.client import CodexCLISettings, _codex_cli_preamble, _render_prompt


def test_codex_cli_preamble_includes_permissions_and_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SHELL", "/bin/bash")
    settings = CodexCLISettings(sandbox="read-only", ask_for_approval="never", search=True)

    preamble = _codex_cli_preamble(settings, tmp_path, delegation_depth=1)
    assert "<permissions instructions>" in preamble
    assert "sandbox_mode: read-only" in preamble
    assert "approval_policy: never" in preamble
    assert "search_enabled: True" in preamble
    assert "<environment_context>" in preamble
    assert f"<cwd>{tmp_path}</cwd>" in preamble
    assert "<shell>bash</shell>" in preamble
    assert "<tokimon_context>" in preamble
    assert "<delegated>true</delegated>" in preamble
    assert "<delegation_depth>1</delegation_depth>" in preamble


def test_render_prompt_is_deterministic_and_sorts_tools() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    tools_a = [
        {"name": "ztool", "actions": ["z", "a"], "signatures": {"a": "a()", "z": "z()"}},
        {"name": "atool", "actions": ["b"], "signatures": {"b": "b()"}},
    ]
    tools_b = list(reversed(tools_a))

    prompt_a = _render_prompt(messages, tools=tools_a, preamble="PRE")
    prompt_b = _render_prompt(messages, tools=tools_b, preamble="PRE")

    assert prompt_a == prompt_b
    assert prompt_a.startswith("PRE\n")
    assert "<tools>" in prompt_a
    assert "</tools>" in prompt_a
    assert prompt_a.index("- atool:") < prompt_a.index("- ztool:")
    assert prompt_a.index("  - a()") < prompt_a.index("  - z()")
    assert "<conversation>" in prompt_a
    assert "</conversation>" in prompt_a
    assert "SYSTEM: sys" in prompt_a
    assert "USER: hi" in prompt_a
