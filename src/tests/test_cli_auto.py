from __future__ import annotations

import argparse
from pathlib import Path

import cli
from cli import _auto_decide_argv, build_parser
from llm.client import MockLLMClient


def _get_subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    subparsers_action = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    return subparsers_action.choices[name]


def test_auto_decide_argv_routes_improve_prompt_to_self_improve() -> None:
    llm = MockLLMClient(
        script=[
            {
                "status": "SUCCESS",
                "argv": ["self-improve", "--goal", "Please improve tokimon CLI help output."],
            }
        ]
    )
    argv = _auto_decide_argv("Please improve tokimon CLI help output.", llm_client=llm)
    assert argv[:2] == ["self-improve", "--goal"]


def test_auto_decide_argv_can_route_suite_and_task() -> None:
    llm = MockLLMClient(script=[{"status": "SUCCESS", "argv": ["run-suite"]}])
    assert _auto_decide_argv("run suite", llm_client=llm) == ["run-suite"]

    llm = MockLLMClient(script=[{"status": "SUCCESS", "argv": ["run-task", "--task-id", "demo-1"]}])
    assert _auto_decide_argv("run task demo-1", llm_client=llm) == ["run-task", "--task-id", "demo-1"]


def test_auto_decide_argv_falls_back_to_heuristic_on_invalid_llm_response() -> None:
    llm = MockLLMClient(script=[{"status": "SUCCESS", "argv": ["auto", "run suite"]}])
    assert _auto_decide_argv("run suite", llm_client=llm) == ["run-suite"]


def test_auto_decide_argv_accepts_string_argv_from_llm() -> None:
    llm = MockLLMClient(script=[{"status": "SUCCESS", "argv": "run-task --task-id demo-1"}])
    assert _auto_decide_argv("execute demo-1", llm_client=llm) == ["run-task", "--task-id", "demo-1"]


def test_self_improve_llm_default_uses_env_else_mixed(monkeypatch) -> None:
    monkeypatch.delenv("TOKIMON_LLM", raising=False)
    parser = build_parser()
    args = parser.parse_args(["self-improve"])
    assert args.llm == "mixed"

    monkeypatch.setenv("TOKIMON_LLM", "mock")
    parser = build_parser()
    args = parser.parse_args(["self-improve"])
    assert args.llm == "mixed"


def test_self_improve_help_hides_advanced_flags() -> None:
    parser = build_parser()
    help_text = _get_subparser(parser, "self-improve").format_help()

    assert "--goal" in help_text
    assert "--input" in help_text
    assert "--sessions" not in help_text
    assert "--batches" not in help_text
    assert "--workers" not in help_text
    assert "--no-merge" not in help_text
    assert "--llm" not in help_text


def test_hidden_flags_still_parse_for_power_users() -> None:
    parser = build_parser()
    args = parser.parse_args(["self-improve", "--sessions", "10", "--llm", "codex"])
    assert args.sessions == 10
    assert args.llm == "codex"


def test_root_help_includes_subcommand_summaries() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "run-suite" in help_text
    assert "Run the benchmark suite" in help_text
    assert "memory" in help_text
    assert "Manage Tokimon's local memory" in help_text


def test_self_improve_codex_preserves_provider_default_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCodexClient:
        def __init__(self, workspace_dir: Path, settings) -> None:
            captured["workspace_dir"] = workspace_dir
            captured["settings"] = settings

    class FakeOrchestrator:
        def __init__(self, master_root: Path, llm_factory, settings) -> None:
            self.llm_factory = llm_factory

        def run(self, goal: str, input_ref: str | None = None):
            captured["goal"] = goal
            captured["input_ref"] = input_ref
            self.llm_factory("1-1", Path.cwd())
            return type("Report", (), {"run_root": str(Path("runs") / "fake-self-improve")})()

    monkeypatch.delenv("TOKIMON_CODEX_TIMEOUT_S", raising=False)
    monkeypatch.delenv("TOKIMON_CODEX_SANDBOX", raising=False)
    monkeypatch.delenv("TOKIMON_CODEX_APPROVAL", raising=False)
    monkeypatch.delenv("TOKIMON_CODEX_SEARCH", raising=False)
    monkeypatch.setattr(cli, "CodexCLIClient", FakeCodexClient)
    monkeypatch.setattr(cli, "SelfImproveOrchestrator", FakeOrchestrator)

    exit_code = cli.main(["self-improve", "--llm", "codex", "--sessions", "1", "--goal", "noop", "--no-merge"])

    assert exit_code == 0
    settings = captured.get("settings")
    assert settings is not None
    assert settings.timeout_s == 900
    assert settings.sandbox == "workspace-write"
    assert settings.ask_for_approval == "never"
    assert settings.search is True


def test_self_improve_claude_preserves_provider_default_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClaudeClient:
        def __init__(self, workspace_dir: Path, settings) -> None:
            captured["workspace_dir"] = workspace_dir
            captured["settings"] = settings

    class FakeOrchestrator:
        def __init__(self, master_root: Path, llm_factory, settings) -> None:
            self.llm_factory = llm_factory

        def run(self, goal: str, input_ref: str | None = None):
            captured["goal"] = goal
            captured["input_ref"] = input_ref
            self.llm_factory("1-1", Path.cwd())
            return type("Report", (), {"run_root": str(Path("runs") / "fake-self-improve")})()

    monkeypatch.delenv("TOKIMON_CLAUDE_TIMEOUT_S", raising=False)
    monkeypatch.setattr(cli, "ClaudeCLIClient", FakeClaudeClient)
    monkeypatch.setattr(cli, "SelfImproveOrchestrator", FakeOrchestrator)

    exit_code = cli.main(["self-improve", "--llm", "claude", "--sessions", "1", "--goal", "noop", "--no-merge"])

    assert exit_code == 0
    settings = captured.get("settings")
    assert settings is not None
    assert settings.timeout_s == 900
