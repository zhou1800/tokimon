from __future__ import annotations

import importlib
import io
import json
import logging
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

debate = importlib.import_module("tokimon.tools.mcp.ai_debate")


def settings_section(section: str, **_: Any) -> dict[str, Any]:
    if section == "ai_debate":
        return {"access": "local-cli", "peer_b": "claude"}
    return {}


def settings_payload(*, peer_b: str = "claude") -> dict[str, Any]:
    return {
        "ai_debate": {"access": "local-cli", "peer_b": peer_b},
    }


class FakeRunner(debate.ProviderRunner):
    def __init__(self, responses: dict[str, Any]) -> None:
        super().__init__(env={"PATH": ""})
        self.responses = {
            task_id: list(response) if isinstance(response, list) else [response]
            for task_id, response in responses.items()
        }
        self.calls: list[tuple[str, str]] = []

    def resolve_executable(self, provider: str) -> str | None:
        return f"/fake/{provider}"

    def _capture_version(self, provider: str) -> str:
        return f"{provider} test"

    def capture_runtime_versions(self) -> dict[str, str]:
        return {
            "python": "3.12.0",
            "codex": "codex test",
            "claude": "claude test",
        }

    def run_json(
        self,
        *,
        provider: str,
        model: str,
        effort: str | None,
        prompt: str,
        timeout_sec: int,
        workspace_dir: Path,
        task_id: str,
        results_dir: Path,
        expected_keys: tuple[str, ...],
        poll_interval_sec: float = 1.0,
        max_wall_time_sec: int | None = None,
    ) -> debate.ProviderCallResult:
        del timeout_sec, poll_interval_sec, max_wall_time_sec
        self.calls.append((task_id, provider))
        if task_id not in self.responses:
            raise AssertionError(f"No fake response configured for {task_id}")
        if not self.responses[task_id]:
            raise AssertionError(f"No remaining fake response configured for {task_id}")
        response = self.responses[task_id].pop(0)
        results_dir.mkdir(parents=True, exist_ok=True)
        schema_path = results_dir / f"{task_id}.schema.json"
        stdout_path = results_dir / f"{task_id}.stdout"
        stderr_path = results_dir / f"{task_id}.stderr"
        schema_path.write_text(
            json.dumps(debate.schema_for_expected_keys(expected_keys)),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        if callable(response):
            response = response(
                provider=provider,
                task_id=task_id,
                results_dir=results_dir,
            )
        if isinstance(response, Exception):
            raise response
        stdout_path.write_text(json.dumps(response), encoding="utf-8")
        return debate.ProviderCallResult(
            payload=response,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )


def workspace_temp_dir() -> tempfile.TemporaryDirectory[str]:
    debate.WORKSPACE_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=debate.WORKSPACE_STATE_ROOT)


def materialize_task_artifact(
    run_dir: Path,
    task_id: str,
    role: str,
    *,
    status: str = "success",
    response_json: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
) -> None:
    (run_dir / "prompts" / f"{task_id}.md").write_text("prompt", encoding="utf-8")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if response_json is None and status == "success":
        response_json = {
            "response_md": "A" if role == "agent_a" else "B",
            "signal": "continue",
            "signal_reason": "Continue." if role == "agent_a" else "Judge.",
            "citations": [],
            "files_created": [],
        }
    stdout_path = run_dir / "results" / f"{task_id}.stdout"
    stderr_path = run_dir / "results" / f"{task_id}.stderr"
    if status == "success":
        expected_keys = (
            ("winner", "reason", "termination", "decisive_round")
            if role == "judge"
            else ("response_md", "signal", "signal_reason")
        )
        (run_dir / "results" / f"{task_id}.schema.json").write_text(
            json.dumps(debate.schema_for_expected_keys(expected_keys)),
            encoding="utf-8",
        )
        stdout_path.write_text(json.dumps(response_json), encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
    payload: dict[str, Any] = {
        "id": task_id,
        "role": role,
        "provider": config["roles"][role]["provider"],
        "model": config["roles"][role]["model"],
        "status": status,
        "stdout_path": (
            failure.get("stdout_path")
            if failure
            else f"results/{task_id}.stdout"
        ),
        "stderr_path": (
            failure.get("stderr_path")
            if failure
            else f"results/{task_id}.stderr"
        ),
        "cli_exit_code": failure.get("cli_exit_code") if failure else 0,
        "finished_at": (
            failure["detected_at"] if failure else config["created_at"]
        ),
    }
    if status == "success":
        payload.update(
            {
                "response_json": response_json,
            }
        )
    else:
        payload["failure"] = failure
    debate.write_json_atomic(
        run_dir / "results" / f"{task_id}.json",
        payload,
    )


def retryable_rate_limit_error(
    *,
    provider: str,
    task_id: str,
    results_dir: Path,
) -> debate.ProviderCallError:
    stderr_path = results_dir / f"{task_id}.stderr"
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.write_text("Rate limit exceeded. Retry-After: 120 seconds", encoding="utf-8")
    return debate.ProviderCallError(
        f"{provider} task {task_id} exited with 1: rate limit exceeded",
        provider=provider,
        task_id=task_id,
        stderr_path=stderr_path,
        cli_exit_code=1,
        retry_reason="rate_limit",
        retry_after_sec=120,
    )


def nonretryable_provider_error(
    *,
    provider: str,
    task_id: str,
    results_dir: Path,
) -> debate.ProviderCallError:
    stderr_path = results_dir / f"{task_id}.stderr"
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.write_text("provider crashed", encoding="utf-8")
    return debate.ProviderCallError(
        f"{provider} task {task_id} exited with 1: provider crashed",
        provider=provider,
        task_id=task_id,
        stderr_path=stderr_path,
        cli_exit_code=1,
    )


class DebateOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings_patch = patch.object(
            debate,
            "load_settings_section",
            side_effect=settings_section,
        )
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)

    def test_run_options_use_ai_debate_settings_when_args_omit_overrides(self) -> None:
        with patch.object(
            debate,
            "load_settings_section",
            side_effect=lambda section, **_: (
                {"access": "local-cli", "peer_b": "codex"}
                if section == "ai_debate"
                else settings_section(section)
            ),
        ):
            options = debate.run_options_from_mapping(
                {"topic": "T", "side_a": "A", "side_b": "B"}
            )

        self.assertEqual(options.access, "local-cli")
        self.assertEqual(options.side_b_provider, "codex")

    def test_run_options_arguments_override_ai_debate_settings(self) -> None:
        with patch.object(
            debate,
            "load_settings_section",
            side_effect=lambda section, **_: (
                {"access": "local-cli", "peer_b": "codex"}
                if section == "ai_debate"
                else settings_section(section)
            ),
        ):
            options = debate.run_options_from_mapping(
                {
                    "topic": "T",
                    "side_a": "A",
                    "side_b": "B",
                    "access": "local-cli",
                    "side_b_provider": "claude",
                }
            )

        self.assertEqual(options.access, "local-cli")
        self.assertEqual(options.side_b_provider, "claude")

    def test_remote_agent_access_is_rejected(self) -> None:
        with self.assertRaises(debate.DebateToolError) as raised:
            debate.validate_options(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    access="mcp.remote-agent-broker",
                )
            )

        self.assertIn("access must be local-cli", str(raised.exception))

    def test_main_thread_options_apply_to_same_provider_roles(self) -> None:
        options = debate.run_options_from_mapping({
            "topic": "T", "side_a": "A", "side_b": "B",
            "model": "parent-model", "effort": "high", "side_b_provider": "codex",
        })
        config = debate.resolve_run_configuration(options)
        for role in ("agent_a", "agent_b", "judge"):
            self.assertEqual(config["roles"][role]["model"], "parent-model")
            self.assertEqual(config["roles"][role]["effort"], "high")

    def test_claude_peer_does_not_inherit_codex_model_or_effort(self) -> None:
        for overrides, expected in (
            ({}, {"model": "default", "effort": "default"}),
            ({"agent_b_model": "claude-peer"}, {"model": "claude-peer", "effort": "default"}),
            ({"agent_b_effort": "max"}, {"model": "default", "effort": "max"}),
        ):
            with self.subTest(overrides=overrides):
                options = debate.RunOptions(topic="T", side_a="A", side_b="B",
                                            model="gpt-parent", effort="high", **overrides)
                config = debate.resolve_run_configuration(options)
                peer = config["roles"]["agent_b"]
                self.assertEqual(peer, {"provider": "claude", **expected})
                runner = debate.ProviderRunner(env={})
                argv, _ = runner._build_command(
                    executable="/fake/claude", provider="claude", model=peer["model"],
                    effort=peer["effort"], prompt="P", workspace_dir=ROOT,
                    schema={}, schema_path=ROOT / "unused.json",
                )
                self.assertNotIn("gpt-parent", argv)
                self.assertNotIn("high", argv)
                self.assertEqual("--model" in argv, expected["model"] != "default")
                self.assertEqual("--effort" in argv, expected["effort"] != "default")
                self.assertEqual(config["roles"]["judge"]["model"], "gpt-parent")

    def test_role_overrides_take_precedence_over_main_thread_options(self) -> None:
        config = debate.resolve_run_configuration(debate.RunOptions(
            topic="T", side_a="A", side_b="B", model="parent-model", effort="high",
            agent_b_model="peer-model", agent_b_effort="max",
        ))
        self.assertEqual(config["roles"]["agent_b"], {
            "provider": "claude", "model": "peer-model", "effort": "max",
        })
        self.assertEqual(config["roles"]["judge"]["model"], "parent-model")

    def test_run_config_records_cli_defaults_and_supports_completed_run_reads(self) -> None:
        runner = FakeRunner(
            {
                "round-01-agent-a": {
                    "response_md": "Agent A opens.",
                    "signal": "continue",
                    "signal_reason": "Agent B should respond.",
                },
                "round-01-agent-b": {
                    "response_md": "Agent B responds.",
                    "signal": "continue",
                    "signal_reason": "The round should be judged.",
                },
                "final-judgment": {
                    "winner": "agent_b",
                    "reason": "Agent B made the stronger case.",
                    "termination": "round_limit",
                    "decisive_round": 1,
                },
            }
        )
        orchestrator = debate.DebateOrchestrator(runner=runner)

        with workspace_temp_dir() as tmpdir:
            summary = orchestrator.run_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    max_rounds=1,
                    output_root=Path(tmpdir),
                )
            )

            config = json.loads(
                (Path(summary["run_dir"]) / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(orchestrator.status(Path(summary["run_dir"]))["status"], "completed")

        self.assertEqual(config["access"], "local-cli")
        self.assertEqual(
            config["roles"]["agent_a"],
            {"provider": "codex", "model": "default", "effort": "default"},
        )
        self.assertEqual(
            config["roles"]["agent_b"],
            {"provider": "claude", "model": "default", "effort": "default"},
        )
        self.assertEqual(
            config["roles"]["judge"],
            {"provider": "codex", "model": "default", "effort": "default"},
        )

    def test_start_new_returns_run_handle_without_provider_calls_then_resume_completes(self) -> None:
        runner = FakeRunner(
            {
                "round-01-agent-a": {
                    "response_md": "Agent A opens.",
                    "signal": "continue",
                    "signal_reason": "Agent B should respond.",
                },
                "round-01-agent-b": {
                    "response_md": "Agent B responds.",
                    "signal": "continue",
                    "signal_reason": "The round should be judged.",
                },
                "final-judgment": {
                    "winner": "agent_b",
                    "reason": "Agent B made the stronger case.",
                    "termination": "round_limit",
                    "decisive_round": 1,
                },
            }
        )
        orchestrator = debate.DebateOrchestrator(runner=runner)

        with workspace_temp_dir() as tmpdir:
            started = orchestrator.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    max_rounds=1,
                    output_root=Path(tmpdir),
                )
            )

            run_dir = Path(started["run_dir"])
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            status = orchestrator.status(run_dir)

            self.assertEqual(started["status"], "in_progress")
            self.assertEqual(started["next_step"], "agent_a")
            self.assertEqual(state["status"], "in_progress")
            self.assertEqual(state["next_step"], "agent_a")
            self.assertEqual(config["topic"], "T")
            self.assertTrue((run_dir / "prompts").is_dir())
            self.assertTrue((run_dir / "results").is_dir())
            self.assertIn("# Debate Topic", (run_dir / "debate_log.md").read_text(encoding="utf-8"))
            self.assertEqual(status["run_dir"], started["run_dir"])
            self.assertEqual(runner.calls, [])

            resumed = orchestrator.resume(run_dir)

        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(
            runner.calls,
            [
                ("round-01-agent-a", "codex"),
                ("round-01-agent-b", "claude"),
                ("final-judgment", "codex"),
            ],
        )
        self.assertEqual(resumed["final_result"]["winner"], "agent_b")

    def test_resume_reconciles_turn_appended_before_state_commit(self) -> None:
        interrupted_runner = FakeRunner(
            {
                "round-01-agent-a": {
                    "response_md": "Uncommitted Agent A response.",
                    "signal": "continue",
                    "signal_reason": "This response must not survive recovery.",
                }
            }
        )
        interrupted = debate.DebateOrchestrator(runner=interrupted_runner)

        with workspace_temp_dir() as tmpdir:
            started = interrupted.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    max_rounds=1,
                    output_root=Path(tmpdir),
                )
            )
            run_dir = Path(started["run_dir"])
            original_persist = interrupted._persist_state
            persist_calls = 0

            def interrupt_second_persist(context: debate.RunContext) -> None:
                nonlocal persist_calls
                persist_calls += 1
                if persist_calls == 2:
                    raise SystemExit("simulated process interruption")
                original_persist(context)

            with (
                patch.object(
                    interrupted,
                    "_persist_state",
                    side_effect=interrupt_second_persist,
                ),
                self.assertRaises(SystemExit),
            ):
                interrupted.resume(run_dir)

            stale_log = (run_dir / "debate_log.md").read_text(encoding="utf-8")
            persisted_before_resume = json.loads(
                (run_dir / "state.json").read_text(encoding="utf-8")
            )
            self.assertIn("Uncommitted Agent A response.", stale_log)
            self.assertEqual(persisted_before_resume["next_step"], "agent_a")

            recovered = debate.DebateOrchestrator(
                runner=FakeRunner(
                    {
                        "round-01-agent-a": {
                            "response_md": "Committed Agent A response.",
                            "signal": "continue",
                            "signal_reason": "Agent B should respond.",
                        },
                        "round-01-agent-b": {
                            "response_md": "Agent B response.",
                            "signal": "continue",
                            "signal_reason": "The round should be judged.",
                        },
                        "final-judgment": {
                            "winner": "agent_a",
                            "reason": "Agent A made the stronger case.",
                            "termination": "round_limit",
                            "decisive_round": 1,
                        },
                    }
                )
            )
            summary = recovered.resume(run_dir)
            final_log = (run_dir / "debate_log.md").read_text(encoding="utf-8")
            final_state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "completed")
        self.assertNotIn("Uncommitted Agent A response.", final_log)
        self.assertIn("Committed Agent A response.", final_log)
        self.assertEqual(
            final_state["rounds"][0]["agent_a"]["response_md"],
            "Committed Agent A response.",
        )

    def test_resume_reconciles_final_log_appended_before_state_commit(self) -> None:
        interrupted_runner = FakeRunner(
            {
                "round-01-agent-a": {
                    "response_md": "Agent A response.",
                    "signal": "continue",
                    "signal_reason": "Agent B should respond.",
                },
                "round-01-agent-b": {
                    "response_md": "Agent B response.",
                    "signal": "continue",
                    "signal_reason": "The round should be judged.",
                },
                "final-judgment": {
                    "winner": "agent_b",
                    "reason": "Uncommitted final reason.",
                    "termination": "round_limit",
                    "decisive_round": 1,
                },
            }
        )
        interrupted = debate.DebateOrchestrator(runner=interrupted_runner)

        with workspace_temp_dir() as tmpdir:
            started = interrupted.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    max_rounds=1,
                    output_root=Path(tmpdir),
                )
            )
            run_dir = Path(started["run_dir"])
            original_persist = interrupted._persist_state
            persist_calls = 0

            def interrupt_fourth_persist(context: debate.RunContext) -> None:
                nonlocal persist_calls
                persist_calls += 1
                if persist_calls == 4:
                    raise SystemExit("simulated process interruption")
                original_persist(context)

            with (
                patch.object(
                    interrupted,
                    "_persist_state",
                    side_effect=interrupt_fourth_persist,
                ),
                self.assertRaises(SystemExit),
            ):
                interrupted.resume(run_dir)

            self.assertIn(
                "Uncommitted final reason.",
                (run_dir / "debate_log.md").read_text(encoding="utf-8"),
            )
            persisted_before_resume = json.loads(
                (run_dir / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted_before_resume["next_step"], "final_judgment")

            recovered = debate.DebateOrchestrator(
                runner=FakeRunner(
                    {
                        "final-judgment": {
                            "winner": "agent_a",
                            "reason": "Committed final reason.",
                            "termination": "round_limit",
                            "decisive_round": 1,
                        }
                    }
                )
            )
            summary = recovered.resume(run_dir)
            final_log = (run_dir / "debate_log.md").read_text(encoding="utf-8")

        self.assertEqual(summary["status"], "completed")
        self.assertNotIn("Uncommitted final reason.", final_log)
        self.assertIn("Committed final reason.", final_log)
        self.assertEqual(summary["final_result"]["reason"], "Committed final reason.")

    def test_status_reports_malformed_persisted_run_as_a_tool_error(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))

        with workspace_temp_dir() as tmpdir:
            started = orchestrator.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    output_root=Path(tmpdir),
                )
            )
            run_dir = Path(started["run_dir"])
            config_path = run_dir / "run_config.json"
            state_path = run_dir / "state.json"
            valid_config = config_path.read_text(encoding="utf-8")
            valid_state = state_path.read_text(encoding="utf-8")

            for target, malformed, expected in (
                (state_path, "{", "Malformed JSON in debate artifact"),
                (state_path, "[]", "state.json must be an object"),
                (state_path, "{}", "state.json is missing required field(s)"),
                (
                    config_path,
                    "{}",
                    "run_config.json is missing required field(s)",
                ),
            ):
                with self.subTest(target=target.name, malformed=malformed):
                    config_path.write_text(valid_config, encoding="utf-8")
                    state_path.write_text(valid_state, encoding="utf-8")
                    target.write_text(malformed, encoding="utf-8")
                    with self.assertRaises(debate.DebateToolError) as raised:
                        orchestrator.status(started["run_dir"])

                    self.assertIn(expected, str(raised.exception))
                    self.assertIn(target.name, str(raised.exception))

    def test_status_rejects_nested_and_cross_field_run_corruption(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))
        agent_a_turn = {
            "response_md": "A",
            "signal": "continue",
            "signal_reason": "Continue.",
            "citations": [],
            "files_created": [],
            "task_id": "round-01-agent-a",
            "result_path": "results/round-01-agent-a.json",
        }
        agent_b_turn = {
            "response_md": "B",
            "signal": "continue",
            "signal_reason": "Judge.",
            "citations": [],
            "files_created": [],
            "task_id": "round-01-agent-b",
            "result_path": "results/round-01-agent-b.json",
        }

        cases = (
            (
                "access-type",
                (("config", ("access",), ["local-cli"]),),
                "access must be local-cli",
            ),
            (
                "side-b-provider-type",
                (("config", ("side_b_provider",), {"provider": "claude"}),),
                "side_b_provider must be claude or codex",
            ),
            (
                "agent-a-provider",
                (("config", ("roles", "agent_a", "provider"), "claude"),),
                "roles.agent_a.provider must be codex",
            ),
            (
                "judge-provider",
                (("config", ("roles", "judge", "provider"), "claude"),),
                "roles.judge.provider must be codex",
            ),
            (
                "agent-b-provider",
                (("config", ("roles", "agent_b", "provider"), "codex"),),
                "roles.agent_b.provider must be claude",
            ),
            (
                "round-number",
                (
                    (
                        "state",
                        ("rounds",),
                        [{"round_number": "bad", "agent_a": None, "agent_b": None}],
                    ),
                ),
                "rounds[0].round_number must be a positive integer",
            ),
            (
                "turn-record",
                (
                    (
                        "state",
                        ("rounds",),
                        [{"round_number": 1, "agent_a": "bad", "agent_b": None}],
                    ),
                ),
                "rounds[0].agent_a must be an object",
            ),
            (
                "agent-b-before-agent-a",
                (
                    ("state", ("next_step",), "agent_b"),
                    (
                        "state",
                        ("rounds",),
                        [{"round_number": 1, "agent_a": None, "agent_b": None}],
                    ),
                ),
                "next_step agent_b requires only Agent A's current-round turn",
            ),
            (
                "completed-without-result",
                (
                    ("config", ("max_rounds",), 1),
                    ("state", ("status",), "completed"),
                    ("state", ("next_step",), "completed"),
                    (
                        "state",
                        ("rounds",),
                        [
                            {
                                "round_number": 1,
                                "agent_a": agent_a_turn,
                                "agent_b": agent_b_turn,
                            }
                        ],
                    ),
                    ("state", ("termination",), "round_limit"),
                ),
                "final_result must be an object",
            ),
            (
                "in-progress-completed-step",
                (
                    ("state", ("next_step",), "completed"),
                    (
                        "state",
                        ("rounds",),
                        [
                            {
                                "round_number": 1,
                                "agent_a": agent_a_turn,
                                "agent_b": agent_b_turn,
                            }
                        ],
                    ),
                ),
                "status completed and next_step completed must occur together",
            ),
            (
                "waiting-without-retry",
                (
                    ("state", ("status",), "waiting_retry"),
                    ("state", ("last_error",), "retry later"),
                    (
                        "state",
                        ("last_failure",),
                        {
                            "retryable": True,
                            "reason": "rate_limit",
                            "message": "retry later",
                            "detected_at": "derived-after-run-creation",
                            "next_step": "agent_a",
                            "provider": "codex",
                            "task_id": "round-01-agent-a",
                            "role": "agent_a",
                            "retry_after_at": "derived-from-detected-at",
                            "retry_after_sec": 3600,
                        },
                    ),
                ),
                "status waiting_retry requires retryable retry metadata",
            ),
            (
                "active-task-type",
                (("state", ("active_task",), "job-1"),),
                "active_task must be an object",
            ),
            (
                "invalid-winner",
                (
                    ("state", ("next_step",), "final_judgment"),
                    (
                        "state",
                        ("rounds",),
                        [
                            {
                                "round_number": 1,
                                "agent_a": {
                                    **agent_a_turn,
                                    "signal": "concede",
                                },
                                "agent_b": None,
                            }
                        ],
                    ),
                    ("state", ("termination",), "concession"),
                    ("state", ("winner",), "bogus"),
                ),
                "winner is unsupported",
            ),
            (
                "round-over-limit",
                (("state", ("current_round",), 25),),
                "current_round exceeds run_config.max_rounds",
            ),
        )

        with workspace_temp_dir() as tmpdir:
            started = orchestrator.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    output_root=Path(tmpdir),
                )
            )
            run_dir = Path(started["run_dir"])
            config_path = run_dir / "run_config.json"
            state_path = run_dir / "state.json"
            valid_config = json.loads(config_path.read_text(encoding="utf-8"))
            valid_state = json.loads(state_path.read_text(encoding="utf-8"))

            for name, assignments, expected in cases:
                with self.subTest(name=name):
                    config = json.loads(json.dumps(valid_config))
                    state = json.loads(json.dumps(valid_state))
                    roots = {"config": config, "state": state}
                    for root_name, keys, replacement in assignments:
                        target = roots[root_name]
                        for key in keys[:-1]:
                            target = target[key]
                        target[keys[-1]] = replacement
                    round_one = next(
                        (
                            record
                            for record in state.get("rounds", [])
                            if isinstance(record, dict)
                            and record.get("round_number") == 1
                        ),
                        {},
                    )
                    for role, task_id in (
                        ("agent_a", "round-01-agent-a"),
                        ("agent_b", "round-01-agent-b"),
                    ):
                        turn = round_one.get(role)
                        response_json = None
                        if isinstance(turn, dict):
                            response_json = {
                                key: turn.get(key, [])
                                for key in (
                                    "response_md",
                                    "signal",
                                    "signal_reason",
                                    "citations",
                                    "files_created",
                                )
                            }
                        materialize_task_artifact(
                            run_dir,
                            task_id,
                            role,
                            response_json=response_json,
                        )
                    if state["status"] == "waiting_retry":
                        detected_at = datetime.fromisoformat(config["created_at"]) + timedelta(
                            seconds=1
                        )
                        retry_after_sec = int(
                            state["last_failure"]["retry_after_sec"]
                        )
                        state["last_failure"]["detected_at"] = detected_at.isoformat()
                        state["last_failure"]["retry_after_at"] = (
                            detected_at + timedelta(seconds=retry_after_sec)
                        ).isoformat()
                        materialize_task_artifact(
                            run_dir,
                            "round-01-agent-a",
                            "agent_a",
                            status="waiting_retry",
                            failure=state["last_failure"],
                        )
                    debate.write_json_atomic(config_path, config)
                    debate.write_json_atomic(state_path, state)

                    with self.assertRaises(debate.DebateToolError) as raised:
                        orchestrator.status(run_dir)

                    self.assertIn(expected, str(raised.exception))

    def test_status_rejects_missing_required_run_artifacts(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))

        with workspace_temp_dir() as tmpdir:
            for artifact_name in (
                "debate_log.md",
                "prompts",
                "results",
                "runtime_versions.json",
            ):
                with self.subTest(artifact_name=artifact_name):
                    started = orchestrator.start_new(
                        debate.RunOptions(
                            topic="T",
                            side_a="A",
                            side_b="B",
                            output_root=Path(tmpdir),
                        )
                    )
                    run_dir = Path(started["run_dir"])
                    artifact_path = run_dir / artifact_name
                    if artifact_path.is_dir():
                        artifact_path.rmdir()
                    else:
                        artifact_path.unlink()

                    with self.assertRaises(debate.DebateToolError) as raised:
                        orchestrator.status(run_dir)

                    self.assertIn("Malformed debate run", str(raised.exception))
                    self.assertIn(artifact_name, str(raised.exception))

    def test_status_and_read_log_preserve_legacy_run_inspection(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))

        with workspace_temp_dir() as tmpdir:
            started = orchestrator.start_new(
                debate.RunOptions(
                    topic="Legacy topic",
                    side_a="A",
                    side_b="B",
                    output_root=Path(tmpdir),
                )
            )
            run_dir = Path(started["run_dir"])
            state_path = run_dir / "state.json"
            config_path = run_dir / "run_config.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state.pop("active_task")
            state.pop("task_attempts")
            debate.write_json_atomic(state_path, state)

            local_summary = orchestrator.status(run_dir)
            self.assertIsNone(local_summary["active_task"])

            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["access"] = "mcp.remote-agent-broker"
            debate.write_json_atomic(config_path, config)

            remote_summary = orchestrator.status(run_dir)
            log_payload = orchestrator.read_log(run_dir)
            with self.assertRaises(debate.DebateRunFailedError) as raised:
                orchestrator.resume(run_dir)

        self.assertEqual(remote_summary["status"], "in_progress")
        self.assertIn("Legacy topic", log_payload["text"])
        self.assertIn("historical mcp.remote-agent-broker", str(raised.exception))
        self.assertEqual(raised.exception.summary["run_dir"], str(run_dir))

    def test_status_gates_historical_lifecycle_metadata_by_access(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))

        with workspace_temp_dir() as tmpdir:
            started = orchestrator.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    output_root=Path(tmpdir),
                )
            )
            run_dir = Path(started["run_dir"])
            state_path = run_dir / "state.json"
            config_path = run_dir / "run_config.json"
            base_state = json.loads(state_path.read_text(encoding="utf-8"))

            local_active = json.loads(json.dumps(base_state))
            local_active["active_task"] = {}
            debate.write_json_atomic(state_path, local_active)
            with self.assertRaises(debate.DebateToolError) as active_error:
                orchestrator.status(run_dir)
            self.assertIn(
                "active_task is allowed only for historical remote-broker runs",
                str(active_error.exception),
            )

            local_attempt = json.loads(json.dumps(base_state))
            local_attempt["task_attempts"] = {"round-01-agent-a": 1}
            debate.write_json_atomic(state_path, local_attempt)
            with self.assertRaises(debate.DebateToolError) as attempt_error:
                orchestrator.status(run_dir)
            self.assertIn(
                "task_attempts are allowed only for historical remote-broker runs",
                str(attempt_error.exception),
            )

            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["access"] = "mcp.remote-agent-broker"
            debate.write_json_atomic(config_path, config)

            remote_active = json.loads(json.dumps(base_state))
            remote_active["active_task"] = {}
            remote_active["task_attempts"] = {"round-01-agent-a": 1}
            debate.write_json_atomic(state_path, remote_active)
            with self.assertRaises(debate.DebateToolError) as malformed_active:
                orchestrator.status(run_dir)
            self.assertIn(
                "active_task is missing required field(s)",
                str(malformed_active.exception),
            )

            remote_attempt = json.loads(json.dumps(base_state))
            remote_attempt["task_attempts"] = {"round-24-agent-b": 1}
            debate.write_json_atomic(state_path, remote_attempt)
            with self.assertRaises(debate.DebateToolError) as unrelated_attempt:
                orchestrator.status(run_dir)
            self.assertIn(
                "outside the reached lifecycle",
                str(unrelated_attempt.exception),
            )

    def test_read_log_rejects_a_malformed_run(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))

        with workspace_temp_dir() as tmpdir:
            started = orchestrator.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    output_root=Path(tmpdir),
                )
            )
            run_dir = Path(started["run_dir"])
            (run_dir / "state.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(debate.DebateToolError) as raised:
                orchestrator.read_log(run_dir)

        self.assertIn("state.json is missing required field(s)", str(raised.exception))

    def test_status_and_log_reject_contained_canonical_symlinks(self) -> None:
        for relative in ("run_config.json", "state.json", "debate_log.md", "prompts"):
            with self.subTest(relative=relative), workspace_temp_dir() as tmpdir:
                orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))
                started = orchestrator.start_new(debate.RunOptions(
                    topic="T", side_a="A", side_b="B", output_root=Path(tmpdir),
                ))
                run_dir = Path(started["run_dir"])
                target = run_dir / relative
                copy = run_dir / (relative + ".copy")
                target.rename(copy)
                target.symlink_to(copy, target_is_directory=copy.is_dir())
                for read in (orchestrator.status, orchestrator.read_log):
                    with self.assertRaisesRegex(debate.DebateToolError, "symbolic link"):
                        read(run_dir)
                self.assertTrue(target.is_symlink())
                self.assertTrue(copy.exists())

    def test_status_rejects_task_artifact_reference_and_symlink_escape(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))
        turn = {
            "response_md": "A",
            "signal": "continue",
            "signal_reason": "Continue.",
            "citations": [],
            "files_created": [],
            "task_id": "round-01-agent-a",
            "result_path": "results/round-01-agent-a.json",
        }

        with workspace_temp_dir() as tmpdir:
            started = orchestrator.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    output_root=Path(tmpdir),
                )
            )
            run_dir = Path(started["run_dir"])
            state_path = run_dir / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["next_step"] = "agent_b"
            state["rounds"] = [
                {"round_number": 1, "agent_a": turn, "agent_b": None}
            ]
            materialize_task_artifact(run_dir, "round-01-agent-a", "agent_a")
            debate.write_json_atomic(state_path, state)
            self.assertEqual(orchestrator.status(run_dir)["next_step"], "agent_b")

            state["rounds"][0]["agent_a"]["result_path"] = "../../outside.json"
            debate.write_json_atomic(state_path, state)
            with self.assertRaises(debate.DebateToolError) as noncanonical:
                orchestrator.status(run_dir)
            self.assertIn("result_path is not canonical", str(noncanonical.exception))

            state["rounds"][0]["agent_a"]["result_path"] = (
                "results/round-01-agent-a.json"
            )
            result_artifact_path = run_dir / "results" / "round-01-agent-a.json"
            result_artifact = json.loads(result_artifact_path.read_text(encoding="utf-8"))
            result_artifact["id"] = "wrong-task"
            debate.write_json_atomic(result_artifact_path, result_artifact)
            debate.write_json_atomic(state_path, state)
            with self.assertRaises(debate.DebateToolError) as wrong_content:
                orchestrator.status(run_dir)
            self.assertIn("id does not match", str(wrong_content.exception))

            outside_result = Path(tmpdir) / "outside-result.json"
            debate.write_json_atomic(
                outside_result,
                {"id": "round-01-agent-a", "role": "agent_a", "status": "success"},
            )
            result_path = run_dir / "results" / "round-01-agent-a.json"
            result_path.unlink()
            result_path.symlink_to(outside_result)
            debate.write_json_atomic(state_path, state)
            with self.assertRaises(debate.DebateToolError) as escaped:
                orchestrator.status(run_dir)

        self.assertIn("artifact escapes the run boundary", str(escaped.exception))

    def test_status_rejects_tampered_success_task_result_fields(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))
        turn = {
            "response_md": "A",
            "signal": "continue",
            "signal_reason": "Continue.",
            "citations": [],
            "files_created": [],
            "task_id": "round-01-agent-a",
            "result_path": "results/round-01-agent-a.json",
        }

        with workspace_temp_dir() as tmpdir:
            started = orchestrator.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    output_root=Path(tmpdir),
                )
            )
            run_dir = Path(started["run_dir"])
            state_path = run_dir / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["next_step"] = "agent_b"
            state["rounds"] = [
                {"round_number": 1, "agent_a": turn, "agent_b": None}
            ]
            debate.write_json_atomic(state_path, state)
            materialize_task_artifact(run_dir, "round-01-agent-a", "agent_a")
            result_path = run_dir / "results" / "round-01-agent-a.json"
            valid_result = json.loads(result_path.read_text(encoding="utf-8"))

            cases = (
                (
                    "status-type",
                    lambda payload: payload.update({"status": ["success"]}),
                    "status must be one of: success",
                ),
                (
                    "provider",
                    lambda payload: payload.update({"provider": "claude"}),
                    "provider does not match run_config",
                ),
                (
                    "model",
                    lambda payload: payload.update({"model": "wrong-model"}),
                    "model does not match run_config",
                ),
                (
                    "response",
                    lambda payload: payload["response_json"].update(
                        {"response_md": "tampered"}
                    ),
                    "response_json does not match state",
                ),
                (
                    "local-citation-object",
                    lambda payload: payload["response_json"].update(
                        {"citations": [{"url": "https://example.test"}]}
                    ),
                    "must contain strings",
                ),
                (
                    "missing-stdout",
                    lambda payload: payload.update({"stdout_path": None}),
                    "successful task requires stdout_path",
                ),
                (
                    "stdout-path",
                    lambda payload: payload.update({"stdout_path": "../../outside"}),
                    "stdout_path is not canonical",
                ),
                (
                    "exit-code",
                    lambda payload: payload.update({"cli_exit_code": 1}),
                    "successful task cli_exit_code must be zero",
                ),
                (
                    "timestamp",
                    lambda payload: payload.update({"finished_at": "not-a-date"}),
                    "finished_at must be an ISO-8601 timestamp",
                ),
                (
                    "pre-run-timestamp",
                    lambda payload: payload.update(
                        {"finished_at": "2000-01-01T00:00:00+00:00"}
                    ),
                    "finished_at cannot precede run creation",
                ),
            )
            for name, mutate, expected in cases:
                with self.subTest(name=name):
                    result = json.loads(json.dumps(valid_result))
                    mutate(result)
                    debate.write_json_atomic(result_path, result)
                    with self.assertRaises(debate.DebateToolError) as raised:
                        orchestrator.status(run_dir)
                    self.assertIn(expected, str(raised.exception))

            debate.write_json_atomic(result_path, valid_result)
            (run_dir / "results" / "round-01-agent-a.schema.json").unlink()
            with self.assertRaises(debate.DebateToolError) as missing_schema:
                orchestrator.status(run_dir)
            self.assertIn("required file is missing", str(missing_schema.exception))

    def test_resume_rejects_preexisting_task_artifact_symlinks(self) -> None:
        targets = (
            "prompts/round-01-agent-a.md",
            "results/round-01-agent-a.schema.json",
            "results/round-01-agent-a.stdout",
            "results/round-01-agent-a.stderr",
            "results/round-01-agent-a.json",
            "results/round-01-agent-a.json.tmp",
        )

        with workspace_temp_dir() as tmpdir:
            output_root = Path(tmpdir)
            for index, relative_path in enumerate(targets):
                with self.subTest(relative_path=relative_path):
                    orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))
                    started = orchestrator.start_new(
                        debate.RunOptions(
                            topic="T",
                            side_a="A",
                            side_b="B",
                            output_root=output_root,
                        )
                    )
                    run_dir = Path(started["run_dir"])
                    outside = output_root / f"outside-{index}.txt"
                    outside.write_text("sentinel", encoding="utf-8")
                    target = run_dir / relative_path
                    target.symlink_to(outside)

                    with self.assertRaises(debate.DebateRunFailedError) as raised:
                        orchestrator.resume(run_dir)

                    self.assertIn("symbolic link", str(raised.exception))
                    self.assertEqual(outside.read_text(encoding="utf-8"), "sentinel")
                    self.assertTrue(target.is_symlink())

    def test_status_rejects_retry_identity_timestamps_and_stale_errors(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))

        with workspace_temp_dir() as tmpdir:
            started = orchestrator.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    output_root=Path(tmpdir),
                )
            )
            run_dir = Path(started["run_dir"])
            config = json.loads(
                (run_dir / "run_config.json").read_text(encoding="utf-8")
            )
            detected_at = datetime.fromisoformat(config["created_at"]) + timedelta(
                seconds=1
            )
            retry_after_at = detected_at + timedelta(seconds=3600)
            failure = {
                "retryable": True,
                "reason": "rate_limit",
                "message": "retry later",
                "detected_at": detected_at.isoformat(),
                "next_step": "agent_a",
                "provider": "codex",
                "task_id": "round-01-agent-a",
                "role": "agent_a",
                "stdout_path": None,
                "stderr_path": None,
                "cli_exit_code": 1,
                "retry_after_at": retry_after_at.isoformat(),
                "retry_after_sec": 3600,
            }
            state_path = run_dir / "state.json"
            base_state = json.loads(state_path.read_text(encoding="utf-8"))
            base_state.update(
                {
                    "status": "waiting_retry",
                    "last_error": "retry later",
                    "last_failure": failure,
                    "retry": failure,
                }
            )
            cases = (
                ("provider", "claude", "provider does not match run_config"),
                ("task_id", "wrong-task", "task_id does not match next_step"),
                ("detected_at", "not-a-date", "must be an ISO-8601 timestamp"),
                (
                    "retry_after_at",
                    (detected_at - timedelta(seconds=1)).isoformat(),
                    "must follow detected_at",
                ),
                (
                    "retry_after_at",
                    (detected_at + timedelta(seconds=1)).isoformat(),
                    "must equal detected_at plus retry_after_sec",
                ),
            )
            for key, value, expected in cases:
                with self.subTest(key=key):
                    state = json.loads(json.dumps(base_state))
                    state["last_failure"][key] = value
                    state["retry"][key] = value
                    materialize_task_artifact(
                        run_dir,
                        "round-01-agent-a",
                        "agent_a",
                        status="waiting_retry",
                        failure=state["last_failure"],
                    )
                    debate.write_json_atomic(state_path, state)
                    with self.assertRaises(debate.DebateToolError) as raised:
                        orchestrator.status(run_dir)
                    self.assertIn(expected, str(raised.exception))

            valid_waiting_state = json.loads(json.dumps(base_state))
            materialize_task_artifact(
                run_dir,
                "round-01-agent-a",
                "agent_a",
                status="waiting_retry",
                failure=valid_waiting_state["last_failure"],
            )
            debate.write_json_atomic(state_path, valid_waiting_state)
            result_path = run_dir / "results" / "round-01-agent-a.json"
            task_result = json.loads(result_path.read_text(encoding="utf-8"))
            task_result["model"] = "wrong-model"
            debate.write_json_atomic(result_path, task_result)
            with self.assertRaises(debate.DebateToolError) as mismatched_model:
                orchestrator.status(run_dir)
            self.assertIn(
                "model does not match run_config",
                str(mismatched_model.exception),
            )

            materialize_task_artifact(
                run_dir,
                "round-01-agent-a",
                "agent_a",
                status="waiting_retry",
                failure=valid_waiting_state["last_failure"],
            )
            task_result = json.loads(result_path.read_text(encoding="utf-8"))
            task_result["model"] = None
            debate.write_json_atomic(result_path, task_result)
            with self.assertRaises(debate.DebateToolError) as null_model:
                orchestrator.status(run_dir)
            self.assertIn(
                "model must be a non-empty string",
                str(null_model.exception),
            )

            materialize_task_artifact(
                run_dir,
                "round-01-agent-a",
                "agent_a",
                status="waiting_retry",
                failure=valid_waiting_state["last_failure"],
            )
            task_result = json.loads(result_path.read_text(encoding="utf-8"))
            del task_result["model"]
            debate.write_json_atomic(result_path, task_result)
            self.assertEqual(
                orchestrator.status(run_dir)["status"],
                "waiting_retry",
            )

            materialize_task_artifact(
                run_dir,
                "round-01-agent-a",
                "agent_a",
                status="waiting_retry",
                failure=valid_waiting_state["last_failure"],
            )
            task_result = json.loads(result_path.read_text(encoding="utf-8"))
            task_result["failure"]["message"] = "tampered"
            debate.write_json_atomic(result_path, task_result)
            with self.assertRaises(debate.DebateToolError) as mismatched_failure:
                orchestrator.status(run_dir)
            self.assertIn("failure does not match state", str(mismatched_failure.exception))

            stale_state = json.loads(json.dumps(base_state))
            stale_state.update(
                {
                    "status": "in_progress",
                    "last_error": "stale",
                    "last_failure": {
                        "retryable": False,
                        "reason": "tool_failure",
                        "message": "stale",
                        "detected_at": "2026-09-03T00:00:00+00:00",
                        "next_step": "agent_a",
                    },
                    "retry": None,
                }
            )
            debate.write_json_atomic(state_path, stale_state)
            with self.assertRaises(debate.DebateToolError) as stale:
                orchestrator.status(run_dir)

        self.assertIn("cannot retain failure or retry metadata", str(stale.exception))

    def test_status_rejects_out_of_order_rounds(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))

        with workspace_temp_dir() as tmpdir:
            started = orchestrator.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    max_rounds=3,
                    output_root=Path(tmpdir),
                )
            )
            run_dir = Path(started["run_dir"])
            state_path = run_dir / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["current_round"] = 3
            state["rounds"] = [
                {"round_number": 2, "agent_a": None, "agent_b": None},
                {"round_number": 1, "agent_a": None, "agent_b": None},
            ]
            debate.write_json_atomic(state_path, state)

            with self.assertRaises(debate.DebateToolError) as raised:
                orchestrator.status(run_dir)

        self.assertIn("ascending contiguous order", str(raised.exception))

    def test_non_finite_poll_interval_and_create_io_errors_are_structured(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))

        with workspace_temp_dir() as tmpdir:
            output_root = Path(tmpdir) / "not-created"
            with patch.object(
                debate,
                "resolve_agent_execution_options",
                return_value={
                    "idle_timeout_sec": 30,
                    "poll_interval_sec": float("inf"),
                    "max_wall_time_sec": None,
                },
            ):
                with self.assertRaises(debate.DebateToolError) as non_finite:
                    orchestrator.start_new(
                        debate.RunOptions(
                            topic="T",
                            side_a="A",
                            side_b="B",
                            output_root=output_root,
                        )
                    )
            self.assertFalse(output_root.exists())

            with patch.object(
                debate,
                "allocate_run_dir",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaises(debate.DebateToolError) as io_error:
                    orchestrator.run_new(
                        debate.RunOptions(
                            topic="T",
                            side_a="A",
                            side_b="B",
                            output_root=Path(tmpdir),
                        )
                    )

        self.assertIn("finite and greater than zero", str(non_finite.exception))
        self.assertIn("Could not create debate run artifacts", str(io_error.exception))

    def test_async_turn_tools_reject_local_only_mode(self) -> None:
        runner = FakeRunner({})
        orchestrator = debate.DebateOrchestrator(runner=runner)

        with workspace_temp_dir() as tmpdir:
            started = orchestrator.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    access="local-cli",
                    output_root=Path(tmpdir),
                )
            )

            for operation in (
                orchestrator.turn_start,
                orchestrator.turn_status,
                orchestrator.turn_finalize,
            ):
                with self.assertRaises(debate.DebateToolError) as raised:
                    operation(started["run_dir"])
                self.assertIn(
                    "AI Debate always uses local-cli",
                    str(raised.exception),
                )

        self.assertEqual(runner.calls, [])

    def test_resume_rejects_historical_active_task_without_async_tool_direction(self) -> None:
        runner = FakeRunner({})
        orchestrator = debate.DebateOrchestrator(runner=runner)

        with workspace_temp_dir() as tmpdir:
            started = orchestrator.start_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    output_root=Path(tmpdir),
                )
            )
            run_dir = Path(started["run_dir"])
            state_path = run_dir / "state.json"
            config_path = run_dir / "run_config.json"
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            config_payload["access"] = "mcp.remote-agent-broker"
            debate.write_json_atomic(config_path, config_payload)
            state_payload = json.loads(state_path.read_text(encoding="utf-8"))
            started_at = config_payload["created_at"]
            (run_dir / "prompts" / "round-01-agent-a.md").write_text(
                "prompt",
                encoding="utf-8",
            )
            state_payload["task_attempts"] = {"round-01-agent-a": 1}
            state_payload["active_task"] = {
                "task_id": "round-01-agent-a",
                "role": "agent_a",
                "phase": "turn",
                "next_step": "agent_a",
                "round_number": 1,
                "attempt": 1,
                "backend": "mcp.remote-agent-broker",
                "access": "mcp.remote-agent-broker",
                "provider": "codex",
                "model": config_payload["roles"]["agent_a"]["model"],
                "effort": config_payload["roles"]["agent_a"]["effort"],
                "job_id": "job_20260903T000000Z_012345abcdef",
                "status": "running",
                "terminal": False,
                "finalized": False,
                "started_at": started_at,
                "updated_at": started_at,
                "prompt_path": "prompts/round-01-agent-a.md",
                "stdout_path": "results/round-01-agent-a.stdout",
                "stderr_path": "results/round-01-agent-a.stderr",
                "expected_keys": ["response_md", "signal", "signal_reason"],
                "broker_exit_code": None,
                "broker_last_error": None,
            }
            debate.write_json_atomic(state_path, state_payload)

            with self.assertRaises(debate.DebateRunFailedError) as raised:
                orchestrator.resume(run_dir)

        message = str(raised.exception)
        self.assertIn("historical state.active_task", message)
        self.assertIn("local agent context", message)
        self.assertNotIn("ai_debate_turn_status", message)
        self.assertNotIn("ai_debate_turn_finalize", message)
        self.assertEqual(runner.calls, [])

    def test_explicit_repository_root_output_path_is_accepted(self) -> None:
        runner = FakeRunner(
            {
                "round-01-agent-a": {
                    "response_md": "Agent A opens.",
                    "signal": "continue",
                    "signal_reason": "Agent B should respond.",
                },
                "round-01-agent-b": {
                    "response_md": "Agent B responds.",
                    "signal": "continue",
                    "signal_reason": "The round should be judged.",
                },
                "final-judgment": {
                    "winner": "agent_b",
                    "reason": "Agent B made the stronger case.",
                    "termination": "round_limit",
                    "decisive_round": 1,
                },
            }
        )
        orchestrator = debate.DebateOrchestrator(runner=runner)

        with tempfile.TemporaryDirectory(
            dir=debate.WORKSPACE_ROOT,
            prefix=".tmp-ai-debate-",
        ) as tmpdir:
            summary = orchestrator.run_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    max_rounds=1,
                    output_root=Path(tmpdir),
                )
            )

            self.assertEqual(summary["status"], "completed")
            self.assertTrue(Path(summary["run_dir"]).is_relative_to(Path(tmpdir)))

    def test_output_path_outside_runtime_and_repository_roots_is_rejected(self) -> None:
        outside_root = Path("/tmp/tokimon-ai-debate-output-root-rejected")
        self.assertFalse(
            any(
                debate.is_relative_to(outside_root.resolve(), root)
                for root in debate.allowed_artifact_roots()
            )
        )

        with self.assertRaises(debate.DebateToolError) as raised:
            debate.resolve_output_root(outside_root)

        self.assertIn("Debate artifacts must stay under", str(raised.exception))

    def test_invalid_configuration_does_not_create_a_run_directory(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))

        with workspace_temp_dir() as tmpdir:
            output_root = Path(tmpdir) / "not-created"
            with (
                patch.object(
                    debate,
                    "load_settings_section",
                    return_value={"access": "invalid", "peer_b": "codex"},
                ),
                self.assertRaises(debate.DebateToolError) as raised,
            ):
                orchestrator.start_new(
                    debate.RunOptions(
                        topic="T",
                        side_a="A",
                        side_b="B",
                        output_root=output_root,
                    )
                )

            self.assertIn("access must be local-cli", str(raised.exception))
            self.assertFalse(output_root.exists())

    def test_run_options_refuse_missing_ai_debate_settings(self) -> None:
        with patch.object(debate, "load_settings_section", return_value={}):
            with self.assertRaises(debate.DebateToolError) as raised:
                debate.run_options_from_mapping({"topic": "T", "side_a": "A", "side_b": "B"})

        self.assertIn("ai_debate.access is required", str(raised.exception))

    def test_round_limit_debate_writes_final_result(self) -> None:
        runner = FakeRunner(
            {
                "round-01-agent-a": {
                    "response_md": "Agent A argues for regulation.",
                    "signal": "continue",
                    "signal_reason": "More arguments remain.",
                },
                "round-01-agent-b": {
                    "response_md": "Agent B argues against regulation.",
                    "signal": "continue",
                    "signal_reason": "The round should be judged.",
                },
                "final-judgment": {
                    "winner": "agent_b",
                    "reason": "Agent B made the stronger practical case.",
                    "termination": "round_limit",
                    "decisive_round": 1,
                },
            }
        )
        orchestrator = debate.DebateOrchestrator(runner=runner)

        with workspace_temp_dir() as tmpdir:
            summary = orchestrator.run_new(
                debate.RunOptions(
                    topic="Should AI development be regulated?",
                    side_a="Yes.",
                    side_b="No.",
                    max_rounds=1,
                    output_root=Path(tmpdir),
                )
            )

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["winner"], None)
            self.assertEqual(summary["final_result"]["winner"], "agent_b")
            self.assertEqual(
                runner.calls,
                [
                    ("round-01-agent-a", "codex"),
                    ("round-01-agent-b", "claude"),
                    ("final-judgment", "codex"),
                ],
            )

            run_dir = Path(summary["run_dir"])
            debate_log = (run_dir / "debate_log.md").read_text(encoding="utf-8")
            self.assertIn("## Round 1", debate_log)
            self.assertIn("### Agent A", debate_log)
            self.assertIn("### Agent B", debate_log)
            self.assertIn("Winner: Agent B", debate_log)
            self.assertTrue((run_dir / "results" / "final-judgment.json").exists())

    def test_agent_a_concession_skips_agent_b_and_overrides_judge_winner(self) -> None:
        runner = FakeRunner(
            {
                "round-01-agent-a": {
                    "response_md": "Agent A concedes.",
                    "signal": "concede",
                    "signal_reason": "Position B is stronger.",
                },
                "final-judgment": {
                    "winner": "agent_a",
                    "reason": "Agent A explicitly conceded.",
                    "termination": "round_limit",
                    "decisive_round": 99,
                },
            }
        )
        orchestrator = debate.DebateOrchestrator(runner=runner)

        with workspace_temp_dir() as tmpdir:
            summary = orchestrator.run_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    output_root=Path(tmpdir),
                )
            )

            self.assertEqual(runner.calls, [("round-01-agent-a", "codex"), ("final-judgment", "codex")])
            self.assertEqual(summary["final_result"]["winner"], "agent_b")
            self.assertEqual(summary["final_result"]["termination"], "concession")
            self.assertNotIn(
                "### Agent B",
                Path(summary["debate_log"]).read_text(encoding="utf-8"),
            )

    def test_mutual_tie_ends_as_tie(self) -> None:
        runner = FakeRunner(
            {
                "round-01-agent-a": {
                    "response_md": "Agent A sees the case as balanced.",
                    "signal": "tie",
                    "signal_reason": "Both sides are plausible.",
                },
                "round-01-agent-b": {
                    "response_md": "Agent B agrees this is inconclusive.",
                    "signal": "tie",
                    "signal_reason": "No decisive argument remains.",
                },
                "final-judgment": {
                    "winner": "agent_a",
                    "reason": "Both agents declared a tie in the same round.",
                    "termination": "round_limit",
                    "decisive_round": 3,
                },
            }
        )
        orchestrator = debate.DebateOrchestrator(runner=runner)

        with workspace_temp_dir() as tmpdir:
            summary = orchestrator.run_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    output_root=Path(tmpdir),
                )
            )

        self.assertEqual(summary["final_result"]["winner"], "tie")
        self.assertEqual(summary["final_result"]["termination"], "mutual_tie")

    def test_retryable_agent_b_failure_waits_and_resume_retries_same_turn(self) -> None:
        runner = FakeRunner(
            {
                "round-01-agent-a": {
                    "response_md": "Agent A opens.",
                    "signal": "continue",
                    "signal_reason": "Agent B should respond.",
                },
                "round-01-agent-b": [
                    retryable_rate_limit_error,
                    {
                        "response_md": "Agent B responds after retry.",
                        "signal": "continue",
                        "signal_reason": "The round should be judged.",
                    },
                ],
                "final-judgment": {
                    "winner": "agent_b",
                    "reason": "Agent B recovered and made the stronger case.",
                    "termination": "round_limit",
                    "decisive_round": 1,
                },
            }
        )
        orchestrator = debate.DebateOrchestrator(runner=runner)

        with workspace_temp_dir() as tmpdir:
            summary = orchestrator.run_new(
                debate.RunOptions(
                    topic="T",
                    side_a="A",
                    side_b="B",
                    max_rounds=1,
                    output_root=Path(tmpdir),
                )
            )

            self.assertEqual(summary["status"], "waiting_retry")
            self.assertEqual(summary["next_step"], "agent_b")
            self.assertEqual(summary["retry"]["reason"], "rate_limit")
            self.assertEqual(summary["retry"]["retry_after_sec"], 120)
            self.assertIn("retry_after_at", summary["retry"])
            self.assertEqual(summary["retry"]["task_id"], "round-01-agent-b")

            run_dir = Path(summary["run_dir"])
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "waiting_retry")
            self.assertEqual(state["current_round"], 1)
            self.assertEqual(state["next_step"], "agent_b")
            self.assertNotIn("### Agent B", (run_dir / "debate_log.md").read_text(encoding="utf-8"))
            task_result = json.loads(
                (run_dir / "results" / "round-01-agent-b.json").read_text(encoding="utf-8")
            )
            self.assertEqual(task_result["status"], "waiting_retry")
            self.assertEqual(task_result["model"], "default")

            resumed = orchestrator.resume(run_dir)

        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(
            runner.calls,
            [
                ("round-01-agent-a", "codex"),
                ("round-01-agent-b", "claude"),
                ("round-01-agent-b", "claude"),
                ("final-judgment", "codex"),
            ],
        )
        self.assertEqual(resumed["final_result"]["winner"], "agent_b")

    def test_malformed_final_judgment_records_failed_run_summary(self) -> None:
        runner = FakeRunner(
            {
                "round-01-agent-a": {
                    "response_md": "Agent A opens.",
                    "signal": "continue",
                    "signal_reason": "Agent B should respond.",
                },
                "round-01-agent-b": {
                    "response_md": "Agent B responds.",
                    "signal": "continue",
                    "signal_reason": "The round should be judged.",
                },
                "final-judgment": {
                    "winner": "agent_b",
                    "reason": "Agent B made the stronger case.",
                    "termination": "round_limit",
                    "decisive_round": "not-an-int",
                },
            }
        )
        orchestrator = debate.DebateOrchestrator(runner=runner)

        with workspace_temp_dir() as tmpdir:
            with self.assertRaises(debate.DebateRunFailedError) as raised:
                orchestrator.run_new(
                    debate.RunOptions(
                        topic="T",
                        side_a="A",
                        side_b="B",
                        max_rounds=1,
                        output_root=Path(tmpdir),
                    )
                )

            summary = raised.exception.summary
            state = json.loads((Path(summary["run_dir"]) / "state.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(state["status"], "failed")
        self.assertIn("invalid decisive_round", summary["last_error"])
        self.assertEqual(summary["last_failure"]["reason"], "tool_failure")

    def test_unsupported_side_b_provider_is_rejected(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))

        with workspace_temp_dir() as tmpdir:
            with self.assertRaises(debate.DebateToolError) as raised:
                orchestrator.run_new(
                    debate.RunOptions(
                        topic="T",
                        side_a="A",
                        side_b="B",
                        side_b_provider="other-provider",
                        output_root=Path(tmpdir),
                    )
                )

        self.assertIn("side_b_provider must be claude or codex", str(raised.exception))


class DebateMCPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings_patch = patch.object(
            debate,
            "load_settings_section",
            side_effect=settings_section,
        )
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)
        self.full_settings_patch = patch.object(
            debate,
            "load_settings",
            return_value=settings_payload(),
        )
        self.full_settings_patch.start()
        self.addCleanup(self.full_settings_patch.stop)

    def test_mcp_lists_debate_tools_and_calls_health(self) -> None:
        server = debate.MCPServer(orchestrator=debate.DebateOrchestrator(runner=FakeRunner({})))

        listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertIsNotNone(listed)
        tool_names = {tool["name"] for tool in listed["result"]["tools"]}
        self.assertIn("ai_debate_health", tool_names)
        self.assertIn("ai_debate_run", tool_names)
        self.assertIn("ai_debate_start", tool_names)
        self.assertIn("ai_debate_resume_start", tool_names)
        self.assertIn("ai_debate_turn_start", tool_names)
        self.assertIn("ai_debate_turn_status", tool_names)
        self.assertIn("ai_debate_turn_finalize", tool_names)
        tools_by_name = {tool["name"]: tool for tool in listed["result"]["tools"]}
        for tool_name in (
            "ai_debate_turn_start",
            "ai_debate_turn_status",
            "ai_debate_turn_finalize",
        ):
            description = tools_by_name[tool_name]["description"]
            lowered = description.lower()
            self.assertIn("local-only compatibility endpoint", lowered)
            self.assertIn("fails fast", lowered)
            self.assertNotIn("async", lowered)
            self.assertNotIn("broker", lowered)
            self.assertNotIn("job handle", lowered)
            self.assertNotIn("advance debate state", lowered)
        run_tool = tools_by_name["ai_debate_run"]
        start_tool = tools_by_name["ai_debate_start"]
        self.assertEqual(
            run_tool["inputSchema"]["properties"]["side_b_provider"]["enum"],
            ["claude", "codex"],
        )
        self.assertEqual(start_tool["inputSchema"], run_tool["inputSchema"])
        self.assertEqual(
            run_tool["inputSchema"]["properties"]["access"]["enum"],
            ["local-cli"],
        )
        self.assertIn("agent_b_effort", run_tool["inputSchema"]["properties"])

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "ai_debate_health", "arguments": {}},
            }
        )
        self.assertIsNotNone(response)
        content = response["result"]["content"][0]["text"]
        payload = json.loads(content)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["peer_b"], "claude")
        self.assertEqual(payload["required_providers"], ["claude", "codex"])
        self.assertTrue(payload["providers"]["codex"]["available"])

    def test_health_requires_a_supported_peer_b_setting(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))

        for configured, expected in (
            ({"access": "local-cli"}, "ai_debate.peer_b must be claude or codex"),
            (
                {"access": "local-cli", "peer_b": "unsupported"},
                "ai_debate.peer_b must be claude or codex",
            ),
        ):
            with self.subTest(configured=configured):
                with patch.object(
                    debate,
                    "load_settings",
                    return_value={
                        "ai_debate": configured,
                    },
                ):
                    payload = orchestrator.health()

                self.assertFalse(payload["ok"])
                self.assertIn(expected, payload["settings_error"])

    def test_health_needs_no_provider_model_settings(self) -> None:
        orchestrator = debate.DebateOrchestrator(runner=FakeRunner({}))
        for peer_b in ("codex", "claude"):
            with self.subTest(peer_b=peer_b):
                with patch.object(debate, "load_settings", return_value={
                    "ai_debate": {"access": "local-cli", "peer_b": peer_b},
                    "agents": {"codex": {"model": "", "effort": ""}},
                }):
                    self.assertTrue(orchestrator.health()["ok"])

    def test_health_requires_only_codex_when_peer_b_is_codex(self) -> None:
        runner = debate.ProviderRunner(
            env={
                "PATH": "",
                "CODEX_CLI": sys.executable,
                "CLAUDE_CODE_CLI": "/definitely/missing/claude",
            }
        )
        orchestrator = debate.DebateOrchestrator(runner=runner)

        with patch.object(
            debate,
            "load_settings",
            return_value=settings_payload(peer_b="codex"),
        ):
            payload = orchestrator.health()

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["peer_b"], "codex")
        self.assertEqual(payload["required_providers"], ["codex"])
        self.assertEqual(payload["readiness_scope"], "executable-discovery")
        self.assertFalse(payload["authentication_checked"])
        self.assertTrue(payload["providers"]["codex"]["available"])
        self.assertEqual(
            payload["providers"]["codex"]["authentication"],
            "not_checked",
        )
        self.assertFalse(payload["providers"]["claude"]["available"])

    def test_mcp_rejects_arguments_outside_declared_schema(self) -> None:
        server = debate.MCPServer(
            orchestrator=debate.DebateOrchestrator(runner=FakeRunner({}))
        )

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": "unexpected",
                "method": "tools/call",
                "params": {
                    "name": "ai_debate_health",
                    "arguments": {"unexpected": True},
                },
            }
        )

        self.assertIsNotNone(response)
        self.assertTrue(response["result"]["isError"])
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertIn("arguments contains unsupported field(s): unexpected", payload["error"])

    def test_mcp_rejects_values_that_violate_declared_schema_types(self) -> None:
        server = debate.MCPServer(
            orchestrator=debate.DebateOrchestrator(runner=FakeRunner({}))
        )

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": "wrong-type",
                "method": "tools/call",
                "params": {
                    "name": "ai_debate_run",
                    "arguments": {
                        "topic": "T",
                        "side_a": "A",
                        "side_b": "B",
                        "max_rounds": "1",
                    },
                },
            }
        )

        self.assertIsNotNone(response)
        self.assertTrue(response["result"]["isError"])
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertIn("arguments.max_rounds must be an integer", payload["error"])

    def test_health_reports_invalid_provider_overrides_unavailable(self) -> None:
        runner = debate.ProviderRunner(
            env={
                "PATH": "",
                "CODEX_CLI": "/definitely/missing/codex",
                "CLAUDE_CODE_CLI": "/definitely/missing/claude",
            }
        )
        orchestrator = debate.DebateOrchestrator(runner=runner)

        payload = orchestrator.health()

        self.assertFalse(payload["ok"])
        self.assertFalse(payload["providers"]["codex"]["available"])
        self.assertFalse(payload["providers"]["claude"]["available"])
        self.assertEqual(payload["providers"]["codex"]["version"], "unavailable")

    def test_health_reports_non_executable_provider_overrides_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_cli = Path(tmpdir) / "provider-cli"
            fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
            fake_cli.chmod(0o644)
            runner = debate.ProviderRunner(
                env={
                    "PATH": "",
                    "CODEX_CLI": str(fake_cli),
                    "CLAUDE_CODE_CLI": str(fake_cli),
                }
            )
            orchestrator = debate.DebateOrchestrator(runner=runner)

            payload = orchestrator.health()

        self.assertFalse(payload["ok"])
        self.assertFalse(payload["providers"]["codex"]["available"])
        self.assertFalse(payload["providers"]["claude"]["available"])
        self.assertEqual(payload["providers"]["codex"]["version"], "unavailable")

    def test_stdio_server_emits_structured_correlated_logs_to_stderr(self) -> None:
        logger = logging.getLogger(debate.LOGGER_NAME)
        previous_handlers = list(logger.handlers)
        previous_level = logger.level
        previous_propagate = logger.propagate
        previous_configured = getattr(logger, "_tokimon_configured", None)

        def restore_logger() -> None:
            logger.handlers.clear()
            for handler in previous_handlers:
                logger.addHandler(handler)
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate
            if previous_configured is None:
                if hasattr(logger, "_tokimon_configured"):
                    delattr(logger, "_tokimon_configured")
            else:
                setattr(logger, "_tokimon_configured", previous_configured)

        self.addCleanup(restore_logger)
        stderr = io.StringIO()
        debate.configure_runtime_logger(stream=stderr)
        stdout = io.StringIO()

        exit_code = debate.serve(
            input_stream=io.StringIO('{"jsonrpc":"2.0","id":"ping-1","method":"ping"}\n'),
            output_stream=stdout,
            server=debate.MCPServer(orchestrator=debate.DebateOrchestrator(runner=FakeRunner({}))),
        )

        self.assertEqual(exit_code, 0)
        response = json.loads(stdout.getvalue())
        self.assertEqual(response["result"], {})
        logs = [json.loads(line) for line in stderr.getvalue().splitlines()]
        self.assertGreaterEqual(len(logs), 2)
        self.assertEqual(logs[0]["message"], "server_start")
        self.assertEqual(logs[0]["service"], "tokimon-ai-debate")
        self.assertEqual(logs[0]["runtime"], "stdio-mcp")
        self.assertEqual(logs[1]["message"], "request_received")
        self.assertEqual(logs[1]["correlation_id"], "ping-1")
        self.assertEqual(logs[1]["request_id"], "ping-1")
        self.assertEqual(logs[1]["method"], "ping")

    def test_mcp_call_returns_tool_error_for_missing_run_dir(self) -> None:
        server = debate.MCPServer(orchestrator=debate.DebateOrchestrator(runner=FakeRunner({})))

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "ai_debate_status", "arguments": {}},
            }
        )

        self.assertIsNotNone(response)
        self.assertTrue(response["result"]["isError"])
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertIn("run_dir is required", payload["error"])

    def test_mcp_run_retryable_failure_returns_waiting_retry(self) -> None:
        runner = FakeRunner(
            {
                "round-01-agent-a": {
                    "response_md": "Agent A opens.",
                    "signal": "continue",
                    "signal_reason": "Agent B should respond.",
                },
                "round-01-agent-b": retryable_rate_limit_error,
            }
        )
        server = debate.MCPServer(orchestrator=debate.DebateOrchestrator(runner=runner))

        with workspace_temp_dir() as tmpdir:
            response = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "ai_debate_run",
                        "arguments": {
                            "topic": "T",
                            "side_a": "A",
                            "side_b": "B",
                            "max_rounds": 1,
                            "output_root": tmpdir,
                        },
                    },
                }
            )

            self.assertIsNotNone(response)
            self.assertNotIn("isError", response["result"])
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertEqual(payload["status"], "waiting_retry")
            self.assertEqual(payload["next_step"], "agent_b")
            self.assertTrue(Path(payload["run_dir"]).exists())
            self.assertEqual(payload["retry"]["reason"], "rate_limit")

    def test_mcp_run_nonretryable_failure_includes_run_summary(self) -> None:
        runner = FakeRunner({"round-01-agent-a": nonretryable_provider_error})
        server = debate.MCPServer(orchestrator=debate.DebateOrchestrator(runner=runner))

        with workspace_temp_dir() as tmpdir:
            response = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {
                        "name": "ai_debate_run",
                        "arguments": {
                            "topic": "T",
                            "side_a": "A",
                            "side_b": "B",
                            "output_root": tmpdir,
                        },
                    },
                }
            )

            self.assertIsNotNone(response)
            self.assertTrue(response["result"]["isError"])
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertIn("error", payload)
            self.assertEqual(payload["run_dir"], payload["run"]["run_dir"])
            self.assertEqual(payload["run"]["status"], "failed")
            self.assertTrue(Path(payload["run"]["run_dir"]).exists())
            self.assertFalse(payload["run"]["last_failure"]["retryable"])

    def test_mcp_start_and_resume_start_return_handle_without_executing_turns(self) -> None:
        runner = FakeRunner(
            {
                "round-01-agent-a": {
                    "response_md": "Agent A opens.",
                    "signal": "continue",
                    "signal_reason": "Agent B should respond.",
                },
                "round-01-agent-b": {
                    "response_md": "Agent B responds.",
                    "signal": "continue",
                    "signal_reason": "The round should be judged.",
                },
                "final-judgment": {
                    "winner": "agent_b",
                    "reason": "Agent B made the stronger case.",
                    "termination": "round_limit",
                    "decisive_round": 1,
                },
            }
        )
        server = debate.MCPServer(orchestrator=debate.DebateOrchestrator(runner=runner))

        with workspace_temp_dir() as tmpdir:
            start_response = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {
                        "name": "ai_debate_start",
                        "arguments": {
                            "topic": "T",
                            "side_a": "A",
                            "side_b": "B",
                            "max_rounds": 1,
                            "output_root": tmpdir,
                        },
                    },
                }
            )

            self.assertIsNotNone(start_response)
            start_payload = json.loads(start_response["result"]["content"][0]["text"])
            run_dir = Path(start_payload["run_dir"])
            self.assertEqual(start_payload["status"], "in_progress")
            self.assertEqual(start_payload["next_step"], "agent_a")
            self.assertTrue(run_dir.exists())
            self.assertEqual(runner.calls, [])

            resume_start_response = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {
                        "name": "ai_debate_resume_start",
                        "arguments": {"run_dir": str(run_dir)},
                    },
                }
            )

            self.assertIsNotNone(resume_start_response)
            resume_start_payload = json.loads(
                resume_start_response["result"]["content"][0]["text"]
            )
            self.assertEqual(resume_start_payload["run_dir"], str(run_dir))
            self.assertEqual(resume_start_payload["next_step"], "agent_a")
            self.assertEqual(runner.calls, [])

            resume_response = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 8,
                    "method": "tools/call",
                    "params": {
                        "name": "ai_debate_resume",
                        "arguments": {"run_dir": str(run_dir)},
                    },
                }
            )

        self.assertIsNotNone(resume_response)
        resume_payload = json.loads(resume_response["result"]["content"][0]["text"])
        self.assertEqual(resume_payload["status"], "completed")
        self.assertEqual(
            runner.calls,
            [
                ("round-01-agent-a", "codex"),
                ("round-01-agent-b", "claude"),
                ("final-judgment", "codex"),
            ],
        )


class DebateParsingTests(unittest.TestCase):
    def test_cli_default_markers_are_not_passed_as_provider_flags(self) -> None:
        runner = debate.ProviderRunner(env={"PATH": ""})
        for provider in ("codex", "claude"):
            with self.subTest(provider=provider):
                cmd, _ = runner._build_command(
                    executable=f"/fake/{provider}", provider=provider,
                    model="default", effort="default", prompt="Prompt",
                    workspace_dir=ROOT, schema={"type": "object"},
                    schema_path=ROOT / "schema.json",
                )
                self.assertNotIn("--model", cmd)
                self.assertNotIn("--effort", cmd)
                self.assertFalse(any("model_reasoning_effort" in item for item in cmd))

    def test_codex_command_reads_prompt_from_stdin(self) -> None:
        runner = debate.ProviderRunner(env={"PATH": ""})
        cmd, stdin_text = runner._build_command(
            executable="/fake/codex",
            provider="codex",
            model="gpt-test",
            effort=None,
            prompt="Long prompt",
            workspace_dir=ROOT,
            schema={"type": "object"},
            schema_path=ROOT / "schema.json",
        )

        self.assertEqual(cmd[-1], "-")
        self.assertEqual(stdin_text, "Long prompt")
        self.assertNotIn("Long prompt", cmd)
        self.assertEqual(cmd[cmd.index("--sandbox") + 1], "read-only")

    def test_codex_command_receives_reasoning_effort(self) -> None:
        runner = debate.ProviderRunner(env={"PATH": ""})
        cmd, _ = runner._build_command(
            executable="/fake/codex",
            provider="codex",
            model="gpt-test",
            effort="xhigh",
            prompt="Long prompt",
            workspace_dir=ROOT,
            schema={"type": "object"},
            schema_path=ROOT / "schema.json",
        )

        self.assertIn("--config", cmd)
        self.assertIn('model_reasoning_effort="xhigh"', cmd)

    def test_claude_command_disables_tools_without_bypassing_permissions(self) -> None:
        runner = debate.ProviderRunner(env={"PATH": ""})
        cmd, stdin_text = runner._build_command(
            executable="/fake/claude",
            provider="claude",
            model="claude-test",
            effort="max",
            prompt="Long prompt",
            workspace_dir=ROOT,
            schema={"type": "object"},
            schema_path=ROOT / "schema.json",
        )

        self.assertEqual(stdin_text, "Long prompt")
        self.assertNotIn("--dangerously-skip-permissions", cmd)
        self.assertIn("--safe-mode", cmd)
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "dontAsk")
        self.assertEqual(cmd[cmd.index("--tools") + 1], "")
        self.assertIn("--no-session-persistence", cmd)

    def test_provider_payload_validation_rejects_type_coercion(self) -> None:
        valid_turn = {
            "response_md": "Argument",
            "signal": "continue",
            "signal_reason": "More remains.",
            "citations": [],
            "files_created": [],
        }
        for field, value in (
            ("response_md", ["not", "text"]),
            ("signal", 1),
            ("signal_reason", True),
            ("citations", [123]),
            ("files_created", {"path": "x"}),
        ):
            with self.subTest(field=field):
                payload = dict(valid_turn)
                payload[field] = value
                with self.assertRaises(debate.DebateToolError):
                    debate.validate_turn_payload(payload)

        valid_state = {
            "termination": "round_limit",
            "current_round": 1,
        }
        with self.assertRaises(debate.DebateToolError) as wrong_type:
            debate.validate_final_payload(
                {
                    "winner": "agent_a",
                    "reason": "A wins.",
                    "termination": "round_limit",
                    "decisive_round": "1",
                },
                valid_state,
            )
        with self.assertRaises(debate.DebateToolError) as wrong_round:
            debate.validate_final_payload(
                {
                    "winner": "agent_a",
                    "reason": "A wins.",
                    "termination": "round_limit",
                    "decisive_round": 99,
                },
                valid_state,
            )

        self.assertIn("invalid decisive_round", str(wrong_type.exception))
        self.assertIn("must match current_round", str(wrong_round.exception))

    def test_historical_object_compatibility_is_limited_to_citations(self) -> None:
        citation = {"url": "https://example.test", "title": "Example"}
        payload = {
            "response_md": "Argument",
            "signal": "continue",
            "signal_reason": "More remains.",
            "citations": [citation],
            "files_created": [],
        }

        normalized = debate.normalize_persisted_task_response(
            payload,
            Path("historical-result.json"),
            role="agent_a",
            allow_historical_objects=True,
        )
        self.assertEqual(normalized["citations"], [str(citation)])

        payload["files_created"] = [{"path": "report.md"}]
        with self.assertRaises(debate.DebateToolError) as raised:
            debate.normalize_persisted_task_response(
                payload,
                Path("historical-result.json"),
                role="agent_a",
                allow_historical_objects=True,
            )

        self.assertIn("response_json.files_created must contain strings", str(raised.exception))

    def test_blocked_stdin_cannot_bypass_timeout(self) -> None:
        class BlockingStdin:
            def __init__(self) -> None:
                self.started = False
                self.released = threading.Event()

            def write(self, _: str) -> None:
                self.started = True
                self.released.wait()

            def flush(self) -> None:
                return None

            def close(self) -> None:
                self.released.set()

        class BlockedProcess:
            def __init__(self) -> None:
                self.stdin = BlockingStdin()
                self.stdout = io.StringIO("")
                self.stderr = io.StringIO("")
                self.returncode: int | None = None
                self.pid = 12345

            def poll(self) -> int | None:
                return self.returncode

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                assert self.returncode is not None
                return self.returncode

            def stop(self) -> None:
                self.returncode = -15
                self.stdin.released.set()

        clock = {"now": 0.0}

        def advance(interval: float) -> None:
            clock["now"] += interval

        runner = debate.ProviderRunner(
            env={"PATH": ""},
            monotonic=lambda: clock["now"],
            sleeper=advance,
        )
        process = BlockedProcess()

        with workspace_temp_dir() as tmpdir:
            with (
                patch.object(runner, "resolve_executable", return_value="/fake/claude"),
                patch.object(debate.subprocess, "Popen", return_value=process),
                patch.object(
                    debate,
                    "terminate_provider_process",
                    side_effect=lambda target, force: target.stop(),
                ),
                patch.object(
                    debate,
                    "terminate_exited_provider_group",
                ) as force_group_cleanup,
            ):
                with self.assertRaises(debate.ProviderCallError) as raised:
                    runner.run_json(
                        provider="claude",
                        model="default",
                        prompt="x" * 1_000_000,
                        timeout_sec=0.05,
                        workspace_dir=Path(tmpdir),
                        task_id="blocked-stdin",
                        results_dir=Path(tmpdir) / "results",
                        expected_keys=("response_md", "signal", "signal_reason"),
                        poll_interval_sec=0.05,
                    )

        self.assertTrue(process.stdin.started)
        self.assertIn("idle timed out", str(raised.exception))
        force_group_cleanup.assert_called_once_with(process, force=True)

    def test_successful_provider_drains_reader_threads_before_json_parsing(self) -> None:
        expected = {
            "response_md": "Complete output",
            "signal": "continue",
            "signal_reason": "More remains.",
            "citations": [],
            "files_created": [],
        }

        class FinishedProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO(json.dumps(expected))
                self.stderr = io.StringIO("")
                self.returncode = 0
                self.pid = 12345

            def poll(self) -> int:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                return 0

        class DeferredThread:
            def __init__(
                self,
                *,
                target: Any,
                args: tuple[Any, ...],
                daemon: bool,
            ) -> None:
                del daemon
                self.target = target
                self.args = args
                self.finished = False

            def start(self) -> None:
                return None

            def join(self, timeout: float | None = None) -> None:
                del timeout
                if not self.finished:
                    self.target(*self.args)
                    self.finished = True

            def is_alive(self) -> bool:
                return not self.finished

        runner = debate.ProviderRunner(env={"PATH": ""})
        process = FinishedProcess()
        with workspace_temp_dir() as tmpdir:
            with (
                patch.object(runner, "resolve_executable", return_value="/fake/codex"),
                patch.object(debate.subprocess, "Popen", return_value=process),
                patch.object(debate.threading, "Thread", DeferredThread),
            ):
                result = runner.run_json(
                    provider="codex",
                    model="gpt-test",
                    prompt="prompt",
                    timeout_sec=30,
                    workspace_dir=Path(tmpdir),
                    task_id="drain-readers",
                    results_dir=Path(tmpdir) / "results",
                    expected_keys=("response_md", "signal", "signal_reason"),
                )

        self.assertEqual(result.payload, expected)

    def test_provider_decode_failure_cannot_publish_valid_truncated_prefix(self) -> None:
        expected = {"response_md": "Complete", "signal": "continue", "signal_reason": "More"}
        encoded = (json.dumps(expected) + " " * 10000).encode() + b"\xff"

        class FinishedProcess:
            stdin = io.StringIO()
            stdout = io.TextIOWrapper(io.BytesIO(encoded), encoding="utf-8")
            stderr = io.StringIO("")
            returncode = 0
            pid = 12345

            def poll(self) -> int:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        runner = debate.ProviderRunner(env={"PATH": ""})
        with (
            workspace_temp_dir() as tmpdir,
            patch.object(runner, "resolve_executable", return_value="/fake/codex"),
            patch.object(debate.subprocess, "Popen", return_value=FinishedProcess()),
            patch.object(debate.threading, "excepthook") as thread_error,
        ):
            with self.assertRaises(debate.ProviderCallError) as failed:
                runner.run_json(
                    provider="codex", model="default", prompt="prompt", timeout_sec=30,
                    workspace_dir=Path(tmpdir), task_id="decode-failure",
                    results_dir=Path(tmpdir) / "results",
                    expected_keys=("response_md", "signal", "signal_reason"),
                )
            self.assertIn("Could not capture provider stream", str(failed.exception))
            self.assertEqual(failed.exception.cli_exit_code, 0)
            prefix = failed.exception.stdout_path.read_text()
            self.assertEqual(json.loads(prefix), expected)
            thread_error.assert_not_called()

    def test_invalid_provider_json_retains_provider_failure_metadata(self) -> None:
        class FinishedProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO("not structured JSON")
                self.stderr = io.StringIO("")
                self.returncode = 0
                self.pid = 12345

            def poll(self) -> int:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                return 0

        runner = debate.ProviderRunner(env={"PATH": ""})
        with workspace_temp_dir() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            with (
                patch.object(runner, "resolve_executable", return_value="/fake/codex"),
                patch.object(
                    debate.subprocess,
                    "Popen",
                    return_value=FinishedProcess(),
                ),
                self.assertRaises(debate.ProviderCallError) as raised,
            ):
                runner.run_json(
                    provider="codex",
                    model="gpt-test",
                    prompt="prompt",
                    timeout_sec=30,
                    workspace_dir=Path(tmpdir),
                    task_id="invalid-json",
                    results_dir=results_dir,
                    expected_keys=("response_md", "signal", "signal_reason"),
                )

            error = raised.exception
            self.assertEqual(error.provider, "codex")
            self.assertEqual(error.task_id, "invalid-json")
            self.assertEqual(error.stdout_path, results_dir / "invalid-json.stdout")
            self.assertEqual(error.stderr_path, results_dir / "invalid-json.stderr")
            self.assertEqual(error.cli_exit_code, 0)
            self.assertIn("invalid structured output", str(error))

    def test_successful_provider_cleans_up_an_unfinished_stdin_writer(self) -> None:
        expected = {
            "response_md": "Complete output",
            "signal": "continue",
            "signal_reason": "More remains.",
            "citations": [],
            "files_created": [],
        }

        class FinishedProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO(json.dumps(expected))
                self.stderr = io.StringIO("")
                self.returncode = 0
                self.pid = 12345

            def poll(self) -> int:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                return 0

        class ControlledThread:
            stdin_thread: "ControlledThread | None" = None

            def __init__(
                self,
                *,
                target: Any,
                args: tuple[Any, ...],
                daemon: bool,
            ) -> None:
                del daemon
                self.target = target
                self.args = args
                self.alive = target is debate.pump_provider_stdin
                if self.alive:
                    type(self).stdin_thread = self

            def start(self) -> None:
                if not self.alive:
                    self.target(*self.args)

            def join(self, timeout: float | None = None) -> None:
                del timeout

            def is_alive(self) -> bool:
                return self.alive

        process = FinishedProcess()

        def force_cleanup(target: Any, *, force: bool) -> None:
            self.assertIs(target, process)
            self.assertTrue(force)
            assert ControlledThread.stdin_thread is not None
            ControlledThread.stdin_thread.alive = False

        runner = debate.ProviderRunner(env={"PATH": ""})
        with workspace_temp_dir() as tmpdir:
            with (
                patch.object(runner, "resolve_executable", return_value="/fake/codex"),
                patch.object(debate.subprocess, "Popen", return_value=process),
                patch.object(debate.threading, "Thread", ControlledThread),
                patch.object(
                    debate,
                    "terminate_exited_provider_group",
                    side_effect=force_cleanup,
                ) as terminate,
                self.assertRaises(debate.ProviderCallError) as raised,
            ):
                runner.run_json(
                    provider="codex",
                    model="gpt-test",
                    prompt="prompt",
                    timeout_sec=30,
                    workspace_dir=Path(tmpdir),
                    task_id="unfinished-stdin",
                    results_dir=Path(tmpdir) / "results",
                    expected_keys=("response_md", "signal", "signal_reason"),
                )

        self.assertIn("before stdin delivery completed", str(raised.exception))
        terminate.assert_called_once_with(process, force=True)

    def test_provider_output_drain_respects_total_wall_timeout(self) -> None:
        class FinishedProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO("{}")
                self.stderr = io.StringIO("")
                self.returncode = 0
                self.pid = 99999999

            def poll(self) -> int:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                return 0

        class StuckThread:
            def __init__(self, **_: Any) -> None:
                self.alive = True

            def start(self) -> None:
                return None

            def join(self, timeout: float | None = None) -> None:
                del timeout

            def is_alive(self) -> bool:
                return self.alive

        clock = {"now": 0.0}

        def advance(interval: float) -> None:
            clock["now"] += interval

        runner = debate.ProviderRunner(
            env={"PATH": ""},
            monotonic=lambda: clock["now"],
            sleeper=advance,
        )
        with workspace_temp_dir() as tmpdir:
            with (
                patch.object(runner, "resolve_executable", return_value="/fake/codex"),
                patch.object(
                    debate.subprocess,
                    "Popen",
                    return_value=FinishedProcess(),
                ),
                patch.object(debate.threading, "Thread", StuckThread),
                patch.object(debate, "terminate_exited_provider_group") as terminate,
                self.assertRaises(debate.ProviderCallError) as raised,
            ):
                runner.run_json(
                    provider="codex",
                    model="gpt-test",
                    prompt="prompt",
                    timeout_sec=30,
                    workspace_dir=Path(tmpdir),
                    task_id="stuck-drain",
                    results_dir=Path(tmpdir) / "results",
                    expected_keys=("response_md", "signal", "signal_reason"),
                    poll_interval_sec=0.05,
                    max_wall_time_sec=0.1,
                )

        self.assertIn("output drain timed out", str(raised.exception))
        terminate.assert_called_once()

    def test_exited_provider_group_cleanup_still_targets_descendants(self) -> None:
        class ExitedProcess:
            pid = 12345

            def poll(self) -> int:
                return 0

        process = ExitedProcess()
        with patch.object(debate.os, "killpg") as kill_group:
            debate.terminate_exited_provider_group(process, force=True)  # type: ignore[arg-type]

        kill_group.assert_called_once_with(process.pid, debate.signal.SIGKILL)

    def test_provider_runner_rejects_symlink_output_before_spawn(self) -> None:
        runner = debate.ProviderRunner(env={"PATH": ""})

        with workspace_temp_dir() as tmpdir:
            run_dir = Path(tmpdir)
            results_dir = run_dir / "results"
            results_dir.mkdir()
            outside = run_dir.parent / f"{run_dir.name}-outside.txt"
            outside.write_text("sentinel", encoding="utf-8")
            self.addCleanup(outside.unlink, missing_ok=True)
            (results_dir / "symlink.schema.json").symlink_to(outside)

            with (
                patch.object(runner, "resolve_executable", return_value="/fake/codex"),
                patch.object(debate.subprocess, "Popen") as popen,
                self.assertRaises(debate.ProviderCallError) as raised,
            ):
                runner.run_json(
                    provider="codex",
                    model="gpt-test",
                    prompt="prompt",
                    timeout_sec=30,
                    workspace_dir=run_dir,
                    task_id="symlink",
                    results_dir=results_dir,
                    expected_keys=("response_md", "signal", "signal_reason"),
                )

            popen.assert_not_called()
            self.assertEqual(outside.read_text(encoding="utf-8"), "sentinel")
            self.assertIn("symbolic link", str(raised.exception))

    def test_streaming_activity_prevents_idle_timeout_without_real_sleep(self) -> None:
        class SequencedProcess:
            def __init__(self) -> None:
                self.poll_count = 0

            def poll(self) -> int | None:
                self.poll_count += 1
                return 0 if self.poll_count >= 4 else None

        clock = {"now": 0.0}
        last_output = {"monotonic": 0.0}

        def advance_with_output(interval: float) -> None:
            clock["now"] += interval
            last_output["monotonic"] = clock["now"]

        runner = debate.ProviderRunner(
            env={"PATH": ""},
            monotonic=lambda: clock["now"],
            sleeper=advance_with_output,
        )
        timeout = runner._wait_for_provider_process(
            SequencedProcess(),  # type: ignore[arg-type]
            provider="claude",
            task_id="streaming",
            last_output=last_output,
            timeout_sec=1,
            poll_interval_sec=0.6,
            max_wall_time_sec=None,
        )

        self.assertIsNone(timeout)

    def test_idle_timeout_is_deterministic_without_real_sleep(self) -> None:
        class RunningProcess:
            def poll(self) -> None:
                return None

        clock = {"now": 0.0}

        def advance(interval: float) -> None:
            clock["now"] += interval

        runner = debate.ProviderRunner(
            env={"PATH": ""},
            monotonic=lambda: clock["now"],
            sleeper=advance,
        )
        timeout = runner._wait_for_provider_process(
            RunningProcess(),  # type: ignore[arg-type]
            provider="claude",
            task_id="idle",
            last_output={"monotonic": 0.0},
            timeout_sec=1,
            poll_interval_sec=0.6,
            max_wall_time_sec=None,
        )

        self.assertEqual(
            timeout,
            "claude task idle idle timed out after 1s without output",
        )
        self.assertEqual(clock["now"], 1.0)

    def test_max_wall_timeout_caps_the_poll_sleep_to_the_remaining_deadline(self) -> None:
        class RunningProcess:
            def poll(self) -> None:
                return None

        clock = {"now": 0.0}

        def advance(interval: float) -> None:
            clock["now"] += interval

        runner = debate.ProviderRunner(
            env={"PATH": ""},
            monotonic=lambda: clock["now"],
            sleeper=advance,
        )
        timeout = runner._wait_for_provider_process(
            RunningProcess(),  # type: ignore[arg-type]
            provider="codex",
            task_id="wall",
            last_output={"monotonic": 0.0},
            timeout_sec=30,
            poll_interval_sec=5,
            max_wall_time_sec=2,
        )

        self.assertEqual(timeout, "codex task wall timed out after 2s")
        self.assertEqual(clock["now"], 2.0)

    def test_unsupported_provider_command_is_rejected(self) -> None:
        runner = debate.ProviderRunner(env={"PATH": ""})
        with self.assertRaises(debate.DebateToolError) as raised:
            runner._build_command(
                executable="/fake/other-provider",
                provider="other-provider",
                model="gpt-5.5",
                effort="xhigh",
                prompt="Long prompt",
                workspace_dir=ROOT,
                schema={"type": "object"},
                schema_path=ROOT / "schema.json",
            )

        self.assertIn("Unsupported provider: other-provider", str(raised.exception))

    def test_turn_schema_requires_all_declared_properties_for_codex_strict_schema(self) -> None:
        schema = debate.schema_for_expected_keys(("response_md", "signal", "signal_reason"))

        self.assertEqual(
            set(schema["required"]),
            {"response_md", "signal", "signal_reason", "citations", "files_created"},
        )

    def test_extracts_json_from_claude_wrapper(self) -> None:
        payload = {
            "type": "result",
            "result": '{"response_md": "hello", "signal": "continue", "signal_reason": "more"}',
        }

        parsed = debate.extract_provider_json(
            json.dumps(payload),
            expected_keys=("response_md", "signal", "signal_reason"),
        )

        self.assertEqual(parsed["response_md"], "hello")

    def test_extracts_json_from_codex_events(self) -> None:
        event = {
            "item": {
                "type": "agent_message",
                "text": '{"winner": "tie", "reason": "balanced", "termination": "round_limit", "decisive_round": 1}',
            }
        }

        parsed = debate.extract_provider_json(
            json.dumps({"noise": True}) + "\n" + json.dumps(event),
            expected_keys=("winner", "reason", "termination", "decisive_round"),
        )

        self.assertEqual(parsed["winner"], "tie")

    def test_extracts_json_from_assistant_message_events(self) -> None:
        event = {
            "type": "assistant.message",
            "data": {
                "content": '{"response_md": "ok", "signal": "continue", "signal_reason": "probe"}'
            },
        }

        parsed = debate.extract_provider_json(
            json.dumps(event),
            expected_keys=("response_md", "signal", "signal_reason"),
        )

        self.assertEqual(parsed["response_md"], "ok")

    def test_classifies_retryable_provider_exhaustion(self) -> None:
        rate = debate.classify_provider_retry_failure("Rate limit exceeded. Retry-After: 90")
        quota = debate.classify_provider_retry_failure("usage limit reached")
        tokens = debate.classify_provider_retry_failure("tokens exhausted")
        token_quota = debate.classify_provider_retry_failure("token quota exceeded")
        tokens_with_retry = debate.classify_provider_retry_failure(
            "tokens exhausted; retry later"
        )

        self.assertEqual(rate, {"reason": "rate_limit", "retry_after_sec": 90})
        self.assertEqual(quota, {"reason": "quota_exhausted", "retry_after_sec": 3600})
        self.assertEqual(tokens, {"reason": "token_exhausted", "retry_after_sec": 3600})
        self.assertEqual(
            token_quota,
            {"reason": "token_exhausted", "retry_after_sec": 3600},
        )
        self.assertEqual(
            tokens_with_retry,
            {"reason": "token_exhausted", "retry_after_sec": 3600},
        )


if __name__ == "__main__":
    unittest.main()
