from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tokimon import settings


class TokimonSettingsTests(unittest.TestCase):
    def test_missing_settings_file_returns_empty_for_optional_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            section = settings.load_settings_section("custom", workspace_root=Path(tmp))

        self.assertEqual(section, {})

    def test_missing_settings_file_raises_for_required_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(settings.TokimonSettingsError):
                settings.load_settings_section(
                    "custom",
                    workspace_root=Path(tmp),
                    required=True,
                )

    def test_loads_settings_from_tokimon_runtime_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".tokimon" / "settings.local.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps({"custom": {"enabled": True}}),
                encoding="utf-8",
            )

            section = settings.load_settings_section("custom", workspace_root=Path(tmp))

        self.assertEqual(section, {"enabled": True})

    def test_rejects_non_object_settings_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".tokimon" / "settings.local.json"
            path.parent.mkdir(parents=True)
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(settings.TokimonSettingsError):
                settings.load_settings(workspace_root=Path(tmp))

    def test_rejects_non_object_settings_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".tokimon" / "settings.local.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"custom": "invalid"}), encoding="utf-8")

            with self.assertRaises(settings.TokimonSettingsError):
                settings.load_settings_section("custom", workspace_root=Path(tmp))

    def test_ignores_legacy_agent_settings_and_preserves_caller_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".tokimon" / "settings.local.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "codex": {"model": "gpt-config", "effort": "xhigh"},
                            "claude": {"model": "claude-config", "effort": "max"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                settings.resolve_agent_options("codex", {}, workspace_root=Path(tmp)),
                {},
            )
            self.assertEqual(
                settings.resolve_agent_options(
                    "claude",
                    {"model": "custom-claude", "reasoning_effort": "high"},
                    workspace_root=Path(tmp),
                ),
                {"model": "custom-claude", "reasoning_effort": "high"},
            )

    def test_agent_options_need_no_settings_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for agent in ("codex", "claude"):
                self.assertEqual(
                    settings.resolve_agent_options(agent, {}, workspace_root=Path(tmp)), {},
                )

    def test_resolves_agent_execution_defaults_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                settings.resolve_agent_execution_options(workspace_root=Path(tmp)),
                {
                    "idle_timeout_sec": 3600,
                    "poll_interval_sec": 5.0,
                    "max_wall_time_sec": None,
                },
            )

            path = Path(tmp) / ".tokimon" / "settings.local.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "agent_execution": {
                            "idle_timeout_sec": 120,
                            "poll_interval_sec": 0.5,
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                settings.resolve_agent_execution_options(
                    {"idle_timeout_sec": 240, "max_wall_time_sec": 3600},
                    workspace_root=Path(tmp),
                ),
                {
                    "idle_timeout_sec": 240,
                    "poll_interval_sec": 0.5,
                    "max_wall_time_sec": 3600,
                },
            )

    def test_execution_options_reject_nonfinite_poll_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for value in (float("nan"), float("inf"), float("-inf"), "NaN", "Infinity"):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(settings.TokimonSettingsError, "poll_interval_sec.*finite"):
                        settings.resolve_agent_execution_options({"poll_interval_sec": value}, workspace_root=root)
            path = root / ".tokimon" / "settings.local.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"agent_execution": {"poll_interval_sec": float("inf")}}))
            with self.assertRaisesRegex(settings.TokimonSettingsError, "poll_interval_sec.*finite"):
                settings.resolve_agent_execution_options(workspace_root=root)

    def test_settings_template_contains_required_sections(self) -> None:
        path = Path("tokimon/settings.example.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            set(payload),
            {"ai_debate", "agent_execution", "pair", "installation"},
        )
        self.assertNotIn("agents", payload)
        self.assertEqual(payload["installation"]["docker_features"], "auto")
        self.assertEqual(payload["ai_debate"]["access"], "local-cli")
        self.assertEqual(payload["ai_debate"]["peer_b"], "codex")
        self.assertEqual(payload["agent_execution"]["idle_timeout_sec"], 3600)
        self.assertIn("poll_interval_sec", payload["agent_execution"])
        self.assertIn("max_wall_time_sec", payload["agent_execution"])
        self.assertEqual(payload["pair"]["access"], "native-subagent")
        self.assertEqual(payload["pair"]["peer_b"], "codex")
        self.assertIn("cwd", payload["pair"])


if __name__ == "__main__":
    unittest.main()
