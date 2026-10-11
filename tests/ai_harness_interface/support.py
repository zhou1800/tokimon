"""Shared test support for AI Harness Interface unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tokimon.ai_harness_interface import claude_cli, plugin_creator


def without_local_claude_cli() -> mock._patch_dict:
    """Keep unit tests hermetic: the Claude CLI adapter must resolve no executable."""
    return mock.patch.dict(
        "os.environ",
        {claude_cli.EXECUTABLE_ENV: "claude-cli-not-installed-for-tests", claude_cli.REQUIRE_ENV: ""},
    )


def without_local_plugin_creator() -> mock._patch_dict:
    """Keep unit tests hermetic: the plugin-creator adapter must resolve no scripts."""
    return mock.patch.dict(
        "os.environ",
        {
            plugin_creator.SCRIPTS_DIR_ENV: "/nonexistent/plugin-creator-scripts-for-tests",
            plugin_creator.REQUIRE_ENV: "",
        },
    )
