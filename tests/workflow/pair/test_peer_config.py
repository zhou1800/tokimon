from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.pair import peer_config


def pair_settings(
    *,
    access: str = "local-cli",
    peer_b: str = "claude",
    cwd: str = "/workspace/tokimon",
) -> dict[str, object]:
    return {
        "access": access,
        "peer_b": peer_b,
        "cwd": cwd,
    }


class PairPeerConfigTests(unittest.TestCase):
    def test_explicit_task_directory_loop_is_a_selection_input_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = Path(tmp) / "loop"
            loop.symlink_to("loop", target_is_directory=True)
            settings = pair_settings()
            before = dict(settings)
            with self.assertRaisesRegex(peer_config.PeerSelectionConfigError, "cannot be resolved"):
                peer_config.select_peers(settings, cwd=loop)
            self.assertEqual(settings, before)

    def test_explicit_task_directory_io_failure_is_a_selection_input_error(self) -> None:
        settings = pair_settings()
        before = dict(settings)
        with mock.patch.object(Path, "resolve", side_effect=PermissionError("synthetic directory resolution denied")):
            with self.assertRaisesRegex(peer_config.PeerSelectionConfigError, "cannot be resolved"):
                peer_config.select_peers(settings, cwd="selected-checkout")
        self.assertEqual(settings, before)

    def test_task_directory_override_leaves_settings_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_checkout = Path(tmp) / "task-checkout"
            task_checkout.mkdir()
            settings = pair_settings()

            peers, _ = peer_config.select_peers(settings, cwd=task_checkout)

            self.assertEqual([peers[slot]["cwd"] for slot in ("a", "b")], [str(task_checkout)] * 2)
            self.assertEqual(settings["cwd"], "/workspace/tokimon")

    def test_invalid_task_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for directory in ("", Path(tmp) / "missing"):
                with self.subTest(directory=directory), self.assertRaises(peer_config.PeerSelectionConfigError):
                    peer_config.select_peers(pair_settings(), cwd=directory)

    def test_select_peers_requires_pair_settings(self) -> None:
        peers, selection = peer_config.select_peers(
            pair_settings(),
            source_path=".tokimon/settings.local.json",
            now=datetime(2026, 5, 5, 1, 2, 3, tzinfo=timezone.utc),
        )

        self.assertEqual(peers["a"]["id"], "local-a-codex")
        self.assertEqual(peers["b"]["id"], "local-b-claude")
        self.assertEqual(selection["selected_peer_ids"], ["local-a-codex", "local-b-claude"])
        self.assertEqual(
            set(selection),
            {"source_path", "settings_section", "selected_peer_ids", "selected_at"},
        )
        self.assertEqual(selection["settings_section"], "pair")

    def test_select_peers_accepts_local_cli_codex_and_claude(self) -> None:
        peers, selection = peer_config.select_peers(
            pair_settings(access="local-cli"),
            source_path=".tokimon/settings.local.json",
            now=datetime(2026, 5, 5, 1, 2, 3, tzinfo=timezone.utc),
        )

        self.assertEqual(peers["a"]["id"], "local-a-codex")
        self.assertEqual(peers["b"]["id"], "local-b-claude")
        self.assertEqual(peers["a"]["capability"], "local-cli")
        self.assertEqual(peers["b"]["capability"], "local-cli")
        self.assertEqual(selection["selected_peer_ids"], ["local-a-codex", "local-b-claude"])

    def test_select_peers_accepts_codex_peer_b_without_duplicate_ids(self) -> None:
        peers, selection = peer_config.select_peers(pair_settings(access="local-cli", peer_b="codex"))

        self.assertEqual(peers["a"]["id"], "local-a-codex")
        self.assertEqual(peers["b"]["id"], "local-b-codex")
        self.assertEqual(selection["selected_peer_ids"], ["local-a-codex", "local-b-codex"])

    def test_select_peers_accepts_native_codex_subagents(self) -> None:
        peers, selection = peer_config.select_peers(
            pair_settings(access="native-subagent", peer_b="codex")
        )

        self.assertEqual(peers["a"]["id"], "native-a-codex")
        self.assertEqual(peers["b"]["id"], "native-b-codex")
        self.assertEqual(peers["a"]["capability"], "native-subagent")
        self.assertEqual(peers["b"]["capability"], "native-subagent")
        self.assertEqual(
            selection["selected_peer_ids"],
            ["native-a-codex", "native-b-codex"],
        )


    def test_native_subagents_reject_claude_peer_b(self) -> None:
        with self.assertRaisesRegex(
            peer_config.PeerSelectionConfigError,
            "pair.peer_b must be codex for native-subagent access",
        ):
            peer_config.select_peers(
                pair_settings(access="native-subagent", peer_b="claude")
            )

    def test_load_and_select_rejects_missing_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(peer_config.PeerSelectionConfigError):
                peer_config.load_and_select(settings_path=Path(tmp) / "missing.json")

    def test_rejects_pair_settings_without_cwd(self) -> None:
        config = pair_settings()
        config.pop("cwd")

        with self.assertRaises(peer_config.PeerSelectionConfigError):
            peer_config.select_peers(config)

    def test_rejects_unsupported_peer_b(self) -> None:
        with self.assertRaises(peer_config.PeerSelectionConfigError) as raised:
            peer_config.select_peers(pair_settings(peer_b="unknown-provider"))

        self.assertIn("pair.peer_b must be claude or codex", str(raised.exception))

    def test_loads_settings_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.local.json"
            path.write_text(
                """{
  "pair": {
    "access": "local-cli",
    "peer_b": "claude",
    "cwd": "/workspace/tokimon"
  }
}
""",
                encoding="utf-8",
            )

            peers, selection = peer_config.load_and_select(settings_path=path)

            self.assertEqual(peers["a"]["agent"], "codex")
            self.assertEqual(selection["source_path"], str(path.resolve()))

    def test_rejects_unsupported_config_keys(self) -> None:
        config = pair_settings()
        config["default_profile"] = "custom-profile"

        with self.assertRaises(peer_config.PeerSelectionConfigError):
            peer_config.select_peers(config)


if __name__ == "__main__":
    unittest.main()
