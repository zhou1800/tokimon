from __future__ import annotations

import concurrent.futures
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tokimon.tools.mcp import remote_browser_broker as broker
from tests.tools.mcp.remote_browser_broker.test_remote_browser_broker import (
    FakeController,
    FakeLauncher,
    FakeNoVNCProxy,
    tool_payload,
)


class CapacityCleanupTests(unittest.TestCase):
    def make_broker(self, **overrides: object) -> broker.RemoteBrowserBroker:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = broker.BrokerConfig(
            runtime_root=Path(temporary.name) / "remote-browser",
            max_active_sessions=1,
            max_total_sessions=2,
        )
        return self.open_broker(broker.replace_config(config, **overrides))

    def open_broker(self, config: broker.BrokerConfig, *, docker: bool = False) -> broker.RemoteBrowserBroker:
        remote = broker.RemoteBrowserBroker(
            config=config,
            launcher=broker.DockerSessionLauncher(config) if docker else FakeLauncher(),
            controller=FakeController(),
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(lambda: self.release(remote))
        return remote

    def release(self, remote: broker.RemoteBrowserBroker) -> None:
        remote.begin_shutdown()
        remote.release_shutdown_leases()

    def ensure(self, remote: broker.RemoteBrowserBroker, key: str) -> dict[str, object]:
        response = broker.MCPServer(remote).handle(
            {
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key=key,
        )
        return tool_payload(response)

    def expire(self, remote: broker.RemoteBrowserBroker, key: str) -> broker.BrowserSession:
        remote.stop(key)
        session = remote._retained_by_mcp[key]
        session.retention_expires_at = "2000-01-01T00:00:00+00:00"
        remote._write_session_metadata(session)
        return session

    def test_total_pressure_preserves_expired_profiles_until_explicit_deletion(self) -> None:
        remote = self.make_broker()
        remote.ensure("expired")
        expired = self.expire(remote, "expired")
        remote.ensure("retained")
        remote.stop("retained")

        result = self.ensure(remote, "new")

        self.assertEqual(result["error_code"], broker.ERROR_SESSION_CAPACITY_EXHAUSTED)
        self.assertTrue(expired.profile_dir.parent.exists())
        self.assertTrue(remote.whoami("retained")["has_browser_session"])
        diagnostics = remote.diagnostics("new")["automatic_cleanup"]
        self.assertEqual(diagnostics["counts"], {})
        self.assertEqual(diagnostics["profile_deletion"], "explicit_only")
        remote.delete_retained_session("expired")
        self.assertEqual(self.ensure(remote, "new")["status"], "ready")

    def test_idle_lru_reclamation_preserves_profile_and_same_session_restoration(self) -> None:
        remote = self.make_broker(max_active_sessions=2, max_total_sessions=4)
        remote.ensure("oldest")
        oldest = remote._sessions_by_mcp["oldest"]
        marker = oldest.profile_dir / "login-state"
        marker.write_text("preserved", encoding="utf-8")
        remote.ensure("other")
        remote._sessions_by_mcp["other"].last_used_at = "2001-01-01T00:00:00+00:00"
        oldest.last_used_at = "2000-01-01T00:00:00+00:00"

        self.assertEqual(self.ensure(remote, "new")["status"], "ready")
        self.assertEqual(remote.whoami("oldest")["browser_session"]["state"], broker.STATE_STOPPED_RETAINED)
        self.assertEqual(remote.whoami("other")["browser_session"]["state"], broker.STATE_RUNNING)

        restored = self.ensure(remote, "oldest")
        self.assertEqual(restored["browser_session"]["id"], oldest.id)
        self.assertIsNot(remote._sessions_by_mcp["oldest"], oldest)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserved")
        self.assertEqual(len(remote.launcher.stopped), 2)

    def test_protected_sessions_are_not_reclaimed(self) -> None:
        for protection in ("recent", "paused", "starting", "editor", "viewer", "disabled"):
            with self.subTest(protection=protection):
                remote = self.make_broker()
                if protection == "editor":
                    remote.create_template("protected", template_name="draft")
                else:
                    remote.ensure("protected")
                session = remote._sessions_by_mcp["protected"]
                session.last_used_at = "2000-01-01T00:00:00+00:00"
                if protection == "recent":
                    remote.current_session_status("protected")
                elif protection == "paused":
                    remote.pause_for_human("protected")
                    session.last_used_at = "2000-01-01T00:00:00+00:00"
                elif protection == "starting":
                    session.state = broker.STATE_STARTING
                elif protection == "viewer":
                    remote.connect_info("protected")
                    session.last_used_at = "2000-01-01T00:00:00+00:00"
                elif protection == "disabled":
                    remote.config = broker.replace_config(remote.config, idle_timeout_seconds=0)

                result = self.ensure(remote, "new")

                self.assertEqual(result["error_code"], broker.ERROR_SESSION_CAPACITY_EXHAUSTED)
                self.assertEqual(remote.launcher.stopped, [])
                self.assertTrue(session.profile_dir.exists())

    def test_default_idle_policy_allows_thirty_minutes_under_capacity_pressure(self) -> None:
        remote = self.make_broker()
        remote.ensure("long-task")
        session = remote._sessions_by_mcp["long-task"]
        now = datetime.now(timezone.utc)
        with patch.object(broker, "datetime") as clock:
            clock.now.return_value = now
            clock.fromisoformat.side_effect = datetime.fromisoformat
            for idle_minutes in (10, 19, 29):
                session.last_used_at = (now - timedelta(minutes=idle_minutes)).isoformat()
                result = self.ensure(remote, "new")
                self.assertEqual(result["error_code"], broker.ERROR_SESSION_CAPACITY_EXHAUSTED)
                self.assertEqual(remote.launcher.stopped, [])
            session.last_used_at = (now - timedelta(minutes=30)).isoformat()
            self.assertEqual(self.ensure(remote, "new")["status"], "ready")
            self.assertEqual(remote.whoami("long-task")["browser_session"]["state"],
                             broker.STATE_STOPPED_RETAINED)

    def test_busy_lifecycle_and_operation_locks_are_skipped_without_waiting(self) -> None:
        for lock_kind in ("lifecycle", "operation"):
            with self.subTest(lock_kind=lock_kind):
                remote = self.make_broker()
                remote.ensure("busy")
                session = remote._sessions_by_mcp["busy"]
                session.last_used_at = "2000-01-01T00:00:00+00:00"
                lock = remote._lifecycle_lock_for_key("busy") if lock_kind == "lifecycle" else session.operation_lock
                with lock, concurrent.futures.ThreadPoolExecutor() as executor:
                    result = executor.submit(self.ensure, remote, "new").result(timeout=2)
                self.assertEqual(result["error_code"], broker.ERROR_SESSION_CAPACITY_EXHAUSTED)
                self.assertEqual(remote.launcher.stopped, [])

    def test_unexpired_retained_profiles_and_expired_template_drafts_are_preserved(self) -> None:
        remote = self.make_broker(max_total_sessions=1)
        remote.ensure("saved")
        remote.stop("saved")
        result = self.ensure(remote, "new")
        self.assertEqual(result["error_code"], broker.ERROR_SESSION_CAPACITY_EXHAUSTED)
        self.assertTrue(remote._retained_by_mcp["saved"].profile_dir.exists())

        editor = self.make_broker(max_total_sessions=1)
        editor.create_template("draft-key", template_name="draft")
        editor._quiesce_template_editor_for_shutdown("draft-key")
        draft = editor._retained_by_mcp["draft-key"]
        draft.retention_expires_at = "2000-01-01T00:00:00+00:00"
        self.assertEqual(self.ensure(editor, "new")["error_code"], broker.ERROR_SESSION_CAPACITY_EXHAUSTED)
        self.assertTrue(draft.profile_dir.exists())

    def test_idle_runtime_is_not_stopped_when_total_capacity_cannot_be_freed(self) -> None:
        remote = self.make_broker(max_total_sessions=1)
        remote.ensure("idle")
        remote._sessions_by_mcp["idle"].last_used_at = "2000-01-01T00:00:00+00:00"
        self.assertEqual(self.ensure(remote, "new")["error_code"], broker.ERROR_SESSION_CAPACITY_EXHAUSTED)
        self.assertEqual(remote.launcher.stopped, [])

    def test_expired_current_session_restores_the_same_profile(self) -> None:
        remote = self.make_broker(max_total_sessions=1)
        remote.ensure("same")
        expired = self.expire(remote, "same")
        result = self.ensure(remote, "same")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["browser_session"]["id"], expired.id)
        self.assertTrue(expired.profile_dir.parent.exists())

    def test_failed_explicit_deletion_keeps_total_capacity_reserved(self) -> None:
        remote = self.make_broker(max_total_sessions=1)
        remote.ensure("saved")
        retained = self.expire(remote, "saved")
        with patch.object(broker.shutil, "rmtree", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                remote.delete_retained_session("saved")
        self.assertTrue(retained.profile_dir.exists())
        self.assertIsNotNone(retained.ownership_lease)
        self.assertEqual(self.ensure(remote, "new")["error_code"], broker.ERROR_SESSION_CAPACITY_EXHAUSTED)
        remote.delete_retained_session("saved")
        self.assertEqual(self.ensure(remote, "new")["status"], "ready")

    def test_failed_idle_checkpoint_is_retried_without_releasing_capacity_early(self) -> None:
        remote = self.make_broker()
        remote.ensure("idle")
        idle = remote._sessions_by_mcp["idle"]
        idle.last_used_at = "2000-01-01T00:00:00+00:00"
        with patch.object(remote, "_write_session_metadata", side_effect=OSError("disk full")):
            result = self.ensure(remote, "new")
        self.assertEqual(result["error_code"], broker.ERROR_SESSION_CAPACITY_EXHAUSTED)
        self.assertEqual(result["automatic_cleanup"]["counts"], {"failed": 1})
        self.assertIs(remote._sessions_by_mcp["idle"], idle)
        self.assertIsNotNone(idle.ownership_lease)
        self.assertEqual(len(remote.launcher.started), 1)
        self.assertEqual(self.ensure(remote, "new")["status"], "ready")

    def test_concurrent_admissions_do_not_reuse_capacity_before_stop_finishes(self) -> None:
        remote = self.make_broker(max_total_sessions=3)
        remote.ensure("idle")
        idle = remote._sessions_by_mcp["idle"]
        idle.last_used_at = "2000-01-01T00:00:00+00:00"
        stop_started = threading.Event()
        finish_stop = threading.Event()
        original_stop = remote.launcher.stop

        def stop(session: broker.BrowserSession) -> None:
            stop_started.set()
            if not finish_stop.wait(2):
                raise AssertionError("test did not release stop")
            original_stop(session)

        with patch.object(remote.launcher, "stop", side_effect=stop), concurrent.futures.ThreadPoolExecutor() as executor:
            first = executor.submit(self.ensure, remote, "one")
            try:
                self.assertTrue(stop_started.wait(2))
                second = executor.submit(self.ensure, remote, "two")
                self.assertEqual(len(remote.launcher.started), 1)
                self.assertTrue(idle.port_ownership_leases)
            finally:
                finish_stop.set()
            results = [first.result(timeout=2), second.result(timeout=2)]
        self.assertEqual(sum(result.get("status") == "ready" for result in results), 1)
        self.assertEqual(len(remote.launcher.started), 2)
        self.assertEqual(len(remote._sessions_by_mcp), 1)

    def test_abandoned_unmounted_profile_is_preserved_under_capacity_pressure(self) -> None:
        owner = self.make_broker(max_total_sessions=1)
        owner.ensure("abandoned")
        abandoned = owner._sessions_by_mcp["abandoned"]
        self.release(owner)
        remote = self.open_broker(owner.config, docker=True)
        with patch.object(remote.launcher, "stop") as stop:
            result = self.ensure(remote, "new")
        self.assertEqual(result["error_code"], broker.ERROR_SESSION_CAPACITY_EXHAUSTED)
        self.assertTrue(abandoned.profile_dir.exists())
        stop.assert_not_called()

    def test_peer_inventory_does_not_consume_capacity_or_allow_cross_session_cleanup(self) -> None:
        owner = self.make_broker(max_total_sessions=1)
        owner.ensure("peer")
        retained = self.expire(owner, "peer")
        remote = self.open_broker(owner.config, docker=True)
        fake = FakeLauncher()
        with patch.object(remote.launcher, "stop") as stop, patch.object(remote.launcher, "start", side_effect=fake.start):
            self.assertEqual(self.ensure(remote, "new")["status"], "ready")
            self.assertEqual(remote.session_list()["session_count"], 2)
            self.assertIn("error", self.ensure(remote, "peer"))
            self.assertEqual(self.ensure(remote, "another")["error_code"], broker.ERROR_SESSION_CAPACITY_EXHAUSTED)
            stop.assert_not_called()
        self.assertTrue(retained.profile_dir.exists())
        self.assertIsNotNone(retained.ownership_lease)


if __name__ == "__main__":
    unittest.main()
