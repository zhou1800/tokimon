from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tokimon.tools.mcp import remote_browser_broker as broker
from tests.tools.mcp.remote_browser_broker import test_remote_browser_broker as fixtures


class SessionRecoveryTests(unittest.TestCase):
    make_broker = fixtures.RemoteBrowserBrokerTests.make_broker
    capture_runtime_logger = fixtures.RemoteBrowserBrokerTests.capture_runtime_logger

    def test_docker_identity_is_checkpointed_before_browser_readiness(self) -> None:
        remote, _, controller = self.make_broker()
        remote.launcher = broker.DockerSessionLauncher(remote.config)
        observed: list[str] = []

        def docker_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertEqual(command[:2], ["docker", "run"])
            Path(command[command.index("--cidfile") + 1]).write_text("a" * 64)
            return subprocess.CompletedProcess(command, 0, "a" * 64, "")

        def ready(session: broker.BrowserSession, **_: object) -> None:
            payload = json.loads((session.profile_dir.parent / "metadata.json").read_text())
            self.assertEqual(payload["state"], broker.STATE_STARTING)
            self.assertEqual(payload["container_id"], "a" * 64)
            restored = remote._session_from_metadata(session.profile_dir.parent / "metadata.json")
            self.assertEqual(restored.container_id, "a" * 64)
            observed.append(session.container_id)

        with (
            patch.object(remote.launcher, "ensure_image"),
            patch.object(remote.launcher, "_ensure_session_network", return_value="owned-network"),
            patch.object(broker.subprocess, "run", side_effect=docker_run),
            patch.object(controller, "wait_until_ready", side_effect=ready),
        ):
            result = remote.ensure("identity-checkpoint")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(observed, ["a" * 64])
        session = remote._sessions_by_mcp["identity-checkpoint"]
        metadata = json.loads((session.profile_dir.parent / "metadata.json").read_text())
        self.assertEqual(metadata["container_id"], "a" * 64)

    def test_identity_checkpoint_failure_preserves_unverified_container_resources(self) -> None:
        remote, _, controller = self.make_broker()
        remote.launcher = broker.DockerSessionLauncher(remote.config)
        persist = remote._write_session_metadata

        def write(session: broker.BrowserSession) -> None:
            if session.container_id is not None:
                raise OSError("checkpoint unavailable")
            persist(session)

        def docker_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertEqual(command[:2], ["docker", "run"])
            Path(command[command.index("--cidfile") + 1]).write_text("a" * 64)
            return subprocess.CompletedProcess(command, 0, "a" * 64, "")

        with (
            patch.object(remote.launcher, "ensure_image"),
            patch.object(remote.launcher, "_ensure_session_network", return_value="owned-network"),
            patch.object(remote.launcher, "_cleanup_failed_start", return_value=False) as cleanup,
            patch.object(broker.subprocess, "run", side_effect=docker_run),
            patch.object(remote, "_write_session_metadata", side_effect=write),
            patch.object(controller, "wait_until_ready") as ready,
        ):
            with self.assertRaises(broker.BrowserBrokerError) as failed:
                remote.ensure("failed-checkpoint")
        self.assertIs(failed.exception.details["cleanup_verified"], False)
        cleanup.assert_called_once()
        ready.assert_not_called()
        session = remote._sessions_by_mcp["failed-checkpoint"]
        self.assertTrue(session.profile_dir.exists())
        self.assertIsNotNone(session.ownership_lease)
        self.assertEqual(session.state, broker.STATE_UNKNOWN)

    def test_docker_stop_rejects_missing_or_replaced_recorded_id_despite_matching_labels(self) -> None:
        remote, _, _ = self.make_broker()
        launcher = broker.DockerSessionLauncher(remote.config)
        profile = remote.config.runtime_root / "sessions" / "session-a" / "profile"
        session = broker.BrowserSession(
            id="session-a", mcp_session_key="identity", profile_dir=profile,
            cdp_port=0, novnc_port=0, novnc_backend_port=0,
            container_name=f"tokimon-rb-{remote.config.installation_id}-session-a",
            created_at="", last_used_at="",
        )
        payload = [{
            "Id": "b" * 64, "Name": "/" + session.container_name,
            "Config": {"Labels": launcher._stable_ownership_labels(session.id)},
            "Mounts": [{"Source": str(profile), "Destination": "/profile"}],
        }]
        for recorded in (None, "a" * 64):
            with self.subTest(recorded=recorded):
                session.container_id = recorded
                with (
                    patch.object(broker.shutil, "which", return_value="/fake/docker"),
                    patch.object(broker.subprocess, "run", return_value=subprocess.CompletedProcess(
                        [], 0, json.dumps(payload), "",
                    )) as run,
                ):
                    with self.assertRaises(broker.BrowserBrokerError):
                        launcher.stop(session)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][:3], ["docker", "container", "inspect"])

    def test_orphan_cleanup_rechecks_inventory_id_and_preserves_active_legacy_checkpoint(self) -> None:
        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint):
                original, _, _ = self.make_broker()
                if checkpoint:
                    original.ensure("legacy")
                    session = original._sessions_by_mcp["legacy"]
                    session_id, profile = session.id, session.profile_dir
                    original.begin_shutdown()
                    original.release_shutdown_leases()
                else:
                    session_id = "orphan-a"
                    profile = original.config.runtime_root / "sessions" / session_id / "profile"
                    profile.mkdir(parents=True)
                remote = broker.RemoteBrowserBroker(
                    config=original.config,
                    launcher=broker.DockerSessionLauncher(original.config),
                    controller=fixtures.FakeController(),
                    novnc_proxy_factory=fixtures.FakeNoVNCProxy,
                )
                self.addCleanup(lambda remote=remote: (
                    remote.begin_shutdown(), remote.release_shutdown_leases(),
                ))
                name = f"tokimon-rb-{remote.config.installation_id}-{session_id}"
                labels = remote.launcher._stable_ownership_labels(session_id)
                inventory = [{"id": "a" * 64, "name": name, "labels": labels,
                              "mount_sources": [str(profile)]}]
                inspected = [{"Id": "b" * 64, "Name": "/" + name,
                              "Config": {"Labels": labels},
                              "Mounts": [{"Source": str(profile), "Destination": "/profile"}]}]
                with (
                    patch.object(broker, "docker_containers_with_label", return_value=(inventory, None)),
                    patch.object(broker.shutil, "which", return_value="/fake/docker"),
                    patch.object(broker.subprocess, "run", return_value=subprocess.CompletedProcess(
                        [], 0, json.dumps(inspected), "",
                    )) as run,
                ):
                    if checkpoint:
                        with self.assertRaisesRegex(broker.BrowserBrokerError, "recorded immutable identity"):
                            remote.cleanup_orphaned_sessions()
                        run.assert_not_called()
                    else:
                        with self.assertRaisesRegex(broker.BrowserBrokerError, "foreign or ambiguous identity"):
                            remote.cleanup_orphaned_sessions()
                        self.assertEqual(run.call_count, 1)
                self.assertTrue(profile.exists())

    def open_broker(self, config: broker.BrokerConfig) -> broker.RemoteBrowserBroker:
        remote = broker.RemoteBrowserBroker(
            config=config,
            launcher=fixtures.FakeLauncher(),
            controller=fixtures.FakeController(),
            novnc_proxy_factory=fixtures.FakeNoVNCProxy,
        )
        self.addCleanup(lambda: (remote.begin_shutdown(), remote.release_shutdown_leases()))
        return remote

    def call(self, remote: broker.RemoteBrowserBroker, name: str, key: str = "original", **arguments: object) -> dict[str, object]:
        return fixtures.tool_payload(broker.MCPServer(remote).handle(
            {
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            mcp_session_key=key,
        ))

    def stdio(self, remote: broker.RemoteBrowserBroker, calls: list[tuple[str, dict[str, object]]]) -> list[dict[str, object]]:
        requests = [
            {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(calls)
        ]
        output = io.StringIO()
        self.assertEqual(broker.serve(
            input_stream=io.StringIO("".join(json.dumps(item) + "\n" for item in requests)),
            output_stream=output,
            server=broker.MCPServer(remote),
        ), 0)
        return [fixtures.tool_payload(json.loads(line)) for line in output.getvalue().splitlines()]

    def test_fresh_stdio_connection_recovers_profile_after_eof(self) -> None:
        self.capture_runtime_logger()
        for explicit_stop in (False, True):
            with self.subTest(explicit_stop=explicit_stop):
                first, launcher, _ = self.make_broker()
                calls = [("browser_ensure", {})]
                if explicit_stop:
                    calls.append(("browser_stop", {}))
                created = self.stdio(first, calls)[0]
                profile = launcher.started[0]["profile_dir"]
                marker = profile / "saved-login"
                marker.write_text("preserved", encoding="utf-8")
                metadata_path = profile.parent / "metadata.json"
                old_key = json.loads(metadata_path.read_text())["mcp_session_key"]
                self.assertEqual(len(launcher.stopped), 1)

                second = self.open_broker(first.config)
                recovered, ready = self.stdio(second, [
                    ("browser_session_recover", {"recovery_token": created["recovery_token"]}),
                    ("browser_ensure", {}),
                ])

                self.assertEqual(recovered["status"], "recovered")
                self.assertEqual(ready["browser_session"]["id"], created["browser_session"]["id"])
                self.assertNotEqual(ready["recovery_token"], created["recovery_token"])
                self.assertNotEqual(json.loads(metadata_path.read_text())["mcp_session_key"], old_key)
                self.assertEqual(marker.read_text(), "preserved")

    def test_recovery_refuses_invalid_token_active_source_and_occupied_destination(self) -> None:
        remote, launcher, _ = self.make_broker()
        created = self.call(remote, "browser_ensure")
        token = created["recovery_token"]
        for value in ("invalid", "x" * 43, created["browser_session"]["id"]):
            denied = self.call(remote, "browser_session_recover", "new", recovery_token=value)
            self.assertEqual(denied["error_code"], broker.ERROR_SESSION_RECOVERY_UNAVAILABLE)
        active = self.call(remote, "browser_session_recover", "new", recovery_token=token)
        self.assertEqual(active["error_code"], broker.ERROR_SESSION_NOT_RETAINED)
        remote.stop("original")
        self.call(remote, "browser_ensure", "occupied")
        occupied = self.call(remote, "browser_session_recover", "occupied", recovery_token=token)
        self.assertEqual(occupied["error_code"], broker.ERROR_SESSION_RECOVERY_UNAVAILABLE)
        self.assertEqual(len(launcher.started), 2)
        self.assertTrue(remote.whoami("original")["has_browser_session"])

    def test_live_peer_is_not_claimed_until_ownership_is_released(self) -> None:
        first, _, _ = self.make_broker()
        created = first.ensure("original")
        first.stop("original")
        second = self.open_broker(first.config)
        denied = self.call(second, "browser_session_recover", "new", recovery_token=created["recovery_token"])
        self.assertIn("error_code", denied)
        self.assertTrue(first.whoami("original")["has_browser_session"])
        first.begin_shutdown()
        first.release_shutdown_leases()

        recovered = self.call(second, "browser_session_recover", "new", recovery_token=created["recovery_token"])

        self.assertEqual(recovered["status"], "recovered")
        self.assertFalse(second.whoami("original")["has_browser_session"])
        self.assertEqual(second.ensure("new")["browser_session"]["id"], created["browser_session"]["id"])

    def test_recovery_checkpoint_failure_preserves_original_binding_and_allows_retry(self) -> None:
        remote, _, _ = self.make_broker()
        created = remote.ensure("original")
        remote.stop("original")
        token = created["recovery_token"]
        with patch.object(broker, "write_json_atomic", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                remote.recover_session("new", recovery_token=token)
        self.assertTrue(remote.whoami("original")["has_browser_session"])
        self.assertFalse(remote.whoami("new")["has_browser_session"])
        self.assertEqual(remote.recover_session("new", recovery_token=token)["status"], "recovered")
        remote.begin_shutdown()
        remote.release_shutdown_leases()
        restarted = self.open_broker(remote.config)
        self.assertFalse(restarted.whoami("original")["has_browser_session"])
        self.assertEqual(restarted.ensure("new")["browser_session"]["id"], created["browser_session"]["id"])

    def test_failed_restart_keeps_recovery_credential_and_profile(self) -> None:
        first, _, _ = self.make_broker()
        created = first.ensure("original")
        first.stop("original")
        recovered = first.recover_session("new", recovery_token=created["recovery_token"])
        with patch.object(first.controller, "wait_until_ready", side_effect=broker.BrowserBrokerError("not ready")):
            with self.assertRaises(broker.BrowserBrokerError):
                first.ensure("new")
        first.begin_shutdown()
        first.release_shutdown_leases()
        second = self.open_broker(first.config)
        second.recover_session("retry", recovery_token=recovered["recovery_token"])
        self.assertEqual(second.ensure("retry")["browser_session"]["id"], created["browser_session"]["id"])

    def test_recovery_token_rotation_and_redaction(self) -> None:
        logs = self.capture_runtime_logger()
        remote, launcher, _ = self.make_broker()
        old = remote.ensure("original")["recovery_token"]
        token = remote.ensure("original")["recovery_token"]
        remote.stop("original")
        denied = self.call(remote, "browser_session_recover", "new", recovery_token=old)
        self.assertEqual(denied["error_code"], broker.ERROR_SESSION_RECOVERY_UNAVAILABLE)
        metadata = (launcher.started[0]["profile_dir"].parent / "metadata.json").read_text()
        public = json.dumps([
            remote.whoami("original"), remote.current_session_status("original"),
            remote.session_list(), remote.diagnostics("original"), denied,
        ]) + logs.getvalue()
        self.assertNotIn(token, metadata + public)
        self.assertNotIn(old, metadata + public)
        self.assertEqual(json.loads(metadata)["recovery_token_hash"], hashlib.sha256(token.encode()).hexdigest())
        self.assertNotIn("recovery_token_hash", public)

    def test_duplicate_and_active_crash_checkpoints_cannot_be_recovered(self) -> None:
        for state in ("duplicate", "active_crash"):
            with self.subTest(state=state):
                remote, launcher, _ = self.make_broker()
                created = remote.ensure("original")
                if state == "duplicate":
                    remote.stop("original")
                    root = launcher.started[0]["profile_dir"].parent
                    duplicate = root.parent / "duplicate"
                    shutil.copytree(root, duplicate)
                    metadata = json.loads((duplicate / "metadata.json").read_text())
                    metadata.update(id="duplicate", profile_dir=str(duplicate / "profile"))
                    (duplicate / "metadata.json").write_text(json.dumps(metadata))
                remote.begin_shutdown()
                remote.release_shutdown_leases()
                second = self.open_broker(remote.config)
                result = self.call(second, "browser_session_recover", "new", recovery_token=created["recovery_token"])
                self.assertIn("error_code", result)
                self.assertEqual(second.launcher.started, [])

    def test_legacy_checkpoint_requires_original_key_then_receives_recovery_token(self) -> None:
        remote, launcher, _ = self.make_broker()
        created = remote.ensure("original")
        remote.stop("original")
        metadata_path = launcher.started[0]["profile_dir"].parent / "metadata.json"
        data = json.loads(metadata_path.read_text())
        del data["recovery_token_hash"]
        metadata_path.write_text(json.dumps(data))
        remote.begin_shutdown()
        remote.release_shutdown_leases()
        second = self.open_broker(remote.config)
        denied = self.call(second, "browser_session_recover", "new", recovery_token=created["recovery_token"])
        self.assertEqual(denied["error_code"], broker.ERROR_SESSION_RECOVERY_UNAVAILABLE)
        ready = second.ensure("original")
        self.assertEqual(ready["browser_session"]["id"], created["browser_session"]["id"])
        second.stop("original")
        self.assertEqual(second.recover_session("new", recovery_token=ready["recovery_token"])["status"], "recovered")

    def test_expired_editor_can_be_recovered_and_sealed_without_operator(self) -> None:
        first, _, _ = self.make_broker()
        created = self.call(first, "browser_template_create", template_name="draft")
        first.shutdown_transport_session("original")
        draft = first._retained_by_mcp["original"]
        draft.retention_expires_at = "2000-01-01T00:00:00+00:00"
        first._write_session_metadata(draft)
        first.begin_shutdown()
        first.release_shutdown_leases()
        second = self.open_broker(first.config)
        self.call(second, "browser_session_recover", "new", recovery_token=created["recovery_token"])
        sealed = self.call(second, "browser_template_seal", "new", template_name="draft")
        self.assertEqual(sealed["status"], "sealed")
        self.assertEqual(second.ensure("clone", template_name="draft")["status"], "ready")

    def test_template_seal_excludes_chrome_lock_links_and_cache_contents(self) -> None:
        remote, launcher, _ = self.make_broker()
        self.call(remote, "browser_template_create", template_name="chrome-profile")
        profile = launcher.started[0]["profile_dir"]
        (profile / "saved-login").write_text("keep")
        for name in ("SingletonLock", "SingletonCookie", "SingletonSocket", "transient.tmp"):
            (profile / name).symlink_to("absent-transient-target")
        cache = profile / "GrShaderCache"
        cache.mkdir()
        (cache / "transient-link").symlink_to("absent-cache-target")
        sealed = self.call(remote, "browser_template_seal", template_name="chrome-profile")
        self.assertEqual(sealed["status"], "sealed")
        self.call(remote, "browser_ensure", "clone", template_name="chrome-profile")
        cloned = launcher.started[-1]["profile_dir"]
        self.assertEqual((cloned / "saved-login").read_text(), "keep")
        self.assertEqual([path.name for path in cloned.iterdir()], ["saved-login"])

    def test_stop_requests_chrome_close_before_waiting_for_container_exit(self) -> None:
        remote, _, _ = self.make_broker()
        remote.ensure("original")
        remote.launcher = broker.DockerSessionLauncher(remote.config)
        cdp = Mock()

        def wait_for_container(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            cdp.call.assert_called_once_with("Browser.close")
            self.assertEqual(args, [remote.config.docker_command, "wait", "a" * 64])
            return subprocess.CompletedProcess(args, 0, "", "")

        with (
            patch.object(broker.shutil, "which", return_value="/usr/bin/docker"),
            patch.object(remote.launcher, "_owned_container_id", return_value="a" * 64),
            patch.object(remote.launcher, "_container_is_absent", return_value=True),
            patch.object(remote.launcher, "_remove_session_network"),
            patch.object(broker.CDPConnection, "for_session", return_value=contextlib.nullcontext(cdp)),
            patch.object(broker.subprocess, "run", side_effect=wait_for_container),
        ):
            result = remote.stop("original")
        self.assertEqual(result["status"], "stopped_retained")

    def test_recovery_skips_busy_original_lifecycle_without_deadlock(self) -> None:
        remote, _, _ = self.make_broker()
        created = remote.ensure("original")
        remote.stop("original")
        with remote._lifecycle_lock_for_key("original"), concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(
                self.call, remote, "browser_session_recover", "new",
                recovery_token=created["recovery_token"],
            ).result(timeout=2)
        self.assertEqual(result["error_code"], broker.ERROR_SESSION_RECOVERY_UNAVAILABLE)
        self.assertTrue(remote.whoami("original")["has_browser_session"])

    def test_concurrent_recovery_can_bind_only_one_new_session(self) -> None:
        remote, _, _ = self.make_broker()
        created = remote.ensure("original")
        remote.stop("original")
        barrier = threading.Barrier(2)

        def recover(key: str) -> dict[str, object]:
            barrier.wait(timeout=2)
            return self.call(remote, "browser_session_recover", key, recovery_token=created["recovery_token"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(recover, ("one", "two")))
        self.assertEqual(sum(result.get("status") == "recovered" for result in results), 1)
        self.assertEqual(remote.session_list()["session_count"], 1)
        self.assertEqual(sum(remote.whoami(key)["has_browser_session"] for key in ("one", "two")), 1)

    def test_artifact_cleanup_releases_quota_and_preserves_profile_downloads_and_peer(self) -> None:
        remote, launcher, _ = self.make_broker()
        created = self.call(remote, "browser_ensure")
        profile = launcher.started[0]["profile_dir"]
        login = profile / "login-state"
        login.write_text("keep-login")
        downloads = profile / "downloads"
        downloads.mkdir(exist_ok=True)
        download = downloads / "keep.txt"
        download.write_text("keep-download")
        source = remote.config.runtime_root / "task-file.txt"
        source.write_text("upload")
        ref = self.call(remote, "browser_snapshot")["page"]["elements"][2]["ref"]
        with patch.object(broker, "MAX_STAGED_UPLOADS", 1), patch.object(broker, "MAX_SCREENSHOT_ARTIFACTS", 1):
            self.assertEqual(self.call(remote, "browser_upload_file", ref=ref, path=str(source))["status"], "uploaded")
            screenshot = self.call(remote, "browser_screenshot")
            self.assertIn("error_code", self.call(remote, "browser_upload_file", ref=ref, path=str(source)))
            remote.ensure("peer")
            peer = remote._sessions_by_mcp["peer"]
            peer_marker = peer.profile_dir / "keep.txt"
            peer_marker.write_text("peer-data")
            remote.stop("original")
            cleared = self.call(remote, "browser_session_cleanup_artifacts")
            self.assertEqual(cleared["status"], "cleaned")
            self.assertEqual(cleared["removed"]["uploads"], {"files": 1, "bytes": 6})
            self.assertEqual(cleared["removed"]["screenshots"]["files"], 1)
            self.assertEqual(login.read_text(), "keep-login")
            self.assertEqual(download.read_text(), "keep-download")
            self.assertEqual(peer_marker.read_text(), "peer-data")
            self.assertFalse(Path(screenshot["screenshot"]["path"]).exists())
            ready = remote.ensure("original")
            self.assertEqual(ready["browser_session"]["id"], created["browser_session"]["id"])
            ref = self.call(remote, "browser_snapshot")["page"]["elements"][2]["ref"]
            self.assertEqual(self.call(remote, "browser_upload_file", ref=ref, path=str(source))["status"], "uploaded")
            self.assertNotIn("error_code", self.call(remote, "browser_screenshot"))

    def test_artifact_cleanup_requires_stopped_ordinary_session(self) -> None:
        remote, _, _ = self.make_broker()
        remote.ensure("original")
        self.assertEqual(self.call(remote, "browser_session_cleanup_artifacts")["error_code"], broker.ERROR_SESSION_NOT_RETAINED)
        remote.create_template("editor", template_name="draft")
        remote.shutdown_transport_session("editor")
        self.assertEqual(self.call(remote, "browser_session_cleanup_artifacts", "editor")["error_code"], broker.ERROR_SESSION_NOT_RETAINED)

    def test_artifact_cleanup_refuses_links_and_special_files_before_deleting_either_tree(self) -> None:
        for kind in ("root_link", "file_link", "fifo"):
            with self.subTest(kind=kind):
                remote, launcher, _ = self.make_broker()
                remote.ensure("original")
                profile = launcher.started[0]["profile_dir"]
                uploads = profile / "uploads"
                uploads.mkdir()
                staged = uploads / "staged.txt"
                staged.write_text("keep")
                screenshots = profile.parent / "screenshots"
                outside = remote.config.runtime_root / "outside"
                outside.mkdir()
                sentinel = outside / "sentinel"
                sentinel.write_text("outside")
                if kind == "root_link":
                    screenshots.symlink_to(outside, target_is_directory=True)
                else:
                    screenshots.mkdir()
                    if kind == "file_link":
                        (screenshots / "linked.png").symlink_to(sentinel)
                    else:
                        os.mkfifo(screenshots / "pipe")
                remote.stop("original")
                result = self.call(remote, "browser_session_cleanup_artifacts")
                self.assertIn("error_code", result)
                self.assertEqual(sentinel.read_text(), "outside")
                self.assertEqual(staged.read_text(), "keep")

    def test_artifact_cleanup_refuses_unverified_container(self) -> None:
        remote, launcher, _ = self.make_broker()
        remote.ensure("original")
        remote.stop("original")
        profile = launcher.started[0]["profile_dir"]
        (profile / "uploads").mkdir()
        marker = profile / "uploads" / "staged"
        marker.write_text("keep")
        remote.launcher = broker.DockerSessionLauncher(remote.config)
        with patch.object(remote.launcher, "stop", side_effect=broker.BrowserBrokerError("foreign container")):
            result = self.call(remote, "browser_session_cleanup_artifacts")
        self.assertIn("error_code", result)
        self.assertEqual(marker.read_text(), "keep")

    def test_restart_waits_until_artifact_cleanup_finishes(self) -> None:
        remote, launcher, _ = self.make_broker()
        created = remote.ensure("original")
        remote.stop("original")
        lifecycle = fixtures.RemoteBrowserBrokerTests.track_lifecycle_lock(self, remote, "original")
        entered = threading.Event()
        release = threading.Event()
        original_remove = shutil.rmtree

        def remove(path: str, **kwargs: object) -> None:
            entered.set()
            if not release.wait(timeout=2):
                raise AssertionError("test did not release artifact cleanup")
            original_remove(path, **kwargs)

        with patch.object(broker.shutil, "rmtree", side_effect=remove), concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="artifacts") as pool:
            clearing = pool.submit(remote.cleanup_session_artifacts, "original")
            try:
                self.assertTrue(entered.wait(timeout=2))
                restoring = pool.submit(remote.ensure, "original")
                self.assertTrue(lifecycle.attempted_by("artifacts_1").wait(timeout=2))
                self.assertEqual(len(launcher.started), 1)
            finally:
                release.set()
            self.assertEqual(clearing.result(timeout=2)["status"], "cleaned")
            self.assertEqual(restoring.result(timeout=2)["browser_session"]["id"], created["browser_session"]["id"])

    def test_partial_artifact_cleanup_failure_can_be_retried_without_losing_login(self) -> None:
        remote, launcher, _ = self.make_broker()
        remote.ensure("original")
        profile = launcher.started[0]["profile_dir"]
        (profile / "login").write_text("keep")
        remote.stop("original")
        original_remove = shutil.rmtree

        def fail_screenshots(path: str, **kwargs: object) -> None:
            if path == "screenshots":
                raise OSError("busy file")
            original_remove(path, **kwargs)

        with patch.object(broker.shutil, "rmtree", side_effect=fail_screenshots):
            with self.assertRaises(OSError):
                remote.cleanup_session_artifacts("original")
        self.assertEqual(remote.cleanup_session_artifacts("original")["status"], "cleaned")
        self.assertEqual((profile / "login").read_text(), "keep")
        self.assertEqual(remote.ensure("original")["status"], "ready")

    def test_upload_rejects_relative_path_instead_of_selecting_tokimon_file(self) -> None:
        remote, _, controller = self.make_broker()
        remote.ensure("original")
        ref = self.call(remote, "browser_snapshot")["page"]["elements"][2]["ref"]
        result = self.call(remote, "browser_upload_file", ref=ref, path="AGENTS.md")
        self.assertEqual(result["error_code"], broker.ERROR_INVALID_ARGUMENT)
        self.assertEqual(controller.uploads, [])


if __name__ == "__main__":
    unittest.main()
