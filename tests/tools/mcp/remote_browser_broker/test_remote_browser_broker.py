from __future__ import annotations

import base64
import contextlib
import http.client
import io
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tokimon.tools.mcp import remote_browser_broker as broker


def owned_network_inspect_payload(
    launcher: broker.DockerSessionLauncher,
    session_id: str,
    *,
    containers: dict[str, object] | None = None,
    labels: dict[str, str] | None = None,
) -> str:
    config = launcher.config
    return json.dumps(
        [
            {
                "Id": "c" * 64,
                "Name": launcher._network_name(session_id),
                "Driver": "bridge",
                "Internal": False,
                "Labels": labels
                or {
                    "tokimon.remote-browser-broker": "true",
                    "tokimon.remote-browser-installation": config.installation_id,
                    "tokimon.remote-browser-owner-user": config.owner_user,
                    "tokimon.remote-browser-checkout": str(config.workspace_root),
                    "tokimon.remote-browser-runtime-root": str(config.runtime_root),
                    "tokimon.remote-browser-session": session_id,
                    "tokimon.remote-browser-network-policy": config.network_policy,
                },
                "Containers": containers or {},
            }
        ]
    )


class FakeLauncher:
    def __init__(self) -> None:
        self.started: list[dict[str, object]] = []
        self.stopped: list[str] = []

    def start(
        self,
        *,
        session_id: str,
        profile_dir: Path,
        cdp_port: int,
        novnc_backend_port: int,
        container_started: Callable[[str], None] | None = None,
    ) -> str:
        container_name = f"fake-{session_id}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        self.started.append(
            {
                "session_id": session_id,
                "profile_dir": profile_dir,
                "cdp_port": cdp_port,
                "novnc_backend_port": novnc_backend_port,
                "container_name": container_name,
            }
        )
        if container_started is not None:
            container_started("a" * 64)
        return container_name

    def stop(self, session: broker.BrowserSession) -> None:
        self.stopped.append(session.container_name)


class FakeController:
    def __init__(self) -> None:
        self.ready: list[str] = []
        self.navigations: list[tuple[str, str]] = []
        self.snapshots: list[tuple[str, int, int]] = []
        self.evaluations: list[tuple[str, str, bool]] = []
        self.screenshots: list[tuple[str, str, int | None]] = []
        self.clicks: list[tuple[str, str]] = []
        self.types: list[tuple[str, str, str, bool]] = []
        self.uploads: list[tuple[str, str, str]] = []
        self.backend_requests: list[dict[str, object]] = []
        self.keys: list[tuple[str, str]] = []
        self.waits: list[tuple[str, float | None, str | None, int]] = []
        self.tabs: list[tuple[str, str, str | None]] = []

    def wait_until_ready(self, session: broker.BrowserSession, *, timeout_sec: float) -> None:
        self.ready.append(session.id)

    def navigate(self, session: broker.BrowserSession, url: str) -> dict[str, object]:
        self.navigations.append((session.id, url))
        return {"url": broker.normalize_url(url), "title": "Fake Page"}

    def snapshot(
        self,
        session: broker.BrowserSession,
        *,
        max_text_chars: int = 20000,
        max_elements: int = 200,
    ) -> dict[str, object]:
        self.snapshots.append((session.id, max_text_chars, max_elements))
        return {
            "url": "https://example.com/",
            "title": "Example",
            "text": "Example Domain",
            "elements": [
                {"tag": "a", "text": "More information"},
                {"tag": "input", "text": "", "placeholder": "Search"},
                {"tag": "input", "text": "", "type": "file"},
            ],
        }

    def evaluate(
        self,
        session: broker.BrowserSession,
        *,
        expression: str,
        await_promise: bool = True,
    ) -> object:
        self.evaluations.append((session.id, expression, await_promise))
        return {"ok": True, "expression": expression}

    def screenshot(
        self,
        session: broker.BrowserSession,
        *,
        image_format: str,
        quality: int | None = None,
    ) -> str:
        self.screenshots.append((session.id, image_format, quality))
        return base64.b64encode(b"fake screenshot").decode("ascii")

    def click(
        self,
        session: broker.BrowserSession,
        ref_data: broker.Mapping[str, object],
    ) -> dict[str, object]:
        ref = str(ref_data["ref"])
        self.clicks.append((session.id, ref))
        return {"status": "clicked", "ref": ref}

    def type_text(
        self,
        session: broker.BrowserSession,
        ref_data: broker.Mapping[str, object],
        *,
        text: str,
        clear: bool,
    ) -> dict[str, object]:
        ref = str(ref_data["ref"])
        self.types.append((session.id, ref, text, clear))
        return {"status": "typed", "ref": ref, "length": len(text)}

    def upload_file(
        self,
        session: broker.BrowserSession,
        ref_data: broker.Mapping[str, object],
        *,
        file_path: str,
    ) -> dict[str, object]:
        ref = str(ref_data["ref"])
        self.uploads.append((session.id, ref, file_path))
        return {"status": "uploaded", "ref": ref, "file_path": file_path}

    def backend_request(
        self,
        session: broker.BrowserSession,
        *,
        method: str,
        path: str,
        body: broker.Mapping[str, object] | None,
        max_response_chars: int,
        timeout_ms: int = broker.DEFAULT_BACKEND_REQUEST_TIMEOUT_MS,
    ) -> dict[str, object]:
        self.backend_requests.append(
            {
                "session_id": session.id,
                "method": method,
                "path": path,
                "body": body,
                "max_response_chars": max_response_chars,
                "timeout_ms": timeout_ms,
            }
        )
        return {"status": 200, "text": "{}", "truncated": False, "url_path": path}

    def press_key(self, session: broker.BrowserSession, key: str) -> dict[str, object]:
        self.keys.append((session.id, key))
        return {"status": "pressed", "key": key}

    def wait(
        self,
        session: broker.BrowserSession,
        *,
        seconds: float | None = None,
        text: str | None = None,
        timeout_ms: int = 5000,
    ) -> dict[str, object]:
        self.waits.append((session.id, seconds, text, timeout_ms))
        return {"status": "waited"}

    def new_tab(self, session: broker.BrowserSession, url: str | None = None) -> dict[str, object]:
        tab_id = f"tab-{len(self.tabs) + 1}"
        self.tabs.append((session.id, "new", url))
        session.active_tab_id = tab_id
        return {"tab_id": tab_id, "active_tab_id": tab_id, "url": url or "about:blank"}

    def switch_tab(self, session: broker.BrowserSession, tab_id: str) -> dict[str, object]:
        self.tabs.append((session.id, "switch", tab_id))
        session.active_tab_id = tab_id
        return {"tab_id": tab_id, "active_tab_id": tab_id}

    def close_tab(self, session: broker.BrowserSession, tab_id: str | None = None) -> dict[str, object]:
        closed = tab_id or session.active_tab_id or "tab-1"
        self.tabs.append((session.id, "close", closed))
        session.active_tab_id = "tab-next"
        return {"closed_tab_id": closed, "active_tab_id": session.active_tab_id}

    def fingerprint_baseline(self, session: broker.BrowserSession) -> dict[str, object]:
        return {
            "signals": {
                "webdriver": False,
                "user_agent": "Fake Chrome",
                "languages": ["en-US"],
                "timezone": "America/Edmonton",
                "viewport": {"width": 1400, "height": 900, "device_pixel_ratio": 1},
                "webgl_present": True,
            },
            "checks": {
                "webdriver_not_true": True,
                "user_agent_present": True,
                "timezone_present": True,
                "viewport_present": True,
                "webgl_present": True,
            },
            "claim": "baseline self-check only; does not claim third-party undetectability",
        }


class FailingOnceReadyController(FakeController):
    def __init__(self) -> None:
        super().__init__()
        self.failures_remaining = 1

    def wait_until_ready(self, session: broker.BrowserSession, *, timeout_sec: float) -> None:
        super().wait_until_ready(session, timeout_sec=timeout_sec)
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("browser did not become ready")


class BlockingController(FakeController):
    def __init__(self, blocked_operation: str) -> None:
        super().__init__()
        self.blocked_operation = blocked_operation
        self.operation_entered = threading.Event()
        self.operation_release = threading.Event()

    def _block(self, operation: str) -> None:
        if self.blocked_operation != operation:
            return
        self.operation_entered.set()
        if not self.operation_release.wait(timeout=2):
            raise RuntimeError(f"timed out waiting to release {operation}")

    def wait_until_ready(self, session: broker.BrowserSession, *, timeout_sec: float) -> None:
        self._block("ready")
        super().wait_until_ready(session, timeout_sec=timeout_sec)

    def screenshot(
        self,
        session: broker.BrowserSession,
        *,
        image_format: str,
        quality: int | None = None,
    ) -> str:
        self._block("screenshot")
        return super().screenshot(
            session,
            image_format=image_format,
            quality=quality,
        )

    def wait(
        self,
        session: broker.BrowserSession,
        *,
        seconds: float | None = None,
        text: str | None = None,
        timeout_ms: int = 5000,
    ) -> dict[str, object]:
        self._block("wait")
        return super().wait(
            session,
            seconds=seconds,
            text=text,
            timeout_ms=timeout_ms,
        )

    def fingerprint_baseline(self, session: broker.BrowserSession) -> dict[str, object]:
        self._block("diagnostics")
        return super().fingerprint_baseline(session)


class TrackedLifecycleLock:
    """RLock wrapper that exposes deterministic per-thread lock boundaries."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._attempted: dict[str, threading.Event] = {}
        self._acquired: dict[str, threading.Event] = {}

    def attempted_by(self, thread_name: str) -> threading.Event:
        return self._attempted.setdefault(thread_name, threading.Event())

    def acquired_by(self, thread_name: str) -> threading.Event:
        return self._acquired.setdefault(thread_name, threading.Event())

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        thread_name = threading.current_thread().name
        self.attempted_by(thread_name).set()
        if timeout == -1:
            acquired = self._lock.acquire(blocking)
        else:
            acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            self.acquired_by(thread_name).set()
        return acquired

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "TrackedLifecycleLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class FakeNoVNCProxy:
    started: list[dict[str, object]] = []
    stopped: list[int] = []

    def __init__(
        self,
        *,
        listen_host: str,
        listen_port: int,
        backend_host: str,
        backend_port: int,
        token: str,
        expires_at: str | None = None,
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.token = token
        self.expires_at = expires_at

    def start(self) -> None:
        self.started.append(
            {
                "listen_host": self.listen_host,
                "listen_port": self.listen_port,
                "backend_host": self.backend_host,
                "backend_port": self.backend_port,
                "token": self.token,
                "expires_at": self.expires_at,
            }
        )

    def stop(self) -> None:
        self.stopped.append(self.listen_port)


def tool_payload(response: dict[str, object]) -> dict[str, object]:
    result = response["result"]
    assert isinstance(result, dict)
    content = result["content"]
    assert isinstance(content, list)
    text = content[0]["text"]
    assert isinstance(text, str)
    return json.loads(text)


def parse_sse_payload(body: bytes) -> dict[str, object]:
    text = body.decode("utf-8")
    data_lines = [line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")]
    if not data_lines:
        raise AssertionError(f"SSE response did not contain a data line: {text!r}")
    payload = json.loads("\n".join(data_lines))
    assert isinstance(payload, dict)
    return payload


class RemoteBrowserBrokerTests(unittest.TestCase):
    def capture_runtime_logger(self) -> io.StringIO:
        logger = logging.getLogger(broker.LOGGER_NAME)
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
        stream = io.StringIO()
        broker.configure_runtime_logger(stream=stream)
        return stream

    def make_broker(
        self,
        *,
        controller: FakeController | None = None,
    ) -> tuple[broker.RemoteBrowserBroker, FakeLauncher, FakeController]:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        config = broker.BrokerConfig(
            runtime_root=Path(tmpdir.name) / "remote-browser",
            cdp_port_range=(31000, 31999),
            novnc_port_range=(41000, 41999),
            startup_timeout_sec=0.1,
        )
        launcher = FakeLauncher()
        controller = controller or FakeController()
        FakeNoVNCProxy.started = []
        FakeNoVNCProxy.stopped = []
        remote_broker = broker.RemoteBrowserBroker(
            config=config,
            launcher=launcher,  # type: ignore[arg-type]
            controller=controller,  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )

        def release_test_leases() -> None:
            remote_broker.begin_shutdown()
            remote_broker.release_shutdown_leases()

        self.addCleanup(release_test_leases)
        return remote_broker, launcher, controller

    def track_lifecycle_lock(
        self,
        remote_broker: broker.RemoteBrowserBroker,
        mcp_session_key: str,
    ) -> TrackedLifecycleLock:
        tracked = TrackedLifecycleLock()
        with remote_broker._lock:
            remote_broker._lifecycle_locks[mcp_session_key] = tracked
        return tracked

    def track_operation_lock(
        self,
        remote_broker: broker.RemoteBrowserBroker,
        mcp_session_key: str,
    ) -> TrackedLifecycleLock:
        tracked = TrackedLifecycleLock()
        remote_broker._session_or_error(mcp_session_key).operation_lock = tracked
        return tracked

    def call_mcp_tool(
        self,
        server: broker.MCPServer,
        *,
        mcp_session_key: str,
        name: str,
        arguments: dict[str, object] | None = None,
    ) -> dict[str, object]:
        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": f"call-{name}",
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            },
            mcp_session_key=mcp_session_key,
        )
        self.assertIsNotNone(response)
        self.assertIn("result", response)
        return tool_payload(response)

    def test_docker_stop_falls_back_to_process_shutdown_when_cdp_is_unavailable(self) -> None:
        config = broker.BrokerConfig()
        launcher = broker.DockerSessionLauncher(config)
        session = broker.BrowserSession(
            id="session-a",
            mcp_session_key="mcp-a",
            profile_dir=Path("/tmp/profile"),
            cdp_port=31001,
            novnc_port=41001,
            novnc_backend_port=41002,
            container_name="tokimon-rb-session-a",
            container_id="a" * 64,
            created_at="2026-01-01T00:00:00+00:00",
            last_used_at="2026-01-01T00:00:00+00:00",
        )
        calls: list[list[str]] = []
        inspect_count = 0
        network_inspect_count = 0

        def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            nonlocal inspect_count, network_inspect_count
            calls.append(args)
            if args[:3] == ["docker", "container", "inspect"]:
                inspect_count += 1
                if inspect_count == 1:
                    payload = [
                        {
                            "Id": "a" * 64,
                            "Name": f"/{session.container_name}",
                            "Config": {
                                "Labels": {
                                    "tokimon.remote-browser-broker": "true",
                                    "tokimon.remote-browser-installation": config.installation_id,
                                    "tokimon.remote-browser-owner-user": config.owner_user,
                                    "tokimon.remote-browser-checkout": str(config.workspace_root),
                                    "tokimon.remote-browser-runtime-root": str(config.runtime_root),
                                    "tokimon.remote-browser-session": session.id,
                                    "tokimon.remote-browser-network-policy": config.network_policy,
                                }
                            },
                            "Mounts": [
                                {
                                    "Source": str(session.profile_dir),
                                    "Destination": "/profile",
                                }
                            ],
                        }
                    ]
                    return subprocess.CompletedProcess(
                        args,
                        0,
                        json.dumps(payload),
                        "",
                    )
                return subprocess.CompletedProcess(
                    args,
                    1,
                    "",
                    "Error: No such container",
                )
            if args[:3] == ["docker", "network", "inspect"]:
                network_inspect_count += 1
                if network_inspect_count == 1:
                    return subprocess.CompletedProcess(
                        args,
                        0,
                        owned_network_inspect_payload(launcher, session.id),
                        "",
                    )
                return subprocess.CompletedProcess(
                    args,
                    1,
                    "",
                    "Error: No such network",
                )
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch.object(
            broker.shutil,
            "which",
            return_value="/usr/bin/docker",
        ), patch.object(broker.subprocess, "run", side_effect=fake_run), patch.object(
            broker.CDPConnection, "for_session", side_effect=broker.BrowserBrokerError("CDP unavailable")
        ):
            launcher.stop(session)

        self.assertEqual(
            calls[0],
            ["docker", "container", "inspect", "tokimon-rb-session-a"],
        )
        self.assertEqual(calls[1][:3], ["docker", "exec", "a" * 64])
        self.assertIn("pkill -TERM", calls[1][-1])
        self.assertIn("[c]hrome.*--user-data-dir=/profile", calls[1][-1])
        self.assertEqual(calls[2], ["docker", "wait", "a" * 64])
        self.assertEqual(
            calls[3],
            ["docker", "container", "inspect", "a" * 64],
        )
        self.assertFalse(any(call[:3] == ["docker", "rm", "-f"] for call in calls))
        self.assertEqual(
            calls[4],
            ["docker", "network", "inspect", launcher._network_name(session.id)],
        )
        self.assertEqual(calls[5], ["docker", "network", "rm", "c" * 64])
        self.assertEqual(calls[6], ["docker", "network", "inspect", "c" * 64])

    def test_docker_start_without_cidfile_preserves_ambiguous_ownership(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        workspace_root = Path(tmpdir.name)
        config = broker.BrokerConfig(
            workspace_root=workspace_root,
            runtime_root=workspace_root / "runtime",
        )
        launcher = broker.DockerSessionLauncher(config)
        commands: list[list[str]] = []
        network_exists = False

        def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            nonlocal network_exists
            commands.append(args)
            if args[:3] == ["docker", "network", "inspect"]:
                if network_exists and args[-1] == launcher._network_name("session-a"):
                    return subprocess.CompletedProcess(
                        args,
                        0,
                        owned_network_inspect_payload(launcher, "session-a"),
                        "",
                    )
                return subprocess.CompletedProcess(
                    args,
                    1,
                    "",
                    "Error: No such network",
                )
            if args[:3] == ["docker", "network", "create"]:
                network_exists = True
                return subprocess.CompletedProcess(args, 0, "c" * 64, "")
            if args[:3] == ["docker", "network", "rm"]:
                network_exists = False
                return subprocess.CompletedProcess(args, 0, "c" * 64, "")
            return subprocess.CompletedProcess(
                args,
                125,
                "",
                "daemon disconnected after accepting run",
            )

        with patch.object(launcher, "ensure_image"), patch.object(
            broker.subprocess,
            "run",
            side_effect=fake_run,
        ):
            with self.assertRaises(broker.BrowserBrokerError) as raised:
                launcher.start(
                    session_id="session-a",
                    profile_dir=workspace_root / "profile",
                    cdp_port=31001,
                    novnc_backend_port=41001,
                )

        self.assertIs(raised.exception.details["cleanup_verified"], False)
        self.assertTrue(any(command[:2] == ["docker", "run"] for command in commands))
        self.assertFalse(any(command[:3] == ["docker", "rm", "-f"] for command in commands))
        self.assertTrue(any(command[:3] == ["docker", "network", "rm"] for command in commands))
        self.assertFalse(network_exists)

    def test_successful_docker_run_without_cidfile_is_still_ambiguous(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        workspace_root = Path(tmpdir.name)
        launcher = broker.DockerSessionLauncher(
            broker.BrokerConfig(
                workspace_root=workspace_root,
                runtime_root=workspace_root / "runtime",
            )
        )

        network_exists = False
        run_completed = False

        def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            nonlocal network_exists, run_completed
            if args[:3] == ["docker", "network", "inspect"]:
                if not network_exists or args[-1] != launcher._network_name("session-a"):
                    return subprocess.CompletedProcess(
                        args,
                        1,
                        "",
                        "Error: No such network",
                    )
                containers = (
                    {"a" * 64: {"Name": "tokimon-rb-session-a"}}
                    if run_completed
                    else {}
                )
                return subprocess.CompletedProcess(
                    args,
                    0,
                    owned_network_inspect_payload(
                        launcher,
                        "session-a",
                        containers=containers,
                    ),
                    "",
                )
            if args[:3] == ["docker", "network", "create"]:
                network_exists = True
                return subprocess.CompletedProcess(args, 0, "c" * 64, "")
            if args[:2] == ["docker", "run"]:
                run_completed = True
                return subprocess.CompletedProcess(args, 0, "a" * 64, "")
            raise AssertionError(f"unexpected Docker command: {args}")

        with patch.object(launcher, "ensure_image"), patch.object(
            broker.subprocess,
            "run",
            side_effect=fake_run,
        ):
            with self.assertRaises(broker.BrowserBrokerError) as raised:
                launcher.start(
                    session_id="session-a",
                    profile_dir=workspace_root / "profile",
                    cdp_port=31001,
                    novnc_backend_port=41001,
                )

        self.assertIs(raised.exception.details["cleanup_verified"], False)
        self.assertTrue(network_exists)

    def test_docker_stop_refuses_foreign_replacement_with_reused_name(self) -> None:
        config = broker.BrokerConfig()
        session = broker.BrowserSession(
            id="session-a",
            mcp_session_key="mcp-a",
            profile_dir=Path("/tmp/profile"),
            cdp_port=31001,
            novnc_port=41001,
            novnc_backend_port=41002,
            container_name="tokimon-rb-session-a",
            created_at="2026-01-01T00:00:00+00:00",
            last_used_at="2026-01-01T00:00:00+00:00",
        )
        calls: list[list[str]] = []
        foreign = [
            {
                "Id": "b" * 64,
                "Name": f"/{session.container_name}",
                "Config": {"Labels": {"unrelated": "true"}},
                "Mounts": [],
            }
        ]

        def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(foreign), "")

        with patch.object(
            broker.shutil,
            "which",
            return_value="/usr/bin/docker",
        ), patch.object(broker.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(broker.BrowserBrokerError):
                broker.DockerSessionLauncher(config).stop(session)

        self.assertEqual(
            calls,
            [["docker", "container", "inspect", session.container_name]],
        )

    def test_docker_stop_removes_owned_network_when_container_is_absent(self) -> None:
        config = broker.BrokerConfig()
        launcher = broker.DockerSessionLauncher(config)
        session = broker.BrowserSession(
            id="session-a",
            mcp_session_key="mcp-a",
            profile_dir=Path("/tmp/profile"),
            cdp_port=31001,
            novnc_port=41001,
            novnc_backend_port=41002,
            container_name="tokimon-rb-session-a",
            created_at="2026-01-01T00:00:00+00:00",
            last_used_at="2026-01-01T00:00:00+00:00",
        )
        commands: list[list[str]] = []

        def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            commands.append(args)
            if args[:3] == ["docker", "container", "inspect"]:
                return subprocess.CompletedProcess(
                    args,
                    1,
                    "",
                    "Error: No such container",
                )
            if args[:3] == ["docker", "network", "inspect"]:
                if args[-1] == launcher._network_name(session.id):
                    return subprocess.CompletedProcess(
                        args,
                        0,
                        owned_network_inspect_payload(launcher, session.id),
                        "",
                    )
                return subprocess.CompletedProcess(
                    args,
                    1,
                    "",
                    "Error: No such network",
                )
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch.object(broker.shutil, "which", return_value="/usr/bin/docker"), patch.object(
            broker.subprocess,
            "run",
            side_effect=fake_run,
        ):
            launcher.stop(session)

        self.assertEqual(commands[0][:3], ["docker", "container", "inspect"])
        self.assertEqual(
            commands[1],
            ["docker", "network", "inspect", launcher._network_name(session.id)],
        )
        self.assertEqual(commands[2], ["docker", "network", "rm", "c" * 64])
        self.assertEqual(commands[3], ["docker", "network", "inspect", "c" * 64])

    def test_docker_network_identity_fails_closed_for_foreign_or_ambiguous_records(self) -> None:
        launcher = broker.DockerSessionLauncher(broker.BrokerConfig())
        foreign = json.loads(owned_network_inspect_payload(launcher, "session-a"))
        foreign[0]["Labels"] = {"unrelated": "true"}
        payloads = {
            "foreign": json.dumps(foreign),
            "ambiguous": json.dumps([]),
        }

        for label, payload in payloads.items():
            with self.subTest(label=label):
                commands: list[list[str]] = []

                def fake_run(
                    args: list[str],
                    **_: object,
                ) -> subprocess.CompletedProcess[str]:
                    commands.append(args)
                    return subprocess.CompletedProcess(args, 0, payload, "")

                with patch.object(broker.subprocess, "run", side_effect=fake_run):
                    with self.assertRaises(broker.BrowserBrokerError):
                        launcher._ensure_session_network("session-a")

                self.assertEqual(len(commands), 1)
                self.assertEqual(commands[0][:3], ["docker", "network", "inspect"])

    def test_docker_network_name_is_short_deterministic_and_session_scoped(self) -> None:
        launcher = broker.DockerSessionLauncher(broker.BrokerConfig())

        first = launcher._network_name("session-a")

        self.assertEqual(first, launcher._network_name("session-a"))
        self.assertRegex(first, r"^tokimon-rbn-[0-9a-f]{20}$")
        self.assertNotEqual(first, launcher._network_name("session-b"))

    def test_docker_network_inspect_accepts_daemon_not_found_wording(self) -> None:
        launcher = broker.DockerSessionLauncher(broker.BrokerConfig())
        network_name = launcher._network_name("session-a")

        with patch.object(
            broker.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                ["docker", "network", "inspect", network_name],
                1,
                "",
                f"Error response from daemon: network {network_name} not found",
            ),
        ):
            inspected = launcher._inspect_owned_network("session-a")

        self.assertIsNone(inspected)

    def test_docker_start_refuses_owned_network_with_attached_container(self) -> None:
        launcher = broker.DockerSessionLauncher(broker.BrokerConfig())
        payload = owned_network_inspect_payload(
            launcher,
            "session-a",
            containers={"d" * 64: {"Name": "unexpected-peer"}},
        )
        commands: list[list[str]] = []

        def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            commands.append(args)
            return subprocess.CompletedProcess(args, 0, payload, "")

        with patch.object(broker.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(broker.BrowserBrokerError) as raised:
                launcher._ensure_session_network("session-a")

        self.assertIn("attached containers", raised.exception.message)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][:3], ["docker", "network", "inspect"])

    def test_docker_network_create_failure_removes_verified_empty_network(self) -> None:
        launcher = broker.DockerSessionLauncher(broker.BrokerConfig())
        network_exists = False
        commands: list[list[str]] = []

        def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            nonlocal network_exists
            commands.append(args)
            if args[:3] == ["docker", "network", "inspect"]:
                if network_exists and args[-1] == launcher._network_name("session-a"):
                    return subprocess.CompletedProcess(
                        args,
                        0,
                        owned_network_inspect_payload(launcher, "session-a"),
                        "",
                    )
                return subprocess.CompletedProcess(
                    args,
                    1,
                    "",
                    "Error: No such network",
                )
            if args[:3] == ["docker", "network", "create"]:
                network_exists = True
                return subprocess.CompletedProcess(
                    args,
                    1,
                    "",
                    "daemon disconnected after creating network",
                )
            if args[:3] == ["docker", "network", "rm"]:
                network_exists = False
                return subprocess.CompletedProcess(args, 0, "c" * 64, "")
            raise AssertionError(f"unexpected Docker command: {args}")

        with patch.object(broker.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(broker.BrowserBrokerError) as raised:
                launcher._ensure_session_network("session-a")

        self.assertIs(raised.exception.details["cleanup_verified"], True)
        self.assertFalse(network_exists)
        self.assertTrue(any(command[:3] == ["docker", "network", "rm"] for command in commands))

    def test_docker_network_post_create_verification_failure_cleans_owned_network(self) -> None:
        launcher = broker.DockerSessionLauncher(broker.BrokerConfig())
        network_exists = False
        post_create_verification_failed = False
        commands: list[list[str]] = []

        def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            nonlocal network_exists, post_create_verification_failed
            commands.append(args)
            if args[:3] == ["docker", "network", "inspect"]:
                if not network_exists:
                    return subprocess.CompletedProcess(
                        args,
                        1,
                        "",
                        "Error: No such network",
                    )
                if not post_create_verification_failed:
                    post_create_verification_failed = True
                    return subprocess.CompletedProcess(args, 0, "not-json", "")
                if args[-1] == "c" * 64:
                    return subprocess.CompletedProcess(
                        args,
                        1,
                        "",
                        "Error: No such network",
                    )
                return subprocess.CompletedProcess(
                    args,
                    0,
                    owned_network_inspect_payload(launcher, "session-a"),
                    "",
                )
            if args[:3] == ["docker", "network", "create"]:
                network_exists = True
                return subprocess.CompletedProcess(args, 0, "c" * 64, "")
            if args[:3] == ["docker", "network", "rm"]:
                network_exists = False
                return subprocess.CompletedProcess(args, 0, "c" * 64, "")
            raise AssertionError(f"unexpected Docker command: {args}")

        with patch.object(broker.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(broker.BrowserBrokerError) as raised:
                launcher._ensure_session_network("session-a")

        self.assertIs(raised.exception.details["cleanup_verified"], True)
        self.assertFalse(network_exists)
        self.assertTrue(post_create_verification_failed)
        self.assertTrue(any(command[:3] == ["docker", "network", "rm"] for command in commands))

    def test_replace_config_preserves_novnc_token_ttl(self) -> None:
        config = broker.BrokerConfig(
            novnc_token_ttl_seconds=123,
        )

        replaced = broker.replace_config(config, broker_port=8940)

        self.assertEqual(replaced.broker_port, 8940)
        self.assertEqual(replaced.novnc_token_ttl_seconds, 123)
        self.assertEqual(replaced.network_policy, broker.DEFAULT_NETWORK_POLICY)

    def test_browser_image_binds_internal_cdp_proxy_to_all_interfaces(self) -> None:
        script = (broker.IMAGE_CONTEXT_DIR / "start-session-browser.sh").read_text(encoding="utf-8")

        self.assertIn("TCP-LISTEN:9223,fork,bind=0.0.0.0", script)
        self.assertIn("--remote-debugging-address=127.0.0.1", script)
        self.assertNotIn("--remote-debugging-address=0.0.0.0", script)

    def make_http_server(
        self,
        *,
        capture_logs: bool = True,
    ) -> tuple[broker.ThreadingHTTPServer, int, broker.RemoteBrowserBroker, FakeLauncher, FakeController]:
        if capture_logs:
            self.capture_runtime_logger()
        remote_broker, launcher, controller = self.make_broker()
        mcp_server = broker.MCPServer(remote_broker)
        httpd = broker.ThreadingHTTPServer(("127.0.0.1", 0), broker.make_handler(mcp_server))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return httpd, httpd.server_address[1], remote_broker, launcher, controller

    def post_mcp(
        self,
        port: int,
        payload: dict[str, object],
        *,
        session_id: str | None = None,
    ) -> tuple[int, dict[str, str], dict[str, object]]:
        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["mcp-session-id"] = session_id
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request("POST", "/mcp", body=json.dumps(payload), headers=headers)
            response = connection.getresponse()
            response_headers = {key.lower(): value for key, value in response.getheaders()}
            body = response.read()
            return response.status, response_headers, parse_sse_payload(body)
        finally:
            connection.close()

    def call_tool_http(
        self,
        port: int,
        name: str,
        arguments: dict[str, object],
        *,
        request_id: object = 1,
        session_id: str | None = None,
    ) -> tuple[dict[str, object], dict[str, str]]:
        status, headers, payload = self.post_mcp(
            port,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            session_id=session_id,
        )
        self.assertEqual(status, 200)
        result = payload["result"]
        self.assertIsInstance(result, dict)
        return tool_payload(payload), headers

    def test_tools_list_contract(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)

        response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, mcp_session_key="mcp-a")

        self.assertIsNotNone(response)
        result = response["result"]
        self.assertIsInstance(result, dict)
        names = {tool["name"] for tool in result["tools"]}
        self.assertEqual(
            names,
            {
                "browser_whoami",
                "browser_ensure",
                "browser_current_session_status",
                "browser_template_list",
                "browser_template_create",
                "browser_template_seal",
                "browser_template_refresh",
                "browser_template_delete",
                "browser_connect_info",
                "browser_session_list",
                "browser_navigate",
                "browser_snapshot",
                "browser_artifact",
                "browser_evaluate",
                "browser_screenshot",
                "browser_click",
                "browser_type",
                "browser_upload_file",
                "browser_backend_request",
                "browser_press_key",
                "browser_wait",
                "browser_new_tab",
                "browser_switch_tab",
                "browser_close_tab",
                "browser_pause_for_human",
                "browser_resume",
                "browser_session_extend_retention",
                "browser_session_delete_retained",
                "browser_session_recover",
                "browser_session_cleanup_artifacts",
                "browser_session_cleanup_expired",
                "browser_session_cleanup_orphaned",
                "browser_diagnostics",
                "browser_stop",
            },
        )
        ensure_tool = next(tool for tool in result["tools"] if tool["name"] == "browser_ensure")
        self.assertEqual(
            ensure_tool["inputSchema"]["properties"]["recreate"],
            {"type": "boolean"},
        )
        seal_tool = next(
            tool for tool in result["tools"]
            if tool["name"] == "browser_template_seal"
        )
        self.assertNotIn("Operator-only", seal_tool["description"])
        screenshot_tool = next(
            tool for tool in result["tools"] if tool["name"] == "browser_screenshot"
        )
        self.assertEqual(
            screenshot_tool["inputSchema"],
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "format": {"type": "string", "enum": ["png", "jpeg"]},
                },
            },
        )
        session_list_tool = next(
            tool for tool in result["tools"] if tool["name"] == "browser_session_list"
        )
        self.assertIn(
            "redacted broker-known browser sessions",
            session_list_tool["description"],
        )
        self.assertNotIn("Operator-only", session_list_tool["description"])
        mutation_tools = {
            "browser_ensure",
            "browser_navigate",
            "browser_evaluate",
            "browser_click",
            "browser_type",
            "browser_upload_file",
            "browser_backend_request",
            "browser_press_key",
            "browser_new_tab",
        }
        for tool in result["tools"]:
            if tool["name"] in mutation_tools:
                properties = tool["inputSchema"]["properties"]
                self.assertNotIn("approval_id", properties)
                self.assertNotIn("action_class", properties)

        pause_tool = next(
            tool for tool in result["tools"] if tool["name"] == "browser_pause_for_human"
        )
        self.assertEqual(set(pause_tool["inputSchema"]["properties"]), {"reason"})

    def test_stdio_serve_reuses_process_session_key_and_logs_to_stderr(self) -> None:
        class RecordingServer:
            def __init__(self) -> None:
                self.session_keys: list[str] = []

            def handle(
                self,
                request: dict[str, object],
                *,
                mcp_session_key: str,
            ) -> dict[str, object]:
                self.session_keys.append(mcp_session_key)
                return broker.response(request.get("id"), {"session_key": mcp_session_key})

        log_stream = self.capture_runtime_logger()
        recording_server = RecordingServer()
        input_stream = io.StringIO(
            "\n".join(
                (
                    '{"jsonrpc":"2.0","id":1,"method":"ping"}',
                    '{"jsonrpc":"2.0","id":2,"method":"ping"}',
                )
            )
            + "\n"
        )
        output_stream = io.StringIO()

        status = broker.serve(
            input_stream=input_stream,
            output_stream=output_stream,
            server=recording_server,  # type: ignore[arg-type]
        )

        responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
        self.assertEqual(status, 0)
        self.assertEqual(len(recording_server.session_keys), 2)
        self.assertEqual(recording_server.session_keys[0], recording_server.session_keys[1])
        self.assertTrue(recording_server.session_keys[0].startswith("mcp_"))
        self.assertEqual(responses[0]["result"]["session_key"], recording_server.session_keys[0])
        self.assertEqual(responses[1]["result"]["session_key"], recording_server.session_keys[0])
        self.assertIn('"runtime": "stdio-mcp"', log_stream.getvalue())
        self.assertNotIn("server_start", output_stream.getvalue())

    def test_stdio_eof_preserves_template_editing_draft_for_restart(self) -> None:
        self.capture_runtime_logger()
        remote_broker, launcher, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "browser_template_create",
                "arguments": {"template_name": "stdio-draft"},
            },
        }
        output = io.StringIO()

        return_code = broker.serve(
            input_stream=io.StringIO(json.dumps(request) + "\n"),
            output_stream=output,
            server=server,
        )

        payload = tool_payload(json.loads(output.getvalue()))
        draft_profile = Path(launcher.started[0]["profile_dir"])
        peer = broker.RemoteBrowserBroker(
            config=remote_broker.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                peer.begin_shutdown(),
                peer.release_shutdown_leases(),
            )
        )
        sessions = peer.session_list()["sessions"]
        self.assertEqual(return_code, 0)
        self.assertEqual(payload["status"], "editing")
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["state"], "stopped_retained")
        self.assertEqual(sessions[0]["template_role"], broker.TEMPLATE_ROLE_EDITING)
        self.assertTrue((draft_profile.parent / "session-metadata.json").exists())
        self.assertEqual(len(launcher.stopped), 1)


    def test_http_post_mcp_tools_call_contract_and_structured_logs(self) -> None:
        log_stream = self.capture_runtime_logger()
        _, port, remote_broker, launcher, _ = self.make_http_server(capture_logs=False)

        list_status, list_headers, list_payload = self.post_mcp(
            port,
            {"jsonrpc": "2.0", "id": "tools", "method": "tools/list"},
        )
        session_id = list_headers["mcp-session-id"]
        ensure_payload, ensure_headers = self.call_tool_http(
            port,
            "browser_ensure",
            {},
            request_id="ensure",
            session_id=session_id,
        )
        whoami_payload, _ = self.call_tool_http(
            port,
            "browser_whoami",
            {},
            request_id="whoami",
            session_id=session_id,
        )

        self.assertEqual(list_status, 200)
        self.assertEqual(list_payload["id"], "tools")
        self.assertIn("tools", list_payload["result"])
        self.assertTrue(session_id.startswith("mcp_"))
        self.assertEqual(ensure_headers["mcp-session-id"], session_id)
        self.assertEqual(whoami_payload["mcp_session_key"], session_id)
        self.assertEqual(whoami_payload["browser_session"]["id"], ensure_payload["browser_session"]["id"])
        self.assertEqual(launcher.started[0]["session_id"], ensure_payload["browser_session"]["id"])
        self.assertTrue(remote_broker.whoami(session_id)["has_browser_session"])

        entries = [json.loads(line) for line in log_stream.getvalue().splitlines()]
        self.assertTrue(
            any(
                entry["message"] == "request_received"
                and entry["request_id"] == "tools"
                and entry["method"] == "tools/list"
                and entry["runtime"] == "http-mcp"
                for entry in entries
            )
        )
        self.assertTrue(
            any(
                entry["message"] == "request_received"
                and entry["request_id"] == "ensure"
                and entry["correlation_id"] == session_id
                and entry["mcp_session_key"] == session_id
                for entry in entries
            )
        )
        self.assertTrue(any(entry["message"] == "http_request" for entry in entries))

    def test_whoami_reports_mcp_session_without_browser_session(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_whoami", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        payload = tool_payload(response)
        self.assertEqual(payload["mcp_session_key"], "mcp-a")
        self.assertFalse(payload["has_browser_session"])

    def test_current_session_status_is_scoped_to_current_mcp_session(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        ensure_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "browser_ensure", "arguments": {}},
        }
        first = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-a"))
        second = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-b"))

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "browser_current_session_status", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        payload = tool_payload(response)
        self.assertTrue(payload["has_browser_session"])
        self.assertEqual(payload["browser_session"]["id"], first["browser_session"]["id"])
        self.assertNotEqual(payload["browser_session"]["id"], second["browser_session"]["id"])
        self.assertNotIn("profile_dir", payload["browser_session"])
        self.assertNotIn("cdp_port", payload["browser_session"])

    def test_session_list_allows_normal_session(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "browser_session_list", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        payload = tool_payload(response)
        current = remote_broker.whoami("mcp-a")["browser_session"]
        self.assertEqual(payload["session_count"], 1)
        self.assertEqual(payload["sessions"][0]["id"], current["id"])
        self.assertNotIn("profile_dir", payload["sessions"][0])
        self.assertNotIn("cdp_port", payload["sessions"][0])

    def test_diagnostics_omit_connect_urls_and_page_content(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "browser_diagnostics", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        payload = tool_payload(response)
        serialized = json.dumps(payload)
        self.assertIn("diagnostics", payload)
        self.assertNotIn("novnc_url", serialized)
        self.assertNotIn("cdp_url", serialized)
        self.assertNotIn("https://example.com", serialized)
        self.assertNotIn("text", payload.get("diagnostics", {}))
        network_policy = payload["diagnostics"]["network_policy"]
        self.assertEqual(network_policy["id"], broker.DEFAULT_NETWORK_POLICY)
        self.assertEqual(
            network_policy["docker_network_mode"],
            "per_session_user_defined_bridge",
        )
        self.assertEqual(
            network_policy["peer_container_isolation"],
            "enforced_for_unattached_containers",
        )
        self.assertEqual(network_policy["internet_egress"], "allowed")
        self.assertEqual(
            network_policy["private_local_networks"],
            "not_enforced_by_broker",
        )
        self.assertEqual(
            network_policy["enforcement"],
            "per_session_network_membership_only",
        )
        self.assertIn("does not enforce", " ".join(network_policy["limitations"]))
        self.assertIn("Docker authority", " ".join(network_policy["limitations"]))
        self.assertTrue(
            payload["diagnostics"]["fingerprint_baseline"]["checks"]["webdriver_not_true"]
        )
        self.assertIn("does not claim", payload["diagnostics"]["fingerprint_baseline"]["claim"])

    def test_health_reports_active_network_policy(self) -> None:
        config = broker.BrokerConfig(
            docker_command="missing-docker-command",
        )

        payload = broker.health_payload(config)

        network_policy = payload["network_policy"]
        self.assertEqual(network_policy["id"], broker.DEFAULT_NETWORK_POLICY)
        self.assertEqual(
            network_policy["docker_network_mode"],
            "per_session_user_defined_bridge",
        )
        self.assertEqual(network_policy["local_test_pages"], "disabled_without_explicit_test_setup")
        self.assertIn("Desktop", " ".join(network_policy["limitations"]))
        self.assertEqual(payload["installation"]["id"], config.installation_id)
        self.assertEqual(
            payload["installation"]["default_http_port"],
            broker.default_broker_port(config.installation_id),
        )
        self.assertIn("resources", payload["docker"])

    def test_docker_resource_inventory_classifies_owned_and_foreign(self) -> None:
        config = broker.BrokerConfig(installation_id="joey-test")

        def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if args[:3] == ["docker", "ps", "-aq"] and "label=tokimon.remote-browser-broker=true" in args:
                return subprocess.CompletedProcess(args, 0, "owned\nforeign\n", "")
            if args[:2] == ["docker", "inspect"] and args[2:] == ["owned", "foreign"]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    json.dumps(
                        [
                            {
                                "Id": "owned-container-id",
                                "Name": "/tokimon-rb-joey-test-session-a",
                                "Config": {
                                    "Image": config.image_name,
                                    "Labels": {
                                        "tokimon.remote-browser-broker": "true",
                                        "tokimon.remote-browser-installation": "joey-test",
                                    },
                                },
                                "State": {"Status": "running"},
                            },
                            {
                                "Id": "foreign-container-id",
                                "Name": "/tokimon-rb-other-session-b",
                                "Config": {
                                    "Image": "tokimon-remote-browser-session:other",
                                    "Labels": {"tokimon.remote-browser-broker": "true"},
                                },
                                "State": {"Status": "running"},
                            },
                        ]
                    ),
                    "",
                )
            raise AssertionError(f"unexpected docker command: {args}")

        with patch.object(broker.subprocess, "run", side_effect=fake_run):
            inventory = broker.docker_resource_inventory(config)

        self.assertEqual(inventory["owned_container_count"], 1)
        self.assertEqual(inventory["foreign_tokimon_container_count"], 1)
        self.assertIn("foreign_or_unscoped_tokimon_browser_containers_present", inventory["warnings"])

    def test_connect_info_omits_raw_cdp_and_uses_token_gated_novnc_proxy(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        payload = tool_payload(response)
        self.assertTrue(payload["connect"]["redacted"])
        self.assertEqual(payload["connect"]["tool"], "browser_connect_info")
        self.assertEqual(FakeNoVNCProxy.started, [])

        connect_payload = self.call_mcp_tool(
            server, mcp_session_key="mcp-a", name="browser_connect_info"
        )
        connect = connect_payload["connect"]
        self.assertNotIn("operator_authorized", connect)
        self.assertNotIn("cdp_url", connect)
        self.assertNotIn("cdp", connect["host_ports"])
        self.assertEqual(
            connect["cdp"]["exposure"],
            "host_loopback_same_os_authority",
        )
        self.assertIn("rb_token=", connect["novnc_url"])
        self.assertEqual(len(FakeNoVNCProxy.started), 1)
        self.assertEqual(
            FakeNoVNCProxy.started[0]["backend_port"],
            launcher.started[0]["novnc_backend_port"],
        )

    def test_novnc_proxy_requires_token_and_sets_cookie(self) -> None:
        class BackendHandler(broker.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server API.
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"backend-ok")

            def log_message(self, format: str, *args: object) -> None:
                return

        backend = broker.ThreadingHTTPServer(("127.0.0.1", 0), BackendHandler)
        backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
        backend_thread.start()
        self.addCleanup(backend.server_close)
        self.addCleanup(backend.shutdown)
        proxy = broker.NoVNCProxy(
            listen_host="127.0.0.1",
            listen_port=0,
            backend_host="127.0.0.1",
            backend_port=backend.server_address[1],
            token="secret-token",
        )
        proxy.start()
        self.addCleanup(proxy.stop)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.listen_port, timeout=2)
        conn.request("GET", "/vnc.html")
        rejected = conn.getresponse()
        rejected.read()
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", proxy.listen_port, timeout=2)
        conn.request("GET", "/vnc.html?rb_token=secret-token")
        accepted = conn.getresponse()
        accepted_body = accepted.read()
        cookie = accepted.getheader("Set-Cookie")
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", proxy.listen_port, timeout=2)
        conn.request("GET", "/app.js", headers={"Cookie": cookie or ""})
        accepted_by_cookie = conn.getresponse()
        accepted_by_cookie_body = accepted_by_cookie.read()
        conn.close()

        self.assertEqual(rejected.status, 403)
        self.assertEqual(accepted.status, 200)
        self.assertEqual(accepted_body, b"backend-ok")
        self.assertIn("tokimon_rb_novnc_token=secret-token", cookie or "")
        self.assertEqual(accepted_by_cookie.status, 200)
        self.assertEqual(accepted_by_cookie_body, b"backend-ok")

    def test_novnc_proxy_rejects_expired_token(self) -> None:
        class BackendHandler(broker.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server API.
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"backend-ok")

            def log_message(self, format: str, *args: object) -> None:
                return

        backend = broker.ThreadingHTTPServer(("127.0.0.1", 0), BackendHandler)
        backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
        backend_thread.start()
        self.addCleanup(backend.server_close)
        self.addCleanup(backend.shutdown)
        proxy = broker.NoVNCProxy(
            listen_host="127.0.0.1",
            listen_port=0,
            backend_host="127.0.0.1",
            backend_port=backend.server_address[1],
            token="secret-token",
            expires_at="2000-01-01T00:00:00+00:00",
        )
        proxy.start()
        self.addCleanup(proxy.stop)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.listen_port, timeout=2)
        conn.request("GET", "/vnc.html?rb_token=secret-token")
        rejected = conn.getresponse()
        body = rejected.read()
        conn.close()

        self.assertEqual(rejected.status, 403)
        self.assertIn(b"expired", body)

    def test_novnc_proxy_closes_active_websocket_when_token_expires(self) -> None:
        backend_ready = threading.Event()
        stop_backend = threading.Event()

        def serve_backend(server_socket: socket.socket) -> None:
            server_socket.listen(1)
            backend_ready.set()
            connection, _ = server_socket.accept()
            with connection:
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = connection.recv(4096)
                    if not chunk:
                        return
                    request += chunk
                connection.sendall(
                    b"HTTP/1.1 101 Switching Protocols\r\n"
                    b"Upgrade: websocket\r\n"
                    b"Connection: Upgrade\r\n\r\n"
                )
                while not stop_backend.is_set():
                    data = connection.recv(65536)
                    if not data:
                        return
                    connection.sendall(data)

        backend_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        backend_socket.bind(("127.0.0.1", 0))
        backend_thread = threading.Thread(target=serve_backend, args=(backend_socket,), daemon=True)
        backend_thread.start()
        self.addCleanup(stop_backend.set)
        self.addCleanup(backend_socket.close)
        self.assertTrue(backend_ready.wait(timeout=2))

        proxy = broker.NoVNCProxy(
            listen_host="127.0.0.1",
            listen_port=0,
            backend_host="127.0.0.1",
            backend_port=backend_socket.getsockname()[1],
            token="secret-token",
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        )
        expired = False

        def fake_is_expired(value: str | None) -> bool:
            return expired

        with patch.object(broker, "is_expired", fake_is_expired):
            proxy.start()
            self.addCleanup(proxy.stop)

            client = socket.create_connection(("127.0.0.1", proxy.listen_port), timeout=2)
            self.addCleanup(client.close)
            client.settimeout(2)
            client.sendall(
                b"GET /websockify?rb_token=secret-token HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Upgrade: websocket\r\n"
                b"Connection: Upgrade\r\n\r\n"
            )
            self.assertIn(b"101 Switching Protocols", client.recv(4096))
            client.sendall(b"before-expiry")
            self.assertEqual(client.recv(len(b"before-expiry")), b"before-expiry")

            expired = True
            client.sendall(b"after-expiry")
            try:
                echoed = client.recv(len(b"after-expiry"))
            except socket.timeout:
                echoed = b""

        self.assertNotEqual(echoed, b"after-expiry")

    def test_novnc_proxy_stop_closes_established_websocket(self) -> None:
        backend_ready = threading.Event()

        def serve_backend(server_socket: socket.socket) -> None:
            server_socket.listen(1)
            backend_ready.set()
            connection, _ = server_socket.accept()
            with connection:
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = connection.recv(4096)
                    if not chunk:
                        return
                    request += chunk
                connection.sendall(
                    b"HTTP/1.1 101 Switching Protocols\r\n"
                    b"Upgrade: websocket\r\n"
                    b"Connection: Upgrade\r\n\r\n"
                )
                while True:
                    data = connection.recv(65536)
                    if not data:
                        return
                    connection.sendall(data)

        backend_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        backend_socket.bind(("127.0.0.1", 0))
        backend_thread = threading.Thread(
            target=serve_backend,
            args=(backend_socket,),
            daemon=True,
        )
        backend_thread.start()
        self.addCleanup(backend_socket.close)
        self.assertTrue(backend_ready.wait(timeout=2))
        proxy = broker.NoVNCProxy(
            listen_host="127.0.0.1",
            listen_port=0,
            backend_host="127.0.0.1",
            backend_port=backend_socket.getsockname()[1],
            token="secret-token",
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        )
        proxy.start()
        self.addCleanup(proxy.stop)
        client = socket.create_connection(("127.0.0.1", proxy.listen_port), timeout=2)
        self.addCleanup(client.close)
        client.settimeout(1)
        client.sendall(
            b"GET /websockify?rb_token=secret-token HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n\r\n"
        )
        self.assertIn(b"101 Switching Protocols", client.recv(4096))
        client.sendall(b"before-stop")
        self.assertEqual(client.recv(len(b"before-stop")), b"before-stop")

        proxy.stop()
        try:
            client.sendall(b"after-stop")
            echoed = client.recv(len(b"after-stop"))
        except OSError:
            echoed = b""

        self.assertNotEqual(echoed, b"after-stop")

    def test_novnc_proxy_stop_failure_retains_server_for_retry(self) -> None:
        proxy = broker.NoVNCProxy(
            listen_host="127.0.0.1",
            listen_port=0,
            backend_host="127.0.0.1",
            backend_port=9,
            token="secret-token",
        )
        proxy.start()
        self.addCleanup(proxy.stop)
        httpd = proxy._httpd
        thread = proxy._thread
        self.assertIsNotNone(httpd)
        self.assertIsNotNone(thread)

        with patch.object(httpd, "shutdown", side_effect=RuntimeError("shutdown failed")):
            with self.assertRaises(RuntimeError):
                proxy.stop()

        self.assertIs(proxy._httpd, httpd)
        self.assertIs(proxy._thread, thread)
        self.assertTrue(thread.is_alive())

        proxy.stop()
        self.assertIsNone(proxy._httpd)
        self.assertIsNone(proxy._thread)
        self.assertFalse(thread.is_alive())

    def test_novnc_proxy_thread_start_failure_can_be_cleaned_without_shutdown_deadlock(self) -> None:
        proxy = broker.NoVNCProxy(
            listen_host="127.0.0.1",
            listen_port=0,
            backend_host="127.0.0.1",
            backend_port=9,
            token="secret-token",
        )

        with patch.object(threading.Thread, "start", side_effect=RuntimeError("no thread")):
            with self.assertRaises(RuntimeError):
                proxy.start()

        cleanup_thread = threading.Thread(target=proxy.stop)
        cleanup_thread.start()
        cleanup_thread.join(timeout=1)
        self.assertFalse(cleanup_thread.is_alive())
        self.assertIsNone(proxy._httpd)
        self.assertIsNone(proxy._thread)

    def test_stop_retains_proxy_ownership_when_revocation_fails(self) -> None:
        remote_broker, _, _ = self.make_broker()
        session = remote_broker._ensure_session("mcp-a")
        remote_broker.connect_info("mcp-a")
        proxy = remote_broker._novnc_proxies[session.id]

        with patch.object(proxy, "stop", side_effect=RuntimeError("stop failed")):
            with self.assertRaises(broker.BrowserBrokerError):
                remote_broker.stop("mcp-a")

        self.assertEqual(session.state, broker.STATE_UNKNOWN)
        self.assertFalse(session.novnc_token_active)
        self.assertIsNone(session.novnc_token)
        self.assertIs(remote_broker._novnc_proxies[session.id], proxy)

        retried = remote_broker.stop("mcp-a")
        self.assertEqual(retried["browser_session"]["state"], broker.STATE_STOPPED_RETAINED)
        self.assertNotIn(session.id, remote_broker._novnc_proxies)

    def test_connect_info_reuses_novnc_link_for_24_hours_then_rotates(self) -> None:
        remote_broker, _, _ = self.make_broker()
        issued_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        now = issued_at

        class ClockDatetime(datetime):
            @classmethod
            def now(cls, tz: object = None) -> datetime:
                return now if tz else now.replace(tzinfo=None)

        with patch.object(broker, "datetime", ClockDatetime):
            remote_broker.ensure("mcp-a")
            first = remote_broker.connect_info("mcp-a")
            self.assertEqual(
                first["connect"]["novnc_token_expires_at"],
                (issued_at + timedelta(hours=24)).isoformat(),
            )
            now = issued_at + timedelta(hours=23, minutes=59)
            reused = remote_broker.connect_info("mcp-a")
            self.assertEqual(reused["connect"], first["connect"])
            self.assertEqual(len(FakeNoVNCProxy.started), 1)

            now = issued_at + timedelta(hours=24)
            second = remote_broker.connect_info("mcp-a")

        self.assertNotEqual(second["connect"]["novnc_url"], first["connect"]["novnc_url"])
        self.assertEqual(len(FakeNoVNCProxy.started), 2)
        self.assertTrue(FakeNoVNCProxy.stopped)
        self.assertEqual(
            second["connect"]["novnc_token_expires_at"],
            (issued_at + timedelta(hours=48)).isoformat(),
        )

    def test_connect_info_proxy_start_failure_leaves_no_active_credential(self) -> None:
        remote_broker, _, _ = self.make_broker()
        session = remote_broker._ensure_session("mcp-a")

        with patch.object(FakeNoVNCProxy, "start", side_effect=RuntimeError("bind failed")):
            with self.assertRaises(RuntimeError):
                remote_broker.connect_info("mcp-a")

        self.assertEqual(remote_broker._novnc_proxies, {})
        self.assertFalse(session.novnc_token_active)
        self.assertIsNone(session.novnc_token)
        self.assertIsNone(session.novnc_token_expires_at)
        self.assertEqual(session.token_exposure_mode, "revoked")

        recovered = remote_broker.connect_info("mcp-a")
        self.assertIn("rb_token=", recovered["connect"]["novnc_url"])
        self.assertTrue(session.novnc_token_active)

    def test_pause_preserves_viewer_link_until_resume(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        remote_broker.ensure("mcp-a")
        initial = remote_broker.connect_info("mcp-a")["connect"]
        paused = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": "pause",
                    "method": "tools/call",
                    "params": {
                        "name": "browser_pause_for_human",
                        "arguments": {"reason": "  human takeover  "},
                    },
                },
                mcp_session_key="mcp-a",
            )
        )
        repeated = remote_broker.pause_for_human("mcp-a")
        self.assertEqual(paused["browser_session"]["state"], broker.STATE_PAUSED_OBSERVED)
        self.assertEqual(paused["reason"], "human takeover")
        self.assertEqual(repeated["browser_session"]["state"], broker.STATE_PAUSED_OBSERVED)
        self.assertEqual(remote_broker.connect_info("mcp-a")["connect"], initial)
        self.assertFalse(FakeNoVNCProxy.stopped)

        resumed = self.call_mcp_tool(
            server, mcp_session_key="mcp-a", name="browser_resume"
        )
        self.assertEqual(resumed["browser_session"]["state"], broker.STATE_RUNNING)
        self.assertEqual(remote_broker.connect_info("mcp-a")["connect"], initial)
        self.assertEqual(remote_broker.navigate("mcp-a", "example.com")["status"], "navigated")

    def test_pause_rejects_removed_mode_argument(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        remote_broker.ensure("mcp-a")
        for mode in ("manual_private", "manual_observed", None):
            with self.subTest(mode=mode):
                result = tool_payload(
                    server.handle(
                        {
                            "jsonrpc": "2.0",
                            "id": "unsupported-pause-mode",
                            "method": "tools/call",
                            "params": {
                                "name": "browser_pause_for_human",
                                "arguments": {"mode": mode},
                            },
                        },
                        mcp_session_key="mcp-a",
                    )
                )
                self.assertEqual(result["error_code"], broker.ERROR_INVALID_ARGUMENT)
                self.assertEqual(
                    remote_broker.current_session_status("mcp-a")["browser_session"]["state"],
                    broker.STATE_RUNNING,
                )

    def test_pause_waits_for_inflight_screenshot_and_text_wait(self) -> None:
        for operation in ("screenshot", "wait"):
            with self.subTest(operation=operation):
                controller = BlockingController(operation)
                remote_broker, _, _ = self.make_broker(controller=controller)
                remote_broker._ensure_session("mcp-a")
                operation_lock = self.track_operation_lock(remote_broker, "mcp-a")
                operation_errors: list[BaseException] = []
                pause_errors: list[BaseException] = []
                pause_started = threading.Event()
                pause_finished = threading.Event()

                def run_operation() -> None:
                    try:
                        if operation == "screenshot":
                            remote_broker.screenshot("mcp-a")
                        else:
                            remote_broker.wait("mcp-a", text="visible text", timeout_ms=1000)
                    except BaseException as exc:  # noqa: BLE001 - surfaced after join.
                        operation_errors.append(exc)

                def run_pause() -> None:
                    pause_started.set()
                    try:
                        remote_broker.pause_for_human("mcp-a")
                    except BaseException as exc:  # noqa: BLE001 - surfaced after join.
                        pause_errors.append(exc)
                    finally:
                        pause_finished.set()

                operation_thread = threading.Thread(
                    target=run_operation,
                    name="blocked-page-operation",
                )
                operation_thread.start()
                self.assertTrue(controller.operation_entered.wait(timeout=1))
                pause_thread = threading.Thread(
                    target=run_pause,
                    name="human-pause",
                )
                pause_thread.start()
                self.assertTrue(pause_started.wait(timeout=1))
                self.assertTrue(
                    operation_lock.attempted_by("human-pause").wait(timeout=1)
                )
                self.assertFalse(pause_finished.is_set())

                controller.operation_release.set()
                operation_thread.join(timeout=2)
                pause_thread.join(timeout=2)

                self.assertFalse(operation_thread.is_alive())
                self.assertFalse(pause_thread.is_alive())
                self.assertEqual(operation_errors, [])
                self.assertEqual(pause_errors, [])
                session = remote_broker._session_or_error("mcp-a")
                self.assertEqual(session.state, broker.STATE_PAUSED_OBSERVED)
                run_operation()
                if operation == "screenshot":
                    self.assertEqual(operation_errors, [])
                else:
                    self.assertEqual(len(operation_errors), 1)
                    self.assertIsInstance(operation_errors[0], broker.BrowserBrokerError)
                    self.assertEqual(operation_errors[0].code, broker.ERROR_PAUSED)

    def test_pause_waits_for_diagnostics_live_page_read(self) -> None:
        controller = BlockingController("diagnostics")
        remote_broker, _, _ = self.make_broker(controller=controller)
        remote_broker._ensure_session("mcp-a")
        operation_lock = self.track_operation_lock(remote_broker, "mcp-a")
        diagnostics_errors: list[BaseException] = []
        pause_errors: list[BaseException] = []
        pause_started = threading.Event()
        pause_finished = threading.Event()

        def run_diagnostics() -> None:
            try:
                remote_broker.diagnostics("mcp-a")
            except BaseException as exc:  # noqa: BLE001 - surfaced after join.
                diagnostics_errors.append(exc)

        def run_pause() -> None:
            pause_started.set()
            try:
                remote_broker.pause_for_human("mcp-a")
            except BaseException as exc:  # noqa: BLE001 - surfaced after join.
                pause_errors.append(exc)
            finally:
                pause_finished.set()

        diagnostics_thread = threading.Thread(
            target=run_diagnostics,
            name="blocked-diagnostics",
        )
        diagnostics_thread.start()
        self.assertTrue(controller.operation_entered.wait(timeout=1))
        pause_thread = threading.Thread(target=run_pause, name="human-pause")
        pause_thread.start()
        self.assertTrue(pause_started.wait(timeout=1))
        self.assertTrue(operation_lock.attempted_by("human-pause").wait(timeout=1))
        self.assertFalse(pause_finished.is_set())

        controller.operation_release.set()
        diagnostics_thread.join(timeout=2)
        pause_thread.join(timeout=2)

        self.assertFalse(diagnostics_thread.is_alive())
        self.assertFalse(pause_thread.is_alive())
        self.assertEqual(diagnostics_errors, [])
        self.assertEqual(pause_errors, [])
        after_pause = remote_broker.diagnostics("mcp-a")
        self.assertEqual(after_pause["diagnostics"]["state"], broker.STATE_PAUSED_OBSERVED)
        self.assertNotIn("fingerprint_baseline", after_pause["diagnostics"])

    def test_pause_waits_for_snapshot_result_construction(self) -> None:
        remote_broker, _, _ = self.make_broker()
        remote_broker._ensure_session("mcp-a")
        operation_lock = self.track_operation_lock(remote_broker, "mcp-a")
        status_entered = threading.Event()
        status_release = threading.Event()
        pause_started = threading.Event()
        pause_finished = threading.Event()
        snapshot_results: list[dict[str, object]] = []
        pause_results: list[dict[str, object]] = []
        original_to_status = broker.BrowserSession.to_status_dict

        def gated_to_status(session: broker.BrowserSession) -> dict[str, object]:
            if threading.current_thread().name == "snapshot-result-construction":
                status_entered.set()
                if not status_release.wait(timeout=2):
                    raise RuntimeError("timed out waiting to release snapshot status")
            return original_to_status(session)

        def run_snapshot() -> None:
            snapshot_results.append(remote_broker.snapshot("mcp-a"))

        def run_pause() -> None:
            pause_started.set()
            try:
                pause_results.append(
                    remote_broker.pause_for_human("mcp-a")
                )
            finally:
                pause_finished.set()

        with patch.object(broker.BrowserSession, "to_status_dict", gated_to_status):
            snapshot_thread = threading.Thread(
                target=run_snapshot,
                name="snapshot-result-construction",
            )
            snapshot_thread.start()
            self.assertTrue(status_entered.wait(timeout=1))
            pause_thread = threading.Thread(target=run_pause, name="human-pause")
            pause_thread.start()
            self.assertTrue(pause_started.wait(timeout=1))
            self.assertTrue(
                operation_lock.attempted_by("human-pause").wait(timeout=1)
            )
            self.assertFalse(pause_finished.is_set())
            status_release.set()
            snapshot_thread.join(timeout=2)
            pause_thread.join(timeout=2)

        self.assertFalse(snapshot_thread.is_alive())
        self.assertFalse(pause_thread.is_alive())
        self.assertEqual(snapshot_results[0]["page"]["text"], "Example Domain")
        self.assertEqual(
            pause_results[0]["browser_session"]["state"],
            broker.STATE_PAUSED_OBSERVED,
        )

    def test_stop_revokes_connect_token_created_by_inflight_request(self) -> None:
        remote_broker, _, _ = self.make_broker()
        session = remote_broker._ensure_session("mcp-a")
        lifecycle_lock = self.track_lifecycle_lock(remote_broker, "mcp-a")
        proxy_start_entered = threading.Event()
        proxy_start_release = threading.Event()
        connect_errors: list[BaseException] = []
        stop_errors: list[BaseException] = []
        stop_finished = threading.Event()
        original_start = FakeNoVNCProxy.start

        def blocking_start(proxy: FakeNoVNCProxy) -> None:
            proxy_start_entered.set()
            if not proxy_start_release.wait(timeout=2):
                raise RuntimeError("timed out waiting to release proxy start")
            original_start(proxy)

        def run_connect() -> None:
            try:
                remote_broker.connect_info("mcp-a")
            except BaseException as exc:  # noqa: BLE001 - surfaced after join.
                connect_errors.append(exc)

        def run_stop() -> None:
            try:
                remote_broker.stop("mcp-a")
            except BaseException as exc:  # noqa: BLE001 - surfaced after join.
                stop_errors.append(exc)
            finally:
                stop_finished.set()

        with patch.object(FakeNoVNCProxy, "start", blocking_start):
            connect_thread = threading.Thread(
                target=run_connect,
                name="blocked-connect",
            )
            connect_thread.start()
            self.assertTrue(proxy_start_entered.wait(timeout=1))
            stop_thread = threading.Thread(target=run_stop, name="stop")
            stop_thread.start()
            self.assertTrue(
                lifecycle_lock.attempted_by("stop").wait(timeout=1)
            )
            self.assertFalse(stop_finished.is_set())
            proxy_start_release.set()
            connect_thread.join(timeout=2)
            stop_thread.join(timeout=2)

        self.assertFalse(connect_thread.is_alive())
        self.assertFalse(stop_thread.is_alive())
        self.assertEqual(connect_errors, [])
        self.assertEqual(stop_errors, [])
        self.assertEqual(remote_broker._novnc_proxies, {})
        self.assertFalse(session.novnc_token_active)
        self.assertIsNone(session.novnc_token)
        self.assertTrue(FakeNoVNCProxy.stopped)
        with self.assertRaises(broker.BrowserBrokerError) as denied_connect:
            remote_broker.connect_info("mcp-a")
        self.assertEqual(denied_connect.exception.code, broker.ERROR_SESSION_NOT_FOUND)

    def test_transport_shutdown_waits_for_inflight_page_operation_and_retains_profile(self) -> None:
        controller = BlockingController("wait")
        remote_broker, _, _ = self.make_broker(controller=controller)
        session = remote_broker._ensure_session("mcp-a")
        operation_lock = self.track_operation_lock(remote_broker, "mcp-a")
        session_root = session.profile_dir.parent
        wait_errors: list[BaseException] = []
        shutdown_errors: list[BaseException] = []
        shutdown_finished = threading.Event()

        def run_wait() -> None:
            try:
                remote_broker.wait("mcp-a", text="still running")
            except BaseException as exc:  # noqa: BLE001 - surfaced after join.
                wait_errors.append(exc)

        def run_shutdown() -> None:
            try:
                remote_broker.shutdown_transport_session("mcp-a")
            except BaseException as exc:  # noqa: BLE001 - surfaced after join.
                shutdown_errors.append(exc)
            finally:
                shutdown_finished.set()

        wait_thread = threading.Thread(target=run_wait, name="blocked-wait")
        wait_thread.start()
        self.assertTrue(controller.operation_entered.wait(timeout=1))
        shutdown_thread = threading.Thread(
            target=run_shutdown,
            name="transport-shutdown",
        )
        shutdown_thread.start()
        self.assertTrue(
            operation_lock.attempted_by("transport-shutdown").wait(timeout=1)
        )
        self.assertFalse(shutdown_finished.is_set())
        self.assertTrue(session_root.exists())

        controller.operation_release.set()
        wait_thread.join(timeout=2)
        shutdown_thread.join(timeout=2)

        self.assertFalse(wait_thread.is_alive())
        self.assertFalse(shutdown_thread.is_alive())
        self.assertEqual(wait_errors, [])
        self.assertEqual(shutdown_errors, [])
        self.assertTrue(session_root.exists())
        self.assertEqual(remote_broker.whoami("mcp-a")["browser_session"]["state"], broker.STATE_STOPPED_RETAINED)

    def test_stop_reserves_session_key_against_concurrent_ensure_and_shutdown(self) -> None:
        controller = BlockingController("wait")
        remote_broker, launcher, _ = self.make_broker(controller=controller)
        session = remote_broker._ensure_session("mcp-a")
        lifecycle_lock = self.track_lifecycle_lock(remote_broker, "mcp-a")
        session_root = session.profile_dir.parent
        wait_thread = threading.Thread(
            target=lambda: remote_broker.wait("mcp-a", text="in flight"),
        )
        stop_results: list[dict[str, object]] = []
        ensure_results: list[dict[str, object]] = []
        shutdown_results: list[dict[str, object]] = []
        stop_thread = threading.Thread(
            target=lambda: stop_results.append(
                remote_broker.stop("mcp-a")
            ),
            name="stop-lifecycle",
        )
        def run_ensure() -> None:
            ensure_results.append(remote_broker.ensure("mcp-a"))

        def run_shutdown() -> None:
            shutdown_results.append(
                remote_broker.shutdown_transport_session("mcp-a")
            )

        ensure_thread = threading.Thread(target=run_ensure, name="ensure-lifecycle")
        shutdown_thread = threading.Thread(target=run_shutdown, name="shutdown-lifecycle")

        wait_thread.start()
        self.assertTrue(controller.operation_entered.wait(timeout=1))
        stop_thread.start()
        self.assertTrue(lifecycle_lock.acquired_by("stop-lifecycle").wait(timeout=1))
        ensure_thread.start()
        shutdown_thread.start()
        self.assertTrue(lifecycle_lock.attempted_by("ensure-lifecycle").wait(timeout=1))
        self.assertTrue(lifecycle_lock.attempted_by("shutdown-lifecycle").wait(timeout=1))

        self.assertEqual(ensure_results, [])
        self.assertEqual(shutdown_results, [])
        self.assertEqual(len(launcher.started), 1)
        self.assertTrue(session_root.exists())

        controller.operation_release.set()
        for thread in (wait_thread, stop_thread, ensure_thread, shutdown_thread):
            thread.join(timeout=3)

        self.assertTrue(all(not thread.is_alive() for thread in (
            wait_thread,
            stop_thread,
            ensure_thread,
            shutdown_thread,
        )))
        self.assertEqual(stop_results[0]["status"], "stopped_retained")
        self.assertEqual(shutdown_results[0]["status"], "stopped_retained")
        self.assertEqual(len(ensure_results), 1)
        self.assertLessEqual(len(launcher.started), 2)

    def test_stop_wins_over_waiting_connect_and_resume_without_reactivation(self) -> None:
        controller = BlockingController("wait")
        remote_broker, _, _ = self.make_broker(controller=controller)
        remote_broker._ensure_session("mcp-a")
        lifecycle_lock = self.track_lifecycle_lock(remote_broker, "mcp-a")
        remote_broker.pause_for_human("mcp-a")
        remote_broker.resume("mcp-a")
        wait_thread = threading.Thread(target=lambda: remote_broker.wait("mcp-a", text="hold"))
        stop_results: list[dict[str, object]] = []
        connect_errors: list[broker.BrowserBrokerError] = []
        resume_errors: list[broker.BrowserBrokerError] = []
        stop_thread = threading.Thread(
            target=lambda: stop_results.append(
                remote_broker.stop("mcp-a")
            ),
            name="stop-lifecycle",
        )

        def run_connect() -> None:
            try:
                remote_broker.connect_info("mcp-a")
            except broker.BrowserBrokerError as exc:
                connect_errors.append(exc)

        def run_resume() -> None:
            try:
                remote_broker.resume("mcp-a")
            except broker.BrowserBrokerError as exc:
                resume_errors.append(exc)

        wait_thread.start()
        self.assertTrue(controller.operation_entered.wait(timeout=1))
        stop_thread.start()
        self.assertTrue(lifecycle_lock.acquired_by("stop-lifecycle").wait(timeout=1))
        connect_thread = threading.Thread(target=run_connect, name="connect-lifecycle")
        resume_thread = threading.Thread(target=run_resume, name="resume-lifecycle")
        connect_thread.start()
        resume_thread.start()
        self.assertTrue(lifecycle_lock.attempted_by("connect-lifecycle").wait(timeout=1))
        self.assertTrue(lifecycle_lock.attempted_by("resume-lifecycle").wait(timeout=1))
        controller.operation_release.set()
        for thread in (wait_thread, stop_thread, connect_thread, resume_thread):
            thread.join(timeout=3)

        self.assertEqual(stop_results[0]["status"], "stopped_retained")
        self.assertEqual(len(connect_errors), 1)
        self.assertEqual(connect_errors[0].code, broker.ERROR_SESSION_NOT_FOUND)
        self.assertEqual(len(resume_errors), 1)
        self.assertEqual(resume_errors[0].code, broker.ERROR_SESSION_NOT_FOUND)
        retained = remote_broker.whoami("mcp-a")["browser_session"]
        self.assertEqual(retained["state"], broker.STATE_STOPPED_RETAINED)
        self.assertEqual(remote_broker._novnc_proxies, {})

    def test_startup_readiness_serializes_second_ensure(self) -> None:
        controller = BlockingController("ready")
        remote_broker, launcher, _ = self.make_broker(controller=controller)
        lifecycle_lock = self.track_lifecycle_lock(remote_broker, "mcp-a")
        first_results: list[dict[str, object]] = []
        second_results: list[dict[str, object]] = []
        second_started = threading.Event()
        first_thread = threading.Thread(
            target=lambda: first_results.append(remote_broker.ensure("mcp-a")),
            name="first-ensure-lifecycle",
        )
        def run_second() -> None:
            second_started.set()
            second_results.append(remote_broker.ensure("mcp-a"))

        second_thread = threading.Thread(target=run_second, name="second-ensure-lifecycle")

        first_thread.start()
        self.assertTrue(controller.operation_entered.wait(timeout=1))
        second_thread.start()
        self.assertTrue(second_started.wait(timeout=1))
        self.assertTrue(
            lifecycle_lock.attempted_by("second-ensure-lifecycle").wait(timeout=1)
        )

        self.assertEqual(first_results, [])
        self.assertEqual(second_results, [])
        self.assertEqual(len(launcher.started), 1)
        self.assertEqual(
            remote_broker.whoami("mcp-a")["browser_session"]["state"],
            broker.STATE_STARTING,
        )

        controller.operation_release.set()
        for thread in (first_thread, second_thread):
            thread.join(timeout=3)

        self.assertTrue(all(not thread.is_alive() for thread in (
            first_thread,
            second_thread,
        )))
        self.assertEqual(first_results[0]["status"], "ready")
        self.assertEqual(second_results[0]["status"], "ready")
        self.assertEqual(first_results[0]["browser_session"]["id"], second_results[0]["browser_session"]["id"])
        self.assertEqual(len(launcher.started), 1)

    def test_startup_readiness_blocks_stop_until_ready(self) -> None:
        controller = BlockingController("ready")
        remote_broker, launcher, _ = self.make_broker(controller=controller)
        lifecycle_lock = self.track_lifecycle_lock(remote_broker, "mcp-a")
        ensure_results: list[dict[str, object]] = []
        stop_results: list[dict[str, object]] = []
        stop_started = threading.Event()
        ensure_thread = threading.Thread(
            target=lambda: ensure_results.append(remote_broker.ensure("mcp-a")),
            name="startup-ensure-lifecycle",
        )
        def run_stop() -> None:
            stop_started.set()
            stop_results.append(
                remote_broker.stop("mcp-a")
            )

        stop_thread = threading.Thread(target=run_stop, name="startup-stop-lifecycle")

        ensure_thread.start()
        self.assertTrue(controller.operation_entered.wait(timeout=1))
        stop_thread.start()
        self.assertTrue(stop_started.wait(timeout=1))
        self.assertTrue(
            lifecycle_lock.attempted_by("startup-stop-lifecycle").wait(timeout=1)
        )

        self.assertEqual(ensure_results, [])
        self.assertEqual(stop_results, [])
        self.assertEqual(len(launcher.stopped), 0)

        controller.operation_release.set()
        ensure_thread.join(timeout=3)
        stop_thread.join(timeout=3)

        self.assertFalse(ensure_thread.is_alive())
        self.assertFalse(stop_thread.is_alive())
        self.assertEqual(ensure_results[0]["status"], "ready")
        self.assertEqual(stop_results[0]["status"], "stopped_retained")
        self.assertEqual(len(launcher.started), 1)
        self.assertEqual(len(launcher.stopped), 1)

    def test_failed_startup_keeps_ports_reserved_until_teardown_finishes(self) -> None:
        controller = FailingOnceReadyController()
        remote_broker, launcher, _ = self.make_broker(controller=controller)
        stop_entered = threading.Event()
        stop_release = threading.Event()
        first_errors: list[BaseException] = []
        original_stop = launcher.stop
        acquire_port = remote_broker._acquire_available_port_lease

        def blocking_stop(session: broker.BrowserSession) -> None:
            stop_entered.set()
            if not stop_release.wait(timeout=2):
                raise RuntimeError("timed out waiting to release failed-start teardown")
            original_stop(session)

        def deterministic_port(
            port_range: tuple[int, int],
            *,
            excluded_ports: set[int],
        ) -> tuple[int, socket.socket]:
            with patch.object(broker.random, "randint", side_effect=range(port_range[0], port_range[1] + 1)):
                return acquire_port(port_range, excluded_ports=excluded_ports)

        def run_first() -> None:
            try:
                remote_broker.ensure("mcp-a")
            except BaseException as exc:  # noqa: BLE001 - asserted after join.
                first_errors.append(exc)

        with (
            patch.object(launcher, "stop", blocking_stop),
            patch.object(remote_broker, "_acquire_available_port_lease", side_effect=deterministic_port) as allocate,
        ):
            first_thread = threading.Thread(target=run_first)
            first_thread.start()
            self.assertTrue(stop_entered.wait(timeout=1))
            second = remote_broker.ensure("mcp-b")
            first_ports = {
                launcher.started[0]["cdp_port"],
                launcher.started[0]["novnc_backend_port"],
            }
            second_ports = {
                launcher.started[1]["cdp_port"],
                launcher.started[1]["novnc_backend_port"],
            }
            self.assertTrue(first_ports.isdisjoint(second_ports))
            self.assertNotEqual(
                launcher.started[0]["novnc_backend_port"],
                launcher.started[1]["novnc_backend_port"],
            )
            self.assertEqual(second["status"], "ready")
            self.assertEqual(allocate.call_count, 6)
            stop_release.set()
            first_thread.join(timeout=3)

        self.assertFalse(first_thread.is_alive())
        self.assertEqual(len(first_errors), 1)
        self.assertFalse(remote_broker.whoami("mcp-a")["has_browser_session"])

    def test_pause_blocks_actions_but_allows_observation(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "browser_pause_for_human",
                    "arguments": {},
                },
            },
            mcp_session_key="mcp-a",
        )

        navigated = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "browser_navigate",
                    "arguments": {"url": "example.com"},
                },
            },
            mcp_session_key="mcp-a",
        )
        snapshot = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "browser_snapshot", "arguments": {}},
                },
                mcp_session_key="mcp-a",
            )
        )
        screenshot = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": "browser_screenshot", "arguments": {}},
                },
                mcp_session_key="mcp-a",
            )
        )

        self.assertEqual(tool_payload(navigated)["error_code"], "PAUSED")
        self.assertEqual(snapshot["page"]["title"], "Example")
        self.assertEqual(screenshot["screenshot"]["capture"], "viewport")
        self.assertEqual(len(controller.snapshots), 1)
        self.assertEqual(len(controller.screenshots), 1)

    def test_ensure_reuses_one_browser_session_for_one_mcp_session(self) -> None:
        remote_broker, launcher, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "browser_ensure", "arguments": {}},
        }

        first = tool_payload(server.handle(request, mcp_session_key="mcp-a"))
        second = tool_payload(server.handle(request, mcp_session_key="mcp-a"))

        self.assertEqual(len(launcher.started), 1)
        self.assertEqual(len(controller.ready), 1)
        self.assertEqual(first["browser_session"]["id"], second["browser_session"]["id"])

    def test_ensure_recreate_replaces_current_mcp_session_browser(self) -> None:
        remote_broker, launcher, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        first = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "browser_ensure", "arguments": {}},
                },
                mcp_session_key="mcp-a",
            )
        )
        first_profile_root = Path(str(launcher.started[0]["profile_dir"])).parent

        remote_broker.pause_for_human("mcp-a")
        recreated = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "browser_ensure",
                        "arguments": {
                            "recreate": True,
                        },
                    },
                },
                mcp_session_key="mcp-a",
            )
        )

        self.assertNotEqual(first["browser_session"]["id"], recreated["browser_session"]["id"])
        self.assertEqual(launcher.stopped, [launcher.started[0]["container_name"]])
        self.assertEqual(len(launcher.started), 2)
        self.assertEqual(len(controller.ready), 2)
        self.assertFalse(first_profile_root.exists())
        self.assertEqual(
            remote_broker.whoami("mcp-a")["browser_session"]["id"],
            recreated["browser_session"]["id"],
        )

    def test_different_mcp_sessions_get_different_browser_sessions(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "browser_ensure", "arguments": {}},
        }

        first = tool_payload(server.handle(request, mcp_session_key="mcp-a"))
        second = tool_payload(server.handle(request, mcp_session_key="mcp-b"))

        self.assertEqual(len(launcher.started), 2)
        self.assertNotEqual(first["browser_session"]["id"], second["browser_session"]["id"])

    def test_lifecycle_lock_registry_does_not_grow_for_one_shot_invalid_keys(self) -> None:
        remote_broker, _, _ = self.make_broker()

        for index in range(200):
            with self.assertRaises(broker.BrowserBrokerError):
                remote_broker.connect_info(
                    f"missing-{index}",
                )

        self.assertLessEqual(len(remote_broker._lifecycle_locks), 1)

    def test_other_session_cannot_reuse_ports_while_stop_is_in_progress(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        remote_broker.ensure("mcp-a")
        first_session = remote_broker._session_or_error("mcp-a")
        stop_entered = threading.Event()
        stop_release = threading.Event()
        original_stop = launcher.stop

        def blocking_stop(session: broker.BrowserSession) -> None:
            stop_entered.set()
            if not stop_release.wait(timeout=2):
                raise RuntimeError("timed out waiting to release launcher stop")
            original_stop(session)

        stop_results: list[dict[str, object]] = []
        stop_thread = threading.Thread(
            target=lambda: stop_results.append(
                remote_broker.stop("mcp-a")
            ),
        )
        with patch.object(launcher, "stop", blocking_stop):
            stop_thread.start()
            self.assertTrue(stop_entered.wait(timeout=1))
            remote_broker.ensure("mcp-b")
            second_session = remote_broker._session_or_error("mcp-b")
            first_ports = {
                launcher.started[0]["cdp_port"],
                launcher.started[0]["novnc_backend_port"],
            }
            second_ports = {
                launcher.started[1]["cdp_port"],
                launcher.started[1]["novnc_backend_port"],
            }
            self.assertTrue(first_ports.isdisjoint(second_ports))
            self.assertNotEqual(
                first_session.novnc_port,
                second_session.novnc_port,
            )
            stop_release.set()
            stop_thread.join(timeout=3)

        self.assertFalse(stop_thread.is_alive())
        self.assertEqual(stop_results[0]["status"], "stopped_retained")

    def test_template_create_seal_and_clone_lifecycle(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        created = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "browser_template_create",
                        "arguments": {
                            "template_name": "example-portal",
                            "description": "Example Portal signed-in profile",
                        },
                    },
                },
                mcp_session_key="mcp-a",
            )
        )
        profile_dir = Path(launcher.started[0]["profile_dir"])
        (profile_dir / "Default").mkdir(parents=True)
        (profile_dir / "Default" / "Preferences").write_text("prefs", encoding="utf-8")
        (profile_dir / "SingletonLock").write_text("lock", encoding="utf-8")

        sealed = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "browser_template_seal",
                        "arguments": {
                            "template_name": "example-portal",
                            "description": "Example Portal signed-in profile",
                        },
                    },
                },
                mcp_session_key="mcp-a",
            )
        )
        cloned = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "browser_ensure",
                        "arguments": {"template_name": "example-portal"},
                    },
                },
                mcp_session_key="mcp-b",
            )
        )

        template_profile = Path(sealed["template"]["profile_dir"])
        session_profile = Path(launcher.started[-1]["profile_dir"])
        self.assertEqual(created["status"], "editing")
        self.assertEqual(created["browser_session"]["template_role"], "template_editing")
        self.assertEqual(sealed["status"], "sealed")
        self.assertTrue(sealed["browser_session_stopped"])
        self.assertEqual(launcher.stopped, [launcher.started[0]["container_name"]])
        self.assertEqual(remote_broker.whoami("mcp-a")["has_browser_session"], False)
        self.assertEqual(sealed["template"]["name"], "example-portal")
        self.assertEqual(sealed["template"]["description"], "Example Portal signed-in profile")
        self.assertTrue((template_profile / "Default" / "Preferences").exists())
        self.assertFalse((template_profile / "SingletonLock").exists())
        self.assertEqual(cloned["browser_session"]["template_name"], "example-portal")
        self.assertEqual(
            cloned["browser_session"]["template_version_id"],
            sealed["template"]["version_id"],
        )
        self.assertTrue((session_profile / "Default" / "Preferences").exists())

    def test_template_seal_allows_only_the_current_editor_without_operator(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        remote_broker.create_template("owner", template_name="owned-template")
        denied = self.call_mcp_tool(
            server, mcp_session_key="other", name="browser_template_seal",
            arguments={"template_name": "owned-template"},
        )
        self.assertIn("error_code", denied)
        self.assertEqual(launcher.stopped, [])
        sealed = self.call_mcp_tool(
            server, mcp_session_key="owner", name="browser_template_seal",
            arguments={"template_name": "owned-template"},
        )
        self.assertEqual(sealed["status"], "sealed")
        self.assertEqual(len(launcher.stopped), 1)

    def test_template_seal_waits_for_inflight_page_operation(self) -> None:
        controller = BlockingController("wait")
        remote_broker, _, _ = self.make_broker(controller=controller)
        created = remote_broker.create_template(
            "mcp-a",
            template_name="serialized-template",
        )
        session = remote_broker._session_or_error("mcp-a")
        operation_lock = self.track_operation_lock(remote_broker, "mcp-a")
        (session.profile_dir / "Default").mkdir(parents=True)
        (session.profile_dir / "Default" / "Preferences").write_text(
            "prefs",
            encoding="utf-8",
        )
        wait_errors: list[BaseException] = []
        seal_errors: list[BaseException] = []
        sealed_results: list[dict[str, object]] = []
        seal_finished = threading.Event()

        def run_wait() -> None:
            try:
                remote_broker.wait("mcp-a", text="editing")
            except BaseException as exc:  # noqa: BLE001 - surfaced after join.
                wait_errors.append(exc)

        def run_seal() -> None:
            try:
                sealed_results.append(
                    remote_broker.seal_template(
                        "mcp-a",
                        template_name="serialized-template",
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - surfaced after join.
                seal_errors.append(exc)
            finally:
                seal_finished.set()

        wait_thread = threading.Thread(target=run_wait, name="blocked-template-wait")
        wait_thread.start()
        self.assertTrue(controller.operation_entered.wait(timeout=1))
        seal_thread = threading.Thread(target=run_seal, name="template-seal")
        seal_thread.start()
        self.assertTrue(
            operation_lock.attempted_by("template-seal").wait(timeout=1)
        )
        self.assertFalse(seal_finished.is_set())

        controller.operation_release.set()
        wait_thread.join(timeout=2)
        seal_thread.join(timeout=2)

        self.assertFalse(wait_thread.is_alive())
        self.assertFalse(seal_thread.is_alive())
        self.assertEqual(wait_errors, [])
        self.assertEqual(seal_errors, [])
        self.assertEqual(created["status"], "editing")
        self.assertEqual(sealed_results[0]["status"], "sealed")
        self.assertTrue(
            (Path(sealed_results[0]["template"]["profile_dir"]) / "Default" / "Preferences").exists()
        )
        self.assertFalse(remote_broker.whoami("mcp-a")["has_browser_session"])
        self.assertEqual(remote_broker._novnc_proxies, {})

    def test_template_editing_session_must_be_sealed_or_discarded_not_stopped(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        created = remote_broker.create_template(
            "mcp-a",
            template_name="stop-safe-template",
        )
        session = remote_broker._session_or_error("mcp-a")
        draft_metadata_path = session.profile_dir.parent / "metadata.json"

        with self.assertRaises(broker.BrowserBrokerError) as error:
            remote_broker.stop("mcp-a")

        self.assertEqual(error.exception.code, broker.ERROR_TEMPLATE_CONFLICT)
        self.assertEqual(launcher.stopped, [])
        self.assertEqual(
            json.loads(draft_metadata_path.read_text(encoding="utf-8"))["state"],
            broker.TEMPLATE_STATE_EDITING,
        )
        active_metadata = json.loads(
            (session.profile_dir.parent / "session-metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(active_metadata["state"], broker.STATE_RUNNING)
        sealed = remote_broker.seal_template(
            "mcp-a",
            template_name="stop-safe-template",
        )
        self.assertEqual(created["status"], "editing")
        self.assertEqual(sealed["status"], "sealed")

    def test_template_seal_metadata_failure_preserves_draft_for_retry(self) -> None:
        remote_broker, _, _ = self.make_broker()
        remote_broker.create_template(
            "mcp-a",
            template_name="retryable-seal",
        )
        session = remote_broker._session_or_error("mcp-a")
        draft_dir = session.profile_dir.parent
        (session.profile_dir / "Default").mkdir(parents=True)
        (session.profile_dir / "Default" / "Preferences").write_text(
            "prefs",
            encoding="utf-8",
        )

        with patch.object(
            remote_broker,
            "_write_template_metadata",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(OSError):
                remote_broker.seal_template(
                    "mcp-a",
                    template_name="retryable-seal",
                )

        self.assertTrue((draft_dir / "profile" / "Default" / "Preferences").exists())
        self.assertIs(remote_broker._sessions_by_mcp["mcp-a"], session)
        versions_root = remote_broker._template_dir("retryable-seal") / "versions"
        self.assertFalse(versions_root.exists() and any(versions_root.iterdir()))

        sealed = remote_broker.seal_template(
            "mcp-a",
            template_name="retryable-seal",
        )
        self.assertEqual(sealed["status"], "sealed")
        self.assertFalse(draft_dir.exists())
        self.assertFalse(remote_broker.whoami("mcp-a")["has_browser_session"])

    def test_template_seal_draft_cleanup_failure_reuses_committed_version(self) -> None:
        remote_broker, _, _ = self.make_broker()
        remote_broker.create_template(
            "mcp-a",
            template_name="committed-seal",
        )
        session = remote_broker._session_or_error("mcp-a")
        draft_dir = session.profile_dir.parent
        original_rmtree = broker.shutil.rmtree

        def fail_draft_cleanup(path: object, *args: object, **kwargs: object) -> None:
            if Path(path) == draft_dir:
                raise OSError("draft busy")
            original_rmtree(path, *args, **kwargs)

        with patch.object(broker.shutil, "rmtree", side_effect=fail_draft_cleanup):
            with self.assertRaises(OSError):
                remote_broker.seal_template(
                    "mcp-a",
                    template_name="committed-seal",
                )

        versions_root = remote_broker._template_dir("committed-seal") / "versions"
        committed_versions = [path for path in versions_root.iterdir() if path.is_dir()]
        self.assertEqual(len(committed_versions), 1)
        self.assertTrue(draft_dir.exists())
        self.assertIs(remote_broker._sessions_by_mcp["mcp-a"], session)

        sealed = remote_broker.seal_template(
            "mcp-a",
            template_name="committed-seal",
        )
        self.assertEqual(sealed["template"]["version_id"], committed_versions[0].name)
        self.assertEqual(len([path for path in versions_root.iterdir() if path.is_dir()]), 1)
        self.assertFalse(draft_dir.exists())

    def test_template_list_and_delete(self) -> None:
        remote_broker, _, _ = self.make_broker()
        template_dir = remote_broker.config.runtime_root / "templates" / "base"
        (template_dir / "profile").mkdir(parents=True)
        (template_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "name": "base",
                    "description": "Base profile",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        server = broker.MCPServer(remote_broker)

        listed = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "browser_template_list", "arguments": {}},
                },
                mcp_session_key="mcp-a",
            )
        )
        deleted = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "browser_template_delete",
                        "arguments": {"template_name": "base"},
                    },
                },
                mcp_session_key="mcp-a",
            )
        )

        self.assertEqual([item["name"] for item in listed["templates"]], ["base"])
        self.assertEqual(deleted["status"], "deleted")
        self.assertFalse(template_dir.exists())

    def test_template_create_overwrites_existing_profile(self) -> None:
        remote_broker, _, _ = self.make_broker()
        template_dir = remote_broker.config.runtime_root / "templates" / "base"
        (template_dir / "profile").mkdir(parents=True)
        old_file = template_dir / "profile" / "old-profile"
        old_file.write_text("replace this profile", encoding="utf-8")
        server = broker.MCPServer(remote_broker)
        created = self.call_mcp_tool(server, mcp_session_key="mcp-a", name="browser_template_create",
                                     arguments={"template_name": "base", "overwrite": True})
        self.assertEqual(created["status"], "editing")
        self.assertTrue(old_file.exists())
        sealed = self.call_mcp_tool(server, mcp_session_key="mcp-a", name="browser_template_seal",
                                   arguments={"template_name": "base"})
        self.assertEqual(sealed["status"], "sealed")
        self.assertFalse(old_file.exists())
        self.assertTrue(template_dir.exists())

    def test_template_refresh_creates_new_editing_draft_from_sealed_version(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        version_profile = (
            remote_broker.config.runtime_root
            / "templates"
            / "base"
            / "versions"
            / "v1"
            / "profile"
        )
        (version_profile / "Default").mkdir(parents=True)
        (version_profile / "Default" / "Preferences").write_text("prefs", encoding="utf-8")
        (version_profile.parent / "metadata.json").write_text(
            json.dumps(
                {
                    "name": "base",
                    "version_id": "v1",
                    "state": "sealed",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        server = broker.MCPServer(remote_broker)

        refreshed = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "browser_template_refresh",
                        "arguments": {"template_name": "base", "version_id": "v1"},
                    },
                },
                mcp_session_key="mcp-a",
            )
        )

        draft_profile = Path(launcher.started[0]["profile_dir"])
        self.assertEqual(refreshed["status"], "editing")
        self.assertEqual(refreshed["template"]["source_version_id"], "v1")
        self.assertTrue((draft_profile / "Default" / "Preferences").exists())

    def test_template_clone_source_allows_independent_mcp_sessions(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        version_profile = (
            remote_broker.config.runtime_root
            / "templates"
            / "base"
            / "versions"
            / "v1"
            / "profile"
        )
        (version_profile / "Default").mkdir(parents=True)
        (version_profile.parent / "metadata.json").write_text(
            json.dumps(
                {
                    "name": "base",
                    "version_id": "v1",
                    "state": "sealed",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        server = broker.MCPServer(remote_broker)
        ensure_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "browser_ensure",
                "arguments": {"template_name": "base"},
            },
        }

        first = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-a"))
        second = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-b"))
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "browser_stop", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )
        third = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-c"))

        self.assertEqual(first["browser_session"]["template_version_id"], "v1")
        self.assertEqual(second["browser_session"]["template_version_id"], "v1")
        self.assertEqual(third["browser_session"]["template_version_id"], "v1")
        self.assertNotEqual(first["browser_session"]["id"], second["browser_session"]["id"])
        self.assertNotEqual(first["browser_session"]["id"], third["browser_session"]["id"])
        self.assertEqual(len(launcher.started), 3)

    def test_idle_template_session_does_not_transfer_to_new_mcp_session(self) -> None:
        remote_broker, launcher, controller = self.make_broker()
        version_profile = (
            remote_broker.config.runtime_root
            / "templates"
            / "base"
            / "versions"
            / "v1"
            / "profile"
        )
        (version_profile / "Default").mkdir(parents=True)
        (version_profile.parent / "metadata.json").write_text(
            json.dumps(
                {
                    "name": "base",
                    "version_id": "v1",
                    "state": "sealed",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        server = broker.MCPServer(remote_broker)
        ensure_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "browser_ensure",
                "arguments": {"template_name": "base"},
            },
        }

        first = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-a"))
        session = remote_broker._sessions_by_mcp["mcp-a"]
        session.last_used_at = broker.utc_after(-3600)
        second = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-b"))

        self.assertNotEqual(second["browser_session"]["id"], first["browser_session"]["id"])
        self.assertEqual(second["browser_session"]["template_version_id"], "v1")
        self.assertEqual(len(launcher.started), 2)
        self.assertEqual(len(controller.ready), 2)
        self.assertTrue(remote_broker.whoami("mcp-a")["has_browser_session"])
        self.assertTrue(remote_broker.whoami("mcp-b")["has_browser_session"])

    def test_failed_template_clone_startup_does_not_block_next_clone(self) -> None:
        failing_controller = FailingOnceReadyController()
        remote_broker, launcher, controller = self.make_broker(controller=failing_controller)
        version_profile = (
            remote_broker.config.runtime_root
            / "templates"
            / "base"
            / "versions"
            / "v1"
            / "profile"
        )
        (version_profile / "Default").mkdir(parents=True)
        (version_profile.parent / "metadata.json").write_text(
            json.dumps(
                {
                    "name": "base",
                    "version_id": "v1",
                    "state": "sealed",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        server = broker.MCPServer(remote_broker)
        ensure_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "browser_ensure",
                "arguments": {"template_name": "base"},
            },
        }

        failed = server.handle(ensure_request, mcp_session_key="mcp-a")
        recovered = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-b"))

        self.assertEqual(failed["error"]["code"], -32603)
        self.assertEqual(len(controller.ready), 2)
        self.assertEqual(launcher.stopped, [launcher.started[0]["container_name"]])
        self.assertEqual(recovered["browser_session"]["template_version_id"], "v1")

    def test_resumed_retained_template_clone_does_not_block_independent_clone(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        version_profile = (
            remote_broker.config.runtime_root
            / "templates"
            / "base"
            / "versions"
            / "v1"
            / "profile"
        )
        (version_profile / "Default").mkdir(parents=True)
        (version_profile.parent / "metadata.json").write_text(
            json.dumps(
                {
                    "name": "base",
                    "version_id": "v1",
                    "state": "sealed",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        server = broker.MCPServer(remote_broker)
        ensure_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "browser_ensure",
                "arguments": {"template_name": "base"},
            },
        }

        first = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-a"))
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "browser_stop", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )
        resumed = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-a"))
        second = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-b"))

        self.assertEqual(first["browser_session"]["id"], resumed["browser_session"]["id"])
        self.assertEqual(resumed["browser_session"]["template_version_id"], "v1")
        self.assertEqual(second["browser_session"]["template_version_id"], "v1")
        self.assertNotEqual(resumed["browser_session"]["id"], second["browser_session"]["id"])
        self.assertEqual(len(launcher.started), 3)

    def test_ensure_can_start_from_template(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        template_profile = remote_broker.config.runtime_root / "templates" / "base" / "profile"
        (template_profile / "Default").mkdir(parents=True)
        (template_profile / "Default" / "Preferences").write_text("prefs", encoding="utf-8")
        server = broker.MCPServer(remote_broker)

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "browser_ensure",
                    "arguments": {"template_name": "base"},
                },
            },
            mcp_session_key="mcp-a",
        )

        payload = tool_payload(response)
        session_profile = Path(launcher.started[0]["profile_dir"])
        self.assertEqual(payload["browser_session"]["template_name"], "base")
        self.assertTrue((session_profile / "Default" / "Preferences").exists())

    def test_template_name_rejects_path_traversal(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "browser_template_delete",
                    "arguments": {"template_name": "../secret"},
                },
            },
            mcp_session_key="operator",
        )

        result = response["result"]
        self.assertTrue(result["isError"])
        payload = tool_payload(response)
        self.assertEqual(payload["error_code"], "INVALID_ARGUMENT")
        self.assertIn("template_name", payload["error"])

    def test_docker_launcher_runs_container_with_host_user_and_writable_home(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        config = broker.BrokerConfig(
            workspace_root=Path(tmpdir.name),
            runtime_root=Path(tmpdir.name) / ".tokimon" / "remote-browser",
            installation_id="joey-test",
        )
        launcher = broker.DockerSessionLauncher(config)
        launcher.ensure_image = lambda: None  # type: ignore[method-assign]
        captured: list[list[str]] = []
        original_run = broker.subprocess.run
        network_exists = False

        class Result:
            returncode = 0
            stdout = "a" * 64
            stderr = ""

        def fake_run(cmd: list[str], **_: object) -> Result:
            nonlocal network_exists
            captured.append(cmd)
            if cmd[:3] == ["docker", "network", "inspect"]:
                if network_exists:
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        owned_network_inspect_payload(launcher, "session-a"),
                        "",
                    )
                return subprocess.CompletedProcess(
                    cmd,
                    1,
                    "",
                    "Error: No such network",
                )
            if cmd[:3] == ["docker", "network", "create"]:
                network_exists = True
                return Result()
            cidfile_index = cmd.index("--cidfile")
            Path(cmd[cidfile_index + 1]).write_text("a" * 64, encoding="utf-8")
            return Result()

        broker.subprocess.run = fake_run  # type: ignore[assignment]
        try:
            container_name = launcher.start(
                session_id="session-a",
                profile_dir=Path(tmpdir.name) / "profile",
                cdp_port=31001,
                novnc_backend_port=41001,
            )
        finally:
            broker.subprocess.run = original_run  # type: ignore[assignment]

        command = next(item for item in captured if item[:2] == ["docker", "run"])
        network_create = next(
            item for item in captured if item[:3] == ["docker", "network", "create"]
        )
        self.assertEqual(container_name, "tokimon-rb-joey-test-session-a")
        expected_network_name = launcher._network_name("session-a")
        self.assertRegex(expected_network_name, r"^tokimon-rbn-[0-9a-f]{20}$")
        self.assertEqual(network_create[-1], expected_network_name)
        for label in (
            "tokimon.remote-browser-broker=true",
            "tokimon.remote-browser-installation=joey-test",
            f"tokimon.remote-browser-owner-user={config.owner_user}",
            f"tokimon.remote-browser-checkout={config.workspace_root}",
            f"tokimon.remote-browser-runtime-root={config.runtime_root}",
            "tokimon.remote-browser-session=session-a",
            f"tokimon.remote-browser-network-policy={broker.DEFAULT_NETWORK_POLICY}",
        ):
            self.assertIn(label, network_create)
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            user_index = command.index("--user")
            self.assertEqual(command[user_index + 1], f"{os.getuid()}:{os.getgid()}")
        self.assertIn("HOME=/tmp", command)
        self.assertIn("--security-opt", command)
        self.assertIn("--cidfile", command)
        seccomp_index = command.index("--security-opt")
        seccomp_arg = command[seccomp_index + 1]
        self.assertRegex(seccomp_arg, r"^seccomp=.*tokimon-rb-seccomp-.*\.json$")
        self.assertIn(str(Path(tmpdir.name)), seccomp_arg)
        self.assertNotEqual(seccomp_arg, f"seccomp={broker.SECCOMP_PROFILE_PATH}")
        self.assertIn("--cap-drop", command)
        self.assertIn("ALL", command)
        self.assertIn("--cap-add", command)
        self.assertIn("SYS_CHROOT", command)
        self.assertIn("--memory", command)
        self.assertIn("--pids-limit", command)
        self.assertIn("--cpus", command)
        network_index = command.index("--network")
        self.assertEqual(command[network_index + 1], expected_network_name)
        self.assertIn("--tmpfs", command)
        self.assertIn(
            f"tokimon.remote-browser-network-policy={broker.DEFAULT_NETWORK_POLICY}",
            command,
        )
        self.assertIn("tokimon.remote-browser-installation=joey-test", command)
        self.assertIn(f"tokimon.remote-browser-owner-user={config.owner_user}", command)
        self.assertIn(f"tokimon.remote-browser-checkout={config.workspace_root}", command)
        self.assertIn(f"tokimon.remote-browser-runtime-root={config.runtime_root}", command)
        self.assertIn(f"TOKIMON_RB_NETWORK_POLICY={broker.DEFAULT_NETWORK_POLICY}", command)
        self.assertEqual(command[-1], "tokimon-remote-browser-session:joey-test")
        self.assertNotIn("--privileged", command)
        self.assertNotIn("/var/run/docker.sock", " ".join(command))
        for index, item in enumerate(command):
            if item == "-v":
                source, target = command[index + 1].split(":", 1)
                self.assertNotEqual(Path(source), ROOT)
                self.assertEqual(target, "/profile")
                self.assertEqual(Path(source), Path(tmpdir.name) / "profile")

    def test_docker_launcher_build_image_uses_temporary_context(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        config = broker.BrokerConfig(
            workspace_root=Path(tmpdir.name),
            runtime_root=Path(tmpdir.name) / ".tokimon" / "remote-browser",
            installation_id="joey-test",
        )
        launcher = broker.DockerSessionLauncher(config)
        captured_contexts: list[Path] = []

        class Result:
            returncode = 0
            stdout = "built"
            stderr = ""

        def fake_run(cmd: list[str], **_: object) -> Result:
            context = Path(cmd[-1])
            captured_contexts.append(context)
            self.assertNotEqual(context, broker.IMAGE_CONTEXT_DIR)
            self.assertTrue(context.is_relative_to(Path(tmpdir.name)))
            self.assertTrue((context / "Dockerfile").exists())
            self.assertTrue((context / "start-session-browser.sh").exists())
            self.assertTrue((context / "seccomp_profile.json").exists())
            return Result()

        with patch.object(broker.shutil, "which", return_value="/usr/bin/docker"):
            with patch.object(broker.subprocess, "run", fake_run):
                launcher.build_image()

        self.assertEqual(len(captured_contexts), 1)
        self.assertFalse(captured_contexts[0].exists())

    def test_browser_image_prefers_chrome_stable_and_keeps_sandbox_enabled(self) -> None:
        dockerfile = (broker.IMAGE_CONTEXT_DIR / "Dockerfile").read_text(encoding="utf-8")
        start_script = (broker.IMAGE_CONTEXT_DIR / "start-session-browser.sh").read_text(encoding="utf-8")

        self.assertRegex(
            dockerfile.splitlines()[0],
            r"^FROM mcr\.microsoft\.com/playwright/mcp:latest@sha256:[0-9a-f]{64}$",
        )
        self.assertIn("google-chrome-stable", dockerfile)
        self.assertIn("https://dl.google.com/linux/chrome/deb/", dockerfile)
        self.assertNotIn("http://dl.google.com/", dockerfile)
        self.assertIn("/usr/bin/google-chrome-stable", start_script)
        self.assertIn("TOKIMON_RB_ALLOW_CHROMIUM_FALLBACK", start_script)
        self.assertIn("Google Chrome Stable is required by default", start_script)
        self.assertLess(
            start_script.index("/usr/bin/google-chrome-stable"),
            start_script.index("/ms-playwright/chromium-*/chrome-linux/chrome"),
        )
        self.assertLess(
            start_script.index('if [[ "$TOKIMON_RB_ALLOW_CHROMIUM_FALLBACK" == "1" ]]'),
            start_script.index("/ms-playwright/chromium-*/chrome-linux/chrome"),
        )
        self.assertNotIn("--no-sandbox", start_script)

        seccomp_profile = (broker.IMAGE_CONTEXT_DIR / "seccomp_profile.json").read_text(encoding="utf-8")
        self.assertIn('"clone"', seccomp_profile)
        self.assertIn('"setns"', seccomp_profile)
        self.assertIn('"unshare"', seccomp_profile)
        self.assertIn("--no-first-run", start_script)
        self.assertIn("--no-default-browser-check", start_script)

    def test_browser_controller_creates_page_target_when_chrome_starts_without_one(
        self,
    ) -> None:
        controller = broker.BrowserController()
        session = broker.BrowserSession(
            id="session-a",
            mcp_session_key="mcp-a",
            profile_dir=Path("/tmp/profile"),
            cdp_port=31001,
            novnc_port=41001,
            novnc_backend_port=41002,
            container_name="container-a",
            created_at=broker.utc_now(),
            last_used_at=broker.utc_now(),
        )
        created = False
        original_fetch_json = broker.fetch_json
        original_urlopen = broker.urllib.request.urlopen

        def fake_fetch_json(url: str, *, timeout: float) -> object:
            if url.endswith("/json/version"):
                return {"Browser": "Chrome/147"}
            if url.endswith("/json"):
                if created:
                    return [{"type": "page", "webSocketDebuggerUrl": "ws://page"}]
                return [{"type": "service_worker", "webSocketDebuggerUrl": "ws://worker"}]
            raise AssertionError(url)

        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"type":"page","webSocketDebuggerUrl":"ws://page"}'

        def fake_urlopen(request: object, *, timeout: float) -> Response:
            nonlocal created
            self.assertEqual(timeout, 2)
            self.assertEqual(request.get_method(), "PUT")  # type: ignore[attr-defined]
            self.assertIn("/json/new?about:blank", request.full_url)  # type: ignore[attr-defined]
            created = True
            return Response()

        broker.fetch_json = fake_fetch_json  # type: ignore[assignment]
        broker.urllib.request.urlopen = fake_urlopen  # type: ignore[assignment]
        try:
            controller.wait_until_ready(session, timeout_sec=2)
        finally:
            broker.fetch_json = original_fetch_json  # type: ignore[assignment]
            broker.urllib.request.urlopen = original_urlopen  # type: ignore[assignment]

        self.assertTrue(created)

    def test_tab_tools_use_production_controller_target_discovery(self) -> None:
        remote_broker, _, _ = self.make_broker()
        remote_broker.ensure("mcp-a")
        remote_broker.controller = broker.BrowserController()
        server = broker.MCPServer(remote_broker)
        targets = [
            {"type": "service_worker", "id": "worker"},
            {"type": "page", "id": "first", "url": "https://example.com/first"},
            {"type": "page", "id": "second", "url": "https://example.com/second"},
        ]
        with patch.object(broker, "fetch_json", return_value=targets), patch.object(
            broker.CDPConnection, "for_session"
        ) as connection:
            switched = self.call_mcp_tool(
                server, mcp_session_key="mcp-a", name="browser_switch_tab", arguments={"tab_id": "second"}
            )
            self.assertEqual(switched["tab"]["active_tab_id"], "second")
            closed = self.call_mcp_tool(
                server, mcp_session_key="mcp-a", name="browser_close_tab", arguments={}
            )
            self.assertEqual(closed["tab"]["closed_tab_id"], "second")
            self.assertEqual(closed["tab"]["active_tab_id"], "first")
            connection.return_value.__enter__.return_value.call.assert_called_once_with(
                "Target.closeTarget", {"targetId": "second"}
            )

    def test_browser_controller_screenshot_uses_page_capture(self) -> None:
        controller = broker.BrowserController()
        session = broker.BrowserSession(
            id="session-a",
            mcp_session_key="mcp-a",
            profile_dir=Path("/tmp/profile"),
            cdp_port=31001,
            novnc_port=41001,
            novnc_backend_port=41002,
            container_name="container-a",
            created_at=broker.utc_now(),
            last_used_at=broker.utc_now(),
        )
        calls: list[tuple[str, dict[str, object]]] = []

        class FakeCDP:
            def __enter__(self) -> "FakeCDP":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def call(
                self,
                method: str,
                params: broker.Mapping[str, object] | None = None,
            ) -> dict[str, object]:
                calls.append((method, dict(params or {})))
                if method == "Page.captureScreenshot":
                    return {"data": "encoded-image"}
                return {}

        def fake_for_session(
            cls: type[broker.CDPConnection],
            target_session: broker.BrowserSession,
            *,
            timeout: float = 10.0,
        ) -> FakeCDP:
            self.assertIs(cls, broker.CDPConnection)
            self.assertIs(target_session, session)
            self.assertEqual(timeout, 10.0)
            return FakeCDP()

        with patch.object(
            broker.CDPConnection,
            "for_session",
            classmethod(fake_for_session),
        ):
            result = controller.screenshot(session, image_format="jpeg", quality=80)

        self.assertEqual(result, "encoded-image")
        self.assertEqual(
            calls,
            [
                ("Page.enable", {}),
                ("Page.captureScreenshot", {"format": "jpeg", "quality": 80}),
            ],
        )

    def test_normalize_url_preserves_non_hierarchical_schemes(self) -> None:
        self.assertEqual(broker.normalize_url("data:text/html,hello"), "data:text/html,hello")
        self.assertEqual(broker.normalize_url("about:blank"), "about:blank")
        self.assertEqual(broker.normalize_url("example.com"), "https://example.com")

    def test_invalid_url_preserves_existing_session_during_recreation(self) -> None:
        remote_broker, launcher, controller = self.make_broker()
        remote_broker.ensure("mcp-a")
        session = remote_broker._session_or_error("mcp-a")
        marker = session.profile_dir / "existing-profile-data"
        marker.write_text("keep this session", encoding="utf-8")

        for url in (
            "https://example.com:not-a-port/",
            "http://localhost:65536/",
            "https://example.com:-1/",
            "https://[::1/",
            "https:///missing-host",
            "https://bad host/",
        ):
            with self.subTest(url=url):
                with self.assertRaises(broker.BrowserBrokerError) as error:
                    remote_broker.ensure("mcp-a", recreate=True, start_url=url)
                self.assertEqual(error.exception.code, broker.ERROR_INVALID_ARGUMENT)
                self.assertIs(remote_broker._session_or_error("mcp-a"), session)
                self.assertEqual(marker.read_text(encoding="utf-8"), "keep this session")
                self.assertEqual(launcher.stopped, [])
                self.assertEqual(len(launcher.started), 1)
                self.assertEqual(controller.navigations, [])


    def test_normalize_url_accepts_valid_explicit_ports(self) -> None:
        for url in ("https://example.com:443/", "http://127.0.0.1:8080/", "http://[::1]:8080/"):
            with self.subTest(url=url):
                self.assertEqual(broker.normalize_url(url), url)


    def test_navigate_uses_current_mcp_sessions_browser(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)

        payload = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_navigate",
            arguments={"url": "example.com"},
        )

        self.assertEqual(payload["status"], "navigated")
        self.assertEqual(payload["page"]["url"], "https://example.com")
        self.assertEqual(len(controller.navigations), 1)
        self.assertEqual(controller.navigations[0][0], payload["browser_session"]["id"])

    def test_task_page_operations_run_without_an_operator_channel(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        self.call_mcp_tool(server, mcp_session_key="mcp-a", name="browser_ensure")
        navigated = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_navigate",
            arguments={"url": "https://example.com/search"},
        )
        snapshot = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_snapshot",
        )
        ref = snapshot["page"]["elements"][1]["ref"]
        clicked = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_click",
            arguments={"ref": ref},
        )
        typed = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_type",
            arguments={"ref": ref, "text": "sample search"},
        )
        pressed = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_press_key",
            arguments={"key": "Enter"},
        )
        evaluated = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_evaluate",
            arguments={"expression": "document.title"},
        )
        self.assertEqual(navigated["status"], "navigated")
        self.assertEqual(clicked["status"], "clicked")
        self.assertEqual(typed["status"], "typed")
        self.assertEqual(pressed["status"], "pressed")
        self.assertEqual(evaluated["result"]["expression"], "document.title")
        self.assertEqual(controller.types[0][2], "sample search")
        self.assertEqual(controller.keys[0][1], "Enter")

    def test_ensure_recreate_with_start_url_keeps_other_sessions(self) -> None:
        remote_broker, launcher, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        first = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_ensure",
            arguments={"start_url": "https://example.com/start"},
        )
        other = self.call_mcp_tool(server, mcp_session_key="mcp-b", name="browser_ensure")
        recreated = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_ensure",
            arguments={"recreate": True, "start_url": "https://example.com/replacement"},
        )
        self.assertNotEqual(first["browser_session"]["id"], recreated["browser_session"]["id"])
        self.assertEqual(launcher.stopped, [launcher.started[0]["container_name"]])
        self.assertEqual(
            controller.navigations[-1],
            (recreated["browser_session"]["id"], "https://example.com/replacement"),
        )
        self.assertEqual(recreated["navigation"]["url"], "https://example.com/replacement")
        self.assertEqual(
            remote_broker.whoami("mcp-b")["browser_session"]["id"],
            other["browser_session"]["id"],
        )

    def test_snapshot_requires_existing_session(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_snapshot", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        result = response["result"]
        self.assertTrue(result["isError"])
        payload = tool_payload(response)
        self.assertEqual(payload["error_code"], "SESSION_NOT_FOUND")
        self.assertIn("browser_ensure", payload["error"])

    def test_snapshot_dispatches_to_current_session(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "browser_snapshot",
                    "arguments": {"max_text_chars": 100, "max_elements": 5},
                },
            },
            mcp_session_key="mcp-a",
        )

        payload = tool_payload(response)
        self.assertEqual(payload["page"]["title"], "Example")
        self.assertIn("ref", payload["page"]["elements"][0])
        self.assertEqual(controller.snapshots, [(payload["browser_session"]["id"], 100, 5)])

    def test_evaluate_dispatches_to_current_session(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        payload = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_evaluate",
            arguments={"expression": "({ok: true})", "await_promise": False},
        )

        self.assertEqual(payload["result"], {"ok": True, "expression": "({ok: true})"})
        self.assertEqual(
            controller.evaluations,
            [(payload["browser_session"]["id"], "({ok: true})", False)],
        )

    def test_screenshot_dispatches_to_current_session_and_writes_artifact(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "browser_screenshot",
                    "arguments": {"format": "jpeg"},
                },
            },
            mcp_session_key="mcp-a",
        )

        payload = tool_payload(response)
        screenshot = payload["screenshot"]
        screenshot_path = Path(screenshot["path"])
        if not screenshot_path.is_absolute():
            screenshot_path = remote_broker.config.workspace_root / screenshot_path
        self.assertEqual(
            controller.screenshots,
            [(payload["browser_session"]["id"], "jpeg", 80)],
        )
        self.assertEqual(screenshot["format"], "jpeg")
        self.assertEqual(screenshot["capture"], "viewport")
        self.assertEqual(screenshot["size_bytes"], len(b"fake screenshot"))
        self.assertEqual(screenshot_path.suffix, ".jpeg")
        self.assertEqual(screenshot_path.parent.name, "screenshots")
        self.assertEqual(screenshot_path.read_bytes(), b"fake screenshot")
        serialized = json.dumps(payload)
        self.assertNotIn("page", payload)
        self.assertNotIn("url", serialized)
        self.assertNotIn("title", serialized)
        self.assertNotIn("ZmFrZSBzY3JlZW5zaG90", serialized)

    def test_screenshot_requires_existing_session(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_screenshot", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        result = response["result"]
        self.assertTrue(result["isError"])
        payload = tool_payload(response)
        self.assertEqual(payload["error_code"], "SESSION_NOT_FOUND")
        self.assertIn("browser_ensure", payload["error"])

    def test_screenshot_rejects_deferred_or_unsupported_arguments(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        full_page_response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "browser_screenshot",
                    "arguments": {"full_page": True},
                },
            },
            mcp_session_key="mcp-a",
        )
        quality_response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "browser_screenshot",
                    "arguments": {"quality": 10},
                },
            },
            mcp_session_key="mcp-a",
        )
        format_response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "browser_screenshot",
                    "arguments": {"format": "webp"},
                },
            },
            mcp_session_key="mcp-a",
        )

        full_page_payload = tool_payload(full_page_response)
        self.assertEqual(full_page_payload["error_code"], "INVALID_ARGUMENT")
        self.assertIn("full_page", full_page_payload["error"])
        self.assertEqual(tool_payload(quality_response)["error_code"], "INVALID_ARGUMENT")
        self.assertEqual(tool_payload(format_response)["error_code"], "INVALID_ARGUMENT")
        self.assertEqual(controller.screenshots, [])

    def test_click_type_key_wait_use_current_snapshot_refs(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )
        snapshot = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "browser_snapshot", "arguments": {}},
                },
                mcp_session_key="mcp-a",
            )
        )
        link_ref = snapshot["page"]["elements"][0]["ref"]
        input_ref = snapshot["page"]["elements"][1]["ref"]

        clicked = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_click",
            arguments={"ref": link_ref},
        )
        typed = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_type",
            arguments={"ref": input_ref, "text": "powell", "clear": True},
        )
        pressed = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_press_key",
            arguments={"key": "Enter"},
        )
        waited = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {"name": "browser_wait", "arguments": {"seconds": 0}},
                },
                mcp_session_key="mcp-a",
            )
        )

        self.assertEqual(clicked["status"], "clicked")
        self.assertEqual(typed["status"], "typed")
        self.assertEqual(pressed["status"], "pressed")
        self.assertEqual(waited["status"], "waited")
        self.assertEqual(controller.clicks[0][1], link_ref)
        self.assertEqual(controller.types[0][1:], (input_ref, "powell", True))
        self.assertEqual(controller.keys[0][1], "Enter")
        self.assertEqual(controller.waits[0][1], 0)

    def test_wait_rejects_unbounded_and_non_finite_durations(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        remote_broker.ensure("mcp-a")

        invalid_arguments = (
            {"seconds": broker.MAX_WAIT_SECONDS + 0.01},
            {"seconds": float("inf")},
            {"seconds": float("nan")},
            {"text": "never", "timeout_ms": broker.MAX_WAIT_TIMEOUT_MS + 1},
        )
        for index, arguments in enumerate(invalid_arguments, start=1):
            with self.subTest(arguments=arguments):
                response = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": index,
                        "method": "tools/call",
                        "params": {"name": "browser_wait", "arguments": arguments},
                    },
                    mcp_session_key="mcp-a",
                )
                self.assertEqual(tool_payload(response)["error_code"], "INVALID_ARGUMENT")

        wait_schema = next(
            item["inputSchema"]
            for item in broker.mcp_tools()
            if item["name"] == "browser_wait"
        )
        self.assertEqual(
            wait_schema["properties"]["seconds"]["maximum"],
            broker.MAX_WAIT_SECONDS,
        )
        self.assertEqual(
            wait_schema["properties"]["timeout_ms"]["maximum"],
            broker.MAX_WAIT_TIMEOUT_MS,
        )
        self.assertEqual(controller.waits, [])

    def test_upload_file_stages_file_and_uses_current_snapshot_ref(self) -> None:
        remote_broker, launcher, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        source_dir = tempfile.TemporaryDirectory()
        self.addCleanup(source_dir.cleanup)
        source = Path(source_dir.name) / "sample log.md"
        source.write_text("debate log", encoding="utf-8")
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )
        snapshot = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "browser_snapshot", "arguments": {}},
                },
                mcp_session_key="mcp-a",
            )
        )
        file_ref = snapshot["page"]["elements"][2]["ref"]

        uploaded = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_upload_file",
            arguments={"ref": file_ref, "path": str(source)},
        )

        self.assertEqual(uploaded["status"], "uploaded")
        self.assertEqual(uploaded["file_name"], "sample log.md")
        self.assertNotEqual(uploaded["upload_file_name"], "sample log.md")
        self.assertTrue(uploaded["upload_file_name"].endswith("-sample log.md"))
        self.assertEqual(uploaded["file_size_bytes"], len("debate log"))
        self.assertEqual(controller.uploads[0][1], file_ref)
        container_path = controller.uploads[0][2]
        self.assertTrue(container_path.startswith("/profile/uploads/"))
        self.assertTrue(container_path.endswith(f"/{uploaded['upload_file_name']}"))
        profile_dir = launcher.started[0]["profile_dir"]
        self.assertIsInstance(profile_dir, Path)
        staged = profile_dir / container_path.removeprefix("/profile/")
        self.assertEqual(staged.read_text(encoding="utf-8"), "debate log")

    def test_upload_file_rejects_missing_source_file(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )
        snapshot = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "browser_snapshot", "arguments": {}},
                },
                mcp_session_key="mcp-a",
            )
        )
        file_ref = snapshot["page"]["elements"][2]["ref"]

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "browser_upload_file",
                    "arguments": {"ref": file_ref, "path": "/missing/source.md"},
                },
            },
            mcp_session_key="mcp-a",
        )

        result = response["result"]
        self.assertTrue(result["isError"])
        self.assertEqual(tool_payload(response)["error_code"], "INVALID_ARGUMENT")
        self.assertEqual(controller.uploads, [])

    def test_backend_request_uses_current_session_and_same_origin_path(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        response = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_backend_request",
            arguments={
                "method": "patch",
                "path": "/api/drafts/draft-1",
                "body": {"archived": True},
                "max_response_chars": 2048,
                "timeout_ms": 45000,
            },
        )

        self.assertEqual(response["status"], "requested")
        self.assertEqual(response["method"], "PATCH")
        self.assertEqual(response["path"], "/api/drafts/draft-1")
        self.assertEqual(response["response"]["status"], 200)
        request = controller.backend_requests[0]
        self.assertEqual(request["method"], "PATCH")
        self.assertEqual(request["path"], "/api/drafts/draft-1")
        self.assertEqual(request["body"], {"archived": True})
        self.assertEqual(request["max_response_chars"], 2048)
        self.assertEqual(request["timeout_ms"], 45000)

    def test_backend_request_rejects_cross_origin_url(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "browser_backend_request",
                    "arguments": {
                        "method": "GET",
                        "path": "https://example.com/api/drafts/draft-1",
                    },
                },
            },
            mcp_session_key="mcp-a",
        )

        result = response["result"]
        self.assertTrue(result["isError"])
        self.assertEqual(tool_payload(response)["error_code"], "INVALID_ARGUMENT")
        self.assertEqual(controller.backend_requests, [])

    def test_backend_request_rejects_removed_authentication_options(self) -> None:
        remote_broker, _, controller = self.make_broker()
        remote_broker.ensure("mcp-a")
        server = broker.MCPServer(remote_broker)
        for key, value in (("include_auth_session_bearer", True),
                           ("auth_session_path", "/auth"), ("auth_token_field", "token")):
            with self.subTest(key=key):
                response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                          "params": {"name": "browser_backend_request",
                                                     "arguments": {"method": "GET", "path": "/draft", key: value}}},
                                         mcp_session_key="mcp-a")
                self.assertTrue(response["result"]["isError"])
                self.assertEqual(tool_payload(response)["error_code"], broker.ERROR_INVALID_ARGUMENT)
        self.assertEqual(controller.backend_requests, [])

    def test_navigation_invalidates_snapshot_refs(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )
        snapshot = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "browser_snapshot", "arguments": {}},
                },
                mcp_session_key="mcp-a",
            )
        )
        ref = snapshot["page"]["elements"][0]["ref"]
        self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_navigate",
            arguments={"url": "example.com"},
        )

        clicked = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "browser_click", "arguments": {"ref": ref}},
            },
            mcp_session_key="mcp-a",
        )

        self.assertEqual(tool_payload(clicked)["error_code"], "STALE_REF")

    def test_tab_tools_return_deterministic_active_tab(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        new_tab = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_new_tab",
            arguments={"url": "https://example.com"},
        )
        switched = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_switch_tab",
            arguments={"tab_id": new_tab["tab"]["tab_id"]},
        )
        closed = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_close_tab",
            arguments={},
        )

        self.assertEqual(new_tab["tab"]["active_tab_id"], new_tab["tab"]["tab_id"])
        self.assertEqual(switched["tab"]["active_tab_id"], new_tab["tab"]["tab_id"])
        self.assertEqual(closed["tab"]["active_tab_id"], "tab-next")
        self.assertEqual([item[1] for item in controller.tabs], ["new", "switch", "close"])


    def test_stop_only_stops_current_session(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        ensure_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "browser_ensure", "arguments": {}},
        }
        first = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-a"))
        second = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-b"))

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "browser_stop", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        payload = tool_payload(response)
        self.assertEqual(payload["status"], "stopped_retained")
        self.assertEqual(payload["browser_session"]["state"], "stopped_retained")
        self.assertEqual(launcher.stopped, [launcher.started[0]["container_name"]])
        self.assertEqual(remote_broker.whoami("mcp-a")["browser_session"]["state"], "stopped_retained")
        self.assertEqual(remote_broker.whoami("mcp-b")["browser_session"]["id"], second["browser_session"]["id"])

    def test_current_session_handoff_without_operator_keeps_other_session(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        remote_broker.ensure("mcp-a")
        other = remote_broker.ensure("mcp-b")["browser_session"]["id"]
        remote_broker.pause_for_human("mcp-a")
        remote_broker.pause_for_human("mcp-b")

        connect = self.call_mcp_tool(
            server, mcp_session_key="mcp-a", name="browser_connect_info"
        )
        resumed = self.call_mcp_tool(
            server, mcp_session_key="mcp-a", name="browser_resume"
        )
        stopped = self.call_mcp_tool(
            server, mcp_session_key="mcp-a", name="browser_stop"
        )

        self.assertIn("rb_token=", connect["connect"]["novnc_url"])
        self.assertEqual(resumed["browser_session"]["state"], broker.STATE_RUNNING)
        self.assertEqual(stopped["status"], "stopped_retained")
        self.assertEqual(launcher.stopped, [launcher.started[0]["container_name"]])
        other_after = remote_broker.whoami("mcp-b")["browser_session"]
        self.assertEqual(other_after["id"], other)
        self.assertEqual(other_after["state"], broker.STATE_PAUSED_OBSERVED)

    def test_stop_retains_profile_until_explicit_deletion(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )
        profile_dir = Path(launcher.started[0]["profile_dir"])
        (profile_dir / "Default").mkdir(parents=True)
        (profile_dir / "Default" / "Preferences").write_text("prefs", encoding="utf-8")

        stopped = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "browser_stop", "arguments": {}},
                },
                mcp_session_key="mcp-a",
            )
        )
        metadata_path = profile_dir.parent / "metadata.json"
        self.assertTrue(metadata_path.exists())
        deleted = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "browser_session_delete_retained",
                        "arguments": {},
                    },
                },
                mcp_session_key="mcp-a",
            )
        )

        self.assertEqual(stopped["browser_session"]["state"], "stopped_retained")
        self.assertEqual(deleted["status"], "deleted")
        self.assertFalse(profile_dir.parent.exists())

    def test_retained_restore_uses_new_session_generation(self) -> None:
        remote_broker, _, _ = self.make_broker()
        first = remote_broker.ensure("mcp-a")
        old_session = remote_broker._session_or_error("mcp-a")

        remote_broker.stop("mcp-a")
        restored = remote_broker.ensure("mcp-a")
        new_session = remote_broker._session_or_error("mcp-a")

        self.assertIsNot(old_session, new_session)
        self.assertEqual(old_session.state, broker.STATE_STOPPED_RETAINED)
        self.assertEqual(new_session.state, broker.STATE_RUNNING)
        self.assertEqual(first["browser_session"]["id"], restored["browser_session"]["id"])

    def test_failed_retained_restart_tears_down_before_fallible_metadata_commit(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        remote_broker.ensure("mcp-a")
        remote_broker.stop("mcp-a")
        failing_controller = FailingOnceReadyController()
        remote_broker.controller = failing_controller  # type: ignore[assignment]

        with patch.object(
            remote_broker,
            "_write_session_metadata",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(OSError):
                remote_broker.ensure("mcp-a")

        self.assertEqual(len(launcher.started), 1)
        self.assertEqual(len(launcher.stopped), 1)
        self.assertNotIn("mcp-a", remote_broker._sessions_by_mcp)
        self.assertEqual(
            remote_broker._retained_by_mcp["mcp-a"].state,
            broker.STATE_STOPPED_RETAINED,
        )

        remote_broker.controller = FakeController()  # type: ignore[assignment]
        recovered = remote_broker.ensure("mcp-a")
        self.assertEqual(recovered["status"], "ready")
        self.assertEqual(len(launcher.started), 2)
        self.assertEqual(len(launcher.stopped), 1)
        self.assertEqual(
            remote_broker._session_or_error("mcp-a").state,
            broker.STATE_RUNNING,
        )

    def test_verified_retained_restart_failure_restores_handoff_metadata(self) -> None:
        first, launcher, _ = self.make_broker()
        original = first.ensure("handoff-retry")
        first.stop("handoff-retry")

        with patch.object(
            launcher,
            "start",
            side_effect=broker.BrowserBrokerError(
                "start failed",
                details={"cleanup_verified": True},
            ),
        ):
            with self.assertRaises(broker.BrowserBrokerError):
                first.ensure("handoff-retry")

        retained = first._retained_by_mcp["handoff-retry"]
        checkpoint = json.loads(
            (retained.profile_dir.parent / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(checkpoint["state"], broker.STATE_STOPPED_RETAINED)
        self.assertIsNotNone(checkpoint["retention_expires_at"])
        first.begin_shutdown()
        first.release_shutdown_leases()

        peer_launcher = FakeLauncher()
        peer = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=peer_launcher,  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                peer.begin_shutdown(),
                peer.release_shutdown_leases(),
            )
        )
        restored = peer.ensure("handoff-retry")
        self.assertEqual(
            restored["browser_session"]["id"],
            original["browser_session"]["id"],
        )

    def test_stop_metadata_failure_keeps_active_ownership_for_retry(self) -> None:
        remote_broker, _, _ = self.make_broker()
        remote_broker.ensure("mcp-a")
        session = remote_broker._session_or_error("mcp-a")
        metadata_path = session.profile_dir.parent / "metadata.json"

        with patch.object(
            remote_broker,
            "_write_session_metadata",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(OSError):
                remote_broker.stop("mcp-a")

        self.assertIs(remote_broker._sessions_by_mcp["mcp-a"], session)
        self.assertNotIn("mcp-a", remote_broker._retained_by_mcp)
        self.assertEqual(session.state, broker.STATE_STOPPED_RETAINED)
        self.assertEqual(
            json.loads(metadata_path.read_text(encoding="utf-8"))["state"],
            broker.STATE_RUNNING,
        )
        with self.assertRaises(broker.BrowserBrokerError) as pause_error:
            remote_broker.pause_for_human("mcp-a")
        with self.assertRaises(broker.BrowserBrokerError) as resume_error:
            remote_broker.resume("mcp-a")
        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker.connect_info("mcp-a")
        self.assertEqual(pause_error.exception.code, broker.ERROR_SESSION_STOPPED)
        self.assertEqual(resume_error.exception.code, broker.ERROR_SESSION_STOPPED)
        self.assertEqual(session.state, broker.STATE_STOPPED_RETAINED)
        self.assertFalse(session.novnc_token_active)

        retried = remote_broker.stop("mcp-a")
        self.assertEqual(retried["status"], "stopped_retained")
        self.assertNotIn("mcp-a", remote_broker._sessions_by_mcp)
        self.assertIs(remote_broker._retained_by_mcp["mcp-a"], session)
        self.assertTrue(metadata_path.exists())

    def test_template_creation_rejects_existing_retained_session_without_index_leak(self) -> None:
        remote_broker, _, _ = self.make_broker()
        remote_broker.ensure("mcp-a")
        retained = remote_broker._session_or_error("mcp-a")
        remote_broker.stop("mcp-a")

        with self.assertRaises(broker.BrowserBrokerError) as error:
            remote_broker.create_template("mcp-a", template_name="new-template")

        self.assertEqual(error.exception.code, broker.ERROR_TEMPLATE_CONFLICT)
        self.assertNotIn("mcp-a", remote_broker._sessions_by_mcp)
        self.assertIs(remote_broker._retained_by_mcp["mcp-a"], retained)
        self.assertEqual(remote_broker._retained_by_id, {retained.id: retained})

    def test_retained_delete_failure_preserves_indexes_metadata_and_retry(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        remote_broker.ensure("mcp-a")
        session = remote_broker._session_or_error("mcp-a")
        session_root = Path(launcher.started[0]["profile_dir"]).parent
        remote_broker.stop("mcp-a")

        with patch.object(broker.shutil, "rmtree", side_effect=OSError("busy")):
            with self.assertRaises(OSError):
                remote_broker.delete_retained_session("mcp-a")

        self.assertIs(remote_broker._retained_by_mcp["mcp-a"], session)
        self.assertIs(remote_broker._retained_by_id[session.id], session)
        self.assertTrue(session_root.exists())
        self.assertTrue((session_root / "metadata.json").exists())

        deleted = remote_broker.delete_retained_session("mcp-a")
        self.assertEqual(deleted["status"], "deleted")
        self.assertFalse(session_root.exists())

    def test_expired_cleanup_failure_preserves_retryable_retained_session(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        remote_broker.ensure("mcp-a")
        session = remote_broker._session_or_error("mcp-a")
        session_root = Path(launcher.started[0]["profile_dir"]).parent
        remote_broker.stop("mcp-a")
        session.retention_expires_at = "2000-01-01T00:00:00+00:00"

        with patch.object(broker.shutil, "rmtree", side_effect=OSError("busy")):
            with self.assertRaises(OSError):
                remote_broker.cleanup_expired_sessions()

        self.assertEqual(session.state, broker.STATE_EXPIRED)
        self.assertIs(remote_broker._retained_by_mcp["mcp-a"], session)
        self.assertIs(remote_broker._retained_by_id[session.id], session)
        self.assertTrue((session_root / "metadata.json").exists())

        cleaned = remote_broker.cleanup_expired_sessions()
        self.assertEqual(cleaned["cleaned_count"], 1)
        self.assertFalse(session_root.exists())

    def test_extend_retained_session_updates_expiry_and_respects_max(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "browser_ensure", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "browser_stop", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )

        extended = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "browser_session_extend_retention",
                        "arguments": {"ttl_seconds": 60},
                    },
                },
                mcp_session_key="mcp-a",
            )
        )
        rejected = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "browser_session_extend_retention",
                    "arguments": {"ttl_seconds": remote_broker.config.max_retention_seconds + 1},
                },
            },
            mcp_session_key="mcp-a",
        )

        self.assertEqual(extended["status"], "extended")
        self.assertIsNotNone(extended["browser_session"]["retention_expires_at"])
        self.assertEqual(tool_payload(rejected)["error_code"], "INVALID_ARGUMENT")

    def test_cleanup_expired_retained_sessions_skips_running_sessions(self) -> None:
        remote_broker, _, _ = self.make_broker()
        server = broker.MCPServer(
            remote_broker,
        )
        ensure_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "browser_ensure", "arguments": {}},
        }
        server.handle(ensure_request, mcp_session_key="mcp-a")
        running = tool_payload(server.handle(ensure_request, mcp_session_key="mcp-b"))
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "browser_stop", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )
        retained = remote_broker._retained_by_mcp["mcp-a"]
        retained.retention_expires_at = "2000-01-01T00:00:00+00:00"
        remote_broker._write_session_metadata(retained)

        cleaned = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "browser_session_cleanup_expired", "arguments": {}},
                },
                mcp_session_key="operator",
            )
        )

        self.assertEqual(cleaned["cleaned_count"], 1)
        self.assertEqual(remote_broker.whoami("mcp-a")["has_browser_session"], False)
        self.assertEqual(
            remote_broker.whoami("mcp-b")["browser_session"]["id"],
            running["browser_session"]["id"],
        )

    def test_reconciliation_loads_and_restarts_retained_session(self) -> None:
        remote_broker, launcher, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        ensured = tool_payload(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "browser_ensure", "arguments": {}},
                },
                mcp_session_key="mcp-a",
            )
        )
        profile_dir = Path(launcher.started[0]["profile_dir"])
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "browser_stop", "arguments": {}},
            },
            mcp_session_key="mcp-a",
        )
        remote_broker.begin_shutdown()
        remote_broker.release_shutdown_leases()
        new_launcher = FakeLauncher()
        new_controller = FakeController()
        FakeNoVNCProxy.started = []
        reconciled = broker.RemoteBrowserBroker(
            config=remote_broker.config,
            launcher=new_launcher,  # type: ignore[arg-type]
            controller=new_controller,  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                reconciled.begin_shutdown(),
                reconciled.release_shutdown_leases(),
            )
        )

        status = reconciled.whoami("mcp-a")
        restarted = reconciled.ensure("mcp-a")

        self.assertEqual(status["browser_session"]["state"], "stopped_retained")
        self.assertEqual(restarted["browser_session"]["id"], ensured["browser_session"]["id"])
        self.assertEqual(new_launcher.started[0]["profile_dir"], profile_dir)
        self.assertEqual(
            json.loads(
                (profile_dir.parent / "metadata.json").read_text(encoding="utf-8")
            )["state"],
            broker.STATE_RUNNING,
        )

    def test_peer_broker_cannot_cleanup_a_live_session_profile(self) -> None:
        first, _, _ = self.make_broker()
        running = first.ensure("mcp-a")
        profile_dir = first._session_or_error("mcp-a").profile_dir
        second = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                second.begin_shutdown(),
                second.release_shutdown_leases(),
            )
        )

        inventory = second.session_list()
        cleaned = second.cleanup_orphaned_sessions()

        self.assertEqual(running["browser_session"]["state"], "running")
        self.assertTrue(profile_dir.exists())
        self.assertEqual(cleaned["cleaned_count"], 0)
        self.assertEqual(cleaned["peer_owned_count"], 1)
        self.assertEqual(inventory["sessions"][0]["state"], "peer_owned")
        self.assertEqual(first.whoami("mcp-a")["browser_session"]["state"], "running")

    def test_peer_owned_index_clears_after_the_real_owner_deletes_root(self) -> None:
        first, _, _ = self.make_broker()
        first.ensure("mcp-a")
        second = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                second.begin_shutdown(),
                second.release_shutdown_leases(),
            )
        )
        self.assertEqual(
            second.session_list()["sessions"][0]["state"],
            "peer_owned",
        )

        first.stop("mcp-a")
        first.delete_retained_session("mcp-a")
        cleaned = second.cleanup_orphaned_sessions()

        self.assertEqual(cleaned["cleaned_count"], 1)
        self.assertEqual(cleaned["peer_owned_count"], 0)
        self.assertEqual(second.session_list()["session_count"], 0)

    def test_peer_broker_cannot_delete_or_overwrite_a_live_template_draft(self) -> None:
        first, _, _ = self.make_broker()
        second = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                second.begin_shutdown(),
                second.release_shutdown_leases(),
            )
        )
        first.create_template("mcp-a", template_name="shared")
        draft_root = first._session_or_error("mcp-a").profile_dir.parent

        with self.assertRaises(broker.BrowserBrokerError) as delete_error:
            second.delete_template(template_name="shared")
        with self.assertRaises(broker.BrowserBrokerError) as overwrite_error:
            second.create_template(
                "mcp-b",
                template_name="shared",
                overwrite=True,
            )

        self.assertEqual(delete_error.exception.code, broker.ERROR_TEMPLATE_CONFLICT)
        self.assertIn(
            overwrite_error.exception.code,
            {broker.ERROR_TEMPLATE_CONFLICT, broker.ERROR_TEMPLATE_EXISTS},
        )
        self.assertTrue(draft_root.exists())
        self.assertEqual(first.whoami("mcp-a")["browser_session"]["state"], "running")

    def test_stale_peer_template_index_clears_after_owner_seals(self) -> None:
        first, _, _ = self.make_broker()
        first.create_template("mcp-a", template_name="handover")
        second = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                second.begin_shutdown(),
                second.release_shutdown_leases(),
            )
        )
        self.assertEqual(second.session_list()["session_count"], 1)

        first.seal_template(
            "mcp-a",
            template_name="handover",
        )
        cleaned = second.cleanup_orphaned_sessions()
        refreshed = second.refresh_template("mcp-b", template_name="handover")

        self.assertEqual(cleaned["cleaned_count"], 1)
        self.assertEqual(refreshed["status"], "editing")

    def test_template_lock_order_does_not_deadlock_same_key_ensure_and_seal(self) -> None:
        remote_broker, _, _ = self.make_broker()
        remote_broker.create_template("mcp-a", template_name="ordered")
        ensure_waiting_for_template = threading.Event()
        allow_ensure_template = threading.Event()
        seal_holds_template = threading.Event()
        results: list[dict[str, object]] = []
        errors: list[BaseException] = []
        original_template_operation = remote_broker._template_operation

        @contextlib.contextmanager
        def tracked_template_operation(template_name: str) -> object:
            thread_name = threading.current_thread().name
            if thread_name == "ensure-thread":
                ensure_waiting_for_template.set()
                if not allow_ensure_template.wait(timeout=2):
                    raise RuntimeError("timed out waiting to acquire template lock")
            with original_template_operation(template_name):
                if thread_name == "seal-thread":
                    seal_holds_template.set()
                yield

        def ensure_template() -> None:
            try:
                results.append(
                    remote_broker.ensure("mcp-a", template_name="ordered")
                )
            except BaseException as exc:  # noqa: BLE001 - asserted below.
                errors.append(exc)

        def seal_template() -> None:
            try:
                results.append(
                    remote_broker.seal_template(
                        "mcp-a",
                        template_name="ordered",
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - asserted below.
                errors.append(exc)

        with patch.object(
            remote_broker,
            "_template_operation",
            tracked_template_operation,
        ):
            ensure_thread = threading.Thread(
                target=ensure_template,
                name="ensure-thread",
            )
            ensure_thread.start()
            self.assertTrue(ensure_waiting_for_template.wait(timeout=1))
            seal_thread = threading.Thread(
                target=seal_template,
                name="seal-thread",
            )
            seal_thread.start()
            self.assertTrue(seal_holds_template.wait(timeout=1))
            seal_thread.join(timeout=3)
            allow_ensure_template.set()
            ensure_thread.join(timeout=3)

        self.assertFalse(seal_thread.is_alive())
        self.assertFalse(ensure_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual({result["status"] for result in results}, {"sealed", "ready"})

    def test_shutdown_closes_admission_and_drains_recreate_before_snapshot(self) -> None:
        remote_broker, launcher, _ = self.make_broker()
        remote_broker.ensure("mcp-a")
        server = broker.MCPServer(remote_broker)
        discard_finished = threading.Event()
        release_recreate = threading.Event()
        ensure_results: list[dict[str, object]] = []
        shutdown_results: list[dict[str, object]] = []
        errors: list[BaseException] = []
        original_discard = remote_broker._discard_current_session

        def blocking_discard(mcp_session_key: str) -> None:
            original_discard(mcp_session_key)
            discard_finished.set()
            if not release_recreate.wait(timeout=2):
                raise RuntimeError("timed out waiting to resume recreate")

        def recreate() -> None:
            try:
                ensure_results.append(
                    remote_broker.ensure(
                        "mcp-a",
                        recreate=True,
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - asserted below.
                errors.append(exc)

        def shutdown() -> None:
            try:
                shutdown_results.append(remote_broker.shutdown_all_sessions())
            except BaseException as exc:  # noqa: BLE001 - asserted below.
                errors.append(exc)

        with patch.object(remote_broker, "_discard_current_session", blocking_discard):
            ensure_thread = threading.Thread(target=recreate)
            ensure_thread.start()
            self.assertTrue(discard_finished.wait(timeout=1))
            shutdown_thread = threading.Thread(target=shutdown)
            shutdown_thread.start()
            self.assertTrue(shutdown_thread.is_alive())
            release_recreate.set()
            ensure_thread.join(timeout=3)
            shutdown_thread.join(timeout=3)

        self.assertFalse(ensure_thread.is_alive())
        self.assertFalse(shutdown_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(ensure_results[0]["status"], "ready")
        self.assertEqual(shutdown_results[0]["stopped_session_count"], 1)
        self.assertEqual(len(launcher.started), 2)
        self.assertEqual(len(launcher.stopped), 2)
        self.assertEqual(
            remote_broker._retained_by_mcp["mcp-a"].state,
            "stopped_retained",
        )

    def test_ensure_response_holds_operation_barrier_through_status_construction(self) -> None:
        remote_broker, _, controller = self.make_broker()
        remote_broker.ensure("mcp-a")
        session = remote_broker._session_or_error("mcp-a")
        status_entered = threading.Event()
        release_status = threading.Event()
        ensure_results: list[dict[str, object]] = []
        snapshot_results: list[dict[str, object]] = []
        original_status = session.to_status_dict

        def blocking_status() -> dict[str, object]:
            status_entered.set()
            if not release_status.wait(timeout=2):
                raise RuntimeError("timed out waiting to finish status construction")
            return original_status()

        with patch.object(session, "to_status_dict", side_effect=blocking_status):
            ensure_thread = threading.Thread(
                target=lambda: ensure_results.append(remote_broker.ensure("mcp-a"))
            )
            ensure_thread.start()
            self.assertTrue(status_entered.wait(timeout=1))
            snapshot_thread = threading.Thread(
                target=lambda: snapshot_results.append(remote_broker.snapshot("mcp-a"))
            )
            snapshot_thread.start()
            self.assertTrue(snapshot_thread.is_alive())
            self.assertEqual(controller.snapshots, [])
            release_status.set()
            ensure_thread.join(timeout=3)
            snapshot_thread.join(timeout=3)

        self.assertFalse(ensure_thread.is_alive())
        self.assertFalse(snapshot_thread.is_alive())
        self.assertEqual(ensure_results[0]["status"], "ready")
        self.assertEqual(snapshot_results[0]["page"]["title"], "Example")

    def test_template_editor_shutdown_reconciles_and_seals_after_restart(self) -> None:
        first, _, _ = self.make_broker()
        first.create_template("mcp-a", template_name="durable")
        draft_profile = first._session_or_error("mcp-a").profile_dir
        (draft_profile / "Default").mkdir(parents=True)
        (draft_profile / "Default" / "Cookies").write_text(
            "authenticated",
            encoding="utf-8",
        )

        stopped = first.shutdown_all_sessions()
        retained_metadata = draft_profile.parent / "session-metadata.json"
        self.assertEqual(stopped["stopped_session_count"], 1)
        self.assertTrue(retained_metadata.exists())
        first.release_shutdown_leases()

        second_launcher = FakeLauncher()
        second = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=second_launcher,  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                second.begin_shutdown(),
                second.release_shutdown_leases(),
            )
        )

        reconciled = second.whoami("mcp-a")
        sealed = second.seal_template(
            "mcp-a",
            template_name="durable",
        )
        sealed_profile = Path(sealed["template"]["profile_dir"])

        self.assertEqual(
            reconciled["browser_session"]["state"],
            "stopped_retained",
        )
        self.assertEqual(second_launcher.stopped, [])
        self.assertEqual(
            (sealed_profile / "Default" / "Cookies").read_text(encoding="utf-8"),
            "authenticated",
        )
        self.assertFalse(draft_profile.parent.exists())

    def test_failed_template_seal_persists_restartable_stopped_draft(self) -> None:
        first, _, _ = self.make_broker()
        first.create_template("mcp-a", template_name="checkpoint")
        session = first._session_or_error("mcp-a")

        with patch.object(first, "_copy_profile", side_effect=RuntimeError("copy failed")):
            with self.assertRaises(RuntimeError):
                first.seal_template(
                    "mcp-a",
                    template_name="checkpoint",
                )

        metadata_path = session.profile_dir.parent / "session-metadata.json"
        self.assertTrue(metadata_path.exists())
        self.assertEqual(session.state, "stopped_retained")
        first.begin_shutdown()
        first.release_shutdown_leases()

        second = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                second.begin_shutdown(),
                second.release_shutdown_leases(),
            )
        )

        self.assertEqual(
            second.whoami("mcp-a")["browser_session"]["state"],
            "stopped_retained",
        )
        self.assertEqual(
            second.seal_template(
                "mcp-a",
                template_name="checkpoint",
            )["status"],
            "sealed",
        )

    def test_template_overwrite_failure_preserves_previous_sealed_version(self) -> None:
        remote_broker, _, controller = self.make_broker()
        remote_broker.create_template("mcp-a", template_name="replaceable")
        profile = remote_broker._session_or_error("mcp-a").profile_dir
        (profile / "Default").mkdir(parents=True)
        (profile / "Default" / "Cookies").write_text("old", encoding="utf-8")
        sealed = remote_broker.seal_template(
            "mcp-a",
            template_name="replaceable",
        )
        old_profile = Path(sealed["template"]["profile_dir"])

        with patch.object(
            controller,
            "wait_until_ready",
            side_effect=RuntimeError("replacement not ready"),
        ):
            with self.assertRaises(RuntimeError):
                remote_broker.create_template(
                    "mcp-a",
                    template_name="replaceable",
                    overwrite=True,
                )

        listed = remote_broker.list_templates()
        self.assertTrue(old_profile.exists())
        self.assertEqual(
            (old_profile / "Default" / "Cookies").read_text(encoding="utf-8"),
            "old",
        )
        self.assertEqual(len(listed["templates"][0]["versions"]), 1)
        self.assertFalse(remote_broker.whoami("mcp-a")["has_browser_session"])

    def test_failed_staging_cleanup_is_immediately_quarantined(self) -> None:
        remote_broker, _, _ = self.make_broker()
        remote_broker.create_template("mcp-a", template_name="quarantine")
        original_rmtree = broker.shutil.rmtree

        def fail_staging_rmtree(path: object, *args: object, **kwargs: object) -> None:
            if "/staging/" in str(path):
                raise OSError("staging cleanup failed")
            original_rmtree(path, *args, **kwargs)

        with patch.object(remote_broker, "_copy_profile", side_effect=RuntimeError("copy failed")), patch.object(
            broker.shutil,
            "rmtree",
            side_effect=fail_staging_rmtree,
        ):
            with self.assertRaises(OSError):
                remote_broker.seal_template(
                    "mcp-a",
                    template_name="quarantine",
                )

        staging = [
            item
            for item in remote_broker.session_list()["sessions"]
            if item["template_role"] == broker.TEMPLATE_ROLE_STAGING
        ]
        self.assertEqual(len(staging), 1)
        self.assertTrue(
            any(
                path.is_dir()
                for path in (
                    remote_broker.config.runtime_root
                    / "templates"
                    / "quarantine"
                    / "staging"
                ).iterdir()
            )
        )
        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker.delete_template(template_name="quarantine")

    def test_template_start_rollback_failure_keeps_unknown_owned_session(self) -> None:
        controller = FailingOnceReadyController()
        remote_broker, launcher, _ = self.make_broker(controller=controller)

        with patch.object(launcher, "stop", side_effect=RuntimeError("stop failed")):
            with self.assertRaises(RuntimeError):
                remote_broker.create_template("mcp-a", template_name="unknown")

        status = remote_broker.whoami("mcp-a")
        self.assertEqual(status["browser_session"]["state"], "unknown")
        self.assertTrue(remote_broker._session_or_error("mcp-a").profile_dir.exists())

    def test_reconciliation_quarantines_malicious_profile_pointer_without_deleting_target(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        runtime_root = root / "runtime"
        record_root = runtime_root / "sessions" / "session-a"
        (record_root / "profile").mkdir(parents=True)
        protected = root / "protected"
        protected.mkdir()
        sentinel = protected / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        metadata = {
            "id": "session-a",
            "mcp_session_key": "mcp-a",
            "state": "stopped_retained",
            "created_at": broker.utc_now(),
            "last_used_at": broker.utc_now(),
            "template_role": "runtime",
            "retention_expires_at": broker.utc_after(60),
            "profile_dir": str(protected),
            "metadata_version": 1,
        }
        (record_root / "metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        remote_broker = broker.RemoteBrowserBroker(
            config=broker.BrokerConfig(
                workspace_root=root,
                runtime_root=runtime_root,
            ),
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                remote_broker.begin_shutdown(),
                remote_broker.release_shutdown_leases(),
            )
        )

        cleaned = remote_broker.cleanup_orphaned_sessions()

        self.assertEqual(cleaned["cleaned_count"], 1)
        self.assertFalse(record_root.exists())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_reconciliation_quarantines_every_duplicate_session_key_record(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        runtime_root = root / "runtime"
        for session_id in ("session-a", "session-b"):
            record_root = runtime_root / "sessions" / session_id
            profile = record_root / "profile"
            profile.mkdir(parents=True)
            (record_root / "metadata.json").write_text(
                json.dumps(
                    {
                        "id": session_id,
                        "mcp_session_key": "duplicate-key",
                        "state": "stopped_retained",
                        "created_at": broker.utc_now(),
                        "last_used_at": broker.utc_now(),
                        "template_role": "runtime",
                        "retention_expires_at": broker.utc_after(60),
                        "profile_dir": str(profile),
                        "metadata_version": 1,
                    }
                ),
                encoding="utf-8",
            )
        remote_broker = broker.RemoteBrowserBroker(
            config=broker.BrokerConfig(
                workspace_root=root,
                runtime_root=runtime_root,
            ),
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                remote_broker.begin_shutdown(),
                remote_broker.release_shutdown_leases(),
            )
        )

        inventory = remote_broker.session_list()

        self.assertFalse(remote_broker.whoami("duplicate-key")["has_browser_session"])
        self.assertEqual(len(inventory["sessions"]), 2)
        self.assertEqual(
            {item["state"] for item in inventory["sessions"]},
            {"orphaned"},
        )

    def test_orphan_cleanup_fails_closed_for_foreign_container_mounting_runtime(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        runtime_root = root / "runtime"
        orphan_root = runtime_root / "sessions" / "orphan-a"
        (orphan_root / "profile").mkdir(parents=True)
        config = broker.BrokerConfig(
            workspace_root=root,
            runtime_root=runtime_root,
        )
        remote_broker = broker.RemoteBrowserBroker(
            config=config,
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                remote_broker.begin_shutdown(),
                remote_broker.release_shutdown_leases(),
            )
        )
        foreign = {
            "id": "c" * 12,
            "name": "foreign-container",
            "labels": {},
            "mount_sources": [str(runtime_root)],
        }

        with patch.object(
            broker,
            "docker_containers_with_label",
            return_value=([foreign], None),
        ):
            with self.assertRaises(broker.BrowserBrokerError):
                remote_broker.cleanup_orphaned_sessions()

        self.assertTrue(orphan_root.exists())

    def test_crashed_template_container_cleanup_uses_labeled_session_identity(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        runtime_root = root / "runtime"
        profile = (
            runtime_root
            / "templates"
            / "template-a"
            / "drafts"
            / "draft-a"
            / "profile"
        )
        profile.mkdir(parents=True)
        config = broker.BrokerConfig(
            workspace_root=root,
            runtime_root=runtime_root,
        )
        remote_broker = broker.RemoteBrowserBroker(
            config=config,
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                remote_broker.begin_shutdown(),
                remote_broker.release_shutdown_leases(),
            )
        )
        session_id = "session-a"
        container_name = f"tokimon-rb-{config.installation_id}-{session_id}"
        labels = {
            "tokimon.remote-browser-broker": "true",
            "tokimon.remote-browser-installation": config.installation_id,
            "tokimon.remote-browser-owner-user": config.owner_user,
            "tokimon.remote-browser-checkout": str(config.workspace_root),
            "tokimon.remote-browser-runtime-root": str(config.runtime_root),
            "tokimon.remote-browser-session": session_id,
            "tokimon.remote-browser-network-policy": "previous-policy",
        }
        inventory_container = {
            "id": "d" * 64,
            "name": container_name,
            "labels": labels,
            "mount_sources": [str(profile)],
        }
        inspect_payload = [
            {
                "Id": "d" * 64,
                "Name": f"/{container_name}",
                "Config": {"Labels": labels},
                "Mounts": [
                    {
                        "Source": str(profile),
                        "Destination": "/profile",
                    }
                ],
            }
        ]
        inspect_count = 0

        def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            nonlocal inspect_count
            if args[:3] == ["docker", "container", "inspect"]:
                inspect_count += 1
                if inspect_count == 1:
                    return subprocess.CompletedProcess(
                        args,
                        0,
                        json.dumps(inspect_payload),
                        "",
                    )
                return subprocess.CompletedProcess(
                    args,
                    1,
                    "",
                    "Error: No such container",
                )
            if args[:3] == ["docker", "network", "inspect"]:
                return subprocess.CompletedProcess(
                    args,
                    1,
                    "",
                    "Error: No such network",
                )
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch.object(
            broker,
            "docker_containers_with_label",
            return_value=([inventory_container], None),
        ), patch.object(
            broker.shutil,
            "which",
            return_value="/usr/bin/docker",
        ), patch.object(
            broker.subprocess,
            "run",
            side_effect=fake_run,
        ):
            cleaned = remote_broker.cleanup_orphaned_sessions(
                )

        self.assertEqual(cleaned["cleaned_count"], 1)
        self.assertFalse(profile.parent.exists())

    def test_template_version_and_profile_symlinks_are_rejected(self) -> None:
        remote_broker, _, _ = self.make_broker()
        template_dir = remote_broker.config.runtime_root / "templates" / "safe"
        versions_root = template_dir / "versions"
        versions_root.mkdir(parents=True)
        external_version = remote_broker.config.runtime_root.parent / "external-version"
        (external_version / "profile").mkdir(parents=True)
        (external_version / "metadata.json").write_text(
            json.dumps({"name": "safe", "version_id": "v-link"}),
            encoding="utf-8",
        )
        (versions_root / "v-link").symlink_to(external_version, target_is_directory=True)
        owned_version = versions_root / "v-safe"
        owned_version.mkdir()
        (owned_version / "profile").symlink_to(
            external_version / "profile",
            target_is_directory=True,
        )
        (owned_version / "metadata.json").write_text(
            json.dumps({"name": "safe", "version_id": "v-safe"}),
            encoding="utf-8",
        )

        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker._read_template("safe", version_id="../escape")
        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker._read_template("safe", version_id="v-link")
        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker._read_template("safe", version_id="v-safe")

    def test_template_directory_symlink_escape_is_rejected(self) -> None:
        remote_broker, _, _ = self.make_broker()
        templates_root = remote_broker.config.runtime_root / "templates"
        templates_root.mkdir(parents=True, exist_ok=True)
        outside_template = remote_broker.config.runtime_root.parent / "outside-template"
        outside_template.mkdir()
        (templates_root / "evil").symlink_to(
            outside_template,
            target_is_directory=True,
        )

        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker.create_template("mcp-a", template_name="evil")
        self.assertEqual(list(outside_template.iterdir()), [])

    def test_snapshot_limits_are_enforced_in_direct_and_mcp_contracts(self) -> None:
        remote_broker, _, controller = self.make_broker()
        remote_broker.ensure("mcp-a")
        server = broker.MCPServer(remote_broker)

        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker.snapshot(
                "mcp-a",
                max_text_chars=broker.MAX_SNAPSHOT_TEXT_CHARS + 1,
            )
        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "browser_snapshot",
                    "arguments": {
                        "max_elements": broker.MAX_SNAPSHOT_ELEMENTS + 1,
                    },
                },
            },
            mcp_session_key="mcp-a",
        )
        listed = server.handle(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            mcp_session_key="mcp-a",
        )
        snapshot_tool = next(
            tool
            for tool in listed["result"]["tools"]
            if tool["name"] == "browser_snapshot"
        )

        self.assertEqual(tool_payload(response)["error_code"], "INVALID_ARGUMENT")
        self.assertEqual(controller.snapshots, [])
        self.assertEqual(
            snapshot_tool["inputSchema"]["properties"]["max_text_chars"]["maximum"],
            broker.MAX_SNAPSHOT_TEXT_CHARS,
        )
        self.assertEqual(
            snapshot_tool["inputSchema"]["properties"]["max_elements"]["maximum"],
            broker.MAX_SNAPSHOT_ELEMENTS,
        )

    def test_cross_process_mcp_lease_blocks_duplicate_live_session(self) -> None:
        first, _, _ = self.make_broker()
        original = first.ensure("shared-key")
        second_launcher = FakeLauncher()
        second = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=second_launcher,  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                second.begin_shutdown(),
                second.release_shutdown_leases(),
            )
        )

        with self.assertRaises(broker.BrowserBrokerError):
            second.ensure("shared-key")

        self.assertEqual(second_launcher.started, [])
        self.assertEqual(
            first.whoami("shared-key")["browser_session"]["id"],
            original["browser_session"]["id"],
        )

    def test_peer_cleanup_promotes_new_runtime_retained_record(self) -> None:
        first, _, _ = self.make_broker()
        original = first.ensure("handoff-key")
        profile_root = first._session_or_error("handoff-key").profile_dir.parent
        second = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                second.begin_shutdown(),
                second.release_shutdown_leases(),
            )
        )

        first.stop("handoff-key")
        first.begin_shutdown()
        first.release_shutdown_leases()
        cleaned = second.cleanup_orphaned_sessions()

        self.assertEqual(cleaned["cleaned_count"], 0)
        self.assertEqual(cleaned["reclassified_count"], 1)
        self.assertTrue(profile_root.exists())
        self.assertEqual(
            second.whoami("handoff-key")["browser_session"]["id"],
            original["browser_session"]["id"],
        )

    def test_peer_cleanup_promotes_new_template_retained_record(self) -> None:
        first, _, _ = self.make_broker()
        created = first.create_template("template-key", template_name="handoff")
        draft_root = first._session_or_error("template-key").profile_dir.parent
        second = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                second.begin_shutdown(),
                second.release_shutdown_leases(),
            )
        )

        first.shutdown_all_sessions()
        first.release_shutdown_leases()
        cleaned = second.cleanup_orphaned_sessions()

        self.assertEqual(cleaned["cleaned_count"], 0)
        self.assertEqual(cleaned["reclassified_count"], 1)
        self.assertTrue(draft_root.exists())
        self.assertEqual(
            second.whoami("template-key")["browser_session"]["id"],
            created["browser_session"]["id"],
        )
        self.assertEqual(
            second.seal_template(
                "template-key",
                template_name="handoff",
            )["status"],
            "sealed",
        )

    def test_released_peer_retained_record_blocks_template_key_reuse(self) -> None:
        first, _, _ = self.make_broker()
        original = first.ensure("retained-key")
        first.stop("retained-key")
        second = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                second.begin_shutdown(),
                second.release_shutdown_leases(),
            )
        )
        first.begin_shutdown()
        first.release_shutdown_leases()

        with self.assertRaises(broker.BrowserBrokerError):
            second.create_template(
                "retained-key",
                template_name="must-not-rebind",
            )

        self.assertEqual(
            second.whoami("retained-key")["browser_session"]["id"],
            original["browser_session"]["id"],
        )

    def test_active_checkpoint_blocks_duplicate_after_abrupt_owner_release(self) -> None:
        for checkpoint_state in (broker.STATE_RUNNING, "paused_private"):
            with self.subTest(checkpoint_state=checkpoint_state):
                first, _, _ = self.make_broker()
                original = first.ensure("crash-key")
                first.stop("crash-key")
                resumed = first.ensure("crash-key")
                active = first._session_or_error("crash-key")
                metadata_path = active.profile_dir.parent / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["state"], broker.STATE_RUNNING)
                self.assertIsNone(metadata["retention_expires_at"])
                metadata["state"] = checkpoint_state
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                first.begin_shutdown()
                first.release_shutdown_leases()

                second_launcher = FakeLauncher()
                second = broker.RemoteBrowserBroker(
                    config=first.config,
                    launcher=second_launcher,  # type: ignore[arg-type]
                    controller=FakeController(),  # type: ignore[arg-type]
                    novnc_proxy_factory=FakeNoVNCProxy,
                )
                self.addCleanup(
                    lambda owned_broker=second: (
                        owned_broker.begin_shutdown(),
                        owned_broker.release_shutdown_leases(),
                    )
                )

                with self.assertRaises(broker.BrowserBrokerError):
                    second.ensure("crash-key")

                inventory = second.session_list()
                self.assertEqual(inventory["sessions"][0]["state"], broker.STATE_UNKNOWN)
                self.assertEqual(second_launcher.started, [])
                self.assertEqual(
                    original["browser_session"]["id"],
                    resumed["browser_session"]["id"],
                )

    def test_active_and_retained_duplicate_key_cannot_be_promoted(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        runtime_root = root / "runtime"
        for session_id, state, expiry in (
            ("active001", broker.STATE_RUNNING, None),
            ("retain001", broker.STATE_STOPPED_RETAINED, broker.utc_after(60)),
        ):
            record_root = runtime_root / "sessions" / session_id
            profile = record_root / "profile"
            profile.mkdir(parents=True)
            (record_root / "metadata.json").write_text(
                json.dumps(
                    {
                        "id": session_id,
                        "mcp_session_key": "duplicate-key",
                        "state": state,
                        "created_at": broker.utc_now(),
                        "last_used_at": broker.utc_now(),
                        "template_role": broker.TEMPLATE_ROLE_RUNTIME,
                        "retention_expires_at": expiry,
                        "profile_dir": str(profile),
                        "metadata_version": 1,
                    }
                ),
                encoding="utf-8",
            )
        launcher = FakeLauncher()
        remote_broker = broker.RemoteBrowserBroker(
            config=broker.BrokerConfig(
                workspace_root=root,
                runtime_root=runtime_root,
            ),
            launcher=launcher,  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                remote_broker.begin_shutdown(),
                remote_broker.release_shutdown_leases(),
            )
        )

        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker.ensure("duplicate-key")

        self.assertEqual(launcher.started, [])
        self.assertEqual(
            {item["state"] for item in remote_broker.session_list()["sessions"]},
            {broker.STATE_ORPHANED},
        )

    def test_partial_duplicate_cleanup_failure_keeps_active_key_blocked(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        runtime_root = root / "runtime"
        for session_id, state, expiry in (
            ("a-retained", broker.STATE_STOPPED_RETAINED, broker.utc_after(60)),
            ("z-active", broker.STATE_RUNNING, None),
        ):
            record_root = runtime_root / "sessions" / session_id
            profile = record_root / "profile"
            profile.mkdir(parents=True)
            (record_root / "metadata.json").write_text(
                json.dumps(
                    {
                        "id": session_id,
                        "mcp_session_key": "cleanup-duplicate-key",
                        "state": state,
                        "created_at": broker.utc_now(),
                        "last_used_at": broker.utc_now(),
                        "template_role": broker.TEMPLATE_ROLE_RUNTIME,
                        "retention_expires_at": expiry,
                        "profile_dir": str(profile),
                        "metadata_version": 1,
                    }
                ),
                encoding="utf-8",
            )
        launcher = FakeLauncher()
        remote_broker = broker.RemoteBrowserBroker(
            config=broker.BrokerConfig(
                workspace_root=root,
                runtime_root=runtime_root,
            ),
            launcher=launcher,  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                remote_broker.begin_shutdown(),
                remote_broker.release_shutdown_leases(),
            )
        )

        def fail_active_stop(session: broker.BrowserSession) -> None:
            if session.id == "z-active":
                raise OSError("active container did not stop")
            launcher.stopped.append(session.container_name)

        with patch.object(launcher, "stop", side_effect=fail_active_stop):
            with self.assertRaises(OSError):
                remote_broker.cleanup_orphaned_sessions()

        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker.ensure("cleanup-duplicate-key")
        self.assertEqual(launcher.started, [])
        self.assertTrue((runtime_root / "sessions" / "z-active").exists())

    def test_active_and_retained_duplicate_id_cannot_be_promoted(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        runtime_root = root / "runtime"
        runtime_record = runtime_root / "sessions" / "shared-id"
        runtime_profile = runtime_record / "profile"
        runtime_profile.mkdir(parents=True)
        (runtime_record / "metadata.json").write_text(
            json.dumps(
                {
                    "id": "shared-id",
                    "mcp_session_key": "active-key",
                    "state": broker.STATE_RUNNING,
                    "created_at": broker.utc_now(),
                    "last_used_at": broker.utc_now(),
                    "template_role": broker.TEMPLATE_ROLE_RUNTIME,
                    "retention_expires_at": None,
                    "profile_dir": str(runtime_profile),
                    "metadata_version": 1,
                }
            ),
            encoding="utf-8",
        )
        draft_root = (
            runtime_root
            / "templates"
            / "shared-template"
            / "drafts"
            / "draft-a"
        )
        draft_profile = draft_root / "profile"
        draft_profile.mkdir(parents=True)
        (draft_root / "session-metadata.json").write_text(
            json.dumps(
                {
                    "id": "shared-id",
                    "mcp_session_key": "draft-key",
                    "state": broker.STATE_STOPPED_RETAINED,
                    "created_at": broker.utc_now(),
                    "last_used_at": broker.utc_now(),
                    "template_name": "shared-template",
                    "template_version_id": "draft-a",
                    "template_role": broker.TEMPLATE_ROLE_EDITING,
                    "retention_expires_at": broker.utc_after(60),
                    "profile_dir": str(draft_profile),
                    "metadata_version": 1,
                }
            ),
            encoding="utf-8",
        )
        launcher = FakeLauncher()
        remote_broker = broker.RemoteBrowserBroker(
            config=broker.BrokerConfig(
                workspace_root=root,
                runtime_root=runtime_root,
            ),
            launcher=launcher,  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                remote_broker.begin_shutdown(),
                remote_broker.release_shutdown_leases(),
            )
        )

        for blocked_key in ("active-key", "draft-key"):
            with self.assertRaises(broker.BrowserBrokerError):
                remote_broker.ensure(blocked_key)

        self.assertEqual(launcher.started, [])
        self.assertEqual(remote_broker.session_list()["session_count"], 2)

    def test_runtime_collection_root_symlink_is_rejected(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        runtime_root = root / "runtime"
        outside = root / "outside-sessions"
        runtime_root.mkdir()
        outside.mkdir()
        sentinel = outside / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        (runtime_root / "sessions").symlink_to(outside, target_is_directory=True)

        with self.assertRaises(broker.BrowserBrokerError):
            broker.RemoteBrowserBroker(
                config=broker.BrokerConfig(
                    workspace_root=root,
                    runtime_root=runtime_root,
                ),
                launcher=FakeLauncher(),  # type: ignore[arg-type]
                controller=FakeController(),  # type: ignore[arg-type]
                novnc_proxy_factory=FakeNoVNCProxy,
            )

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_template_collection_symlinks_are_rejected_without_traversal(self) -> None:
        remote_broker, _, _ = self.make_broker()
        templates_root = remote_broker.config.runtime_root / "templates"
        for collection_name in ("versions", "drafts", "staging"):
            template_name = f"unsafe-{collection_name}"
            template_dir = templates_root / template_name
            outside = remote_broker.config.runtime_root.parent / f"outside-{collection_name}"
            template_dir.mkdir(parents=True)
            outside.mkdir()
            sentinel = outside / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            (template_dir / collection_name).symlink_to(
                outside,
                target_is_directory=True,
            )

            if collection_name == "versions":
                with self.assertRaises(broker.BrowserBrokerError):
                    remote_broker.list_templates()
            else:
                with self.assertRaises(broker.BrowserBrokerError):
                    remote_broker.create_template(
                        f"key-{collection_name}",
                        template_name=template_name,
                    )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_abstract_lease_churn_leaves_no_lock_artifacts(self) -> None:
        remote_broker, _, _ = self.make_broker()
        for index in range(20):
            key = f"churn-{index}"
            remote_broker.ensure(key)
            remote_broker.stop(key)
            remote_broker.delete_retained_session(key)
        for index in range(3):
            key = f"template-churn-{index}"
            name = f"template-{index}"
            remote_broker.create_template(key, template_name=name)
            remote_broker.seal_template(
                key,
                template_name=name,
            )
            remote_broker.delete_template(template_name=name)

        artifact_names = {
            path.name
            for path in remote_broker.config.runtime_root.rglob("*")
            if "lock" in path.name.lower() or "lease" in path.name.lower()
        }
        self.assertEqual(artifact_names, set())
        self.assertEqual(list((remote_broker.config.runtime_root / "sessions").iterdir()), [])
        self.assertEqual(list((remote_broker.config.runtime_root / "templates").iterdir()), [])

    def test_constructor_reconciliation_failure_releases_acquired_leases(self) -> None:
        first, _, _ = self.make_broker()
        first.ensure("expired-key")
        first.stop("expired-key")
        retained = first._retained_by_mcp["expired-key"]
        retained.state = broker.STATE_EXPIRED
        retained.retention_expires_at = "2000-01-01T00:00:00+00:00"
        first._write_session_metadata(retained)
        first.begin_shutdown()
        first.release_shutdown_leases()

        with patch.object(
            broker.RemoteBrowserBroker,
            "_write_session_metadata",
            side_effect=OSError("disk failed"),
        ):
            with self.assertRaises(OSError) as failed:
                broker.RemoteBrowserBroker(
                    config=first.config,
                    launcher=FakeLauncher(),  # type: ignore[arg-type]
                    controller=FakeController(),  # type: ignore[arg-type]
                    novnc_proxy_factory=FakeNoVNCProxy,
                )
        self.assertEqual(str(failed.exception), "disk failed")

        peer = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                peer.begin_shutdown(),
                peer.release_shutdown_leases(),
            )
        )
        self.assertEqual(
            peer.whoami("expired-key")["browser_session"]["state"],
            broker.STATE_EXPIRED,
        )

    def test_session_root_acquisition_exception_releases_mcp_lease(self) -> None:
        first, _, _ = self.make_broker()
        with patch.object(
            first,
            "_try_acquire_session_lease",
            side_effect=OSError("root failed"),
        ):
            with self.assertRaises(OSError):
                first.ensure("reusable-key")

        peer_launcher = FakeLauncher()
        peer = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=peer_launcher,  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                peer.begin_shutdown(),
                peer.release_shutdown_leases(),
            )
        )
        self.assertEqual(peer.ensure("reusable-key")["status"], "ready")

    def test_template_admission_failure_closes_socket_editor_lease(self) -> None:
        first, _, _ = self.make_broker()
        first.create_template("setup-key", template_name="source")
        first.seal_template(
            "setup-key",
            template_name="source",
        )
        first.ensure("held-key")
        second = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                second.begin_shutdown(),
                second.release_shutdown_leases(),
            )
        )

        with self.assertRaises(broker.BrowserBrokerError):
            second.create_template("held-key", template_name="new-template")
        create_lease = second._try_acquire_template_editor_lease("new-template")
        self.assertIsNotNone(create_lease)
        assert create_lease is not None
        create_lease.close()

        with self.assertRaises(broker.BrowserBrokerError):
            second.refresh_template("held-key", template_name="source")
        refresh_lease = second._try_acquire_template_editor_lease("source")
        self.assertIsNotNone(refresh_lease)
        assert refresh_lease is not None
        refresh_lease.close()

    def test_template_collection_failure_closes_unpublished_editor_lease(self) -> None:
        remote_broker, _, _ = self.make_broker()
        remote_broker.create_template("setup", template_name="refresh-source")
        remote_broker.seal_template(
            "setup",
            template_name="refresh-source",
        )
        original_collection_root = remote_broker._template_collection_root

        def fail_create(
            template_name: str,
            collection_name: str,
            *,
            create: bool,
        ) -> Path:
            if create and collection_name == "drafts":
                raise OSError("draft root failed")
            return original_collection_root(
                template_name,
                collection_name,
                create=create,
            )

        with patch.object(
            remote_broker,
            "_template_collection_root",
            side_effect=fail_create,
        ):
            with self.assertRaises(OSError) as create_failure:
                remote_broker.create_template(
                    "create-key",
                    template_name="create-failure",
                )
        self.assertEqual(str(create_failure.exception), "draft root failed")
        create_lease = remote_broker._try_acquire_template_editor_lease(
            "create-failure"
        )
        self.assertIsNotNone(create_lease)
        assert create_lease is not None
        create_lease.close()

        with patch.object(
            remote_broker,
            "_template_collection_root",
            side_effect=fail_create,
        ):
            with self.assertRaises(OSError) as refresh_failure:
                remote_broker.refresh_template(
                    "refresh-key",
                    template_name="refresh-source",
                )
        self.assertEqual(str(refresh_failure.exception), "draft root failed")
        refresh_lease = remote_broker._try_acquire_template_editor_lease(
            "refresh-source"
        )
        self.assertIsNotNone(refresh_lease)
        assert refresh_lease is not None
        refresh_lease.close()

    def test_failed_start_delete_checkpoint_blocks_replacement_after_handoff(self) -> None:
        first, _, _ = self.make_broker(controller=FailingOnceReadyController())

        with patch.object(
            broker.shutil,
            "rmtree",
            side_effect=OSError("profile delete failed"),
        ):
            with self.assertRaises(OSError):
                first.ensure("failed-start-key")

        failed = first._sessions_by_mcp["failed-start-key"]
        metadata_path = failed.profile_dir.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(failed.state, broker.STATE_UNKNOWN)
        self.assertEqual(metadata["state"], broker.STATE_UNKNOWN)
        self.assertEqual(metadata["mcp_session_key"], "failed-start-key")
        first.begin_shutdown()
        first.release_shutdown_leases()

        peer = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                peer.begin_shutdown(),
                peer.release_shutdown_leases(),
            )
        )
        with self.assertRaises(broker.BrowserBrokerError):
            peer.ensure("failed-start-key")
        self.assertEqual(
            peer.session_list()["sessions"][0]["state"],
            broker.STATE_UNKNOWN,
        )

    def test_failed_clone_cleanup_remains_indexed_and_checkpointed(self) -> None:
        first, _, _ = self.make_broker()
        first.create_template("setup", template_name="clone-source")
        first.seal_template(
            "setup",
            template_name="clone-source",
        )

        def fail_partial_copy(_: Path, destination: Path) -> None:
            destination.mkdir(parents=True, exist_ok=True)
            raise OSError("copy failed")

        with patch.object(first, "_copy_profile", side_effect=fail_partial_copy), patch.object(
            broker.shutil,
            "rmtree",
            side_effect=OSError("cleanup failed"),
        ):
            with self.assertRaises(OSError):
                first.ensure("clone-key", template_name="clone-source")

        failed = first._sessions_by_mcp["clone-key"]
        checkpoint = json.loads(
            (failed.profile_dir.parent / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(failed.state, broker.STATE_UNKNOWN)
        self.assertEqual(checkpoint["state"], broker.STATE_UNKNOWN)
        self.assertIsNotNone(failed.mcp_ownership_lease)
        self.assertIsNotNone(failed.ownership_lease)
        first.begin_shutdown()
        first.release_shutdown_leases()

        peer = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                peer.begin_shutdown(),
                peer.release_shutdown_leases(),
            )
        )
        with self.assertRaises(broker.BrowserBrokerError):
            peer.ensure("clone-key", template_name="clone-source")


    def test_http_bind_failure_releases_retained_leases(self) -> None:
        first, _, _ = self.make_broker()
        first.ensure("http-retained")
        first.stop("http-retained")
        retained = first._retained_by_mcp["http-retained"]

        with patch.object(
            broker,
            "ThreadingHTTPServer",
            side_effect=OSError("bind failed"),
        ):
            with self.assertRaises(OSError):
                broker.serve_http(config=first.config, broker=first)

        self.assertEqual(first._retained_by_mcp, {})
        self.assertIsNone(retained.ownership_lease)
        self.assertIsNone(retained.mcp_ownership_lease)

    def test_release_waits_for_inflight_retained_delete_before_handoff(self) -> None:
        first, _, _ = self.make_broker()
        first.ensure("handoff-key")
        first.stop("handoff-key")
        delete_entered = threading.Event()
        delete_release = threading.Event()
        release_finished = threading.Event()
        release_wait_entered = threading.Event()
        delete_errors: list[BaseException] = []
        original_delete = first._delete_session_runtime
        original_wait_for_drain = first._wait_for_admission_drain

        def blocking_delete(session: broker.BrowserSession) -> None:
            delete_entered.set()
            if not delete_release.wait(timeout=2):
                raise RuntimeError("timed out waiting to release retained delete")
            original_delete(session)

        def delete_retained() -> None:
            try:
                first.delete_retained_session("handoff-key")
            except BaseException as exc:  # noqa: BLE001 - asserted below.
                delete_errors.append(exc)

        def tracked_wait_for_admission_drain() -> None:
            release_wait_entered.set()
            original_wait_for_drain()

        with (
            patch.object(first, "_delete_session_runtime", blocking_delete),
            patch.object(
                first,
                "_wait_for_admission_drain",
                tracked_wait_for_admission_drain,
            ),
        ):
            delete_thread = threading.Thread(target=delete_retained)
            delete_thread.start()
            self.assertTrue(delete_entered.wait(timeout=1))
            first.begin_shutdown()
            release_thread = threading.Thread(
                target=lambda: (
                    first.release_shutdown_leases(),
                    release_finished.set(),
                )
            )
            release_thread.start()
            self.assertTrue(release_wait_entered.wait(timeout=1))
            self.assertFalse(release_finished.is_set())
            delete_release.set()
            delete_thread.join(timeout=2)
            release_thread.join(timeout=2)

        self.assertEqual(delete_errors, [])
        self.assertTrue(release_finished.is_set())
        peer = broker.RemoteBrowserBroker(
            config=first.config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                peer.begin_shutdown(),
                peer.release_shutdown_leases(),
            )
        )
        peer.ensure("handoff-key")
        peer_root = peer._session_or_error("handoff-key").profile_dir.parent
        with self.assertRaises(broker.BrowserBrokerError):
            first.delete_retained_session("handoff-key")
        self.assertTrue(peer_root.exists())

    def test_event_stream_exits_promptly_when_broker_shutdown_begins(self) -> None:
        httpd, port, remote_broker, _, _ = self.make_http_server()
        httpd.daemon_threads = False
        httpd.block_on_close = True
        sock = socket.create_connection(("127.0.0.1", port), timeout=2)
        self.addCleanup(sock.close)
        sock.settimeout(2)
        sock.sendall(
            b"GET /mcp HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Accept: text/event-stream\r\n"
            b"mcp-session-id: shutdown-stream\r\n\r\n"
        )
        response = b""
        while b": connected" not in response:
            response += sock.recv(2048)

        remote_broker.begin_shutdown()
        close_errors: list[BaseException] = []

        def close_server() -> None:
            try:
                httpd.shutdown()
                httpd.server_close()
            except BaseException as exc:  # noqa: BLE001 - asserted below.
                close_errors.append(exc)

        closer = threading.Thread(target=close_server)
        closer.start()
        closer.join(timeout=2)
        self.assertFalse(closer.is_alive())
        self.assertEqual(close_errors, [])

    def test_snapshot_payload_is_bounded_and_whitelisted_at_broker_boundary(self) -> None:
        remote_broker, _, controller = self.make_broker()
        remote_broker.ensure("mcp-a")
        oversized = "x" * (broker.MAX_SNAPSHOT_FIELD_CHARS + 100)
        adversarial = {
            "url": oversized,
            "title": oversized,
            "text": oversized,
            "unknown": oversized,
            "elements": [
                {
                    "tag": oversized,
                    "role": oversized,
                    "text": oversized,
                    "aria_label": oversized,
                    "placeholder": oversized,
                    "type": oversized,
                    "name": oversized,
                    "id": oversized,
                    "href": oversized,
                    "unknown": oversized,
                },
                {"tag": "button", "text": "extra"},
            ],
        }
        with patch.object(controller, "snapshot", return_value=adversarial):
            result = remote_broker.snapshot(
                "mcp-a",
                max_text_chars=7,
                max_elements=1,
            )

        page = result["page"]
        self.assertEqual(len(page["url"]), broker.MAX_SNAPSHOT_FIELD_CHARS)
        self.assertEqual(len(page["title"]), broker.MAX_SNAPSHOT_FIELD_CHARS)
        self.assertEqual(len(page["text"]), 7)
        self.assertNotIn("unknown", page)
        self.assertEqual(len(page["elements"]), 1)
        element = page["elements"][0]
        self.assertEqual(len(element["tag"]), broker.MAX_SNAPSHOT_FIELD_CHARS)
        self.assertEqual(len(element["text"]), 240)
        self.assertNotIn("unknown", element)
        self.assertIn("ref", element)
        expression = broker.snapshot_expression(snapshot_id="bounded-snapshot", max_text_chars=7, max_elements=1)
        self.assertIn(f"const fieldLimit = {broker.MAX_SNAPSHOT_FIELD_CHARS}", expression)
        self.assertIn("url: bounded(location.href)", expression)

    def test_screenshot_rejects_symlink_artifact_directory_before_capture(self) -> None:
        remote_broker, _, controller = self.make_broker()
        remote_broker.ensure("mcp-a")
        session = remote_broker._session_or_error("mcp-a")
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))
        (session.profile_dir.parent / "screenshots").symlink_to(
            outside,
            target_is_directory=True,
        )

        with self.assertRaises(broker.BrowserBrokerError) as error:
            remote_broker.screenshot("mcp-a")

        self.assertEqual(error.exception.code, broker.ERROR_BROKER)
        self.assertEqual(controller.screenshots, [])
        self.assertEqual(list(outside.iterdir()), [])

    def test_upload_rejects_symlink_staging_directory_before_copy(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        remote_broker.ensure("mcp-a")
        snapshot = remote_broker.snapshot("mcp-a")
        ref = snapshot["page"]["elements"][2]["ref"]
        source_root = Path(tempfile.mkdtemp())
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(source_root, ignore_errors=True))
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))
        source = source_root / "upload.txt"
        source.write_text("bounded upload", encoding="utf-8")
        session = remote_broker._session_or_error("mcp-a")
        (session.profile_dir / "uploads").symlink_to(
            outside,
            target_is_directory=True,
        )

        completed = self.call_mcp_tool(
            server,
            mcp_session_key="mcp-a",
            name="browser_upload_file",
            arguments={"ref": ref, "path": str(source)},
        )

        self.assertEqual(completed["error_code"], broker.ERROR_BROKER)
        self.assertEqual(controller.uploads, [])
        self.assertEqual(list(outside.iterdir()), [])

    def test_upload_rejects_source_changed_during_staging_and_removes_copy(self) -> None:
        remote_broker, _, controller = self.make_broker()
        server = broker.MCPServer(remote_broker)
        remote_broker.ensure("mcp-a")
        ref = remote_broker.snapshot("mcp-a")["page"]["elements"][2]["ref"]
        session = remote_broker._session_or_error("mcp-a")
        with tempfile.TemporaryDirectory() as source_root:
            source = Path(source_root) / "upload.txt"
            source.write_bytes(b"first")
            read_source = remote_broker._read_bounded_upload_source

            def change_after_read(
                path: Path,
                *,
                destination_stream: object | None = None,
            ) -> tuple[str, int]:
                result = read_source(path, destination_stream=destination_stream)
                if destination_stream is None:
                    source.write_bytes(b"other")
                return result

            with patch.object(
                remote_broker,
                "_read_bounded_upload_source",
                side_effect=change_after_read,
            ):
                result = self.call_mcp_tool(
                    server,
                    mcp_session_key="mcp-a",
                    name="browser_upload_file",
                    arguments={"ref": ref, "path": str(source)},
                )

        self.assertEqual(result["error_code"], broker.ERROR_INVALID_ARGUMENT)
        self.assertEqual(controller.uploads, [])
        self.assertEqual(list((session.profile_dir / "uploads").iterdir()), [])

    def test_upload_initial_digest_bounds_zero_size_hint_stream(self) -> None:
        source_root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(source_root, ignore_errors=True))
        source = source_root / "zero-size-hint.bin"
        source.write_bytes(b"placeholder")
        remote_broker, _, controller = self.make_broker()
        remote_broker.ensure("mcp-a")
        ref = remote_broker.snapshot("mcp-a")["page"]["elements"][2]["ref"]

        class ZeroSizeRegularFile:
            st_mode = broker.stat.S_IFREG | 0o600
            st_size = 0

        reads = [b"x" * 4097]

        def bounded_overflow_read(_: int, requested: int) -> bytes:
            self.assertLessEqual(requested, 4097)
            return reads.pop(0) if reads else b""

        with (
            patch.object(broker, "MAX_UPLOAD_BYTES", 4096),
            patch.object(broker.os, "open", return_value=123) as source_open,
            patch.object(broker.os, "fstat", return_value=ZeroSizeRegularFile()),
            patch.object(broker.os, "read", side_effect=bounded_overflow_read),
            patch.object(broker.os, "close") as source_close,
        ):
            with self.assertRaises(broker.BrowserBrokerError) as error:
                remote_broker.upload_file(
                    "mcp-a",
                    ref=ref,
                    path=str(source),
                )

        self.assertEqual(error.exception.code, broker.ERROR_INVALID_ARGUMENT)
        source_open.assert_called_once_with(
            source.resolve(),
            broker.os.O_RDONLY
            | broker.os.O_NOFOLLOW
            | getattr(broker.os, "O_NONBLOCK", 0),
        )
        source_close.assert_called_once_with(123)
        self.assertEqual(controller.uploads, [])

    def test_sealed_template_requires_real_sealed_metadata(self) -> None:
        remote_broker, _, _ = self.make_broker()
        versions_root = remote_broker._template_collection_root(
            "partial-template",
            "versions",
            create=True,
        )
        missing_metadata = versions_root / "v-missing"
        (missing_metadata / "profile").mkdir(parents=True)
        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker._read_template(
                "partial-template",
                version_id="v-missing",
            )

        wrong_state = versions_root / "v-editing"
        (wrong_state / "profile").mkdir(parents=True)
        (wrong_state / "metadata.json").write_text(
            json.dumps(
                {
                    "name": "partial-template",
                    "version_id": "v-editing",
                    "state": broker.TEMPLATE_STATE_EDITING,
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker._read_template(
                "partial-template",
                version_id="v-editing",
            )

    def test_template_clone_refuses_different_configured_runtime_image(self) -> None:
        remote_broker, _, _ = self.make_broker()
        versions_root = remote_broker._template_collection_root(
            "image-bound-template",
            "versions",
            create=True,
        )
        version = versions_root / "v-image-bound"
        (version / "profile").mkdir(parents=True)
        (version / "metadata.json").write_text(
            json.dumps(
                {
                    "name": "image-bound-template",
                    "version_id": "v-image-bound",
                    "state": broker.TEMPLATE_STATE_SEALED,
                    "runtime_image": "different-browser-image@sha256:deadbeef",
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaises(broker.BrowserBrokerError) as error:
            remote_broker.ensure(
                "mcp-a",
                template_name="image-bound-template",
            )

        self.assertEqual(error.exception.code, broker.ERROR_TEMPLATE_CONFLICT)
        self.assertFalse(remote_broker.whoami("mcp-a")["has_browser_session"])

    def test_backend_response_limit_must_be_positive_before_execution(self) -> None:
        remote_broker, _, controller = self.make_broker()
        remote_broker.ensure("mcp-a")

        for invalid_limit in (0, broker.MAX_BACKEND_RESPONSE_MAX_CHARS + 1):
            with self.subTest(max_response_chars=invalid_limit):
                with self.assertRaises(broker.BrowserBrokerError) as error:
                    remote_broker.backend_request(
                        "mcp-a",
                        method="GET",
                        path="/bounded",
                        max_response_chars=invalid_limit,
                    )

                self.assertEqual(error.exception.code, broker.ERROR_INVALID_ARGUMENT)
        self.assertEqual(controller.backend_requests, [])
        schema = next(
            item["inputSchema"]
            for item in broker.mcp_tools()
            if item["name"] == "browser_backend_request"
        )
        self.assertEqual(
            schema["properties"]["max_response_chars"]["maximum"],
            broker.MAX_BACKEND_RESPONSE_MAX_CHARS,
        )

    def test_profile_copy_rejects_nested_symlinks(self) -> None:
        remote_broker, _, _ = self.make_broker()
        source_root = Path(tempfile.mkdtemp())
        outside_root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(source_root, ignore_errors=True))
        self.addCleanup(lambda: shutil.rmtree(outside_root, ignore_errors=True))
        sentinel = outside_root / "secret.txt"
        sentinel.write_text("host secret", encoding="utf-8")
        (source_root / "linked-secret").symlink_to(sentinel)
        destination = remote_broker.config.runtime_root / "copy-target"

        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker._copy_profile(source_root, destination)

        self.assertFalse(destination.exists())

    def test_backend_timeout_is_capped_before_controller_execution(self) -> None:
        remote_broker, _, controller = self.make_broker()
        remote_broker.ensure("mcp-a")

        with self.assertRaises(broker.BrowserBrokerError) as error:
            remote_broker.backend_request(
                "mcp-a",
                method="GET",
                path="/bounded",
                timeout_ms=broker.MAX_BACKEND_REQUEST_TIMEOUT_MS + 1,
            )

        self.assertEqual(error.exception.code, broker.ERROR_INVALID_ARGUMENT)
        self.assertEqual(controller.backend_requests, [])
        schema = next(
            item["inputSchema"]
            for item in broker.mcp_tools()
            if item["name"] == "browser_backend_request"
        )
        self.assertEqual(
            schema["properties"]["timeout_ms"]["maximum"],
            broker.MAX_BACKEND_REQUEST_TIMEOUT_MS,
        )

    def test_http_rejects_oversized_jsonrpc_body_without_reading_it(self) -> None:
        _, port, _, _, _ = self.make_http_server()
        with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
            sock.settimeout(2)
            sock.sendall(
                b"POST /mcp HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                + f"Content-Length: {broker.MAX_MCP_REQUEST_BYTES + 1}\r\n".encode()
                + b"Content-Type: application/json\r\n\r\n"
            )
            response = sock.recv(2048)
        self.assertIn(b" 413 ", response)

    def test_cdp_rejects_oversized_frame_before_payload_read(self) -> None:
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        connection = broker.CDPConnection("ws://127.0.0.1/devtools/page/1")
        connection.sock = client
        server.sendall(
            bytes([0x81, 0x7F])
            + broker.struct.pack("!Q", broker.MAX_CDP_FRAME_BYTES + 1)
        )
        with self.assertRaises(broker.BrowserBrokerError):
            connection._recv_frame()

    def test_cdp_rejects_oversized_websocket_handshake(self) -> None:
        class OversizedHandshakeSocket:
            def __init__(self) -> None:
                self.remaining = bytearray(
                    b"HTTP/1.1 101 Switching Protocols\r\nX-Fill: "
                    + b"x" * broker.MAX_CDP_HANDSHAKE_BYTES
                )
                self.closed = False

            def settimeout(self, timeout: float) -> None:
                self.timeout = timeout

            def sendall(self, payload: bytes) -> None:
                self.request = payload

            def recv(self, length: int) -> bytes:
                chunk = bytes(self.remaining[:length])
                del self.remaining[:length]
                return chunk

            def close(self) -> None:
                self.closed = True

        fake_socket = OversizedHandshakeSocket()
        connection = broker.CDPConnection(
            "ws://127.0.0.1/devtools/page/1",
            timeout=1.0,
        )
        with patch.object(broker.socket, "create_connection", return_value=fake_socket):
            with self.assertRaises(broker.BrowserBrokerError) as error:
                connection.connect()

        self.assertIn("handshake exceeds", str(error.exception))
        self.assertTrue(fake_socket.closed)
        self.assertIsNone(connection.sock)

    def test_cdp_websocket_handshake_uses_one_deadline_while_bytes_drip(self) -> None:
        class DripHandshakeSocket:
            def __init__(self) -> None:
                self.closed = False
                self.recv_count = 0

            def settimeout(self, timeout: float) -> None:
                self.timeout = timeout

            def sendall(self, payload: bytes) -> None:
                self.request = payload

            def recv(self, length: int) -> bytes:
                self.recv_count += 1
                return b"H"

            def close(self) -> None:
                self.closed = True

        now = -0.2

        def advancing_monotonic() -> float:
            nonlocal now
            now += 0.2
            return now

        fake_socket = DripHandshakeSocket()
        connection = broker.CDPConnection(
            "ws://127.0.0.1/devtools/page/1",
            timeout=0.7,
        )
        with (
            patch.object(broker.socket, "create_connection", return_value=fake_socket),
            patch.object(broker.time, "monotonic", side_effect=advancing_monotonic),
        ):
            with self.assertRaises(broker.BrowserBrokerError) as error:
                connection.connect()

        self.assertIn("handshake timed out", str(error.exception))
        self.assertEqual(fake_socket.recv_count, 1)
        self.assertTrue(fake_socket.closed)

    def test_cdp_preserves_first_frame_received_with_websocket_handshake(self) -> None:
        result_payload = b'{"id":1,"result":{"value":"ready"}}'
        first_frame = bytes([0x81, len(result_payload)]) + result_payload

        class HandshakeAndFrameSocket:
            def __init__(self) -> None:
                self.remaining = bytearray(
                    b"HTTP/1.1 101 Switching Protocols\r\n"
                    b"Upgrade: websocket\r\n\r\n"
                    + first_frame
                )
                self.sent: list[bytes] = []

            def settimeout(self, timeout: float) -> None:
                self.timeout = timeout

            def sendall(self, payload: bytes) -> None:
                self.sent.append(payload)

            def recv(self, length: int) -> bytes:
                chunk = bytes(self.remaining[:length])
                del self.remaining[:length]
                return chunk

            def close(self) -> None:
                self.closed = True

        fake_socket = HandshakeAndFrameSocket()
        connection = broker.CDPConnection(
            "ws://127.0.0.1/devtools/page/1",
            timeout=1.0,
        )
        with patch.object(broker.socket, "create_connection", return_value=fake_socket):
            connection.connect()
        self.addCleanup(connection.close)

        self.assertEqual(connection.call("Runtime.evaluate"), {"value": "ready"})
        self.assertEqual(fake_socket.remaining, bytearray())

    def test_cdp_command_rejects_unmatched_event_flood(self) -> None:
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        connection = broker.CDPConnection(
            "ws://127.0.0.1/devtools/page/1",
            timeout=10.0,
        )
        connection.sock = client
        event_payload = (b'{"method":"Runtime.consoleAPICalled"}', 0x1)
        with (
            patch.object(connection, "_send_json"),
            patch.object(connection, "_recv_frame", return_value=event_payload) as recv_frame,
        ):
            with self.assertRaises(broker.BrowserBrokerError) as error:
                connection.call("Runtime.evaluate")

        self.assertIn("unmatched-message limit", str(error.exception))
        self.assertEqual(recv_frame.call_count, broker.MAX_CDP_UNMATCHED_MESSAGES)

    def test_cdp_command_event_loop_uses_one_total_deadline(self) -> None:
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        connection = broker.CDPConnection(
            "ws://127.0.0.1/devtools/page/1",
            timeout=1.0,
        )
        connection.sock = client
        now = -0.25

        def advancing_monotonic() -> float:
            nonlocal now
            now += 0.25
            return now

        event_payload = (b'{"method":"Runtime.consoleAPICalled"}', 0x1)
        with (
            patch.object(connection, "_send_json"),
            patch.object(connection, "_recv_frame", return_value=event_payload) as recv_frame,
            patch.object(broker.time, "monotonic", side_effect=advancing_monotonic),
        ):
            with self.assertRaises(broker.BrowserBrokerError) as error:
                connection.call("Runtime.evaluate")

        self.assertIn("command response timed out", str(error.exception))
        self.assertEqual(recv_frame.call_count, 1)

    def test_session_capacity_limits_active_and_total_profiles(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        config = broker.BrokerConfig(
            runtime_root=Path(tmpdir.name) / "remote-browser",
            cdp_port_range=(32000, 32999),
            novnc_port_range=(42000, 42999),
            max_active_sessions=1,
            max_total_sessions=2,
        )
        remote_broker = broker.RemoteBrowserBroker(
            config=config,
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(
            lambda: (
                remote_broker.begin_shutdown(),
                remote_broker.release_shutdown_leases(),
            )
        )
        remote_broker.ensure("one")
        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker.ensure("two")
        remote_broker.stop("one")
        remote_broker.ensure("two")
        remote_broker.stop("two")
        with self.assertRaises(broker.BrowserBrokerError):
            remote_broker.ensure("three")

    def test_host_port_leases_prevent_cross_broker_reuse_until_stop(self) -> None:
        first_tmp = tempfile.TemporaryDirectory()
        second_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(first_tmp.cleanup)
        self.addCleanup(second_tmp.cleanup)
        first = broker.RemoteBrowserBroker(
            config=broker.BrokerConfig(
                runtime_root=Path(first_tmp.name) / "runtime",
                installation_id="port-owner-a",
                cdp_port_range=(33000, 33000),
                novnc_port_range=(43000, 43001),
            ),
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        second = broker.RemoteBrowserBroker(
            config=broker.BrokerConfig(
                runtime_root=Path(second_tmp.name) / "runtime",
                installation_id="port-owner-b",
                cdp_port_range=(33000, 33000),
                novnc_port_range=(43000, 43001),
            ),
            launcher=FakeLauncher(),  # type: ignore[arg-type]
            controller=FakeController(),  # type: ignore[arg-type]
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        for item in (first, second):
            self.addCleanup(
                lambda candidate=item: (
                    candidate.begin_shutdown(),
                    candidate.release_shutdown_leases(),
                )
            )

        first.ensure("first")
        with self.assertRaises(broker.BrowserBrokerError):
            second.ensure("second")
        first.stop("first")
        second.ensure("second")

    def test_screenshot_count_quota_blocks_capture_before_controller_call(self) -> None:
        remote_broker, _, controller = self.make_broker()
        remote_broker.ensure("mcp-a")
        with patch.object(broker, "MAX_SCREENSHOT_ARTIFACTS", 1):
            remote_broker.screenshot("mcp-a")
            with self.assertRaises(broker.BrowserBrokerError):
                remote_broker.screenshot("mcp-a")
        self.assertEqual(len(controller.screenshots), 1)

    def test_operation_parameter_budget_rejects_oversized_page_operation(self) -> None:
        remote_broker, _, controller = self.make_broker()
        remote_broker.ensure("mcp-a")
        expression = "x" * (broker.MAX_OPERATION_PARAMETER_BYTES + 1)

        with self.assertRaises(broker.BrowserBrokerError) as error:
            remote_broker.evaluate("mcp-a", expression=expression)

        self.assertEqual(error.exception.code, broker.ERROR_INVALID_ARGUMENT)
        self.assertEqual(controller.evaluations, [])

    def test_stdio_rejects_oversized_jsonrpc_line(self) -> None:
        remote_broker, _, _ = self.make_broker()
        output = io.StringIO()
        status = broker.serve(
            input_stream=io.StringIO("x" * (broker.MAX_MCP_REQUEST_BYTES + 1) + "\n"),
            output_stream=output,
            server=broker.MCPServer(remote_broker),
        )
        response = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(response["error"]["code"], -32600)
        self.assertIn("exceeds", response["error"]["message"])

    def test_http_get_mcp_opens_event_stream(self) -> None:
        _, port, _, _, _ = self.make_http_server()

        with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
            sock.settimeout(2)
            sock.sendall(
                b"GET /mcp HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Accept: text/event-stream\r\n"
                b"mcp-session-id: mcp-a\r\n"
                b"\r\n"
            )
            chunks = []
            while b": connected" not in b"".join(chunks):
                chunks.append(sock.recv(2048))
            response = b"".join(chunks)

        self.assertIn(b" 200 ", response)
        self.assertIn(b"Content-Type: text/event-stream", response)
        self.assertIn(b"mcp-session-id: mcp-a", response)
        self.assertIn(b": connected", response)

    def test_http_delete_mcp_retains_current_profile_without_operator(self) -> None:
        _, port, remote_broker, launcher, _ = self.make_http_server()
        created = remote_broker.ensure("mcp-a")
        remote_broker.pause_for_human("mcp-a")
        profile_root = Path(str(launcher.started[0]["profile_dir"])).parent

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("DELETE", "/mcp", headers={"mcp-session-id": "mcp-a"})
        response = conn.getresponse()
        response.read()
        conn.close()

        self.assertEqual(response.status, 202)
        self.assertEqual(launcher.stopped, [launcher.started[0]["container_name"]])
        self.assertEqual(remote_broker.whoami("mcp-a")["browser_session"]["state"], broker.STATE_STOPPED_RETAINED)
        self.assertTrue(profile_root.exists())
        remote_broker.recover_session("mcp-new", recovery_token=created["recovery_token"])
        restored = remote_broker.ensure("mcp-new")
        self.assertEqual(restored["browser_session"]["id"], created["browser_session"]["id"])


if __name__ == "__main__":
    unittest.main()
