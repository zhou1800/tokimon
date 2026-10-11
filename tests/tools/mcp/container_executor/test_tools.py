from __future__ import annotations

import base64
import io
import json
import logging
import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from tokimon.tools.mcp.container_executor.config import default_config, worker_config
from tokimon.tools.mcp.container_executor.server import ExecutorServer, SocketServer


class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.readonly = self.root / "readonly"
        self.readonly.mkdir()
        config = worker_config(default_config("test"))
        config["roots"] = {str(self.workspace): False, str(self.readonly): True}
        config["limits"].update(output_bytes=1024, concurrency=2, timeout_seconds=3)
        self.log_output = io.StringIO()
        log = logging.getLogger(self.id())
        log.handlers = [logging.StreamHandler(self.log_output)]
        log.setLevel(logging.INFO)
        self.server = ExecutorServer(config, self.root / "state", {}, log=log)

    def call(self, name, **args):
        result = self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})
        self.assertNotIn("error", result)
        return result["result"], json.loads(result["result"]["content"][0]["text"])

    def start(self, command, **args):
        result, payload = self.call("executor_start", command=command, cwd=str(self.workspace), **args)
        self.assertFalse(result.get("isError"), payload)
        return payload["job_id"]

    def wait(self, job):
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            _, status = self.call("executor_status", job_id=job)
            if status["state"] != "running":
                return status
            time.sleep(0.02)
        self.fail("job did not terminate")

    def test_success_failure_output_and_logs_exclude_command(self):
        job = self.start("printf 'private-command-canary'; printf 'bad' >&2; exit 7")
        status = self.wait(job)
        self.assertEqual((status["state"], status["exit_code"]), ("failed", 7))
        _, output = self.call("executor_read", job_id=job, offset=8, max_bytes=7)
        self.assertEqual(output["content"], "command")
        _, error = self.call("executor_read", job_id=job, stream="stderr")
        self.assertEqual(error["content"], "bad")
        self.assertNotIn("private-command-canary", self.log_output.getvalue())
        self.assertEqual(self.wait(self.start("true"))["state"], "succeeded")

    def test_output_capture_is_bounded(self):
        status = self.wait(self.start("head -c 10000 /dev/zero"))
        self.assertEqual(status["stdout_bytes"], 1024)
        self.assertTrue(status["stdout_truncated"])

    def test_timeout_is_terminal(self):
        self.assertEqual(self.wait(self.start("sleep 5", timeout_seconds=1))["state"], "timed_out")

    def test_cancel_is_terminal_and_idempotent(self):
        job = self.start("sleep 5")
        self.call("executor_cancel", job_id=job)
        status = self.wait(job)
        self.assertEqual(status["state"], "cancelled")
        _, cancelled_again = self.call("executor_cancel", job_id=job)
        self.assertEqual(cancelled_again, status)

    def test_pipe_holding_descendant_does_not_hold_job_open(self):
        status = self.wait(self.start("setsid sleep 2 & printf done"))
        self.assertEqual(status["state"], "succeeded")

    def test_concurrency_limit_fails_explicitly(self):
        first, second = self.start("sleep 5"), self.start("sleep 5")
        result, _ = self.call("executor_start", command="true", cwd=str(self.workspace))
        self.assertTrue(result["isError"])
        for job in (first, second):
            self.call("executor_cancel", job_id=job)
            self.wait(job)

    def test_job_count_limit_fails_explicitly(self):
        config = self.server.config
        config["limits"]["max_jobs"] = 2
        self.server = ExecutorServer(config, self.root / "quota-state", {}, log=self.server.log)
        self.wait(self.start("true"))
        self.wait(self.start("true"))
        result, payload = self.call("executor_start", command="true", cwd=str(self.workspace))
        self.assertTrue(result["isError"])
        self.assertIn("job-count", payload["error"])

    def test_failed_monitor_start_kills_reaps_and_releases_slot(self):
        with mock.patch("threading.Thread.start", side_effect=RuntimeError("thread limit exhausted")):
            result, _ = self.call("executor_start", command="sleep 5", cwd=str(self.workspace))
        self.assertTrue(result["isError"])
        _, jobs = self.call("executor_jobs")
        self.assertEqual(jobs["jobs"][0]["state"], "failed")
        self.assertLess(jobs["jobs"][0]["exit_code"], 0)
        self.assertEqual(self.wait(self.start("true"))["state"], "succeeded")

    def test_corrupted_output_and_metadata_fifo_fail_without_blocking(self):
        job = self.start("true")
        self.wait(job)
        stdout = self.root / "state" / job / "stdout"
        stdout.unlink()
        os.mkfifo(stdout)
        started = time.monotonic()
        result, _ = self.call("executor_read", job_id=job, max_bytes=1)
        self.assertTrue(result["isError"])
        self.assertLess(time.monotonic() - started, 0.5)
        metadata = stdout.parent / "job.json"
        metadata.unlink()
        os.mkfifo(metadata)
        with self.assertRaisesRegex(ValueError, "regular"):
            ExecutorServer(self.server.config, self.root / "state", {}, log=self.server.log)

    def test_maintenance_atomically_fences_job_admission(self):
        result = self.server.handle({"id": 1, "method": "tokimon/maintenance", "params": {"paused": True}})
        self.assertTrue(result["result"]["paused"])
        result, _ = self.call("executor_start", command="true", cwd=str(self.workspace))
        self.assertTrue(result["isError"])
        self.server.handle({"id": 2, "method": "tokimon/maintenance", "params": {"paused": False}})
        job = self.start("sleep 5")
        result = self.server.handle({"id": 3, "method": "tokimon/maintenance", "params": {"paused": True}})
        self.assertIn("active jobs", result["error"]["message"])
        self.call("executor_cancel", job_id=job)
        self.wait(job)

    def test_request_and_job_logs_preserve_correlation_without_command(self):
        correlation = "a" * 32
        response = self.server.handle({"id": 1, "method": "tools/call", "params": {"name": "executor_start", "arguments": {"command": "printf secret-marker", "cwd": str(self.workspace)}}}, correlation_id=correlation)
        job = json.loads(response["result"]["content"][0]["text"])
        self.wait(job["job_id"])
        self.assertEqual(job["request_id"], correlation)
        self.assertNotIn("secret-marker", self.log_output.getvalue())

    def test_persistence_recovers_terminal_and_interrupted_without_rerun(self):
        terminal = self.start("printf persisted")
        self.wait(terminal)
        state = self.root / "state" / ("a" * 32)
        state.mkdir()
        record = {"job_id": "a" * 32, "state": "running", "started_at": "2020-01-01", "finished_at": None,
                  "exit_code": None, "stdout_bytes": 0, "stderr_bytes": 0, "stdout_truncated": False, "stderr_truncated": False}
        (state / "job.json").write_text(json.dumps(record))
        recovered = ExecutorServer(self.server.config, self.root / "state", {}, log=self.server.log)
        response = recovered.call_tool({"name": "executor_jobs"})
        jobs = json.loads(response["content"][0]["text"])["jobs"]
        self.assertEqual({j["job_id"]: j["state"] for j in jobs}, {terminal: "succeeded", "a" * 32: "interrupted"})

    def test_file_create_overwrite_byte_read_and_base64(self):
        path = str(self.workspace / "binary")
        encoded = base64.b64encode(b"\x00\xffABC").decode()
        result, _ = self.call("executor_file_write", path=path, content=encoded, encoding="base64")
        self.assertFalse(result.get("isError"))
        result, _ = self.call("executor_file_write", path=path, content="replace")
        self.assertTrue(result["isError"])
        _, payload = self.call("executor_file_read", path=path, offset=1, max_bytes=2, encoding="base64")
        self.assertEqual(base64.b64decode(payload["content"]), b"\xffA")
        self.assertEqual(payload["next_offset"], 3)
        self.call("executor_file_write", path=path, content="ok", overwrite=True)
        self.assertEqual((self.workspace / "binary").read_text(), "ok")

    def test_file_symlink_special_parent_escape_and_readonly_denied(self):
        outside = self.root / "outside"
        outside.write_text("secret")
        (self.workspace / "link").symlink_to(outside)
        (self.workspace / "dirlink").symlink_to(self.root, target_is_directory=True)
        os.mkfifo(self.workspace / "pipe")
        for path in (str(self.workspace / "link"), str(self.workspace / "dirlink/outside"), str(self.workspace / "pipe"), str(self.workspace / "../outside"), str(outside)):
            with self.subTest(path=path):
                result, _ = self.call("executor_file_read", path=path)
                self.assertTrue(result["isError"])
        result, _ = self.call("executor_file_write", path=str(self.readonly / "new"), content="x")
        self.assertTrue(result["isError"])
        self.assertFalse((self.readonly / "new").exists())
        _, listed = self.call("executor_list", path=str(self.workspace), limit=1)
        self.assertTrue(listed["truncated"])

    def test_validation_rejects_unknown_wrong_types_and_bounds(self):
        for name, args in [("executor_start", {"command": "true", "env": {}}), ("executor_jobs", {"limit": True}),
                           ("executor_read", {"job_id": "../../", "offset": -1}), ("executor_file_write", {"path": str(self.workspace / "large"), "content": "x" * 1048577}),
                           ("executor_file_read", {"path": str(self.workspace / "x"), "max_bytes": 1048577}), ("executor_health", {"extra": 1})]:
            with self.subTest(name=name):
                result, _ = self.call(name, **args)
                self.assertTrue(result["isError"])

    def test_concurrent_leaf_replacement_never_follows_external_symlink(self):
        outside = self.root / "unmounted-secret"
        outside.write_text("must-never-be-read")
        target = self.workspace / "racing-file"
        target.write_text("allowed")
        stopping = threading.Event()
        def replace():
            while not stopping.is_set():
                temporary = self.workspace / "replacement"
                temporary.symlink_to(outside)
                temporary.replace(target)
                temporary.write_text("allowed")
                temporary.replace(target)
        thread = threading.Thread(target=replace)
        thread.start()
        try:
            for _ in range(100):
                result, payload = self.call("executor_file_read", path=str(target))
                if not result.get("isError"):
                    self.assertEqual(payload["content"], "allowed")
        finally:
            stopping.set()
            thread.join(2)

    def test_socket_mcp_reconnect_and_bounded_frames(self):
        path = self.root / "socket"
        with SocketServer(str(path), self.server) as service:
            thread = threading.Thread(target=service.serve_forever, daemon=True)
            thread.start()
            def send(request):
                with socket.socket(socket.AF_UNIX) as connection:
                    connection.settimeout(3)
                    connection.connect(str(path))
                    connection.sendall(json.dumps(request).encode() + b"\n")
                    with connection.makefile("rb") as source:
                        return json.loads(source.readline())
            request = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "executor_start", "arguments": {"command": "sleep 0.1; printf reconnected", "cwd": str(self.workspace)}}}
            result = send(request)
            job = json.loads(result["result"]["content"][0]["text"])["job_id"]
            self.wait(job)
            result = send({"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "executor_read", "arguments": {"job_id": job}}})
            self.assertEqual(json.loads(result["result"]["content"][0]["text"])["content"], "reconnected")
            with socket.socket(socket.AF_UNIX) as connection:
                connection.settimeout(3)
                connection.connect(str(path))
                connection.sendall(b"x" * (2 * 1024 * 1024 + 1))
                self.assertEqual(connection.recv(1), b"")
            service.shutdown()
            thread.join(2)
