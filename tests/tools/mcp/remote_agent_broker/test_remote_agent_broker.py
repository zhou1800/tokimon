from __future__ import annotations

import fcntl
import io
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tokimon.tools.mcp import remote_agent_broker as broker


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


class RequestHeaders(dict[str, str]):
    def get(self, key: str, default: str | None = None) -> str | None:
        return super().get(key.lower(), default)

    def get_all(self, key: str, default: object = None):
        value = self.get(key)
        return default if value is None else [value]


class DisconnectingWriter(io.BytesIO):
    """Capture the initial event-stream frame, then model a disconnected client."""

    def flush(self) -> None:
        if self.getvalue():
            raise BrokenPipeError


def invoke_handler(
    server: broker.MCPServer,
    *,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = RequestHeaders(
        {"host": "127.0.0.1:8938", "content-type": "application/json",
         **{str(key).lower(): str(value) for key, value in (headers or {}).items()}}
    )
    if body:
        request_headers["content-length"] = str(len(body))
    base_handler = broker.make_handler(server)

    class InMemoryHandler(base_handler):
        def __init__(self) -> None:
            pass

        def send_response(self, code: int, message: str | None = None) -> None:
            self.response_status = code
            self.log_request(code)

        def send_header(self, keyword: str, value: str) -> None:
            self.response_headers[keyword.lower()] = value

        def end_headers(self) -> None:
            return

        def send_error(self, code: int, message: str | None = None) -> None:
            self.response_status = code

    handler = InMemoryHandler()
    handler.command = method
    handler.path = path
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 12345)
    handler.server = SimpleNamespace(server_address=("127.0.0.1", 8938))
    handler.headers = request_headers
    handler.rfile = io.BytesIO(body)
    handler.wfile = DisconnectingWriter() if method == "GET" and path == "/mcp" else io.BytesIO()
    handler.response_status = 0
    handler.response_headers: dict[str, str] = {}
    getattr(handler, f"do_{method}")()
    return handler.response_status, handler.response_headers, handler.wfile.getvalue()


def write_fake_cli(bin_dir: Path, name: str) -> Path:
    path = bin_dir / name
    path.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import json",
                "import sys",
                "if '--version' in sys.argv:",
                f"    print('{name} fake 1.0')",
                "    raise SystemExit(0)",
                "stdin = sys.stdin.read()",
                f"print('{name.upper()} ARGS=' + json.dumps(sys.argv[1:]))",
                "if stdin:",
                "    print('STDIN=' + stdin.strip())",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def write_settings(root: Path) -> None:
    path = root / ".tokimon" / "settings.local.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "agents": {
                    "claude": {"model": "claude-config", "effort": "max"},
                },
                "agent_execution": {
                    "idle_timeout_sec": 3600,
                    "poll_interval_sec": 0.05,
                    "max_wall_time_sec": None,
                },
            }
        ),
        encoding="utf-8",
    )


class RemoteAgentBrokerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capture_runtime_logger()

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
        with_settings: bool = True,
        monotonic_scale: float = 1.0,
    ) -> tuple[broker.RemoteAgentBroker, tempfile.TemporaryDirectory[str], Path, Path]:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        if with_settings:
            write_settings(root)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        for name in ("copilot", "claude"):
            write_fake_cli(bin_dir, name)
        config = broker.BrokerConfig(
            workspace_root=root,
            runtime_root=root / "runtime" / "remote-agent",
            shell_command="/bin/sh",
            shell_sync_timeout_sec=5,
        )
        remote_broker = broker.RemoteAgentBroker(
            config=config,
            env={"PATH": os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")])},
            monotonic_clock=lambda: time.monotonic() * monotonic_scale,
        )
        return remote_broker, tmpdir, root, bin_dir

    def start_http_server(
        self,
        remote_broker: broker.RemoteAgentBroker,
        *,
        capture_logs: bool = True,
    ) -> broker.MCPServer:
        if capture_logs:
            self.capture_runtime_logger()
        return broker.MCPServer(remote_broker)

    def schema_probe_server(self) -> broker.MCPServer:
        probe = Mock()
        for method_name in (
            "health",
            "list_agents",
            "start_agent",
            "status",
            "read",
            "send_input",
            "cancel",
            "diff",
            "artifacts",
            "start_shell",
        ):
            getattr(probe, method_name).return_value = {"ok": True}
        return broker.MCPServer(probe)

    def valid_tool_arguments(self) -> dict[str, dict[str, object]]:
        return {
            "remote_agent_health": {},
            "remote_agent_list_agents": {},
            "remote_agent_start": {
                "agent": "claude",
                "cwd": "/workspace",
                "prompt": "probe",
            },
            "remote_agent_status": {"job_id": "job-probe"},
            "remote_agent_read": {"job_id": "job-probe"},
            "remote_agent_send_input": {
                "job_id": "job-probe",
                "text": "probe",
            },
            "remote_agent_cancel": {"job_id": "job-probe"},
            "remote_agent_diff": {"job_id": "job-probe"},
            "remote_agent_artifacts": {"job_id": "job-probe"},
            "remote_shell_start": {"command": "true"},
        }

    def schema_probe_result(
        self,
        server: broker.MCPServer,
        *,
        tool_name: str,
        arguments: object,
    ) -> dict[str, object]:
        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": tool_name,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
        )
        assert response is not None
        result = response["result"]
        assert isinstance(result, dict)
        return result

    def post_mcp(
        self,
        server: broker.MCPServer,
        payload: dict[str, object],
        *,
        session_id: str | None = None,
    ) -> tuple[int, dict[str, str], dict[str, object]]:
        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["mcp-session-id"] = session_id
        status, response_headers, body = invoke_handler(
            server,
            method="POST",
            path="/mcp",
            payload=payload,
            headers=headers,
        )
        return status, response_headers, parse_sse_payload(body)

    def call_tool_http(
        self,
        server: broker.MCPServer,
        name: str,
        arguments: dict[str, object],
        *,
        request_id: object = 1,
        session_id: str | None = None,
    ) -> tuple[dict[str, object], dict[str, str]]:
        status, headers, payload = self.post_mcp(
            server,
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

    def get_json(
        self,
        server: broker.MCPServer,
        path: str,
    ) -> tuple[int, dict[str, str], dict[str, object]]:
        status, headers, body = invoke_handler(server, method="GET", path=path)
        payload = json.loads(body.decode("utf-8"))
        self.assertIsInstance(payload, dict)
        return status, headers, payload

    def wait_for_status(
        self,
        remote_broker: broker.RemoteAgentBroker,
        job_id: str,
        status: str,
        *,
        timeout_sec: float = 5.0,
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout_sec
        last: dict[str, object] | None = None
        while time.monotonic() < deadline:
            last = remote_broker.status(job_id)
            job = last["job"]
            assert isinstance(job, dict)
            if job["status"] == status:
                if status in {"completed", "failed", "cancelled"}:
                    final = remote_broker.read(job_id=job_id, stream="final")
                    if not final["text"]:
                        time.sleep(0.01)
                        continue
                return last
            time.sleep(0.01)
        raise AssertionError(f"Job did not reach {status}: {last}")

    def seed_persisted_job(
        self,
        remote_broker: broker.RemoteAgentBroker,
        *,
        cwd: Path,
        status: str,
        updated_at: str | None = None,
        ttl_days: int | None = None,
    ) -> tuple[str, Path]:
        job_id, job_dir = remote_broker.artifact_manager.allocate_job()
        job = remote_broker.job_store.initial_job(
            job_id=job_id,
            kind="shell",
            agent="shell",
            cwd=cwd,
            command=[sys.executable],
            runner_options={},
            prompt=None,
            ttl_days=ttl_days,
            capture_limit_bytes=None,
            timeout_sec=5,
            idle_timeout_sec=None,
            max_wall_time_sec=None,
            artifacts=remote_broker.artifact_manager.artifact_paths(job_id),
            request_context=broker.RequestContext.direct_job(job_id),
        )
        job["status"] = status
        if updated_at is not None:
            job["updated_at"] = updated_at
        remote_broker.job_store.write(job_dir, job)
        return job_id, job_dir

    def test_tools_list_contract(self) -> None:
        remote_broker, _, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)

        response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

        self.assertIsNotNone(response)
        result = response["result"]
        self.assertIsInstance(result, dict)
        tools = {tool["name"]: tool for tool in result["tools"]}
        names = set(tools)
        self.assertEqual(
            names,
            {
                "remote_agent_health",
                "remote_agent_list_agents",
                "remote_agent_start",
                "remote_agent_status",
                "remote_agent_read",
                "remote_agent_send_input",
                "remote_agent_cancel",
                "remote_agent_diff",
                "remote_agent_artifacts",
                "remote_shell_start",
            },
        )
        self.assertEqual(
            tools["remote_agent_start"]["inputSchema"]["properties"]["agent"]["enum"],
            ["copilot", "claude"],
        )

    def test_mcp_rejects_unsupported_agents_before_allocating_artifacts(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)

        for unsupported_agent in ("codex", "unknown"):
            with self.subTest(agent=unsupported_agent):
                response = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": unsupported_agent,
                        "method": "tools/call",
                        "params": {
                            "name": "remote_agent_start",
                            "arguments": {
                                "agent": unsupported_agent,
                                "cwd": str(root),
                                "prompt": "must be rejected",
                            },
                        },
                    }
                )
                assert response is not None
                self.assertTrue(response["result"]["isError"])

        self.assertFalse(remote_broker.jobs_root.exists())

    def test_each_declared_tool_accepts_its_minimal_schema_shape(self) -> None:
        server = self.schema_probe_server()

        for tool_name, arguments in self.valid_tool_arguments().items():
            with self.subTest(tool=tool_name):
                result = self.schema_probe_result(
                    server,
                    tool_name=tool_name,
                    arguments=arguments,
                )
                self.assertNotIn("isError", result)

    def test_each_declared_tool_rejects_nonobject_argument_roots(self) -> None:
        server = self.schema_probe_server()

        for tool_name in self.valid_tool_arguments():
            for arguments in (None, [], "invalid"):
                with self.subTest(tool=tool_name, arguments=arguments):
                    result = self.schema_probe_result(
                        server,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                    self.assertTrue(result["isError"])
                    self.assertIn(
                        "arguments must be an object",
                        tool_payload({"result": result})["error"],
                    )

    def test_argument_omission_defaults_to_an_empty_object(self) -> None:
        server = self.schema_probe_server()

        for tool_name in ("remote_agent_health", "remote_agent_list_agents"):
            with self.subTest(tool=tool_name):
                response = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": tool_name,
                        "method": "tools/call",
                        "params": {"name": tool_name},
                    }
                )
                assert response is not None
                self.assertNotIn("isError", response["result"])

    def test_each_declared_tool_rejects_undeclared_arguments(self) -> None:
        server = self.schema_probe_server()

        for tool_name, arguments in self.valid_tool_arguments().items():
            with self.subTest(tool=tool_name):
                result = self.schema_probe_result(
                    server,
                    tool_name=tool_name,
                    arguments={**arguments, "unexpected": True},
                )
                self.assertTrue(result["isError"])
                self.assertIn("unsupported field", tool_payload({"result": result})["error"])

    def test_each_declared_tool_enforces_required_fields(self) -> None:
        server = self.schema_probe_server()
        declarations = {item["name"]: item for item in broker.mcp_tools()}

        for tool_name, arguments in self.valid_tool_arguments().items():
            schema = declarations[tool_name]["inputSchema"]
            for required_name in schema.get("required", []):
                with self.subTest(tool=tool_name, required=required_name):
                    invalid = dict(arguments)
                    invalid.pop(required_name)
                    result = self.schema_probe_result(
                        server,
                        tool_name=tool_name,
                        arguments=invalid,
                    )
                    self.assertTrue(result["isError"])
                    self.assertIn(
                        f"{required_name} is required",
                        tool_payload({"result": result})["error"],
                    )

    def test_each_declared_property_enforces_its_json_type(self) -> None:
        server = self.schema_probe_server()
        declarations = {item["name"]: item for item in broker.mcp_tools()}
        invalid_by_type: dict[str, object] = {
            "string": 1,
            "boolean": "true",
            "integer": "1",
            "object": [],
        }

        for tool_name, base_arguments in self.valid_tool_arguments().items():
            properties = declarations[tool_name]["inputSchema"].get("properties", {})
            for property_name, property_schema in properties.items():
                expected_type = property_schema.get("type")
                invalid_value = (
                    invalid_by_type.get(expected_type, {})
                    if "oneOf" not in property_schema
                    else {}
                )
                with self.subTest(tool=tool_name, property=property_name):
                    result = self.schema_probe_result(
                        server,
                        tool_name=tool_name,
                        arguments={**base_arguments, property_name: invalid_value},
                    )
                    self.assertTrue(result["isError"])

    def test_each_declared_enum_and_minimum_is_enforced(self) -> None:
        server = self.schema_probe_server()
        declarations = {item["name"]: item for item in broker.mcp_tools()}

        for tool_name, base_arguments in self.valid_tool_arguments().items():
            properties = declarations[tool_name]["inputSchema"].get("properties", {})
            for property_name, property_schema in properties.items():
                if "enum" in property_schema:
                    with self.subTest(tool=tool_name, enum=property_name):
                        result = self.schema_probe_result(
                            server,
                            tool_name=tool_name,
                            arguments={
                                **base_arguments,
                                property_name: "outside-declared-enum",
                            },
                        )
                        self.assertTrue(result["isError"])
                if "minimum" in property_schema:
                    with self.subTest(tool=tool_name, minimum=property_name):
                        result = self.schema_probe_result(
                            server,
                            tool_name=tool_name,
                            arguments={
                                **base_arguments,
                                property_name: property_schema["minimum"] - 1,
                            },
                        )
                        self.assertTrue(result["isError"])

    def test_mcp_rejects_nested_schema_type_mismatches_before_allocation(self) -> None:
        remote_broker, _, _, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)
        cases = (
            (
                "boolean-is-not-integer",
                "remote_shell_start",
                {"command": "true", "timeout_sec": True},
                "must be an integer",
            ),
            (
                "float-is-not-integer",
                "remote_shell_start",
                {"command": "true", "timeout_sec": 1.5},
                "must be an integer",
            ),
            (
                "non-string-env",
                "remote_shell_start",
                {"command": "true", "env": {"COUNT": 2}},
                "must be a string",
            ),
            (
                "non-string-argv-item",
                "remote_shell_start",
                {"command": ["true", 2]},
                "must be a string",
            ),
        )

        for request_id, tool_name, arguments, expected_error in cases:
            with self.subTest(request_id=request_id):
                response = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "tools/call",
                        "params": {"name": tool_name, "arguments": arguments},
                    }
                )
                assert response is not None
                self.assertTrue(response["result"]["isError"])
                self.assertIn(expected_error, tool_payload(response)["error"])

        self.assertFalse(remote_broker.jobs_root.exists())

    def test_stdio_serve_ping_contract_and_structured_logs(self) -> None:
        log_stream = self.capture_runtime_logger()
        remote_broker, _, _, _ = self.make_broker()
        input_stream = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
        output_stream = io.StringIO()

        status = broker.serve(
            input_stream=input_stream,
            output_stream=output_stream,
            server=broker.MCPServer(remote_broker),
        )

        self.assertEqual(status, 0)
        self.assertEqual(
            json.loads(output_stream.getvalue()),
            {"jsonrpc": "2.0", "id": 1, "result": {}},
        )
        self.assertIn('"runtime": "stdio-mcp"', log_stream.getvalue())
        self.assertNotIn("server_start", output_stream.getvalue())

    def test_health_reports_all_first_version_runners(self) -> None:
        remote_broker, _, _, _ = self.make_broker()

        payload = remote_broker.health()

        self.assertTrue(payload["ok"])
        agents = {item["agent"]: item for item in payload["agents"]}
        self.assertEqual(set(agents), {"shell", "copilot", "claude"})
        self.assertTrue(agents["copilot"]["available"])
        self.assertTrue(agents["claude"]["available"])
        self.assertEqual(
            payload["agent_settings"],
            {"ok": True, "model_settings_required": False},
        )

    def test_health_needs_no_provider_model_settings(self) -> None:
        remote_broker, _, _, _ = self.make_broker(with_settings=False)

        payload = remote_broker.health()

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["agent_settings"]["ok"])

    def test_health_rejects_non_executable_runner_overrides(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        write_settings(root)
        override = root / "provider-cli"
        override.write_text("#!/bin/sh\nprintf 'provider test\\n'\n", encoding="utf-8")
        override.chmod(0o644)
        config = broker.BrokerConfig(
            workspace_root=root,
            runtime_root=root / "runtime" / "remote-agent",
            shell_command="/bin/sh",
            shell_sync_timeout_sec=5,
        )
        remote_broker = broker.RemoteAgentBroker(
            config=config,
            env={
                "PATH": "",
                "COPILOT_CLI": str(override),
                "CLAUDE_CODE_CLI": str(override),
            },
        )

        payload = remote_broker.health()

        self.assertTrue(payload["ok"])
        agents = {item["agent"]: item for item in payload["agents"]}
        self.assertTrue(agents["shell"]["available"])
        self.assertFalse(agents["copilot"]["available"])
        self.assertFalse(agents["claude"]["available"])
        self.assertTrue(payload["agent_settings"]["ok"])

    def test_shell_start_sync_writes_artifacts_and_output(self) -> None:
        remote_broker, _, root, _ = self.make_broker()

        payload = remote_broker.start_shell(
            command=[sys.executable, "-c", "print('shell-ok')"],
            cwd=str(root),
            timeout_sec=5,
        )

        job = payload["job"]
        self.assertEqual(job["status"], "completed")
        job_id = str(job["job_id"])
        stdout = remote_broker.read(job_id=job_id, stream="stdout")
        self.assertIn("shell-ok", stdout["text"])
        artifacts = remote_broker.artifacts(job_id=job_id)["artifacts"]
        self.assertIn("job.json", {item["name"] for item in artifacts})
        self.assertIn("final.json", {item["name"] for item in artifacts})
        self.assertIn("transcript.md", {item["name"] for item in artifacts})
        transcript = remote_broker.read(job_id=job_id, stream="transcript")["text"]
        self.assertIn("## Output", transcript)
        self.assertIn("shell-ok", transcript)

    def test_stream_logs_redact_multi_character_secret_patterns(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        token = "sk-" + ("a" * 24)

        with patch.object(broker, "STREAM_READ_CHARS", 5):
            payload = remote_broker.start_shell(
                command=[
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        f"print('api_key={token}'); "
                        f"print('token={token}', file=sys.stderr)"
                    ),
                ],
                cwd=str(root),
                timeout_sec=5,
            )
        job_id = str(payload["job"]["job_id"])

        stdout = remote_broker.read(job_id=job_id, stream="stdout")
        stderr = remote_broker.read(job_id=job_id, stream="stderr")

        self.assertNotIn(token, stdout["text"])
        self.assertNotIn(token, stderr["text"])
        self.assertIn("api_key=<redacted>", stdout["text"])
        self.assertIn("token=<redacted>", stderr["text"])

    def test_mcp_remote_shell_start_dispatches(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "remote_shell_start",
                    "arguments": {
                        "command": [sys.executable, "-c", "print('mcp-shell-ok')"],
                        "cwd": str(root),
                        "timeout_sec": 5,
                    },
                },
            },
        )

        payload = tool_payload(response)
        self.assertEqual(payload["job"]["status"], "completed")
        stdout = remote_broker.read(job_id=payload["job"]["job_id"], stream="stdout")
        self.assertIn("mcp-shell-ok", stdout["text"])

    def test_http_mcp_transport_contract_and_structured_logs(self) -> None:
        log_stream = self.capture_runtime_logger()
        remote_broker, _, _, _ = self.make_broker()
        server = self.start_http_server(remote_broker, capture_logs=False)

        health_status, health_headers, health = self.get_json(server, "/health")
        list_status, list_headers, list_payload = self.post_mcp(
            server,
            {"jsonrpc": "2.0", "id": "tools", "method": "tools/list"},
        )
        session_id = list_headers["mcp-session-id"]
        stream_status, _, stream_body = invoke_handler(
            server,
            method="GET",
            path="/mcp",
            headers={"mcp-session-id": session_id},
        )
        delete_status, delete_headers, _ = invoke_handler(
            server,
            method="DELETE",
            path="/mcp",
            headers={"mcp-session-id": session_id},
        )

        self.assertEqual(health_status, 200)
        self.assertEqual(health_headers["content-type"], "application/json; charset=utf-8")
        self.assertTrue(health["ok"])
        self.assertEqual(list_status, 200)
        self.assertEqual(list_payload["id"], "tools")
        self.assertIn("tools", list_payload["result"])
        self.assertTrue(session_id.startswith("mcp_"))
        self.assertEqual(stream_status, 200)
        self.assertEqual(stream_body.splitlines()[0], b": connected")
        self.assertEqual(delete_status, 202)
        self.assertEqual(delete_headers["mcp-session-id"], session_id)

        entries = [json.loads(line) for line in log_stream.getvalue().splitlines()]
        self.assertTrue(
            any(item["message"] == "health_requested" and item["runtime"] == "http-mcp" for item in entries)
        )
        self.assertTrue(
            any(
                item["message"] == "request_received"
                and item["request_id"] == "tools"
                and item["method"] == "tools/list"
                and item["correlation_id"] == session_id
                for item in entries
            )
        )
        self.assertTrue(any(item["message"] == "event_stream_opened" for item in entries))
        self.assertTrue(any(item["message"] == "session_shutdown" for item in entries))
        self.assertTrue(any(item["message"] == "http_request" for item in entries))

    def test_http_agent_status_read_artifacts_and_correlation(self) -> None:
        log_stream = self.capture_runtime_logger()
        remote_broker, _, root, _ = self.make_broker()
        server = self.start_http_server(remote_broker, capture_logs=False)

        agent_payload, headers = self.call_tool_http(
            server,
            "remote_agent_start",
            {
                "agent": "claude",
                "cwd": str(root),
                "prompt": "http agent",
                "timeout_sec": 5,
            },
            request_id="start-agent",
        )
        session_id = headers["mcp-session-id"]
        agent_job_id = str(agent_payload["job"]["job_id"])
        self.assertEqual(
            agent_payload["job"]["request_context"],
            {
                "correlation_id": session_id,
                "request_id": "start-agent",
                "transport": "http-mcp",
            },
        )
        self.wait_for_status(remote_broker, agent_job_id, "completed")

        status_payload, _ = self.call_tool_http(
            server,
            "remote_agent_status",
            {"job_id": agent_job_id},
            request_id="status",
            session_id=session_id,
        )
        read_payload, _ = self.call_tool_http(
            server,
            "remote_agent_read",
            {"job_id": agent_job_id, "stream": "stdout"},
            request_id="read",
            session_id=session_id,
        )
        artifacts_payload, _ = self.call_tool_http(
            server,
            "remote_agent_artifacts",
            {"job_id": agent_job_id},
            request_id="artifacts",
            session_id=session_id,
        )

        self.assertEqual(status_payload["job"]["status"], "completed")
        self.assertIn("CLAUDE", read_payload["text"])
        self.assertIn("job.json", {item["name"] for item in artifacts_payload["artifacts"]})
        job_events = [
            json.loads(line)
            for line in log_stream.getvalue().splitlines()
            if '"runtime": "broker"' in line
        ]
        self.assertEqual(
            {entry["message"] for entry in job_events},
            {"job_accepted", "job_completed"},
        )
        self.assertTrue(
            all(
                entry["correlation_id"] == session_id
                and entry["request_id"] == "start-agent"
                and entry["job_id"] == agent_job_id
                for entry in job_events
            )
        )

    def test_http_shell_input_lifecycle(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        server = self.start_http_server(remote_broker)

        shell_payload, _ = self.call_tool_http(
            server,
            "remote_shell_start",
            {
                "command": [
                    sys.executable,
                    "-c",
                    "import sys; line=sys.stdin.readline(); print('HTTP-ECHO:' + line.strip(), flush=True)",
                ],
                "cwd": str(root),
                "async": True,
                "timeout_sec": 5,
            },
            request_id="shell",
        )
        shell_job_id = str(shell_payload["job"]["job_id"])
        self.call_tool_http(
            server,
            "remote_agent_send_input",
            {"job_id": shell_job_id, "text": "ping\n"},
            request_id="send",
        )
        self.wait_for_status(remote_broker, shell_job_id, "completed")
        shell_stdout, _ = self.call_tool_http(
            server,
            "remote_agent_read",
            {"job_id": shell_job_id, "stream": "stdout"},
            request_id="shell-read",
        )
        self.assertIn("HTTP-ECHO:ping", shell_stdout["text"])

    def test_http_shell_cancel_lifecycle(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        server = self.start_http_server(remote_broker)

        cancel_payload, _ = self.call_tool_http(
            server,
            "remote_shell_start",
            {
                "command": [sys.executable, "-c", "import sys; sys.stdin.read()"],
                "cwd": str(root),
                "async": True,
                "timeout_sec": 5,
            },
            request_id="cancel-shell",
        )
        cancel_job_id = str(cancel_payload["job"]["job_id"])
        cancelled, _ = self.call_tool_http(
            server,
            "remote_agent_cancel",
            {"job_id": cancel_job_id},
            request_id="cancel",
        )
        self.assertEqual(cancelled["job"]["status"], "cancelled")

    def test_http_diff_dispatches_base_ref(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        server = self.start_http_server(remote_broker)
        completed = remote_broker.start_shell(
            command=[sys.executable, "-c", "print('diff-ready')"],
            cwd=str(root),
            timeout_sec=5,
        )
        job_id = str(completed["job"]["job_id"])

        with patch(
            "tokimon.tools.mcp.remote_agent_broker.capture_git_diff",
            return_value={"available": True, "diff": "diff --git a/example b/example\n"},
        ) as capture:
            diff_payload, _ = self.call_tool_http(
                server,
                "remote_agent_diff",
                {"job_id": job_id, "base_ref": "HEAD~1"},
                request_id="diff",
            )

        self.assertTrue(diff_payload["available"])
        self.assertIn("diff --git", diff_payload["diff"])
        capture.assert_called_once_with(root, base_ref="HEAD~1")

    def test_async_shell_send_input(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        payload = remote_broker.start_shell(
            command=[
                sys.executable,
                "-c",
                "import sys; line=sys.stdin.readline(); print('ECHO:' + line.strip(), flush=True)",
            ],
            cwd=str(root),
            async_job=True,
            timeout_sec=5,
        )
        job_id = str(payload["job"]["job_id"])

        remote_broker.send_input(job_id=job_id, text="ping\n")
        self.wait_for_status(remote_broker, job_id, "completed")

        stdout = remote_broker.read(job_id=job_id, stream="stdout")
        self.assertIn("ECHO:ping", stdout["text"])

    def test_each_noninteractive_agent_rejects_input_without_transcript_change(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        server = broker.MCPServer(remote_broker)

        for agent_name in ("copilot", "claude"):
            with self.subTest(agent=agent_name):
                agent_runner = remote_broker.runners[agent_name]
                slow_spec = broker.CommandSpec(
                    argv=[sys.executable, "-c", "import time; time.sleep(10)"],
                    stdin_text="prompt",
                    close_stdin=True,
                )
                with patch.object(agent_runner, "build", return_value=slow_spec):
                    payload = remote_broker.start_agent(
                        agent=agent_name,
                        cwd=str(root),
                        prompt="non-interactive",
                    )
                job_id = str(payload["job"]["job_id"])
                transcript_before = remote_broker.read(
                    job_id=job_id,
                    stream="transcript",
                )["text"]
                try:
                    response = server.handle(
                        {
                            "jsonrpc": "2.0",
                            "id": f"input-{agent_name}",
                            "method": "tools/call",
                            "params": {
                                "name": "remote_agent_send_input",
                                "arguments": {
                                    "job_id": job_id,
                                    "text": "follow-up\n",
                                },
                            },
                        }
                    )
                    assert response is not None
                    self.assertTrue(response["result"]["isError"])
                    self.assertIn(
                        "does not support interactive input",
                        tool_payload(response)["error"],
                    )
                    transcript_after = remote_broker.read(
                        job_id=job_id,
                        stream="transcript",
                    )["text"]
                    self.assertEqual(transcript_after, transcript_before)
                finally:
                    remote_broker.cancel(job_id=job_id, signal_name="SIGKILL")

    def test_initial_prompt_backpressure_is_bounded_by_idle_and_wall_deadlines(self) -> None:
        for deadline in ("idle", "wall"):
            with self.subTest(deadline=deadline):
                remote_broker, _, root, _ = self.make_broker(monotonic_scale=100.0)
                spec = broker.CommandSpec(
                    argv=[sys.executable, "-c", "import time; time.sleep(10)"],
                    stdin_text="x" * (4 * 1024 * 1024),
                    close_stdin=True,
                )
                returned = threading.Event()
                outcomes: list[object] = []

                def start() -> None:
                    try:
                        outcomes.append(remote_broker.start_agent(
                            agent="claude", cwd=str(root), prompt="blocked prompt",
                            timeout_sec=1 if deadline == "wall" else 60,
                            max_wall_time_sec=1 if deadline == "wall" else 60,
                            idle_timeout_sec=1 if deadline == "idle" else 60,
                        ))
                    except Exception as exc:
                        outcomes.append(exc)
                    finally:
                        returned.set()

                with patch.object(remote_broker.runners["claude"], "build", return_value=spec):
                    starter = threading.Thread(target=start, daemon=True)
                    starter.start()
                    try:
                        self.assertTrue(returned.wait(2), "start blocked while delivering stdin")
                        self.assertEqual(len(outcomes), 1)
                        self.assertIsInstance(outcomes[0], dict)
                        job_id = outcomes[0]["job"]["job_id"]
                        result = self.wait_for_status(remote_broker, job_id, "failed")
                        self.assertEqual(result["job"]["status"], "failed")
                        expected = "idle timed out" if deadline == "idle" else "Job timed out"
                        self.assertIn(expected, result["job"]["last_error"])
                        runtime = remote_broker._runtime(job_id)
                        if runtime is not None:
                            self.assertFalse(runtime.stdin_thread.is_alive())
                    finally:
                        for job_id in list(remote_broker._runtimes):
                            remote_broker.cancel(job_id=job_id, signal_name="SIGKILL")
                        starter.join(3)

    def test_cancel_is_available_while_initial_prompt_delivery_is_blocked(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        entered = threading.Event()
        release = threading.Event()
        returned = threading.Event()
        results: list[dict[str, object]] = []
        popen = broker.subprocess.Popen
        terminate = remote_broker._terminate_process

        def launch(*args: object, **kwargs: object) -> object:
            process = popen(*args, **kwargs)
            if process.stdin is None:
                return process
            write = process.stdin.write

            def blocked_write(text: str) -> int:
                entered.set()
                if not release.wait(5):
                    raise OSError("fixture stdin delivery was not released")
                return write(text)

            process.stdin.write = blocked_write
            return process

        def signal_process(process: object, *, force: bool) -> object:
            result = terminate(process, force=force)
            release.set()
            return result

        spec = broker.CommandSpec(
            argv=[sys.executable, "-c", "import time; time.sleep(10)"],
            stdin_text="blocked", close_stdin=True,
        )
        with (
            patch.object(remote_broker.runners["claude"], "build", return_value=spec),
            patch.object(broker.subprocess, "Popen", side_effect=launch),
            patch.object(remote_broker, "_terminate_process", side_effect=signal_process),
        ):
            started = remote_broker.start_agent(
                agent="claude", cwd=str(root), prompt="prompt",
                idle_timeout_sec=60, max_wall_time_sec=60,
            )
            job_id = started["job"]["job_id"]
            self.assertTrue(entered.wait(2))

            def cancel() -> None:
                try:
                    results.append(remote_broker.cancel(job_id=job_id, signal_name="SIGKILL"))
                finally:
                    returned.set()

            caller = threading.Thread(target=cancel, daemon=True)
            caller.start()
            try:
                self.assertTrue(returned.wait(2), "cancellation blocked behind stdin delivery")
                self.assertEqual(results[0]["job"]["status"], "cancelled")
                self.assertEqual(remote_broker.status(job_id)["job"]["status"], "cancelled")
                self.assertIsNone(remote_broker._runtime(job_id))
            finally:
                release.set()
                remote_broker.cancel(job_id=job_id, signal_name="SIGKILL")
                caller.join(3)

    def test_parent_exit_does_not_block_monitor_on_stalled_initial_stdin(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        entered = threading.Event()
        release = threading.Event()
        popen = broker.subprocess.Popen

        def launch(*args: object, **kwargs: object) -> object:
            process = popen(*args, **kwargs)
            if process.stdin is None:
                return process
            write = process.stdin.write

            def blocked_write(text: str) -> int:
                entered.set()
                release.wait(5)
                return write(text)

            process.stdin.write = blocked_write
            return process

        spec = broker.CommandSpec(
            argv=[sys.executable, "-c", "print('parent exit', flush=True)"],
            stdin_text="blocked", close_stdin=True,
        )
        with (
            patch.object(remote_broker.runners["claude"], "build", return_value=spec),
            patch.object(broker.subprocess, "Popen", side_effect=launch),
            patch.object(broker, "POST_EXIT_STREAM_DRAIN_SEC", 0.01),
            patch.object(broker, "POST_EXIT_PROCESS_GROUP_TERM_SEC", 0.01),
            patch.object(broker, "POST_EXIT_PROCESS_GROUP_KILL_SEC", 0.01),
        ):
            started = remote_broker.start_agent(
                agent="claude", cwd=str(root), prompt="prompt",
                idle_timeout_sec=60, max_wall_time_sec=60,
            )
            job_id = started["job"]["job_id"]
            runtime = remote_broker._runtime(job_id)
            try:
                self.assertTrue(entered.wait(2))
                failed = self.wait_for_status(remote_broker, job_id, "failed", timeout_sec=2)
                self.assertEqual(failed["job"]["exit_code"], 0)
                self.assertIn("stdin capture failed", failed["job"]["last_error"])
                self.assertIn("bounded process-group cleanup", failed["job"]["last_error"])
                runtime.monitor_thread.join(2)
                self.assertFalse(runtime.monitor_thread.is_alive())
            finally:
                release.set()
                runtime.stdin_thread.join(2)
            self.assertFalse(runtime.stdin_thread.is_alive())

    def test_cancel_running_shell_job(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        payload = remote_broker.start_shell(
            command=[sys.executable, "-c", "import sys; sys.stdin.read()"],
            cwd=str(root),
            async_job=True,
            timeout_sec=60,
        )
        job_id = str(payload["job"]["job_id"])

        cancelled = remote_broker.cancel(job_id=job_id)

        self.assertEqual(cancelled["job"]["status"], "cancelled")

    def test_cancel_can_force_process_that_ignores_sigterm(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        payload = remote_broker.start_shell(
            command=[
                sys.executable,
                "-c",
                (
                    "import signal, sys; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "sys.stdin.read()"
                ),
            ],
            cwd=str(root),
            async_job=True,
            timeout_sec=60,
        )
        job_id = str(payload["job"]["job_id"])

        cancelled = remote_broker.cancel(job_id=job_id, signal_name="SIGKILL")

        self.assertEqual(cancelled["job"]["status"], "cancelled")
        self.assertEqual(remote_broker.status(job_id)["job"]["status"], "cancelled")

    def test_cancel_signal_failure_is_visible_and_retryable(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        payload = remote_broker.start_shell(
            command=[sys.executable, "-c", "import sys; sys.stdin.read()"],
            cwd=str(root),
            async_job=True,
            timeout_sec=60,
        )
        job_id = str(payload["job"]["job_id"])
        pid = int(remote_broker.status(job_id)["job"]["pid"])

        def stop_process() -> None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        self.addCleanup(stop_process)
        with patch.object(broker.os, "killpg", side_effect=PermissionError("denied")):
            failed_signal = remote_broker.cancel(job_id=job_id, signal_name="SIGKILL")

        self.assertEqual(failed_signal["job"]["status"], "cancelled")
        self.assertIn("signal delivery failed", str(failed_signal["job"]["last_error"]))
        self.assertIn("PermissionError", str(failed_signal["job"]["cancellation_signal_error"]))
        os.kill(pid, 0)

        retried = remote_broker.cancel(job_id=job_id, signal_name="SIGKILL")

        self.assertEqual(retried["job"]["status"], "cancelled")
        self.assertNotIn("cancellation_signal_error", retried["job"])
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_send_input_waits_for_cancel_and_rejects_terminal_job(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        payload = remote_broker.start_shell(
            command=[sys.executable, "-c", "import sys; sys.stdin.read()"],
            cwd=str(root),
            async_job=True,
            timeout_sec=60,
        )
        job_id = str(payload["job"]["job_id"])
        signal_started = threading.Event()
        release_signal = threading.Event()
        input_started = threading.Event()
        cancel_result: dict[str, object] = {}
        input_errors: list[Exception] = []
        original_killpg = broker.os.killpg
        input_done = threading.Event()

        def blocking_killpg(pid: int, signal_number: int) -> None:
            signal_started.set()
            release_signal.wait(5)
            original_killpg(pid, signal_number)

        def cancel_job() -> None:
            cancel_result.update(remote_broker.cancel(job_id=job_id, signal_name="SIGKILL"))

        def send_late_input() -> None:
            input_started.set()
            try:
                remote_broker.send_input(job_id=job_id, text="late-input\n")
            except Exception as exc:
                input_errors.append(exc)
            finally:
                input_done.set()

        try:
            with patch.object(broker.os, "killpg", side_effect=blocking_killpg):
                cancel_thread = threading.Thread(target=cancel_job)
                cancel_thread.start()
                self.assertTrue(signal_started.wait(5))
                self.assertEqual(remote_broker.job_store.read(job_id)["status"], "cancelled")

                input_thread = threading.Thread(target=send_late_input)
                input_thread.start()
                self.assertTrue(input_started.wait(5))
                self.assertFalse(input_done.is_set())

                release_signal.set()
                cancel_thread.join(5)
                input_thread.join(5)
        finally:
            release_signal.set()

        self.assertTrue(input_done.is_set())
        self.assertEqual(cancel_result["job"]["status"], "cancelled")
        self.assertEqual(len(input_errors), 1)
        self.assertIsInstance(input_errors[0], broker.RemoteAgentBrokerError)
        transcript = remote_broker.read(job_id=job_id, stream="transcript")["text"]
        self.assertNotIn("late-input", transcript)

    def test_dual_stream_output_is_exact_when_cancellation_races_monitor(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        ready_path = root / "dual-stream-ready"
        line_count = 4000
        script = "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "import threading",
                f"count = {line_count}",
                "def emit(stream, prefix):",
                "    for index in range(count):",
                "        stream.write(f'{prefix}-{index:04d}\\n'.encode('utf-8'))",
                "    stream.flush()",
                "stdout_thread = threading.Thread(target=emit, args=(sys.stdout.buffer, 'stdout-λ'))",
                "stderr_thread = threading.Thread(target=emit, args=(sys.stderr.buffer, 'stderr-雪'))",
                "stdout_thread.start()",
                "stderr_thread.start()",
                "stdout_thread.join()",
                "stderr_thread.join()",
                "Path(sys.argv[1]).write_text('ready', encoding='utf-8')",
                "sys.stdin.read()",
            ]
        )
        payload = remote_broker.start_shell(
            command=[sys.executable, "-c", script, str(ready_path)],
            cwd=str(root),
            async_job=True,
            timeout_sec=60,
        )
        job_id = str(payload["job"]["job_id"])
        deadline = time.monotonic() + 5
        while not ready_path.exists() and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        self.assertTrue(ready_path.exists(), "dual-stream child did not become ready")

        cancelled = remote_broker.cancel(job_id=job_id)

        expected_stdout = "".join(f"stdout-λ-{index:04d}\n" for index in range(line_count))
        expected_stderr = "".join(f"stderr-雪-{index:04d}\n" for index in range(line_count))
        self.assertEqual(cancelled["job"]["status"], "cancelled")
        self.assertIn("Cancellation requested", str(cancelled["job"]["last_error"]))
        self.assertEqual(
            remote_broker.read(job_id=job_id, stream="stdout")["text"],
            expected_stdout,
        )
        self.assertEqual(
            remote_broker.read(job_id=job_id, stream="stderr")["text"],
            expected_stderr,
        )
        self.assertEqual(remote_broker.status(job_id)["job"]["status"], "cancelled")
        final = json.loads(remote_broker.read(job_id=job_id, stream="final")["text"])
        self.assertEqual(final["status"], "cancelled")
        self.assertIn("Cancellation requested", str(final["last_error"]))

    def test_malformed_utf8_is_replaced_without_losing_later_output(self) -> None:
        remote_broker, _, root, _ = self.make_broker()

        payload = remote_broker.start_shell(
            command=[
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'before\\xffafter\\n')",
            ],
            cwd=str(root),
            timeout_sec=5,
        )
        job_id = str(payload["job"]["job_id"])

        stdout = remote_broker.read(job_id=job_id, stream="stdout")

        self.assertEqual(payload["job"]["status"], "completed")
        self.assertEqual(stdout["text"], "before\ufffdafter\n")

    def test_monitor_waits_for_stream_capture_before_terminal_artifacts(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        pump_entered = threading.Event()
        release_pump = threading.Event()
        sentinel = "blocked-pump-output"
        original_redact = broker.redact_text

        def blocking_redact(value: str) -> str:
            if sentinel in value and "_pump_stream" in threading.current_thread().name:
                pump_entered.set()
                release_pump.wait(5)
            return original_redact(value)

        try:
            with patch.object(broker, "redact_text", side_effect=blocking_redact):
                payload = remote_broker.start_shell(
                    command=[sys.executable, "-c", f"print('{sentinel}')"],
                    cwd=str(root),
                    async_job=True,
                    timeout_sec=5,
                )
                job_id = str(payload["job"]["job_id"])
                self.assertTrue(pump_entered.wait(5))

                interim = remote_broker.status(job_id)
                self.assertEqual(interim["job"]["status"], "running")
                artifact_names = {
                    item["name"]
                    for item in remote_broker.artifacts(job_id=job_id)["artifacts"]
                }
                self.assertNotIn("final.json", artifact_names)

                release_pump.set()
                completed = self.wait_for_status(remote_broker, job_id, "completed")
        finally:
            release_pump.set()

        self.assertEqual(completed["job"]["status"], "completed")
        final = json.loads(remote_broker.read(job_id=job_id, stream="final")["text"])
        self.assertIn(sentinel, final["stdout_tail"])

    @unittest.skipUnless(hasattr(os, "killpg"), "requires POSIX process groups")
    def test_monitor_terminates_same_group_descendant_holding_streams(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        child_pid_path = root / "same-group-child.pid"
        child_cleanup_needed = threading.Event()
        child_cleanup_needed.set()
        child_script = "\n".join(
            [
                "from pathlib import Path",
                "import os",
                "import sys",
                "import time",
                "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8')",
                "print('same-group-descendant-output', flush=True)",
                "time.sleep(60)",
            ]
        )
        parent_script = "\n".join(
            [
                "from pathlib import Path",
                "import subprocess",
                "import sys",
                "import time",
                "pid_path = Path(sys.argv[1])",
                "subprocess.Popen([sys.executable, '-c', sys.argv[2], str(pid_path)])",
                "deadline = time.monotonic() + 5",
                "while not pid_path.exists() and time.monotonic() < deadline:",
                "    time.sleep(0.01)",
                "if not pid_path.exists():",
                "    raise RuntimeError('same-group descendant did not start')",
                "print('parent-exit-output', flush=True)",
            ]
        )

        def stop_child() -> None:
            if not child_cleanup_needed.is_set() or not child_pid_path.exists():
                return
            try:
                os.kill(int(child_pid_path.read_text(encoding="utf-8")), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
            finally:
                child_cleanup_needed.clear()

        self.addCleanup(stop_child)
        with (
            patch.object(broker, "POST_EXIT_STREAM_DRAIN_SEC", 0.05),
            patch.object(broker, "POST_EXIT_PROCESS_GROUP_TERM_SEC", 0.1),
            patch.object(broker, "POST_EXIT_PROCESS_GROUP_KILL_SEC", 0.1),
        ):
            payload = remote_broker.start_shell(
                command=[
                    sys.executable,
                    "-c",
                    parent_script,
                    str(child_pid_path),
                    child_script,
                ],
                cwd=str(root),
                async_job=True,
                timeout_sec=5,
            )
            job_id = str(payload["job"]["job_id"])
            completed = self.wait_for_status(
                remote_broker,
                job_id,
                "completed",
                timeout_sec=3,
            )
            child_cleanup_needed.clear()

        stdout = remote_broker.read(job_id=job_id, stream="stdout")["text"]
        self.assertEqual(completed["job"]["status"], "completed")
        self.assertIsNone(completed["job"]["last_error"])
        self.assertIn("parent-exit-output", stdout)
        self.assertIn("same-group-descendant-output", stdout)

    @unittest.skipUnless(hasattr(os, "killpg"), "requires POSIX process groups")
    def test_monitor_abandons_detached_stream_capture_without_late_appends(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        child_pid_path = root / "detached-child.pid"
        release_path = root / "release-detached-output"
        emitted_path = root / "detached-output-emitted"
        child_cleanup_needed = threading.Event()
        child_cleanup_needed.set()
        child_script = "\n".join(
            [
                "from pathlib import Path",
                "import os",
                "import signal",
                "import sys",
                "import time",
                "pid_path = Path(sys.argv[1])",
                "release_path = Path(sys.argv[2])",
                "emitted_path = Path(sys.argv[3])",
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "pid_path.write_text(str(os.getpid()), encoding='utf-8')",
                "print('detached-before-terminal', flush=True)",
                "while not release_path.exists():",
                "    time.sleep(0.01)",
                "print('detached-after-terminal-stdout', flush=True)",
                "print('detached-after-terminal-stderr', file=sys.stderr, flush=True)",
                "emitted_path.write_text('done', encoding='utf-8')",
                "time.sleep(60)",
            ]
        )
        parent_script = "\n".join(
            [
                "from pathlib import Path",
                "import subprocess",
                "import sys",
                "import time",
                "pid_path = Path(sys.argv[1])",
                "subprocess.Popen(",
                "    [sys.executable, '-c', sys.argv[4], *sys.argv[1:4]],",
                "    start_new_session=True,",
                ")",
                "deadline = time.monotonic() + 5",
                "while not pid_path.exists() and time.monotonic() < deadline:",
                "    time.sleep(0.01)",
                "if not pid_path.exists():",
                "    raise RuntimeError('detached descendant did not start')",
                "print('detached-parent-exit', flush=True)",
            ]
        )

        def stop_child() -> None:
            if not child_cleanup_needed.is_set() or not child_pid_path.exists():
                return
            try:
                os.kill(int(child_pid_path.read_text(encoding="utf-8")), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
            finally:
                child_cleanup_needed.clear()

        self.addCleanup(stop_child)
        try:
            with (
                patch.object(broker, "POST_EXIT_STREAM_DRAIN_SEC", 0.05),
                patch.object(broker, "POST_EXIT_PROCESS_GROUP_TERM_SEC", 0.05),
                patch.object(broker, "POST_EXIT_PROCESS_GROUP_KILL_SEC", 0.05),
            ):
                payload = remote_broker.start_shell(
                    command=[
                        sys.executable,
                        "-c",
                        parent_script,
                        str(child_pid_path),
                        str(release_path),
                        str(emitted_path),
                        child_script,
                    ],
                    cwd=str(root),
                    async_job=True,
                    timeout_sec=5,
                )
                job_id = str(payload["job"]["job_id"])
                failed = self.wait_for_status(
                    remote_broker,
                    job_id,
                    "failed",
                    timeout_sec=3,
                )

            last_error = str(failed["job"]["last_error"])
            self.assertEqual(failed["job"]["exit_code"], 0)
            self.assertIn("stdout capture failed", last_error)
            self.assertIn("stderr capture failed", last_error)
            self.assertIn("bounded process-group cleanup", last_error)
            stdout_before = remote_broker.read(job_id=job_id, stream="stdout")["text"]
            stderr_before = remote_broker.read(job_id=job_id, stream="stderr")["text"]
            self.assertNotIn("after-terminal", stdout_before)
            self.assertNotIn("after-terminal", stderr_before)

            release_path.write_text("release", encoding="utf-8")
            deadline = time.monotonic() + 2
            while not emitted_path.exists() and time.monotonic() < deadline:
                threading.Event().wait(0.01)
            self.assertTrue(emitted_path.exists(), "detached child did not emit late output")
            stop_child()

            stdout_after = remote_broker.read(job_id=job_id, stream="stdout")["text"]
            stderr_after = remote_broker.read(job_id=job_id, stream="stderr")["text"]
            self.assertEqual(stdout_after, stdout_before)
            self.assertEqual(stderr_after, stderr_before)
            final = json.loads(remote_broker.read(job_id=job_id, stream="final")["text"])
            self.assertEqual(final["status"], "failed")
            self.assertEqual(final["last_error"], last_error)
        finally:
            stop_child()

    def test_large_output_coalesces_job_state_writes(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        line_count = 2000
        script = (
            "import sys; "
            f"sys.stdout.write(''.join(f'line-{{index:04d}}\\n' for index in range({line_count})))"
        )

        with (
            patch.object(broker, "OUTPUT_STATE_FLUSH_INTERVAL_SEC", 60.0),
            patch.object(
                remote_broker.job_store,
                "write_unlocked",
                wraps=remote_broker.job_store.write_unlocked,
            ) as write_job,
        ):
            payload = remote_broker.start_shell(
                command=[sys.executable, "-c", script],
                cwd=str(root),
                timeout_sec=5,
            )

        job_id = str(payload["job"]["job_id"])
        expected = "".join(f"line-{index:04d}\n" for index in range(line_count))
        self.assertEqual(payload["job"]["status"], "completed")
        self.assertEqual(remote_broker.read(job_id=job_id, stream="stdout")["text"], expected)
        self.assertLessEqual(write_job.call_count, 4)
        self.assertEqual(remote_broker.cancel(job_id=job_id)["job"]["status"], "completed")

    def test_status_does_not_mark_exited_job_lost_while_monitor_finishes(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        monitor_in_diff = threading.Event()
        release_monitor = threading.Event()
        original_capture_git_diff = broker.capture_git_diff

        def slow_capture_git_diff(cwd: Path, *, base_ref: str | None = None) -> dict[str, object]:
            monitor_in_diff.set()
            release_monitor.wait(5)
            return original_capture_git_diff(cwd, base_ref=base_ref)

        broker.capture_git_diff = slow_capture_git_diff
        try:
            payload = remote_broker.start_shell(
                command=[sys.executable, "-c", "print('done')"],
                cwd=str(root),
                async_job=True,
                timeout_sec=5,
            )
            job_id = str(payload["job"]["job_id"])

            self.assertTrue(monitor_in_diff.wait(5))
            status = remote_broker.status(job_id)

            self.assertNotEqual(status["job"]["status"], "lost")
            release_monitor.set()
            self.wait_for_status(remote_broker, job_id, "completed")
        finally:
            release_monitor.set()
            broker.capture_git_diff = original_capture_git_diff

    def test_read_uses_job_capture_limit_as_default_ceiling(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        payload = remote_broker.start_shell(
            command=[sys.executable, "-c", "print('abcdef')"],
            cwd=str(root),
            timeout_sec=5,
            capture_limit_bytes=4,
        )
        job_id = str(payload["job"]["job_id"])

        stdout = remote_broker.read(job_id=job_id, stream="stdout")
        expanded = remote_broker.read(job_id=job_id, stream="stdout", max_bytes=100)

        self.assertEqual(stdout["text"], "abcd")
        self.assertEqual(expanded["text"], "abcd")

    def test_incremental_read_preserves_utf8_across_byte_page_boundaries(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        payload = remote_broker.start_shell(
            command=[sys.executable, "-c", "import sys; sys.stdout.write('ab€c')"],
            cwd=str(root),
            timeout_sec=5,
        )
        job_id = str(payload["job"]["job_id"])

        first = remote_broker.read(job_id=job_id, stream="stdout", max_bytes=3)
        second = remote_broker.read(
            job_id=job_id,
            stream="stdout",
            since_offset=int(first["next_offset"]),
            max_bytes=3,
        )
        third = remote_broker.read(
            job_id=job_id,
            stream="stdout",
            since_offset=int(second["next_offset"]),
            max_bytes=3,
        )

        self.assertEqual(first["text"], "ab")
        self.assertEqual(second["text"], "€")
        self.assertEqual(third["text"], "c")
        self.assertEqual(first["next_offset"], 2)
        self.assertEqual(second["next_offset"], 5)
        self.assertTrue(third["eof"])
        with self.assertRaisesRegex(broker.RemoteAgentBrokerError, "character boundary"):
            remote_broker.read(
                job_id=job_id,
                stream="stdout",
                since_offset=3,
                max_bytes=3,
            )

    def test_failed_shell_job_records_failed_state(self) -> None:
        remote_broker, _, root, _ = self.make_broker()

        payload = remote_broker.start_shell(
            command=[sys.executable, "-c", "import sys; sys.exit(7)"],
            cwd=str(root),
            timeout_sec=5,
        )
        job_id = str(payload["job"]["job_id"])

        final = remote_broker.read(job_id=job_id, stream="final")

        self.assertEqual(payload["job"]["status"], "failed")
        self.assertEqual(payload["job"]["exit_code"], 7)
        self.assertIn("Process exited with 7", str(payload["job"]["last_error"]))
        self.assertIn('"exit_code": 7', final["text"])

    def test_status_marks_unmanaged_running_job_lost(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        job_id, _ = self.seed_persisted_job(
            remote_broker,
            cwd=root,
            status="running",
        )

        def add_pid(job: dict[str, object]) -> None:
            job["pid"] = 999999

        remote_broker.job_store.update(job_id, add_pid)

        status = remote_broker.status(job_id)

        self.assertEqual(status["job"]["status"], "lost")
        self.assertIn("no longer managing", str(status["job"]["last_error"]))

    def test_agent_start_supports_copilot_and_claude(self) -> None:
        remote_broker, _, root, _ = self.make_broker()

        for agent in ("copilot", "claude"):
            with self.subTest(agent=agent):
                payload = remote_broker.start_agent(
                    agent=agent,
                    cwd=str(root),
                    prompt=f"hello from {agent}",
                    runner_options={
                        "model": "fake-model",
                        "effort": "high",
                        "output_format": "text",
                    },
                    timeout_sec=5,
                )
                job_id = str(payload["job"]["job_id"])
                self.wait_for_status(remote_broker, job_id, "completed")

                stdout = remote_broker.read(job_id=job_id, stream="stdout")
                self.assertIn(agent.upper(), stdout["text"])
                if agent == "claude":
                    self.assertIn(f"hello from {agent}", stdout["text"])

    def test_claude_jobs_ignore_legacy_model_settings(self) -> None:
        remote_broker, _, root, _ = self.make_broker()

        payload = remote_broker.start_agent(
            agent="claude",
            cwd=str(root),
            prompt="hello from default claude",
            timeout_sec=5,
        )
        job = payload["job"]

        self.assertNotIn("model", job["runner_options"])
        self.assertNotIn("effort", job["runner_options"])
        command = job["command"]
        self.assertNotIn("--model", command)
        self.assertNotIn("--effort", command)
        self.wait_for_status(remote_broker, str(job["job_id"]), "completed")

    def test_claude_jobs_forward_main_thread_options_without_settings(self) -> None:
        remote_broker, _, root, _ = self.make_broker(with_settings=False)
        payload = remote_broker.start_agent(
            agent="claude", cwd=str(root), prompt="hello", timeout_sec=5,
            runner_options={"model": "parent-model", "effort": "high"},
        )
        job = payload["job"]
        self.assertEqual(job["runner_options"], {"model": "parent-model", "effort": "high"})
        self.assertIn("parent-model", job["command"])
        self.assertIn("--effort", job["command"])
        self.wait_for_status(remote_broker, str(job["job_id"]), "completed", timeout_sec=10)

    def test_agent_idle_timeout_requires_no_output(self) -> None:
        remote_broker, _, root, bin_dir = self.make_broker(monotonic_scale=20.0)
        (bin_dir / "claude").write_text(
            "\n".join(
                [
                    f"#!{sys.executable}",
                    "import sys",
                    "import time",
                    "sys.stdin.read()",
                    "time.sleep(60)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (bin_dir / "claude").chmod(0o755)

        payload = remote_broker.start_agent(
            agent="claude",
            cwd=str(root),
            prompt="silent",
            idle_timeout_sec=1,
        )
        job_id = str(payload["job"]["job_id"])
        status = self.wait_for_status(remote_broker, job_id, "failed", timeout_sec=5)

        self.assertIn("idle timed out", str(status["job"]["last_error"]))

    def test_agent_streaming_output_refreshes_idle_timeout(self) -> None:
        remote_broker, _, root, bin_dir = self.make_broker(monotonic_scale=20.0)
        (bin_dir / "claude").write_text(
            "\n".join(
                [
                    f"#!{sys.executable}",
                    "import sys",
                    "import time",
                    "sys.stdin.read()",
                    "for _ in range(3):",
                    "    sys.stderr.write('thinking')",
                    "    sys.stderr.flush()",
                    "    time.sleep(0.02)",
                    "print('done')",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (bin_dir / "claude").chmod(0o755)

        payload = remote_broker.start_agent(
            agent="claude",
            cwd=str(root),
            prompt="stream",
            idle_timeout_sec=1,
        )
        job_id = str(payload["job"]["job_id"])
        status = self.wait_for_status(remote_broker, job_id, "completed", timeout_sec=5)
        stderr = remote_broker.read(job_id=job_id, stream="stderr")

        self.assertEqual(status["job"]["status"], "completed")
        self.assertIn("thinking", stderr["text"])
        self.assertIsNotNone(status["job"]["last_output_at"])

    def test_diff_captures_worktree_changes(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        with patch(
            "tokimon.tools.mcp.remote_agent_broker.capture_git_diff",
            return_value={"available": True, "diff": "diff --git a/tracked.txt b/tracked.txt\n-before\n+after\n"},
        ) as capture:
            payload = remote_broker.start_shell(
                command=[sys.executable, "-c", "print('diff-ready')"],
                cwd=str(root),
                timeout_sec=5,
            )
            job_id = str(payload["job"]["job_id"])
            self.wait_for_status(remote_broker, job_id, "completed", timeout_sec=5)
            diff = remote_broker.diff(job_id=job_id)

        self.assertTrue(diff["available"])
        self.assertIn("diff --git a/tracked.txt b/tracked.txt", diff["diff"])
        self.assertIn("-before", diff["diff"])
        self.assertIn("+after", diff["diff"])
        self.assertTrue(Path(str(diff["path"])).exists())
        capture.assert_called_once_with(root, base_ref=None)

    def test_diff_passes_base_ref_to_capture_git_diff(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        payload = remote_broker.start_shell(
            command=[sys.executable, "-c", "print('diff-ready')"],
            cwd=str(root),
            timeout_sec=5,
        )
        job_id = str(payload["job"]["job_id"])

        with patch(
            "tokimon.tools.mcp.remote_agent_broker.capture_git_diff",
            return_value={"available": True, "diff": "diff --git a/staged.txt b/staged.txt\n"},
        ) as capture:
            diff = remote_broker.diff(job_id=job_id, base_ref="HEAD~1")

        self.assertTrue(diff["available"])
        self.assertIn("staged.txt", diff["diff"])
        capture.assert_called_once_with(root, base_ref="HEAD~1")

    def test_diff_reports_git_timeout_as_unavailable(self) -> None:
        _, _, root, _ = self.make_broker()

        with (
            patch("tokimon.tools.mcp.remote_agent_broker.shutil.which", return_value="/fake/git"),
            patch(
                "tokimon.tools.mcp.remote_agent_broker.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["git"], timeout=10),
            ),
        ):
            diff = broker.capture_git_diff(root)

        self.assertFalse(diff["available"])
        self.assertIn("timed out", str(diff["error"]))

    def test_cleanup_removes_expired_terminal_jobs_only(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        payload = remote_broker.start_shell(
            command=[sys.executable, "-c", "print('old')"],
            cwd=str(root),
            timeout_sec=5,
        )
        job = payload["job"]
        job_id = str(job["job_id"])
        old_timestamp = "2020-01-01T00:00:00+00:00"

        def make_old(stored_job: dict[str, object]) -> None:
            stored_job["completed_at"] = old_timestamp
            stored_job["updated_at"] = old_timestamp

        remote_broker.job_store.update(job_id, make_old)
        expired_job_id, _ = self.seed_persisted_job(
            remote_broker,
            cwd=root,
            status="expired",
            updated_at=old_timestamp,
        )
        remote_broker.job_store.update(expired_job_id, make_old)
        running_job_id, _ = self.seed_persisted_job(
            remote_broker,
            cwd=root,
            status="running",
            updated_at=old_timestamp,
        )

        cleanup = remote_broker.cleanup(ttl_days=7)

        self.assertEqual(set(cleanup["removed"]), {job_id, expired_job_id})
        self.assertFalse((remote_broker.jobs_root / job_id).exists())
        self.assertFalse((remote_broker.jobs_root / expired_job_id).exists())
        self.assertTrue((remote_broker.jobs_root / running_job_id).exists())

    def test_cleanup_honors_per_job_retention_and_explicit_override(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        completed_at = "2026-08-23T00:00:00+00:00"
        short_job_id, short_job_dir = self.seed_persisted_job(
            remote_broker,
            cwd=root,
            status="completed",
            updated_at=completed_at,
            ttl_days=3,
        )
        long_job_id, long_job_dir = self.seed_persisted_job(
            remote_broker,
            cwd=root,
            status="completed",
            updated_at=completed_at,
            ttl_days=10,
        )
        remote_broker.cleanup_manager.now = lambda: datetime(
            2026,
            8,
            28,
            tzinfo=timezone.utc,
        )

        per_job = remote_broker.cleanup()

        self.assertEqual(per_job["retention_policy"], "per_job")
        self.assertEqual(per_job["removed"], [short_job_id])
        self.assertFalse(short_job_dir.exists())
        self.assertTrue(long_job_dir.exists())

        overridden = remote_broker.cleanup(ttl_days=3)

        self.assertEqual(overridden["retention_policy"], "override")
        self.assertEqual(overridden["removed"], [long_job_id])
        self.assertFalse(long_job_dir.exists())

    def test_cleanup_rejects_nonpositive_retention_before_scanning(self) -> None:
        remote_broker, _, _, _ = self.make_broker()

        with self.assertRaisesRegex(
            broker.RemoteAgentBrokerError,
            "ttl_days must be a positive integer",
        ):
            remote_broker.cleanup(ttl_days=0)
        self.assertFalse(remote_broker.jobs_root.exists())

        with (
            patch("sys.stderr", new=io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            broker.build_parser().parse_args(["cleanup", "--ttl-days", "-1"])

    def test_separate_broker_cleanup_waits_for_in_flight_artifact_read(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        cleanup_broker = broker.RemoteAgentBroker(
            config=remote_broker.config,
            env=remote_broker.env,
        )
        payload = remote_broker.start_shell(
            command=[sys.executable, "-c", "print('old-output')"],
            cwd=str(root),
            timeout_sec=5,
        )
        job_id = str(payload["job"]["job_id"])
        job_dir = remote_broker.job_store.job_dir(job_id)

        def expire(job: dict[str, object]) -> None:
            job["completed_at"] = "2020-01-01T00:00:00+00:00"
            job["updated_at"] = "2020-01-01T00:00:00+00:00"

        remote_broker.job_store.update(job_id, expire)
        read_entered = threading.Event()
        release_read = threading.Event()
        cleanup_scan_reached = threading.Event()
        read_done = threading.Event()
        cleanup_done = threading.Event()
        read_results: list[dict[str, object]] = []
        cleanup_results: list[dict[str, object]] = []
        errors: list[Exception] = []
        original_read_page = remote_broker.artifact_manager.read_page
        original_runtime_lookup = cleanup_broker.cleanup_manager.runtime_lookup

        def blocking_read_page(*args: object, **kwargs: object) -> tuple[str, int, int]:
            read_entered.set()
            release_read.wait(5)
            return original_read_page(*args, **kwargs)  # type: ignore[arg-type]

        def read_artifact() -> None:
            try:
                read_results.append(remote_broker.read(job_id=job_id, stream="stdout"))
            except Exception as exc:
                errors.append(exc)
            finally:
                read_done.set()

        def run_cleanup() -> None:
            try:
                cleanup_results.append(cleanup_broker.cleanup(ttl_days=7))
            except Exception as exc:
                errors.append(exc)
            finally:
                cleanup_done.set()

        def observed_runtime_lookup(candidate_job_id: str) -> object | None:
            if candidate_job_id == job_id:
                cleanup_scan_reached.set()
            return original_runtime_lookup(candidate_job_id)

        try:
            with (
                patch.object(
                    remote_broker.artifact_manager,
                    "read_page",
                    side_effect=blocking_read_page,
                ),
                patch.object(
                    cleanup_broker.cleanup_manager,
                    "runtime_lookup",
                    side_effect=observed_runtime_lookup,
                ),
            ):
                read_thread = threading.Thread(target=read_artifact)
                read_thread.start()
                self.assertTrue(read_entered.wait(5))

                cleanup_thread = threading.Thread(target=run_cleanup)
                cleanup_thread.start()
                self.assertTrue(cleanup_scan_reached.wait(5))
                self.assertFalse(cleanup_done.is_set())
                self.assertTrue(job_dir.exists())

                release_read.set()
                self.assertTrue(read_done.wait(5))
                self.assertTrue(cleanup_done.wait(5))
        finally:
            release_read.set()

        self.assertEqual(errors, [])
        self.assertEqual(read_results[0]["text"], "old-output\n")
        self.assertEqual(cleanup_results[0]["removed"], [job_id])
        self.assertFalse(job_dir.exists())

    def test_cleanup_fails_closed_when_process_lock_is_held(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        job_id, job_dir = self.seed_persisted_job(
            remote_broker,
            cwd=root,
            status="completed",
            updated_at="2020-01-01T00:00:00+00:00",
        )
        lock_path = remote_broker.config.runtime_root / "locks" / "cleanup.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            cleanup = remote_broker.cleanup(ttl_days=7)

        self.assertFalse(cleanup["ok"])
        self.assertIn("cleanup lock failure", cleanup["error"])
        self.assertEqual(cleanup["removed"], [])
        self.assertTrue(job_dir.exists())
        self.assertTrue((remote_broker.jobs_root / job_id).exists())

    def test_cleanup_fails_closed_for_in_root_job_directory_symlink(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        retained_job_id, retained_job_dir = self.seed_persisted_job(
            remote_broker,
            cwd=root,
            status="completed",
            updated_at="2020-01-01T00:00:00+00:00",
        )
        backing_dir = remote_broker.config.runtime_root / "symlink-backing-job"
        backing_dir.mkdir(parents=True)
        (backing_dir / "job.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "completed_at": "2020-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        linked_job_dir = (
            remote_broker.jobs_root / "job_20200101T000000Z_deadbeefcafe"
        )
        linked_job_dir.symlink_to(backing_dir, target_is_directory=True)

        cleanup = remote_broker.cleanup(ttl_days=7)

        self.assertFalse(cleanup["ok"])
        self.assertIn("cleanup safety failure", str(cleanup["error"]))
        self.assertIn("symbolic link", str(cleanup["error"]))
        self.assertEqual(cleanup["removed"], [])
        self.assertTrue(retained_job_dir.exists())
        self.assertTrue((remote_broker.jobs_root / retained_job_id).exists())
        self.assertTrue(linked_job_dir.is_symlink())
        self.assertTrue(backing_dir.exists())

    def test_cleanup_reports_filesystem_delete_failure(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        job_id, job_dir = self.seed_persisted_job(
            remote_broker,
            cwd=root,
            status="completed",
            updated_at="2020-01-01T00:00:00+00:00",
        )

        with patch(
            "tokimon.tools.mcp.remote_agent_broker.storage.shutil.rmtree",
            side_effect=OSError("delete denied"),
        ):
            cleanup = remote_broker.cleanup(ttl_days=7)

        self.assertFalse(cleanup["ok"])
        self.assertIn("cleanup failure", str(cleanup["error"]))
        self.assertIn("delete denied", str(cleanup["error"]))
        self.assertEqual(cleanup["removed"], [])
        self.assertTrue(job_dir.exists())
        self.assertTrue((remote_broker.jobs_root / job_id).exists())

    def test_cleanup_fails_closed_for_outside_job_directory(self) -> None:
        remote_broker, _, root, _ = self.make_broker()
        payload = remote_broker.start_shell(
            command=[sys.executable, "-c", "print('old')"],
            cwd=str(root),
            timeout_sec=5,
        )
        job_id = str(payload["job"]["job_id"])
        job_path = Path(str(payload["job"]["artifacts"]["job"]))
        stored = json.loads(job_path.read_text(encoding="utf-8"))
        stored["completed_at"] = "2020-01-01T00:00:00+00:00"
        stored["updated_at"] = "2020-01-01T00:00:00+00:00"
        job_path.write_text(json.dumps(stored), encoding="utf-8")

        external_job_dir = root / "outside-runtime-job"
        external_job_dir.mkdir()
        (external_job_dir / "job.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "completed_at": "2020-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        bad_job_dir = remote_broker.jobs_root / "job_20200101T000000Z_deadbeefcafe"
        bad_job_dir.symlink_to(external_job_dir, target_is_directory=True)

        cleanup = remote_broker.cleanup(ttl_days=7)

        self.assertFalse(cleanup["ok"])
        self.assertIn("cleanup boundary failure", str(cleanup["error"]))
        self.assertEqual(cleanup["removed"], [])
        self.assertTrue((remote_broker.jobs_root / job_id).exists())
        self.assertTrue(external_job_dir.exists())


if __name__ == "__main__":
    unittest.main()
