from __future__ import annotations

import io
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tokimon.tools.mcp import ai_debate, remote_agent_broker, remote_browser_broker, telegram_bot
from tokimon.tools.mcp.container_executor import gateway, server as executor
from tests.tools.mcp.remote_agent_broker.test_http_security import MemoryConnection
from tests.tools.mcp.test_protocol import service_endpoints


UNICODE_TEXT = "中文🚀 café \\ud800\n\"quoted\"\u2028\u2029"
SURROGATE_TEXT = "中文🚀 high:\ud800 low:\udcff literal:\\ud800"
ESCAPED_SURROGATE_TEXT = "中文🚀 high:\\ud800 low:\\udcff literal:\\ud800"


def request_bytes(request: dict) -> bytes:
    # The test input encoder is independent of the response encoder under test.
    text = json.dumps(request, ensure_ascii=False)
    try:
        return text.encode("utf-8") + b"\n"
    except UnicodeEncodeError:
        return json.dumps(request, ensure_ascii=True).encode("ascii") + b"\n"


class MCPWireEncodingTests(unittest.TestCase):
    def run_stdio(self, module, endpoint, requests):
        output = io.BytesIO()
        with (
            io.TextIOWrapper(io.BytesIO(b"".join(request_bytes(r) for r in requests)),
                             encoding="utf-8", errors="strict") as incoming,
            io.TextIOWrapper(output, encoding="utf-8", errors="strict", write_through=True) as outgoing,
            patch.object(module, "configure_runtime_logger", return_value=Mock()),
        ):
            status = module.serve(input_stream=incoming, output_stream=outgoing, server=endpoint)
            outgoing.flush()
            raw = output.getvalue()
        self.assertEqual(status, 0)
        if module is remote_browser_broker:
            endpoint.broker.begin_shutdown.assert_called_once_with()
            endpoint.broker.shutdown_transport_session.assert_called_once()
            endpoint.broker.shutdown_all_sessions.assert_called_once_with()
            endpoint.broker.release_shutdown_leases.assert_called_once_with()
        return raw

    def decoded_lines(self, raw: bytes):
        text = raw.decode("utf-8", errors="strict")
        self.assertTrue(raw.endswith(b"\n"))
        return [json.loads(line) for line in text.split("\n") if line]

    def invoke_http(self, module, endpoint, request):
        body = request_bytes(request).rstrip(b"\n")
        raw = (
            b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:8938\r\n"
            b"Content-Type: application/json\r\nmcp-session-id: unicode-session\r\n"
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body
        )
        connection = MemoryConnection(raw)
        with patch.object(module, "configure_runtime_logger", return_value=Mock()):
            module.make_handler(endpoint)(
                connection, ("127.0.0.1", 1234), SimpleNamespace(server_address=("127.0.0.1", 8938)),
            )
        headers, response_body = bytes(connection.output).split(b"\r\n\r\n", 1)
        self.assertIn(b" 200 ", headers.split(b"\r\n", 1)[0])
        parsed_headers = {
            name.lower(): value.strip()
            for name, value in (line.split(b":", 1) for line in headers.split(b"\r\n")[1:])
        }
        self.assertEqual(int(parsed_headers[b"content-length"]), len(response_body))
        self.assertEqual(parsed_headers[b"mcp-session-id"], b"unicode-session")
        self.assertIn(b"text/event-stream", parsed_headers[b"content-type"])
        text = response_body.decode("utf-8", errors="strict")
        self.assertTrue(text.endswith("\n\n"))
        data = [line.removeprefix("data: ") for line in text.split("\n") if line.startswith("data: ")]
        self.assertEqual(len(data), 1)
        return json.loads(data[0]), response_body

    def install_tool_payload(self, module, endpoint, payload):
        if module is ai_debate:
            endpoint.orchestrator.health = Mock(return_value=payload)
            return "ai_debate_health"
        if module is remote_agent_broker:
            endpoint.broker.health = Mock(return_value=payload)
            return "remote_agent_health"
        if module is remote_browser_broker:
            endpoint.broker.whoami = Mock(return_value=payload)
            return "browser_whoami"
        self.enterContext(patch.object(telegram_bot, "telegram_health_payload", return_value=payload))
        return "telegram_bot_health"

    def assert_ping_sequence(self, replies, ids):
        self.assertEqual(replies, [{"jsonrpc": "2.0", "id": value, "result": {}} for value in ids])

    def test_stdio_surrogate_ids_do_not_interrupt_following_multibyte_requests(self):
        ids = ["\ud800", "\udcff", "literal:\\ud800", "请求🚀", 0]
        requests = [{"jsonrpc": "2.0", "id": value, "method": "ping"} for value in ids]
        for endpoint, _, identity, _, _, module in service_endpoints()[:4]:
            with self.subTest(service=identity):
                raw = self.run_stdio(module, endpoint, requests)
                self.assert_ping_sequence(self.decoded_lines(raw), ids)
                self.assertIn("请求🚀".encode("utf-8"), raw)

    def test_stdio_unknown_method_errors_escape_surrogates_and_continue(self):
        method = "unsupported-" + SURROGATE_TEXT
        requests = [{"id": 1, "method": method}, {"id": "请求🚀", "method": "ping"}]
        for endpoint, _, identity, _, _, module in service_endpoints()[:4]:
            with self.subTest(service=identity):
                replies = self.decoded_lines(self.run_stdio(module, endpoint, requests))
                self.assertEqual(replies[0]["error"], {
                    "code": -32601, "message": "Method not found: unsupported-" + ESCAPED_SURROGATE_TEXT,
                })
                replies[0]["error"]["message"].encode("utf-8", errors="strict")
                self.assertEqual(replies[1], {"jsonrpc": "2.0", "id": "请求🚀", "result": {}})

    def test_stdio_tool_content_roundtrips_surrogates_without_losing_regular_text(self):
        payload = {"text": SURROGATE_TEXT, "ordinary": UNICODE_TEXT}
        for endpoint, _, identity, _, _, module in service_endpoints()[:4]:
            with self.subTest(service=identity):
                name = self.install_tool_payload(module, endpoint, payload)
                requests = [
                    {"id": "工具🚀", "method": "tools/call", "params": {"name": name}},
                    {"id": 0, "method": "ping"},
                ]
                raw = self.run_stdio(module, endpoint, requests)
                replies = self.decoded_lines(raw)
                self.assertEqual(replies[0]["id"], "工具🚀")
                self.assertEqual(json.loads(replies[0]["result"]["content"][0]["text"]), payload)
                replies[0]["result"]["content"][0]["text"].encode("utf-8", errors="strict")
                self.assertEqual(replies[1], {"jsonrpc": "2.0", "id": 0, "result": {}})
                self.assertIn("中文🚀".encode("utf-8"), raw)

    def test_http_surrogate_ids_roundtrip_and_following_request_succeeds(self):
        for endpoint, _, identity, _, _, module in service_endpoints()[1:3]:
            for request_id in ("\ud800", "\udcff", "请求🚀"):
                with self.subTest(service=identity, request_id=request_id):
                    reply, _ = self.invoke_http(module, endpoint, {"id": request_id, "method": "ping"})
                    self.assertEqual(reply, {"jsonrpc": "2.0", "id": request_id, "result": {}})
                    reply, _ = self.invoke_http(module, endpoint, {"id": 0, "method": "ping"})
                    self.assertEqual(reply, {"jsonrpc": "2.0", "id": 0, "result": {}})

    def test_http_error_and_tool_content_are_utf8_safe_with_byte_accurate_lengths(self):
        payload = {"text": SURROGATE_TEXT, "ordinary": UNICODE_TEXT}
        for endpoint, _, identity, _, _, module in service_endpoints()[1:3]:
            with self.subTest(service=identity):
                reply, _ = self.invoke_http(module, endpoint, {"id": 1, "method": SURROGATE_TEXT})
                self.assertEqual(reply["error"]["message"], "Method not found: " + ESCAPED_SURROGATE_TEXT)
                reply["error"]["message"].encode("utf-8", errors="strict")
                name = self.install_tool_payload(module, endpoint, payload)
                reply, raw = self.invoke_http(module, endpoint, {
                    "id": "工具🚀", "method": "tools/call", "params": {"name": name},
                })
                self.assertEqual(json.loads(reply["result"]["content"][0]["text"]), payload)
                reply["result"]["content"][0]["text"].encode("utf-8", errors="strict")
                self.assertIn("中文🚀".encode("utf-8"), raw)
                reply, _ = self.invoke_http(module, endpoint, {"id": 0, "method": "ping"})
                self.assertEqual(reply, {"jsonrpc": "2.0", "id": 0, "result": {}})

    def test_http_health_json_uses_the_same_safe_utf8_encoding(self):
        payload = {"text": SURROGATE_TEXT, "ordinary": UNICODE_TEXT}
        for endpoint, _, identity, _, _, module in service_endpoints()[1:3]:
            with self.subTest(service=identity):
                connection = MemoryConnection(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1:8938\r\n\r\n")
                if module is remote_agent_broker:
                    endpoint.broker.health = Mock(return_value=payload)
                else:
                    self.enterContext(patch.object(module, "health_payload", return_value=payload))
                with patch.object(module, "configure_runtime_logger", return_value=Mock()):
                    module.make_handler(endpoint)(
                        connection, ("127.0.0.1", 1234), SimpleNamespace(server_address=("127.0.0.1", 8938)),
                    )
                headers, body = bytes(connection.output).split(b"\r\n\r\n", 1)
                self.assertIn(b" 200 ", headers.split(b"\r\n", 1)[0])
                self.assertIn(b"Content-Length: " + str(len(body)).encode("ascii"), headers)
                self.assertEqual(json.loads(body.decode("utf-8", errors="strict")), payload)
                self.assertIn("中文🚀".encode("utf-8"), body)

    def test_browser_http_shutdown_error_body_remains_valid_utf8(self):
        endpoint = service_endpoints()[2][0]
        failure = remote_browser_broker.BrowserBrokerError(SURROGATE_TEXT, details={"path": SURROGATE_TEXT})
        endpoint.broker.shutdown_transport_session.side_effect = failure
        connection = MemoryConnection(
            b"DELETE /mcp HTTP/1.1\r\nHost: 127.0.0.1:8938\r\nmcp-session-id: unicode-session\r\n\r\n"
        )
        with patch.object(remote_browser_broker, "configure_runtime_logger", return_value=Mock()):
            remote_browser_broker.make_handler(endpoint)(
                connection, ("127.0.0.1", 1234), SimpleNamespace(server_address=("127.0.0.1", 8938)),
            )
        headers, body = bytes(connection.output).split(b"\r\n\r\n", 1)
        self.assertIn(b" 409 ", headers.split(b"\r\n", 1)[0])
        self.assertIn(b"Content-Length: " + str(len(body)).encode("ascii"), headers)
        self.assertEqual(json.loads(body.decode("utf-8", errors="strict")), failure.to_payload())
        endpoint.broker.shutdown_transport_session.assert_called_once_with("unicode-session")

    def run_container_port(self, port, requests, *, tool_payload=None):
        endpoint = executor.ExecutorServer.__new__(executor.ExecutorServer) if port == "worker" else (
            gateway.GatewayServer.__new__(gateway.GatewayServer)
        )
        endpoint.log = Mock()
        if tool_payload is not None:
            if port == "worker":
                endpoint.jobs = SimpleNamespace(jobs=Mock(return_value=tool_payload))
            else:
                # Socket RPC is the Gateway's external dependency.
                endpoint.routes = {"ws-" + "a" * 20: {"name": "fixture", "socket": "/fixture/mcp.sock"}}
                self.enterContext(patch.object(gateway, "request", return_value={
                    "jsonrpc": "2.0", "id": 1, "result": {"content": [
                        {"type": "text", "text": json.dumps(tool_payload, ensure_ascii=False)},
                    ]},
                }))
        incoming = io.BytesIO(b"".join(request_bytes(r) for r in requests))
        outgoing = io.BytesIO()
        if port == "worker":
            handler = executor.RequestHandler.__new__(executor.RequestHandler)
            handler.connection = Mock()
            handler.rfile, handler.wfile = incoming, outgoing
            handler.server = SimpleNamespace(executor=endpoint)
            handler.handle()
        else:
            with (
                patch.object(gateway, "no_new_privileges"),
                patch.object(gateway, "load_routes", return_value={}),
                patch.object(gateway, "GatewayServer", return_value=endpoint),
                patch.object(sys, "stdin", SimpleNamespace(buffer=incoming)),
                patch.object(sys, "stdout", SimpleNamespace(buffer=outgoing)),
            ):
                status = gateway.serve()
            self.assertEqual(status, 0)
        return outgoing.getvalue()

    def test_container_ports_preserve_surrogate_ids_and_following_requests(self):
        ids = ["\ud800", "\udcff", "literal:\\ud800", "请求🚀", 0]
        requests = [{"jsonrpc": "2.0", "id": value, "method": "ping"} for value in ids]
        for port in ("worker", "gateway"):
            with self.subTest(port=port):
                raw = self.run_container_port(port, requests)
                self.assert_ping_sequence(self.decoded_lines(raw), ids)
                self.assertIn("请求🚀".encode("utf-8"), raw)

    def test_container_ports_escape_error_and_nested_tool_content(self):
        payload = {"text": SURROGATE_TEXT, "ordinary": UNICODE_TEXT}
        for port in ("worker", "gateway"):
            with self.subTest(port=port):
                arguments = {} if port == "worker" else {"workspace_id": "ws-" + "a" * 20}
                requests = [
                    {"id": 1, "method": SURROGATE_TEXT},
                    {"id": "工具🚀", "method": "tools/call",
                     "params": {"name": "executor_jobs", "arguments": arguments}},
                    {"id": 0, "method": "ping"},
                ]
                raw = self.run_container_port(port, requests, tool_payload=payload)
                replies = self.decoded_lines(raw)
                self.assertEqual(replies[0]["error"]["message"], "Method not found: " + ESCAPED_SURROGATE_TEXT)
                replies[0]["error"]["message"].encode("utf-8", errors="strict")
                self.assertEqual(json.loads(replies[1]["result"]["content"][0]["text"]), payload)
                replies[1]["result"]["content"][0]["text"].encode("utf-8", errors="strict")
                self.assertEqual(replies[2], {"jsonrpc": "2.0", "id": 0, "result": {}})
                self.assertIn("中文🚀".encode("utf-8"), raw)

    def test_container_oversized_unicode_tool_reply_keeps_bounded_error_frame(self):
        payload = {"text": "中文🚀" * (executor.MAX_FRAME // 10)}
        for port in ("worker", "gateway"):
            with self.subTest(port=port):
                arguments = {} if port == "worker" else {"workspace_id": "ws-" + "a" * 20}
                requests = [
                    {"id": "请求🚀", "method": "tools/call",
                     "params": {"name": "executor_jobs", "arguments": arguments}},
                    {"id": 0, "method": "ping"},
                ]
                raw = self.run_container_port(port, requests, tool_payload=payload)
                self.assertTrue(all(len(line) + 1 <= executor.MAX_FRAME for line in raw.splitlines()))
                replies = self.decoded_lines(raw)
                self.assertEqual(replies[0]["error"]["code"], -32603)
                self.assertEqual(replies[1], {"jsonrpc": "2.0", "id": 0, "result": {}})
