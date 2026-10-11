from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tokimon.tools.mcp import remote_agent_broker as broker


class MemoryConnection:
    def __init__(self, request: bytes) -> None:
        self.input = io.BytesIO(request)
        self.output = bytearray()
        self.timeout = None

    def makefile(self, *_: object) -> io.BytesIO:
        return self.input

    def sendall(self, data: bytes) -> None:
        self.output.extend(data)
        if data.startswith(b": connected"):
            raise BrokenPipeError("Synthetic event-stream client disconnected")

    def settimeout(self, value: float) -> None:
        self.timeout = value


class RemoteAgentHTTPBoundaryTests(unittest.TestCase):
    def invoke(
        self, *, method="POST", headers=(), body=None,
        client="127.0.0.1", bind="127.0.0.1",
    ):
        if body is None:
            body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
        entries = [("Host", "127.0.0.1:8938"), ("Content-Type", "application/json"),
                   ("Content-Length", str(len(body)))]
        overrides = {key.lower() for key, _ in headers}
        entries = [(key, value) for key, value in entries if key.lower() not in overrides]
        entries.extend((key, value) for key, value in headers if value is not None)
        raw = (f"{method} /mcp HTTP/1.1\r\n" +
               "".join(f"{key}: {value}\r\n" for key, value in entries) + "\r\n").encode() + body
        connection = MemoryConnection(raw)
        endpoint = Mock(spec=broker.MCPServer)
        endpoint.handle.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
        logger = Mock()
        with patch.object(broker, "configure_runtime_logger", return_value=logger):
            broker.make_handler(endpoint)(connection, (client, 1234),
                                          SimpleNamespace(server_address=(bind, 8938)))
        status = int(bytes(connection.output).split(b" ", 2)[1])
        return status, endpoint, connection

    def test_origin_and_foreign_host_rejected_before_any_dispatch(self):
        for method in ("GET", "DELETE", "POST"):
            for headers in ((("Origin", "https://untrusted.example"),), (("Origin", "null"),),
                            (("Origin", "http://127.0.0.1:8938"),), (("Origin", ""),),
                            (("Host", "untrusted.example:8938"),), (("Host", None),),
                            (("Host", "localhost:8938"), ("Host", "untrusted.example")),
                            (("Host", "attacker@localhost:8938"),),
                            (("Host", "localhost:8938/path"),)):
                with self.subTest(method=method, headers=headers):
                    status, endpoint, _ = self.invoke(method=method, headers=headers)
                    self.assertEqual(status, 403)
                    endpoint.handle.assert_not_called()

    def test_nonlocal_client_or_listener_rejected(self):
        for client, bind in (("192.0.2.1", "127.0.0.1"), ("127.0.0.1", "0.0.0.0")):
            with self.subTest(client=client, bind=bind):
                status, endpoint, _ = self.invoke(client=client, bind=bind)
                self.assertEqual(status, 403)
                endpoint.handle.assert_not_called()

    def test_local_and_ssh_tunnel_clients_keep_jsonrpc_contract(self):
        for host in ("127.0.0.1:8938", "localhost:8938", "[::1]:8938"):
            with self.subTest(host=host):
                status, endpoint, connection = self.invoke(headers=(("Host", host),))
                self.assertEqual(status, 200)
                self.assertEqual(endpoint.handle.call_args.kwargs["request_context"].transport, "http-mcp")
                self.assertIn(b"text/event-stream", connection.output)
                self.assertEqual(connection.timeout, 10)

    def test_invalid_framing_and_content_types_rejected_before_dispatch(self):
        cases = [((("Content-Length", value),), 400) for value in (None, "", "-1", "0", "1.5", "+1")]
        cases += [((("Content-Length", str(1024 * 1024 + 1)),), 413),
                  ((("Content-Length", "2"), ("Content-Length", "3")), 400),
                  ((("Transfer-Encoding", "chunked"),), 400),
                  ((("Content-Type", "text/plain"),), 415),
                  ((("Content-Type", None),), 415)]
        for headers, expected in cases:
            with self.subTest(headers=headers):
                status, endpoint, _ = self.invoke(headers=headers)
                self.assertEqual(status, expected)
                endpoint.handle.assert_not_called()

    def test_truncated_body_and_invalid_utf8_do_not_dispatch(self):
        status, endpoint, _ = self.invoke(headers=(("Content-Length", "20"),), body=b"{}")
        self.assertEqual(status, 400)
        endpoint.handle.assert_not_called()
        status, endpoint, connection = self.invoke(body=b"\xff")
        self.assertEqual(status, 200)
        self.assertIn(b"-32700", connection.output)
        endpoint.handle.assert_not_called()

    def test_deep_array_and_oversized_integer_json_do_not_dispatch(self):
        for body in (b"[" * 1100 + b"]" * 1100, b'{"value":' + b"9" * 5000 + b"}"):
            with self.subTest(body_size=len(body)):
                status, endpoint, connection = self.invoke(body=body)
                self.assertEqual(status, 200)
                self.assertTrue(b"-32700" in connection.output or b"-32600" in connection.output)
                endpoint.handle.assert_not_called()

    def test_body_limit_accepts_exact_boundary_and_rejects_the_next_byte(self):
        body = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'
        padded = body.ljust(1024 * 1024, b" ")
        status, endpoint, _ = self.invoke(body=padded)
        self.assertEqual(status, 200)
        endpoint.handle.assert_called_once()
        status, endpoint, _ = self.invoke(body=padded + b" ")
        self.assertEqual(status, 413)
        endpoint.handle.assert_not_called()

    def test_nonloopback_startup_fails_before_socket_or_broker_creation(self):
        with patch.object(broker, "ThreadingHTTPServer") as http, patch.object(broker, "RemoteAgentBroker") as runtime:
            for host in ("0.0.0.0", "::", "192.0.2.1", "untrusted.example"):
                with self.subTest(host=host), self.assertRaises(broker.RemoteAgentBrokerError):
                    broker.serve_http(config=broker.BrokerConfig(bind_host=host))
            http.assert_not_called()
            runtime.assert_not_called()
