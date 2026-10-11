from __future__ import annotations

import io
import json
import logging
import sys
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tokimon.tools.mcp import ai_debate, remote_agent_broker, remote_browser_broker, telegram_bot
from tokimon.tools.mcp.container_executor import gateway, server as executor
from tests.tools.mcp.remote_agent_broker.test_http_security import MemoryConnection


def service_endpoints():
    browser = SimpleNamespace(
        config=remote_browser_broker.BrokerConfig(),
        begin_shutdown=Mock(), _shutdown_internal_operations=nullcontext,
        shutdown_transport_session=Mock(), shutdown_all_sessions=Mock(),
        release_shutdown_leases=Mock(),
    )
    return (
        (ai_debate.MCPServer(SimpleNamespace()), {}, ai_debate.SERVER_NAME,
         ai_debate.SERVER_VERSION, ai_debate.SUPPORTED_PROTOCOL_VERSION, ai_debate),
        (remote_agent_broker.MCPServer(SimpleNamespace()), {}, remote_agent_broker.SERVER_NAME,
         remote_agent_broker.SERVER_VERSION, remote_agent_broker.SUPPORTED_PROTOCOL_VERSION, remote_agent_broker),
        (remote_browser_broker.MCPServer(browser), {"mcp_session_key": "test-session"},
         remote_browser_broker.SERVER_NAME, remote_browser_broker.SERVER_VERSION,
         remote_browser_broker.SUPPORTED_PROTOCOL_VERSION, remote_browser_broker),
        (telegram_bot.MCPServer(), {}, telegram_bot.SERVER_NAME, telegram_bot.SERVER_VERSION,
         telegram_bot.SUPPORTED_PROTOCOL_VERSION, telegram_bot),
        (executor.ExecutorServer.__new__(executor.ExecutorServer), {"correlation_id": "test-correlation"},
         "tokimon-container-executor", executor.VERSION, "2024-11-05", executor),
        (gateway.GatewayServer.__new__(gateway.GatewayServer), {"correlation": "test-correlation"},
         "tokimon-workspace-gateway", gateway.VERSION, "2024-11-05", gateway),
    )


class MCPProtocolCompatibilityTests(unittest.TestCase):
    def test_service_entrypoints_preserve_protocol_and_tool_error_boundaries(self) -> None:
        for server, context, identity, version, protocol, _ in service_endpoints():
            with self.subTest(service=identity):
                initialized = server.handle({
                    "id": "请求", "method": "initialize",
                    "params": {"protocolVersion": protocol},
                }, **context)
                self.assertEqual(initialized, {
                    "jsonrpc": "2.0", "id": "请求", "result": {
                        "protocolVersion": protocol,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": identity, "version": version},
                    },
                })
                self.assertIsNone(server.handle({"method": "notifications/initialized"}, **context))
                self.assertEqual(server.handle({"id": 0, "method": "ping", "params": []}, **context),
                                 {"jsonrpc": "2.0", "id": 0, "result": {}})
                self.assertEqual(server.handle({"id": 3, "method": "unsupported"}, **context), {
                    "jsonrpc": "2.0", "id": 3,
                    "error": {"code": -32601, "message": "Method not found: unsupported"},
                })
                result = server.handle({
                    "id": 4, "method": "tools/call", "params": {"name": "absent-tool"},
                }, **context)
                self.assertTrue(result["result"]["isError"])
                self.assertEqual(result["result"]["content"][0]["type"], "text")

    def test_initialization_returns_only_the_service_supported_revision(self) -> None:
        for server, context, identity, _, protocol, _ in service_endpoints():
            for proposal in (protocol, "2099-01-01", "2020-01-01", "client-version"):
                with self.subTest(service=identity, proposal=proposal):
                    result = server.handle({
                        "jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": proposal},
                    }, **context)
                    self.assertEqual(result["result"]["protocolVersion"], protocol)
                    self.assertEqual(result["id"], 1)

    def test_missing_or_malformed_version_proposals_keep_the_declared_revision(self) -> None:
        params_cases = ({}, None, [], {"protocolVersion": None}, {"protocolVersion": ""},
                        {"protocolVersion": 42}, {"protocolVersion": {}}, {"protocolVersion": []})
        for server, context, identity, _, protocol, _ in service_endpoints():
            for params in params_cases:
                with self.subTest(service=identity, params=params):
                    result = server.handle({"id": 1, "method": "initialize", "params": params}, **context)
                    self.assertEqual(result["result"]["protocolVersion"], protocol)
            result = server.handle({"id": 2, "method": "initialize"}, **context)
            self.assertEqual(result["result"]["protocolVersion"], protocol)

    def test_notifications_never_reply_or_dispatch_request_only_operations(self) -> None:
        for server, context, identity, _, _, module in service_endpoints():
            # Dispatch is the application boundary of the shared protocol under test.
            if module is remote_agent_broker:
                target, attribute = server.dispatcher, "dispatch"
            else:
                target, attribute = server, "call_tool" if module in (executor, gateway) else "_call_tool"
            with patch.object(target, attribute) as dispatch:
                for method in ("notifications/initialized", "notifications/cancelled", "notifications/unknown",
                               "initialize", "ping", "tools/list", "tools/call", "unsupported"):
                    with self.subTest(service=identity, method=method):
                        result = server.handle({
                            "jsonrpc": "2.0", "method": method,
                            "params": {"name": "absent-tool", "requestId": 1},
                        }, **context)
                        self.assertIsNone(result)
                dispatch.assert_not_called()

    def test_explicit_falsey_ids_keep_success_and_error_responses(self) -> None:
        for server, context, identity, _, _, _ in service_endpoints():
            for request_id in (0, "", None):
                with self.subTest(service=identity, request_id=request_id):
                    self.assertEqual(server.handle({
                        "jsonrpc": "2.0", "id": request_id, "method": "ping",
                    }, **context), {"jsonrpc": "2.0", "id": request_id, "result": {}})
                    for method in ("unsupported", "notifications/initialized", "notifications/cancelled"):
                        result = server.handle({"id": request_id, "method": method}, **context)
                        self.assertEqual(result["id"], request_id)
                        self.assertEqual(result["error"]["code"], -32601)

    def test_malformed_methods_are_invalid_requests_rather_than_notifications(self) -> None:
        for server, context, identity, _, _, _ in service_endpoints():
            for method_fields in ({}, {"method": None}, {"method": 42}, {"method": []}, {"method": {}}):
                for id_fields in ({}, {"id": 0}):
                    with self.subTest(service=identity, method_fields=method_fields, id_fields=id_fields):
                        result = server.handle({"jsonrpc": "2.0", **method_fields, **id_fields}, **context)
                        self.assertEqual(result["id"], id_fields.get("id"))
                        self.assertEqual(result["error"]["code"], -32600)

    def test_idless_maintenance_cannot_change_executor_state(self) -> None:
        endpoint = executor.ExecutorServer.__new__(executor.ExecutorServer)
        endpoint.jobs = Mock()
        endpoint.jobs.maintenance.return_value = {"paused": True}
        payload = {"jsonrpc": "2.0", "method": "tokimon/maintenance", "params": {"paused": True}}
        self.assertIsNone(endpoint.handle(payload))
        endpoint.jobs.maintenance.assert_not_called()
        result = endpoint.handle({**payload, "id": 0})
        self.assertEqual(result, {"jsonrpc": "2.0", "id": 0, "result": {"paused": True}})
        endpoint.jobs.maintenance.assert_called_once_with(True)


class MCPTransportRecoveryTests(unittest.TestCase):
    def assert_stdio_recovery(self, bad_line: str, *, code: int = -32700, decoder_failure=None) -> None:
        real_loads = json.loads
        for endpoint, _, identity, _, _, module in service_endpoints()[:4]:
            with self.subTest(service=identity):
                output, log_output = io.StringIO(), io.StringIO()
                logger = logging.Logger(identity)
                handler = logging.StreamHandler(log_output)
                formatter = module.JsonLogFormatter if module is telegram_bot else module.StructuredLogFormatter
                handler.setFormatter(formatter())
                logger.addHandler(handler)
                first = json.dumps({"jsonrpc": "2.0", "id": 0, "method": "ping"})
                last = json.dumps({"jsonrpc": "2.0", "id": "after-error", "method": "ping"})
                with ExitStack() as stack:
                    stack.enter_context(patch.object(module, "configure_runtime_logger", return_value=logger))
                    if decoder_failure is not None:
                        def decode(line, **kwargs):
                            if line.strip() == bad_line:
                                raise decoder_failure
                            return real_loads(line, **kwargs)
                        stack.enter_context(patch.object(module.json, "loads", side_effect=decode))
                    status = module.serve(
                        input_stream=io.StringIO(first + "\n" + bad_line + "\n" + last + "\n"),
                        output_stream=output, server=endpoint,
                    )
                self.assertEqual(status, 0)
                replies = [real_loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(len(replies), 3)
                self.assertEqual(replies[0], {"jsonrpc": "2.0", "id": 0, "result": {}})
                self.assertEqual(replies[1]["id"], None)
                self.assertEqual(replies[1]["error"]["code"], code)
                self.assertEqual(replies[2], {"jsonrpc": "2.0", "id": "after-error", "result": {}})
                if code == -32700:
                    logs = [real_loads(line) for line in log_output.getvalue().splitlines()]
                    parse_errors = [record for record in logs if record["message"] == "invalid_jsonrpc_request"]
                    self.assertEqual(len(parse_errors), 1)
                    self.assertTrue(parse_errors[0]["correlation_id"])
                if module is remote_browser_broker:
                    endpoint.broker.begin_shutdown.assert_called_once_with()
                    endpoint.broker.shutdown_transport_session.assert_called_once()
                    endpoint.broker.shutdown_all_sessions.assert_called_once_with()
                    endpoint.broker.release_shutdown_leases.assert_called_once_with()

    def test_stdio_recovers_after_malformed_json(self) -> None:
        self.assert_stdio_recovery('{"jsonrpc":')

    def test_stdio_recovers_after_non_object_json(self) -> None:
        self.assert_stdio_recovery("[]", code=-32600)

    def test_stdio_recovers_after_interpreter_integer_limit(self) -> None:
        previous = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(4300)
        self.addCleanup(sys.set_int_max_str_digits, previous)
        self.assert_stdio_recovery('{"jsonrpc":"2.0","id":1,"method":"ping","value":' + "9" * 5000 + "}")
        self.assertEqual(sys.get_int_max_str_digits(), 4300)

    def test_stdio_recovers_after_decoder_recursion_error(self) -> None:
        # Decoder depth varies by Python build; inject only that external failure.
        self.assert_stdio_recovery('{"deep":[]}', decoder_failure=RecursionError("Decoder nesting exceeded"))

    def test_stdio_notifications_leave_following_request_correlation_intact(self) -> None:
        for endpoint, _, identity, _, _, module in service_endpoints()[:4]:
            with self.subTest(service=identity):
                requests = (
                    {"id": 0, "method": "ping"},
                    {"method": "notifications/cancelled", "params": {"requestId": 0}},
                    {"method": "notifications/unknown"},
                    {"id": "", "method": "ping"},
                )
                output = io.StringIO()
                with patch.object(module, "configure_runtime_logger", return_value=Mock()):
                    status = module.serve(
                        input_stream=io.StringIO("\n".join(json.dumps(request) for request in requests) + "\n"),
                        output_stream=output, server=endpoint,
                    )
                self.assertEqual(status, 0)
                self.assertEqual([json.loads(line) for line in output.getvalue().splitlines()], [
                    {"jsonrpc": "2.0", "id": 0, "result": {}}, {"jsonrpc": "2.0", "id": "", "result": {}},
                ])

    def invoke_http(self, module, endpoint, body: bytes, *, decoder_failure=None):
        raw = (
            b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:8938\r\nContent-Type: application/json\r\n"
            b"mcp-session-id: test-session\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )
        connection, logger = MemoryConnection(raw), Mock()
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "configure_runtime_logger", return_value=logger))
            if decoder_failure is not None:
                stack.enter_context(patch.object(module.json, "loads", side_effect=decoder_failure))
            module.make_handler(endpoint)(
                connection, ("127.0.0.1", 1234), SimpleNamespace(server_address=("127.0.0.1", 8938)),
            )
        headers, response_body = bytes(connection.output).split(b"\r\n\r\n", 1)
        status = int(headers.split(b" ", 2)[1])
        return status, headers, response_body, logger

    def test_http_notifications_return_empty_202_with_session_header(self) -> None:
        for endpoint, _, identity, _, _, module in service_endpoints()[1:3]:
            for method in ("notifications/initialized", "notifications/cancelled", "notifications/unknown", "tools/call"):
                with self.subTest(service=identity, method=method):
                    body = json.dumps({"jsonrpc": "2.0", "method": method, "params": {"name": "absent-tool"}}).encode()
                    status, headers, response_body, _ = self.invoke_http(module, endpoint, body)
                    self.assertEqual(status, 202)
                    self.assertIn(b"mcp-session-id: test-session", headers)
                    self.assertEqual(response_body, b"")

    def test_http_parser_failures_return_parse_errors_without_poisoning_next_request(self) -> None:
        previous = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(4300)
        self.addCleanup(sys.set_int_max_str_digits, previous)
        cases = (b'{"jsonrpc":', b"\xff", b'{"value":' + b"9" * 5000 + b"}")
        for endpoint, _, identity, _, _, module in service_endpoints()[1:3]:
            for body in cases:
                with self.subTest(service=identity, body_size=len(body)):
                    status, _, response_body, logger = self.invoke_http(module, endpoint, body)
                    self.assertEqual(status, 200)
                    payload = json.loads(response_body.decode().split("data: ", 1)[1].strip())
                    self.assertEqual(payload["id"], None)
                    self.assertEqual(payload["error"]["code"], -32700)
                    failures = [call for call in logger.log.call_args_list
                                if call.args[1] == "invalid_jsonrpc_request"]
                    self.assertEqual(len(failures), 1)
                    self.assertEqual(failures[0].kwargs["extra"]["correlation_id"], "test-session")
                    status, _, response_body, _ = self.invoke_http(module, endpoint, b'{"id":0,"method":"ping"}')
                    self.assertEqual(status, 200)
                    self.assertIn(b'"id": 0, "result": {}', response_body)

    def test_http_decoder_recursion_error_is_a_parse_error(self) -> None:
        for endpoint, _, identity, _, _, module in service_endpoints()[1:3]:
            with self.subTest(service=identity):
                status, _, response_body, _ = self.invoke_http(
                    module, endpoint, b'{"deep":[]}', decoder_failure=RecursionError("Decoder nesting exceeded"),
                )
                self.assertEqual(status, 200)
                payload = json.loads(response_body.decode().split("data: ", 1)[1].strip())
                self.assertEqual(payload["error"]["code"], -32700)
