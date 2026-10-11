from __future__ import annotations

import unittest
from types import SimpleNamespace

from tokimon.tools.mcp import ai_debate, remote_agent_broker, remote_browser_broker, telegram_bot


class MCPProtocolCompatibilityTests(unittest.TestCase):
    def test_service_entrypoints_preserve_protocol_and_tool_error_boundaries(self) -> None:
        servers = (
            (ai_debate.MCPServer(SimpleNamespace()), {}, "tokimon-ai-debate", "0.1.0"),
            (remote_agent_broker.MCPServer(SimpleNamespace()), {}, "tokimon-remote-agent-broker", "0.1.0"),
            (remote_browser_broker.MCPServer(
                broker=SimpleNamespace(config=remote_browser_broker.BrokerConfig()),
            ), {"mcp_session_key": "test-session"}, "tokimon-remote-browser-broker", "0.2.0"),
            (telegram_bot.MCPServer(), {}, "tokimon-telegram-bot", "0.1.0"),
        )
        for server, context, identity, version in servers:
            with self.subTest(service=identity):
                initialized = server.handle({
                    "id": "请求", "method": "initialize",
                    "params": {"protocolVersion": "client-version"},
                }, **context)
                self.assertEqual(initialized, {
                    "jsonrpc": "2.0", "id": "请求", "result": {
                        "protocolVersion": "client-version",
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
