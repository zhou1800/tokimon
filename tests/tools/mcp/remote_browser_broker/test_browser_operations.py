from __future__ import annotations

import http.client
import io
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from tokimon.tools.mcp import remote_browser_broker as broker
from tests.tools.mcp.remote_browser_broker.test_remote_browser_broker import (
    FakeController,
    FakeLauncher,
    FakeNoVNCProxy,
    tool_payload,
)


class BrowserOperationsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.controller = FakeController()
        self.remote = broker.RemoteBrowserBroker(
            config=broker.BrokerConfig(runtime_root=self.root / "browser"),
            controller=self.controller,
            launcher=FakeLauncher(),
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(self.release)
        self.server = broker.MCPServer(self.remote)

    def release(self) -> None:
        self.remote.shutdown_all_sessions()
        self.remote.release_shutdown_leases()

    def call(self, name: str, *, mcp_key: str = "mcp-a", **arguments: object) -> dict:
        response = self.server.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": name, "arguments": arguments}},
            mcp_session_key=mcp_key,
        )
        payload = tool_payload(response)
        self.assertFalse(response["result"].get("isError"), (name, payload))
        return payload

    def ref(self, index: int) -> str:
        snapshot = self.call("browser_snapshot")
        return snapshot["page"]["elements"][index]["ref"]

    def test_legacy_mode_environment_cannot_enable_approval_gates(self) -> None:
        for value in ("operator", "typo"):
            with self.subTest(value=value):
                config = broker.BrokerConfig.from_env(env={
                    "TOKIMON_RB_AUTHORIZATION_MODE": value,
                    "TOKIMON_RB_OPERATOR_TOKEN": "unused-legacy-token",
                })
                self.assertEqual(config, broker.BrokerConfig.from_env(env={}))
        with self.assertRaises(broker.BrowserBrokerError):
            broker.replace_config(self.remote.config, bind_host="0.0.0.0")

    def test_page_operations_execute_without_tokens_or_approvals(self) -> None:
        self.call("browser_ensure", start_url="https://example.com/")
        self.call("browser_navigate", url="https://example.com/")
        self.call("browser_click", ref=self.ref(0), pacing="fast")
        self.call("browser_type", ref=self.ref(1), text="example text", pacing="batch")
        source = self.root / "attachment.txt"
        source.write_text("attachment", encoding="utf-8")
        self.call("browser_upload_file", ref=self.ref(2), path=str(source))
        self.call("browser_backend_request", method="POST", path="/draft", body={"text": "draft"})
        self.call("browser_press_key", key="Escape")
        self.call("browser_evaluate", expression="document.title")
        tab = self.call("browser_new_tab", url="https://example.org/")["tab"]["tab_id"]
        self.call("browser_switch_tab", tab_id=tab)
        self.call("browser_close_tab", tab_id=tab)
        self.assertEqual(len(self.controller.navigations), 2)
        self.assertEqual(len(self.controller.clicks), 1)
        self.assertEqual(self.controller.types[0][2], "example text")
        self.assertEqual(len(self.controller.uploads), 1)
        self.assertEqual(self.controller.backend_requests[0]["method"], "POST")
        self.assertEqual(len(self.controller.keys), 1)
        self.assertEqual(len(self.controller.evaluations), 1)
        self.assertEqual(len(self.controller.tabs), 3)

    def test_lifecycle_cleanup_and_template_management_need_no_operator(self) -> None:
        created = self.call("browser_ensure")
        self.assertEqual(created["connect"]["tool"], "browser_connect_info")
        self.assertNotIn("required_operator_tool", created["connect"])
        first = created["browser_session"]["id"]
        second = self.call("browser_ensure", recreate=True, start_url="https://example.com/")
        self.assertNotEqual(first, second["browser_session"]["id"])
        self.call("browser_connect_info")
        self.call("browser_stop")
        self.assertEqual(self.call("browser_session_delete_retained")["status"], "deleted")
        self.call("browser_session_cleanup_expired")
        self.call("browser_session_list")
        self.call("browser_session_cleanup_orphaned")
        self.call("browser_template_create", template_name="test-template")
        self.call("browser_template_seal", template_name="test-template")
        self.call("browser_ensure", template_name="test-template")
        self.call("browser_stop")
        self.call("browser_session_delete_retained")
        self.assertEqual(self.call("browser_template_delete", template_name="test-template")["status"], "deleted")

    def test_pause_remains_effective_and_resume_needs_no_operator(self) -> None:
        self.call("browser_ensure")
        self.call("browser_pause_for_human")
        self.call("browser_snapshot")
        with self.assertRaises(broker.BrowserBrokerError) as blocked:
            self.remote.navigate("mcp-a", url="https://example.com/")
        self.assertEqual(blocked.exception.code, broker.ERROR_PAUSED)
        self.call("browser_connect_info")
        self.call("browser_resume")
        self.call("browser_navigate", url="https://example.com/")

    def test_catalog_and_diagnostics_do_not_request_approvals(self) -> None:
        response = self.server.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, mcp_session_key="mcp-a",
        )
        catalog = response["result"]["tools"]
        names = {tool["name"] for tool in catalog}
        self.assertNotIn("browser_action_approve", names)
        self.assertNotIn("browser_action_request_inspect", names)
        self.assertNotIn("browser_development_grant", names)
        self.assertNotIn("browser_risk_acknowledge", names)
        self.assertNotIn("approval_id", json.dumps(catalog))
        self.assertNotIn("operator-only", json.dumps(catalog).lower())
        self.assertNotIn("browser_operator_session_list", names)
        self.assertNotIn("browser_operator_cleanup_orphaned", names)
        for tool in catalog:
            self.assertTrue({"approval_id", "confirm", "confirm_name"}.isdisjoint(
                tool["inputSchema"]["properties"]
            ))
        diagnostics = self.call("browser_diagnostics")
        self.assertNotIn("authorization_mode", diagnostics)
        self.assertNotIn("operator_channel", diagnostics)
        self.assertNotIn("operator_authority", diagnostics)

    def test_stdio_executes_operations_with_legacy_environment(self) -> None:
        output = io.StringIO()
        with patch.dict("os.environ", {"TOKIMON_RB_OPERATOR_TOKEN": "unused-legacy-token"}):
            result = broker.serve(
                input_stream=io.StringIO(json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "browser_ensure", "arguments": {"start_url": "https://example.com/"}},
                }) + "\n"),
                output_stream=output, server=self.server,
            )
        self.assertEqual(result, 0)
        self.assertEqual(tool_payload(json.loads(output.getvalue()))["status"], "ready")
        self.assertEqual(len(self.controller.navigations), 1)
        self.assertNotIn("unused-legacy-token", output.getvalue())

    def test_type_preserves_literal_text_and_rejects_other_types(self) -> None:
        self.call("browser_ensure")
        for text in ("  exact text\n", " \t\n", ""):
            with self.subTest(text=text):
                refs = self.call("browser_snapshot")["page"]["elements"]
                result = self.call("browser_type", ref=refs[1]["ref"], text=text, clear=True)
                self.assertEqual(result["status"], "typed")
                self.assertEqual(self.controller.types[-1][2:], (text, True))
        before = list(self.controller.types)
        for text in (None, 123, True, [], {}):
            with self.subTest(text=text):
                response = self.server.handle({
                    "jsonrpc": "2.0", "id": "invalid-text", "method": "tools/call",
                    "params": {"name": "browser_type", "arguments": {"ref": "e1", "text": text}},
                }, mcp_session_key="mcp-a")
                self.assertTrue(response["result"]["isError"])
                self.assertEqual(tool_payload(response)["error_code"], broker.ERROR_INVALID_ARGUMENT)
        self.assertEqual(self.controller.types, before)

    def test_session_and_input_boundaries(self) -> None:
        self.call("browser_ensure")
        other = self.call("browser_ensure", mcp_key="other")
        self.assertNotEqual(other["browser_session"]["id"], self.call("browser_whoami")["browser_session"]["id"])
        with self.assertRaises(broker.BrowserBrokerError) as invalid:
            self.remote.evaluate("mcp-a", expression="x" * (broker.MAX_OPERATION_PARAMETER_BYTES + 1))
        self.assertEqual(invalid.exception.code, broker.ERROR_INVALID_ARGUMENT)
        self.assertEqual(self.controller.evaluations, [])

    def test_http_accepts_local_calls_and_delete_without_token(self) -> None:
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), broker.make_handler(self.server))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            self.call("browser_ensure")
            connection = http.client.HTTPConnection(*httpd.server_address, timeout=3)
            connection.request("POST", "/mcp", json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "browser_navigate", "arguments": {"url": "https://example.com/"}},
            }), {"mcp-session-id": "mcp-a", "Content-Type": "application/json"})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertNotIn("CONFIRMATION_REQUIRED", response.read().decode())
            self.assertEqual(len(self.controller.navigations), 1)
            connection.request("DELETE", "/mcp", headers={"mcp-session-id": "mcp-a"})
            response = connection.getresponse()
            self.assertEqual(response.status, 202)
            response.read()
            self.assertEqual(
                self.call("browser_whoami")["browser_session"]["state"],
                broker.STATE_STOPPED_RETAINED,
            )
            connection.close()
            for headers in ({"Origin": "https://untrusted.example"}, {"Host": "untrusted.example"}):
                connection = http.client.HTTPConnection(*httpd.server_address, timeout=3)
                connection.request("DELETE", "/mcp", headers=headers)
                response = connection.getresponse()
                self.assertEqual(response.status, 403)
                response.read()
                connection.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
