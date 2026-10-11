from __future__ import annotations

import json
import subprocess
import sys
import unittest
from unittest import mock

from tokimon.tools.mcp.container_executor.gateway import GatewayServer

A, B = "ws-" + "a" * 20, "ws-" + "b" * 20


def result(payload):
    return {"result": {"content": [{"type": "text", "text": json.dumps(payload)}]}}


class GatewayTests(unittest.TestCase):
    def setUp(self):
        routes = {key: {"directory": "/projects/" + key, "socket_volume": key + "-socket", "owner_id": "a" * 32} for key in (A, B)}
        self.server = GatewayServer({"schema_version": 1, "routes": routes})
        self.requests = []
        self.jobs = {A: {"1" * 32}, B: {"2" * 32}}
        def worker(method, params, correlation, *, socket_path, timeout):
            identity = socket_path.split("/")[-2]
            self.requests.append((identity, params, correlation))
            args = params["arguments"]
            if "job_id" in args and args["job_id"] not in self.jobs[identity]:
                payload = result({"error": "unknown job"})
                payload["result"]["isError"] = True
                return payload
            return result({"directory": identity, "cwd": args.get("cwd"), "path": args.get("path"), "deployment": identity,
                           "ok": True, "roots": {"/mounts/project": False}})
        patch = mock.patch("tokimon.tools.mcp.container_executor.gateway.request", side_effect=worker)
        patch.start(); self.addCleanup(patch.stop)

    def call(self, name, args):
        return self.server.handle({"id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}}, "c" * 32)["result"]

    def test_two_chats_route_explicit_ids_with_project_defaults_and_scoped_jobs(self):
        for identity in (A, B, A):
            tool = self.call("executor_start", {"workspace_id": identity, "command": "pwd"})
            payload = json.loads(tool["content"][0]["text"])
            self.assertEqual(payload["directory"], identity)
            self.assertEqual(payload["cwd"], "/mounts/project")
        self.assertTrue(self.call("executor_status", {"workspace_id": B, "job_id": "1" * 32})["isError"])
        self.assertNotIn("isError", self.call("executor_status", {"workspace_id": A, "job_id": "1" * 32}))
        self.assertTrue(all(correlation == "c" * 32 for _, _, correlation in self.requests))

    def test_no_missing_unknown_traversal_or_lifecycle_route_fallback(self):
        for args in ({"command": "pwd"}, {"workspace_id": "../../host", "command": "pwd"},
                     {"workspace_id": "ws-" + "d" * 20, "command": "pwd"}, {"workspace_id": A, "command": "pwd", "socket_path": "/tmp/host"}):
            self.assertTrue(self.call("executor_start", args)["isError"])
        self.assertTrue(self.call("tokimon/maintenance", {"workspace_id": A})["isError"])
        self.assertFalse(self.requests)

    def test_discovery_and_catalog_do_not_need_dynamic_tool_refresh(self):
        tools = self.server.handle({"id": 1, "method": "tools/list"}, "c" * 32)["result"]["tools"]
        self.assertEqual(len(tools), 10)
        self.assertTrue(all("workspace_id" in t["inputSchema"]["required"] for t in tools if t["name"] != "executor_workspaces"))
        workspaces = json.loads(self.call("executor_workspaces", {})["content"][0]["text"])["workspaces"]
        self.assertEqual([w["workspace_id"] for w in workspaces], [A, B])
        self.assertTrue(all(w["local_ready"] for w in workspaces))
        self.assertTrue(all("socket_volume" not in w and "owner_id" not in w for w in workspaces))

    def test_bad_worker_response_is_isolated_from_healthy_discovery(self):
        good = result({"deployment": B, "ok": True, "roots": {"/mounts/project": False}})
        malformed = [None, {"result": None}, {"result": {"content": []}},
                     {"result": {"content": [None]}}, result(None), result({"ok": True, "deployment": A, "roots": []})]
        for bad in malformed:
            def response(method, params, correlation, *, socket_path, timeout):
                return bad if A in socket_path else good
            with self.subTest(bad=bad), mock.patch("tokimon.tools.mcp.container_executor.gateway.request", side_effect=response):
                payload = self.call("executor_workspaces", {})
                self.assertNotIn("isError", payload)
                a, b = json.loads(payload["content"][0]["text"])["workspaces"]
                self.assertEqual(a["error"], "workspace_unavailable")
                self.assertTrue(b["local_ready"])

    def test_stdio_surrogate_id_does_not_terminate_following_requests(self):
        script = 'from tokimon.tools.mcp.container_executor import gateway; gateway.load_routes=lambda:{"schema_version":1,"routes":{}}; raise SystemExit(gateway.serve())'
        payload = b'{"id":"\\ud800","method":"ping"}\n{"id":"normal","method":"ping"}\n'
        completed = subprocess.run([sys.executable, "-c", script], input=payload, capture_output=True, timeout=5)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        replies = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual([r["id"] for r in replies], ["\ud800", "normal"])
        self.assertEqual(replies[-1]["result"], {})
