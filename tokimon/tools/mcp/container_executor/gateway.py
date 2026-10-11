"""Fixed workspace-aware MCP dispatcher; no host filesystem or Docker control port."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tokimon.tools.mcp._protocol import error_response, handle_request, tool_result, json_wire_text, utf8_safe_text

from . import MAX_FRAME, VERSION
from .config import fields, integer
from .isolation import no_new_privileges, process_identity
from .logging import logger, log_failure
from .server import TOOL_ARGUMENTS, list_tools as worker_tools
from .transport import request

WORKSPACE_ID = re.compile(r"ws-[0-9a-f]{20}\Z")


def validate_routes(payload: object) -> dict:
    config = fields(payload, {"schema_version", "routes"})
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("gateway schema_version must be 1")
    routes = config["routes"]
    if not isinstance(routes, dict) or len(routes) > 256:
        raise ValueError("gateway routes must be bounded to 256 workspaces")
    for identity, route in routes.items():
        if not isinstance(identity, str) or not WORKSPACE_ID.fullmatch(identity):
            raise ValueError("invalid workspace_id")
        fields(route, {"directory", "socket_volume", "owner_id"})
        if not isinstance(route["directory"], str) or len(route["directory"].encode()) > 4096 or not route["directory"].startswith("/"):
            raise ValueError("invalid workspace directory label")
        if not isinstance(route["socket_volume"], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", route["socket_volume"]):
            raise ValueError("invalid socket volume identity")
        if not isinstance(route["owner_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", route["owner_id"]):
            raise ValueError("invalid socket owner")
    return config


def list_tools() -> list[dict]:
    tools = worker_tools()
    for tool in tools:
        schema = tool["inputSchema"]
        schema["properties"]["workspace_id"] = {"type": "string", "description": "Explicit ID returned by executor_workspaces."}
        schema["required"] = ["workspace_id", *schema["required"]]
    return [{"name": "executor_workspaces", "description": "List registered directory workspaces and actual availability before selecting a workspace_id.",
             "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}, "additionalProperties": False}}, *tools]


class GatewayServer:
    def __init__(self, config: dict):
        self.routes = validate_routes(config)["routes"]
        self.log = logger()

    def _request(self, identity: str, name: str, args: dict, correlation: str) -> dict:
        if not isinstance(identity, str) or not WORKSPACE_ID.fullmatch(identity) or identity not in self.routes:
            raise ValueError("unknown workspace_id; call executor_workspaces")
        result = request("tools/call", {"name": name, "arguments": args}, correlation,
                         socket_path=f"/run/workspaces/{identity}/mcp.sock", timeout=2 if name == "executor_health" else 15)
        if not isinstance(result, dict):
            raise ValueError("invalid workspace response envelope")
        if "error" in result:
            raise RuntimeError("workspace request failed")
        tool = fields(result.get("result"), {"content"}, {"isError"})
        if type(tool.get("isError", False)) is not bool or not isinstance(tool["content"], list) or not tool["content"]:
            raise ValueError("invalid workspace tool result")
        for content in tool["content"]:
            fields(content, {"type", "text"})
            if content["type"] != "text" or not isinstance(content["text"], str):
                raise ValueError("invalid workspace tool content")
            content["text"] = utf8_safe_text(content["text"])
        return tool

    def _workspace(self, item: tuple[str, dict], correlation: str) -> dict:
        identity, route = item
        result = {"workspace_id": identity, "directory": route["directory"], "project_root": "/mounts/project", "local_ready": False}
        try:
            health = self._request(identity, "executor_health", {}, correlation)
            payload = json.loads(health["content"][0]["text"])
            if not isinstance(payload, dict) or not isinstance(payload.get("roots"), dict) or any(not isinstance(k, str) or type(v) is not bool for k, v in payload["roots"].items()):
                raise ValueError("invalid workspace health payload")
            result["local_ready"] = not health.get("isError", False) and payload.get("deployment") == identity and payload.get("ok") is True
            if result["local_ready"]:
                result["roots"] = payload["roots"]
        except (ValueError, KeyError, OSError, RuntimeError, RecursionError):
            pass
        if not result["local_ready"]:
            result["error"] = "workspace_unavailable"
        return result

    def call_tool(self, params: dict, correlation: str) -> dict:
        try:
            fields(params, {"name"}, {"arguments", "_meta"})
            name = params["name"]
            if name == "executor_workspaces":
                args = fields(params.get("arguments", {}), set(), {"limit"})
                limit = integer(args.get("limit", 200), "limit", 1, 200)
                items = sorted(self.routes.items())[:limit]
                with ThreadPoolExecutor(max_workers=32) as pool:
                    workspaces = list(pool.map(lambda item: self._workspace(item, correlation), items))
                return tool_result({"workspaces": workspaces, "truncated": len(self.routes) > limit})
            if not isinstance(name, str) or name not in TOOL_ARGUMENTS:
                raise ValueError("unknown gateway tool")
            properties, required = TOOL_ARGUMENTS[name]
            args = dict(fields(params.get("arguments", {}), {"workspace_id", *required}, set(properties) - set(required)))
            identity = args.pop("workspace_id")
            for key, value in args.items():
                if type(value) is not {"string": str, "integer": int, "boolean": bool}[properties[key]]:
                    raise ValueError(f"{key} has invalid type")
            if name == "executor_start":
                args.setdefault("cwd", "/mounts/project")
            elif name == "executor_list":
                args.setdefault("path", "/mounts/project")
            return self._request(identity, name, args, correlation)
        except (ValueError, KeyError, OSError, RuntimeError, RecursionError) as exc:
            return tool_result({"error": str(exc)}, is_error=True)

    def handle(self, payload: dict, correlation: str) -> dict | None:
        return handle_request(payload, server_name="tokimon-workspace-gateway", server_version=VERSION,
                              protocol_version="2024-11-05", list_tools=list_tools,
                              call_tool=lambda params: self.call_tool(params, correlation))


def load_routes() -> dict:
    source = Path("/etc/tokimon/gateway.json")
    if source.stat().st_size > MAX_FRAME:
        raise ValueError("gateway configuration exceeds frame limit")
    return validate_routes(json.loads(source.read_text()))


def serve() -> int:
    no_new_privileges()
    server = GatewayServer(load_routes())
    while line := sys.stdin.buffer.readline(MAX_FRAME + 1):
        if len(line) > MAX_FRAME or not line.endswith(b"\n"):
            return 1
        correlation = uuid.uuid4().hex
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                result = error_response(None, -32600, "Request must be an object")
            else:
                result = server.handle(payload, correlation)
        except (ValueError, RecursionError):
            result = error_response(None, -32700, "Invalid JSON request")
        server.log.info("gateway_request_completed", extra={"correlation_id": correlation})
        if result is not None:
            output = json_wire_text(result).encode("utf-8") + b"\n"
            if len(output) > MAX_FRAME:
                output = json.dumps(error_response(None, -32603, "Response exceeds frame limit")).encode() + b"\n"
            sys.stdout.buffer.write(output)
            sys.stdout.buffer.flush()
    return 0


def probe() -> dict:
    no_new_privileges()
    routes = load_routes()
    configured = json.loads(Path("/etc/tokimon/tunnel.json").read_text()) is not None
    ready = False
    if configured:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/readyz", timeout=2) as result:
                ready = result.status == 200
        except (OSError, urllib.error.URLError):
            pass
    return {"ok": True, "identity": process_identity(), "workspaces": sorted(routes["routes"]),
            "discovery": GatewayServer(routes).handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, uuid.uuid4().hex),
            "tunnel_configured": configured, "tunnel_ready": ready, "remote_acceptance": "unverified"}


def supervise() -> int:
    no_new_privileges()
    tunnel = json.loads(Path("/etc/tokimon/tunnel.json").read_text())
    if tunnel is None:
        threading.Event().wait()
        return 0
    subprocess.run([
        "/usr/local/bin/tunnel-client", "init", "--profile", "tokimon", "--profile-dir", "/tmp/tunnel-profile",
        "--tunnel-id", tunnel["tunnel_id"], "--control-plane-api-key-ref", "file:/run/secrets/tunnel-key",
        "--mcp-command", "/usr/local/bin/python -I -m tokimon.tools.mcp.container_executor.gateway serve",
        "--health-listen-addr", "127.0.0.1:8080",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Client diagnostics may contain provider details; keep them off Docker logs.
    with open(os.devnull, "wb") as sink:
        process = subprocess.Popen(["/usr/local/bin/tunnel-client", "run", "--profile", "tokimon",
                                    "--profile-dir", "/tmp/tunnel-profile"], stdout=sink, stderr=sink)
        return process.wait()


if __name__ == "__main__":
    try:
        operation = sys.argv[1]
        if operation == "serve":
            raise SystemExit(serve())
        if operation == "probe":
            print(json.dumps(probe()))
        elif operation == "supervise":
            raise SystemExit(supervise())
        else:
            raise ValueError("unknown gateway operation")
    except Exception as exc:
        log_failure("gateway", uuid.uuid4().hex, exc)
        raise SystemExit(1) from None
