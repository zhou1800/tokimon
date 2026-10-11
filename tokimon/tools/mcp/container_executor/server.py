"""Bounded MCP tools and Unix-socket serving inside the unprivileged executor."""

from __future__ import annotations

import json
import logging
import os
import socketserver
import threading
import uuid
from pathlib import Path

from tokimon.tools.mcp._protocol import error_response, handle_request, response, tool_result, json_wire_text

from . import MAX_FRAME, VERSION
from .config import fields, integer
from .files import FileBoundary, MAX_BYTES
from .isolation import process_identity, verify_unprivileged
from .jobs import JobRuntime
from .logging import logger, log_failure


TOOL_ARGUMENTS = {
    "executor_health": ({}, []),
    "executor_start": ({"command": "string", "cwd": "string", "timeout_seconds": "integer"}, ["command"]),
    "executor_status": ({"job_id": "string"}, ["job_id"]),
    "executor_read": ({"job_id": "string", "stream": "string", "offset": "integer", "max_bytes": "integer"}, ["job_id"]),
    "executor_cancel": ({"job_id": "string"}, ["job_id"]),
    "executor_jobs": ({"limit": "integer"}, []),
    "executor_list": ({"path": "string", "limit": "integer"}, []),
    "executor_file_read": ({"path": "string", "offset": "integer", "max_bytes": "integer", "encoding": "string"}, ["path"]),
    "executor_file_write": ({"path": "string", "content": "string", "encoding": "string", "overwrite": "boolean"}, ["path", "content"]),
}
DESCRIPTIONS = {
    "executor_health": "Inspect executor identity, isolation, roots and resource limits.",
    "executor_start": "Start a Bash command independently of this MCP connection; poll/read/cancel by returned job_id.",
    "executor_status": "Read persistent job status and captured output sizes.",
    "executor_read": "Read bounded captured stdout/stderr by byte offset.",
    "executor_cancel": "Request process-group cancellation; terminal outcomes are unchanged.",
    "executor_jobs": "List recent jobs for reconnect/continuation.",
    "executor_list": "List bounded entries under /workspace or a declared /mounts alias.",
    "executor_file_read": "Read a bounded regular file, rejecting symlink traversal.",
    "executor_file_write": "Create a bounded regular file; replacement requires overwrite=true.",
}


def list_tools() -> list[dict]:
    return [{"name": name, "description": DESCRIPTIONS[name],
             "inputSchema": {"type": "object", "properties": {k: {"type": t} for k, t in props.items()},
                             "required": required, "additionalProperties": False}}
            for name, (props, required) in TOOL_ARGUMENTS.items()]


class ExecutorServer:
    def __init__(self, config: dict, state: Path, policy: dict, *, log: logging.Logger | None = None):
        self.config, self.policy = config, policy
        self.boot_id = os.environ.get("TOKIMON_BOOT_ID") or uuid.uuid4().hex
        self.log = log or logger()
        self.files = FileBoundary(config["roots"])
        self.jobs = JobRuntime(state, config["limits"], self.files, self.log)

    def call_tool(self, params: dict, *, correlation_id: str | None = None) -> dict:
        try:
            fields(params, {"name"}, {"arguments", "_meta"})
            name = params["name"]
            if not isinstance(name, str) or name not in TOOL_ARGUMENTS:
                raise ValueError("unknown executor tool")
            props, required = TOOL_ARGUMENTS[name]
            args = fields(params.get("arguments", {}), set(required), set(props) - set(required))
            for key, value in args.items():
                expected = {"string": str, "integer": int, "boolean": bool}[props[key]]
                if type(value) is not expected:
                    raise ValueError(f"{key} has invalid type")
            return tool_result(self._dispatch(name, args, correlation_id or uuid.uuid4().hex))
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            return tool_result({"error": str(exc)}, is_error=True)

    def _dispatch(self, name: str, args: dict, correlation_id: str) -> dict:
        if name == "executor_health":
            with self.jobs.lock:
                active_jobs = sum(r["state"] == "running" for r in self.jobs.records.values())
            return {"ok": True, "version": VERSION, "deployment": self.config["name"], "boot_id": self.boot_id,
                    "identity": process_identity(), "isolation": self.policy, "roots": self.config["roots"], "limits": self.config["limits"],
                    "active_jobs": active_jobs, "maintenance": self.jobs.paused}
        if name == "executor_start":
            command = args["command"]
            if not command or "\0" in command or len(command.encode()) > 65536:
                raise ValueError("command must be nonempty and at most 64 KiB")
            timeout = integer(args.get("timeout_seconds", self.config["limits"]["timeout_seconds"]), "timeout_seconds", 1, self.config["limits"]["timeout_seconds"])
            return self.jobs.start(command, args.get("cwd", "/workspace"), timeout, correlation_id=correlation_id)
        if "job_id" in args and (len(args["job_id"]) != 32 or any(c not in "0123456789abcdef" for c in args["job_id"])):
            raise ValueError("invalid job_id")
        if name == "executor_status":
            return self.jobs.status(args["job_id"])
        if name == "executor_cancel":
            return self.jobs.cancel(args["job_id"])
        if name == "executor_jobs":
            return self.jobs.jobs(integer(args.get("limit", 20), "limit", 1, 200))
        if name == "executor_list":
            return self.files.list(args.get("path", "/workspace"), integer(args.get("limit", 100), "limit", 1, 200))
        if name == "executor_file_write":
            return self.files.write(args["path"], args["content"], self._encoding(args), args.get("overwrite", False))
        offset = integer(args.get("offset", 0), "offset", 0, 2**63 - 1)
        maximum = integer(args.get("max_bytes", 65536), "max_bytes", 1, MAX_BYTES)
        if name == "executor_read":
            stream = args.get("stream", "stdout")
            if stream not in {"stdout", "stderr"}:
                raise ValueError("stream must be stdout or stderr")
            return self.jobs.read(args["job_id"], stream, offset, maximum)
        return self.files.read(args["path"], offset, maximum, self._encoding(args))

    @staticmethod
    def _encoding(args: dict) -> str:
        encoding = args.get("encoding", "utf-8")
        if encoding not in {"utf-8", "base64"}:
            raise ValueError("encoding must be utf-8 or base64")
        return encoding

    def handle(self, request: dict, *, correlation_id: str | None = None) -> dict | None:
        if request.get("method") == "tokimon/maintenance" and "id" in request:
            try:
                params = fields(request.get("params", {}), {"paused"}, {"_meta"})
                if type(params["paused"]) is not bool:
                    raise ValueError("paused must be boolean")
                return response(request.get("id"), self.jobs.maintenance(params["paused"]))
            except ValueError as exc:
                return error_response(request.get("id"), -32602, str(exc))
        return handle_request(request, server_name="tokimon-container-executor", server_version=VERSION,
                              protocol_version="2024-11-05", list_tools=list_tools,
                              call_tool=lambda params: self.call_tool(params, correlation_id=correlation_id))


class SocketServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, path: str, executor: ExecutorServer):
        self.executor = executor
        self.connections = threading.BoundedSemaphore(32)
        super().__init__(path, RequestHandler)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()

    def handle_error(self, request, client_address):
        self.executor.log.warning("connection_failed", extra={"correlation_id": "socket"})


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(300)
        while line := self.rfile.readline(MAX_FRAME + 1):
            if len(line) > MAX_FRAME or not line.endswith(b"\n"):
                return
            correlation = uuid.uuid4().hex
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    result = error_response(None, -32600, "Request must be an object")
                else:
                    params = request.get("params", {})
                    meta = params.get("_meta", {}) if isinstance(params, dict) else {}
                    candidate = meta.get("tokimonCorrelationId") if isinstance(meta, dict) else None
                    if isinstance(candidate, str) and len(candidate) == 32 and all(c in "0123456789abcdef" for c in candidate):
                        correlation = candidate
                    result = self.server.executor.handle(request, correlation_id=correlation)
            except (ValueError, RecursionError):
                result = error_response(None, -32700, "Invalid JSON request")
            self.server.executor.log.info("request_completed", extra={"correlation_id": correlation})
            if result is not None:
                output = json_wire_text(result).encode("utf-8") + b"\n"
                if len(output) > MAX_FRAME:
                    output = json.dumps(error_response(request.get("id"), -32603, "Response exceeds frame limit")).encode() + b"\n"
                self.wfile.write(output)
                self.wfile.flush()


def main() -> None:
    config = json.loads(Path("/etc/tokimon/worker.json").read_text())
    policy = json.loads(Path("/etc/tokimon/policy.json").read_text())
    verify_unprivileged(process_identity(), config["uid"], config["gid"])
    executor = ExecutorServer(config, Path("/state/jobs"), policy)
    socket = Path("/run/tokimon/mcp.sock")
    socket.unlink(missing_ok=True)
    with SocketServer(str(socket), executor) as server:
        socket.chmod(0o666)
        executor.log.info("server_ready", extra={"correlation_id": executor.boot_id})
        server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_failure("service", os.environ.get("TOKIMON_BOOT_ID") or uuid.uuid4().hex, exc)
        raise SystemExit(1) from None
