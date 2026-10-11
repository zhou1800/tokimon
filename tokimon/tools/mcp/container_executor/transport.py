"""Fixed MCP socket connector, used after privilege drop inside an Executor."""

from __future__ import annotations

import json
import os
import selectors
import socket
import sys
import time
import uuid

from . import MAX_FRAME
from .isolation import no_new_privileges
from .logging import log_failure

SOCKET_PATH = "/run/tokimon/mcp.sock"


def connect(path: str = SOCKET_PATH, timeout: float = 15) -> socket.socket:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(timeout)
        connection.connect(path)
        connection.settimeout(None)
    except Exception:
        connection.close()
        raise
    return connection


def relay() -> int:
    no_new_privileges()
    with connect() as connection:
        with selectors.SelectSelector() as selector:
            stdin = sys.stdin.fileno()
            selector.register(stdin, selectors.EVENT_READ, "input")
            selector.register(connection, selectors.EVENT_READ, "output")
            buffers = {"input": bytearray(), "output": bytearray()}
            input_closed = False
            while True:
                for key, _ in selector.select():
                    data = os.read(stdin, 65536) if key.data == "input" else connection.recv(65536)
                    if not data:
                        if key.data == "output":
                            return 0 if input_closed and not buffers["output"] else 1
                        if buffers["input"]:
                            return 1
                        input_closed = True
                        selector.unregister(stdin)
                        connection.shutdown(socket.SHUT_WR)
                        continue
                    buffer = buffers[key.data]
                    buffer.extend(data)
                    while b"\n" in buffer:
                        end = buffer.index(b"\n") + 1
                        if end > MAX_FRAME:
                            return 1
                        line = bytes(buffer[:end])
                        del buffer[:end]
                        if key.data == "input":
                            connection.sendall(line)
                        else:
                            sys.stdout.buffer.write(line)
                            sys.stdout.buffer.flush()
                    if len(buffer) > MAX_FRAME:
                        return 1


def request(method: str, params: dict | None = None, correlation_id: str | None = None, *, socket_path: str = SOCKET_PATH, timeout: float = 15) -> dict:
    no_new_privileges()
    deadline = time.monotonic() + timeout
    def remaining():
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise TimeoutError("executor request deadline exceeded")
        return budget
    params = dict(params or {})
    if correlation_id:
        params["_meta"] = {"tokimonCorrelationId": correlation_id}
    frame = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode() + b"\n"
    if len(frame) > MAX_FRAME:
        raise ValueError("request exceeds frame limit")
    with connect(socket_path, remaining()) as connection:
        connection.settimeout(remaining())
        connection.sendall(frame)
        buffer = bytearray()
        while b"\n" not in buffer and len(buffer) <= MAX_FRAME:
            connection.settimeout(remaining())
            data = connection.recv(min(65536, MAX_FRAME + 1 - len(buffer)))
            if not data:
                break
            buffer.extend(data)
        remaining()
        line = bytes(buffer).split(b"\n", 1)[0] + b"\n" if b"\n" in buffer else bytes(buffer)
        if len(line) > MAX_FRAME or not line.endswith(b"\n"):
            raise RuntimeError("invalid bounded executor response")
        return json.loads(line)


if __name__ == "__main__":
    try:
        command = sys.argv[1]
        if command == "relay":
            raise SystemExit(relay())
        if command == "request":
            print(json.dumps(request(sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else None, sys.argv[4] if len(sys.argv) > 4 else None)))
        else:
            raise ValueError("unknown connector operation")
    except Exception as exc:
        log_failure("transport", uuid.uuid4().hex, exc)
        raise SystemExit(1) from None
