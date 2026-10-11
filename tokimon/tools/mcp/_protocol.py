"""MCP envelopes, method dispatch and stateless stdio processing."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Mapping, TextIO


def json_text(payload: Any) -> str:
    return json_wire_text(payload, indent=2)


def utf8_safe_text(text: str) -> str:
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def json_wire_text(payload: Any, *, ensure_ascii: bool = False, indent: int | None = None) -> str:
    """Keep Unicode readable while escaping surrogates for strict UTF-8 transports."""
    return utf8_safe_text(json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent))


def tool_result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": json_text(payload)}]}
    if is_error:
        result["isError"] = True
    return result


def response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": utf8_safe_text(message)}}


def handle_request(
    request: Mapping[str, Any],
    *,
    server_name: str,
    server_version: str,
    protocol_version: str,
    list_tools: Callable[[], list[dict[str, Any]]],
    call_tool: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    if not isinstance(method, str):
        return error_response(request_id, -32600, "Method must be a string")
    if "id" not in request:
        return None
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    try:
        if method == "initialize":
            return response(request_id, {
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": server_name, "version": server_version},
            })
        if method == "ping":
            return response(request_id, {})
        if method == "tools/list":
            return response(request_id, {"tools": list_tools()})
        if method == "tools/call":
            return response(request_id, call_tool(params))
        return error_response(request_id, -32601, f"Method not found: {method}")
    except Exception as exc:  # noqa: BLE001 - JSON-RPC must return errors.
        return error_response(request_id, -32603, str(exc))


def serve_stdio(
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    server: Any,
    logger: logging.Logger,
    log_event: Callable[..., None],
    method_as_text: bool = False,
    ensure_ascii_errors: bool = False,
) -> int:
    log_event(logger, logging.INFO, "server_start", correlation_id="server-start")
    for line in input_stream:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except (ValueError, RecursionError) as exc:
            log_event(logger, logging.WARNING, "invalid_jsonrpc_request", correlation_id="parse-error")
            print(json_wire_text(error_response(None, -32700, str(exc)), ensure_ascii=ensure_ascii_errors),
                  file=output_stream, flush=True)
            continue
        if not isinstance(request, dict):
            log_event(logger, logging.WARNING, "invalid_jsonrpc_request", correlation_id="invalid-request")
            print(json_wire_text(error_response(None, -32600, "Request must be an object"),
                             ensure_ascii=ensure_ascii_errors), file=output_stream, flush=True)
            continue
        method = request.get("method")
        log_event(logger, logging.INFO, "request_received", request_id=request.get("id"),
                  method=str(method) if method_as_text else method)
        result = server.handle(request)
        if result is not None:
            print(json_wire_text(result), file=output_stream, flush=True)
    return 0
