"""MCP tool declaration, input validation, and broker dispatch boundary."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, cast

from tokimon.tools.mcp._protocol import tool_result as _tool_result
from tokimon.tools.mcp.remote_agent_broker.model import (
    AGENT_RUNNER_NAMES,
    RemoteAgentBrokerError,
    RequestContext,
)


def _build_mcp_tools() -> list[dict[str, Any]]:
    empty_schema = {"type": "object", "additionalProperties": False, "properties": {}}
    return [
        {
            "name": "remote_agent_health",
            "description": "Report broker availability, runtime root, transport settings, and runner discovery.",
            "inputSchema": empty_schema,
        },
        {
            "name": "remote_agent_list_agents",
            "description": "List available remote shell and AI CLI runners.",
            "inputSchema": empty_schema,
        },
        {
            "name": "remote_agent_start",
            "description": "Start a remote AI-agent job and return a job id.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["agent", "cwd", "prompt"],
                "properties": {
                    "agent": {"type": "string", "enum": list(AGENT_RUNNER_NAMES)},
                    "cwd": {"type": "string"},
                    "prompt": {"type": "string"},
                    "runner_options": {"type": "object"},
                    "timeout_sec": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Compatibility alias for idle_timeout_sec on AI-agent jobs.",
                    },
                    "idle_timeout_sec": {"type": "integer", "minimum": 1},
                    "max_wall_time_sec": {"type": "integer", "minimum": 1},
                    "ttl_days": {"type": "integer", "minimum": 1},
                    "capture_limit_bytes": {"type": "integer", "minimum": 1},
                },
            },
        },
        {
            "name": "remote_agent_status",
            "description": "Read current job state.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["job_id"],
                "properties": {"job_id": {"type": "string"}},
            },
        },
        {
            "name": "remote_agent_read",
            "description": "Read incremental stdout, stderr, transcript, or final output for a job.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["job_id"],
                "properties": {
                    "job_id": {"type": "string"},
                    "stream": {
                        "type": "string",
                        "enum": ["stdout", "stderr", "transcript", "final"],
                    },
                    "since_offset": {"type": "integer", "minimum": 0},
                    "max_bytes": {"type": "integer", "minimum": 1},
                },
            },
        },
        {
            "name": "remote_agent_send_input",
            "description": "Send stdin text to a running job.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["job_id", "text"],
                "properties": {
                    "job_id": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
        },
        {
            "name": "remote_agent_cancel",
            "description": "Cancel a running job.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["job_id"],
                "properties": {
                    "job_id": {"type": "string"},
                    "signal": {"type": "string"},
                },
            },
        },
        {
            "name": "remote_agent_diff",
            "description": "Return git diff for a job working tree when available.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["job_id"],
                "properties": {
                    "job_id": {"type": "string"},
                    "base_ref": {"type": "string"},
                },
            },
        },
        {
            "name": "remote_agent_artifacts",
            "description": "List persisted artifacts for a job.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["job_id"],
                "properties": {"job_id": {"type": "string"}},
            },
        },
        {
            "name": "remote_shell_start",
            "description": "Run a remote shell command, synchronously for short commands or asynchronously when async=true.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["command"],
                "properties": {
                    "command": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                    "cwd": {"type": "string"},
                    "async": {"type": "boolean"},
                    "timeout_sec": {"type": "integer", "minimum": 1},
                    "env": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "capture_limit_bytes": {"type": "integer", "minimum": 1},
                },
            },
        },
    ]


_MCP_TOOLS = _build_mcp_tools()
_TOOL_SCHEMAS = {
    str(tool["name"]): cast(Mapping[str, Any], tool["inputSchema"])
    for tool in _MCP_TOOLS
}


def mcp_tools() -> list[dict[str, Any]]:
    """Return independent copies of the authoritative tool declarations."""

    return deepcopy(_MCP_TOOLS)


class ToolCallDispatcher:
    """Validate one tools/call payload and dispatch it to the broker port."""

    def __init__(self, broker: Any) -> None:
        self.broker = broker

    def dispatch(
        self,
        params: Mapping[str, Any],
        *,
        request_context: RequestContext,
    ) -> dict[str, Any]:
        name_value = params.get("name")
        if not isinstance(name_value, str):
            return _tool_result({"error": "tool name must be a string"}, is_error=True)
        name = name_value
        arguments = params["arguments"] if "arguments" in params else {}
        if not isinstance(arguments, dict):
            return _tool_result({"error": "arguments must be an object"}, is_error=True)
        try:
            schema = _TOOL_SCHEMAS.get(name)
            if schema is None:
                return _tool_result({"error": f"Unknown tool: {name}"}, is_error=True)
            _validate_schema(arguments, schema, path="arguments")
            if name == "remote_agent_health":
                return _tool_result(self.broker.health())
            if name == "remote_agent_list_agents":
                return _tool_result(self.broker.list_agents())
            if name == "remote_agent_start":
                return _tool_result(
                    self.broker.start_agent(
                        agent=_required_string(arguments, "agent"),
                        cwd=_required_string(arguments, "cwd"),
                        prompt=_required_string(arguments, "prompt"),
                        runner_options=_optional_mapping(arguments, "runner_options"),
                        timeout_sec=_optional_positive_int(arguments, "timeout_sec"),
                        idle_timeout_sec=_optional_positive_int(
                            arguments, "idle_timeout_sec"
                        ),
                        max_wall_time_sec=_optional_positive_int(
                            arguments, "max_wall_time_sec"
                        ),
                        ttl_days=_optional_positive_int(arguments, "ttl_days"),
                        capture_limit_bytes=_optional_positive_int(
                            arguments, "capture_limit_bytes"
                        ),
                        request_context=request_context,
                    )
                )
            if name == "remote_agent_status":
                return _tool_result(
                    self.broker.status(_required_string(arguments, "job_id"))
                )
            if name == "remote_agent_read":
                return _tool_result(
                    self.broker.read(
                        job_id=_required_string(arguments, "job_id"),
                        stream=_optional_string(arguments, "stream") or "stdout",
                        since_offset=_optional_nonnegative_int(
                            arguments, "since_offset", 0
                        ),
                        max_bytes=_optional_positive_int(arguments, "max_bytes"),
                    )
                )
            if name == "remote_agent_send_input":
                return _tool_result(
                    self.broker.send_input(
                        job_id=_required_string(arguments, "job_id"),
                        text=_required_raw_string(arguments, "text"),
                    )
                )
            if name == "remote_agent_cancel":
                return _tool_result(
                    self.broker.cancel(
                        job_id=_required_string(arguments, "job_id"),
                        signal_name=_optional_string(arguments, "signal"),
                    )
                )
            if name == "remote_agent_diff":
                return _tool_result(
                    self.broker.diff(
                        job_id=_required_string(arguments, "job_id"),
                        base_ref=_optional_string(arguments, "base_ref"),
                    )
                )
            if name == "remote_agent_artifacts":
                return _tool_result(
                    self.broker.artifacts(
                        job_id=_required_string(arguments, "job_id")
                    )
                )
            if name == "remote_shell_start":
                return _tool_result(
                    self.broker.start_shell(
                        command=_required_command(arguments, "command"),
                        cwd=_optional_string(arguments, "cwd"),
                        async_job=_optional_bool(arguments, "async", False),
                        timeout_sec=_optional_positive_int(arguments, "timeout_sec"),
                        env=_optional_mapping(arguments, "env"),
                        capture_limit_bytes=_optional_positive_int(
                            arguments, "capture_limit_bytes"
                        ),
                        request_context=request_context,
                    )
                )
            raise AssertionError(f"Unimplemented declared tool: {name}")
        except RemoteAgentBrokerError as exc:
            return _tool_result({"error": str(exc)}, is_error=True)


def _required_string(arguments: Mapping[str, Any], name: str) -> str:
    return cast(str, arguments[name])


def _required_raw_string(arguments: Mapping[str, Any], name: str) -> str:
    value = cast(str, arguments[name])
    if value == "":
        raise RemoteAgentBrokerError(f"{name} is required")
    return value


def _optional_string(arguments: Mapping[str, Any], name: str) -> str | None:
    value = arguments.get(name)
    return cast(str | None, value)


def _optional_bool(arguments: Mapping[str, Any], name: str, default: bool) -> bool:
    value = arguments.get(name)
    return default if value is None else cast(bool, value)


def _optional_positive_int(arguments: Mapping[str, Any], name: str) -> int | None:
    return cast(int | None, arguments.get(name))


def _optional_nonnegative_int(
    arguments: Mapping[str, Any],
    name: str,
    default: int,
) -> int:
    value = cast(int | None, arguments.get(name))
    return default if value is None else value


def _optional_mapping(
    arguments: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any] | None:
    return cast(Mapping[str, Any] | None, arguments.get(name))


def _required_command(arguments: Mapping[str, Any], name: str) -> str | list[str]:
    value = arguments.get(name)
    if isinstance(value, str):
        if not value.strip():
            raise RemoteAgentBrokerError(f"{name} must not be empty")
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if not value:
            raise RemoteAgentBrokerError(f"{name} must not be empty")
        return value
    raise RemoteAgentBrokerError(f"{name} must be a string or string array")


def _validate_schema(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    """Validate the JSON Schema subset used by the MCP tool declarations."""

    alternatives = schema.get("oneOf")
    if isinstance(alternatives, list):
        failures: list[str] = []
        for alternative in alternatives:
            try:
                _validate_schema(value, cast(Mapping[str, Any], alternative), path=path)
            except RemoteAgentBrokerError as exc:
                failures.append(str(exc))
            else:
                return
        detail = failures[0] if failures else f"{path} does not match any allowed shape"
        raise RemoteAgentBrokerError(detail)

    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise RemoteAgentBrokerError(f"{path} must be an object")
        properties = cast(Mapping[str, Mapping[str, Any]], schema.get("properties", {}))
        required = cast(list[str], schema.get("required", []))
        missing = [name for name in required if name not in value]
        if missing:
            raise RemoteAgentBrokerError(f"{path}.{missing[0]} is required")
        extras = sorted(set(value) - set(properties))
        additional = schema.get("additionalProperties", True)
        if extras and additional is False:
            raise RemoteAgentBrokerError(
                f"{path} contains unsupported field(s): {', '.join(extras)}"
            )
        for name, item in value.items():
            property_schema = properties.get(name)
            if property_schema is not None:
                _validate_schema(item, property_schema, path=f"{path}.{name}")
            elif isinstance(additional, Mapping):
                _validate_schema(
                    item,
                    cast(Mapping[str, Any], additional),
                    path=f"{path}.{name}",
                )
    elif expected == "array":
        if not isinstance(value, list):
            raise RemoteAgentBrokerError(f"{path} must be an array")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema(
                    item,
                    cast(Mapping[str, Any], item_schema),
                    path=f"{path}[{index}]",
                )
    elif expected == "string":
        if not isinstance(value, str):
            raise RemoteAgentBrokerError(f"{path} must be a string")
    elif expected == "boolean":
        if type(value) is not bool:
            raise RemoteAgentBrokerError(f"{path} must be a boolean")
    elif expected == "integer":
        if type(value) is not int:
            raise RemoteAgentBrokerError(f"{path} must be an integer")

    allowed = schema.get("enum")
    if isinstance(allowed, list) and value not in allowed:
        choices = ", ".join(str(item) for item in allowed)
        raise RemoteAgentBrokerError(f"{path} must be one of {choices}")
    minimum = schema.get("minimum")
    if type(value) is int and type(minimum) is int and value < minimum:
        raise RemoteAgentBrokerError(f"{path} must be >= {minimum}")
