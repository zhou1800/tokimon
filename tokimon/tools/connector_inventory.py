"""Read and validate optional local MCP connector inventory."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tokimon.runtime import resolve_runtime_path


DEFAULT_MCP_CONNECTOR_INVENTORY_RELATIVE_PATH = (
    Path(".tokimon") / "tools" / "mcp" / "connectors.json"
)
CONNECTOR_STATUS_VALUES = frozenset(("available", "unavailable"))


@dataclass(frozen=True)
class MCPConnector:
    id: str
    capability_id: str
    source: str
    status: str
    transport: str
    tools: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.status == "available"


class MCPConnectorInventoryError(RuntimeError):
    exit_code = 4


def _entry_error(
    source: str,
    index: int,
    field: str,
    expectation: str,
) -> MCPConnectorInventoryError:
    return MCPConnectorInventoryError(
        f"Malformed MCP connector inventory in {source}: connector {index} "
        f"field '{field}' {expectation}."
    )


def _nonempty_string(
    data: Mapping[str, object],
    field: str,
    *,
    source: str,
    index: int,
) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise _entry_error(source, index, field, "must be a nonempty string")
    return value.strip()


def _connector_from_mapping(
    data: Mapping[str, object],
    *,
    source: str,
    index: int,
) -> MCPConnector:
    connector_id = _nonempty_string(data, "id", source=source, index=index)
    capability_id = _nonempty_string(data, "capability_id", source=source, index=index)
    connector_source = _nonempty_string(data, "source", source=source, index=index)
    status = _nonempty_string(data, "status", source=source, index=index).lower()
    if status not in CONNECTOR_STATUS_VALUES:
        supported = ", ".join(sorted(CONNECTOR_STATUS_VALUES))
        raise _entry_error(source, index, "status", f"must be one of: {supported}")
    transport = _nonempty_string(data, "transport", source=source, index=index)

    raw_tools = data.get("tools")
    if not isinstance(raw_tools, list):
        raise _entry_error(source, index, "tools", "must be a list of nonempty strings")
    tools: list[str] = []
    for tool_index, tool in enumerate(raw_tools, start=1):
        if not isinstance(tool, str) or not tool.strip():
            raise _entry_error(
                source,
                index,
                f"tools[{tool_index}]",
                "must be a nonempty string",
            )
        tools.append(tool.strip())

    return MCPConnector(
        id=connector_id,
        capability_id=capability_id,
        source=connector_source,
        status=status,
        transport=transport,
        tools=tuple(tools),
    )


def load_mcp_connectors(
    *,
    workspace_root: Path,
    env: Mapping[str, str] | None = None,
) -> tuple[MCPConnector, ...]:
    effective_env = dict(os.environ if env is None else env)
    raw_payload = effective_env.get("TOKIMON_MCP_CONNECTORS_JSON")
    inventory_source = "TOKIMON_MCP_CONNECTORS_JSON"
    if raw_payload is None:
        inventory_path = resolve_runtime_path(
            DEFAULT_MCP_CONNECTOR_INVENTORY_RELATIVE_PATH,
            workspace_root=workspace_root,
        )
        if not inventory_path.exists():
            return ()
        inventory_source = str(inventory_path)
        try:
            raw_payload = inventory_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise MCPConnectorInventoryError(
                f"Unable to read MCP connector inventory from {inventory_source}: {exc}."
            ) from exc

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise MCPConnectorInventoryError(
            f"Malformed MCP connector inventory in {inventory_source}: {exc.msg}."
        ) from exc

    if isinstance(payload, list):
        raw_connectors: object = payload
    elif isinstance(payload, dict):
        raw_connectors = payload.get("connectors", [])
    else:
        raise MCPConnectorInventoryError(
            f"Malformed MCP connector inventory in {inventory_source}: expected object or list."
        )
    if not isinstance(raw_connectors, list):
        raise MCPConnectorInventoryError(
            f"Malformed MCP connector inventory in {inventory_source}: connectors must be a list."
        )

    connectors: list[MCPConnector] = []
    for index, item in enumerate(raw_connectors, start=1):
        if not isinstance(item, Mapping):
            raise MCPConnectorInventoryError(
                f"Malformed MCP connector inventory in {inventory_source}: connector {index} "
                "must be an object."
            )
        connectors.append(_connector_from_mapping(item, source=inventory_source, index=index))
    return tuple(connectors)


def available_connector_for_capability(
    capability_id: str,
    connectors: tuple[MCPConnector, ...],
) -> MCPConnector | None:
    return next(
        (
            connector
            for connector in connectors
            if connector.capability_id == capability_id and connector.available
        ),
        None,
    )
