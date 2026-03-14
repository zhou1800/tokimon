"""Central registry for tool-call risk classification.

This is inspired by OpenClaw's single-source-of-truth approach to "dangerous tools"
to avoid policy drift across runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


RiskTier = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ToolRisk:
    risk_tier: RiskTier
    requires_approval: bool
    notes: str = ""


def _normalize(value: str) -> str:
    return str(value or "").strip().lower()


_REGISTRY: dict[tuple[str, str], ToolRisk] = {
    ("file", "read"): ToolRisk(risk_tier="low", requires_approval=False, notes="read-only workspace access"),
    ("file", "write"): ToolRisk(risk_tier="high", requires_approval=True, notes="writes to workspace"),
    ("patch", "apply"): ToolRisk(risk_tier="high", requires_approval=True, notes="applies patches to workspace"),
    ("grep", "search"): ToolRisk(risk_tier="low", requires_approval=False, notes="bounded repo search"),
    ("pytest", "run"): ToolRisk(risk_tier="medium", requires_approval=False, notes="executes local tests"),
    ("web", "fetch"): ToolRisk(risk_tier="medium", requires_approval=False, notes="network access"),
    ("web", "search"): ToolRisk(risk_tier="medium", requires_approval=False, notes="network access"),
}


def tool_catalog() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for (tool_name, action), risk in sorted(_REGISTRY.items()):
        catalog.append(
            {
                "tool": tool_name,
                "action": action,
                "risk_tier": risk.risk_tier,
                "requires_approval": risk.requires_approval,
                "notes": risk.notes,
            }
        )
    return catalog


def tool_risk(tool_name: str, action: str) -> ToolRisk | None:
    key = (_normalize(tool_name), _normalize(action))
    return _REGISTRY.get(key)


def risk_tier_for(tool_name: str, action: str) -> RiskTier:
    risk = tool_risk(tool_name, action)
    if risk is None:
        return "low"
    return risk.risk_tier


def requires_approval(tool_name: str, action: str) -> bool:
    risk = tool_risk(tool_name, action)
    if risk is None:
        return False
    return bool(risk.requires_approval)


def is_side_effectful(tool_name: str, action: str) -> bool:
    key = (_normalize(tool_name), _normalize(action))
    return key in {("file", "write"), ("patch", "apply")}
