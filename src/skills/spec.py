"""Skill specification structures."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal


SkillKind = Literal["code", "prompt"]
SkillContract = dict[str, Any] | str


@dataclass
class SkillSpec:
    name: str
    purpose: str
    contract: SkillContract
    kind: SkillKind = "code"
    preconditions: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    retrieval_prefs: dict[str, Any] = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    safety_notes: dict[str, Any] = field(default_factory=lambda: {"hard": [], "soft": []})
    cost_energy_notes: str = ""
    validation_method: Any = field(default_factory=dict)
    version: str = "0.0.0"
    deprecation_policy: str = ""
    module: str | None = None
    prompt_template: str | None = None
    prompt_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillSpec":
        if not isinstance(data, dict):
            raise TypeError("SkillSpec.from_dict expects a dict")

        name = str(data.get("name") or "").strip()
        if not name:
            raise ValueError("SkillSpec missing required field: name")

        normalized: dict[str, Any] = {"name": name}
        normalized["purpose"] = data.get("purpose", "")
        normalized["contract"] = data.get("contract", "")
        normalized["kind"] = data.get("kind") or "code"

        for field_info in fields(cls):
            key = field_info.name
            if key in {"name", "purpose", "contract", "kind"}:
                continue
            if key in data:
                normalized[key] = data[key]

        if normalized.get("required_tools") is None:
            normalized["required_tools"] = []
        if normalized.get("retrieval_prefs") is None:
            normalized["retrieval_prefs"] = {}
        if normalized.get("preconditions") is None:
            normalized["preconditions"] = []
        if normalized.get("failure_modes") is None:
            normalized["failure_modes"] = []

        prompt_template = normalized.get("prompt_template")
        if prompt_template is not None and not isinstance(prompt_template, str):
            normalized["prompt_template"] = str(prompt_template)
        prompt_path = normalized.get("prompt_path")
        if prompt_path is not None and not isinstance(prompt_path, str):
            normalized["prompt_path"] = str(prompt_path)

        safety_notes = normalized.get("safety_notes")
        if not isinstance(safety_notes, dict):
            normalized["safety_notes"] = {"hard": [], "soft": []}
        else:
            normalized["safety_notes"] = {
                "hard": safety_notes.get("hard", []) if isinstance(safety_notes.get("hard"), list) else [],
                "soft": safety_notes.get("soft", []) if isinstance(safety_notes.get("soft"), list) else [],
            }

        kind = str(normalized.get("kind") or "code").strip().lower()
        normalized["kind"] = "prompt" if kind == "prompt" else "code"

        allowed = {field_info.name for field_info in fields(cls)}
        filtered = {key: value for key, value in normalized.items() if key in allowed}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        allowed = {field_info.name for field_info in fields(self)}
        return {name: getattr(self, name) for name in allowed}
