"""Skill registry for built-in and generated skills."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skills.spec import SkillSpec


@dataclass
class SkillEntry:
    spec: SkillSpec
    module: Any


@dataclass(frozen=True)
class PromptSkillAsset:
    prompt_template: str
    prompt_path: Path | None = None


class SkillRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.generated_dir = root / "skills_generated"
        self.manifest_path = self.generated_dir / "manifest.json"
        self._skills: dict[str, SkillEntry] = {}

    def load(self) -> None:
        self._skills.clear()
        self._load_builtin()
        self._load_generated()

    def _load_builtin(self) -> None:
        try:
            module = importlib.import_module("skills_builtin")
        except ModuleNotFoundError:
            return
        specs = getattr(module, "SKILLS", [])
        for spec in specs:
            self._skills[spec.name] = SkillEntry(spec=spec, module=module)

    def _load_generated(self) -> None:
        if not self.manifest_path.exists():
            return
        data = json.loads(self.manifest_path.read_text())
        for entry in data.get("skills", []):
            if not isinstance(entry, dict):
                continue
            spec_data = entry.get("spec")
            if not isinstance(spec_data, dict):
                spec_data = {}
            kind = (spec_data.get("kind") or entry.get("kind") or "code")
            spec_data = {**spec_data, "kind": kind}
            try:
                spec = SkillSpec.from_dict(spec_data)
            except (TypeError, ValueError):
                continue

            if spec.kind == "prompt":
                prompt_template = str(spec.prompt_template or "").strip()
                prompt_path_value = (
                    str(spec.prompt_path or "").strip()
                    or str(entry.get("prompt_path") or "").strip()
                    or str(spec_data.get("prompt_path") or "").strip()
                    or f"prompts/{spec.name}.md"
                )
                prompt_path = _resolve_prompt_path(self.generated_dir, prompt_path_value)
                if not prompt_template:
                    if prompt_path is None or not prompt_path.exists():
                        continue
                    prompt_template = prompt_path.read_text()
                if not spec.prompt_path:
                    spec.prompt_path = prompt_path_value
                module = PromptSkillAsset(prompt_template=prompt_template, prompt_path=prompt_path)
                self._skills[spec.name] = SkillEntry(spec=spec, module=module)
                continue

            module_name = entry.get("module") or spec_data.get("module")
            if not module_name:
                continue
            module = importlib.import_module(str(module_name))
            if not spec.module:
                spec.module = str(module_name)
            self._skills[spec.name] = SkillEntry(spec=spec, module=module)

    def reload(self) -> None:
        for entry in self._skills.values():
            if entry.spec.kind != "code":
                continue
            importlib.reload(entry.module)
        self.load()

    def list_skills(self) -> list[SkillSpec]:
        return [entry.spec for entry in self._skills.values()]

    def get(self, name: str) -> SkillEntry | None:
        return self._skills.get(name)


def _resolve_prompt_path(generated_dir: Path, prompt_path_value: str) -> Path | None:
    prompt_path_value = str(prompt_path_value or "").strip()
    if not prompt_path_value:
        return None
    raw_path = Path(prompt_path_value)
    base = generated_dir.resolve()
    candidate = raw_path if raw_path.is_absolute() else (generated_dir / raw_path)
    try:
        resolved = candidate.resolve()
    except FileNotFoundError:
        resolved = candidate.absolute()
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return resolved
