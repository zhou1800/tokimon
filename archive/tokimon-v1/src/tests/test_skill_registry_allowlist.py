from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

from skills.registry import SkillRegistry


def _write_generated_manifest(root: Path, *, skill_name: str, module_name: str) -> None:
    generated_dir = root / "skills_generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    (generated_dir / "manifest.json").write_text(
        json.dumps(
            {
                "skills": [
                    {
                        "kind": "code",
                        "module": module_name,
                        "spec": {"name": skill_name, "kind": "code", "purpose": "test", "contract": "none"},
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_skill_registry_skips_non_allowlisted_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module_name = f"untrusted_mod_{uuid.uuid4().hex[:8]}"
    skill_name = f"test-skill-{uuid.uuid4().hex[:8]}"
    sentinel = tmp_path / "sentinel.txt"
    (tmp_path / f"{module_name}.py").write_text(
        "from pathlib import Path\n"
        f"Path(r\"{sentinel}\").write_text(\"imported\")\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("TOKIMON_SKILL_MODULE_ALLOWLIST", raising=False)

    _write_generated_manifest(tmp_path, skill_name=skill_name, module_name=module_name)

    registry = SkillRegistry(tmp_path)
    registry.load()

    assert registry.get(skill_name) is None
    assert module_name not in sys.modules
    assert not sentinel.exists()


def test_skill_registry_allows_module_with_env_allowlist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module_name = f"untrusted_mod_{uuid.uuid4().hex[:8]}"
    skill_name = f"test-skill-{uuid.uuid4().hex[:8]}"
    sentinel = tmp_path / "sentinel.txt"
    (tmp_path / f"{module_name}.py").write_text(
        "from pathlib import Path\n"
        f"Path(r\"{sentinel}\").write_text(\"imported\")\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("TOKIMON_SKILL_MODULE_ALLOWLIST", module_name)

    _write_generated_manifest(tmp_path, skill_name=skill_name, module_name=module_name)

    try:
        registry = SkillRegistry(tmp_path)
        registry.load()
        entry = registry.get(skill_name)
        assert entry is not None
        assert sentinel.exists()
        assert entry.provenance.get("module") == module_name
        sha = entry.provenance.get("sha256")
        assert isinstance(sha, str) and len(sha) == 64
    finally:
        sys.modules.pop(module_name, None)

