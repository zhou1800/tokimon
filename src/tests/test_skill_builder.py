from __future__ import annotations

import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

from memory.store import MemoryStore
from skills.builder import SkillBuilder
from skills.registry import SkillRegistry
from skills.spec import SkillSpec


def _valid_skill_spec(name: str, *, kind: str) -> SkillSpec:
    required_tools = ["file"] if kind == "code" else []
    return SkillSpec(
        name=name,
        kind=kind,
        purpose="Temporary skill for testing.",
        contract={"inputs": {"request": "string"}, "outputs": {"summary": "string"}, "side_effects": "none"},
        preconditions=["Workspace is writable."],
        required_tools=required_tools,
        retrieval_prefs={"stage1": "recent context", "stage2": "component lessons", "stage3": "cross-task patterns"},
        failure_modes=["Invalid input.", "Missing context."],
        safety_notes={"hard": ["MUST NOT violate Tokimon Non-goals."], "soft": []},
        cost_energy_notes="Low: intended for tests only.",
        validation_method={"type": "pytest" if kind == "code" else "checklist"},
        version="0.1.0",
        deprecation_policy="Deprecate when superseded.",
        module=None,
        prompt_template=None,
        prompt_path=None,
    )


def _read_lesson_metadata(path: Path) -> dict[str, object]:
    header = path.read_text().split("\n", 1)[0].strip()
    return json.loads(header) if header else {}


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value.lower())


def _audit_append_entries(audit_path: Path, before_text: str) -> list[dict[str, object]]:
    if not audit_path.exists():
        return []
    after_text = audit_path.read_text(encoding="utf-8", errors="replace")
    appended = after_text[len(before_text) :] if after_text.startswith(before_text) else after_text
    entries: list[dict[str, object]] = []
    for line in appended.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries


def test_build_and_registry_code_skill(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    memory_store = MemoryStore(tmp_path / "memory")
    builder = SkillBuilder(repo_root, memory_store)

    name = f"test-code-skill-{uuid.uuid4().hex[:8]}"
    spec = _valid_skill_spec(name, kind="code")

    manifest_path = repo_root / "skills_generated" / "manifest.json"
    module_basename = name.replace("-", "_")
    module_name = f"skills_generated.{module_basename}"
    module_path = repo_root / "skills_generated" / f"{module_basename}.py"
    pycache_dir = repo_root / "skills_generated" / "__pycache__"
    shipped_test_dir = repo_root / "tests" / "skills_generated"
    shipped_test_path = shipped_test_dir / f"test_{module_basename}.py"
    shipped_test_pycache = shipped_test_dir / "__pycache__"
    shipped_test_dir_existed = shipped_test_dir.exists()
    original_manifest = manifest_path.read_text() if manifest_path.exists() else None
    audit_path = workspace_root / ".tokimon-tmp" / "audit" / "config.jsonl"
    audit_existed = audit_path.exists()
    audit_before = audit_path.read_text(encoding="utf-8", errors="replace") if audit_existed else ""

    try:
        ok = builder.build_skill(spec, "test justification")
        assert ok is True
        assert shipped_test_path.exists()

        registry = SkillRegistry(repo_root)
        registry.load()
        entry = registry.get(name)
        assert entry is not None
        assert entry.spec.kind == "code"
        assert entry.spec.module
        assert entry.provenance.get("module") == module_name
        assert isinstance(entry.provenance.get("module_file"), str)
        assert _is_sha256(entry.provenance.get("sha256"))

        audit_entries = _audit_append_entries(audit_path, audit_before)
        audit_paths = {entry.get("path") for entry in audit_entries}
        assert "src/skills_generated/manifest.json" in audit_paths
        assert f"src/skills_generated/{module_basename}.py" in audit_paths
        for audit_entry in audit_entries:
            if audit_entry.get("path") not in {
                "src/skills_generated/manifest.json",
                f"src/skills_generated/{module_basename}.py",
            }:
                continue
            assert audit_entry.get("action") == "write"
            assert _is_sha256(audit_entry.get("sha256_after"))
            assert isinstance(audit_entry.get("ts"), str) and audit_entry.get("ts")
            assert isinstance(audit_entry.get("reason"), str) and audit_entry.get("reason")

        expected_sha = hashlib.sha256(module_path.read_bytes()).hexdigest()
        matching = [e for e in audit_entries if e.get("path") == f"src/skills_generated/{module_basename}.py"]
        assert matching
        assert matching[-1].get("sha256_after") == expected_sha
    finally:
        sys.modules.pop(module_name, None)
        if module_path.exists():
            module_path.unlink()
        if pycache_dir.exists():
            shutil.rmtree(pycache_dir)
        if shipped_test_path.exists():
            shipped_test_path.unlink()
        if shipped_test_pycache.exists():
            shutil.rmtree(shipped_test_pycache)
        if not shipped_test_dir_existed and shipped_test_dir.exists() and not any(shipped_test_dir.iterdir()):
            shipped_test_dir.rmdir()
        if original_manifest is not None:
            manifest_path.write_text(original_manifest)
        elif manifest_path.exists():
            manifest_path.unlink()
        if audit_existed:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(audit_before, encoding="utf-8")
        elif audit_path.exists():
            audit_path.unlink()


def test_build_and_registry_prompt_skill(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    memory_store = MemoryStore(tmp_path / "memory")
    builder = SkillBuilder(repo_root, memory_store)

    name = f"test-prompt-skill-{uuid.uuid4().hex[:8]}"
    spec = _valid_skill_spec(name, kind="prompt")

    manifest_path = repo_root / "skills_generated" / "manifest.json"
    prompt_path = repo_root / "skills_generated" / "prompts" / f"{name}.md"
    validation_path = repo_root / "skills_generated" / "prompts" / f"{name}.validation.md"
    original_manifest = manifest_path.read_text() if manifest_path.exists() else None
    audit_path = workspace_root / ".tokimon-tmp" / "audit" / "config.jsonl"
    audit_existed = audit_path.exists()
    audit_before = audit_path.read_text(encoding="utf-8", errors="replace") if audit_existed else ""

    try:
        ok = builder.build_skill(spec, "test justification")
        assert ok is True
        assert prompt_path.exists()
        assert validation_path.exists()

        registry = SkillRegistry(repo_root)
        registry.load()
        entry = registry.get(name)
        assert entry is not None
        assert entry.spec.kind == "prompt"
        assert hasattr(entry.module, "prompt_template")
        prompt_template = getattr(entry.module, "prompt_template")
        assert isinstance(prompt_template, str)
        assert f"# {name}" in prompt_template
        assert spec.purpose in prompt_template

        audit_entries = _audit_append_entries(audit_path, audit_before)
        audit_paths = {entry.get("path") for entry in audit_entries}
        assert "src/skills_generated/manifest.json" in audit_paths
        assert f"src/skills_generated/prompts/{name}.md" in audit_paths
        assert f"src/skills_generated/prompts/{name}.validation.md" in audit_paths
        for audit_entry in audit_entries:
            if audit_entry.get("path") not in {
                "src/skills_generated/manifest.json",
                f"src/skills_generated/prompts/{name}.md",
                f"src/skills_generated/prompts/{name}.validation.md",
            }:
                continue
            assert audit_entry.get("action") == "write"
            assert _is_sha256(audit_entry.get("sha256_after"))
    finally:
        if prompt_path.exists():
            prompt_path.unlink()
        if validation_path.exists():
            validation_path.unlink()
        if original_manifest is not None:
            manifest_path.write_text(original_manifest)
        elif manifest_path.exists():
            manifest_path.unlink()
        if audit_existed:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(audit_before, encoding="utf-8")
        elif audit_path.exists():
            audit_path.unlink()


def test_validation_failure_keeps_candidate_and_does_not_update_manifest(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    memory_store = MemoryStore(tmp_path / "memory")
    builder = SkillBuilder(repo_root, memory_store)

    name = f"test-invalid-skill-{uuid.uuid4().hex[:8]}"
    spec = _valid_skill_spec(name, kind="code")
    spec.preconditions = []

    manifest_path = repo_root / "skills_generated" / "manifest.json"
    candidates_dir = repo_root / "skills_generated" / "candidates"
    original_manifest = manifest_path.read_text() if manifest_path.exists() else None
    candidates_before = set(p.name for p in candidates_dir.iterdir()) if candidates_dir.exists() else set()

    try:
        ok = builder.build_skill(spec, "test justification")
        assert ok is False

        if original_manifest is not None:
            assert manifest_path.read_text() == original_manifest
        else:
            assert not manifest_path.exists()

        candidates_after = set(p.name for p in candidates_dir.iterdir()) if candidates_dir.exists() else set()
        new_candidates = candidates_after - candidates_before
        assert len(new_candidates) == 1
        candidate_dir = candidates_dir / next(iter(new_candidates))
        assert (candidate_dir / "spec.json").exists()
        assert (candidate_dir / "failure_reason.txt").exists()

        lessons = list((tmp_path / "memory" / "lessons").glob("lesson-*.md"))
        assert lessons
        metadata = _read_lesson_metadata(lessons[0])
        assert metadata.get("tags") == ["skill", "candidate", name]
        assert "candidate_dir" in metadata
    finally:
        candidates_after = set(p.name for p in candidates_dir.iterdir()) if candidates_dir.exists() else set()
        for created in candidates_after - candidates_before:
            shutil.rmtree(candidates_dir / created)
        if original_manifest is not None:
            manifest_path.write_text(original_manifest)
        elif manifest_path.exists():
            manifest_path.unlink()


def test_safety_guardrail_blocks_unsafe_skill(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    memory_store = MemoryStore(tmp_path / "memory")
    builder = SkillBuilder(repo_root, memory_store)

    name = f"test-unsafe-skill-{uuid.uuid4().hex[:8]}"
    spec = _valid_skill_spec(name, kind="code")
    spec.purpose = "Hack credentials from a config file."

    manifest_path = repo_root / "skills_generated" / "manifest.json"
    candidates_dir = repo_root / "skills_generated" / "candidates"
    original_manifest = manifest_path.read_text() if manifest_path.exists() else None
    candidates_before = set(p.name for p in candidates_dir.iterdir()) if candidates_dir.exists() else set()

    try:
        ok = builder.build_skill(spec, "test justification")
        assert ok is False

        if original_manifest is not None:
            assert manifest_path.read_text() == original_manifest
        else:
            assert not manifest_path.exists()

        candidates_after = set(p.name for p in candidates_dir.iterdir()) if candidates_dir.exists() else set()
        new_candidates = candidates_after - candidates_before
        assert len(new_candidates) == 1

        lessons = list((tmp_path / "memory" / "lessons").glob("lesson-*.md"))
        assert lessons
        body = lessons[0].read_text()
        assert "Safety guardrail rejected skill" in body
    finally:
        candidates_after = set(p.name for p in candidates_dir.iterdir()) if candidates_dir.exists() else set()
        for created in candidates_after - candidates_before:
            shutil.rmtree(candidates_dir / created)
        if original_manifest is not None:
            manifest_path.write_text(original_manifest)
        elif manifest_path.exists():
            manifest_path.unlink()
