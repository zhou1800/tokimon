"""Skill builder pipeline."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit.config_audit import write_text_with_audit
from memory.store import MemoryStore
from skills.spec import SkillSpec
from tools.pytest_tool import PytestTool


_KEBAB_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


class SkillBuilder:
    def __init__(self, repo_root: Path, memory_store: MemoryStore) -> None:
        self.repo_root = repo_root
        self.workspace_root = repo_root.parent if (repo_root.parent / "docs").is_dir() else repo_root
        self.generated_dir = repo_root / "skills_generated"
        self.prompts_dir = self.generated_dir / "prompts"
        self.candidates_dir = self.generated_dir / "candidates"
        self.staging_root = repo_root / ".tokimon-tmp" / "skill-builder"
        self.manifest_path = self.generated_dir / "manifest.json"
        self.memory_store = memory_store
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        init_file = self.generated_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Generated skills live here."""\n')

    def build_skill(self, spec: SkillSpec, justification: str) -> bool:
        spec.kind = "prompt" if str(getattr(spec, "kind", "code")).strip().lower() == "prompt" else "code"

        staging_dir = self._new_staging_dir(spec.name)
        staging_dir.mkdir(parents=True, exist_ok=True)
        (staging_dir / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
        try:
            metadata_issues = _validate_skill_metadata(spec)
            if metadata_issues:
                return self._finalize_candidate_failure(
                    spec,
                    justification,
                    staging_dir,
                    reason="Metadata validation failed: " + "; ".join(metadata_issues),
                )

            unsafe_reason = _unsafe_skill_reason(spec)
            if unsafe_reason:
                return self._finalize_candidate_failure(
                    spec,
                    justification,
                    staging_dir,
                    reason=f"Safety guardrail rejected skill: {unsafe_reason}",
                )

            if spec.kind == "prompt":
                ok, reason = self._promote_prompt_skill(spec, staging_dir)
                if not ok:
                    return self._finalize_candidate_failure(spec, justification, staging_dir, reason=reason)
                self._write_lesson(spec, justification, status="promoted", reason="Prompt skill promoted after validation.")
                shutil.rmtree(staging_dir)
                return True

            ok, reason = self._promote_code_skill(spec, staging_dir)
            if not ok:
                return self._finalize_candidate_failure(spec, justification, staging_dir, reason=reason)
            self._write_lesson(spec, justification, status="promoted", reason="Code skill promoted after pytest validation.")
            shutil.rmtree(staging_dir)
            return True
        except Exception as exc:
            return self._finalize_candidate_failure(spec, justification, staging_dir, reason=f"Build error: {exc}")

    def _new_staging_dir(self, skill_name: str) -> Path:
        safe_name = _safe_fs_slug(skill_name)
        return self.staging_root / f"{uuid.uuid4().hex}-{safe_name}"

    def _finalize_candidate_failure(self, spec: SkillSpec, justification: str, staging_dir: Path, *, reason: str) -> bool:
        try:
            (staging_dir / "failure_reason.txt").write_text(reason.strip() + "\n")
        except Exception:
            pass
        candidate_dir = self._move_to_candidate_dir(staging_dir, spec.name)
        self._write_lesson(spec, justification, status="candidate", reason=reason, candidate_dir=candidate_dir)
        return False

    def _move_to_candidate_dir(self, staging_dir: Path, skill_name: str) -> Path | None:
        if not staging_dir.exists():
            return None
        timestamp = _timestamp_slug()
        safe_name = _safe_fs_slug(skill_name)
        candidate_dir = self.candidates_dir / f"{timestamp}-{safe_name}"
        try:
            shutil.move(str(staging_dir), str(candidate_dir))
            return candidate_dir
        except Exception:
            try:
                shutil.copytree(staging_dir, candidate_dir)
                shutil.rmtree(staging_dir)
                return candidate_dir
            except Exception:
                return None

    def _promote_prompt_skill(self, spec: SkillSpec, staging_dir: Path) -> tuple[bool, str]:
        candidate_prompt = staging_dir / "prompt.md"
        candidate_validation = staging_dir / "validation.md"
        prompt_source = self._prompt_source(spec)
        candidate_prompt.write_text(prompt_source)
        candidate_validation.write_text(_prompt_validation_checklist(spec))

        validation_issues = _validate_prompt_candidate(candidate_prompt, candidate_validation, spec)
        if validation_issues:
            return False, "Prompt validation failed: " + "; ".join(validation_issues)

        prompt_relpath = Path("prompts") / f"{spec.name}.md"
        validation_relpath = Path("prompts") / f"{spec.name}.validation.md"
        spec.prompt_path = str(prompt_relpath)
        spec.prompt_template = None
        if isinstance(spec.validation_method, dict):
            spec.validation_method = {**spec.validation_method, "artifact": str(validation_relpath)}
        promoted_prompt = self.generated_dir / prompt_relpath
        promoted_prompt.parent.mkdir(parents=True, exist_ok=True)
        write_text_with_audit(
            self.workspace_root,
            promoted_prompt,
            candidate_prompt.read_text(),
            reason=f"promote prompt skill {spec.name}",
        )
        promoted_validation = self.generated_dir / validation_relpath
        write_text_with_audit(
            self.workspace_root,
            promoted_validation,
            candidate_validation.read_text(),
            reason=f"promote prompt skill validation {spec.name}",
        )

        manifest = _load_manifest(self.manifest_path)
        manifest["skills"] = [entry for entry in manifest.get("skills", []) if _manifest_entry_name(entry) != spec.name]
        manifest["skills"].append({"kind": "prompt", "prompt_path": spec.prompt_path, "spec": spec.to_dict()})
        write_text_with_audit(
            self.workspace_root,
            self.manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True),
            reason=f"update skills manifest for prompt skill {spec.name}",
        )
        return True, ""

    def _prompt_source(self, spec: SkillSpec) -> str:
        prompt_template = str(spec.prompt_template or "")
        if prompt_template.strip():
            return prompt_template
        prompt_path = str(spec.prompt_path or "").strip()
        if prompt_path:
            resolved = _resolve_within_root(self.repo_root, prompt_path)
            if resolved is not None and resolved.exists():
                return resolved.read_text()
        return _prompt_template(spec)

    def _promote_code_skill(self, spec: SkillSpec, staging_dir: Path) -> tuple[bool, str]:
        module_basename = _module_basename(spec.name)
        module_name = f"skills_generated.{module_basename}"
        spec.module = module_name
        (staging_dir / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))

        candidate_module = staging_dir / f"{module_basename}.py"
        candidate_module.write_text(_module_template(spec))
        validation_test = staging_dir / f"test_validation_{module_basename}.py"
        validation_test.write_text(_code_candidate_validation_test_template(spec, candidate_module))

        pytest_tool = PytestTool(self.repo_root)
        result = pytest_tool.run([str(validation_test)])
        if not result.ok:
            summary = result.data.get("failing_tests") if isinstance(result.data, dict) else None
            extra = f" failing_tests={summary}" if summary else ""
            return False, f"Code skill pytest validation failed:{extra}"

        promoted_module = self.generated_dir / f"{module_basename}.py"
        write_text_with_audit(
            self.workspace_root,
            promoted_module,
            candidate_module.read_text(),
            reason=f"promote code skill {spec.name}",
        )

        shipped_test_relpath = Path("tests") / "skills_generated" / f"test_{module_basename}.py"
        shipped_test_path = self.repo_root / shipped_test_relpath
        shipped_test_path.parent.mkdir(parents=True, exist_ok=True)
        shipped_test_path.write_text(_code_shipped_validation_test_template(spec))
        if isinstance(spec.validation_method, dict):
            spec.validation_method = {**spec.validation_method, "test_path": str(shipped_test_relpath)}

        manifest = _load_manifest(self.manifest_path)
        manifest["skills"] = [entry for entry in manifest.get("skills", []) if _manifest_entry_name(entry) != spec.name]
        manifest["skills"].append({"kind": "code", "module": module_name, "spec": spec.to_dict()})
        write_text_with_audit(
            self.workspace_root,
            self.manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True),
            reason=f"update skills manifest for code skill {spec.name}",
        )
        return True, ""

    def _write_lesson(
        self,
        spec: SkillSpec,
        justification: str,
        *,
        status: str,
        reason: str,
        candidate_dir: Path | None = None,
    ) -> None:
        lesson_id = str(uuid.uuid4())
        metadata = {
            "id": lesson_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tags": ["skill", status, spec.name],
            "component": "skills",
        }
        if candidate_dir is not None:
            metadata["candidate_dir"] = str(candidate_dir)
        body = (
            f"Created skill {spec.name}.\n"
            f"Kind: {spec.kind}\n"
            f"Purpose: {spec.purpose}\n"
            f"Contract: {spec.contract}\n"
            f"Justification: {justification}\n"
            f"Outcome: {status}\n"
            f"Reason: {reason}\n"
        )
        self.memory_store.write_lesson(metadata, body)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"skills": []}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"skills": []}
    return data if isinstance(data, dict) else {"skills": []}


def _manifest_entry_name(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    spec = entry.get("spec")
    if isinstance(spec, dict):
        name = spec.get("name")
        return str(name) if name else None
    name = entry.get("name")
    return str(name) if name else None


def _module_basename(skill_name: str) -> str:
    return skill_name.replace("-", "_")


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _validate_skill_metadata(spec: SkillSpec) -> list[str]:
    issues: list[str] = []
    if spec.kind not in {"code", "prompt"}:
        issues.append("kind must be 'code' or 'prompt'")
    if not _KEBAB_CASE_PATTERN.match(spec.name):
        issues.append("name must be kebab-case (e.g. my-skill)")
    if _is_empty_value(spec.purpose):
        issues.append("purpose is required")

    if not isinstance(spec.contract, dict):
        issues.append("contract must be an object with inputs/outputs")
    else:
        if _is_empty_value(spec.contract.get("inputs")):
            issues.append("contract.inputs is required")
        if _is_empty_value(spec.contract.get("outputs")):
            issues.append("contract.outputs is required")

    if _is_empty_value(spec.preconditions):
        issues.append("preconditions is required")
    if spec.required_tools is None or not isinstance(spec.required_tools, list):
        issues.append("required_tools must be a list")
    if _is_empty_value(spec.retrieval_prefs):
        issues.append("retrieval_prefs is required")
    if _is_empty_value(spec.failure_modes):
        issues.append("failure_modes is required")

    if not isinstance(spec.safety_notes, dict):
        issues.append("safety_notes must be an object with hard/soft")
    else:
        hard = spec.safety_notes.get("hard")
        soft = spec.safety_notes.get("soft")
        if not isinstance(hard, list) or not hard:
            issues.append("safety_notes.hard is required")
        if not isinstance(soft, list):
            issues.append("safety_notes.soft must be a list")

    if _is_empty_value(spec.cost_energy_notes):
        issues.append("cost_energy_notes is required")
    if _is_empty_value(spec.validation_method):
        issues.append("validation_method is required")
    if _is_empty_value(spec.version) or not _SEMVER_PATTERN.match(str(spec.version).strip()):
        issues.append("version must be semantic version (e.g. 1.0.0)")
    if _is_empty_value(spec.deprecation_policy):
        issues.append("deprecation_policy is required")

    return issues


def _unsafe_skill_reason(spec: SkillSpec) -> str | None:
    text = " ".join([str(spec.purpose or ""), json.dumps(spec.contract, sort_keys=True, default=str)])
    lowered = text.lower()
    disallowed = [
        ("credential theft", "credential theft"),
        ("steal credentials", "credential theft"),
        ("phish", "phishing"),
        ("malware", "malware"),
        ("ransomware", "malware"),
        ("keylogger", "malware"),
        ("exploit", "cyber exploitation"),
        ("hack", "cyber exploitation"),
    ]
    for token, reason in disallowed:
        if token in lowered:
            return reason
    return None


def _module_template(spec: SkillSpec) -> str:
    spec_json = json.dumps(spec.to_dict(), sort_keys=True)
    return (
        "from __future__ import annotations\n\n"
        "import json\n\n"
        "from skills.spec import SkillSpec\n\n"
        f"SKILL_SPEC = SkillSpec.from_dict(json.loads({spec_json!r}))\n\n"
        "def execute(context: dict[str, Any] | None = None) -> dict[str, Any]:\n"
        "    context = context or {}\n"
        "    return {\"status\": \"SUCCESS\", \"summary\": f\"{SKILL_SPEC.name} executed\"}\n"
    )


def _code_candidate_validation_test_template(spec: SkillSpec, module_path: Path) -> str:
    return (
        "from __future__ import annotations\n\n"
        "import importlib.util\n"
        "from pathlib import Path\n\n"
        f"_MODULE_PATH = Path({str(module_path)!r})\n\n"
        "def _load_module():\n"
        "    spec = importlib.util.spec_from_file_location('candidate_skill', _MODULE_PATH)\n"
        "    assert spec and spec.loader\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        "    return module\n\n"
        "def test_generated_skill_spec():\n"
        "    module = _load_module()\n"
        f"    assert module.SKILL_SPEC.name == {spec.name!r}\n"
        f"    assert module.SKILL_SPEC.kind == {spec.kind!r}\n\n"
        "def test_generated_skill_execute_contract():\n"
        "    module = _load_module()\n"
        "    result = module.execute({})\n"
        "    assert isinstance(result, dict)\n"
        "    assert \"status\" in result\n"
        "    assert \"summary\" in result\n"
    )


def _code_shipped_validation_test_template(spec: SkillSpec) -> str:
    module_basename = _module_basename(spec.name)
    module_import = f"skills_generated.{module_basename}"
    return (
        "from __future__ import annotations\n\n"
        "import importlib\n\n"
        f"_MODULE_IMPORT = {module_import!r}\n\n"
        "def _load_module():\n"
        "    return importlib.import_module(_MODULE_IMPORT)\n\n"
        "def test_generated_skill_spec():\n"
        "    module = _load_module()\n"
        "    assert hasattr(module, \"SKILL_SPEC\")\n"
        f"    assert module.SKILL_SPEC.name == {spec.name!r}\n"
        f"    assert module.SKILL_SPEC.kind == {spec.kind!r}\n\n"
        "def test_generated_skill_execute_contract():\n"
        "    module = _load_module()\n"
        "    assert hasattr(module, \"execute\")\n"
        "    result = module.execute({})\n"
        "    assert isinstance(result, dict)\n"
        "    assert \"status\" in result\n"
        "    assert \"summary\" in result\n"
    )


def _prompt_template(spec: SkillSpec) -> str:
    contract = spec.contract if isinstance(spec.contract, dict) else {}
    inputs = contract.get("inputs", {})
    outputs = contract.get("outputs", {})
    side_effects = contract.get("side_effects", "")
    return (
        f"# {spec.name}\n\n"
        "## Purpose\n"
        f"{spec.purpose}\n\n"
        "## Contract\n"
        f"- Inputs: {inputs}\n"
        f"- Outputs: {outputs}\n"
        f"- Side effects: {side_effects}\n\n"
        "## Prompt Template\n"
        "You are a Tokimon prompt skill. Follow the contract exactly.\n"
    )


def _prompt_validation_checklist(spec: SkillSpec) -> str:
    return (
        f"# Validation Checklist: {spec.name}\n\n"
        "- [ ] Metadata complete (required fields present)\n"
        "- [ ] Prompt contains Purpose and Contract\n"
        "- [ ] Prompt includes at least one safety reminder\n"
    )


def _validate_prompt_candidate(prompt_path: Path, validation_path: Path, spec: SkillSpec) -> list[str]:
    issues: list[str] = []
    if not prompt_path.exists():
        issues.append("prompt.md missing")
    else:
        prompt = prompt_path.read_text()
        if len(prompt.strip()) < 40:
            issues.append("prompt.md too short")
        if spec.purpose.strip() and spec.purpose.strip() not in prompt:
            issues.append("prompt.md must include purpose")
        if "## Contract" not in prompt:
            issues.append("prompt.md must include Contract section")
    if not validation_path.exists():
        issues.append("validation.md missing")
    else:
        content = validation_path.read_text()
        if "Validation Checklist" not in content:
            issues.append("validation.md must include checklist header")
    return issues


_FS_SLUG_PATTERN = re.compile(r"[^a-z0-9._-]+")


def _safe_fs_slug(value: str) -> str:
    normalized = str(value or "").strip().lower()
    normalized = normalized.replace("/", "-").replace("\\", "-").replace(os.sep, "-")
    normalized = _FS_SLUG_PATTERN.sub("-", normalized)
    normalized = normalized.strip("-") or "skill"
    return normalized[:120]


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _resolve_within_root(root: Path, path_value: str) -> Path | None:
    raw = Path(str(path_value))
    base = root.resolve()
    candidate = raw if raw.is_absolute() else (root / raw)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return resolved
