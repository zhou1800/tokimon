"""Packaged root skill directory generation and validation."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokimon.ai_harness_interface import contract, route, templates


class SkillPackagingError(ValueError):
    """Raised when a source or packaged root skill is invalid."""


RESOURCE_DIR_NAMES = ("assets", "reference", "references", "scripts", "agents")
RESOURCE_PATH_RE = re.compile(r"(?<![\w.-])(?:assets|reference|references|scripts|agents)/[^\s`'\"),]+")


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    body: str


def load_source_skill(*, source_root: Path, skill_id: str) -> str:
    path = source_root / "skills" / skill_id / "skill.md"
    if path.is_symlink() or not path.is_file():
        raise SkillPackagingError(f"source skill {skill_id} must be a regular file")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SkillPackagingError(f"unable to read source skill {skill_id}: {exc}") from exc


def source_openai_metadata_path(*, source_root: Path, skill_id: str) -> Path:
    return source_root / "skills" / skill_id / "agents" / "openai.yaml"


def load_source_openai_metadata(*, source_root: Path, skill_id: str) -> str | None:
    path = source_openai_metadata_path(source_root=source_root, skill_id=skill_id)
    if path.is_symlink():
        raise SkillPackagingError(f"{skill_id} agents/openai.yaml must be a regular file")
    if not path.exists():
        return None
    if not path.is_file():
        raise SkillPackagingError(f"{skill_id} agents/openai.yaml must be a regular file")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SkillPackagingError(f"unable to read Codex metadata for {skill_id}: {exc}") from exc
    if not text.strip():
        raise SkillPackagingError(f"{skill_id} agents/openai.yaml must not be empty")
    return text if text.endswith("\n") else text + "\n"


def parse_skill(text: str) -> SkillMetadata:
    if not text.startswith("---\n"):
        raise SkillPackagingError("skill must start with YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise SkillPackagingError("skill frontmatter is not closed")
    frontmatter = text[4:end]
    body = text[end + len("\n---\n") :]
    data: dict[str, str] = {}
    for line in frontmatter.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise SkillPackagingError(f"unsupported frontmatter line: {line}")
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    name = data.get("name", "")
    description = data.get("description", "")
    if not name or not description:
        raise SkillPackagingError("skill frontmatter must include name and description")
    return SkillMetadata(name=name, description=description, body=body)


def build_installed_routing_note(*, install_route: route.InstallRoute) -> str:
    return templates.render_text_template(
        "installed-routing-note.md",
        {
            "liveCheckoutPath": str(install_route.liveCheckoutPath),
            "runtimeRoot": str(install_route.runtimeRoot),
        },
    )


def build_packaged_skill(*, skill_id: str, source_text: str, install_route: route.InstallRoute) -> str:
    metadata = parse_skill(source_text)
    if metadata.name != skill_id:
        raise SkillPackagingError(
            f"source skill folder {skill_id} does not match frontmatter name {metadata.name}"
        )
    body = metadata.body.replace(
        "{{tokimonPython}}",
        str(install_route.python_executable),
    )
    command = (
        f"PYTHONPATH={shlex.quote(str(install_route.liveCheckoutPath))}"
        '"${PYTHONPATH:+:$PYTHONPATH}" '
        f"{shlex.quote(str(install_route.python_executable))} -P"
    )
    body = body.replace("{{tokimonCommand}}", command)
    return "\n".join(
        (
            "---",
            f"name: {metadata.name}",
            f"description: {metadata.description}",
            "---",
            "",
            build_installed_routing_note(install_route=install_route).rstrip(),
            "",
            body.strip(),
            "",
        )
    )


def scan_unhandled_resource_dependencies(*, source_root: Path, skill_id: str) -> list[str]:
    skill_dir = source_root / "skills" / skill_id
    source_text = load_source_skill(source_root=source_root, skill_id=skill_id)
    problems: list[str] = []
    for dirname in RESOURCE_DIR_NAMES:
        resource_dir = skill_dir / dirname
        if not resource_dir.exists():
            continue
        if dirname == "agents":
            if resource_dir.is_symlink() or not resource_dir.is_dir():
                problems.append(f"{skill_id} has unsupported package-local resource `agents/`")
                continue
            for path in sorted(resource_dir.rglob("*")):
                relative_path = path.relative_to(skill_dir).as_posix()
                if relative_path == "agents/openai.yaml" and path.is_file() and not path.is_symlink():
                    continue
                problems.append(f"{skill_id} has unsupported package-local resource `{relative_path}`")
            continue
        problems.append(f"{skill_id} has package-local resource directory `{dirname}/`")
    for match in RESOURCE_PATH_RE.finditer(source_text):
        relative = match.group(0)
        if (skill_dir / relative).exists():
            problems.append(f"{skill_id} references package-local resource `{relative}`")
    return problems


def validate_source_skill_inventory(*, source_root: Path) -> tuple[str, ...]:
    discovered = contract.repo_skill_ids(source_root=source_root)
    problems: list[str] = []
    for skill_id in discovered:
        problems.extend(scan_unhandled_resource_dependencies(source_root=source_root, skill_id=skill_id))
        try:
            load_source_openai_metadata(source_root=source_root, skill_id=skill_id)
        except SkillPackagingError as exc:
            problems.append(str(exc))
    if problems:
        raise SkillPackagingError("; ".join(problems))
    return discovered


def validate_packaged_skill(
    *,
    skill_id: str,
    text: str,
    install_route: route.InstallRoute | None = None,
) -> dict[str, Any]:
    metadata = parse_skill(text)
    issues: list[str] = []
    if metadata.name != skill_id:
        issues.append(f"frontmatter name {metadata.name!r} does not match {skill_id!r}")
    if "## Tokimon Routing" not in text:
        issues.append("missing installed Tokimon routing note")
    if (
        skill_id != contract.STARTUP_BOOTSTRAP_SKILL_ID
        and "starting any conversation" in metadata.description.lower()
    ):
        issues.append("only the startup bootstrap skill may use conversation-start trigger wording")
    if "Treat the current repository as the task subject" not in text:
        issues.append("missing adopting-repository task subject rule")
    if "Resolve Tokimon-owned command entrypoints" not in text:
        issues.append("missing live-checkout command routing rule")
    if "{{tokimonPython}}" in text:
        issues.append("unresolved tokimonPython command placeholder")
    if "{{tokimonCommand}}" in text:
        issues.append("unresolved tokimonCommand command placeholder")
    if "When editing source files, use the current task checkout" not in text:
        issues.append("missing source-edit checkout rule")
    if skill_id in {
        contract.STARTUP_BOOTSTRAP_SKILL_ID,
        contract.MEMORY_ACTIVATION_SKILL_ID,
    }:
        if 'briefing --project-root "$PWD" --include-scope global --limit 8' not in text:
            issues.append("missing adopting-project memory briefing route")
        if 'search "<keywords>" --project-root "$PWD" --include-scope global' not in text:
            issues.append("missing adopting-project prior-context search route")
        if "briefing --scope tokimon" in text:
            issues.append("installed memory briefing hardcodes the Tokimon project scope")
    if install_route is not None:
        if str(install_route.liveCheckoutPath) not in text:
            issues.append("missing resolved liveCheckoutPath")
        if str(install_route.runtimeRoot) not in text:
            issues.append("missing resolved runtimeRoot")
    return {
        "skill_id": skill_id,
        "ok": not issues,
        "issues": issues,
    }
