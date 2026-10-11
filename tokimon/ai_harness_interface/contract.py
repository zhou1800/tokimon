"""Static AI Harness Interface contract declarations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokimon import runtime
from tokimon.tools import catalog as tool_catalog


TARGET_PLATFORM = "codex"
PACKAGE_ID = "tokimon"
PACKAGE_VERSION = "0.2.1"
MARKETPLACE_NAME = "tokimon-local"
MARKETPLACE_DISPLAY_NAME = "Tokimon Local"
MARKETPLACE_OWNER_NAME = "Tokimon"
MARKETPLACE_SOURCE_PATH = "./plugins/tokimon"
MARKETPLACE_CATEGORY = "Productivity"
CLAUDE_MARKETPLACE_CATEGORY = "productivity"
LICENSE_FILE_PATHS: tuple[str, ...] = (
    "LICENSE",
    "THIRD_PARTY_NOTICES",
    "licenses/Matt-Pocock-MIT.txt",
    "licenses/Superpowers-MIT.txt",
    "licenses/Playwright-Apache-2.0.txt",
    "licenses/Playwright-NOTICE.txt",
)


@dataclass(frozen=True)
class TargetPlatform:
    platform_id: str
    manifest_dir_name: str
    marketplace_relative_path: Path
    generated_output_relative_path: Path
    account_level_install_owner: str


CODEX_PLATFORM = TargetPlatform(
    platform_id="codex",
    manifest_dir_name=".codex-plugin",
    marketplace_relative_path=Path(".agents") / "plugins" / "marketplace.json",
    generated_output_relative_path=Path(".tokimon") / "ai-harness-interface" / "codex",
    account_level_install_owner="Codex plugin-creator guidance and Codex plugin CLI",
)

CLAUDE_PLATFORM = TargetPlatform(
    platform_id="claude",
    manifest_dir_name=".claude-plugin",
    marketplace_relative_path=Path(".claude-plugin") / "marketplace.json",
    generated_output_relative_path=Path(".tokimon") / "ai-harness-interface" / "claude",
    account_level_install_owner="Claude Code plugin CLI guided by the install operation document",
)

TARGET_PLATFORMS: dict[str, TargetPlatform] = {
    CODEX_PLATFORM.platform_id: CODEX_PLATFORM,
    CLAUDE_PLATFORM.platform_id: CLAUDE_PLATFORM,
}

def target_platform_for(platform_id: str) -> TargetPlatform:
    try:
        return TARGET_PLATFORMS[platform_id]
    except KeyError as exc:
        supported = ", ".join(sorted(TARGET_PLATFORMS))
        raise ValueError(f"unsupported target platform {platform_id!r}; supported: {supported}") from exc

STARTUP_BOOTSTRAP_SKILL_ID = "using-tokimon"
MEMORY_ACTIVATION_SKILL_ID = "memory"

CLI_COMMANDS: tuple[str, ...] = (
    "bootstrap-codex",
    "configure",
    "contract",
    "generate",
    "validate",
    "diagnose",
    "post-install",
)


McpEntry = tool_catalog.McpCatalogEntry
MCP_ENTRIES: tuple[McpEntry, ...] = tool_catalog.package_entries()


def _mcp_entry_payload(entry: McpEntry) -> dict[str, Any]:
    return {
        "capability_id": entry.capability_id,
        "server_id": entry.server_id,
        "module": entry.module,
        "serve_args": list(entry.serve_args),
        "health_args": list(entry.health_args),
    }


def package_root(workspace_root: Path) -> Path:
    return workspace_root / "plugins" / PACKAGE_ID


def marketplace_path(workspace_root: Path, platform: TargetPlatform = CODEX_PLATFORM) -> Path:
    return workspace_root / platform.marketplace_relative_path


def manifest_path(package_root_path: Path, platform: TargetPlatform = CODEX_PLATFORM) -> Path:
    return package_root_path / platform.manifest_dir_name / "plugin.json"


def repo_skill_ids(*, source_root: Path) -> tuple[str, ...]:
    skills_root = source_root / "skills"
    try:
        entries = tuple(sorted(path.name for path in skills_root.iterdir() if (path / "skill.md").is_file()))
    except OSError as exc:
        raise ValueError(f"unable to read root skill catalog {skills_root}: {exc}") from exc
    if not entries:
        raise ValueError(f"root skill catalog has no skill entrypoints: {skills_root}")
    return entries


def package_skill_ids(*, source_root: Path) -> tuple[str, ...]:
    repo_ids = repo_skill_ids(source_root=source_root)
    priority = (STARTUP_BOOTSTRAP_SKILL_ID, MEMORY_ACTIVATION_SKILL_ID)
    ordered = tuple(skill_id for skill_id in priority if skill_id in repo_ids)
    return (*ordered, *(skill_id for skill_id in repo_ids if skill_id not in ordered))


def contract_payload(
    *,
    workspace_root: Path | None = None,
    target_platform: str = TARGET_PLATFORM,
) -> dict[str, Any]:
    root = (workspace_root or Path.cwd()).resolve(strict=False)
    platform = target_platform_for(target_platform)
    repo_skills = repo_skill_ids(source_root=root)
    package_skills = package_skill_ids(source_root=root)
    payload: dict[str, Any] = {
        "capability_id": "ai-harness-interface",
        "target_platform": platform.platform_id,
        "supported_target_platforms": sorted(TARGET_PLATFORMS),
        "package_id": PACKAGE_ID,
        "package_version": PACKAGE_VERSION,
        "marketplace_name": MARKETPLACE_NAME,
        "marketplace_source_path": MARKETPLACE_SOURCE_PATH,
        "generated_output_relative_path": platform.generated_output_relative_path.as_posix(),
        "repo_skill_ids": list(repo_skills),
        "startup_bootstrap_skill_id": STARTUP_BOOTSTRAP_SKILL_ID,
        "memory_activation_skill_id": MEMORY_ACTIVATION_SKILL_ID,
        "package_skill_ids": list(package_skills),
        "mcp_entries": [_mcp_entry_payload(entry) for entry in MCP_ENTRIES],
        "commands": list(CLI_COMMANDS),
    }
    output_root = runtime.resolve_runtime_path(platform.generated_output_relative_path, workspace_root=root)
    payload["workspace_root"] = str(root)
    payload["output_root"] = str(output_root)
    payload["package_root"] = str(package_root(output_root))
    payload["marketplace_path"] = str(marketplace_path(output_root, platform))
    return payload
