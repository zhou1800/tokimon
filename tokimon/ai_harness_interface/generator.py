"""Generate and validate AI Harness Interface package output per target platform."""

from __future__ import annotations

import json
import os
import shutil
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokimon import runtime
from tokimon.ai_harness_interface import (
    claude_cli,
    configuration,
    contract,
    installation_policy,
    marketplace,
    mcp,
    plugin_creator,
    plugin_manifest,
    route,
    skills,
)


class PackageValidationError(ValueError):
    """Raised when generated package output fails validation."""


class PackageScaffoldError(PackageValidationError):
    """Raised when plugin-creator scaffold files are missing."""


@dataclass(frozen=True)
class GenerationPaths:
    source_root: Path
    output_root: Path
    package_root: Path
    marketplace_path: Path
    platform: contract.TargetPlatform = contract.CODEX_PLATFORM

    @property
    def manifest_path(self) -> Path:
        return contract.manifest_path(self.package_root, self.platform)


def paths_for(
    *,
    source_root: Path,
    output_root: Path | None = None,
    target_platform: str = contract.TARGET_PLATFORM,
) -> GenerationPaths:
    platform = contract.target_platform_for(target_platform)
    source = source_root.expanduser().resolve(strict=False)
    if output_root is None:
        output = runtime.resolve_runtime_path(
            platform.generated_output_relative_path,
            workspace_root=source,
        ).resolve(strict=False)
    else:
        output = output_root.expanduser().resolve(strict=False)
    return GenerationPaths(
        source_root=source,
        output_root=output,
        package_root=contract.package_root(output),
        marketplace_path=contract.marketplace_path(output, platform),
        platform=platform,
    )


def install_route_for(paths: GenerationPaths) -> route.InstallRoute:
    return route.build_route(
        live_checkout_path=paths.source_root,
        runtime_root=runtime.runtime_root(paths.source_root),
    )


def _require_valid_source_skill_inventory(source_root: Path) -> tuple[str, ...]:
    try:
        return skills.validate_source_skill_inventory(source_root=source_root)
    except ValueError as exc:
        raise PackageValidationError(f"source skill inventory is invalid: {exc}") from exc


def bootstrap_scaffold(paths: GenerationPaths, *, force: bool = False) -> dict[str, Any]:
    return plugin_creator.bootstrap_codex_scaffold(
        output_root=paths.output_root,
        marketplace_path=paths.marketplace_path,
        force=force,
        required=None,
    )


def generate_package(
    paths: GenerationPaths,
    *,
    version: str = contract.PACKAGE_VERSION,
    cachebuster: str | None = None,
    settings_path: Path | None = None,
) -> dict[str, Any]:
    is_codex = paths.platform.platform_id == "codex"
    if is_codex:
        _require_scaffold(paths)
    elif cachebuster:
        raise PackageValidationError(
            "cachebuster refresh is Codex plugin-creator tooling; "
            f"regenerate the {paths.platform.platform_id} package and refresh through its own plugin update commands"
        )
    setup = configuration.require_configured_settings(
        source_root=paths.source_root,
        settings_path=settings_path,
    )
    _require_valid_source_skill_inventory(paths.source_root)
    license_files = _load_source_license_files(paths.source_root)
    try:
        plugin_manifest.validate_version(version)
        policy = installation_policy.resolve(setup["settings"])
    except ValueError as exc:
        raise PackageValidationError(str(exc)) from exc
    _validate_output_targets(paths)
    package_skill_ids = policy.skill_ids(paths.source_root)
    active_route = install_route_for(paths)
    route.validate_route(active_route, create_runtime_root=True)
    _clean_owned_package_content(paths)

    if is_codex:
        manifest = plugin_manifest.apply_manifest_overlay(_read_json(paths.manifest_path), version=version)
    else:
        manifest = plugin_manifest.build_claude_manifest(version=version)
    _write_json(paths.manifest_path, manifest)
    for relative_path, data in license_files.items():
        _write_bytes(paths.package_root / relative_path, data)

    _write_json(
        paths.package_root / ".mcp.json",
        mcp.build_mcp_manifest(
            install_route=active_route,
            platform=paths.platform,
            entries=policy.mcp_entries,
        ),
    )
    try:
        for skill_id in package_skill_ids:
            source_text = skills.load_source_skill(
                source_root=paths.source_root,
                skill_id=skill_id,
            )
            _write_text(
                paths.package_root / "skills" / skill_id / "SKILL.md",
                skills.build_packaged_skill(
                    skill_id=skill_id,
                    source_text=source_text,
                    install_route=active_route,
                ),
            )
            if is_codex:
                metadata = skills.load_source_openai_metadata(
                    source_root=paths.source_root,
                    skill_id=skill_id,
                )
                if metadata is not None:
                    _write_text(
                        paths.package_root / "skills" / skill_id / "agents" / "openai.yaml",
                        metadata,
                    )
    except skills.SkillPackagingError as exc:
        raise PackageValidationError(f"could not package canonical skills: {exc}") from exc

    if is_codex:
        marketplace_payload = marketplace.apply_tokimon_marketplace_content(_read_json(paths.marketplace_path))
    else:
        marketplace_payload = marketplace.build_claude_marketplace_content()
    _write_json(paths.marketplace_path, marketplace_payload)
    cachebuster_result = None
    if cachebuster:
        cachebuster_result = plugin_creator.update_cachebuster(
            package_root=paths.package_root,
            cachebuster=cachebuster,
            required=None,
        )
        if not cachebuster_result["ok"]:
            raise PackageValidationError("plugin-creator cachebuster update failed")
    effective_version = _read_json(paths.manifest_path).get("version", version)
    policy_path = paths.output_root / installation_policy.POLICY_FILENAME
    _write_json(policy_path, policy.to_json())
    return {
        "ok": True,
        "target_platform": paths.platform.platform_id,
        "package_root": str(paths.package_root),
        "marketplace_path": str(paths.marketplace_path),
        "output_root": str(paths.output_root),
        "liveCheckoutPath": str(active_route.liveCheckoutPath),
        "runtimeRoot": str(active_route.runtimeRoot),
        "version": effective_version,
        "cachebuster": cachebuster_result,
        "skill_ids": list(package_skill_ids),
        "mcp_server_ids": [entry.server_id for entry in policy.mcp_entries],
        "installation_policy": policy.to_json(),
        "installation_policy_path": str(policy_path),
    }


def validate_generated_package(
    paths: GenerationPaths,
    *,
    install_route: route.InstallRoute | None = None,
) -> dict[str, Any]:
    _require_valid_source_skill_inventory(paths.source_root)
    active_route = install_route or install_route_for(paths)
    is_codex = paths.platform.platform_id == "codex"
    policy = load_installation_policy(paths)
    checks: list[dict[str, Any]] = []
    catalog_check = _check_exact_file_catalog(paths, policy=policy)
    checks.append(catalog_check)
    if catalog_check.get("unsafe_links"):
        raise PackageValidationError(
            "generated package validation failed: exact-file-catalog"
        )
    checks.append(_check_manifest(paths) if is_codex else _check_claude_manifest(paths))
    checks.append(_check_marketplace(paths) if is_codex else _check_claude_marketplace(paths))
    checks.append(
        _check_mcp(
            paths,
            install_route=active_route,
            entries=policy.mcp_entries,
        )
    )
    checks.append(_check_skills(paths, install_route=active_route, policy=policy))
    checks.append(_check_license_files(paths))
    if is_codex:
        checks.append(plugin_creator.validate_plugin(package_root=paths.package_root))
    else:
        checks.append(claude_cli.validate_plugin(validation_root=paths.output_root))
    ok = all(check["ok"] for check in checks)
    payload = {
        "ok": ok,
        "package_root": str(paths.package_root),
        "marketplace_path": str(paths.marketplace_path),
        "output_root": str(paths.output_root),
        "checks": checks,
        "installation_policy": policy.to_json(),
    }
    if not ok:
        failures = [check["name"] for check in checks if not check["ok"]]
        raise PackageValidationError(f"generated package validation failed: {', '.join(failures)}")
    return payload


def load_installation_policy(paths: GenerationPaths) -> installation_policy.InstallationPolicy:
    path = paths.output_root / installation_policy.POLICY_FILENAME
    if not path.exists() and not path.is_symlink():
        manifest = _read_json(paths.manifest_path)
        if str(manifest.get("version", "")).split("+", 1)[0] == contract.PACKAGE_VERSION:
            raise PackageValidationError("generated installation policy snapshot is missing")
        return installation_policy.InstallationPolicy("auto", True)
    try:
        return installation_policy.from_json(_read_json(path))
    except ValueError as exc:
        raise PackageValidationError(str(exc)) from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise PackageValidationError(f"generated JSON target must be a regular file: {path}")
    _write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _write_text(path: Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


def _check_output_target(path: Path, *, directory: bool = False) -> None:
    for parent in reversed(path.parents):
        if parent.is_symlink():
            raise PackageValidationError(f"generated output must not follow symbolic links: {parent}")
        if parent.exists() and not parent.is_dir():
            raise PackageValidationError(f"generated output parent must be a directory: {parent}")
    if path.is_symlink():
        raise PackageValidationError(f"generated output must not follow symbolic links: {path}")
    if path.exists() and not (path.is_dir() if directory else path.is_file()):
        kind = "directory" if directory else "file"
        raise PackageValidationError(f"generated output must be a regular {kind}: {path}")


def _validate_output_targets(paths: GenerationPaths) -> None:
    _require_directory_operations()
    directories = (paths.output_root, paths.package_root, paths.package_root / "skills", paths.package_root / "hooks")
    for path in directories:
        _check_output_target(path, directory=True)
    targets = (
        paths.manifest_path,
        paths.package_root / ".mcp.json",
        paths.marketplace_path,
        paths.output_root / installation_policy.POLICY_FILENAME,
        *(paths.package_root / relative for relative in contract.LICENSE_FILE_PATHS),
    )
    for path in targets:
        _check_output_target(path)
    for path in paths.package_root.rglob("*"):
        _check_output_target(path, directory=path.is_dir())


@contextmanager
def _output_parent(path: Path):
    """Pin each parent directory before replacing a generated file."""
    _require_directory_operations()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent = path.absolute().parent
    descriptor = os.open(parent.anchor, flags)
    try:
        for part in parent.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(part, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _require_directory_operations() -> None:
    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY")) or os.open not in os.supports_dir_fd:
        raise PackageValidationError("safe package generation requires POSIX directory-relative file operations")


def _write_bytes(path: Path, data: bytes) -> None:
    temporary = f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with _output_parent(path) as parent:
            created = False
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=parent)
                created = True
                with os.fdopen(descriptor, "wb") as output:
                    output.write(data)
                try:
                    target = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    target = None
                if target is not None and not stat.S_ISREG(target.st_mode):
                    raise PackageValidationError(f"generated output must be a regular file without symbolic links: {path}")
                os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
            finally:
                if created:
                    try:
                        os.unlink(temporary, dir_fd=parent)
                    except FileNotFoundError:
                        pass
    except OSError as exc:
        raise PackageValidationError(f"unable to safely write generated file {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PackageValidationError(f"{path} must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise PackageValidationError(f"unable to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PackageValidationError(f"{path} must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise PackageValidationError(f"{path} must contain a JSON object")
    return payload


def _load_source_license_files(source_root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for relative_path in contract.LICENSE_FILE_PATHS:
        path = source_root / relative_path
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise PackageValidationError(f"source license file {relative_path} must be a regular file")
        try:
            data = path.read_bytes()
            data.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise PackageValidationError(f"unable to read source license file {relative_path}: {exc}") from exc
        if not data.strip():
            raise PackageValidationError(f"source license file {relative_path} must not be empty")
        files[relative_path] = data
    return files


def _check_license_files(paths: GenerationPaths) -> dict[str, Any]:
    issues: list[str] = []
    try:
        source_files = _load_source_license_files(paths.source_root)
    except PackageValidationError as exc:
        return {"name": "license-notices", "ok": False, "issues": [str(exc)]}
    for relative_path, source_data in source_files.items():
        path = paths.package_root / relative_path
        if not path.is_file():
            issues.append(f"missing packaged license file {relative_path}")
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            issues.append(f"unable to read packaged license file {relative_path}: {exc}")
            continue
        if data != source_data:
            issues.append(f"packaged license file {relative_path} differs from its canonical source")
    return {"name": "license-notices", "ok": not issues, "issues": issues}


def _require_scaffold(paths: GenerationPaths) -> None:
    missing = [
        path
        for path in (
            paths.manifest_path,
            paths.package_root / ".mcp.json",
            paths.package_root / "skills",
            paths.marketplace_path,
        )
        if not path.exists()
    ]
    if missing:
        display = ", ".join(str(path) for path in missing)
        raise PackageScaffoldError(f"missing plugin-creator scaffold files: {display}")


def _clean_owned_package_content(paths: GenerationPaths) -> None:
    skills_root = paths.package_root / "skills"
    if skills_root.exists():
        shutil.rmtree(skills_root)
    skills_root.mkdir(parents=True, exist_ok=True)
    hooks_root = paths.package_root / "hooks"
    if hooks_root.exists():
        shutil.rmtree(hooks_root)


def _check_exact_file_catalog(paths: GenerationPaths, *, policy: installation_policy.InstallationPolicy) -> dict[str, Any]:
    issues: list[str] = []
    try:
        metadata_files = (
            {
                f"skills/{skill_id}/agents/openai.yaml"
                for skill_id in policy.skill_ids(paths.source_root)
                if skills.load_source_openai_metadata(
                    source_root=paths.source_root,
                    skill_id=skill_id,
                )
                is not None
            }
            if paths.platform.platform_id == "codex"
            else set()
        )
    except ValueError as exc:
        metadata_files = set()
        issues.append(f"unable to inspect canonical skill metadata: {exc}")
    expected = {
        f"{paths.platform.manifest_dir_name}/plugin.json",
        ".mcp.json",
        *(f"skills/{skill_id}/SKILL.md" for skill_id in policy.skill_ids(paths.source_root)),
        *metadata_files,
        *contract.LICENSE_FILE_PATHS,
    }
    entries = tuple(paths.package_root.rglob("*"))
    unsafe_links = sorted(
        path.relative_to(paths.package_root).as_posix()
        for path in entries
        if path.is_symlink()
    )
    actual = {
        str(path.relative_to(paths.package_root).as_posix())
        for path in entries
        if not path.is_symlink() and path.is_file()
    }
    if unsafe_links:
        issues.append(f"symbolic links are not allowed: {', '.join(unsafe_links)}")
    return {
        "name": "exact-file-catalog",
        "ok": actual == expected and not issues,
        "expected": sorted(expected),
        "actual": sorted(actual),
        "issues": issues,
        "unsafe_links": unsafe_links,
    }


def _check_manifest(paths: GenerationPaths) -> dict[str, Any]:
    manifest = _read_json(paths.manifest_path)
    expected = plugin_manifest.apply_manifest_overlay({}, version=str(manifest.get("version", "")))
    issues: list[str] = []
    try:
        plugin_manifest.validate_version(str(manifest.get("version", "")))
    except ValueError as exc:
        issues.append(str(exc))
    if manifest.get("name") != contract.PACKAGE_ID:
        issues.append("manifest name mismatch")
    if manifest.get("description") != expected.get("description"):
        issues.append("manifest description mismatch")
    if manifest.get("license") != expected.get("license"):
        issues.append("manifest license mismatch")
    if manifest.get("author") != expected.get("author"):
        issues.append("manifest author mismatch")
    if manifest.get("skills") != "./skills/":
        issues.append("manifest skills path mismatch")
    if manifest.get("mcpServers") != "./.mcp.json":
        issues.append("manifest mcpServers path mismatch")
    interface = manifest.get("interface")
    if not isinstance(interface, dict):
        issues.append("manifest interface must be an object")
    else:
        for key, value in expected["interface"].items():
            if interface.get(key) != value:
                issues.append(f"manifest interface {key} mismatch")
    if "hooks" in manifest:
        issues.append("manifest must not include hooks")
    if "apps" in manifest:
        issues.append("manifest must not include apps for v1")
    return {"name": "plugin-manifest", "ok": not issues, "issues": issues}


def _check_claude_manifest(paths: GenerationPaths) -> dict[str, Any]:
    manifest = _read_json(paths.manifest_path)
    expected = plugin_manifest.build_claude_manifest(version=str(manifest.get("version", "")))
    issues: list[str] = []
    for key, value in expected.items():
        if manifest.get(key) != value:
            issues.append(f"manifest {key} mismatch")
    for field in plugin_manifest.CLAUDE_MANIFEST_FORBIDDEN_FIELDS:
        if field in manifest:
            issues.append(f"manifest must not include {field}")
    return {"name": "plugin-manifest", "ok": not issues, "issues": issues}


def _check_marketplace(paths: GenerationPaths) -> dict[str, Any]:
    payload = _read_json(paths.marketplace_path)
    issues = marketplace.validate_tokimon_marketplace_content(payload)
    marketplace_name = plugin_creator.read_marketplace_name(marketplace_path=paths.marketplace_path)
    if (
        marketplace_name["ok"]
        and not marketplace_name.get("skipped")
        and marketplace_name.get("stdout", "").strip() != contract.MARKETPLACE_NAME
    ):
        issues.append("plugin-creator marketplace name mismatch")
    return {
        "name": "generated-local-marketplace",
        "ok": not issues and marketplace_name["ok"],
        "issues": issues,
        "actual": payload,
        "plugin_creator_marketplace_name": marketplace_name,
    }


def _check_claude_marketplace(paths: GenerationPaths) -> dict[str, Any]:
    payload = _read_json(paths.marketplace_path)
    issues = marketplace.validate_claude_marketplace_content(payload)
    return {
        "name": "generated-local-marketplace",
        "ok": not issues,
        "issues": issues,
        "actual": payload,
    }


def _check_mcp(
    paths: GenerationPaths,
    *,
    install_route: route.InstallRoute,
    entries: tuple[contract.McpEntry, ...],
) -> dict[str, Any]:
    payload = _read_json(paths.package_root / ".mcp.json")
    servers = payload.get("mcpServers")
    issues: list[str] = []
    actual_ids = set(servers) if isinstance(servers, dict) else set()
    package_entries = {entry.server_id: entry for entry in entries}
    expected_ids = set(package_entries)
    if actual_ids != expected_ids:
        issues.append(
            f"MCP server ids do not match package contract: expected {sorted(expected_ids)}, "
            f"actual {sorted(actual_ids)}"
        )
    for server_id in sorted(actual_ids & expected_ids):
        entry = package_entries[server_id]
        server = servers.get(entry.server_id, {}) if isinstance(servers, dict) else {}
        if not isinstance(server, dict):
            issues.append(f"{entry.server_id} must be a JSON object")
            continue
        rendered = json.dumps(server)
        expected_server = mcp.build_mcp_server(
            entry=entry,
            install_route=install_route,
            platform=paths.platform,
        )
        if "/plugins/cache/" in rendered:
            issues.append(f"{entry.server_id} references plugin cache")
        for key, value in expected_server.items():
            if server.get(key) != value:
                issues.append(f"{entry.server_id} {key} does not match the live checkout route")
        for key in set(server) - set(expected_server):
            issues.append(f"{entry.server_id} has unexpected field {key}")
        if entry.module not in rendered:
            issues.append(f"{entry.server_id} does not route to {entry.module}")
    return {"name": "mcp-manifest", "ok": not issues, "issues": issues}


def _check_skills(paths: GenerationPaths, *, install_route: route.InstallRoute, policy: installation_policy.InstallationPolicy) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    expected_ids = policy.skill_ids(paths.source_root)
    actual_ids = sorted(path.parent.name for path in (paths.package_root / "skills").glob("*/SKILL.md"))
    if tuple(actual_ids) != tuple(sorted(expected_ids)):
        issues.append({"skill_inventory": actual_ids})
    for skill_id in expected_ids:
        skill_path = paths.package_root / "skills" / skill_id / "SKILL.md"
        if skill_path.is_symlink() or not skill_path.is_file():
            issues.append(
                {
                    "skill_id": skill_id,
                    "entrypoint": "generated SKILL.md must be a regular file",
                }
            )
            text = None
        else:
            try:
                text = skill_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                issues.append(
                    {
                        "skill_id": skill_id,
                        "entrypoint": f"unable to read generated SKILL.md: {exc}",
                    }
                )
                text = None

        try:
            expected_text = skills.build_packaged_skill(
                skill_id=skill_id,
                source_text=skills.load_source_skill(
                    source_root=paths.source_root,
                    skill_id=skill_id,
                ),
                install_route=install_route,
            )
        except (skills.SkillPackagingError, UnicodeError) as exc:
            issues.append(
                {
                    "skill_id": skill_id,
                    "source": f"unable to render canonical SKILL.md: {exc}",
                }
            )
            expected_text = None

        if text is not None:
            try:
                result = skills.validate_packaged_skill(
                    skill_id=skill_id,
                    text=text,
                    install_route=install_route,
                )
            except skills.SkillPackagingError as exc:
                issues.append(
                    {
                        "skill_id": skill_id,
                        "entrypoint": f"invalid generated SKILL.md: {exc}",
                    }
                )
            else:
                if not result["ok"]:
                    issues.append(result)

            if expected_text is not None and text != expected_text:
                issues.append(
                    {
                        "skill_id": skill_id,
                        "entrypoint": "generated SKILL.md differs from canonical source rendering",
                    }
                )

        metadata_path = paths.package_root / "skills" / skill_id / "agents" / "openai.yaml"
        try:
            source_metadata = skills.load_source_openai_metadata(
                source_root=paths.source_root,
                skill_id=skill_id,
            )
        except skills.SkillPackagingError as exc:
            issues.append(
                {
                    "skill_id": skill_id,
                    "source_metadata": f"unable to read canonical agents/openai.yaml: {exc}",
                }
            )
            source_metadata = None
        if paths.platform.platform_id == "codex" and source_metadata is not None:
            if metadata_path.is_symlink() or not metadata_path.is_file():
                issues.append({"skill_id": skill_id, "metadata": "missing regular agents/openai.yaml"})
            else:
                try:
                    generated_metadata = metadata_path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    issues.append(
                        {
                            "skill_id": skill_id,
                            "metadata": f"unable to read generated agents/openai.yaml: {exc}",
                        }
                    )
                else:
                    if generated_metadata != source_metadata:
                        issues.append(
                            {
                                "skill_id": skill_id,
                                "metadata": "agents/openai.yaml differs from source",
                            }
                        )
        elif metadata_path.exists():
            issues.append({"skill_id": skill_id, "metadata": "unexpected agents/openai.yaml"})
    return {"name": "packaged-skills", "ok": not issues, "issues": issues}
