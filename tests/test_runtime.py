from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tokimon.runtime as runtime_helpers


class RuntimeRootTests(unittest.TestCase):
    def test_source_checkout_does_not_switch_to_another_installed_workspace(self) -> None:
        checkout = Path(runtime_helpers.__file__).resolve().parent.parent
        with mock.patch.object(runtime_helpers.sys, "prefix", "/unrelated-environment"):
            self.assertEqual(runtime_helpers.source_root(), checkout)

    def test_installed_package_uses_its_bundled_workspace(self) -> None:
        for linked_share in (False, True):
            with self.subTest(linked_share=linked_share), tempfile.TemporaryDirectory() as tmpdir:
                environment = Path(tmpdir)
                site = environment / "lib/python3.12/site-packages"
                if linked_share:
                    assets = environment / "assets"
                    assets.mkdir()
                    (environment / "share").symlink_to(assets, target_is_directory=True)
                workspace = (environment / "share/tokimon").resolve()
                required = ("tokimon/__init__.py", "tokimon/settings.example.json", "skills/using-tokimon/skill.md", "docs/guidelines/repository-guidelines.md", "LICENSE", "THIRD_PARTY_NOTICES")
                for relative in required:
                    path = workspace / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("Fixture\n")
                python = environment / "bin/python"
                with mock.patch.object(runtime_helpers, "__file__", str(site / "tokimon/runtime.py")), mock.patch.object(runtime_helpers.sysconfig, "get_path", return_value=str(site)), mock.patch.object(runtime_helpers.sys, "prefix", str(environment)), mock.patch.object(runtime_helpers.sys, "executable", str(python)):
                    self.assertEqual(runtime_helpers.source_root(), workspace)
                    self.assertEqual(runtime_helpers.runtime_root(workspace), workspace / ".tokimon")
                    self.assertEqual(runtime_helpers.python_executable(workspace), python)

    def test_incomplete_installed_workspace_fails_instead_of_using_site_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = Path(tmpdir)
            site = environment / "site-packages"
            with mock.patch.object(runtime_helpers, "__file__", str(site / "tokimon/runtime.py")), mock.patch.object(runtime_helpers.sysconfig, "get_path", return_value=str(site)), mock.patch.object(runtime_helpers.sys, "prefix", str(environment)):
                with self.assertRaisesRegex(runtime_helpers.RuntimePathError, "installed Tokimon workspace is incomplete"):
                    runtime_helpers.source_root()

    def test_generated_python_preserves_the_primary_worktree_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary, linked = self.linked_worktree_fixture(Path(tmpdir))
            self.assertEqual(runtime_helpers.python_executable(linked), primary / ".venv/bin/python")

    def register_worktree(self, primary: Path, linked: Path) -> None:
        linked.mkdir(parents=True)
        worktree_git_dir = primary / ".git" / "worktrees" / linked.name
        worktree_git_dir.mkdir(parents=True)
        (linked / ".git").write_text(f"gitdir: {worktree_git_dir}\n", encoding="utf-8")
        (worktree_git_dir / "commondir").write_text("../..\n", encoding="utf-8")
        (worktree_git_dir / "gitdir").write_text(f"{linked / '.git'}\n", encoding="utf-8")

    def linked_worktree_fixture(self, root: Path) -> tuple[Path, Path]:
        primary = root / "primary"
        linked = root / "linked"
        self.register_worktree(primary, linked)
        return primary, linked

    def test_runtime_root_uses_primary_worktree_for_linked_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary, linked = self.linked_worktree_fixture(Path(tmpdir))

            self.assertEqual(runtime_helpers.primary_worktree_root(linked), primary)
            self.assertEqual(runtime_helpers.runtime_workspace_root(linked), primary)
            self.assertEqual(runtime_helpers.runtime_root(linked), primary / ".tokimon")

    def test_runtime_root_follows_relative_gitdir_and_commondir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primary = root / "primary"
            (primary / ".git").mkdir(parents=True)
            linked = root / "linked"
            linked.mkdir()
            metadata = root / "metadata"
            metadata.mkdir()
            (linked / ".git").write_text("gitdir: ../metadata\n", encoding="utf-8")
            (metadata / "commondir").write_text("../primary/.git\n", encoding="utf-8")

            self.assertEqual(runtime_helpers.runtime_root(linked), primary / ".tokimon")

    def test_runtime_root_uses_standard_layout_when_commondir_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary, linked = self.linked_worktree_fixture(Path(tmpdir))
            (primary / ".git" / "worktrees" / "linked" / "commondir").unlink()

            self.assertEqual(runtime_helpers.runtime_root(linked), primary / ".tokimon")

    def test_runtime_root_stays_in_direct_checkout_when_not_linked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".git").mkdir()

            self.assertIsNone(runtime_helpers.primary_worktree_root(root))
            self.assertEqual(runtime_helpers.runtime_workspace_root(root), root)
            self.assertEqual(runtime_helpers.runtime_root(root), root / ".tokimon")

    def test_resolve_runtime_path_maps_tokimon_relative_paths_to_runtime_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary, linked = self.linked_worktree_fixture(Path(tmpdir))

            self.assertEqual(
                runtime_helpers.resolve_runtime_path(".tokimon/workflow/pair", workspace_root=linked),
                primary / ".tokimon" / "workflow" / "pair",
            )
            self.assertEqual(
                runtime_helpers.resolve_runtime_path("tmp/output", workspace_root=linked),
                linked / "tmp" / "output",
            )

    def test_resolve_env_runtime_path_uses_primary_runtime_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary, linked = self.linked_worktree_fixture(Path(tmpdir))

            self.assertEqual(
                runtime_helpers.resolve_env_runtime_path(
                    default_relative_path=Path("tools") / "mcp" / "remote-browser",
                    env_var="TOKIMON_RB_RUNTIME_ROOT",
                    env={},
                    workspace_root=linked,
                ),
                primary / ".tokimon" / "tools" / "mcp" / "remote-browser",
            )

    def test_resolve_env_runtime_path_rejects_linked_worktree_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _primary, linked = self.linked_worktree_fixture(Path(tmpdir))
            bad_root = linked / ".tokimon" / "tools" / "mcp" / "remote-browser"

            with self.assertRaisesRegex(runtime_helpers.RuntimePathError, "linked worktree"):
                runtime_helpers.resolve_env_runtime_path(
                    default_relative_path=Path("tools") / "mcp" / "remote-browser",
                    env_var="TOKIMON_RB_RUNTIME_ROOT",
                    env={"TOKIMON_RB_RUNTIME_ROOT": str(bad_root)},
                    workspace_root=linked,
                )

    def test_resolve_env_runtime_path_resolves_relative_override_from_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary, linked = self.linked_worktree_fixture(Path(tmpdir))

            self.assertEqual(
                runtime_helpers.resolve_env_runtime_path(
                    default_relative_path=Path("unused"),
                    env_var="TOKIMON_RB_RUNTIME_ROOT",
                    env={"TOKIMON_RB_RUNTIME_ROOT": "runtime/browser"},
                    workspace_root=linked,
                ),
                primary / "runtime" / "browser",
            )

    def test_resolve_env_runtime_path_allows_explicit_primary_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary, linked = self.linked_worktree_fixture(Path(tmpdir))
            override = primary / ".tokimon" / "custom-browser"

            self.assertEqual(
                runtime_helpers.resolve_env_runtime_path(
                    default_relative_path=Path("unused"),
                    env_var="TOKIMON_RB_RUNTIME_ROOT",
                    env={"TOKIMON_RB_RUNTIME_ROOT": str(override)},
                    workspace_root=linked,
                ),
                override,
            )

    def test_resolve_env_runtime_path_rejects_nested_worktree_from_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "primary"
            linked = primary / ".worktrees" / "nested"
            self.register_worktree(primary, linked)

            with self.assertRaisesRegex(runtime_helpers.RuntimePathError, "linked worktree"):
                runtime_helpers.resolve_env_runtime_path(
                    default_relative_path=Path("unused"),
                    env_var="TOKIMON_RB_RUNTIME_ROOT",
                    env={"TOKIMON_RB_RUNTIME_ROOT": ".worktrees/nested/runtime"},
                    workspace_root=primary,
                )

    def test_resolve_env_runtime_path_rejects_sibling_worktree_and_symlink_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primary, linked = self.linked_worktree_fixture(root)
            sibling = root / "sibling"
            self.register_worktree(primary, sibling)
            alias = root / "alias"
            alias.symlink_to(sibling, target_is_directory=True)

            for override in (sibling / "runtime", alias / "runtime"):
                with self.subTest(override=override):
                    with self.assertRaisesRegex(runtime_helpers.RuntimePathError, "linked worktree"):
                        runtime_helpers.resolve_env_runtime_path(
                            default_relative_path=Path("unused"),
                            env_var="TOKIMON_RB_RUNTIME_ROOT",
                            env={"TOKIMON_RB_RUNTIME_ROOT": str(override)},
                            workspace_root=linked,
                        )

    def test_resolve_env_runtime_path_allows_non_worktree_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            external_runtime = root / "runtime" / "remote-browser"
            _primary, linked = self.linked_worktree_fixture(root)

            self.assertEqual(
                runtime_helpers.resolve_env_runtime_path(
                    default_relative_path=Path("tools") / "mcp" / "remote-browser",
                    env_var="TOKIMON_RB_RUNTIME_ROOT",
                    env={"TOKIMON_RB_RUNTIME_ROOT": str(external_runtime)},
                    workspace_root=linked,
                ),
                external_runtime,
            )

    def test_runtime_context_rejects_invalid_git_marker_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".git").write_text("not a gitdir marker\n", encoding="utf-8")

            with self.assertRaisesRegex(runtime_helpers.RuntimePathError, "Invalid gitdir marker"):
                runtime_helpers.runtime_path_context(workspace_root=root)

    def test_runtime_context_rejects_unreadable_common_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _primary, linked = self.linked_worktree_fixture(Path(tmpdir))
            git_dir = Path((linked / ".git").read_text(encoding="utf-8").split(":", 1)[1].strip())
            (git_dir / "commondir").unlink()
            (git_dir / "commondir").mkdir()

            with self.assertRaisesRegex(runtime_helpers.RuntimePathError, "Unable to read git commondir"):
                runtime_helpers.runtime_path_context(workspace_root=linked)


if __name__ == "__main__":
    unittest.main()
