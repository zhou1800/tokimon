from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.memory.mem_palace import mem_palace as wm
from tokimon.memory.mem_palace import exchange as exchange_helpers
from tokimon.memory.mem_palace import project_context

FIXED_TIMESTAMP = "2026-05-14T00:00:00+00:00"


def manifest_text(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def memory_exchange_row(
    memory_id: int = 1,
    *,
    summary: str = "Portable exchange memory",
    **updates: object,
) -> dict:
    row = {
        "id": memory_id,
        "kind": "decision",
        "summary": summary,
        "details": "Portable exchange keeps the module standalone.",
        "tags": "exchange",
        "source": "test",
        "scope": "memory",
        "room": "exchange",
        "entities": "",
        "importance": 3.0,
        "confidence": 1.0,
        "metadata": "{}",
        "pinned": 0,
        "created_at": FIXED_TIMESTAMP,
        "updated_at": FIXED_TIMESTAMP,
        "use_count": 0,
        "last_used_at": None,
    }
    row.update(updates)
    if "content_hash" not in updates:
        row["content_hash"] = wm.memory_content_hash(
            kind=str(row["kind"]),
            summary=str(row["summary"]),
            details=str(row["details"]),
            tags=wm.normalize_tags([str(row["tags"])]),
            source=str(row["source"]),
            scope=str(row["scope"]),
            room=str(row["room"]),
        )
    return row


def relationship_exchange_row() -> dict:
    return {
        "id": "rel_missing_subject_exports_missing_object_test",
        "subject_id": "missing_subject",
        "predicate": "exports",
        "object_id": "missing_object",
        "valid_from": None,
        "valid_to": None,
        "confidence": 1.0,
        "source_memory_id": None,
        "source": "test",
        "created_at": FIXED_TIMESTAMP,
        "updated_at": FIXED_TIMESTAMP,
    }


def write_exchange_fixture(
    root: Path,
    records: list[tuple[str, dict]],
    *,
    manifest_updates: dict | None = None,
    write_manifest: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "memory.jsonl"
    header = {
        "type": "header",
        "format": exchange_helpers.EXPORT_FORMAT,
        "version": exchange_helpers.EXPORT_VERSION,
        "exported_at": FIXED_TIMESTAMP,
        "tables": list(exchange_helpers.EXPORT_TABLES),
    }
    lines = [json.dumps(header, sort_keys=True)]
    counts = {table: 0 for table in exchange_helpers.EXPORT_TABLES}
    for table, data in records:
        counts[table] += 1
        lines.append(
            json.dumps(
                {
                    "type": table[:-1],
                    "table": table,
                    "data": data,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if write_manifest:
        manifest = {
            "format": exchange_helpers.EXPORT_MANIFEST_FORMAT,
            "schema_version": exchange_helpers.EXPORT_MANIFEST_SCHEMA_VERSION,
            "source_instance_id": "test-instance",
            "exported_at": FIXED_TIMESTAMP,
            "jsonl_format": exchange_helpers.EXPORT_FORMAT,
            "jsonl_version": exchange_helpers.EXPORT_VERSION,
            "streams": list(exchange_helpers.EXPORT_TABLES),
            "counts": counts,
            "jsonl_sha256": hashlib.sha256(jsonl_path.read_bytes()).hexdigest(),
        }
        manifest.update(manifest_updates or {})
        (root / "manifest.json").write_text(manifest_text(manifest), encoding="utf-8")
    return jsonl_path


def provenance_memory_kwargs(summary: str = "Imported relationship provenance memory") -> dict:
    return {
        "kind": "decision",
        "summary": summary,
        "details": "Relationship provenance must follow the imported memory row.",
        "tags": ["exchange", "provenance"],
        "source": "test",
        "scope": "memory",
        "room": "exchange",
        "entities": ["Imported Memory"],
        "metadata": {"owner": "memory"},
        "auto_extract_entities": False,
    }


def write_provenance_export(root: Path, *, summary: str = "Imported relationship provenance memory") -> Path:
    export_path = root / "memory.jsonl"
    with wm.MemPalace(root / "source.db") as palace:
        memory = palace.remember(**provenance_memory_kwargs(summary))
        palace.add_fact(
            "Imported Memory",
            "supports",
            "Relationship Provenance",
            source_memory_id=memory.id,
            source="test",
        )
        palace.export_jsonl(export_path)
    return export_path


def facade_state(palace: wm.MemPalace) -> dict[str, object]:
    return {
        "memories": [record.to_dict() for record in palace.recent(limit=1_000)],
        "relationships": [fact.to_dict() for fact in palace.timeline(limit=1_000)],
        "stats": palace.stats(),
    }


class MemPalaceFacadeTests(unittest.TestCase):
    def test_add_and_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            with wm.MemPalace(db_path) as palace:
                memory = palace.remember(
                    kind="avoid",
                    summary="Avoid deleting durable memory from use count alone",
                    details="Use count is a signal, not enough evidence by itself.",
                    tags=["memory", "tokimon"],
                    source="test",
                    pinned=True,
                )
                memories = palace.recent(limit=5)

        self.assertEqual(memory.id, 1)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].kind, "avoid")
        self.assertEqual(memories[0].summary, "Avoid deleting durable memory from use count alone")
        self.assertTrue(memories[0].pinned)
        self.assertEqual(memories[0].scope, "tokimon")
        self.assertEqual(memories[0].room, "general")
        self.assertEqual(memories[0].importance, 3.0)
        self.assertEqual(memories[0].use_count, 0)
        self.assertIsNone(memories[0].last_used_at)

    def test_search_matches_details_and_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            with wm.MemPalace(db_path) as palace:
                palace.remember(
                    kind="context",
                    summary="Tool registry diagnostics guide missing capability work",
                    details="Use the tool registry before workflow work when a capability is missing.",
                    tags=["tool", "registry"],
                    source="test",
                )
                results = palace.recall("tool registry", limit=5)
                used_memory = palace.get(results[0].memory.id)

        self.assertEqual(len(results), 1)
        self.assertIn("registry", results[0].memory.tags)
        self.assertEqual(results[0].rank, 1)
        self.assertGreater(results[0].score, 0)
        self.assertIsNotNone(used_memory)
        self.assertEqual(used_memory.use_count, 1)
        self.assertIsNotNone(used_memory.last_used_at)

    def test_search_rejects_queries_without_searchable_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            with wm.MemPalace(db_path) as palace:
                with self.assertRaisesRegex(ValueError, "searchable token"):
                    palace.recall("???")

    def test_scope_room_entities_metadata_and_duplicate_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            with wm.MemPalace(db_path) as palace:
                first = palace.remember(
                    kind="decision",
                    summary="Route example workflow artifacts to workflow memory",
                    details="The example workflow keeps local run state under .tokimon/workflow/example-workflow.",
                    tags=["example-workflow", "workflow"],
                    source="test",
                    scope="workflow",
                    room="example-workflow",
                    entities=["Example Workflow", "Durable Memory"],
                    importance=4.5,
                    confidence=0.9,
                    metadata={"owner": "workflow", "version": 1},
                    pinned=True,
                    auto_extract_entities=False,
                )
                duplicate = palace.remember(
                    kind="decision",
                    summary="Route example workflow artifacts to workflow memory",
                    details="The example workflow keeps local run state under .tokimon/workflow/example-workflow.",
                    tags=["example-workflow", "workflow"],
                    source="test",
                    scope="workflow",
                    room="example-workflow",
                    entities=["Durable Memory", "Example Workflow"],
                    importance=4.5,
                    confidence=0.9,
                    metadata={"version": 1, "owner": "workflow"},
                    pinned=True,
                    auto_extract_entities=False,
                )
                filtered = palace.recent(scope="workflow", room="example-workflow", entity="Example")

        self.assertEqual(first.id, duplicate.id)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].scope, "workflow")
        self.assertEqual(filtered[0].room, "example-workflow")
        self.assertEqual(filtered[0].entities, ("Example Workflow", "Durable Memory"))
        self.assertEqual(filtered[0].metadata, {"owner": "workflow", "version": 1})

    def test_supersession_hides_old_memory_only_from_default_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            export_path = root / "memory.jsonl"
            with wm.MemPalace(root / "memory.db") as palace:
                old = palace.remember(
                    kind="decision",
                    summary="Use optional package selection",
                    details="The package decision changed later.",
                    source="test",
                    auto_extract_entities=False,
                )
                current = palace.remember(
                    kind="decision",
                    summary="Use the fixed supported package set",
                    details="This package decision replaces the old decision.",
                    source="test",
                    metadata={"supersedes_memory_ids": [old.id, old.id]},
                    auto_extract_entities=False,
                )
                recent_ids = {memory.id for memory in palace.recent(limit=10)}
                essential_ids = {memory.id for memory in palace.essential(limit=10)}
                recalled_ids = {
                    result.memory.id
                    for result in palace.recall("package decision", limit=10)
                }
                retained_old = palace.get(old.id)
                report = palace.export_jsonl(export_path)
                memory_count = palace.stats()["memories"]

        self.assertEqual(current.metadata["supersedes_memory_ids"], [old.id])
        self.assertEqual(recent_ids, {current.id})
        self.assertEqual(essential_ids, {current.id})
        self.assertEqual(recalled_ids, {current.id})
        self.assertIsNotNone(retained_old)
        self.assertEqual(memory_count, 2)
        self.assertEqual(report["counts"]["memories"], 2)

    def test_forward_dangling_and_self_supersession_cannot_hide_later_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with wm.MemPalace(Path(tmpdir) / "memory.db") as palace:
                first = palace.remember(
                    kind="decision",
                    summary="First independent memory",
                    metadata={
                        "supersedes_memory_ids": [2, 99, 0, -1, True, "1"],
                    },
                    auto_extract_entities=False,
                )
                second = palace.remember(
                    kind="decision",
                    summary="Second independent memory",
                    metadata={"supersedes_memory_ids": [2]},
                    auto_extract_entities=False,
                )
                stored_first = palace.get(first.id)
                stored_second = palace.get(second.id)
                visible_ids = {memory.id for memory in palace.recent(limit=10)}

                palace.conn.execute(
                    "UPDATE memories SET metadata = ? WHERE id = ?",
                    [json.dumps({"supersedes_memory_ids": [second.id]}), first.id],
                )
                palace.conn.commit()
                guarded_ids = {memory.id for memory in palace.recent(limit=10)}

                palace.conn.execute(
                    "UPDATE memories SET metadata = ? WHERE id = ?",
                    ["not-json", first.id],
                )
                palace.conn.commit()
                malformed_ids = {memory.id for memory in palace.recent(limit=10)}

        self.assertEqual(stored_first.metadata, {})
        self.assertEqual(stored_second.metadata, {})
        self.assertEqual(visible_ids, {first.id, second.id})
        self.assertEqual(guarded_ids, {first.id, second.id})
        self.assertEqual(malformed_ids, {first.id, second.id})

    def test_same_content_semantic_conflicts_are_explicit_and_non_mutating(self) -> None:
        variants = {
            "pinned": {"pinned": False},
            "entities": {"entities": ["Entity A", "Entity C"]},
            "importance": {"importance": 4.0},
            "confidence": {"confidence": 0.8},
            "metadata": {"metadata": {"owner": "other"}},
        }
        base = {
            "kind": "decision",
            "summary": "Preserve semantic duplicate attributes",
            "details": "Same-content records must not silently lose requested attributes.",
            "tags": ["memory", "duplicates"],
            "source": "test",
            "scope": "memory",
            "room": "identity",
            "entities": ["Entity A", "Entity B"],
            "importance": 4.5,
            "confidence": 0.9,
            "metadata": {"owner": "memory"},
            "pinned": True,
            "auto_extract_entities": False,
        }

        for field, updates in variants.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmpdir:
                with wm.MemPalace(Path(tmpdir) / "memory.db") as palace:
                    memory = palace.remember(**base)
                    requested = dict(base)
                    requested.update(updates)
                    with self.assertRaises(ValueError) as caught:
                        palace.remember(**requested)
                    stored = palace.get(memory.id)
                    count = palace.stats()["memories"]

            message = str(caught.exception)
            self.assertIn("existing memory ids [1]", message)
            self.assertIn(field, message)
            self.assertNotIn("other", message)
            self.assertEqual(count, 1)
            self.assertIsNotNone(stored)
            self.assertTrue(stored.pinned)
            self.assertEqual(stored.entities, ("Entity A", "Entity B"))
            self.assertEqual(stored.importance, 4.5)
            self.assertEqual(stored.confidence, 0.9)
            self.assertEqual(stored.metadata, {"owner": "memory"})

    def test_allow_duplicate_preserves_same_content_attribute_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with wm.MemPalace(Path(tmpdir) / "memory.db") as palace:
                first = palace.remember(
                    kind="decision",
                    summary="Keep intentional same-content variants",
                    source="test",
                    pinned=False,
                    auto_extract_entities=False,
                )
                second = palace.remember(
                    kind="decision",
                    summary="Keep intentional same-content variants",
                    source="test",
                    pinned=True,
                    allow_duplicate=True,
                    auto_extract_entities=False,
                )
                exact_variant = palace.remember(
                    kind="decision",
                    summary="Keep intentional same-content variants",
                    source="test",
                    pinned=True,
                    auto_extract_entities=False,
                )
                rows = sorted((record.id, record.pinned) for record in palace.recent(limit=10))

        self.assertNotEqual(first.id, second.id)
        self.assertEqual(exact_variant.id, second.id)
        self.assertEqual(rows, [(1, False), (2, True)])

    def test_memory_weights_reject_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            with wm.MemPalace(db_path) as palace:
                with self.assertRaisesRegex(ValueError, "importance must be a number"):
                    palace.remember(
                        kind="decision",
                        summary="Reject invalid importance",
                        importance="bad",
                    )
                with self.assertRaisesRegex(ValueError, "confidence must be between 0 and 1"):
                    palace.add_fact("Mem Palace", "stores", "memory", confidence=2)
                with self.assertRaisesRegex(ValueError, "JSON-compatible"):
                    palace.remember(
                        kind="decision",
                        summary="Reject non-JSON metadata",
                        metadata={"score": float("nan")},
                    )

    def test_mem_palace_facade_recall_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"

            with wm.MemPalace(db_path) as palace:
                record = palace.remember(
                    kind="preference",
                    summary="Prefer scoped recall for workflow work",
                    details="Use scope and room filters before broad search.",
                    tags=["memory", "workflow"],
                    scope="workflow",
                    room="recall",
                    importance=5,
                )
                results = palace.recall("scoped recall", scope="workflow")
                stats = palace.stats()

        self.assertEqual(record.scope, "workflow")
        self.assertEqual(results[0].memory.summary, "Prefer scoped recall for workflow work")
        self.assertEqual(stats["memories"], 1)

    def test_temporal_relationships_query_timeline_and_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            with wm.MemPalace(db_path) as palace:
                memory = palace.remember(
                    kind="fact",
                    summary="Mem Palace stores local durable memory",
                    tags=["memory"],
                    source="test",
                )
                relationship_id = palace.add_fact(
                    "Mem Palace",
                    "stores",
                    "local durable memory",
                    valid_from="2026-04-24",
                    source_memory_id=memory.id,
                )
                duplicate_id = palace.add_fact(
                    "Mem Palace",
                    "stores",
                    "local durable memory",
                    valid_from="2026-04-24",
                    source_memory_id=memory.id,
                )
                facts = palace.entity_facts("Mem Palace", as_of="2026-04-24")
                history = palace.timeline("local durable memory")
                count = palace.invalidate_fact(
                    "Mem Palace",
                    "stores",
                    "local durable memory",
                    ended="2026-05-01",
                )
                current_after = palace.entity_facts("Mem Palace", as_of="2026-05-02")

        self.assertEqual(relationship_id, duplicate_id)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].object, "local durable memory")
        self.assertEqual(history[0].subject, "Mem Palace")
        self.assertEqual(count, 1)
        self.assertEqual(current_after, [])

    def test_briefing_includes_repository_guidelines_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            c4_overview_path = Path(tmpdir) / "docs" / "guidelines" / "repository-guidelines.md"
            c4_overview_path.parent.mkdir(parents=True, exist_ok=True)
            c4_overview_path.write_text(
                """---
summary: "Top-level repository routing."
read_when:
  - Starting Tokimon repository work
---

# Tokimon Repository Guidelines

## Purpose
Test repository guidelines
""",
                encoding="utf-8",
            )

            with wm.MemPalace(db_path, c4_overview_path=c4_overview_path) as palace:
                palace.remember(
                    kind="decision",
                    summary="Use briefing at startup",
                    details="Read C4 architecture and durable memory together.",
                    tags=["memory", "briefing"],
                    source="test",
                )
                briefing = palace.briefing(limit=5)

        self.assertIn("== Repository Guidelines", briefing)
        self.assertIn("Test repository guidelines", briefing)
        self.assertNotIn("summary: \"Top-level repository routing.\"", briefing)
        self.assertIn("== Essential Memory ==", briefing)
        self.assertIn("== Recent Memory ==", briefing)
        self.assertIn("Use briefing at startup", briefing)

    def test_scoped_startup_briefing_excludes_unrelated_memory_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            overview = tmp / "repository-guidelines.md"
            overview.write_text("# Repository Guidelines\n", encoding="utf-8")
            with wm.MemPalace(tmp / "memory.db", c4_overview_path=overview) as palace:
                palace.remember(
                    kind="decision",
                    summary="Tokimon startup memory",
                    scope="tokimon",
                    pinned=True,
                    auto_extract_entities=False,
                )
                palace.remember(
                    kind="context",
                    summary="Unrelated job search memory",
                    scope="job-search",
                    pinned=True,
                    auto_extract_entities=False,
                )
                scoped = palace.briefing(scope="tokimon")
                generic = palace.briefing()

        self.assertIn("Tokimon startup memory", scoped)
        self.assertNotIn("Unrelated job search memory", scoped)
        self.assertIn("Unrelated job search memory", generic)

    def test_tokimon_repository_startup_keeps_its_compatibility_scope(self) -> None:
        self.assertIn(
            "briefing --scope tokimon --limit 8",
            (ROOT / "AGENTS.md").read_text(encoding="utf-8"),
        )

    def test_installed_memory_skills_select_the_adopting_project(self) -> None:
        for relative_path in ("skills/using-tokimon/skill.md", "skills/memory/skill.md"):
            with self.subTest(path=relative_path):
                text = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(
                    'briefing --project-root "$PWD" --include-scope global --limit 8',
                    text,
                )
                self.assertNotIn("briefing --scope tokimon --limit 8", text)

    def test_scope_parser_accepts_explicit_and_project_selection(self) -> None:
        parser = wm.build_parser()
        self.assertEqual(parser.parse_args(["briefing", "--scope", "tokimon"]).scope, "tokimon")
        self.assertIsNone(parser.parse_args(["briefing", "--limit", "8"]).scope)
        parsed = parser.parse_args(
            ["briefing", "--project-root", str(ROOT), "--include-scope", "global"]
        )
        self.assertEqual(parsed.project_root, str(ROOT))
        self.assertEqual(parsed.include_scope, ["global"])

    def test_project_scope_preserves_tokimon_compatibility_and_separates_projects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tokimon = Path(tmpdir) / "tokimon"
            first = Path(tmpdir) / "First Project"
            second = Path(tmpdir) / "Second Project"
            tokimon.mkdir()
            first.mkdir()
            second.mkdir()
            non_git = subprocess.CompletedProcess(
                args=[],
                returncode=128,
                stdout="",
                stderr="fatal: not a git repository (or any parent): .git\n",
            )
            with patch.object(project_context.subprocess, "run", return_value=non_git):
                tokimon_scope = project_context.resolve_project_memory_scope(
                    tokimon,
                    tokimon_root=tokimon,
                )
                first_scope = project_context.resolve_project_memory_scope(
                    first,
                    tokimon_root=tokimon,
                )
                repeated_scope = project_context.resolve_project_memory_scope(
                    first,
                    tokimon_root=tokimon,
                )
                second_scope = project_context.resolve_project_memory_scope(
                    second,
                    tokimon_root=tokimon,
                )

        self.assertEqual(tokimon_scope.scope, "tokimon")
        self.assertEqual(first_scope, repeated_scope)
        self.assertRegex(first_scope.scope, r"^project:first-project:[0-9a-f]{12}$")
        self.assertNotEqual(first_scope.scope, second_scope.scope)

    def test_linked_git_worktrees_share_project_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tokimon = Path(tmpdir) / "tokimon"
            primary = Path(tmpdir) / "primary"
            linked = Path(tmpdir) / "linked"
            tokimon.mkdir()
            primary.mkdir()
            linked.mkdir()

            def git_identity(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                root = Path(command[2])
                if root == linked:
                    top_level = linked
                    common_dir = primary / ".git"
                else:
                    top_level = root
                    common_dir = root / ".git"
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout=f"{top_level}\n{common_dir}\n",
                    stderr="",
                )

            with patch.object(project_context.subprocess, "run", side_effect=git_identity):
                primary_scope = project_context.resolve_project_memory_scope(
                    primary,
                    tokimon_root=tokimon,
                )
                linked_scope = project_context.resolve_project_memory_scope(
                    linked,
                    tokimon_root=tokimon,
                )

        self.assertEqual(primary_scope.scope, linked_scope.scope)
        self.assertEqual(primary_scope.identity_kind, "git-common-dir")

    def test_project_scope_fails_when_git_cannot_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(project_context.subprocess, "run", side_effect=OSError("missing")):
                with self.assertRaisesRegex(
                    ValueError,
                    "unable to execute Git while resolving project identity",
                ):
                    project_context.resolve_project_memory_scope(tmpdir, tokimon_root=tmpdir)

    def test_project_scope_fails_when_git_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            timeout = subprocess.TimeoutExpired(cmd="git", timeout=5)
            with patch.object(project_context.subprocess, "run", side_effect=timeout):
                with self.assertRaisesRegex(
                    ValueError,
                    "Git timed out while resolving project identity",
                ):
                    project_context.resolve_project_memory_scope(tmpdir, tokimon_root=tmpdir)

    def test_project_scope_fails_on_unexpected_git_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            failed = subprocess.CompletedProcess(
                args=[],
                returncode=128,
                stdout="",
                stderr="fatal: detected dubious ownership in repository\n",
            )
            with patch.object(project_context.subprocess, "run", return_value=failed):
                with self.assertRaisesRegex(ValueError, "Git could not resolve project identity"):
                    project_context.resolve_project_memory_scope(tmpdir, tokimon_root=tmpdir)

    def test_project_scope_fails_on_malformed_git_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            malformed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=f"{tmpdir}\n",
                stderr="",
            )
            with patch.object(project_context.subprocess, "run", return_value=malformed):
                with self.assertRaisesRegex(
                    ValueError,
                    "Git returned an invalid project identity",
                ):
                    project_context.resolve_project_memory_scope(tmpdir, tokimon_root=tmpdir)

    def test_project_and_global_scope_selection_excludes_unrelated_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tokimon = tmp / "tokimon"
            project = tmp / "adopting-project"
            tokimon.mkdir()
            project.mkdir()
            non_git = subprocess.CompletedProcess(
                args=[],
                returncode=128,
                stdout="",
                stderr="fatal: not a git repository (or any parent): .git\n",
            )
            with patch.object(project_context.subprocess, "run", return_value=non_git):
                selected_scope = project_context.resolve_project_memory_scope(
                    project,
                    tokimon_root=tokimon,
                ).scope
            with wm.MemPalace(tmp / "memory.db") as palace:
                for summary, scope in (
                    ("Target project memory", selected_scope),
                    ("Global preference", "global"),
                    ("Unrelated project memory", "project:unrelated:000000000000"),
                ):
                    palace.remember(
                        kind="context",
                        summary=summary,
                        scope=scope,
                        auto_extract_entities=False,
                    )
                memories = palace.recent(
                    scopes=(selected_scope, "global"),
                    limit=10,
                )

        summaries = {memory.summary for memory in memories}
        self.assertEqual(summaries, {"Target project memory", "Global preference"})

    def test_cli_routes_project_and_global_memory_without_cross_project_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            project = tmp / "adopting-project"
            project.mkdir()
            db_path = tmp / "memory.db"
            overview = tmp / "repository-guidelines.md"
            overview.write_text("# Repository Guidelines\n", encoding="utf-8")
            non_git = subprocess.CompletedProcess(
                args=[],
                returncode=128,
                stdout="",
                stderr="fatal: not a git repository (or any parent): .git\n",
            )

            def run_cli(*arguments: str) -> str:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exit_code = wm.main(
                        [
                            "--db",
                            str(db_path),
                            "--c4-overview-path",
                            str(overview),
                            *arguments,
                        ]
                    )
                self.assertEqual(exit_code, 0)
                return output.getvalue()

            with patch.object(project_context.subprocess, "run", return_value=non_git):
                run_cli(
                    "add",
                    "--kind",
                    "context",
                    "--summary",
                    "Target project guidance",
                    "--project-root",
                    str(project),
                )
                run_cli(
                    "add",
                    "--kind",
                    "context",
                    "--summary",
                    "Global guidance",
                    "--scope",
                    "global",
                )
                run_cli(
                    "add",
                    "--kind",
                    "context",
                    "--summary",
                    "Unrelated project guidance",
                    "--scope",
                    "project:unrelated:000000000000",
                )
                search_payload = json.loads(
                    run_cli(
                        "search",
                        "guidance",
                        "--project-root",
                        str(project),
                        "--include-scope",
                        "global",
                        "--json",
                    )
                )
                briefing_payload = json.loads(
                    run_cli(
                        "briefing",
                        "--project-root",
                        str(project),
                        "--include-scope",
                        "global",
                        "--json",
                    )
                )

        search_summaries = {item["summary"] for item in search_payload}
        briefing_summaries = {
            item["summary"] for item in briefing_payload["memories"]
        }
        self.assertEqual(search_summaries, {"Target project guidance", "Global guidance"})
        self.assertEqual(briefing_summaries, {"Target project guidance", "Global guidance"})

    def test_cli_rejects_a_missing_project_before_creating_the_memory_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "memory.db"
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                exit_code = wm.main(
                    [
                        "--db",
                        str(db_path),
                        "briefing",
                        "--project-root",
                        str(tmp / "missing-project"),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("project root does not exist", error.getvalue())
            self.assertFalse(db_path.exists())

    def test_cli_rejects_git_failure_before_creating_the_memory_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            project = tmp / "project"
            project.mkdir()
            db_path = tmp / "memory.db"
            error = io.StringIO()
            with patch.object(
                project_context.subprocess,
                "run",
                side_effect=OSError("missing"),
            ), contextlib.redirect_stderr(error):
                exit_code = wm.main(
                    [
                        "--db",
                        str(db_path),
                        "briefing",
                        "--project-root",
                        str(project),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn(
                "unable to execute Git while resolving project identity",
                error.getvalue(),
            )
            self.assertFalse(db_path.exists())

    def test_cli_reports_actionable_sqlite_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "database-is-a-directory"
            db_path.mkdir()
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                exit_code = wm.main(["--db", str(db_path), "stats"])

        self.assertEqual(exit_code, 2)
        self.assertIn("Mem Palace command 'stats' failed for database", error.getvalue())
        self.assertIn(str(db_path), error.getvalue())
        self.assertIn("unable to open database file", error.getvalue())

    def test_export_import_and_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_db = tmp / "source.db"
            export_path = tmp / "memory.jsonl"
            backup_path = tmp / "backup.db"
            target_db = tmp / "target.db"

            with wm.MemPalace(source_db) as source:
                source.remember(
                    kind="decision",
                    summary="Export memory as JSONL",
                    details="Portable exchange keeps the module standalone.",
                    tags=["exchange"],
                    source="test",
                    scope="memory",
                    room="exchange",
                    entities=["Entity B", "Entity A"],
                    metadata={"label": "mémoire"},
                    auto_extract_entities=False,
                )
                source.add_fact("Mem Palace", "exports", "JSONL")
                export_report = source.export_jsonl(export_path)
                backup_report = source.backup(backup_path)
                source_memories = source.recent(limit=10)
                source_facts = source.timeline(limit=10)

            with wm.MemPalace(target_db) as target:
                import_report = target.import_jsonl(export_path)
                target_memories = target.recent(limit=10)
                target_facts = target.timeline(limit=10)
                memories = target.recall("portable exchange", scope="memory")
                facts = target.entity_facts("Mem Palace")

            manifest_path = Path(export_report["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(export_report["counts"]["memories"], 1)
            self.assertEqual(manifest_path, export_path.with_name("manifest.json").resolve())
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), manifest_text(manifest))
            self.assertEqual(manifest["counts"], export_report["counts"])
            self.assertEqual(manifest["streams"], list(exchange_helpers.EXPORT_TABLES))
            self.assertEqual(manifest["jsonl_sha256"], hashlib.sha256(export_path.read_bytes()).hexdigest())
            self.assertEqual(export_report["jsonl_sha256"], manifest["jsonl_sha256"])
            self.assertTrue(export_report["source_instance_id"])
            self.assertGreater(backup_report["bytes"], 0)
            self.assertEqual(import_report["counts"]["memories"], 1)
            self.assertEqual(import_report["manifest_path"], str(manifest_path))
            self.assertEqual(
                [memory.to_dict() for memory in target_memories],
                [memory.to_dict() for memory in source_memories],
            )
            self.assertEqual(
                [fact.to_dict() for fact in target_facts],
                [fact.to_dict() for fact in source_facts],
            )
            self.assertEqual(memories[0].memory.summary, "Export memory as JSONL")
            self.assertEqual(facts[0].object, "JSONL")

    def test_import_rejects_manifest_validation_errors_before_writing(self) -> None:
        cases = {
            "wrong_hash": {"jsonl_sha256": "not-the-real-hash"},
            "wrong_count": {"counts": {"memories": 2, "entities": 0, "relationships": 0}},
            "unsupported_schema": {"schema_version": 999},
        }

        for name, manifest_updates in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    exchange_path = write_exchange_fixture(
                        tmp,
                        [("memories", memory_exchange_row())],
                        manifest_updates=manifest_updates,
                    )
                    db_path = tmp / "target.db"
                    with wm.MemPalace(db_path) as target:
                        before = facade_state(target)
                        with self.assertRaisesRegex(ValueError, "stage=manifest"):
                            target.import_jsonl(exchange_path)
                        after = facade_state(target)

                self.assertEqual(after, before)

    def test_import_rejects_malformed_records_with_or_without_manifest(self) -> None:
        for require_manifest in (True, False):
            with self.subTest(require_manifest=require_manifest):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    malformed = memory_exchange_row()
                    del malformed["summary"]
                    exchange_path = write_exchange_fixture(
                        tmp,
                        [("memories", malformed)],
                        write_manifest=require_manifest,
                    )
                    with wm.MemPalace(tmp / "target.db") as target:
                        with self.assertRaisesRegex(
                            ValueError,
                            "stage=record_schema stream=memories record_offset=1",
                        ):
                            target.import_jsonl(exchange_path, require_manifest=require_manifest)

    def test_import_rejects_invalid_memory_identifiers_before_writing(self) -> None:
        for invalid_id in (True, 0, -1, 1.5, "1"):
            with self.subTest(invalid_id=invalid_id), tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                row = memory_exchange_row()
                row["id"] = invalid_id
                exchange_path = write_exchange_fixture(
                    tmp / "exchange",
                    [("memories", row)],
                )
                with wm.MemPalace(tmp / "target.db") as target:
                    before = facade_state(target)
                    with self.assertRaisesRegex(
                        ValueError,
                        "memory id must be a positive integer",
                    ):
                        target.import_jsonl(exchange_path)
                    after = facade_state(target)

            self.assertEqual(after, before)

    def test_import_apply_failure_rolls_back_partial_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            exchange_path = write_exchange_fixture(
                tmp,
                [
                    ("memories", memory_exchange_row()),
                    ("relationships", relationship_exchange_row()),
                ],
            )
            with wm.MemPalace(tmp / "target.db") as target:
                target.remember(
                    kind="context",
                    summary="Existing row remains after failed import",
                    source="test",
                )
                before = facade_state(target)
                with self.assertRaisesRegex(
                    ValueError,
                    "stage=apply stream=relationships record_offset=1",
                ):
                    target.import_jsonl(exchange_path)
                after = facade_state(target)

        self.assertEqual(after, before)

    def test_no_manifest_import_accepts_legacy_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            exchange_path = write_exchange_fixture(
                tmp,
                [("memories", memory_exchange_row(summary="Legacy JSONL import"))],
                write_manifest=False,
            )
            with wm.MemPalace(tmp / "target.db") as target:
                report = target.import_jsonl(exchange_path, require_manifest=False)
                results = target.recall("legacy jsonl", scope="memory")

        self.assertIsNone(report["manifest_path"])
        self.assertEqual(report["counts"]["memories"], 1)
        self.assertEqual(results[0].memory.summary, "Legacy JSONL import")

    def test_import_populates_omitted_canonical_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            row = memory_exchange_row(summary="Legacy row without a content hash")
            expected_hash = row.pop("content_hash")
            exchange_path = write_exchange_fixture(tmp, [("memories", row)])
            with wm.MemPalace(tmp / "target.db") as target:
                report = target.import_jsonl(exchange_path)
                stored = target.get(1)

        self.assertEqual(report["counts"]["memories"], 1)
        self.assertIsNotNone(stored)
        self.assertEqual(stored.content_hash, expected_hash)

    def test_import_preserves_existing_hash_contract_for_legacy_kind_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            row = memory_exchange_row(kind="Decision")
            exchange_path = write_exchange_fixture(tmp, [("memories", row)])
            with wm.MemPalace(tmp / "target.db") as target:
                report = target.import_jsonl(exchange_path)
                stored = target.get(1)

        self.assertEqual(report["counts"]["memories"], 1)
        self.assertIsNotNone(stored)
        self.assertEqual(stored.kind, "Decision")
        self.assertEqual(stored.content_hash, row["content_hash"])

    def test_import_rejects_noncanonical_content_hash_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            row = memory_exchange_row(content_hash="not-canonical")
            exchange_path = write_exchange_fixture(tmp, [("memories", row)])
            with wm.MemPalace(tmp / "target.db") as target:
                before = facade_state(target)
                with self.assertRaisesRegex(
                    ValueError,
                    "stage=record_schema stream=memories record_offset=1.*content_hash",
                ):
                    target.import_jsonl(exchange_path)
                after = facade_state(target)

        self.assertEqual(after, before)

    def test_import_duplicate_semantics_skip_conflict_and_explicit_keep(self) -> None:
        base = {
            "kind": "decision",
            "summary": "Portable exchange memory",
            "details": "Portable exchange keeps the module standalone.",
            "tags": ["exchange"],
            "source": "test",
            "scope": "memory",
            "room": "exchange",
            "entities": ["Entity A", "Entity B"],
            "metadata": {"owner": "memory", "version": 1},
            "auto_extract_entities": False,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            exact_row = memory_exchange_row(
                memory_id=42,
                entities="Entity B, Entity A",
                metadata={"version": 1, "owner": "memory"},
            )
            exact_path = write_exchange_fixture(tmp / "exact", [("memories", exact_row)])
            conflict_row = memory_exchange_row(
                memory_id=1,
                entities="Entity A, Entity B",
                metadata={"owner": "memory", "version": 1},
                pinned=1,
            )
            conflict_path = write_exchange_fixture(tmp / "conflict", [("memories", conflict_row)])
            with wm.MemPalace(tmp / "target.db") as target:
                original = target.remember(**base)
                exact_report = target.import_jsonl(exact_path)
                before_conflict = facade_state(target)
                with self.assertRaises(ValueError) as caught:
                    target.import_jsonl(conflict_path)
                after_conflict = facade_state(target)
                keep_report = target.import_jsonl(
                    conflict_path,
                    skip_duplicate_memories=False,
                )
                rows = sorted((memory.id, memory.pinned) for memory in target.recent(limit=10))

        self.assertEqual(original.id, 1)
        self.assertEqual(exact_report["counts"]["memories"], 0)
        self.assertEqual(exact_report["skipped"], 1)
        self.assertEqual(before_conflict, after_conflict)
        self.assertRegex(
            str(caught.exception),
            r"stage=apply stream=memories record_offset=1.*existing memory ids \[1\].*pinned",
        )
        self.assertEqual(keep_report["counts"]["memories"], 1)
        self.assertEqual(rows, [(1, False), (2, True)])

    def test_import_remaps_occupied_memory_id_and_relationship_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            exchange_path = write_provenance_export(tmp)
            with wm.MemPalace(tmp / "target.db") as target:
                existing = target.remember(
                    kind="context",
                    summary="Unrelated destination memory",
                    source="test",
                    auto_extract_entities=False,
                )
                report = target.import_jsonl(exchange_path)
                memories = sorted((memory.id, memory.summary) for memory in target.recent(limit=10))
                relationship = next(fact for fact in target.timeline() if fact.predicate == "supports")

        self.assertEqual(existing.id, 1)
        self.assertEqual(report["counts"]["memories"], 1)
        self.assertEqual(
            memories,
            [(1, "Unrelated destination memory"), (2, "Imported relationship provenance memory")],
        )
        self.assertEqual(relationship.source_memory_id, 2)

    def test_import_remaps_supersession_after_id_collision_and_reordered_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            old_row = memory_exchange_row(
                memory_id=1,
                summary="Imported obsolete package decision",
            )
            current_row = memory_exchange_row(
                memory_id=2,
                summary="Imported current package decision",
                metadata={"supersedes_memory_ids": [1, 2, 99, True]},
            )
            exchange_path = write_exchange_fixture(
                tmp / "exchange",
                [("memories", current_row), ("memories", old_row)],
            )
            with wm.MemPalace(tmp / "target.db") as target:
                unrelated = target.remember(
                    kind="context",
                    summary="Unrelated destination memory",
                    source="test",
                    auto_extract_entities=False,
                )
                report = target.import_jsonl(exchange_path)
                all_rows = {
                    memory.summary: memory
                    for memory in (
                        target.get(1),
                        target.get(2),
                        target.get(3),
                    )
                    if memory is not None
                }
                visible = {memory.summary for memory in target.recent(limit=10)}
                recalled = {
                    result.memory.summary
                    for result in target.recall("package decision", limit=10)
                }

        self.assertEqual(unrelated.id, 1)
        self.assertEqual(report["counts"]["memories"], 2)
        self.assertEqual(
            all_rows["Imported current package decision"].metadata,
            {"supersedes_memory_ids": [2]},
        )
        self.assertGreater(
            all_rows["Imported current package decision"].id,
            all_rows["Imported obsolete package decision"].id,
        )
        self.assertEqual(
            visible,
            {"Unrelated destination memory", "Imported current package decision"},
        )
        self.assertEqual(recalled, {"Imported current package decision"})

    def test_import_remaps_supersession_through_exact_skipped_predecessor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            old_row = memory_exchange_row(
                memory_id=1,
                summary="Exact imported predecessor",
            )
            current_row = memory_exchange_row(
                memory_id=2,
                summary="Successor of exact imported predecessor",
                metadata={"supersedes_memory_ids": [1]},
            )
            exchange_path = write_exchange_fixture(
                tmp / "exchange",
                [("memories", old_row), ("memories", current_row)],
            )
            with wm.MemPalace(tmp / "target.db") as target:
                target.remember(
                    kind="context",
                    summary="Unrelated destination memory",
                    source="test",
                    auto_extract_entities=False,
                )
                exact = target.remember(
                    kind="decision",
                    summary=str(old_row["summary"]),
                    details=str(old_row["details"]),
                    tags=[str(old_row["tags"])],
                    source=str(old_row["source"]),
                    scope=str(old_row["scope"]),
                    room=str(old_row["room"]),
                    auto_extract_entities=False,
                )
                report = target.import_jsonl(exchange_path)
                visible = {memory.summary: memory for memory in target.recent(limit=10)}
                successor = visible["Successor of exact imported predecessor"]

        self.assertEqual(exact.id, 2)
        self.assertEqual(report["counts"]["memories"], 1)
        self.assertEqual(report["skipped"], 1)
        self.assertEqual(successor.metadata, {"supersedes_memory_ids": [exact.id]})
        self.assertGreater(successor.id, exact.id)
        self.assertNotIn("Exact imported predecessor", visible)

    def test_import_builds_complete_memory_map_before_applying_reordered_relationships(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            exchange_path = write_provenance_export(tmp / "exchange")
            lines = exchange_path.read_text(encoding="utf-8").splitlines()
            header, *records = lines
            parsed_records = [json.loads(line) for line in records]
            stream_order = {"relationships": 0, "entities": 1, "memories": 2}
            parsed_records.sort(key=lambda row: stream_order[str(row["table"])])
            exchange_path.write_text(
                "\n".join([header, *(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in parsed_records)])
                + "\n",
                encoding="utf-8",
            )
            manifest_path = exchange_path.with_name("manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["jsonl_sha256"] = hashlib.sha256(exchange_path.read_bytes()).hexdigest()
            manifest_path.write_text(manifest_text(manifest), encoding="utf-8")

            with wm.MemPalace(tmp / "target.db") as target:
                existing = target.remember(
                    kind="context",
                    summary="Unrelated row occupying the exported memory id",
                    source="test",
                    auto_extract_entities=False,
                )
                report = target.import_jsonl(exchange_path)
                imported = next(
                    memory
                    for memory in target.recent(limit=10)
                    if memory.summary == "Imported relationship provenance memory"
                )
                relationship = next(fact for fact in target.timeline() if fact.predicate == "supports")

        self.assertEqual(existing.id, 1)
        self.assertEqual(report["counts"]["memories"], 1)
        self.assertEqual(relationship.source_memory_id, imported.id)

    def test_import_exact_skipped_memory_remaps_relationship_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            exchange_path = write_provenance_export(tmp)
            with wm.MemPalace(tmp / "target.db") as target:
                target.remember(
                    kind="context",
                    summary="Unrelated destination memory",
                    source="test",
                    auto_extract_entities=False,
                )
                exact = target.remember(**provenance_memory_kwargs())
                report = target.import_jsonl(exchange_path)
                relationship = next(fact for fact in target.timeline() if fact.predicate == "supports")
                memory_count = target.stats()["memories"]

        self.assertEqual(exact.id, 2)
        self.assertEqual(report["counts"]["memories"], 0)
        self.assertEqual(report["skipped"], 1)
        self.assertEqual(memory_count, 2)
        self.assertEqual(relationship.source_memory_id, 2)

    def test_import_replace_keeps_exported_memory_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imported = memory_exchange_row(memory_id=1, summary="Replacement memory")
            exchange_path = write_exchange_fixture(tmp, [("memories", imported)])
            with wm.MemPalace(tmp / "target.db") as target:
                target.remember(
                    kind="context",
                    summary="Destination row to replace",
                    source="test",
                    auto_extract_entities=False,
                )
                report = target.import_jsonl(exchange_path, replace=True)
                rows = sorted((memory.id, memory.summary) for memory in target.recent(limit=10))

        self.assertEqual(report["counts"]["memories"], 1)
        self.assertEqual(rows, [(1, "Replacement memory")])

    def test_import_replace_keeps_valid_supersession_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            old_row = memory_exchange_row(
                memory_id=1,
                summary="Replacement predecessor",
            )
            current_row = memory_exchange_row(
                memory_id=2,
                summary="Replacement successor",
                metadata={"supersedes_memory_ids": [1]},
            )
            exchange_path = write_exchange_fixture(
                tmp / "exchange",
                [("memories", current_row), ("memories", old_row)],
            )
            with wm.MemPalace(tmp / "target.db") as target:
                target.remember(
                    kind="context",
                    summary="Destination row one",
                    auto_extract_entities=False,
                )
                target.remember(
                    kind="context",
                    summary="Destination row two",
                    auto_extract_entities=False,
                )
                report = target.import_jsonl(exchange_path, replace=True)
                predecessor = target.get(1)
                successor = target.get(2)
                visible = target.recent(limit=10)

        self.assertEqual(report["counts"]["memories"], 2)
        self.assertIsNotNone(predecessor)
        self.assertIsNotNone(successor)
        assert predecessor is not None
        assert successor is not None
        self.assertEqual(predecessor.summary, "Replacement predecessor")
        self.assertEqual(successor.metadata, {"supersedes_memory_ids": [1]})
        self.assertEqual([memory.id for memory in visible], [2])

    def test_ingest_paths_stores_text_file_as_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            doc = tmp / "docs" / "note.md"
            doc.parent.mkdir(parents=True)
            doc.write_text("# Durable Memory\n\nMem Palace should ingest explicit docs only.\n", encoding="utf-8")
            db_path = tmp / "memory.db"

            with wm.MemPalace(db_path) as palace:
                report = palace.ingest_paths(
                    [doc],
                    workspace_root=tmp,
                    scope="memory",
                    room="ingest",
                    tags=["docs"],
                )
                results = palace.recall("explicit docs", scope="memory", room="ingest")

        self.assertEqual(report["documents"], 1)
        self.assertEqual(len(report["memories"]), 1)
        self.assertEqual(report["memories"][0]["kind"], "artifact")
        self.assertIn("ingest:docs/note.md#", report["memories"][0]["source"])
        self.assertEqual(results[0].memory.summary, "Ingested docs/note.md: Durable Memory")

    def test_add_memory_rejects_secret_like_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            with wm.MemPalace(db_path) as palace:
                before = palace.stats()["memories"]
                with self.assertRaisesRegex(ValueError, "secret-like material"):
                    palace.remember(
                        kind="context",
                        summary="Do not store credential material",
                        details="api_key=sensitive_value_abcdef123456",
                        source="test",
                    )
                after = palace.stats()["memories"]

        self.assertEqual(before, 0)
        self.assertEqual(after, 0)

    def test_import_rejects_secret_like_memory_records_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            secret_like_row = memory_exchange_row()
            secret_like_row["details"] = "password=sensitive_value_abcdef123456"
            exchange_path = write_exchange_fixture(
                tmp,
                [("memories", secret_like_row)],
            )
            with wm.MemPalace(tmp / "target.db") as target:
                before = facade_state(target)
                with self.assertRaisesRegex(ValueError, "secret-like material"):
                    target.import_jsonl(exchange_path)
                after = facade_state(target)

        self.assertEqual(after, before)

    def test_ingest_skips_secret_like_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            doc = tmp / "docs" / "secret-note.md"
            doc.parent.mkdir(parents=True)
            doc.write_text(
                "# Local Credential\n\napi_key=sensitive_value_abcdef123456\n",
                encoding="utf-8",
            )
            with wm.MemPalace(tmp / "memory.db") as palace:
                report = palace.ingest_paths([doc], workspace_root=tmp)
                stats = palace.stats()

        self.assertEqual(report["documents"], 1)
        self.assertEqual(report["memories"], [])
        self.assertEqual(report["skipped_secret_documents"], 1)
        self.assertEqual(stats["memories"], 0)

    def test_recall_evaluation_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cases_path = tmp / "recall.jsonl"
            cases_path.write_text(
                json.dumps({"query": "startup briefing", "expected": ["Use briefing at startup"]}) + "\n",
                encoding="utf-8",
            )
            with wm.MemPalace(tmp / "memory.db") as palace:
                palace.remember(
                    kind="decision",
                    summary="Use briefing at startup",
                    details="Read C4 architecture and durable memory together.",
                    tags=["memory", "briefing"],
                    source="test",
                )
                report = palace.evaluate_recall(cases_path)

        self.assertEqual(report["total"], 1)
        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["failed"], 0)

    def test_recall_evaluation_preserves_usage_and_pending_caller_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = root / "recall.jsonl"
            cases.write_text(json.dumps({"query": "startup briefing", "expected": ["Use briefing at startup"]}) + "\n")
            with wm.MemPalace(root / "memory.db") as palace:
                memory_id = palace.remember(kind="decision", summary="Use briefing at startup").id
                palace.conn.execute("UPDATE memories SET details = ? WHERE id = ?", ["Pending caller change", memory_id])
                before = [tuple(row) for row in palace.conn.execute("SELECT * FROM memories ORDER BY id")]

                report = palace.evaluate_recall(cases)

                self.assertEqual(report["passed"], 1)
                self.assertEqual([tuple(row) for row in palace.conn.execute("SELECT * FROM memories ORDER BY id")], before)
                self.assertTrue(palace.conn.in_transaction)
                palace.conn.commit()
                self.assertEqual(palace.get(memory_id).details, "Pending caller change")
                self.assertEqual(palace.get(memory_id).use_count, 0)
                self.assertIsNone(palace.get(memory_id).last_used_at)
                palace.recall("startup briefing")
                self.assertEqual(palace.get(memory_id).use_count, 1)
                self.assertIsNotNone(palace.get(memory_id).last_used_at)

    def test_entity_secret_rejection_preserves_existing_state(self) -> None:
        token = "ghp_" + "a" * 25
        inputs = (
            {"name": token},
            {"entity_type": token},
            {"aliases": [token]},
            {"metadata": {"nested": {"access_token": "sensitive-value"}}},
            {"metadata": json.dumps({"token": token})},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with wm.MemPalace(Path(tmpdir) / "memory.db") as palace:
                wm.add_entity(palace.conn, "Existing entity", aliases=["retained"])
                before = [tuple(row) for row in palace.conn.execute("SELECT * FROM entities ORDER BY id")]
                for fields in inputs:
                    with self.subTest(fields=tuple(fields)):
                        kwargs = {"name": "Existing entity", **fields}
                        with self.assertRaisesRegex(ValueError, "secret-like") as caught:
                            wm.add_entity(palace.conn, **kwargs)
                        self.assertNotIn(token, str(caught.exception))
                        self.assertEqual([tuple(row) for row in palace.conn.execute("SELECT * FROM entities ORDER BY id")], before)

    def test_fact_secret_rejection_occurs_before_entity_writes(self) -> None:
        token = "ghp_" + "a" * 25
        inputs = (
            {"subject": token}, {"predicate": token}, {"obj": token},
            {"source": token}, {"valid_from": token}, {"valid_to": token},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with wm.MemPalace(Path(tmpdir) / "memory.db") as palace:
                for fields in inputs:
                    with self.subTest(fields=tuple(fields)):
                        kwargs = {"subject": "New subject", "predicate": "uses", "obj": "New object", **fields}
                        with self.assertRaisesRegex(ValueError, "secret-like") as caught:
                            palace.add_fact(**kwargs)
                        self.assertNotIn(token, str(caught.exception))
                        self.assertEqual(palace.stats()["entities"], 0)
                        self.assertEqual(palace.stats()["relationships"], 0)

    def test_fact_invalidation_rejects_secret_before_changing_existing_fact(self) -> None:
        token = "ghp_" + "a" * 25
        with tempfile.TemporaryDirectory() as tmpdir:
            with wm.MemPalace(Path(tmpdir) / "memory.db") as palace:
                palace.add_fact("Subject", "uses", "Object", valid_from="2026-10-01")
                before = [tuple(row) for row in palace.conn.execute("SELECT * FROM relationships ORDER BY id")]
                with self.assertRaisesRegex(ValueError, "secret-like") as caught:
                    palace.invalidate_fact("Subject", "uses", "Object", ended=token)
                self.assertNotIn(token, str(caught.exception))
                self.assertEqual([tuple(row) for row in palace.conn.execute("SELECT * FROM relationships ORDER BY id")], before)
                self.assertEqual(palace.invalidate_fact("Subject", "uses", "Object", ended="2026-10-02"), 1)
                self.assertEqual(palace.timeline("Subject")[0].valid_to, "2026-10-02")


class MemPalaceSqliteStoreAndMigrationAdapterTests(unittest.TestCase):
    def test_migration_adapter_upgrades_legacy_memory_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            legacy = sqlite3.connect(db_path)
            try:
                legacy.execute(
                    """
                    CREATE TABLE memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        kind TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        details TEXT NOT NULL DEFAULT '',
                        tags TEXT NOT NULL DEFAULT '',
                        source TEXT NOT NULL DEFAULT '',
                        pinned INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                legacy.execute(
                    """
                    INSERT INTO memories(kind, summary, details, tags, source, pinned, created_at, updated_at)
                    VALUES ('decision', 'Legacy memory row', '', 'legacy', 'test', 0, '2026-04-24T00:00:00+00:00', '2026-04-24T00:00:00+00:00')
                    """
                )
                legacy.commit()
            finally:
                legacy.close()

            with wm.MemPalace(db_path) as palace:
                row = palace.get(1)
                stats = palace.stats()

        self.assertIsNotNone(row)
        self.assertEqual(row.scope, "tokimon")
        self.assertEqual(row.room, "general")
        self.assertEqual(row.use_count, 0)
        self.assertIsNone(row.last_used_at)
        self.assertEqual(stats["schema_version"], 3)

    def test_store_adapter_falls_back_when_fts_table_is_missing(self) -> None:
        class MissingFtsQueryConnection:
            def __init__(self, delegate: sqlite3.Connection) -> None:
                self.delegate = delegate

            def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
                if "memory_fts MATCH" in sql:
                    raise sqlite3.OperationalError("no such table: memory_fts")
                return self.delegate.execute(sql, parameters)

            def __getattr__(self, name: str) -> object:
                return getattr(self.delegate, name)

        with tempfile.TemporaryDirectory() as tmpdir:
            with wm.MemPalace(Path(tmpdir) / "memory.db") as palace:
                palace.remember(kind="context", summary="FTS fallback still recalls memory")
                palace.conn = MissingFtsQueryConnection(palace.conn)
                results = palace.recall("fallback recalls")

        self.assertEqual(results[0].memory.summary, "FTS fallback still recalls memory")

    def test_store_adapter_propagates_unexpected_fts_operational_error(self) -> None:
        class LockedFtsQueryConnection:
            def __init__(self, delegate: sqlite3.Connection) -> None:
                self.delegate = delegate

            def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
                if "memory_fts MATCH" in sql:
                    raise sqlite3.OperationalError("database is locked")
                return self.delegate.execute(sql, parameters)

            def __getattr__(self, name: str) -> object:
                return getattr(self.delegate, name)

        with tempfile.TemporaryDirectory() as tmpdir:
            with wm.MemPalace(Path(tmpdir) / "memory.db") as palace:
                palace.remember(kind="context", summary="Unexpected FTS failures propagate")
                palace.conn = LockedFtsQueryConnection(palace.conn)
                with self.assertRaisesRegex(sqlite3.OperationalError, "database is locked"):
                    palace.recall("failures propagate")


if __name__ == "__main__":
    unittest.main()
