"""Operator CLI for the memory maintenance workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from tokimon.memory.mem_palace import mem_palace as mem
from tokimon.workflow.memory_maintenance import operations


def required_tool_capabilities() -> tuple[str, ...]:
    return ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokimon memory maintenance workflow.")
    parser.add_argument("--db", default=str(mem.DEFAULT_DB_PATH), help="Mem Palace SQLite database path")
    parser.add_argument(
        "--state-dir",
        default=str(operations.DEFAULT_RUN_ROOT),
        help="Local workflow artifact root for backups and run logs",
    )
    subparsers = parser.add_subparsers(dest="command")

    snapshot_parser = subparsers.add_parser("snapshot", help="Emit a factual memory snapshot")
    snapshot_parser.add_argument("--scope", default=None)
    snapshot_parser.add_argument("--room", default=None)
    snapshot_parser.add_argument("--json", action="store_true")

    delete_parser = subparsers.add_parser("delete", help="Delete one unpinned memory after backup")
    delete_parser.add_argument("--id", type=int, required=True, dest="memory_id")
    delete_parser.add_argument("--reason", required=True)
    delete_parser.add_argument("--run-id", default=None)
    delete_parser.add_argument("--json", action="store_true")

    weight_parser = subparsers.add_parser("update-weight", help="Update one unpinned memory weight after backup")
    weight_parser.add_argument("--id", type=int, required=True, dest="memory_id")
    weight_parser.add_argument("--importance", type=float, required=True)
    weight_parser.add_argument("--confidence", type=float, required=True)
    weight_parser.add_argument("--reason", required=True)
    weight_parser.add_argument("--run-id", default=None)
    weight_parser.add_argument("--json", action="store_true")

    repair_parser = subparsers.add_parser("repair", help="Repair deterministic memory store indexes and counts")
    repair_parser.add_argument("--run-id", default=None)
    repair_parser.add_argument("--json", action="store_true")

    return parser


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_snapshot(payload: Mapping[str, Any]) -> None:
    stats = payload["stats"]
    print(f"memory_count: {stats['total_count']} ({stats['pinned_count']} pinned)")
    print(f"exact_duplicate_groups: {len(payload['exact_duplicate_groups'])}")
    print(f"database: {payload['database']}")


def _print_mutation(payload: Mapping[str, Any]) -> None:
    print(f"action: {payload['action']}")
    if "memory_id" in payload:
        print(f"memory_id: {payload['memory_id']}")
    if "reason" in payload:
        print(f"reason: {payload['reason']}")
    print(f"backup: {payload['backup']['path']}")
    print(f"run_log: {payload['run_log_path']}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    try:
        db_path = Path(args.db)
        if args.command == "snapshot":
            payload = operations.snapshot(db_path, scope=args.scope, room=args.room)
            if args.json:
                _print_json(payload)
            else:
                _print_snapshot(payload)
            return 0

        if args.command == "delete":
            payload = operations.delete_memory(
                db_path,
                memory_id=args.memory_id,
                reason=args.reason,
                run_id=args.run_id,
                run_root=args.state_dir,
            )
            if args.json:
                _print_json(payload)
            else:
                _print_mutation(payload)
            return 0

        if args.command == "update-weight":
            payload = operations.update_weight(
                db_path,
                memory_id=args.memory_id,
                importance=args.importance,
                confidence=args.confidence,
                reason=args.reason,
                run_id=args.run_id,
                run_root=args.state_dir,
            )
            if args.json:
                _print_json(payload)
            else:
                _print_mutation(payload)
            return 0

        if args.command == "repair":
            payload = operations.repair(db_path, run_id=args.run_id, run_root=args.state_dir)
            if args.json:
                _print_json(payload)
            else:
                _print_mutation(payload)
            return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
