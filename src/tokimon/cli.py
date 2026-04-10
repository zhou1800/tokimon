from __future__ import annotations

import argparse
from pathlib import Path

from tokimon.engine import (
    add_direction,
    create_state,
    feed_tokens,
    format_status,
    prepare_for_task,
    run_idle_cycle,
)
from tokimon.persistence import DEFAULT_STATE_PATH, TokimonPersistenceError, load_state, save_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokimon v2")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Path to Tokimon state JSON file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a Tokimon v2 state file.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite an existing state file.")

    feed_parser = subparsers.add_parser("feed", help="Feed Tokimon new tokens.")
    feed_parser.add_argument("--tokens", type=int, required=True, help="Number of tokens to add.")

    direct_parser = subparsers.add_parser("direct", help="Add or update a learning direction.")
    direct_parser.add_argument("--skill", required=True, help="Skill Tokimon should prioritize.")
    direct_parser.add_argument("--priority", type=int, default=5, help="Priority from 1 to 10.")
    direct_parser.add_argument("--note", default="", help="Optional reason for the direction.")

    idle_parser = subparsers.add_parser("idle", help="Run Tokimon's idle self-improvement loop.")
    idle_parser.add_argument("--max-cycles", type=int, default=None, help="Maximum tokens to spend this run.")

    task_parser = subparsers.add_parser("task", help="Prepare Tokimon for a task and produce guidance.")
    task_parser.add_argument("--summary", required=True, help="Task summary.")
    task_parser.add_argument(
        "--skill",
        action="append",
        default=[],
        help="Task-relevant skill. Repeat for multiple skills.",
    )
    task_parser.add_argument(
        "--prep-budget",
        type=int,
        default=None,
        help="Maximum tokens Tokimon may spend preparing for this task.",
    )

    subparsers.add_parser("status", help="Print Tokimon state summary.")

    return parser


def command_init(state_path: Path, force: bool) -> int:
    if state_path.exists() and not force:
        raise SystemExit(f"state file already exists: {state_path} (use --force to overwrite)")
    state = create_state()
    save_state(state, state_path)
    print(f"initialized Tokimon v2 at {state_path}")
    return 0


def command_feed(state_path: Path, tokens: int) -> int:
    state = load_state(state_path)
    state.session.current_command_context = "feed"
    available = feed_tokens(state, tokens)
    save_state(state, state_path)
    print(f"fed {tokens} token(s); available_tokens={available}")
    return 0


def command_direct(state_path: Path, skill: str, priority: int, note: str) -> int:
    state = load_state(state_path)
    state.session.current_command_context = "direct"
    direction = add_direction(state, skill_name=skill, priority=priority, note=note)
    save_state(state, state_path)
    print(f"direction set: skill={direction.skill} priority={direction.priority}")
    return 0


def command_idle(state_path: Path, max_cycles: int | None) -> int:
    state = load_state(state_path)
    state.session.current_command_context = "idle"
    effective_max_cycles = max_cycles
    if effective_max_cycles is None:
        effective_max_cycles = state.settings.default_idle_max_cycles
    records = run_idle_cycle(state, max_cycles=effective_max_cycles)
    save_state(state, state_path)
    print(f"idle cycles completed: {len(records)}")
    if records:
        last = records[-1]
        print(f"last improvement: {last.skill} -> level {last.after_level}")
    return 0


def command_task(state_path: Path, summary: str, skills: list[str], prep_budget: int | None) -> int:
    state = load_state(state_path)
    state.session.current_command_context = "task"
    effective_prep_budget = prep_budget
    if effective_prep_budget is None:
        effective_prep_budget = state.settings.default_task_prep_budget
    advice = prepare_for_task(
        state,
        summary=summary,
        requested_skills=skills,
        prep_budget=effective_prep_budget,
    )
    save_state(state, state_path)
    print(f"task: {advice.summary}")
    print(f"focus_skills: {', '.join(advice.focus_skills)}")
    print(f"auto_training_spent: {advice.auto_training_spent}")
    print(f"confidence: {advice.confidence}")
    print("approach:")
    for step in advice.approach:
        print(f"- {step}")
    if advice.gaps:
        print(f"gaps: {', '.join(advice.gaps)}")
    else:
        print("gaps: none")
    return 0


def command_status(state_path: Path) -> int:
    state = load_state(state_path)
    state.session.current_command_context = "status"
    print(format_status(state))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            return command_init(args.state_file, args.force)
        if args.command == "feed":
            return command_feed(args.state_file, args.tokens)
        if args.command == "direct":
            return command_direct(args.state_file, args.skill, args.priority, args.note)
        if args.command == "idle":
            return command_idle(args.state_file, args.max_cycles)
        if args.command == "task":
            return command_task(args.state_file, args.summary, args.skill, args.prep_budget)
        if args.command == "status":
            return command_status(args.state_file)
    except TokimonPersistenceError as exc:
        raise SystemExit(str(exc)) from exc

    raise SystemExit(f"unknown command: {args.command}")
