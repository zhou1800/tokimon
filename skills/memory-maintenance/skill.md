---
name: memory-maintenance
description: Review and maintain Tokimon's Mem Palace; use when the user asks to review, clean up, or repair memory.
---

Review the current Mem Palace state.
Decide what to keep, delete, deweight, repair, or report.
Apply maintenance mutations through the memory-maintenance workflow CLI.

Start with `{{tokimonCommand}} -m tokimon.workflow.memory_maintenance snapshot --json`.
Treat the snapshot as the primary input.
Do not write `.tokimon/memory/memory.db` directly.
If direct inspection is truly needed, keep it read-only and use the workflow CLI help before acting.

Identify candidates: stale, duplicated, low-signal, or misleading records.
Judge each one using the full snapshot record, not the summary alone.
Use these judgment signals:
- **stale**: old or unused-looking records.
  Infer staleness from `created_at`, `updated_at`, `last_used_at`, `use_count`, and current content relevance.
- **duplicated**: exact duplicate groups from `content_hash`, plus manual semantic review for near-duplicates.
- **low-signal**: vague, generic, not actionable, low `importance`/`confidence`, or low/no usage.
- **misleading**: stale or incorrect guidance that could steer future agents wrong.
- **uncertain**: keep and report uncertainty.

Apply mutations through the same workflow entrypoint:
- `delete` for redundant or invalid records, with a reason that names the evidence.
- `update-weight` to deweight low-signal records that may still be useful.
  Use `--importance` on the `0` to `5` scale and `--confidence` on the `0` to `1` scale.
- `repair` when the snapshot or repair output shows missing hashes, FTS drift, or schema gaps.

Pinned records are read-only for this workflow; surface concerns in the report instead of mutating them.
Mutation JSON output includes the backup path at `backup.path` and the run-log path at `run_log_path`.
Use those fields for the final report.
When a durable mistake or anti-pattern is discovered during review, capture it as a `kind=avoid` memory through the Mem Palace CLI.

Use `{{tokimonCommand}} -m tokimon.workflow.memory_maintenance --help` and subcommand `--help` for command syntax.
If the CLI behavior and this skill disagree, follow the CLI for the current run and report the mismatch.

Report:
- decisions made
- mutations applied
- backup and run-log paths
- repair status
- final total and pinned counts
- exact duplicate group count
- pinned concerns
- new `avoid` memories
- unresolved items
- whether the run changed tracked files
