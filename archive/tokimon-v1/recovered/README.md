# Tokimon-v1 Recovered Prompt Dumps

Generated from the archived Tokimon-v1 runtime data under `archive/tokimon-v1/` and git history for `docs/todo.md`.

## Counts
- Unique self-improve top-level goals: 238
- Self-improve runs with preserved top-level goals: 335
- Preserved `session.json` files: 524
- Preserved internal session attempt prompts: 758
- Unique TODO document states: 26
- Unique TODO items ever seen: 24

## Files
- `manifest.json`: counts and file map.
- `unique_self_improve_goals.json`: the full 238 unique top-level self-improve goal prompts, with run IDs and occurrence counts.
- `internal_session_prompts.jsonl`: the full 758 internal session attempt prompts, one JSON record per attempt.
- `todo_history.json`: reconstructed TODO timeline with full document content for each unique state.
- `todo_items_ever_seen.json`: every unique TODO item text observed across history, with first/last seen dates and open/done states.

## Notes
- The archived durable memory DB is present, but empty in this snapshot; these dumps therefore come from run artifacts and git history, not from a populated long-term lesson store.
- `internal_session_prompts.jsonl` preserves raw multiline prompts in JSON string form.
