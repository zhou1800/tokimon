---
name: memory
description: Use for explicit Mem Palace search, durable memory writes, and deeper memory operations.
---

## When to Use Memory

- Use Mem Palace for local durable memory and project continuity.
- Treat Codex memory as supplemental; Mem Palace remains authoritative.
- Before the final response on substantive work, evaluate whether future Tokimon sessions should reuse what was learned.
  If yes, add a short memory.
- Record user feedback and behavior corrections as memory by default after judging reuse value.
- Keep memories concise, scoped, and searchable, then verify writes with `recent` or `search`.

## Activation

- Keep activation to memory routing only; do not activate Tokimon tools, workflows, MCP servers, audits, or general Tokimon guidance.
- When memory work or a routed memory task needs current context, run one bounded Mem Palace briefing unless this session or task already has a current briefing.

```bash
{{tokimonCommand}} -m tokimon.memory.mem_palace.mem_palace briefing --project-root "$PWD" --include-scope global --limit 8
```

- If the Mem Palace command is unavailable, report that Tokimon memory is degraded and continue from checked-in docs with reduced confidence.
- Do not create `.venv`, install packages, or write `.tokimon/` runtime state just to satisfy memory activation.

## Search

- Use `search` for explicit prior-context or memory lookup requests.

```bash
{{tokimonCommand}} -m tokimon.memory.mem_palace.mem_palace search "<keywords>" --project-root "$PWD" --include-scope global
```

## Writes

- Use `add` only for durable local preferences, decisions, facts, or continuity notes.
  Keep summaries short and include scope, room, tags.
- Use `--project-root "$PWD"` for project-specific memory. Use `--scope global` only for genuinely cross-project memory.
- Use `--kind avoid` for durable mistakes or anti-patterns to avoid.
  Include the trigger, consequence, and preferred behavior in details.
- When a new memory corrects an older memory, keep the history and pass the older existing ids as `--metadata-json '{"supersedes_memory_ids":[<id>]}'`.
  Confirm the referenced memories first; forward or missing ids are intentionally ignored.
- Use `link`, `entity`, `timeline`, and `invalidate` when a fact is a relationship that may change over time.

Example durable memory write:

```bash
{{tokimonCommand}} -m tokimon.memory.mem_palace.mem_palace add \
  --kind decision \
  --summary "Short durable summary" \
  --details "Why this matters and when to use it" \
  --tags "tag-one,tag-two" \
  --project-root "$PWD" \
  --room "mem-palace" \
  --importance 3 \
  --confidence 1
```

Example relationship write:

```bash
{{tokimonCommand}} -m tokimon.memory.mem_palace.mem_palace link "<subject>" "<predicate>" "<object>" --valid-from YYYY-MM-DD
```

## Boundaries

- Never store secrets, tokens, passwords, cookies, credentials, or sensitive headers.
- Do not write `.tokimon/memory/memory.db` directly; use the Mem Palace CLI or Python API.
