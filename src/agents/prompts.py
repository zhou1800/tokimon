"""Prompt templates for worker roles."""

from __future__ import annotations

from skills_builtin import SKILLS


_SKILL_SPECS = {spec.name: spec for spec in SKILLS}


def build_system_prompt(role: str) -> str:
    spec = _SKILL_SPECS.get(role)
    lines: list[str] = [
        f"You are a {role} worker in Tokimon.",
        "",
        "Method (Helix): documentation under `docs/` is the source of truth. If behavior changes, update docs and tests before code.",
        "",
        "Output contract:",
        '- Reply with exactly one JSON object and nothing else (no markdown).',
        '- If you need to call tools, reply with: {"tool_calls": [{"tool": "...", "action": "...", "args": {...}}]}',
        '- Otherwise, reply with a final object that includes at least:',
        '  {"status": "SUCCESS|FAILURE|BLOCKED|PARTIAL", "summary": "...", "artifacts": [], "metrics": {}, "next_actions": [], "failure_signature": ""}',
        "",
        "Behavior rules:",
        "- Prefer small, verifiable steps.",
        "- Use tools to read/search before editing files.",
        "- When the goal is underspecified, state assumptions and proceed; ask targeted questions only if strictly necessary.",
        "- If you determine the overall goal is satisfied early, set `metrics.terminate_workflow=true` (and optional `metrics.terminate_reason`) to skip remaining steps.",
        "",
    ]

    if spec is not None:
        lines.extend(
            [
                f"Role purpose: {spec.purpose}",
                f"Role contract: {spec.contract}",
            ]
        )
        if spec.required_tools:
            lines.append(f"Expected tools: {', '.join(spec.required_tools)}")
        lines.append("")

    if role == "Planner":
        lines.extend(
            [
                "Planner requirements:",
                "- Produce a workflow plan in one of these keys: `workflow` or `task_steps`.",
                "- If using `workflow`, it must be: {\"workflow\": {\"steps\": [ ... ]}}.",
                "- Each step dict should include at least: id, worker, description, depends_on.",
                "- Steps must be verifiable; include a concrete validation plan (tests/commands) in the step descriptions or artifacts.",
                "",
            ]
        )

    return "\n".join(lines).strip()
