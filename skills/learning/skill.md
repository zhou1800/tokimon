---
name: learning
description: Use when Tokimon should learn from a concrete failure or explicit learning request and turn approved evidence into a product-development intake package for an activated Skill, MCP, workflow, or existing-surface update.
---

# Learning

Use this skill when the current task asks Tokimon to learn, or when a concrete failure or review finding should become visible product behavior.
Learning is agent-driven and CLI-backed: the agent performs judgment, source inspection, scope selection, and product-development activation; the CLI validates and records deterministic workflow artifacts.
Learning has two workflows:
- **Reactive Learning:** starts from a real correction, tool failure, test failure, code review finding, or document review finding.
- **Active Learning:** starts from an explicit operator request to learn a target capability from selected sources.

## Route

1. Classify the trigger as Reactive Learning or Active Learning.
2. Inspect the evidence or selected sources, then create a candidate packet with exact evidence, pressure scenario, scoped guidance, negative scope, claim boundary, and acceptance expectations.
3. Validate the candidate through `{{tokimonCommand}} -m tokimon.workflow.learning`.
4. Record independent review and operator intake approval before creating an intake package.
5. Create a runtime product-development intake package for a Skill, MCP, workflow, or existing-surface update.
6. Send the intake package through normal product-development before activation.
7. Record activation evidence only after the resulting Tokimon capability is active.

## Guardrails

- Candidate packets are reviewable evidence, not canonical guidance or active product behavior.
- Intake packages are local runtime handoffs, not checked-in product behavior.
- GitHub score, broad repo browsing, startup memory observations, and open-ended repository mining are not accepted learning evidence.
- Do not fabricate Reactive Learning if no real failure occurred.
- Do not turn learning evidence directly into a Skill or MCP.
- Mem Palace may point to the activated capability or local runtime evidence, but it is not canonical.
- Do not startup-load Learning runtime artifacts.

## Activation Output

A Learning run succeeds only when the result becomes visible in Tokimon capability.
Acceptable activation targets are a new or updated Skill, MCP server, workflow, or existing product surface with matching C4/code/tests when those are required.

Command shape:

```bash
{{tokimonCommand}} -m tokimon.workflow.learning create-intake-package --packet <candidate.json> --proposed-surface skill_and_mcp --json
```
