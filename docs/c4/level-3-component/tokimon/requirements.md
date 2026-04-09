# Tokimon v2 Requirements

## Product Rules

### R1. Idle Self-Improvement

When Tokimon has available tokens and is placed into an idle improvement cycle, it must spend tokens on self-improvement until the cycle budget or token budget is exhausted.

### R2. Directed Learning Priority

When the user provides a direction for a skill, Tokimon must prioritize that skill ahead of non-directed learning during future improvement cycles.

### R3. Task-Time Preparation

When the user asks Tokimon to help with a task and relevant tokens are available, Tokimon must spend a bounded preparation budget on task-relevant skills before returning its task guidance.

### R4. Persistence

Tokimon state must survive across CLI runs through a state file on disk.

### R5. Benchmark Honesty

Tokimon may expose local heuristic signals such as `quality_score`, but claims of being better than baseline AI systems must be justified by explicit benchmarks rather than by the heuristic alone.
