---
name: github-pr-review
description: Review GitHub PR links and return P1/P2/P3 findings on modified lines with constructive comments. Use when a user provides a GitHub PR URL and asks for a review.
---

# GitHub PR Review

Review the pull request read-only unless the user explicitly asks you to post or mutate anything.
Use GitHub connector tools when available; otherwise use `gh` or the public GitHub web view.

Return only review findings.
If there are none, say `No findings.`

## Contract

- Use priorities `P1`, `P2`, and `P3`, ordered by severity.
- Target only files and lines modified by the PR.
  A range must be continuous and every line in it must be modified.
- Report bugs, regressions, security issues, data-loss risks, broken contracts, and important test gaps introduced or exposed by the PR.
- Do not report style preferences, broad redesigns, duplicated existing comments, or unrelated cleanup.
- If a real issue cannot be mapped to a modified line, mention it only in a brief note after the findings.
- Trace changed contracts through their callers and consumers, including code outside the diff.

## Comment Style

- Be polite and humble.
- Use questions when they invite real discussion, but avoid judgmental questions.
- Explain the problem and why it matters with evidence from the PR.
- Stay scoped to the changed behavior; do not pile on unrelated cleanup.
- Usually avoid prescribing a fix unless the contract leaves only one reasonable option.
- Avoid sarcasm, blame, surprise, or wording that assumes negligence.

## Output

```text
1. Priority: P2
   File: path/to/file.ext
   Line: 42
   Comment: Should we account for ...?
```

Use `Line` for a single line and `Lines` for a range.
