---
title: "Language and Communication Guidelines"
summary: "Repo-wide rules for repository language, user communication language, and conservative user-input interpretation."
read_when:
  - Creating, editing, reviewing, or auditing repository-authored written material.
  - Deciding whether non-English source material should be preserved.
  - Responding to human-authored requests that may need language choice or input clarification.
---

# Language and Communication Guidelines

This file owns repository-wide language, communication-language, and user-input interpretation rules.

## Repository Language

Use English for repository-authored control-plane content.
Representative examples include code identifiers, comments, documentation, command contracts, schemas, commit messages, issues, pull requests, and agent-facing instructions.

Preserve the original language for data-plane content when language is part of the source material, expected output, or behavior under test.
Representative examples include raw inputs, quoted text, data fields from user-facing systems, fixtures, expected outputs, localization and translation artifacts, and non-English test cases.

Surrounding explanations, contracts, metadata, and verification notes should still be English unless the workflow explicitly requires another language for the user-facing artifact.

## Communication Language

The user may communicate in Chinese or English.
Respond in the language that best supports clarity and reasoning.

For longer explanatory responses, prefer Chinese because the user reads it faster.
Technical terms may remain in English when that is clearer or more natural.

## User Input Interpretation

Human-authored requests may be imperfectly phrased.
Before choosing tools or making changes, interpret the request conservatively so the user's original intention is preserved.

Use local context to resolve low-risk wording issues when the intended meaning is clear.
Treat explicit references as exact unless context makes a low-risk correction clear.

State the assumption when it matters for the work or result.
Ask for clarification when ambiguity remains after reasonable context checks and could materially change the intended result.
