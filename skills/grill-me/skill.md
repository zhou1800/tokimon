---
name: grill-me
description: Interview the user relentlessly about a plan or design until reaching shared understanding, resolving each branch of the decision tree. Use when user wants to stress-test a plan, get grilled on their design, or mentions "grill me".
---

<!-- Upstream wording: Matt Pocock's grill-me skill, copyright (c) 2026 Matt Pocock, MIT License. See ../../licenses/Matt-Pocock-MIT.txt and ../../THIRD_PARTY_NOTICES. -->

Interview the user relentlessly about every aspect of the plan until there is shared understanding.
Walk down each branch of the design tree, resolving dependencies between decisions one by one.

Ask one question at a time.
For each question, provide the recommended answer.

Before each question, have a context-check subagent review it against available task context, the current conversation, and repo evidence.
Use answers already known or locally discoverable from that evidence.
Skip answered, low-value, or locally answerable questions; ask only the first unresolved high-value question requiring human judgment.
