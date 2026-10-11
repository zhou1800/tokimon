---
name: pair
description: Use only when the user explicitly asks for Pair collaboration.
---

# Pair Skill

Use this skill only when the human explicitly invokes Pair.

## Operating Procedure

Pair produces `consensus.md` or `no_consensus.md` from two independent Peer reviews.
Use `{{tokimonCommand}} -m tokimon.workflow.pair` as the operator entrypoint.
Use `--help` and `contract --json` for exact subcommands, flags, and artifact names.
The CLI records and validates artifacts; the Mediator runs the protocol.
Peer A is Codex. Peer B is Codex for `native-subagent`, or Codex/Claude for `local-cli`.
Native placement requires Codex subagent controls; local placement requires the selected provider CLIs and their authentication.

## Workflow Protocol

The Mediator does not:
- draft, merge, rank, filter, or rewrite final findings or final text
- judge which Peer is technically correct
- hide unresolved disagreement

After each round, send Peer A's note to Peer B and Peer B's note to Peer A.
Consensus requires both Peers to accept the same final text.
That final text must be authored by a Peer or explicitly supplied by the human.
Validation is structural only.

1. Initialize the run.
2. Select the configured Peers with one behavior:
   - `read_only`: shared review with no file changes.
   - `independent_outputs`: separate Peer outputs are the requested result.
   - `shared_mutable_output`: one shared artifact will be created or edited.
   - `unique_irreversible_side_effect`: one non-repeatable action will be executed.
   Pass the current task checkout with `--cwd`.
3. Use the selected placement:
   Persist notes verbatim before exchange and keep the same Peers across rounds.
   - For `native-subagent`, create each selected native Peer as a direct Codex subagent before collecting either Peer's first note.
     When native history control is available, start each Peer with no inherited Mediator conclusions and provide the required brief and context explicitly.
     Use native follow-up and wait controls for later rounds.
     If a Peer thread is lost, create a replacement from the persisted prompt and round artifacts and record the replacement in the transcript.
     If native controls or the required Peer slots are unavailable, stop with an actionable request to select `local-cli`; do not silently change placement.
   - For `local-cli`, use the configured Codex or Claude provider placement and preserve the same Pair artifacts and protocol.
     For a Codex main thread, use `CODEX_THREAD_ID` to locate its session JSONL under `$CODEX_HOME/sessions` (`~/.codex/sessions` when `CODEX_HOME` is unset).
     Read `model` and `effort` from the latest `turn_context` record and pass both to Codex CLI Peers.
     For a Claude CLI Peer, use explicit Claude model and effort options or omit both to use its CLI defaults, following `docs/guidelines/configuration-guidelines.md`.
4. Give both Peers the same brief and context.
   You may add complementary task-relevant review roles, but keep their first notes independent and do not reveal either Peer's work to the other.
5. Continue until both Peers accept the same final text or identify a no-consensus issue.
   Default to five rounds; pause for human direction when the run reaches its configured limit without consensus.
   If a Peer needs external clarification, ask the user, record the answer in the run, broadcast it to both Peers, then continue.
6. For mutable work, one Peer writes and the other reviews.
   Do not allow simultaneous Peer writes to the shared workspace.
   Return unresolved objections to Peer discussion.
7. Do not execute an irreversible action until both Peers converge on the exact action.
8. Write the terminal Pair artifact, update run state, and validate the run structure.
   Record the terminal evidence required by `contract --json`, including the independent Peer exchange and, for mutable work, distinct writer/reviewer evidence.
9. Deliver the run id, selected Peers, rounds completed, result, artifact paths, and verification performed.
   No-consensus delivery includes the unresolved issue, both Peer positions and evidence, the human decision needed, and continuation options.
