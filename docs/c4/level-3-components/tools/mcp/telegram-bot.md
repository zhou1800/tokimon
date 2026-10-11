---
title: "Telegram Bot MCP - Component"
summary: "L3 component contract for send-only Telegram MCP operations, secrets, and sanitized logs."
read_when:
  - Changing Telegram MCP tools, secret loading, send behavior, logging, error normalization, or AI Harness Interface route-back behavior.
  - Reviewing the Telegram MCP command and behavior contract.
---

# Telegram Bot MCP - Component

## Contents

- [Overview](#overview)
- [Components](#components)
- [Interfaces/Contracts](#interfacescontracts)
- [Data Ownership](#data-ownership)
- [Failure Modes](#failure-modes)
- [Acceptance Checks](#acceptance-checks)
- [References](#references)

## Overview

Telegram Bot MCP owns Tokimon's send-only Telegram MCP server.
It exposes health, chat diagnostics, message send, document send, and log-tail operations through stdio MCP while keeping secrets and sanitized runtime evidence outside the repository source tree.

## Components

- **MCP server:** Owns newline-delimited stdio JSON-RPC handling, supported MCP methods, exposed tool names, and MCP text-content response shape.
- **Secrets loader:** Owns local config and `TELEGRAM_BOT_TOKEN` resolution without printing or persisting tokens.
- **Destination resolver:** Owns explicit chat ids, named `allowed_chats`, and `default_chat_id` selection.
- **Telegram API client:** Owns `getMe`, `getChat`, `sendMessage`, and `sendDocument` requests.
- **Send-log writer:** Owns sanitized metadata logs without tokens, message bodies, caption bodies, or document bodies.
- **Health CLI:** Owns server startup diagnostics for operators and AI Harness Interface route-back checks.
- **Error normalizer:** Owns missing config, Telegram API failure, rate-limit, invalid-destination, and MCP tool-error diagnostics.

## Interfaces/Contracts

Canonical machine-readable tool contracts are the MCP `tools/list` response produced by `.venv/bin/python -m tokimon.tools.mcp.telegram_bot serve` and the health payload from `.venv/bin/python -m tokimon.tools.mcp.telegram_bot health --json`.

### Command Surface

```bash
.venv/bin/python -m tokimon.tools.mcp.telegram_bot serve
.venv/bin/python -m tokimon.tools.mcp.telegram_bot health --json
.venv/bin/python -m tokimon.tools.mcp.telegram_bot health --live --json
```

The `serve` command reads newline-delimited JSON-RPC requests from stdin and writes newline-delimited JSON-RPC responses to stdout.
The `health --json` command is non-network by default and must not require a token.
Use `--live` to run a live Telegram `getMe` check when a token is configured.

### MCP Protocol Methods

The server supports the minimum MCP stdio methods needed by Codex-compatible clients:

- `initialize`
- `notifications/initialized`
- `ping`
- `tools/list`
- `tools/call`

`initialize` returns the service's declared protocol version, including for unsupported, missing, or malformed client proposals.
Missing or non-string methods return `-32600`.
Messages with a string method and no `id` are notifications: they produce no reply and cannot invoke request-only methods or tools.
Explicit request ids, including `0`, an empty string, and `null`, keep their response correlation.
Wire responses escape unpaired Unicode surrogates as JSON escapes and preserve ordinary Unicode text.
Tool text and error messages also escape malformed characters before wrapping, so decoded display text is valid UTF-8.
Strict UTF-8 output remains valid for ids, errors, and tool content, and the next request still receives its response.
Malformed JSON, interpreter integer limits, and decoder recursion errors return `-32700`, log a correlated parse failure, and leave stdio ready for the next line.

Tool names exposed through `tools/list`:

- `telegram_bot_health`
- `telegram_bot_get_me`
- `telegram_bot_get_chat`
- `telegram_bot_send_message`
- `telegram_bot_send_document`
- `telegram_bot_log_tail`

Tool responses are returned as MCP text content blocks containing formatted JSON.
Tool failures are returned as MCP tool errors with sanitized JSON payloads.

### Tool Input Contracts

#### `telegram_bot_health`

Optional:

- `live`: boolean; when true, call Telegram `getMe` if a token is configured.

#### `telegram_bot_get_me`

No arguments.
Requires a configured token.

#### `telegram_bot_get_chat`

Optional:

- `chat`: configured chat name.
- `chat_id`: explicit chat id.

Exactly one target source may be supplied.
When both are omitted, the configured default chat is used.

#### `telegram_bot_send_message`

Required:

- `text`: non-empty plain text message.

Optional:

- `chat`: configured chat name.
- `chat_id`: explicit chat id.
- `correlation_id`: workflow, run, task, or request identifier recorded in output and logs.
- `dry_run`: boolean; when true, do not call Telegram and do not write a send log.

#### `telegram_bot_send_document`

Required:

- `file`: Tokimon repository-owned, Tokimon runtime-owned, or configured adopter-owned document path.

Optional:

- `caption`: plain text caption.
- `chat`: configured chat name.
- `chat_id`: explicit chat id.
- `correlation_id`: workflow, run, task, or request identifier recorded in output and logs.
- `dry_run`: boolean; when true, do not call Telegram and do not write a send log.

#### `telegram_bot_log_tail`

Optional:

- `limit`: positive integer, default `20`.

### Document Root Boundary

`telegram_bot_send_document` accepts files under Tokimon source, Tokimon runtime, or configured adopter document roots.
Configured adopter roots come from `TOKIMON_TELEGRAM_ALLOWED_DOCUMENT_ROOTS`, an `os.pathsep`-separated list of absolute existing directories.
Blank entries are ignored.
Relative or missing roots are rejected.
Each document is resolved and checked against these roots, then read through pinned directory and file descriptors that reject symbolic links introduced during opening.
Only regular files are accepted; this boundary requires POSIX no-follow filesystem operations.
The multipart upload, byte count, and SHA-256 log use the same immutable content snapshot.
Multipart filenames escape quotes and line breaks so file metadata cannot add headers.
Replacing a file or ancestor after approval cannot redirect a send to another path.

### Secrets File Contract

The file is user-managed and ignored by git:

```json
{
  "telegram": {
    "bot_token": "<telegram-bot-token>",
    "default_chat_id": "<telegram-chat-id>",
    "allowed_chats": {
      "ops": "<telegram-chat-id>"
    }
  }
}
```

`TELEGRAM_BOT_TOKEN` overrides the file token.
The server warns when `.tokimon/secrets.local.json` is not `0600` on POSIX hosts.
The server never writes this file.

## Data Ownership

- Consumes read-only bot token and optional chat ids from `.tokimon/secrets.local.json` or `TELEGRAM_BOT_TOKEN`.
- Consumes read-only configured adopter document roots from `TOKIMON_TELEGRAM_ALLOWED_DOCUMENT_ROOTS`.
- Owns sanitized send logs at `.tokimon/tools/mcp/telegram-bot/sends.jsonl`.
- Derives destination diagnostics, text/document hashes, dry-run payloads, MCP response payloads, and sanitized success/failure log entries.
- Does not store tokens, cookies, full message bodies, caption bodies, document bodies, Telegram updates, chat history, or MCP credentials.

## Failure Modes

Telegram recovery fixes the local secret, destination, permission, payload, or network condition before retrying; send logs preserve sanitized evidence without storing message bodies or tokens.

- Missing or invalid local secrets JSON.
- Bot token missing, invalid, or revoked.
- Secrets file mode is broader than `0600`.
- No destination selected and no `default_chat_id` configured.
- Named chat missing from `allowed_chats`.
- Bot is not present in the target group or lacks permission to send.
- Chat id is wrong or forbidden.
- Telegram API returns 429 with `retry_after`.
- Network timeout or non-JSON Telegram response.
- Message text is empty or longer than Telegram's 4096-character limit.
- Document path is missing, outside the allowed document-root boundary, or is not a file.
- Configured adopter document root is relative, missing, or not a directory.
- Document caption exceeds Telegram's 1024-character caption limit.
- MCP requests are malformed or call unknown tools.

## Acceptance Checks

1. Verify Telegram MCP behavior:
   ```bash
   .venv/bin/python -m unittest discover -s tests/tools/mcp/telegram_bot -t . -p 'test_*.py'
   ```
   Expected outcome: tests pass for secrets loading, destination resolution, request handling, sanitized logs, MCP protocol behavior, health diagnostics, and parser coverage.
   Document tests cover file and ancestor replacement, symlink escapes, regular-file checks, matching upload/log bytes, and dry runs without network or log writes.
   Shared protocol tests in `tests/tools/mcp/test_protocol.py` cover version negotiation, silent notifications, explicit request ids, and stdio parser recovery.
   Wire tests in `tests/tools/mcp/test_wire_encoding.py` use strict UTF-8 byte streams for surrogate ids, errors, and tool content.

2. Verify registry contract:
   ```bash
   .venv/bin/python -m unittest discover -s tests/tools/registry -t . -p 'test_registry.py'
   .venv/bin/python -m tokimon.tools.registry show mcp.telegram-bot --json
   ```
   Expected outcome: the registry reports `mcp.telegram-bot`; health does not require a live Telegram API call.

3. Verify AI Harness Interface MCP route-back contract:
   ```bash
   .venv/bin/python -m unittest discover -s tests/ai_harness_interface -t . -p 'test_*.py'
   .venv/bin/python -m tokimon.ai_harness_interface validate --json
   ```
   Expected outcome: generated MCP entries include `tokimon-telegram-bot` routed to `tokimon.tools.mcp.telegram_bot`.

4. Ask an AI coding agent to verify the component contract:
   ```text
   Review docs/c4/level-3-components/tools/mcp/telegram-bot.md, tokimon/tools/mcp/telegram_bot/, tests/tools/mcp/telegram_bot/, tests/tools/registry/, and tests/ai_harness_interface/.
   Confirm that the documented MCP tools, command surface, secret boundary, sanitized log contract, failure modes, registry behavior, AI Harness Interface route-back behavior, and tests match the implementation.
   Report any drift with file and line evidence.
   ```
   Expected outcome: the agent reports no contract drift, or lists precise mismatches with recommended owner documents or implementation files.

## References

- L1 Context: `docs/c4/level-1-context/tools-context.md`
- L2 Container: `docs/c4/level-2-containers/tools/mcp/telegram-bot.md`
- Tool Registry: `docs/c4/level-2-containers/tools/tool-registry.md`
- AI Harness Interface Contract: `docs/c4/level-3-components/ai-harness-interface/ai-harness-interface.md`
