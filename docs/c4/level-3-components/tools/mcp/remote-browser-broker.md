---
title: "Remote Browser Broker MCP - Component"
summary: "L3 component contract for isolated browser sessions, page operations, human takeover and profile templates."
read_when:
  - Changing Remote Browser Broker MCP tools, transport, Docker lifecycle, CDP helpers, pause behavior, or template behavior.
  - Reviewing Remote Browser Broker behavior, safety, and recovery contracts.
---

# Remote Browser Broker MCP - Component

## Contents

- [Overview](#overview)
- [Components](#components)
- [Interfaces/Contracts](#interfacescontracts)
- [Data Ownership](#data-ownership)
- [Failure Modes](#failure-modes)
- [Acceptance Checks](#acceptance-checks)
- [References](#references)

## Overview

Remote Browser Broker MCP is Tokimon's isolated visible-browser runtime for agents and human handoff.
At L3, this document owns the MCP transport contract, browser-session lifecycle, Docker runtime boundary, page-operation contract, pause behavior, profile templates, runtime state ownership, failure behavior, and acceptance checks for `mcp.remote-browser-broker`.
Its primary outcome is one independently controlled browser runtime per MCP session, with explicit human takeover and redacted diagnostics.

## Components

- **MCP transports:** Own stdio and `/mcp` HTTP transport, MCP session id handling, JSON-RPC method support, and tool-call response shape.
- **Browser lifecycle manager:** Owns one browser runtime per MCP session, session reuse, recreation, stop/delete behavior, retained-session metadata, and startup reconciliation.
- **Docker runtime manager:** Owns container creation, profile mounting, runtime hardening, network policy, and cleanup for browser sessions and templates.
- **Page operation adapter:** Owns structured page actions, snapshots, screenshots, refs, waits, active-tab behavior, and stale-ref handling.
- **Human takeover controller:** Owns pause/resume, noVNC connection credentials, and token lifecycle.
- **Template manager:** Owns profile template create/edit/seal/clone/refresh/delete behavior and sealed-template immutability.
- **Diagnostics and redaction layer:** Owns safe health/status output, technical logs, and sensitive-data exclusion.

## Interfaces/Contracts

Canonical machine-readable tool contracts are the MCP `tools/list` response produced by `.venv/bin/python -m tokimon.tools.mcp.remote_browser_broker serve` and the health payload from `.venv/bin/python -m tokimon.tools.mcp.remote_browser_broker health --json`.

The local MCP client acts for the user and executes browser and management operations directly.
The calling agent owns task authorization and platform-required confirmations.
HTTP access is loopback-only and rejects browser-origin requests and non-loopback Host headers.

### Command Surface

```bash
.venv/bin/python -m tokimon.tools.mcp.remote_browser_broker serve
.venv/bin/python -m tokimon.tools.mcp.remote_browser_broker serve-http --host 127.0.0.1
.venv/bin/python -m tokimon.tools.mcp.remote_browser_broker health --json
.venv/bin/python -m tokimon.tools.mcp.remote_browser_broker build-image
```

`serve` starts a stdio MCP endpoint for target-platform managed MCP startup.
Stdout is reserved for line-delimited JSON-RPC responses; runtime logs go to stderr.
The stdio process uses one MCP session key for that client process.
`serve-http` starts a Streamable-HTTP-style MCP endpoint at `/mcp`.
When `--port` is omitted, the port is derived from the installation id.
Use `health --json` to read the exact `broker_url`.
JSON-RPC requests are sent with `POST /mcp`; responses are sent as Server-Sent Events with the `mcp-session-id` response header.
Clients that require a shared event stream can open `GET /mcp`, which returns a long-lived Server-Sent Events stream for the MCP session.
`DELETE /mcp` stops the caller's browser and retains its profile.
HTTP session termination, graceful stdio EOF, and broker-process shutdown preserve ordinary profiles and template drafts.
Graceful shutdown closes new ownership/browser admission, waits for admitted work, then stops or checkpoints owned runtimes before releasing ownership.
It checkpoints an editing template as a stopped retained draft with recovery metadata so transport teardown does not discard human editing work.


### MCP JSON-RPC Contract

Supported JSON-RPC methods:

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
Strict UTF-8 stdio and HTTP replies remain valid for ids, errors, and tool content, and later requests still receive their responses.
Malformed JSON, invalid HTTP UTF-8, interpreter integer limits, and decoder recursion errors return `-32700` and log a correlated parse failure.
Stdio continues with the next line and retains its normal EOF shutdown; HTTP notifications receive an empty `202` response with the MCP session header.

The HTTP handler returns or preserves `mcp-session-id`.
The broker maps that header to the browser session for all normal tool calls.
New MCP sessions are isolated unless the caller explicitly recovers its retained browser using the recovery token returned to the original session.

Tool responses are MCP text content blocks containing formatted JSON.
Broker tool failures are returned as MCP tool errors with `isError=true` and stable machine-readable error codes where applicable.

### MCP Tool Catalog

Session tools:

- `browser_whoami`
- `browser_ensure`
- `browser_current_session_status`
- `browser_connect_info`
- `browser_navigate`
- `browser_snapshot`
- `browser_artifact`
- `browser_evaluate`
- `browser_screenshot`
- `browser_click`
- `browser_type`
- `browser_upload_file`
- `browser_backend_request`
- `browser_press_key`
- `browser_wait`
- `browser_new_tab`
- `browser_switch_tab`
- `browser_close_tab`
- `browser_pause_for_human`
- `browser_resume`
- `browser_stop`
- `browser_session_extend_retention`
- `browser_session_delete_retained`
- `browser_session_recover`
- `browser_session_cleanup_artifacts`
- `browser_diagnostics`

Management tools:

- `browser_session_list`
- `browser_session_cleanup_expired`
- `browser_session_cleanup_orphaned`

Template tools:

- `browser_template_create`
- `browser_template_seal`
- `browser_template_list`
- `browser_template_refresh`
- `browser_template_delete`


Normal session tools operate on the caller's current MCP session; recovery may rebind only token-proven retained ownership.
`browser_session_list` returns redacted broker-wide inventory.
The broker has no raw-CDP disclosure tool.

### State Model

The core invariant is `one MCP session -> one Docker container -> one browser`.
The browser session id is debug metadata and never grants recovery authority.
Normal agent calls use their MCP session; recovery additionally requires the original session's private recovery token.
The design must not move the coordination problem from owner ids to browser session ids.
The client integration is responsible for giving each Agent one independent MCP session.
Template source does not affect this runtime ownership chain.

Session states:

- `starting`: container startup/readiness is in progress and page/connect operations are not yet admitted.
- `running`: broker owns the container and can operate it.
- `paused_observed`: running but automatic page actions are blocked.
- `stopped_retained`: container stopped and runtime profile retained.
- `expired`: retention deadline passed.
- `orphaned`: container or profile exists but ownership cannot be safely proven.
- `peer_owned`: a contained runtime root is currently owned by another broker process and must not be cleaned by this process.
- `unknown`: broker cannot inspect enough to classify.

State records include the MCP session key, internal browser session id, container id/name, profile path, active tab id, retention expiry, template source, pacing mode, token exposure mode, and technical reachability.
Status outputs expose only fields appropriate to the caller scope.
Diagnostics are redacted by default and must not leak page content or bearer connection material.

### Session Lifecycle

`browser_ensure` starts or reuses this MCP session's browser, restores its retained profile even after its retention deadline, or clones a sealed template.
`recreate=true` replaces this session's runtime and profile.
`start_url` navigates the resulting browser immediately when page actions are allowed.
HTTP(S) URLs require a valid host and port; invalid input fails before recreation removes the existing runtime or profile.
The broker reserves ownership and capacity before startup and publishes `running` only after readiness succeeds, then returns status and a private recovery token.
The caller requests human-view information separately.
A template with a recorded runtime-image reference must match the configured image.

Lifecycle operations for one session are serialized, and shutdown drains admitted work before ownership changes hands.
The active session remains globally reserved until its container has stopped, so neither the same MCP session nor another session can reuse its capacity or ports early.
The default per-process capacity is eight active sessions and 32 active-plus-retained sessions, configurable through `TOKIMON_RB_MAX_ACTIVE_SESSIONS` and `TOKIMON_RB_MAX_TOTAL_SESSIONS`.
Redacted `peer_owned` inventory does not consume this process's capacity; exclusive ownership still prevents it from adopting or mutating peer sessions.
Capacity pressure never deletes profiles, including expired and abandoned records.
For active capacity it stops the least recently used eligible ordinary runtime and retains its profile for same-session restoration.
An idle runtime is eligible after 30 minutes without session activity, controlled by `TOKIMON_RB_IDLE_TIMEOUT_SECONDS`; `0` disables idle stopping.
Idle-timeout overrides should leave room for the caller's polling interval and delayed wakeups.
Starting, busy, paused, template-editing, and unexpired human-view sessions are protected; retained profiles and template drafts are never evicted for capacity.
Each admission makes one bounded pass, skips busy candidates without waiting, and stops reclaiming as soon as the requested capacity is available.
Cleanup failures retain ownership for retry and are logged with redacted technical context; a failed candidate does not prevent checking other eligible candidates.
If capacity remains unavailable, `SESSION_CAPACITY_EXHAUSTED` includes capacity counts and idle-stop results; freeing total capacity requires explicit profile deletion.
Health and diagnostics report the automatic-cleanup policy and diagnostics include cumulative cleanup counts.
Expiry marks a retained profile as eligible for explicit cleanup; it never deletes or prevents recovery of that profile.
Cross-broker ownership prevents different processes from claiming the same MCP key, runtime root, template mutation, or host port.
Startup remains `starting` until readiness succeeds; a stopped retained profile restarts as a new session generation so stale calls cannot execute after restoration.
One MCP session key may appear in the active-session index or the retained-session indexes, never both; template create/refresh refuses a key with retained ownership until that retained session is resumed or explicitly deleted.
If container stop succeeds but retained metadata persistence fails, the stopped generation remains active-indexed and rejects connect, pause, and resume until a later stop/ensure recovery call persists and transfers its ownership safely.
Startup failure tears down the launched container and its dedicated network before releasing the session mapping and its port ownership; cleanup failure stays visible instead of admitting a colliding replacement.
Before Docker start, the broker checkpoints `starting` ownership with the real MCP session key and concrete runtime identity; after readiness it checkpoints `running` before admitting page work.
If the owner process disappears while that checkpoint is active, reconciliation does not promote it into the retained indexes or make it resumable.
The new broker indexes the contained root as `unknown` orphan ownership (or `peer_owned` while another process still owns it) and rejects duplicate admission for that key.
Recovery of a remaining container requires explicit orphan cleanup, which must verify and stop the exact stable Docker container identity before deleting the profile; ordinary `browser_ensure` cannot adopt or resume the abandoned generation.

Retained state is reusable only when its checkpoint is explicitly `stopped_retained` or `expired` with a valid retention expiry.
Before retained deletion, expired cleanup, or retained restart touches or reuses its mounted profile, the Docker manager resolves the expected container name, verifies the full stable identity, and stops that exact container by id if it still exists.
An absent container is safe to continue, but ambiguous or foreign identity leaves the retained root owned and fails closed.
Before orphan cleanup mutates a contained root, the broker reacquires that root's ownership and any associated template-editor ownership.
Peer-owned records remain visible and untouched until the peer releases ownership; a later cleanup pass removes stale inventory or reclassifies the root.

A new MCP session never claims another MCP session's running or idle browser runtime because they requested the same template.
`browser_ensure`, `browser_template_create`, and `browser_template_refresh` return a private recovery token for the current task to keep in its transient context.
After transport loss, `browser_session_recover(recovery_token)` associates that retained profile or draft with a fresh MCP session key without starting a browser.
The destination must have no other browser; active, unknown, duplicate, or peer-owned checkpoints cannot be claimed.
Recovery verifies the token and exclusive ownership, checkpoints the new binding before releasing the old key, and preserves the previous binding if checkpointing fails.
Only a token hash is persisted; tokens never appear in status, inventory, logs, or diagnostics and must not be copied into repository files or durable memory.
Successful ensure/create/refresh/recovery issues a replacement token, invalidating the previous token so only one reconnect can claim it.
A failed restart preserves the latest token and profile for retry.
Older checkpoints without a token remain recoverable through their original MCP key, but cannot be claimed from a new key.
If the MCP session is still usable but its Docker container or browser is not, `recreate=true` replaces only that MCP session's runtime.

Transport shutdown checkpoints retained ownership after admitted work finishes.
Use `browser_stop` when the current session's container should stop but its runtime profile should remain available for same-session reuse.

`browser_stop` behavior:

1. Requests `Browser.close` over the owned runtime's CDP connection so Chrome can flush pending profile writes, then waits for container exit and removes it.
2. Revokes noVNC / Browser Use observer tokens.
3. Retains the runtime profile and records its eligibility deadline for explicit cleanup.
4. Moves the session to `stopped_retained`.

If CDP is unavailable or Chrome does not exit, the Docker shutdown fallback still retains the profile but cannot guarantee the last unflushed writes.
Template sealing uses the same shutdown path before copying the profile.

Immediate delete and retention extension are explicit tool actions with max-retention enforcement.
Expired cleanup must never delete running sessions.

`browser_session_cleanup_artifacts` clears only the current ordinary retained session's staged uploads, screenshots and saved artifacts, returning removed file and byte counts.
It requires a stopped session, verifies that its exact container is absent or stopped, and serializes with browser work and lifecycle changes.
All artifact trees must pass containment and regular-file validation before deletion; links and special files are refused.
Login profiles, browser-managed downloads, templates, and recovery metadata remain intact; `browser_ensure` restarts the same profile afterward.

### Connect Info and noVNC

`browser_connect_info` returns noVNC human-view information and optional SSH tunnel guidance, including during human pause.
It issues a fresh credential when needed; raw CDP endpoints remain internal.
Ensure, template-create, refresh, pause, status, and diagnostics responses contain no viewer credentials.

noVNC URLs are bearer-secret URLs.
New viewer credentials expire after 24 hours by default; `TOKIMON_RB_NOVNC_TOKEN_TTL_SECONDS` overrides this lifetime in seconds when the broker starts.
The returned `novnc_token_expires_at` is authoritative for each issued link; changing the default does not extend credentials already issued by a running broker.
The broker issues random per-session tokens, binds noVNC exposure to loopback by default, revokes tokens on stop, template seal, cleanup, retained-profile deletion, and token expiry, and keeps tokens out of logs.
Token issuance and revocation are coordinated with page observation and lifecycle changes.
Revocation marks the credential inactive before stopping the proxy, retains ownership when proxy shutdown fails so cleanup can be retried, and forcibly closes established WebSocket relays rather than waiting for token expiry.

Raw CDP is browser control and credential access.
The broker uses CDP internally and does not return the endpoint through MCP tools.
Its Docker-published CDP port is bound to host loopback, which excludes remote network callers but not principals with the same host/OS authority; those principals are trusted by this design.
Internal browser-control traffic has bounded connection, command, message-size, and unmatched-message limits so an untrusted page cannot cause unbounded buffering or waiting.

### Human Pause Contract

`browser_pause_for_human` accepts only an optional `reason` and blocks automatic page actions until resume.
Unsupported arguments are rejected.
Page observation, technical status, and existing noVNC links remain available.
Pause is idempotent and waits for admitted page work before changing state.

The agent calls `browser_resume` after the user indicates that it should continue.

### Browser Operation Path

Page operations use broker-owned structured browser control against the same headed browser that a human can view through noVNC.
The adapter implementation may evolve without changing the MCP operation contract.
Agents should not drive page actions through the noVNC canvas when a structured operation exists.

This layer is about tabs and page operations inside the one browser owned by the current MCP session.
Agents open tabs as needed for the task.
Site-specific state belongs to the workflow, operations, or guidelines document that owns that site, not to broker session ownership.

`browser_snapshot` returns URL-safe technical state, title when allowed, visible text when allowed, and structured element refs.
`max_text_chars` is bounded to `1..100000` and `max_elements` to `1..1000`; the defaults are 20,000 characters and 200 elements.
Snapshot results include only documented fields and bound both individual values and aggregate text/element output.
Refs are stable only within their declared lifetime.
Navigation, tab switch, DOM refresh, timeout, or page reload invalidates refs.
Ref actions verify the captured document, element identity, position, and snapshot description immediately before use; changed targets fail with `STALE_REF`.
`browser_evaluate` evaluates a caller-provided JavaScript expression and returns its value so workflow adapters can perform page-specific public data extraction.
Its optional `await_promise` input defaults to `true` and controls whether promise results are awaited.
Caller-provided JavaScript follows the same task authorization and pause rules as other page operations; it is not inherently read-only.
It must not be used to capture or persist credentials, cookies, bearer tokens, passwords, or other secrets.
Workflow adapters are responsible for keeping returned values inside their documented artifact and auth boundaries.
`browser_screenshot` captures the current viewport, writes a PNG or JPEG artifact under `.tokimon/tools/mcp/remote-browser/sessions/<session-id>/screenshots/`, and returns only the artifact path plus format, capture type, and byte size.
It accepts only optional `format` values `png` or `jpeg`; JPEG uses broker-owned quality `80`.
It rejects `full_page` because bounded full-page capture is deferred to a separate design.
It must not return image bytes, base64 data, page URL, page title, text, HTML, DOM, or other page-derived metadata.
The broker rejects screenshots over 16 MiB and caps each session at 100 screenshot files and 256 MiB total; artifacts are published atomically inside a contained, non-symlink directory.

Structured operations:

- `browser_navigate` normalizes the URL and waits for page readiness.
- `browser_click` targets an element ref.
- `browser_type` targets an element ref and uses human-paced typing by default.
- `browser_upload_file` copies one host file into the session runtime and sets an `input[type=file]` ref without mounting the host workspace.
  Callers must supply an absolute host path from the task workspace; relative paths are rejected so another repository's file cannot be selected accidentally.
  The broker revalidates the bounded regular-file source during copying; a zero or inaccurate size hint cannot bypass the 100 MiB source cap.
  One session may retain at most 64 staged uploads and 512 MiB total; a source change between validation and copying aborts the upload and removes the partial copy.
- `browser_backend_request` runs generic same-origin requests with the active page's ordinary cookies. Absolute/cross-origin URLs and unsupported arguments are rejected.
  It does not read auth-session endpoints or acquire bearer tokens.
  `timeout_ms` is bounded to `1..60000` and `max_response_chars` to `1..10000000`.
  One deadline covers the bounded response.
- `browser_press_key` targets the active page or a ref when supplied.
- `browser_wait` requires any supplied `seconds` value to be finite and in `0..10`, and requires `timeout_ms` to be in `1..10000`; calls outside those caps fail before controller execution.
- `browser_new_tab`, `browser_switch_tab`, and `browser_close_tab` operate within the current browser session and invalidate old element refs.
- tab tools return the active tab id deterministically.

Coordinate actions are optional and require explicit coordinate mode.
Stale refs fail with `STALE_REF`.
Pacing accepts `human`, `fast`, or `batch` directly; `human` is the default.
Mutation-capable tool parameters are limited to 1 MiB before controller execution.
The MCP catalog owns parameter bounds. Page and lifecycle errors describe invalid input, pause, missing tabs/sessions, or unavailable runtime resources.


### Page Artifacts

`browser_artifact` saves content for local processing and returns only path, MIME type, byte size, SHA-256 and session/tab identity.

- **Save page content:** `kind="html"` captures the rendered body or optional `scope_selector` as sanitized HTML.
  Scripts, hidden/unloaded content, form values, link destinations, images and styling are omitted; required citation URLs need separate extraction.
  A hidden scope fails with `SCOPE_NOT_VISIBLE`.
- **Download an existing file:** `kind="download"` fetches a fresh current-page anchor `ref` without filtering file types or changing the bytes.
  Downloads require a same-origin HTTP(S) link without redirects, bounded streaming and a 15-second deadline; other links need the site's normal download flow.
  Metadata reports the response MIME type, defaulting to `application/octet-stream` when absent. A generated filename uses a MIME-derived extension or `.bin`.

`max_bytes` defaults to 4 MiB, with a 16-MiB maximum; each session allows 100 files and 256 MiB total.
Files are published atomically as private regular files in the owning session's `artifacts/` directory.
Paused sessions allow HTML capture but block downloads.


### Template Lifecycle

Templates are broker-local sensitive browser profiles prepared by the human.
They are not git artifacts.
Profile validation and copying use the same exclusion list for transient Chrome locks and caches; all copied entries must remain real directories or regular files.

Template lifecycle:

1. `browser_template_create` creates an editing browser and returns status plus a recovery token; its current editor requests takeover information separately.
2. The human configures the browser, including login state when desired.
3. `browser_template_seal` lets the current editor stop the browser gracefully, flush profile state, and atomically publish an immutable cloneable version.
4. `browser_ensure` may clone a sealed version into a live runtime profile.
5. Live sessions never open the sealed profile directly.
6. Live session changes never write back to a sealed version automatically.
7. `browser_template_refresh` creates a new editing draft from a sealed version and later seals a new version.
8. `browser_template_delete` deletes the selected template profile data.

Seal conflicts report the current session role, state, template name and applicable next operations; only the template editor can seal.

New sealed-version metadata includes template id, version id, state, created/updated timestamps, configured runtime-image reference, runtime channel marker, viewport, screen, network-policy label, locale, timezone, and language hints.
Sealed template versions are immutable clone sources only.
Multiple MCP sessions may clone the same sealed version into independent runtime profiles and containers.
Live runtime changes never write back to the sealed version automatically, and template name never transfers browser ownership across MCP sessions.
Only one broker process may edit or mutate a template at a time.
Seal waits for in-flight work, stops Chrome, revokes observer access, and preserves a restartable draft before publication.
If publication fails, the draft remains restartable and no partial version is visible; if post-publication cleanup fails, retry reuses the committed version instead of creating a duplicate.
`overwrite=true` creates a replacement draft without deleting existing sealed versions.
Only after the replacement version publishes does seal prune the superseded versions, so replacement startup or pre-publication failure preserves the prior sealed template and the retryable draft.
If cleanup of an unpublished profile or version workspace fails, the broker immediately indexes that contained root as `orphaned` or `peer_owned`; template create, refresh, overwrite, and delete remain blocked until safe explicit cleanup or peer release resolves it.
Transport shutdown stops an active editor and retains its draft; recreation or retained-session deletion explicitly discards it.
A clone or refresh refuses a different configured runtime-image reference when the sealed version records one; legacy versions without that field remain readable.
This comparison does not resolve mutable tags to immutable image ids, so users must refresh templates deliberately after rebuilding an image under the same tag.

### Docker Session Runtime

The default Docker image tag is installation-scoped, derived from the current user and primary runtime-owning checkout path.
Primary and linked worktrees share the default installation id, image, and HTTP port; distinct repositories and users remain isolated.
`TOKIMON_RB_INSTALLATION_ID` may override that derived id, and `TOKIMON_RB_IMAGE` may override the full image name.
The default HTTP broker port is also installation-scoped, selected from the non-ephemeral `20000-20999` range, and reported in health diagnostics.
`TOKIMON_RB_PORT` or `serve-http --port` may override it for local client configuration.

The broker labels each session container and network with stable broker, installation, owner, checkout, runtime-root, and session identity; network policy remains diagnostic metadata.
The checkout identity in labels, network names, and ownership checks is the primary runtime owner; source checkout paths remain available for source provenance and artifact resolution.
It creates one installation-scoped user-defined bridge per session.
Before reusing or removing a same-named network, the broker verifies its immutable identity, expected `bridge` behavior, and stable ownership labels.
An owned stale network is recreated only when it has no attached containers; an attached, foreign, or ambiguous network fails closed.
Container startup uses only that exact network, and stop removes it after the container is absent, including when stop finds no container.

Container startup records a full immutable container id; missing or malformed identity makes cleanup ambiguous and keeps ownership reserved.
Before stop or removal, the broker verifies that id, the expected installation/session name, stable ownership labels, and the exact broker-owned profile mount.
Destructive Docker commands address only the verified immutable id.
Mutable network-policy metadata is not accepted as ownership proof.
Cleanup tools must not claim containers by generic image name, template name, exposed port, image tag, network-policy label, or unscoped `tokimon-rb-*` prefix.

The runtime provides a headed Chrome session plus broker-gated human view and internal browser-control paths.
Chrome Stable is the default; Chrome for Testing or Playwright Chromium may be used for tests, CI, or explicit fallback.
Chrome uses a non-default per-session profile directory, and the normal launch path preserves Chrome's sandbox.

The broker starts session containers with a hardening baseline:

- no privileged mode;
- no Docker socket mount;
- no host home mount;
- no live project checkout mount by default;
- Chrome-sandbox-compatible syscall policy and least Linux capabilities;
- CPU, memory, process, and tmpfs limits;
- minimal mounts for profile, downloads, and runtime needs;
- downloads treated as untrusted.

The container waits for Xvfb to accept display connections before starting display clients.
Display readiness is bounded; Xvfb exit or timeout fails startup with a diagnostic.

The image may start from a Playwright base image for bundled dependencies, but must install and require Google Chrome Stable by default.
The owning Dockerfile pins that base by OCI digest.
Debian and Chrome packages remain live rebuild-time inputs, so an image rebuild is a deliberate dependency-refresh event rather than a bit-for-bit reproducible build.
The current Chrome Stable package path supports signed Linux `amd64` packages; other architectures require an explicit supported image design.
Fallback to Chrome for Testing or Playwright Chromium requires explicit opt-in, such as `TOKIMON_RB_ALLOW_CHROMIUM_FALLBACK=1`, so stale images cannot silently downgrade production sessions.

### Network Policy

Default runtime networking is a dedicated user-defined Docker `bridge` per browser session, with only that session's browser container attached by the broker.
This prevents ordinary sibling containers attached to other networks from reaching the browser's container-IP CDP or noVNC listeners; the published host ports remain bound to loopback.
Docker-authority principals can attach another container to the network and remain trusted by this design.

The bridge preserves internet egress and does not itself deny the Docker host, metadata services, or private/local-network destinations.
The network-policy id remains configuration metadata rather than proof of destination filtering.
Diagnostics report the enforced per-session membership boundary separately from unrestricted destination egress.
Deployments that require destination denial must supply a host firewall or another external egress control.

## Data Ownership

Default runtime root:

```text
.tokimon/tools/mcp/remote-browser/
```

`TOKIMON_RB_RUNTIME_ROOT` may override the runtime root for local setup.
Relative overrides resolve from the primary worktree.
Overrides that point inside a linked worktree are rejected because persistent runtime state belongs under the primary worktree's `.tokimon/` or another non-worktree path.
The runtime root and every owned session, template, screenshot, upload, and profile collection must remain real contained non-symlink directories.
Recursive copies reject symlinks and special files.
Path containment and current ownership are both required; a safe-looking path alone does not authorize cleanup while another broker owns it.

Session profile path:

```text
.tokimon/tools/mcp/remote-browser/sessions/<session-id>/profile/
```

Session screenshot artifact path:

```text
.tokimon/tools/mcp/remote-browser/sessions/<session-id>/screenshots/
```

Session artifact directory for saved HTML and downloaded files:

```text
.tokimon/tools/mcp/remote-browser/sessions/<session-id>/artifacts/
```

Template profile path:

```text
.tokimon/tools/mcp/remote-browser/templates/<template-id>/versions/<version-id>/profile/
```

Retained-session metadata path:

```text
.tokimon/tools/mcp/remote-browser/sessions/<session-id>/metadata.json
```

The same owned metadata location is the crash checkpoint for active runtime ownership; template editors use corresponding draft metadata.
Active checkpoints preserve the real MCP session key and carry `starting`, `running`, paused, or `unknown` state without a retention expiry.
Only terminal retained checkpoints with a valid expiry are eligible for restart; active checkpoints reconcile into unknown/orphan ownership until explicit cleanup.

Runtime state is local-only and ignored by git.
Do not commit browser profiles, cookies, localStorage, screenshots, captured credentials, noVNC tokens, CDP endpoints, or session metadata that contains sensitive connection material.
Profile templates can contain cookies, localStorage, extension data, credentials, and other account state, so they follow the same local-only sensitive-data boundary.
Protocol inputs, action parameters, screenshots, uploads, artifact counts, and active/retained-session limits are enforced per broker process.
The bind-mounted profile and downloads directories do not have an aggregate filesystem quota enforced by the broker; host-level disk quotas and installation-wide capacity remain host responsibilities.

- Owns browser session metadata, profile directories, screenshot artifacts, retained-session records, template profiles, noVNC token metadata, and technical diagnostics under `.tokimon/tools/mcp/remote-browser/`.
- Consumes read-only user-provided template metadata, configured network/runtime policy, retained-session metadata during status calls, and Docker/Chrome diagnostics.
- Consumes mutable runtime inputs supplied through MCP page operations, uploads, pause/resume requests, and cleanup requests.
- Derives redacted status, session inventory, screenshot path metadata, template metadata, and diagnostics for MCP callers.

### Redaction and Logging

Default logs and diagnostics may include technical reachability, state, pacing mode, configured browser image/channel marker, template id/version, retention expiry, hardening summary, and network policy.
Health diagnostics also report this installation id, owner user, Docker image, owned browser containers, and browser containers outside this installation scope.

Default logs and diagnostics must not include:

- page URL;
- page title;
- page text;
- typed text;
- DOM;
- screenshots;
- cookies;
- tokens;
- raw page-operation parameters;
- connect URLs;
- CDP endpoints;
- credentials;
- page-content transcripts.

If a future explicit debug mode captures sensitive material, it must be explicitly enabled, time-limited, clearly labeled, and kept out of committed artifacts.

## Failure Modes

Remote Browser recovery preserves session metadata and diagnostics first, then recreates only owned sessions, profiles, containers, or templates after confirming the affected runtime boundary.

- **Host runtime unavailable:** Health reports missing Docker, Chrome, noVNC, or another required dependency; restore the dependency before creating a session.
- **Human-view startup or shutdown failure:** No failed startup credential becomes active, and revoked access remains revoked while the broker keeps enough ownership for a visible cleanup retry.
- **Missing or abandoned session:** Stale, unreachable, or active-at-crash state becomes retained, expired, unknown, or orphaned as evidence permits; unknown/orphan state is never adopted automatically and requires explicit cleanup.
- **Pause or page-state conflict:** Paused sessions block page actions; stale refs or changed tabs return stable errors so callers can refresh state.
- **Template invalid or publication interrupted:** Unsealed, missing, image-incompatible, or unsafe templates are refused; failed publication preserves the prior version and a retryable draft or quarantines unsafe residue.
- **Unsafe runtime path:** A linked-worktree override, symlink, special file, or escaping path is refused before template use or cleanup.
- **Foreign or concurrently owned resource:** Foreign/ambiguous containers and networks are reported but not adopted; `peer_owned` roots remain untouched until ownership is released.
- **Retained cleanup or persistence failure:** Sensitive profiles and their ownership remain reserved and retryable until exact identity verification, stop, metadata persistence, and deletion succeed.
- **Shutdown, capacity, or input bound reached:** Shutdown drains admitted work; capacity admission stops eligible idle runtimes before reporting `SESSION_CAPACITY_EXHAUSTED`. Retained profiles and uncertain resources stay owned. Oversized protocol/action/artifact input is rejected before allocation.
- **External host limit:** Disk capacity and destination-restricted egress remain host controls; diagnostics do not claim the per-session bridge supplies either guarantee.

## Acceptance Checks

Document-first verification:

```bash
git diff --check
.venv/bin/python -m unittest tests.test_c4_contract_language
.venv/bin/python -m unittest discover -s tests/tools/registry -t . -p 'test_registry.py'
```

Remote browser broker tests:

```bash
.venv/bin/python -m unittest discover -s tests/tools/mcp/remote_browser_broker -t . -p 'test_*.py'
```

Unit tests must cover these behavior families:

- direct browser and management operations, catalogs and diagnostics without approval/configuration surfaces
- human-view credential lifecycle and human pause

- MCP/CLI catalogs, session reuse and isolation, page operations, bounded artifacts, stable errors, and registry availability
- protocol negotiation, silent notifications, explicit request ids, and parser recovery through `tests/tools/mcp/test_protocol.py`
- strict UTF-8 byte-stream replies, accurate HTTP byte lengths, and normal EOF cleanup through `tests/tools/mcp/test_wire_encoding.py`
- sanitized HTML capture and linked-file downloads, metadata-only replies, file limits, and pause behavior
- current-session viewer access, resume, and stop; human-view link reuse for 24 hours and renewal after expiry; immediate credential revocation and redacted status
- ordinary task operations without per-action approval, snapshot-ref invalidation, paused-action rejection with continued observation, and direct pacing selection
- lifecycle concurrency, display readiness before client startup, bounded display failure, readiness publication, shutdown drain, crash reconciliation, retained-state recovery, and peer-owned exclusion
- profile preservation on EOF, HTTP termination, expiry, and capacity pressure; idle restoration and protected-session exclusion
- recovery across fresh MCP keys and broker restarts, invalid-token/peer/duplicate refusal, checkpoint failure rollback, and token redaction
- stopped-session upload/screenshot/artifact cleanup, quota reuse, active-work exclusion, and profile preservation
- template create/edit/seal/clone/refresh/delete, immutable versions, interrupted publication, and replacement rollback
- primary/linked worktree template reuse and Docker ownership, unchanged primary defaults, and explicit installation isolation
- path containment, input/resource limits, verified Docker ownership, runtime hardening, network boundaries, and foreign-resource refusal
- generic same-origin requests with one bounded fetch, no auth-session fetch or bearer header, and rejection of removed authentication arguments

Live checks must include:

- low-risk public site navigation and harmless interaction without fixed ports;
- human creates/seals a login template, agent clones it, and cleanup does not mutate the sealed version;
- human pause blocks automatic page actions until resume;
- fingerprint baseline self-check reports coherent runtime signals without claiming third-party undetectability.

Expected outcome: document-first checks pass; unit tests cover MCP dispatch, session isolation, page operations, screenshots, templates, redaction, and registry availability; live checks produce concrete Docker/browser evidence.

Ask an AI coding agent to verify the component contract:

```text
Review docs/c4/level-3-components/tools/mcp/remote-browser-broker.md, tokimon/tools/mcp/remote_browser_broker/, tests/tools/mcp/remote_browser_broker/, and tests/tools/registry/.
Confirm that the documented MCP transport contract, session lifecycle, page-operation contract, pause behavior, template lifecycle, runtime state ownership, redaction rules, failure modes, and tests match the implementation.
Report any drift with file and line evidence.
```

Expected outcome: the agent reports no contract drift, or lists precise mismatches with recommended owner documents or implementation files.

## References

- L1 Context: `docs/c4/level-1-context/tools-context.md`
- L2 Container: `docs/c4/level-2-containers/tools/mcp/remote-browser-broker.md`
