# Tokimon Gateway (WebSocket)

Tokimon Gateway is a local server mode that exposes:

- Existing Chat UI HTTP endpoints: `GET /healthz`, `POST /api/send`
- A new WebSocket control plane endpoint: `GET /gateway` (WS upgrade)

This is inspired by OpenClaw's Gateway protocol, but is intentionally minimal.
Phase 3 adds protocol negotiation up to v3, a basic presence snapshot, and
OpenClaw-inspired device identity + challenge signing for protocol v3.

## CLI

Start the Gateway server:

```bash
tokimon gateway --host 127.0.0.1 --port 8765 --llm codex
```

Notes:
- `tokimon chat-ui` remains available and unchanged.
- Gateway is a superset server: it supports `/healthz` + `/api/send` and adds `/gateway`.
- Gateway binds loopback by default. To bind to a non-loopback interface (e.g. `0.0.0.0`), you must:
  - Configure `TOKIMON_GATEWAY_AUTH_TOKEN`, and
  - Pass `--dangerously-expose` (explicit opt-in).
- If `tokimon health` / `tokimon logs` fail with a WS handshake error (expected HTTP 101), the URL is pointing at a non-Gateway HTTP server (commonly `tokimon chat-ui` or another service on that port). Start `tokimon gateway run` on a free port and pass `--url ws://127.0.0.1:<port>/gateway`.

## Transport

- WebSocket over HTTP upgrade at `GET /gateway`.
- Text frames with UTF-8 JSON payloads only.
- The first client message **must** be a `connect` request.

## Framing

All WS messages are JSON objects with one of the following shapes:

- **Request**: `{ "type": "req", "id": "<string>", "method": "<string>", "params": { ... } }`
- **Response**: `{ "type": "res", "id": "<string>", "ok": <bool>, "payload": { ... } }`
  - On error: `{ "type": "res", "id": "<string>", "ok": false, "error": { "message": "<string>", "details"?: { ... } } }`
- **Event**: `{ "type": "event", "event": "<string>", "payload": { ... }, "seq"?: <int>, "stateVersion"?: <int> }`

## Handshake (`connect`)

On socket open, Gateway sends a pre-connect challenge event:

```json
{ "type": "event", "event": "connect.challenge", "payload": { "nonce": "…", "ts": 1737264000000 } }
```

The client must then send a `connect` request as its first frame:

```json
{
  "type": "req",
  "id": "1",
  "method": "connect",
  "params": {
    "minProtocol": 1,
    "maxProtocol": 3,
    "challenge": { "nonce": "…" },
    "client": { "id": "cli", "version": "0.1.0", "platform": "linux", "deviceFamily": "laptop", "mode": "operator" },
    "auth": { "mode": "token", "credential": "…" },
    "device": {
      "id": "…",        // sha256(publicKey_raw_bytes).hexdigest()
      "publicKey": "…", // base64url raw Ed25519 public key bytes
      "signature": "…", // base64url Ed25519 signature over the payload string
      "signedAt": 1737264000000,
      "nonce": "…"      // must equal connect.challenge nonce
    },
    "role": "operator",
    "scopes": ["operator.read", "operator.write"]
  }
}
```

Gateway responds:

```json
{
  "type": "res",
  "id": "1",
  "ok": true,
  "payload": { "type": "hello-ok", "protocol": 3, "policy": { "tickIntervalMs": 15000 } }
}
```

### Protocol versioning

- The server supports protocol versions `1..3`.
- Clients send `connect.params.minProtocol` and `connect.params.maxProtocol`.
- The server selects the highest common supported version in the inclusive range.
- `hello-ok.payload.protocol` is the negotiated protocol.
- If there is no overlap, the server responds with `ok=false` and closes the socket.

### Challenge echo

- The client MUST echo back the `connect.challenge.payload.nonce` in `connect.params.challenge.nonce`.
- On mismatch or missing nonce, the server responds with `ok=false` and closes the socket.

### Token auth (optional)

- When `TOKIMON_GATEWAY_AUTH_TOKEN` is configured on the server, the client MUST include one of:
  - `connect.params.auth = { "mode": "token", "credential": "<token>" }`
  - `connect.params.auth = { "token": "<token>" }` (OpenClaw-style)
- When `TOKIMON_GATEWAY_AUTH_TOKEN` is not configured, `connect.params.auth` is optional and ignored.

### Device identity + challenge signing (protocol 3)

- For negotiated protocol versions `1..2`, `connect.params.device` is accepted (type-checked) but ignored.
- For negotiated protocol versions `>= 3`, Gateway requires device identity + challenge signing unless
  `TOKIMON_GATEWAY_DANGEROUSLY_DISABLE_DEVICE_AUTH=1` is set.

When required, the client MUST include `connect.params.device` with:

- `id` (string): `sha256(publicKey_raw_bytes).hexdigest()`
- `publicKey` (string): base64url raw Ed25519 public key bytes
- `signature` (string): base64url Ed25519 signature
- `signedAt` (int): epoch ms
- `nonce` (string): non-blank and MUST equal the `connect.challenge` nonce

Validation rules:

- `device.id` MUST equal `sha256(device.publicKey_raw_bytes).hexdigest()`.
- `device.signedAt` MUST be within ±2 minutes of server time.
- `device.nonce` MUST be present (non-blank after trim) and MUST equal the `connect.challenge` nonce.
- `device.signature` MUST verify (Ed25519) over a single UTF-8 payload string. Gateway tries:
  - v3 payload first, then falls back to v2 payload (OpenClaw-compatible)
  - Token field selection: use `connect.params.auth.token` if present, else `connect.params.auth.credential`
    when `auth.mode == "token"`, else `""`.
  - v3 payload includes `platform` and `deviceFamily`, normalized by trimming and lowercasing ASCII A–Z only
    (non-ASCII unchanged).

On failure, Gateway responds to `connect` with `ok=false`, includes OpenClaw-compatible `error.details.code`
and `error.details.reason` (when applicable), and closes the socket.

### Optional connect params (accepted; semantics mostly ignored)

The server accepts (and type-checks) these optional `connect.params` fields:

- `caps` (list)
- `commands` (list)
- `permissions` (object)
- `locale` (string)
- `userAgent` (string)
- `device` (object)
  - Protocol `1..2`: accepted keys (all optional): `id` (string), `publicKey` (string), `signature` (string), `signedAt` (int), `nonce` (string)
  - Protocol `>=3`: required unless `TOKIMON_GATEWAY_DANGEROUSLY_DISABLE_DEVICE_AUTH=1` is set; see above.

## JSON-schema validation (Phase 1)

Gateway performs deterministic validation for:

- Frame envelope fields (`type`, `id`, `method`, `params` for requests).
- `connect.params` minimum fields (`minProtocol`, `maxProtocol`, `client.*`, `role`).
- Method-specific params for `send`.

Invalid frames return an `ok=false` response (when an `id` is available) and the
socket may be closed for handshake violations.

## Idempotency keys

Side-effecting methods must include an idempotency key:

- `send.params.idempotencyKey` is required and must be a non-empty string.
- Repeating the same `idempotencyKey` on the same connection returns the cached
  payload from the first call.

## Methods

### `health`

Request:

```json
{ "type": "req", "id": "2", "method": "health", "params": {} }
```

Response payload:

```json
{ "ok": true }
```

### `send`

Invokes the same worker logic as `POST /api/send`.

Request params:

- `message` (string, required)
- `history` (list, optional)
- `model` (string, optional)
- `idempotencyKey` (string, required)

Response payload includes `status`, `reply`, and other worker output fields.

### `logs.tail`

Returns recent log entries from an in-memory ring buffer.

Request params:

- `limit` (int, optional; default: 200)
- `after` (int, optional; return only entries with `id > after`)

Response payload:

```json
{ "entries": [{ "id": 1, "ts_ms": 1737264000000, "event": "send", "payload": {} }], "cursor": 12 }
```

### `methods.list`

Returns the list of server-supported WebSocket RPC methods (excluding `connect`)
in deterministic order, gated by the negotiated protocol version:

- Protocol `1` (and `2`): Phase 1 method list.
- Protocol `3`: Phase 1 list plus `system-presence`.

Request:

```json
{ "type": "req", "id": "4", "method": "methods.list", "params": {} }
```

Response payload:

```json
{ "methods": ["health", "logs.tail", "methods.list", "send", "tools.catalog"] }
```

### `system-presence` (protocol 3 only)

Returns a deterministic snapshot of active WebSocket connections.

Request:

```json
{ "type": "req", "id": "6", "method": "system-presence", "params": {} }
```

Response payload:

```json
{
  "connections": [
    {
      "device": { "id": "…" },
      "role": "operator",
      "scopes": ["operator.read"],
      "client": { "id": "pytest-v3", "version": "0", "platform": "linux", "mode": "operator" }
    }
  ]
}
```

Ordering is stable and deterministic (sorted by `device.id`, then `role`, then `client.id`, then connection arrival order).

### `tools.catalog`

Returns a deterministic catalog of tool/action pairs and their risk
classification derived from `src/policy/dangerous_tools.py`.

Request:

```json
{ "type": "req", "id": "5", "method": "tools.catalog", "params": {} }
```

Response payload:

```json
{
  "tools": [
    {
      "tool": "file",
      "action": "read",
      "risk_tier": "low",
      "requires_approval": false,
      "notes": "read-only workspace access"
    }
  ]
}
```

## TODO (Future phases)

- Explicit `node` role and semantics for `caps`/`commands`/`permissions` and richer presence APIs.
- Multi-channel connectors and server-push events beyond handshake.

## Tests

Gateway WS contract is covered by deterministic pytest tests:

- `src/tests/test_gateway_ws.py` covers: handshake + protocol negotiation + health + methods.list (protocol-gated) + tools.catalog + send + system-presence

Run:

```bash
pytest --maxfail=1 -c src/pyproject.toml src/tests
```
