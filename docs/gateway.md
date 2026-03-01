# Tokimon Gateway (WebSocket)

Tokimon Gateway is a local server mode that exposes:

- Existing Chat UI HTTP endpoints: `GET /healthz`, `POST /api/send`
- A new WebSocket control plane endpoint: `GET /gateway` (WS upgrade)

This is inspired by OpenClaw's Gateway protocol, but is intentionally minimal
in Phase 1.

## CLI

Start the Gateway server:

```bash
tokimon gateway --host 127.0.0.1 --port 8765 --llm mock
```

Notes:
- `tokimon chat-ui` remains available and unchanged.
- Gateway is a superset server: it supports `/healthz` + `/api/send` and adds `/gateway`.
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
    "maxProtocol": 1,
    "client": { "id": "cli", "version": "0.1.0", "platform": "linux", "mode": "operator" },
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
  "payload": { "type": "hello-ok", "protocol": 1, "policy": { "tickIntervalMs": 15000 } }
}
```

### Protocol versioning

- `PROTOCOL_VERSION = 1` for Phase 1.
- The server accepts the connection only if `minProtocol <= PROTOCOL_VERSION <= maxProtocol`.
- On mismatch, the server responds with `ok=false` and closes the socket.

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

## Methods (Phase 1)

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
- `idempotencyKey` (string, required)

Response payload includes `status`, `reply`, and other worker output fields.

## TODO (Phase 2, docs only)

- Device identity + signatures (challenge signing).
- Explicit `node` role, caps/commands/permissions, presence APIs.
- Multi-channel connectors and server-push events beyond handshake.

## Tests

Gateway WS contract is covered by deterministic pytest tests:

- `src/tests/test_gateway_ws.py` covers: handshake + health + send

Run:

```bash
pytest --maxfail=1 -c src/pyproject.toml src/tests
```
