"""Save rendered HTML or download linked files into private session files."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import mimetypes
import os
import re
import secrets
from typing import Any

MAX_BYTES = 16 * 1024 * 1024
MAX_FILES = 100
MAX_TOTAL_BYTES = 256 * 1024 * 1024


def render_script(template: str, **values: str) -> str:
    """Replace template markers once; user selectors/IDs never become new markers."""
    return re.sub(r"\b(" + "|".join(re.escape(key) for key in values) + r")\b",
                  lambda match: values[match.group()], template)


def invalid(message: str, code: str = "INVALID_ARGUMENT") -> None:
    from . import BrowserBrokerError
    raise BrowserBrokerError(message, code=code)


def html_expression(scope_selector: str | None, max_bytes: int) -> str:
    return render_script("""(() => {
const visible = el => {
  if (!el || !el.getClientRects().length) return false;
  for (let node = el; node; node = node.parentElement) {
    if (node.hidden || node.getAttribute('aria-hidden') === 'true') return false;
    const style = getComputedStyle(node);
    if (style.display === 'none' || ['hidden','collapse'].includes(style.visibility)
        || style.contentVisibility === 'hidden' || style.opacity === '0') return false;
  }
  return true;
};
let scope = document.body;
const selector = SELECTOR;
if (selector) {
  const scopes = document.querySelectorAll(selector);
  if (scopes.length !== 1) throw new Error('SCOPE_NOT_FOUND_OR_AMBIGUOUS');
  scope = scopes[0];
}
if (!scope) throw new Error('SCOPE_NOT_FOUND_OR_AMBIGUOUS');
if (!visible(scope)) throw new Error('SCOPE_NOT_VISIBLE');
const skip = new Set(['SCRIPT','STYLE','TEMPLATE','NOSCRIPT','INPUT','TEXTAREA','SELECT','OPTION','IFRAME','OBJECT','EMBED','SVG','CANVAS']);
const allowed = new Set(['BODY','MAIN','ARTICLE','SECTION','DIV','P','SPAN','A','H1','H2','H3','H4','H5','H6','UL','OL','LI','TABLE','THEAD','TBODY','TR','TH','TD','PRE','CODE','BLOCKQUOTE','STRONG','EM','B','I','BR','HR']);
let total = 0;
const bounded = value => {total += new TextEncoder().encode(value).length; if(total > LIMIT) throw new Error('ARTIFACT_TOO_LARGE'); return value;};
const escape = value => value.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
function render(node) {
  if (node.nodeType === 3) return bounded(escape(node.textContent || ''));
  if (node.nodeType !== 1 || skip.has(node.tagName) || !visible(node)) return '';
  const tag = allowed.has(node.tagName) ? node.tagName.toLowerCase() : 'div';
  const open = bounded('<' + tag + '>');
  const children = Array.from(node.childNodes).map(render).join('');
  return open + children + bounded('</' + tag + '>');
}
return '<!doctype html>' + render(scope);
})()""", SELECTOR=json.dumps(scope_selector), LIMIT=str(max_bytes - 15))


def download_expression(ref_data: dict, max_bytes: int) -> str:
    from . import element_ref_expression
    return render_script("""(async () => {
REF_CHECK
if (!(el instanceof HTMLAnchorElement)) throw new Error('ARTIFACT_UNSUPPORTED');
const url = new URL(el.href, location.href);
if (!['http:','https:'].includes(url.protocol) || url.origin !== location.origin || url.username || url.password) throw new Error('ARTIFACT_UNSUPPORTED');
const controller = new AbortController();
const timer = setTimeout(() => controller.abort(), 15000);
let reader;
try {
  const response = await fetch(url.href, {method:'GET', credentials:'same-origin', redirect:'manual', signal:controller.signal});
  if (!response.ok || response.type === 'opaqueredirect') throw new Error('ARTIFACT_UNSUPPORTED');
  const size = Number(response.headers.get('content-length') || 0);
  if (size > LIMIT) throw new Error('ARTIFACT_TOO_LARGE');
  reader = response.body && response.body.getReader();
  let total = 0;
  const chunks = [];
  while (reader) {
    const {done, value} = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > LIMIT) throw new Error('ARTIFACT_TOO_LARGE');
    chunks.push(value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {bytes.set(chunk, offset); offset += chunk.length;}
  let binary = '';
  for(let i=0; i<bytes.length; i+=8192) binary += String.fromCharCode(...bytes.subarray(i,i+8192));
  return {data: btoa(binary), mime_type: (response.headers.get('content-type') || 'application/octet-stream').split(';')[0].trim().toLowerCase()};
} finally {
  if(reader) {try {await reader.cancel();} catch (_) {}}
  clearTimeout(timer); controller.abort();
}
})()""", REF_CHECK=element_ref_expression(ref_data), LIMIT=str(max_bytes))


def capture(broker: Any, key: str, *, kind: str, scope_selector: str | None = None,
            ref: str | None = None, max_bytes: int = 4194304) -> dict:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= MAX_BYTES:
        invalid(f"max_bytes must be an integer in 1..{MAX_BYTES}")
    if kind not in {"html", "download"} or (kind == "html" and ref is not None) or (kind == "download" and (scope_selector is not None or not isinstance(ref, str) or not ref)):
        invalid("Use kind=html with optional scope_selector, or kind=download with a fresh anchor ref")
    if scope_selector is not None and (not isinstance(scope_selector, str) or not 1 <= len(scope_selector) <= 1000):
        invalid("scope_selector must be a nonempty selector of at most 1000 characters")
    session = broker._session_or_error(key)
    with session.operation_lock:
        if kind == "download":
            broker._require_page_action_allowed(session)
        else:
            broker._require_page_observation_allowed(session)
        directory = broker._owned_child_directory(broker._validated_session_root(session), "artifacts")
        count, total = broker._owned_artifact_usage(directory)
        if count >= MAX_FILES or total >= MAX_TOTAL_BYTES:
            invalid("Artifact quota exhausted; stop and clean this session's artifacts", "ARTIFACT_QUOTA_EXCEEDED")
        script = html_expression(scope_selector, max_bytes) if kind == "html" else download_expression(session.resolve_ref(ref), max_bytes)
        raw = broker.controller.isolated_evaluate(session, script, timeout=15)
        mime_type = "text/html"
        if kind == "download":
            if not isinstance(raw, dict) or not isinstance(raw.get("mime_type"), str):
                invalid("Invalid download response", "ARTIFACT_UNSUPPORTED")
            mime_type = raw["mime_type"] or "application/octet-stream"
            raw = raw.get("data")
        if not isinstance(raw, str) or len(raw) > (max_bytes * 4 // 3 + 4 if kind == "download" else max_bytes):
            invalid("Artifact exceeded the requested byte limit", "ARTIFACT_TOO_LARGE")
        try:
            data = base64.b64decode(raw, validate=True) if kind == "download" else raw.encode("utf-8")
        except (ValueError, binascii.Error):
            invalid("Artifact data is invalid", "ARTIFACT_UNSUPPORTED")
        if len(data) > max_bytes:
            invalid("Artifact exceeded the requested byte limit", "ARTIFACT_TOO_LARGE")
        if total + len(data) > MAX_TOTAL_BYTES:
            invalid("Artifact byte quota exceeded", "ARTIFACT_QUOTA_EXCEEDED")
        extension = mimetypes.guess_extension(mime_type) or ".bin"
        name = f"{secrets.token_hex(16)}{extension}"
        temporary = f".{name}.tmp"
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            out = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            with os.fdopen(out, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=fd)
            except FileNotFoundError:
                pass
            os.close(fd)
        session.touch()
        return dict(identity=dict(session_id=session.id, tab_id=session.active_tab_id), artifact=dict(
            path=str(directory / name), mime_type=mime_type,
            size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
