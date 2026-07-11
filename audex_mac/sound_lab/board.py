"""Local blind-audition board for Audex Sound Lab."""

from __future__ import annotations

import json
import threading
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .catalog import SoundLabCatalog

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_MAX_REQUEST_BYTES = 64 * 1024


class SoundLabBoard:
    """Expose one catalog through a loopback-only browser interface."""

    def __init__(
        self,
        catalog: SoundLabCatalog,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        opener: Callable[[str], object] | None = webbrowser.open,
    ) -> None:
        if host not in _LOOPBACK_HOSTS:
            raise ValueError("Sound Lab board must bind to a loopback host")
        self._catalog = catalog
        self._host = host
        self._port = port
        self._opener = opener
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("Sound Lab board has not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> str:
        if self._server is not None:
            return self.url
        handler = _handler_for(self._catalog)
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="audex-sound-lab-board",
            daemon=True,
        )
        self._thread.start()
        if self._opener is not None:
            self._opener(self.url)
        return self.url

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def __enter__(self) -> SoundLabBoard:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _handler_for(catalog: SoundLabCatalog) -> type[BaseHTTPRequestHandler]:
    class SoundLabRequestHandler(BaseHTTPRequestHandler):
        server_version = "AudexSoundLab/0.1"

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/":
                self._send_bytes(
                    HTTPStatus.OK,
                    _BOARD_HTML.encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return
            if path == "/api/state":
                self._send_json(HTTPStatus.OK, catalog.public_snapshot())
                return
            if path.startswith("/audio/"):
                asset_id = unquote(path.removeprefix("/audio/"))
                if not asset_id or "/" in asset_id or "\\" in asset_id:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                try:
                    audio_path = catalog.audio_path(asset_id)
                except (KeyError, FileNotFoundError):
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                self._send_file(audio_path)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            try:
                payload = self._read_json()
                if path == "/api/preferences":
                    catalog.record_preference(
                        job_id=_required_string(payload, "job_id"),
                        selected_label=_required_string(payload, "selected_label"),
                        rejected_labels=_string_tuple(payload, "rejected_labels"),
                        note=str(payload.get("note", "")),
                    )
                elif path == "/api/reveal":
                    catalog.reveal_job(_required_string(payload, "job_id"))
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"ok": True})

        def log_message(self, _format: str, *args: object) -> None:
            del args

        def _read_json(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length", "")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if not 0 < length <= _MAX_REQUEST_BYTES:
                raise ValueError("request body size is invalid")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _send_file(self, path: Path) -> None:
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as source:
                while chunk := source.read(64 * 1024):
                    self.wfile.write(chunk)

        def _send_json(self, status: HTTPStatus, payload: object) -> None:
            self._send_bytes(
                status,
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _send_bytes(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; media-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
            )
            self.end_headers()
            self.wfile.write(body)

    return SoundLabRequestHandler


def _required_string(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _string_tuple(payload: dict[str, Any], name: str) -> tuple[str, ...]:
    value = payload.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a string array")
    return tuple(value)


_BOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audex Sound Lab</title>
<style>
:root { color-scheme: dark; font-family: ui-monospace, SFMono-Regular, monospace; }
body { margin: 0 auto; max-width: 1050px; padding: 24px; background: #111; color: #eee; }
h1 { font-size: 1.25rem; font-weight: 600; }
.hint { color: #aaa; margin-bottom: 24px; }
.job { border: 1px solid #444; border-radius: 8px; padding: 16px; margin: 14px 0; }
.rack { display: grid; grid-template-columns: repeat(auto-fit,minmax(170px,1fr)); gap: 10px; }
.candidate { border: 1px solid #333; padding: 12px; border-radius: 6px; }
.label { font-size: 2rem; margin-bottom: 8px; }
audio { width: 100%; }
button,input { font: inherit; margin: 5px 4px 0 0; }
.meta { color: #aaa; font-size: .85rem; white-space: pre-wrap; }
</style>
</head>
<body>
<h1>Audex Sound Lab</h1>
<div class="hint">Blind rack: production details stay hidden until you record a preference and reveal them.</div>
<main id="jobs">Waiting for a sound request in the terminal.</main>
<script>
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function post(path, payload) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  if (!response.ok) throw new Error((await response.json()).error || response.statusText);
  await refresh();
}
function renderCandidate(candidate, revealed) {
  const audio = candidate.audio_url ? `<audio controls preload="metadata" src="${esc(candidate.audio_url)}"></audio>` : `<div class="meta">${esc(candidate.state)}</div>`;
  const details = revealed ? `<div class="meta">${esc(candidate.caption)}\nseed ${esc(candidate.seed)}\n${esc(candidate.difference)}</div>` : '';
  return `<section class="candidate"><div class="label">${esc(candidate.label)}</div>${audio}${details}</section>`;
}
function renderJob(job) {
  const preference = job.preference || {};
  return `<article class="job" data-job="${esc(job.job_id)}">
    <div class="meta">${esc(job.job_id)} | ${esc(job.state)}</div>
    <div class="rack">${job.candidates.map(c => renderCandidate(c, job.revealed)).join('')}</div>
    <div><input class="winner" size="3" maxlength="1" placeholder="winner"><input class="rejected" size="12" placeholder="reject B,C"><input class="note" size="32" placeholder="Why did it win?">
    <button onclick="savePreference('${esc(job.job_id)}')">Save winner</button>
    <button onclick="post('/api/reveal',{job_id:'${esc(job.job_id)}'})">Reveal recipe</button></div>
    ${preference.selected_label ? `<div class="meta">Winner ${esc(preference.selected_label)}: ${esc(preference.note)}</div>` : ''}
  </article>`;
}
async function savePreference(jobId) {
  const root = document.querySelector(`[data-job="${jobId}"]`);
  const selected = root.querySelector('.winner').value.trim().toUpperCase();
  const rejected = root.querySelector('.rejected').value.split(',').map(value => value.trim().toUpperCase()).filter(Boolean);
  const note = root.querySelector('.note').value.trim();
  await post('/api/preferences',{job_id:jobId,selected_label:selected,rejected_labels:rejected,note});
}
let lastRenderedState = '';
function previewIsPlaying() {
  return [...document.querySelectorAll('audio')].some(audio => !audio.paused && !audio.ended);
}
async function refresh() {
  if (previewIsPlaying()) return;
  const state = await (await fetch('/api/state',{cache:'no-store'})).json();
  if (previewIsPlaying()) return;
  const serialized = JSON.stringify(state);
  if (serialized === lastRenderedState) return;
  lastRenderedState = serialized;
  document.getElementById('jobs').innerHTML = state.jobs.length ? state.jobs.map(renderJob).join('') : 'Waiting for a sound request in the terminal.';
}
refresh(); setInterval(refresh, 1000);
</script>
</body>
</html>
"""
