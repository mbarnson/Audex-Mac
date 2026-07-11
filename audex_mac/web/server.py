"""Dependency-free localhost HTTP server for the Audex browser interface."""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .chat import ChatCoordinator
from .modes import ChatMode, mode_catalog

STATIC_ROOT = Path(__file__).with_name("static")
MAX_REQUEST_BYTES = 64 * 1024 * 1024
_CHAT_ROUTE = re.compile(r"^/api/chats/([^/]+)$")
_TURN_ROUTE = re.compile(r"^/api/chats/([^/]+)/turns$")
_MEDIA_ROUTE = re.compile(r"^/api/chats/([^/]+)/media/([^/]+)$")
_ASSET_MEDIA_ROUTE = re.compile(r"^/api/chats/([^/]+)/media/([^/]+)/assets/(\d+)$")


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    body: bytes
    content_type: str = "application/json; charset=utf-8"
    headers: dict[str, str] = field(default_factory=dict)


class AudexWebApplication:
    """Route HTTP requests through the cache-preserving chat interface."""

    def __init__(self, *, coordinator: ChatCoordinator, upload_root: Path) -> None:
        self.coordinator = coordinator
        self.upload_root = Path(upload_root)

    def dispatch(self, method: str, raw_path: str, body: bytes = b"") -> HttpResponse:
        path = unquote(urlsplit(raw_path).path)
        try:
            return self._dispatch(method.upper(), path, body)
        except json.JSONDecodeError:
            return _error(400, "Request body must be valid JSON.")
        except (binascii.Error, UnicodeError):
            return _error(400, "Audio must be valid base64 data.")
        except (ValueError, FileNotFoundError) as exc:
            return _error(400, str(exc))
        except KeyError as exc:
            return _error(404, str(exc).strip("'"))

    def _dispatch(self, method: str, path: str, body: bytes) -> HttpResponse:
        if method == "GET" and path == "/api/bootstrap":
            return _json_response(
                200,
                {
                    "name": "Audex",
                    "modes": mode_catalog(),
                    "chats": [
                        chat.to_dict() for chat in self.coordinator.store.list_chats()
                    ],
                },
            )
        if method == "POST" and path == "/api/chats":
            payload = _payload(body)
            mode = _mode(payload.get("mode", ChatMode.TEXT_TEXT.value))
            return _json_response(
                201,
                {"chat": self.coordinator.create_chat(mode=mode).to_dict()},
            )

        match = _CHAT_ROUTE.fullmatch(path)
        if match and method == "GET":
            return _json_response(
                200,
                {"chat": self.coordinator.store.load(match.group(1)).to_dict()},
            )
        if match and method == "PATCH":
            payload = _payload(body)
            return _json_response(
                200,
                {
                    "chat": self.coordinator.rename_chat(
                        match.group(1), str(payload.get("title", ""))
                    ).to_dict()
                },
            )

        match = _TURN_ROUTE.fullmatch(path)
        if match and method == "POST":
            chat_id = match.group(1)
            payload = _payload(body)
            mode = _mode(payload.get("mode"))
            audio_path = self._decode_audio(chat_id, payload.get("audio"))
            turn = self.coordinator.submit(
                chat_id,
                mode=mode,
                text=(
                    str(payload["text"]) if payload.get("text") is not None else None
                ),
                audio_path=audio_path,
            )
            return _json_response(201, turn.to_dict())

        match = _ASSET_MEDIA_ROUTE.fullmatch(path)
        if match and method == "GET":
            media_path = self.coordinator.media_path(
                match.group(1),
                match.group(2),
                asset_index=int(match.group(3)),
            )
            return _media_response(media_path)

        match = _MEDIA_ROUTE.fullmatch(path)
        if match and method == "GET":
            media_path = self.coordinator.media_path(match.group(1), match.group(2))
            return _media_response(media_path)

        if method == "GET" and path in {"/", "/index.html"}:
            return _static_response("index.html")
        if method == "GET" and path.startswith("/assets/"):
            return _static_response(path.removeprefix("/assets/"))
        return _error(404, f"Audex route not found: {method} {path}")

    def _decode_audio(self, chat_id: str, raw_audio: object) -> Path | None:
        if raw_audio is None:
            return None
        if not isinstance(raw_audio, dict):
            raise ValueError("Audio payload must contain name and base64 fields.")
        encoded = raw_audio.get("base64")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("Audio payload requires base64 data.")
        binary = base64.b64decode(encoded, validate=True)
        if len(binary) > MAX_REQUEST_BYTES:
            raise ValueError("Audio upload exceeds the 64 MB local limit.")
        if (
            len(binary) < 12
            or not binary.startswith(b"RIFF")
            or binary[8:12] != b"WAVE"
        ):
            raise ValueError("Browser audio must be a PCM WAV file.")
        destination = self.upload_root / chat_id
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"input-{uuid.uuid4().hex}.wav"
        path.write_bytes(binary)
        return path


def serve(
    application: AudexWebApplication,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    """Serve Audex until interrupted."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "AudexLocal/0.1"

        def do_GET(self) -> None:  # noqa: N802
            self._respond(application.dispatch("GET", self.path))

        def do_POST(self) -> None:  # noqa: N802
            self._respond(application.dispatch("POST", self.path, self._body()))

        def do_PATCH(self) -> None:  # noqa: N802
            self._respond(application.dispatch("PATCH", self.path, self._body()))

        def _body(self) -> bytes:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError:
                length = 0
            if length > MAX_REQUEST_BYTES * 2:
                return b""
            return self.rfile.read(max(0, length))

        def _respond(self, response: HttpResponse) -> None:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(response.body)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"Audex web: {format % args}", flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{server.server_port}"
    print(f"Audex browser interface: {url}", flush=True)
    if on_ready is not None:
        on_ready(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _payload(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def _mode(raw: object) -> ChatMode:
    try:
        return ChatMode(str(raw))
    except ValueError as exc:
        raise ValueError("Request requires a valid mode.") from exc


def _json_response(status: int, payload: dict[str, Any]) -> HttpResponse:
    return HttpResponse(
        status,
        json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    )


def _error(status: int, message: str) -> HttpResponse:
    return _json_response(status, {"error": message})


def _static_response(name: str) -> HttpResponse:
    if name not in {"index.html", "app.css", "app.js"}:
        return _error(404, "Static asset not found.")
    path = STATIC_ROOT / name
    if not path.is_file():
        return _error(404, "Static asset not found.")
    return HttpResponse(
        HTTPStatus.OK,
        path.read_bytes(),
        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        {"Cache-Control": "no-cache"},
    )


def _media_response(path: Path) -> HttpResponse:
    return HttpResponse(
        200,
        path.read_bytes(),
        (
            "audio/wav"
            if path.suffix.casefold() == ".wav"
            else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        ),
        {"Cache-Control": "private, max-age=3600"},
    )
