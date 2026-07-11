from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from audex_mac.web.chat import ChatCoordinator, RuntimeTurn
from audex_mac.web.modes import ChatMode
from audex_mac.web.server import AudexWebApplication
from audex_mac.web.store import WebChatStore


class RecordingRuntime:
    def __init__(self, output_audio: Path) -> None:
        self.output_audio = output_audio
        self.calls: list[tuple[ChatMode, str | None, Path | None]] = []

    def respond(
        self,
        *,
        mode: ChatMode,
        text: str | None,
        audio_path: Path | None,
    ) -> RuntimeTurn:
        self.calls.append((mode, text, audio_path))
        return RuntimeTurn(
            transcript=text or "A clear spoken browser fixture.",
            response_text="A colorful local response.",
            output_audio_path=(
                self.output_audio if mode.output_kind == "speech" else None
            ),
        )


class RecordingFactory:
    def __init__(self, output_audio: Path) -> None:
        self.runtime = RecordingRuntime(output_audio)

    def create(self, _chat_id: str) -> RecordingRuntime:
        return self.runtime


def _json(response) -> dict:
    return json.loads(response.body)


@pytest.mark.fast
def test_http_application_drives_chat_lifecycle_and_returns_bootstrap_state(
    tmp_path: Path,
) -> None:
    output_audio = tmp_path / "reply.wav"
    output_audio.write_bytes(b"RIFF audex reply")
    store = WebChatStore(tmp_path / "chats")
    coordinator = ChatCoordinator(
        store=store,
        runtime_factory=RecordingFactory(output_audio),
    )
    application = AudexWebApplication(
        coordinator=coordinator,
        upload_root=tmp_path / "uploads",
    )

    created = application.dispatch("POST", "/api/chats", b"{}")
    chat_id = _json(created)["chat"]["id"]
    renamed = application.dispatch(
        "PATCH",
        f"/api/chats/{chat_id}",
        json.dumps({"title": "Birdsong ideas"}).encode(),
    )
    turn = application.dispatch(
        "POST",
        f"/api/chats/{chat_id}/turns",
        json.dumps({"mode": "text-speech", "text": "Tell me what you hear"}).encode(),
    )
    bootstrap = application.dispatch("GET", "/api/bootstrap")

    assert created.status == 201
    assert _json(renamed)["chat"]["title"] == "Birdsong ideas"
    assert _json(turn)["assistant"]["transcript"] == "A colorful local response."
    assert _json(turn)["assistant"]["audio_url"].startswith(
        f"/api/chats/{chat_id}/media/"
    )
    assert _json(bootstrap)["chats"][0]["title"] == "Birdsong ideas"
    assert len(_json(bootstrap)["modes"]) == 7

    media = application.dispatch(
        "GET",
        _json(turn)["assistant"]["audio_url"],
    )
    assert media.status == 200
    assert media.body == b"RIFF audex reply"
    assert media.content_type == "audio/wav"


@pytest.mark.fast
def test_http_audio_turn_decodes_browser_wav_and_persists_visible_transcript(
    tmp_path: Path,
) -> None:
    output_audio = tmp_path / "unused.wav"
    output_audio.write_bytes(b"RIFF unused")
    factory = RecordingFactory(output_audio)
    application = AudexWebApplication(
        coordinator=ChatCoordinator(
            store=WebChatStore(tmp_path / "chats"),
            runtime_factory=factory,
        ),
        upload_root=tmp_path / "uploads",
    )
    chat_id = _json(application.dispatch("POST", "/api/chats", b"{}"))["chat"]["id"]
    wav = b"RIFF\x18\x00\x00\x00WAVEfmt browser microphone fixture"

    response = application.dispatch(
        "POST",
        f"/api/chats/{chat_id}/turns",
        json.dumps(
            {
                "mode": "speech-text",
                "audio": {
                    "name": "recording.wav",
                    "base64": base64.b64encode(wav).decode(),
                },
            }
        ).encode(),
    )

    assert response.status == 201
    assert _json(response)["user"]["transcript"] == ("A clear spoken browser fixture.")
    submitted_path = factory.runtime.calls[0][2]
    assert submitted_path is not None
    assert submitted_path.read_bytes() == wav
    assert submitted_path.parent == tmp_path / "uploads" / chat_id


@pytest.mark.fast
def test_http_application_serves_the_browser_shell_and_assets(tmp_path: Path) -> None:
    output = tmp_path / "unused.wav"
    output.write_bytes(b"RIFF")
    application = AudexWebApplication(
        coordinator=ChatCoordinator(
            store=WebChatStore(tmp_path / "chats"),
            runtime_factory=RecordingFactory(output),
        ),
        upload_root=tmp_path / "uploads",
    )

    index = application.dispatch("GET", "/")
    styles = application.dispatch("GET", "/assets/app.css")
    script = application.dispatch("GET", "/assets/app.js")
    audio_script = application.dispatch("GET", "/assets/audio.js")

    assert index.status == 200
    assert b"Choose a mode" in index.body
    assert b"chatUI" not in index.body
    assert styles.content_type == "text/css"
    assert b".message-bubble" in styles.body
    assert "javascript" in script.content_type
    assert b"createWavRecorder" in script.body
    assert b"cancelRecording" in script.body
    assert b"recordingEpoch" in script.body
    assert b"recording.cancel()" in script.body
    assert b"setSidebarOpen" in script.body
    assert b'elements.conversationPanel.setAttribute("inert", "")' in script.body
    assert b'item.setAttribute("role", "alert")' in script.body
    assert b'aria-label="Play ${escapeHtml(label)}"' in script.body
    assert b"button.title = mode.description" in script.body
    assert b"state.chat.messages = state.chat.messages.filter" in script.body
    assert b'aria-controls="sidebar"' in index.body
    assert b'aria-pressed="false"' in index.body
    assert b'id="attach-button"' in index.body
    assert b".message-group.has-assets .message-bubble" in styles.body
    assert b"width: 44px" in styles.body
    assert b"module.exports" in audio_script.body


@pytest.mark.fast
def test_http_application_returns_structured_model_errors(tmp_path: Path) -> None:
    output = tmp_path / "unused.wav"
    output.write_bytes(b"RIFF")
    coordinator = ChatCoordinator(
        store=WebChatStore(tmp_path / "chats"),
        runtime_factory=RecordingFactory(output),
    )
    application = AudexWebApplication(
        coordinator=coordinator,
        upload_root=tmp_path / "uploads",
    )
    chat_id = _json(application.dispatch("POST", "/api/chats", b"{}"))["chat"]["id"]

    def explode(*_args, **_kwargs):
        raise RuntimeError("local model ran out of memory")

    coordinator.submit = explode
    response = application.dispatch(
        "POST",
        f"/api/chats/{chat_id}/turns",
        json.dumps({"mode": "text-text", "text": "Hello"}).encode(),
    )

    assert response.status == 500
    assert _json(response)["error"] == ("RuntimeError: local model ran out of memory")


@pytest.mark.fast
@pytest.mark.parametrize(
    ("method", "path", "body", "status", "message"),
    [
        ("GET", "/api/nope", b"", 404, "not found"),
        ("POST", "/api/chats", b"{bad json", 400, "valid JSON"),
        (
            "POST",
            "/api/chats/missing/turns",
            b"{}",
            400,
            "valid mode",
        ),
    ],
)
def test_http_application_returns_structured_client_errors(
    tmp_path: Path,
    method: str,
    path: str,
    body: bytes,
    status: int,
    message: str,
) -> None:
    output = tmp_path / "unused.wav"
    output.write_bytes(b"RIFF")
    application = AudexWebApplication(
        coordinator=ChatCoordinator(
            store=WebChatStore(tmp_path / "chats"),
            runtime_factory=RecordingFactory(output),
        ),
        upload_root=tmp_path / "uploads",
    )

    response = application.dispatch(method, path, body)

    assert response.status == status
    assert message in _json(response)["error"]
