from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.web.chat import ChatCoordinator, RuntimeTurn
from audex_mac.web.live import LiveTurnHandler, decode_pcm_frame
from audex_mac.web.modes import ChatMode
from audex_mac.web.server import AudexWebApplication
from audex_mac.web.store import WebChatStore

pytestmark = pytest.mark.fast


class StreamingRuntime:
    def __init__(self, output: Path) -> None:
        self.output = output

    def respond(self, *, mode, text, audio_path, stream=None) -> RuntimeTurn:
        assert mode is ChatMode.TEXT_SPEECH
        assert text == "Hello streaming"
        assert audio_path is None
        assert stream is not None
        stream.assistant_text("Hello")
        stream.assistant_pcm(24_000, b"\x01\x00\xff\xff")
        stream.assistant_text("Hello back")
        stream.assistant_pcm(24_000, b"\x02\x00\xfe\xff")
        return RuntimeTurn(
            transcript=text,
            response_text="Hello back",
            output_audio_path=self.output,
        )


class StreamingFactory:
    def __init__(self, output: Path) -> None:
        self.output = output

    def create(self, _chat_id: str) -> StreamingRuntime:
        return StreamingRuntime(self.output)


class FakeConnection:
    def __init__(self, request: dict[str, object]) -> None:
        self.request = request
        self.sent: list[str | bytes] = []

    def recv(self):
        return json.dumps(self.request)

    def send(self, message) -> None:
        self.sent.append(message)


def _events(connection: FakeConnection) -> list[dict[str, object]]:
    return [json.loads(item) for item in connection.sent if isinstance(item, str)]


def test_live_turn_streams_text_and_pcm_before_completed_replay_message(
    tmp_path: Path,
) -> None:
    output = tmp_path / "response.wav"
    output.write_bytes(b"RIFF response")
    coordinator = ChatCoordinator(
        store=WebChatStore(tmp_path / "chats"),
        runtime_factory=StreamingFactory(output),
    )
    chat = coordinator.create_chat()
    application = AudexWebApplication(
        coordinator=coordinator,
        upload_root=tmp_path / "uploads",
    )
    connection = FakeConnection(
        {
            "chat_id": chat.chat_id,
            "mode": "text-speech",
            "text": "Hello streaming",
        }
    )

    LiveTurnHandler(application).handle(connection)

    events = _events(connection)
    assert [event["type"] for event in events] == [
        "turn.started",
        "assistant.text.delta",
        "assistant.audio.started",
        "assistant.text.delta",
        "assistant.audio.finished",
        "user.transcript.final",
        "turn.finished",
    ]
    binary = [item for item in connection.sent if isinstance(item, bytes)]
    assert [decode_pcm_frame(item)[1:] for item in binary] == [
        (0, b"\x01\x00\xff\xff"),
        (1, b"\x02\x00\xfe\xff"),
    ]
    turn_id = events[0]["turn_id"]
    assert all(decode_pcm_frame(item)[0] == turn_id for item in binary)
    assert events[2]["sample_rate"] == 24_000
    assert "audio_url" not in events[1]
    assistant = events[-1]["turn"]["assistant"]
    assert assistant["transcript"] == "Hello back"
    assert assistant["audio_url"].endswith(f"/{assistant['message_id']}")
    assert coordinator.store.load(chat.chat_id).messages[-1].transcript == "Hello back"
