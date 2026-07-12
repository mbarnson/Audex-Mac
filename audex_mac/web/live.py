"""Live conversational turn transport for incremental browser speech playback."""

from __future__ import annotations

import json
import struct
import uuid
from contextlib import suppress
from typing import Any, Protocol

from .chat import ChatTurn, TurnStream
from .server import AudexWebApplication

_PCM_HEADER = struct.Struct("<4s16sI")
_PCM_MAGIC = b"APCM"


class LiveConnection(Protocol):
    def recv(self) -> str | bytes: ...

    def send(self, message: str | bytes) -> None: ...


def encode_pcm_frame(turn_id: str, sequence: int, pcm: bytes) -> bytes:
    return _PCM_HEADER.pack(_PCM_MAGIC, uuid.UUID(hex=turn_id).bytes, sequence) + pcm


def decode_pcm_frame(frame: bytes) -> tuple[str, int, bytes]:
    if len(frame) < _PCM_HEADER.size:
        raise ValueError("Audex PCM frame is shorter than its header.")
    magic, turn_bytes, sequence = _PCM_HEADER.unpack_from(frame)
    if magic != _PCM_MAGIC:
        raise ValueError("Audex PCM frame has an invalid marker.")
    return uuid.UUID(bytes=turn_bytes).hex, sequence, frame[_PCM_HEADER.size :]


class WebSocketTurnStream(TurnStream):
    """Translate one model turn into ordered WebSocket control and PCM frames."""

    def __init__(self, connection: LiveConnection, *, turn_id: str) -> None:
        self.connection = connection
        self.turn_id = turn_id
        self._audio_started = False
        self._sample_rate: int | None = None
        self._sequence = 0
        self._last_text = ""

    def start(self) -> None:
        self._event("turn.started")

    def assistant_text(self, text: str) -> None:
        if text == self._last_text:
            return
        self._last_text = text
        self._event("assistant.text.delta", text=text)

    def assistant_pcm(self, sample_rate: int, pcm: bytes) -> None:
        if not pcm:
            return
        if not self._audio_started:
            self._audio_started = True
            self._sample_rate = int(sample_rate)
            self._event(
                "assistant.audio.started",
                sample_rate=self._sample_rate,
                channels=1,
                sample_format="s16le",
            )
        elif sample_rate != self._sample_rate:
            raise ValueError(
                "Audex speech stream changed sample rate during one turn: "
                f"{self._sample_rate} to {sample_rate}."
            )
        self.connection.send(encode_pcm_frame(self.turn_id, self._sequence, pcm))
        self._sequence += 1

    def finish(self, turn: ChatTurn) -> None:
        if self._audio_started:
            self._event(
                "assistant.audio.finished",
                chunks=self._sequence,
                sample_rate=self._sample_rate,
            )
        self._event("user.transcript.final", text=turn.user.transcript)
        self._event("turn.finished", turn=turn.to_dict())

    def fail(self, error: BaseException) -> None:
        self._event(
            "turn.failed",
            error=f"{type(error).__name__}: {error}",
        )

    def _event(self, event_type: str, **payload: Any) -> None:
        self.connection.send(
            json.dumps(
                {"type": event_type, "turn_id": self.turn_id, **payload},
                separators=(",", ":"),
            )
        )


class LiveTurnHandler:
    """Run one submitted browser turn over one WebSocket connection."""

    def __init__(self, application: AudexWebApplication) -> None:
        self.application = application

    def handle(self, connection: LiveConnection) -> None:
        stream = WebSocketTurnStream(connection, turn_id=uuid.uuid4().hex)
        try:
            raw_request = connection.recv()
            if not isinstance(raw_request, str):
                raise ValueError("Audex live turn request must be JSON text.")
            payload = json.loads(raw_request)
            if not isinstance(payload, dict):
                raise ValueError("Audex live turn request must be a JSON object.")
            chat_id = str(payload.get("chat_id", "")).strip()
            if not chat_id:
                raise ValueError("Audex live turn request requires chat_id.")
            stream.start()
            turn = self.application.submit_turn_payload(
                chat_id,
                payload,
                stream=stream,
            )
            stream.finish(turn)
        except Exception as exc:
            with suppress(Exception):
                stream.fail(exc)
