"""Durable browser chat metadata, separate from model conversation state."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .modes import ChatMode

DEFAULT_WEB_CHAT_ROOT = Path(".audex/web/chats")
_TITLE_WORDS = (
    "Aurora",
    "Echo",
    "Prism",
    "Sonata",
    "Pulse",
    "Lumen",
    "Chorus",
    "Orbit",
    "Cadence",
    "Ripple",
    "Spectrum",
    "Resonance",
)


@dataclass(slots=True)
class WebMessage:
    message_id: str
    role: str
    transcript: str
    mode: ChatMode
    created_at: str
    audio_path: str | None = None
    audio_url: str | None = None
    kind: str = "message"
    assets: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


@dataclass(slots=True)
class WebChat:
    chat_id: str
    title: str
    created_at: str
    updated_at: str
    current_mode: ChatMode
    messages: list[WebMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chat_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_mode": self.current_mode.value,
            "mode_icon": self.current_mode.spec.icon,
            "messages": [message.to_dict() for message in self.messages],
        }


class WebChatStore:
    def __init__(self, root: Path = DEFAULT_WEB_CHAT_ROOT) -> None:
        self.root = Path(root)

    def create(self, *, mode: ChatMode = ChatMode.TEXT_TEXT) -> WebChat:
        self.root.mkdir(parents=True, exist_ok=True)
        existing = len(tuple(self.root.glob("*.json")))
        now = _now()
        chat = WebChat(
            chat_id=f"chat-{time.time_ns():x}-{uuid.uuid4().hex[:6]}",
            title=f"Audex {_TITLE_WORDS[existing % len(_TITLE_WORDS)]}",
            created_at=now,
            updated_at=now,
            current_mode=mode,
        )
        self.save(chat)
        return chat

    def load(self, chat_id: str) -> WebChat:
        path = self._path(chat_id)
        if not path.is_file():
            raise KeyError(f"unknown Audex web chat: {chat_id}")
        return _chat_from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_chats(self) -> list[WebChat]:
        chats = (
            [
                _chat_from_dict(json.loads(path.read_text(encoding="utf-8")))
                for path in self.root.glob("*.json")
            ]
            if self.root.is_dir()
            else []
        )
        return sorted(chats, key=lambda chat: (chat.created_at, chat.chat_id))

    def save(self, chat: WebChat) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        chat.updated_at = _now()
        path = self._path(chat.chat_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(chat.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def rename(self, chat_id: str, title: str) -> WebChat:
        normalized = " ".join(title.split())
        if not normalized:
            raise ValueError("Chat title must not be empty")
        if len(normalized) > 80:
            raise ValueError("Chat title must be at most 80 characters")
        chat = self.load(chat_id)
        chat.title = normalized
        self.save(chat)
        return chat

    def _path(self, chat_id: str) -> Path:
        if not chat_id.startswith("chat-") or any(
            part in chat_id for part in ("/", "\\", "..")
        ):
            raise ValueError(f"invalid Audex web chat id: {chat_id}")
        return self.root / f"{chat_id}.json"


def _chat_from_dict(payload: dict[str, Any]) -> WebChat:
    messages = [
        WebMessage(
            message_id=str(item["message_id"]),
            role=str(item["role"]),
            transcript=str(item.get("transcript", "")),
            mode=ChatMode(str(item["mode"])),
            created_at=str(item["created_at"]),
            audio_path=(str(item["audio_path"]) if item.get("audio_path") else None),
            audio_url=(str(item["audio_url"]) if item.get("audio_url") else None),
            kind=str(item.get("kind", "message")),
            assets=list(item.get("assets", [])),
        )
        for item in payload.get("messages", [])
    ]
    return WebChat(
        chat_id=str(payload["id"]),
        title=str(payload["title"]),
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
        current_mode=ChatMode(str(payload["current_mode"])),
        messages=messages,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
