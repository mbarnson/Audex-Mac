"""Cache-preserving orchestration for Audex browser conversations."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .modes import ChatMode
from .store import WebChat, WebChatStore, WebMessage


@dataclass(frozen=True, slots=True)
class RuntimeTurn:
    transcript: str
    response_text: str
    output_audio_path: Path | None = None
    assets: tuple[dict[str, Any], ...] = ()


class ConversationRuntime(Protocol):
    def respond(
        self,
        *,
        mode: ChatMode,
        text: str | None,
        audio_path: Path | None,
    ) -> RuntimeTurn: ...


class RuntimeFactory(Protocol):
    def create(self, chat_id: str) -> ConversationRuntime: ...


@dataclass(frozen=True, slots=True)
class ChatTurn:
    chat: WebChat
    user: WebMessage
    assistant: WebMessage

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat": self.chat.to_dict(),
            "user": self.user.to_dict(),
            "assistant": self.assistant.to_dict(),
        }


class ChatCoordinator:
    """Own one warm model runtime per chat, regardless of selected mode."""

    def __init__(self, *, store: WebChatStore, runtime_factory: RuntimeFactory) -> None:
        self.store = store
        self.runtime_factory = runtime_factory
        self._runtimes: dict[str, ConversationRuntime] = {}
        self._runtime_lock = threading.Lock()
        self._chat_locks: dict[str, threading.Lock] = {}

    def create_chat(self, *, mode: ChatMode = ChatMode.TEXT_TEXT) -> WebChat:
        return self.store.create(mode=mode)

    def rename_chat(self, chat_id: str, title: str) -> WebChat:
        with self._lock_for(chat_id):
            return self.store.rename(chat_id, title)

    def submit(
        self,
        chat_id: str,
        *,
        mode: ChatMode,
        text: str | None = None,
        audio_path: Path | None = None,
    ) -> ChatTurn:
        normalized = text.strip() if text is not None else None
        if mode.input_kind == "text" and not normalized:
            raise ValueError(f"{mode.spec.label} requires text input")
        if mode.input_kind != "text" and audio_path is None:
            raise ValueError(f"{mode.spec.label} requires audio input")
        if (
            mode.input_kind != "text"
            and audio_path is not None
            and not audio_path.is_file()
        ):
            raise FileNotFoundError(f"Audex input audio does not exist: {audio_path}")

        with self._lock_for(chat_id):
            chat = self.store.load(chat_id)
            runtime = self._runtime_for(chat_id)
            result = runtime.respond(
                mode=mode,
                text=normalized,
                audio_path=audio_path,
            )
            now = _now()
            user_id = uuid.uuid4().hex
            assistant_id = uuid.uuid4().hex
            assets = []
            for index, raw_asset in enumerate(result.assets):
                asset = dict(raw_asset)
                if asset.get("audio_path"):
                    asset["audio_path"] = str(asset["audio_path"])
                    asset["audio_url"] = (
                        f"/api/chats/{chat_id}/media/{assistant_id}/assets/{index}"
                    )
                assets.append(asset)
            user = WebMessage(
                message_id=user_id,
                role="user",
                transcript=result.transcript.strip(),
                mode=mode,
                created_at=now,
                audio_path=str(audio_path) if audio_path is not None else None,
                audio_url=(
                    f"/api/chats/{chat_id}/media/{user_id}"
                    if audio_path is not None
                    else None
                ),
                kind="sound-prompt" if mode.input_kind == "audio" else "message",
            )
            assistant = WebMessage(
                message_id=assistant_id,
                role="assistant",
                transcript=result.response_text.strip(),
                mode=mode,
                created_at=now,
                audio_path=(
                    str(result.output_audio_path)
                    if result.output_audio_path is not None
                    else None
                ),
                audio_url=(
                    f"/api/chats/{chat_id}/media/{assistant_id}"
                    if result.output_audio_path is not None
                    else None
                ),
                kind="sound-result" if mode.output_kind == "audio" else "message",
                assets=assets,
            )
            chat.current_mode = mode
            chat.messages.extend((user, assistant))
            self.store.save(chat)
            return ChatTurn(chat=chat, user=user, assistant=assistant)

    def media_path(
        self,
        chat_id: str,
        message_id: str,
        *,
        asset_index: int | None = None,
    ) -> Path:
        chat = self.store.load(chat_id)
        for message in chat.messages:
            if message.message_id != message_id:
                continue
            raw_path = message.audio_path
            if asset_index is not None:
                if not 0 <= asset_index < len(message.assets):
                    raise KeyError(f"unknown Audex sound asset: {asset_index}")
                raw_path = message.assets[asset_index].get("audio_path")
            if raw_path:
                path = Path(str(raw_path))
                if path.is_file():
                    return path
                raise FileNotFoundError(f"Audex media is missing: {path}")
        raise KeyError(f"unknown Audex chat media: {message_id}")

    def _runtime_for(self, chat_id: str) -> ConversationRuntime:
        with self._runtime_lock:
            runtime = self._runtimes.get(chat_id)
            if runtime is None:
                runtime = self.runtime_factory.create(chat_id)
                self._runtimes[chat_id] = runtime
            return runtime

    def _lock_for(self, chat_id: str) -> threading.Lock:
        with self._runtime_lock:
            return self._chat_locks.setdefault(chat_id, threading.Lock())


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
