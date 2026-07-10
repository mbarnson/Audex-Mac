"""Persistent text conversations for the Audex speech CLI."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .speech_output import RUNS_DIR

CONVERSATIONS_DIR = RUNS_DIR.parent / "conversations"
CURRENT_CONVERSATION_PATH = CONVERSATIONS_DIR / "current"
CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_DEMO_CONTEXT_TOKENS = 262_144


@dataclass(slots=True)
class Conversation:
    root: Path
    conversation_id: str
    created_at: str
    updated_at: str
    persona_id: str
    persona_path: str
    messages: list[dict[str, str]]
    token_count: int | None = None
    max_context_tokens: int = DEFAULT_DEMO_CONTEXT_TOKENS
    user_name: str | None = None

    @property
    def json_path(self) -> Path:
        return self.root / f"{self.conversation_id}.json"

    @property
    def transcript_path(self) -> Path:
        return self.root / f"{self.conversation_id}.md"


class ConversationStore:
    def __init__(self, root: Path = CONVERSATIONS_DIR) -> None:
        self.root = root
        self.current_path = root / "current"

    def create(
        self,
        *,
        persona_id: str,
        persona_path: Path,
        system_prompt: str,
        max_context_tokens: int = DEFAULT_DEMO_CONTEXT_TOKENS,
    ) -> Conversation:
        now = _utc_timestamp()
        conversation = Conversation(
            root=self.root,
            conversation_id=_new_conversation_id(),
            created_at=now,
            updated_at=now,
            persona_id=persona_id,
            persona_path=str(persona_path),
            messages=[{"role": "system", "content": system_prompt}],
            max_context_tokens=max_context_tokens,
        )
        self.save(conversation)
        self.set_current(conversation.conversation_id)
        return conversation

    def load(self, conversation_id: str) -> Conversation:
        if not CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
            raise ValueError(f"Invalid conversation id: {conversation_id}")
        path = self.root / f"{conversation_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Conversation not found: {conversation_id}")
        return _conversation_from_dict(
            json.loads(path.read_text(encoding="utf-8")),
            root=self.root,
        )

    def current_id(self) -> str | None:
        if not self.current_path.is_file():
            return None
        conversation_id = self.current_path.read_text(encoding="utf-8").strip()
        return conversation_id or None

    def resume_current_or_create(
        self,
        *,
        persona_id: str,
        persona_path: Path,
        system_prompt: str,
        max_context_tokens: int = DEFAULT_DEMO_CONTEXT_TOKENS,
    ) -> tuple[Conversation, bool]:
        conversation_id = self.current_id()
        if conversation_id is not None:
            conversation = self.load(conversation_id)
            if conversation.max_context_tokens != max_context_tokens:
                conversation.max_context_tokens = max_context_tokens
                self.save(conversation)
            return conversation, True
        return (
            self.create(
                persona_id=persona_id,
                persona_path=persona_path,
                system_prompt=system_prompt,
                max_context_tokens=max_context_tokens,
            ),
            False,
        )

    def save(self, conversation: Conversation) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        conversation.updated_at = _utc_timestamp()
        conversation.user_name = (
            infer_user_name(conversation.messages) or conversation.user_name
        )
        payload = {
            "id": conversation.conversation_id,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "persona_id": conversation.persona_id,
            "persona_path": conversation.persona_path,
            "token_count": conversation.token_count,
            "max_context_tokens": conversation.max_context_tokens,
            "user_name": conversation.user_name,
            "messages": conversation.messages,
        }
        conversation.json_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        conversation.transcript_path.write_text(
            render_transcript(conversation),
            encoding="utf-8",
        )

    def set_current(self, conversation_id: str) -> None:
        if not CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
            raise ValueError(f"Invalid conversation id: {conversation_id}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_path.write_text(f"{conversation_id}\n", encoding="utf-8")


def render_transcript(conversation: Conversation) -> str:
    lines = [
        f"# Audex Conversation {conversation.conversation_id}",
        "",
        f"Persona: {conversation.persona_id}",
        f"Created: {conversation.created_at}",
        f"Updated: {conversation.updated_at}",
        f"Context tokens: {_format_token_count(conversation)}",
        "",
    ]
    for message in conversation.messages:
        role = message.get("role", "")
        if role == "system":
            continue
        heading = "User" if role == "user" else "Assistant"
        lines.extend([f"## {heading}", "", message.get("content", "").strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _conversation_from_dict(payload: dict[str, Any], *, root: Path) -> Conversation:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Conversation file has no messages.")
    return Conversation(
        root=root,
        conversation_id=str(payload["id"]),
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
        persona_id=str(payload["persona_id"]),
        persona_path=str(payload["persona_path"]),
        messages=[
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ],
        token_count=(
            int(payload["token_count"])
            if payload.get("token_count") is not None
            else None
        ),
        max_context_tokens=int(
            payload.get("max_context_tokens", DEFAULT_DEMO_CONTEXT_TOKENS)
        ),
        user_name=(
            infer_user_name(
                [
                    {"role": str(message["role"]), "content": str(message["content"])}
                    for message in messages
                ]
            )
            or (
                str(payload["user_name"]).strip()
                if payload.get("user_name") is not None
                else None
            )
        ),
    )


def infer_user_name(messages: list[dict[str, str]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        name = _infer_user_name_from_text(message.get("content", ""))
        if name is not None:
            return name
    return None


def _infer_user_name_from_text(text: str) -> str | None:
    normalized = " ".join(text.strip().split())
    patterns = (
        r"\bmy name is\s+([A-Za-z][A-Za-z .'-]{0,40})",
        r"\bcall me\s+([A-Za-z][A-Za-z .'-]{0,40})",
        r"\byou can call me\s+([A-Za-z][A-Za-z .'-]{0,40})",
        r"\bi am\s+([A-Za-z][A-Za-z .'-]{0,40})",
        r"\bi'm\s+([A-Za-z][A-Za-z .'-]{0,40})",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match is None:
            continue
        return _clean_user_name(match.group(1))
    return None


def _clean_user_name(raw_name: str) -> str | None:
    name = re.split(
        r"[,.!?;:]|\s+(?:and|but|so|because)\b",
        raw_name.strip(),
        maxsplit=1,
    )[0]
    name = " ".join(part.capitalize() for part in name.split())
    if not name or len(name) > 40:
        return None
    return name


def _format_token_count(conversation: Conversation) -> str:
    if conversation.token_count is None:
        return f"unknown/{conversation.max_context_tokens}"
    return f"{conversation.token_count}/{conversation.max_context_tokens}"


def _new_conversation_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
