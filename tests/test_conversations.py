from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.conversations import (
    DEFAULT_DEMO_CONTEXT_TOKENS,
    ConversationStore,
    infer_user_name,
)

pytestmark = pytest.mark.fast


def test_conversation_store_creates_text_and_json_logs(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)

    conversation = store.create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System prompt.",
        max_context_tokens=1_000_000,
    )
    conversation.messages.extend(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there."},
        ]
    )
    conversation.token_count = 42
    store.save(conversation)

    assert store.current_id() == conversation.conversation_id
    assert conversation.json_path.is_file()
    assert conversation.transcript_path.is_file()
    payload = json.loads(conversation.json_path.read_text(encoding="utf-8"))
    assert payload["token_count"] == 42
    transcript = conversation.transcript_path.read_text(encoding="utf-8")
    assert "Context tokens: 42/1000000" in transcript
    assert "## User\n\nHello" in transcript
    assert "## Assistant\n\nHi there." in transcript


def test_conversation_store_can_use_browser_chat_id_as_cache_identity(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "conversations")

    conversation = store.create(
        conversation_id="chat-browser-123",
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System",
    )

    assert conversation.conversation_id == "chat-browser-123"
    assert store.load("chat-browser-123").conversation_id == "chat-browser-123"
    with pytest.raises(ValueError, match="Invalid conversation id"):
        store.create(
            conversation_id="../escape",
            persona_id="assistant",
            persona_path=tmp_path / "assistant.md",
            system_prompt="System",
        )


def test_conversation_store_resumes_current_by_default(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    created = store.create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System prompt.",
    )

    resumed, did_resume = store.resume_current_or_create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="Different prompt.",
    )

    assert did_resume is True
    assert resumed.conversation_id == created.conversation_id
    assert resumed.messages == created.messages
    assert resumed.max_context_tokens == DEFAULT_DEMO_CONTEXT_TOKENS


def test_resuming_migrates_old_million_token_policy_to_demo_limit(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    created = store.create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System prompt.",
        max_context_tokens=1_000_000,
    )

    resumed, did_resume = store.resume_current_or_create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System prompt.",
    )

    assert did_resume is True
    assert resumed.conversation_id == created.conversation_id
    assert resumed.max_context_tokens == DEFAULT_DEMO_CONTEXT_TOKENS
    payload = json.loads(resumed.json_path.read_text(encoding="utf-8"))
    assert payload["max_context_tokens"] == DEFAULT_DEMO_CONTEXT_TOKENS


def test_conversation_store_persists_inferred_user_name(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    conversation = store.create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System prompt.",
    )
    conversation.messages.append(
        {"role": "user", "content": "My name is pat barnson, let's talk."}
    )

    store.save(conversation)

    reloaded = store.load(conversation.conversation_id)
    assert reloaded.user_name == "Pat Barnson"
    payload = json.loads(conversation.json_path.read_text(encoding="utf-8"))
    assert payload["user_name"] == "Pat Barnson"


def test_infer_user_name_recognizes_spoken_identity_phrases() -> None:
    assert (
        infer_user_name([{"role": "user", "content": "You can call me Alex."}])
        == "Alex"
    )
    assert infer_user_name([{"role": "user", "content": "I'm Sam!"}]) == "Sam"


def test_latest_explicit_identity_replaces_stale_user_name(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    conversation = store.create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System prompt.",
    )
    conversation.user_name = "Wrong Benchmark Words"
    conversation.messages.extend(
        [
            {"role": "user", "content": "Call me Old Name."},
            {"role": "assistant", "content": "Hello."},
            {
                "role": "user",
                "content": "Call me Matt and let's talk about Gilroy.",
            },
        ]
    )

    store.save(conversation)

    assert conversation.user_name == "Matt"
    assert store.load(conversation.conversation_id).user_name == "Matt"
