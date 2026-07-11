from __future__ import annotations

from pathlib import Path

import pytest

from audex_mac.web.chat import ChatCoordinator, RuntimeTurn
from audex_mac.web.modes import ChatMode, mode_catalog
from audex_mac.web.store import WebChatStore


class FakeConversationRuntime:
    def __init__(self, chat_id: str) -> None:
        self.chat_id = chat_id
        self.calls: list[tuple[ChatMode, str | None, Path | None]] = []

    def respond(
        self,
        *,
        mode: ChatMode,
        text: str | None,
        audio_path: Path | None,
    ) -> RuntimeTurn:
        self.calls.append((mode, text, audio_path))
        transcript = text or f"Transcript of {audio_path.name}"
        return RuntimeTurn(
            transcript=transcript,
            response_text=f"Audex response {len(self.calls)}",
            output_audio_path=(
                Path(f"/tmp/response-{len(self.calls)}.wav")
                if mode.output_kind in {"speech", "audio"}
                else None
            ),
        )


class FakeRuntimeFactory:
    def __init__(self) -> None:
        self.created: list[FakeConversationRuntime] = []

    def create(self, chat_id: str) -> FakeConversationRuntime:
        runtime = FakeConversationRuntime(chat_id)
        self.created.append(runtime)
        return runtime


@pytest.mark.fast
def test_conversational_mode_switches_reuse_one_runtime_and_keep_transcripts(
    tmp_path: Path,
) -> None:
    store = WebChatStore(tmp_path / "web-chats")
    factory = FakeRuntimeFactory()
    coordinator = ChatCoordinator(store=store, runtime_factory=factory)
    chat = coordinator.create_chat()
    speech = tmp_path / "spoken.wav"
    speech.write_bytes(b"RIFF fixture")

    first = coordinator.submit(
        chat.chat_id,
        mode=ChatMode.TEXT_SPEECH,
        text="Read this aloud",
    )
    second = coordinator.submit(
        chat.chat_id,
        mode=ChatMode.SPEECH_SPEECH,
        audio_path=speech,
    )
    third = coordinator.submit(
        chat.chat_id,
        mode=ChatMode.SPEECH_TEXT,
        audio_path=speech,
    )
    fourth = coordinator.submit(
        chat.chat_id,
        mode=ChatMode.TEXT_TEXT,
        text="Now answer silently",
    )

    assert len(factory.created) == 1
    assert [call[0] for call in factory.created[0].calls] == [
        ChatMode.TEXT_SPEECH,
        ChatMode.SPEECH_SPEECH,
        ChatMode.SPEECH_TEXT,
        ChatMode.TEXT_TEXT,
    ]
    assert first.user.transcript == "Read this aloud"
    assert second.user.transcript == "Transcript of spoken.wav"
    assert third.user.transcript == "Transcript of spoken.wav"
    assert fourth.user.transcript == "Now answer silently"
    assert first.assistant.audio_url is not None
    assert third.assistant.audio_url is None
    reloaded = store.load(chat.chat_id)
    assert reloaded.current_mode is ChatMode.TEXT_TEXT
    assert [
        message.transcript for message in reloaded.messages if message.role == "user"
    ] == [
        "Read this aloud",
        "Transcript of spoken.wav",
        "Transcript of spoken.wav",
        "Now answer silently",
    ]


@pytest.mark.fast
def test_mode_catalog_exposes_all_requested_input_output_combinations() -> None:
    modes = {item["id"]: item for item in mode_catalog()}

    assert set(modes) == {
        "text-text",
        "text-speech",
        "speech-text",
        "speech-speech",
        "audio-text",
        "text-audio",
        "audio-audio",
    }
    assert modes["audio-text"]["description"]
    assert modes["text-audio"]["label"] == "Text in, Audio out"
    assert modes["text-speech"]["label"] == "Text in, Speech out"
    assert modes["speech-speech"]["preserves_conversation_cache"] is True
    assert modes["text-audio"]["preserves_conversation_cache"] is False


@pytest.mark.fast
def test_chat_titles_are_audex_named_editable_and_persisted(tmp_path: Path) -> None:
    store = WebChatStore(tmp_path / "web-chats")
    coordinator = ChatCoordinator(store=store, runtime_factory=FakeRuntimeFactory())

    first = coordinator.create_chat()
    second = coordinator.create_chat()
    renamed = coordinator.rename_chat(first.chat_id, "Field recordings")

    assert first.title.startswith("Audex ")
    assert second.title.startswith("Audex ")
    assert first.title != second.title
    assert renamed.title == "Field recordings"
    assert [chat.title for chat in store.list_chats()] == [
        "Field recordings",
        second.title,
    ]


@pytest.mark.fast
def test_submit_rejects_payloads_that_do_not_match_the_selected_mode(
    tmp_path: Path,
) -> None:
    coordinator = ChatCoordinator(
        store=WebChatStore(tmp_path / "web-chats"),
        runtime_factory=FakeRuntimeFactory(),
    )
    chat = coordinator.create_chat()

    with pytest.raises(ValueError, match="requires text input"):
        coordinator.submit(chat.chat_id, mode=ChatMode.TEXT_TEXT)
    with pytest.raises(ValueError, match="requires audio input"):
        coordinator.submit(
            chat.chat_id,
            mode=ChatMode.SPEECH_TEXT,
            text="not an audio recording",
        )


@pytest.mark.fast
def test_generated_sound_assets_receive_chat_scoped_playback_urls(
    tmp_path: Path,
) -> None:
    sound = tmp_path / "rain.wav"
    sound.write_bytes(b"RIFF generated sound")

    class AssetRuntime(FakeConversationRuntime):
        def respond(self, **_kwargs) -> RuntimeTurn:
            return RuntimeTurn(
                transcript="Generate rain",
                response_text="One sound is ready.",
                assets=(
                    {
                        "label": "Rain",
                        "caption": "Rain falls on leaves.",
                        "audio_path": str(sound),
                    },
                ),
            )

    class AssetFactory:
        def create(self, chat_id: str) -> AssetRuntime:
            return AssetRuntime(chat_id)

    coordinator = ChatCoordinator(
        store=WebChatStore(tmp_path / "chats"),
        runtime_factory=AssetFactory(),
    )
    chat = coordinator.create_chat()

    turn = coordinator.submit(
        chat.chat_id,
        mode=ChatMode.TEXT_AUDIO,
        text="Generate rain",
    )

    asset = turn.assistant.assets[0]
    assert asset["caption"] == "Rain falls on leaves."
    assert asset["audio_url"].endswith(f"/{turn.assistant.message_id}/assets/0")
    assert (
        coordinator.media_path(
            chat.chat_id,
            turn.assistant.message_id,
            asset_index=0,
        )
        == sound
    )
