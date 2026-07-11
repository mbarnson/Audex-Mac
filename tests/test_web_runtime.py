from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from audex_mac.conversations import ConversationStore
from audex_mac.web.modes import ChatMode
from audex_mac.web.runtime import (
    AudexConversationRuntime,
    GeneratedAudioAsset,
    SharedAudexRuntimeFactory,
    SoundLabWebBackend,
)


class FakeAudexSession:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[str, object]] = []

    def run_text_only_turn_from_text(self, *, user_text: str):
        self.calls.append(("text-text", user_text))
        return SimpleNamespace(transcript=user_text, response_text="Written answer")

    def run_turn_from_text(self, *, user_text: str, play: bool):
        self.calls.append(("text-speech", (user_text, play)))
        return SimpleNamespace(
            transcript=user_text,
            response_text="Spoken answer",
            output_wav_path=self.root / "speech.wav",
        )

    def run_text_only_turn_from_wav(self, *, input_wav_path: Path):
        self.calls.append(("speech-text", input_wav_path))
        return SimpleNamespace(
            transcript="Spoken question", response_text="Written answer"
        )

    def run_turn_from_wav(self, *, input_wav_path: Path, play: bool):
        self.calls.append(("speech-speech", (input_wav_path, play)))
        return SimpleNamespace(
            transcript="Spoken question",
            response_text="Spoken answer",
            output_wav_path=self.root / "speech.wav",
        )

    def understand_audio(self, *, input_wav_path: Path, prompt: str):
        self.calls.append(("understand", (input_wav_path, prompt)))
        return SimpleNamespace(
            transcript=prompt,
            response_text="Soft rain on a metal roof",
        )


class FakeSoundBackend:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.prompts: list[str] = []

    def generate(self, prompt: str):
        self.prompts.append(prompt)
        return (
            "I generated two playable variations.",
            (
                GeneratedAudioAsset(
                    label="Rain 1",
                    caption="Gentle rain on a corrugated metal roof.",
                    audio_path=self.root / "rain-one.wav",
                ),
                GeneratedAudioAsset(
                    label="Rain 2",
                    caption="Heavy rain rattles a distant metal awning.",
                    audio_path=self.root / "rain-two.wav",
                ),
            ),
        )


@pytest.mark.fast
def test_runtime_routes_four_conversation_modes_without_replacing_session(
    tmp_path: Path,
) -> None:
    session = FakeAudexSession(tmp_path)
    runtime = AudexConversationRuntime(
        session=session,
        sound_backend=FakeSoundBackend(tmp_path),
    )
    input_wav = tmp_path / "input.wav"

    text_text = runtime.respond(
        mode=ChatMode.TEXT_TEXT, text="Type this", audio_path=None
    )
    text_speech = runtime.respond(
        mode=ChatMode.TEXT_SPEECH, text="Read this", audio_path=None
    )
    speech_text = runtime.respond(
        mode=ChatMode.SPEECH_TEXT, text=None, audio_path=input_wav
    )
    speech_speech = runtime.respond(
        mode=ChatMode.SPEECH_SPEECH, text=None, audio_path=input_wav
    )

    assert [name for name, _payload in session.calls] == [
        "text-text",
        "text-speech",
        "speech-text",
        "speech-speech",
    ]
    assert text_text.output_audio_path is None
    assert speech_text.output_audio_path is None
    assert text_speech.output_audio_path == tmp_path / "speech.wav"
    assert speech_speech.output_audio_path == tmp_path / "speech.wav"
    assert speech_speech.transcript == "Spoken question"
    assert all(call[1][1] is False for call in (session.calls[1], session.calls[3]))


@pytest.mark.fast
def test_runtime_routes_understanding_and_unblinded_sound_generation(
    tmp_path: Path,
) -> None:
    session = FakeAudexSession(tmp_path)
    sounds = FakeSoundBackend(tmp_path)
    runtime = AudexConversationRuntime(session=session, sound_backend=sounds)
    input_wav = tmp_path / "reference.wav"

    understood = runtime.respond(
        mode=ChatMode.AUDIO_TEXT,
        text="What is happening in this recording?",
        audio_path=input_wav,
    )
    generated = runtime.respond(
        mode=ChatMode.TEXT_AUDIO,
        text="Rain on a metal roof",
        audio_path=None,
    )
    continued = runtime.respond(
        mode=ChatMode.AUDIO_AUDIO,
        text=None,
        audio_path=input_wav,
    )

    assert understood.transcript == "What is happening in this recording?"
    assert understood.response_text == "Soft rain on a metal roof"
    assert generated.transcript == "Rain on a metal roof"
    assert generated.response_text == "I generated two playable variations."
    assert [asset["caption"] for asset in generated.assets] == [
        "Gentle rain on a corrugated metal roof.",
        "Heavy rain rattles a distant metal awning.",
    ]
    assert all(asset["audio_path"].endswith(".wav") for asset in generated.assets)
    assert continued.transcript == "Soft rain on a metal roof"
    assert sounds.prompts == [
        "Rain on a metal roof",
        "Soft rain on a metal roof",
    ]


@pytest.mark.fast
def test_sound_lab_web_backend_returns_ready_assets_without_voting_or_reveal(
    tmp_path: Path,
) -> None:
    first = tmp_path / "one.wav"
    second = tmp_path / "two.wav"
    first.write_bytes(b"RIFF one")
    second.write_bytes(b"RIFF two")

    class Session:
        def handle(self, prompt: str):
            assert prompt == "Rain on leaves"
            return SimpleNamespace(job_id="job-1", ready_count=2, failed_count=0)

    class Catalog:
        def public_snapshot(self, *, reveal_all: bool = False):
            assert reveal_all is True
            return {
                "jobs": [
                    {
                        "job_id": "job-1",
                        "revealed": False,
                        "candidates": [
                            {
                                "asset_id": "one",
                                "label": "A",
                                "state": "ready",
                                "caption": "Light rain taps broad leaves.",
                            },
                            {
                                "asset_id": "two",
                                "label": "B",
                                "state": "ready",
                                "caption": "Heavy drops strike dry leaves.",
                            },
                        ],
                    }
                ]
            }

        def audio_path(self, asset_id: str) -> Path:
            return {"one": first, "two": second}[asset_id]

    backend = SoundLabWebBackend(session=Session(), catalog=Catalog())

    message, assets = backend.generate("Rain on leaves")

    assert message == "I generated 2 playable sound variations."
    assert [asset.caption for asset in assets] == [
        "Light rain taps broad leaves.",
        "Heavy drops strike dry leaves.",
    ]
    assert [asset.audio_path for asset in assets] == [first, second]


@pytest.mark.fast
def test_shared_factory_keeps_one_model_session_and_switches_cache_keys(
    tmp_path: Path,
) -> None:
    loaded_for: list[str] = []
    sound_loads: list[object] = []

    class SharedSession(FakeAudexSession):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.activated: list[str] = []

        def activate_conversation(self, conversation, _store) -> None:
            self.activated.append(conversation.conversation_id)

    shared_session = SharedSession(tmp_path)

    def session_loader(conversation, _store, _persona):
        loaded_for.append(conversation.conversation_id)
        return shared_session

    def sound_loader(session):
        sound_loads.append(session)
        return FakeSoundBackend(tmp_path)

    factory = SharedAudexRuntimeFactory(
        conversation_store=ConversationStore(tmp_path / "model-conversations"),
        persona=SimpleNamespace(
            persona_id="assistant",
            path=tmp_path / "assistant.md",
            system_prompt="System.",
        ),
        session_loader=session_loader,
        sound_backend_loader=sound_loader,
    )
    first = factory.create("chat-first")
    second = factory.create("chat-second")

    first.respond(mode=ChatMode.TEXT_TEXT, text="First", audio_path=None)
    second.respond(mode=ChatMode.TEXT_TEXT, text="Second", audio_path=None)
    first.respond(mode=ChatMode.TEXT_AUDIO, text="Bell", audio_path=None)

    assert loaded_for == ["chat-first"]
    assert shared_session.activated == ["chat-second", "chat-first"]
    assert sound_loads == [shared_session]
    assert [name for name, _payload in shared_session.calls] == [
        "text-text",
        "text-text",
    ]
    assert factory.loaded is True
