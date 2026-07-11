"""User-selectable Audex browser modes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

InputKind = Literal["text", "speech", "audio"]
OutputKind = Literal["text", "speech", "audio"]


@dataclass(frozen=True, slots=True)
class ModeSpec:
    label: str
    description: str
    input_kind: InputKind
    output_kind: OutputKind
    icon: str
    preserves_conversation_cache: bool


class ChatMode(StrEnum):
    TEXT_TEXT = "text-text"
    TEXT_SPEECH = "text-speech"
    SPEECH_TEXT = "speech-text"
    SPEECH_SPEECH = "speech-speech"
    AUDIO_TEXT = "audio-text"
    TEXT_AUDIO = "text-audio"
    AUDIO_AUDIO = "audio-audio"

    @property
    def spec(self) -> ModeSpec:
        return MODE_SPECS[self]

    @property
    def input_kind(self) -> InputKind:
        return self.spec.input_kind

    @property
    def output_kind(self) -> OutputKind:
        return self.spec.output_kind


MODE_SPECS: dict[ChatMode, ModeSpec] = {
    ChatMode.TEXT_TEXT: ModeSpec(
        "Text in, Text out",
        "Type a message and receive a written Audex response.",
        "text",
        "text",
        "message-square",
        True,
    ),
    ChatMode.TEXT_SPEECH: ModeSpec(
        "Text in, Speech out",
        "Type a message; Audex replies in text and natural speech.",
        "text",
        "speech",
        "message-volume",
        True,
    ),
    ChatMode.SPEECH_TEXT: ModeSpec(
        "Speech in, Text out",
        "Speak naturally; your transcript and Audex's written reply stay visible.",
        "speech",
        "text",
        "mic-message",
        True,
    ),
    ChatMode.SPEECH_SPEECH: ModeSpec(
        "Speech in, Speech out",
        "Have a spoken conversation with visible transcripts on both sides.",
        "speech",
        "speech",
        "waveform",
        True,
    ),
    ChatMode.AUDIO_TEXT: ModeSpec(
        "Audio in, Text out",
        "Upload or record audio for Audex to understand and describe in text.",
        "audio",
        "text",
        "audio-search",
        False,
    ),
    ChatMode.TEXT_AUDIO: ModeSpec(
        "Text in, Audio out",
        "Describe a sound and generate playable variations immediately.",
        "text",
        "audio",
        "sparkles-wave",
        False,
    ),
    ChatMode.AUDIO_AUDIO: ModeSpec(
        "Audio in, Audio out",
        "Use a reference sound as the prompt for a newly generated sound.",
        "audio",
        "audio",
        "audio-loop",
        False,
    ),
}


def mode_catalog() -> list[dict[str, object]]:
    return [
        {
            "id": mode.value,
            "label": spec.label,
            "description": spec.description,
            "input_kind": spec.input_kind,
            "output_kind": spec.output_kind,
            "icon": spec.icon,
            "preserves_conversation_cache": spec.preserves_conversation_cache,
        }
        for mode, spec in MODE_SPECS.items()
    ]
