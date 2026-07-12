"""Adapters from browser modes to warm Audex model sessions."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ..conversations import Conversation, ConversationStore
from ..personas import Persona
from .chat import RuntimeTurn, TurnStream
from .modes import ChatMode

DEFAULT_AUDIO_UNDERSTANDING_PROMPT = (
    "Describe what is audible in this recording, including the sound sources, "
    "actions, environment, distance, and timing."
)
REFERENCE_AUDIO_CAPTION_PROMPT = (
    "Write one literal present-tense AudioCaps-style sentence describing only "
    "what is audible. Include the source, action, environment, distance, and "
    "timing needed to generate a similar sound."
)


@dataclass(frozen=True, slots=True)
class GeneratedAudioAsset:
    label: str
    caption: str
    audio_path: Path

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["audio_path"] = str(self.audio_path)
        return payload


class SoundGenerationBackend(Protocol):
    def generate(
        self,
        prompt: str,
    ) -> tuple[str, tuple[GeneratedAudioAsset, ...]]: ...


class SoundLabWebBackend:
    """Expose ready Sound Lab candidates immediately, with no voting ceremony."""

    def __init__(self, *, session: Any, catalog: Any) -> None:
        self.session = session
        self.catalog = catalog

    def generate(
        self,
        prompt: str,
    ) -> tuple[str, tuple[GeneratedAudioAsset, ...]]:
        turn = self.session.handle(prompt)
        if turn.job_id is None:
            return (str(getattr(turn, "message", "Audex returned no sounds.")), ())
        snapshot = self.catalog.public_snapshot(reveal_all=True)
        job = next(
            (
                item
                for item in snapshot.get("jobs", [])
                if item.get("job_id") == turn.job_id
            ),
            None,
        )
        if job is None:
            raise RuntimeError(f"Sound Lab job disappeared: {turn.job_id}")
        assets = tuple(
            GeneratedAudioAsset(
                label=f"Variation {candidate.get('label', index + 1)}",
                caption=str(candidate.get("caption", "Generated Audex sound")),
                audio_path=self.catalog.audio_path(str(candidate["asset_id"])),
            )
            for index, candidate in enumerate(job.get("candidates", []))
            if candidate.get("state") == "ready"
        )
        noun = "variation" if len(assets) == 1 else "variations"
        return (f"I generated {len(assets)} playable sound {noun}.", assets)


class AudexConversationRuntime:
    """Route every mode through one already-warm conversational session."""

    def __init__(self, *, session: Any, sound_backend: SoundGenerationBackend) -> None:
        self.session = session
        self.sound_backend = sound_backend

    def respond(
        self,
        *,
        mode: ChatMode,
        text: str | None,
        audio_path: Path | None,
        stream: TurnStream | None = None,
    ) -> RuntimeTurn:
        if mode is ChatMode.TEXT_TEXT:
            result = self.session.run_text_only_turn_from_text(
                user_text=_require_text(text, mode)
            )
            return _conversation_turn(result)
        if mode is ChatMode.TEXT_SPEECH:
            result = self.session.run_turn_from_text(
                user_text=_require_text(text, mode),
                play=False,
                pcm_chunk_sink=(stream.assistant_pcm if stream is not None else None),
                text_delta_sink=(stream.assistant_text if stream is not None else None),
            )
            return _conversation_turn(result, output_audio=True)
        if mode is ChatMode.SPEECH_TEXT:
            result = self.session.run_text_only_turn_from_wav(
                input_wav_path=_require_audio(audio_path, mode)
            )
            return _conversation_turn(result)
        if mode is ChatMode.SPEECH_SPEECH:
            result = self.session.run_turn_from_wav(
                input_wav_path=_require_audio(audio_path, mode),
                play=False,
                pcm_chunk_sink=(stream.assistant_pcm if stream is not None else None),
                text_delta_sink=(stream.assistant_text if stream is not None else None),
            )
            return _conversation_turn(result, output_audio=True)
        if mode is ChatMode.AUDIO_TEXT:
            source = _require_audio(audio_path, mode)
            caption = self.session.understand_audio(
                input_wav_path=source,
                prompt=REFERENCE_AUDIO_CAPTION_PROMPT,
            )
            question = text or DEFAULT_AUDIO_UNDERSTANDING_PROMPT
            answer = self.session.understand_audio(
                input_wav_path=source,
                prompt=question,
            )
            visible_prompt = caption.response_text.strip()
            if text:
                visible_prompt = f"{visible_prompt}\n\nQuestion: {text.strip()}"
            return RuntimeTurn(
                transcript=visible_prompt,
                response_text=answer.response_text,
            )
        if mode is ChatMode.TEXT_AUDIO:
            prompt = _require_text(text, mode)
            message, assets = self.sound_backend.generate(prompt)
            return RuntimeTurn(
                transcript=prompt,
                response_text=message,
                assets=tuple(asset.to_dict() for asset in assets),
            )
        if mode is ChatMode.AUDIO_AUDIO:
            understood = self.session.understand_audio(
                input_wav_path=_require_audio(audio_path, mode),
                prompt=REFERENCE_AUDIO_CAPTION_PROMPT,
            )
            caption = understood.response_text.strip()
            if not caption:
                raise ValueError("Audex could not describe the reference audio.")
            direction = text.strip() if text else ""
            generation_prompt = caption
            visible_prompt = caption
            if direction:
                generation_prompt = f"{caption}\n\nCreative direction: {direction}"
                visible_prompt = f"{caption}\n\nDirection: {direction}"
            message, assets = self.sound_backend.generate(generation_prompt)
            return RuntimeTurn(
                transcript=visible_prompt,
                response_text=message,
                assets=tuple(asset.to_dict() for asset in assets),
            )
        raise ValueError(f"Unsupported Audex browser mode: {mode}")


SessionLoader = Callable[[Conversation, ConversationStore, Persona], Any]
SoundBackendLoader = Callable[[Any], SoundGenerationBackend]


class SharedAudexRuntimeFactory:
    """Share one heavy model/decoder while retaining independent chat cache keys."""

    def __init__(
        self,
        *,
        conversation_store: ConversationStore,
        persona: Persona,
        session_loader: SessionLoader,
        sound_backend_loader: SoundBackendLoader,
    ) -> None:
        self.conversation_store = conversation_store
        self.persona = persona
        self.session_loader = session_loader
        self.sound_backend_loader = sound_backend_loader
        self._session: Any | None = None
        self._sound_backend: SoundGenerationBackend | None = None
        self._active_chat_id: str | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def create(self, chat_id: str) -> BoundAudexConversationRuntime:
        return BoundAudexConversationRuntime(factory=self, chat_id=chat_id)

    def respond(
        self,
        chat_id: str,
        *,
        mode: ChatMode,
        text: str | None,
        audio_path: Path | None,
        stream: TurnStream | None = None,
    ) -> RuntimeTurn:
        with self._lock:
            conversation = self._load_or_create_conversation(chat_id)
            if self._session is None:
                self._session = self.session_loader(
                    conversation,
                    self.conversation_store,
                    self.persona,
                )
                self._active_chat_id = chat_id
            elif self._active_chat_id != chat_id:
                self._session.activate_conversation(
                    conversation,
                    self.conversation_store,
                )
                self._active_chat_id = chat_id
            if mode.output_kind == "audio" and self._sound_backend is None:
                self._sound_backend = self.sound_backend_loader(self._session)
            runtime = AudexConversationRuntime(
                session=self._session,
                sound_backend=self._sound_backend or _UnavailableSoundBackend(),
            )
            return runtime.respond(
                mode=mode,
                text=text,
                audio_path=audio_path,
                stream=stream,
            )

    def shutdown(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.shutdown()

    def _load_or_create_conversation(self, chat_id: str) -> Conversation:
        try:
            return self.conversation_store.load(chat_id)
        except FileNotFoundError:
            return self.conversation_store.create(
                conversation_id=chat_id,
                persona_id=self.persona.persona_id,
                persona_path=self.persona.path,
                system_prompt=self.persona.system_prompt,
            )


@dataclass(frozen=True, slots=True)
class BoundAudexConversationRuntime:
    factory: SharedAudexRuntimeFactory
    chat_id: str

    def respond(
        self,
        *,
        mode: ChatMode,
        text: str | None,
        audio_path: Path | None,
        stream: TurnStream | None = None,
    ) -> RuntimeTurn:
        return self.factory.respond(
            self.chat_id,
            mode=mode,
            text=text,
            audio_path=audio_path,
            stream=stream,
        )


class _UnavailableSoundBackend:
    def generate(
        self,
        _prompt: str,
    ) -> tuple[str, tuple[GeneratedAudioAsset, ...]]:
        raise RuntimeError("Audex sound-generation runtime is not loaded.")


def _conversation_turn(result: Any, *, output_audio: bool = False) -> RuntimeTurn:
    return RuntimeTurn(
        transcript=str(result.transcript),
        response_text=str(result.response_text),
        output_audio_path=(Path(result.output_wav_path) if output_audio else None),
    )


def _require_text(text: str | None, mode: ChatMode) -> str:
    normalized = text.strip() if text is not None else ""
    if not normalized:
        raise ValueError(f"{mode.spec.label} requires text input")
    return normalized


def _require_audio(audio_path: Path | None, mode: ChatMode) -> Path:
    if audio_path is None:
        raise ValueError(f"{mode.spec.label} requires audio input")
    return audio_path
