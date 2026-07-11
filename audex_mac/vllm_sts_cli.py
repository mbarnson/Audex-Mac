"""vLLM-backed Audex speech-to-speech CLI path."""

from __future__ import annotations

import asyncio
import gc
import json
import os
import re
import resource
import subprocess
import time
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from .audio_encoder import (
    encode_audio_features_mlx,
    load_audio_encoder_config,
    load_audio_encoder_weights_mlx,
)
from .audio_features import extract_audex_input_features
from .audio_pcm import SAMPLE_RATE, load_wav_pcm, prepare_audex_wav_clips
from .audio_projector import (
    load_audio_projector_config,
    load_audio_projector_weights_mlx,
    project_audio_hidden_states_mlx,
)
from .conversations import (
    DEFAULT_DEMO_CONTEXT_TOKENS,
    Conversation,
    ConversationStore,
)
from .personas import DEFAULT_PERSONA_NAME, Persona, load_persona
from .preemptive_turn import PreemptiveTurnCoordinator
from .speech_decoder import (
    AudexSpeechDecoderSession,
    configured_speech_decoder_device,
    load_speech_decoder_config,
    load_speech_decoder_weights_mlx,
)
from .speech_output import (
    RUNS_DIR,
    SpeechOutputSmokeResult,
    write_pcm16_wav,
)
from .sts_cli import (
    DEFAULT_RESPONSE_MAX_TOKENS,
    DEFAULT_S2S_TTS_MAX_TOKENS,
    DEFAULT_SPEECH_TOKENS_PER_TEXT_TOKEN,
    SpeechToSpeechTurnResult,
    _ContinuousPcmPlayer,
    _PlaybackStartGate,
    _Recording,
    play_wav,
    startup_greeting_text,
)
from .tts_text import (
    DEFAULT_TTS_TARGET_SEGMENTS as DEFAULT_VLLM_TTS_TARGET_SEGMENTS,
)
from .tts_text import (
    streamed_tts_chunks_from_text,
)
from .vllm_runtime import (
    AudexAsyncVllmRuntime,
    AudexVllmRuntime,
    VllmRequestResult,
    VllmStreamDelta,
    extract_spoken_answer,
    scrub_spoken_answer,
)
from .vllm_speech import (
    DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES,
    SpeechSynthesisRequest,
    VllmSpeechSynthesizer,
    _QueuedTtsChunkSource,
)
from .vllm_sts_requests import (
    VllmTtsSamplingConfig,
    build_audio_messages_response_request,
    build_audio_response_prefix_token_ids,
    build_text_messages_history_prompt,
    build_text_messages_response_request,
)

DEFAULT_VLLM_TEXT_CONTEXT_TOKENS = DEFAULT_DEMO_CONTEXT_TOKENS
DEFAULT_VLLM_TTS_MIN_TOKENS_PER_CHUNK = 512
VLLM_TTS_CFG_ENV = "AUDEX_VLLM_TTS_CFG"
VLLM_CFG_WIRING_ENV = "AUDEX_VLLM_ENABLE_CFG_WIRING"
VLLM_RESET_PREFIX_CACHE_BEFORE_TTS_ENV = "AUDEX_VLLM_RESET_PREFIX_CACHE_BEFORE_TTS"
VLLM_DIRECT_AUDIO_RESPONSE_ENV = "AUDEX_VLLM_DIRECT_AUDIO_RESPONSE"
SEMANTIC_AUDIO_GATE_SECONDS = 1.0
RESUME_STARTUP_GREETING_MAX_TOKENS = 96
RESUME_STARTUP_GREETING_PROMPT = (
    "Open this resumed spoken conversation with a warm greeting. Do not "
    "introduce yourself, state your name, or explain that you remember a "
    "transcript, history, cache, or previous session. In one or two short "
    "spoken sentences, briefly mention the main topic from the most recent "
    "substantive exchange, then ask what the user wants to talk about today. "
    "Only output words meant to be spoken aloud."
)


def _has_substantive_conversation_history(
    messages: list[dict[str, str]],
) -> bool:
    return any(
        message.get("role") == "user" and message.get("content", "").strip()
        for message in messages
    )


def _limit_spoken_sentences(text: str, *, max_sentences: int) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    sentences = [
        sentence.strip()
        for sentence in re.findall(r"[^.!?]+(?:[.!?]+|$)", cleaned)
        if sentence.strip()
    ]
    return " ".join(sentences[: max(1, int(max_sentences))]).strip()


@dataclass(frozen=True, slots=True)
class VllmSpeechToSpeechSessionStats:
    model_load_seconds: float
    audio_component_load_seconds: float
    decoder_load_seconds: float
    turns: int


@dataclass(frozen=True, slots=True)
class VllmTextTurnResult:
    """A conversational turn that deliberately stops before speech synthesis."""

    transcript: str
    response_text: str
    input_wav_path: Path | None
    run_log_path: Path


@dataclass(frozen=True, slots=True)
class _PreparedSpokenTurn:
    started_at: float
    asr_started_at: float
    asr_finished_at: float
    asr: VllmRequestResult
    transcript: str
    asr_mlx_clear_cache_seconds: float
    process_memory_before_text: dict[str, object]
    text_started_at: float
    pending_messages: list[dict[str, str]]
    text_to_tts_interleaved: bool
    tts_prefix_cache_reset: bool
    tts_started_at: float
    text: VllmRequestResult
    response_text: str
    text_mlx_clear_cache_seconds: float
    speech: SpeechOutputSmokeResult
    response_source: str = "transcript"
    staged_voice_revision: int | None = None
    staged_sample_count: int | None = None


class VllmSpeechToSpeechSession:
    """Reusable vLLM Metal Audex state for CLI speech-to-speech turns."""

    def __init__(
        self,
        *,
        full_model_path: Path,
        decoder_path: Path,
        selected_model_repo: str | None = None,
        output_dir: Path = RUNS_DIR,
        thinking_enabled: bool = False,
        response_max_tokens: int = DEFAULT_RESPONSE_MAX_TOKENS,
        speech_max_tokens: int | None = None,
        conversation: Conversation | None = None,
        conversation_store: ConversationStore | None = None,
        persona: Persona | None = None,
        runtime: AudexVllmRuntime | None = None,
        async_runtime: AudexAsyncVllmRuntime | None = None,
        tts_sampling_config: VllmTtsSamplingConfig | None = None,
    ) -> None:
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise RuntimeError("Audex vLLM STS session requires mlx.") from exc

        self.full_model_path = full_model_path
        self.decoder_path = decoder_path
        self.selected_model_repo = selected_model_repo
        self.output_dir = output_dir
        self.thinking_enabled = thinking_enabled
        self.response_max_tokens = response_max_tokens
        self.speech_max_tokens = speech_max_tokens
        self.tts_cfg_enabled = (
            tts_sampling_config.cfg_enabled
            if tts_sampling_config is not None
            else _vllm_tts_cfg_enabled()
        )
        self.persona = persona or load_persona(DEFAULT_PERSONA_NAME)
        self.conversation_store = conversation_store
        self.conversation = conversation
        if conversation is not None:
            self.messages = _sanitize_prompt_history(
                conversation.messages,
                system_prompt=self.persona.system_prompt,
            )
            self._sanitized_resumed_history = self.messages != conversation.messages
        else:
            self.messages = [{"role": "system", "content": self.persona.system_prompt}]
            self._sanitized_resumed_history = False
        self.turns = 0
        self._last_interleaved_text_stream_stats: dict[str, object] = {}
        self._last_text_context_stats: dict[str, object] = {}
        self._audio_conversation_state_cache: dict[str, object] | None = None
        self.mx = mx
        self.mx.set_default_device(self.mx.gpu)

        if runtime is None and async_runtime is None:
            if tts_sampling_config is None:
                _enable_cfg_wiring_if_tts_cfg_requested()
            async_runtime = AudexAsyncVllmRuntime.from_model_path(
                full_model_path,
                tts_sampling_config=tts_sampling_config,
            )
        self.runtime = runtime
        self.async_runtime = async_runtime
        self._align_conversation_context_limit()
        self._async_loop = (
            asyncio.new_event_loop()
            if self.async_runtime is not None and self.runtime is None
            else None
        )

        self.encoder_config = None
        self.encoder_weights = None
        self.projector_config = None
        self.projector_weights = None
        self.audio_component_load_seconds = 0.0

        decoder_started_at = time.time()
        self.decoder_config = load_speech_decoder_config(decoder_path)
        self.decoder_device = configured_speech_decoder_device(self.mx)
        stream_context = getattr(self.mx, "stream", None)
        if callable(stream_context):
            with stream_context(self.decoder_device):
                self.decoder_weights = load_speech_decoder_weights_mlx(decoder_path)
        else:
            self.decoder_weights = load_speech_decoder_weights_mlx(decoder_path)
        self.decoder_load_seconds = round(time.time() - decoder_started_at, 3)
        if self._sanitized_resumed_history:
            self._persist_conversation(announce=False, invalidate_kv_cache=True)
        if (
            self.async_runtime is not None
            and self.runtime is None
            and _direct_audio_response_enabled(thinking_enabled=self.thinking_enabled)
        ):
            self._run_async(self._prime_audio_response_history_async())

    @property
    def stats(self) -> VllmSpeechToSpeechSessionStats:
        return VllmSpeechToSpeechSessionStats(
            model_load_seconds=self._model_runtime_stats().model_load_seconds,
            audio_component_load_seconds=self.audio_component_load_seconds,
            decoder_load_seconds=self.decoder_load_seconds,
            turns=self.turns,
        )

    def activate_conversation(
        self,
        conversation: Conversation,
        conversation_store: ConversationStore,
    ) -> None:
        """Select cached history by identity without reconstructing the model runtime."""

        if (
            self.conversation is not None
            and self.conversation.conversation_id == conversation.conversation_id
        ):
            return
        messages = _sanitize_prompt_history(
            conversation.messages,
            system_prompt=self.persona.system_prompt,
        )
        self.conversation = conversation
        self.conversation_store = conversation_store
        self.messages = messages
        self.turns = sum(
            1
            for message in messages
            if message.get("role") == "user" and message.get("content", "").strip()
        )
        self._audio_conversation_state_cache = None
        self._align_conversation_context_limit()
        if messages != conversation.messages:
            self._persist_conversation(announce=False, invalidate_kv_cache=True)

    def shutdown(self, timeout: float | None = 5.0) -> None:
        for runtime in (self.async_runtime, self.runtime):
            engine = getattr(runtime, "engine", None)
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                with suppress(Exception):
                    shutdown(timeout=timeout)
        self._close_async_loop()

    def _new_speech_decoder_session(self, **kwargs: Any) -> Any:
        decoder = AudexSpeechDecoderSession(**kwargs)
        decoder.device = getattr(self, "decoder_device", None)
        return decoder

    def run_text_only_turn_from_text(self, *, user_text: str) -> VllmTextTurnResult:
        """Append a typed turn while retaining the warm conversation state."""

        text = user_text.strip()
        if not text:
            raise ValueError("Typed Audex input must not be empty.")
        return self._complete_text_only_turn(
            transcript=text,
            input_wav_path=None,
            input_mode="text",
            asr_elapsed_seconds=None,
        )

    def run_text_only_turn_from_wav(
        self,
        *,
        input_wav_path: Path,
    ) -> VllmTextTurnResult:
        """Transcribe speech and append a text-response turn without running TTS."""

        started_at = time.time()
        input_audio = load_wav_pcm(input_wav_path)
        if input_audio.sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Audex vLLM speech input must be {SAMPLE_RATE} Hz PCM WAV, "
                f"got {input_audio.sample_rate} Hz."
            )
        print(
            "Audex STS: transcribing browser speech with vLLM Metal...",
            flush=True,
        )
        asr = self._transcribe_audio(
            input_audio.samples,
            sample_rate=input_audio.sample_rate,
        )
        transcript = asr.text.strip()
        if not transcript:
            raise ValueError("Audex could not find speech in the input audio.")
        print(f"Audex STS: transcript: {transcript}", flush=True)
        return self._complete_text_only_turn(
            transcript=transcript,
            input_wav_path=input_wav_path,
            input_mode="speech",
            asr_elapsed_seconds=round(time.time() - started_at, 3),
        )

    def _complete_text_only_turn(
        self,
        *,
        transcript: str,
        input_wav_path: Path | None,
        input_mode: str,
        asr_elapsed_seconds: float | None,
    ) -> VllmTextTurnResult:
        started_at = time.time()
        pending_messages = [
            *self.messages,
            {"role": "user", "content": transcript},
        ]
        pending_messages = self._validate_text_prompt_messages(
            pending_messages,
            max_tokens=self.response_max_tokens,
        )
        print("Audex STS: generating text response with vLLM Metal...", flush=True)
        text = self._generate_text_response_from_messages(
            pending_messages,
            enable_reasoning=self.thinking_enabled,
            max_tokens=self.response_max_tokens,
            **self._text_conversation_state_kwargs(),
        )
        response_text = scrub_spoken_answer(text.text)
        self._clear_mlx_cache()
        print(f"Audex STS: response text: {response_text}", flush=True)
        self.messages = [
            *pending_messages,
            {"role": "assistant", "content": response_text},
        ]
        self.turns += 1
        self._persist_conversation()
        if self.async_runtime is not None and self.runtime is None:
            self._run_async(self._prime_audio_response_history_async())

        elapsed_seconds = round(time.time() - started_at, 3)
        run_log_path = self.output_dir / (f"text-turn-vllm-{time.time_ns()}.json")
        run_log = {
            "backend": "vllm",
            "selected_model": self.selected_model_repo,
            "input_mode": input_mode,
            "output_mode": "text",
            "input_wav_path": (
                str(input_wav_path) if input_wav_path is not None else None
            ),
            "transcript": transcript,
            "response_text": response_text,
            "turn_index": self.turns,
            "conversation_id": (
                self.conversation.conversation_id
                if self.conversation is not None
                else None
            ),
            "conversation_token_count": (
                self.conversation.token_count
                if self.conversation is not None
                else self._count_messages_tokens()
            ),
            "max_context_tokens": (
                self.conversation.max_context_tokens
                if self.conversation is not None
                else None
            ),
            "timings": {
                "elapsed_seconds": elapsed_seconds,
                "asr_elapsed_seconds": asr_elapsed_seconds,
                "text_elapsed_seconds": text.elapsed_seconds,
            },
            "text_context": dict(self._last_text_context_stats),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
        return VllmTextTurnResult(
            transcript=transcript,
            response_text=response_text,
            input_wav_path=input_wav_path,
            run_log_path=run_log_path,
        )

    def understand_audio(
        self,
        *,
        input_wav_path: Path,
        prompt: str,
    ) -> VllmTextTurnResult:
        """Answer a free-form question about non-speech audio outside chat state."""

        normalized_prompt = " ".join(prompt.split())
        if not normalized_prompt:
            raise ValueError("Audio understanding requires a prompt.")
        input_audio = load_wav_pcm(input_wav_path)
        if input_audio.sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Audex audio understanding requires {SAMPLE_RATE} Hz PCM WAV, "
                f"got {input_audio.sample_rate} Hz."
            )
        request = build_audio_messages_response_request(
            self._model_tokenizer(),
            [{"role": "system", "content": self.persona.system_prompt}],
            input_audio.samples,
            sample_rate=input_audio.sample_rate,
            prompt_text=normalized_prompt,
            enable_reasoning=self.thinking_enabled,
            max_tokens=self.response_max_tokens,
            trim_padded_audio_embeddings=True,
        )
        request = replace(request, debug_name="audio-understanding")
        started_at = time.time()
        if self.async_runtime is not None:
            result = self._run_async(self.async_runtime.generate_one_final(request))
        elif self.runtime is not None:
            result = self.runtime.generate_one(request)
        else:
            raise RuntimeError("Audex vLLM runtime is not configured.")
        response_text = extract_spoken_answer(result.text)
        self._clear_mlx_cache()
        run_log_path = self.output_dir / (
            f"audio-understanding-vllm-{time.time_ns()}.json"
        )
        run_log = {
            "backend": "vllm",
            "selected_model": self.selected_model_repo,
            "input_mode": "audio",
            "output_mode": "text",
            "input_wav_path": str(input_wav_path),
            "transcript": normalized_prompt,
            "response_text": response_text,
            "conversation_mutated": False,
            "timings": {
                "elapsed_seconds": round(time.time() - started_at, 3),
                "generation_elapsed_seconds": result.elapsed_seconds,
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
        return VllmTextTurnResult(
            transcript=normalized_prompt,
            response_text=response_text,
            input_wav_path=input_wav_path,
            run_log_path=run_log_path,
        )

    def run_turn_from_wav(
        self,
        *,
        input_wav_path: Path,
        play: bool = True,
        turn_submitted_at: float | None = None,
    ) -> SpeechToSpeechTurnResult:
        if self.async_runtime is not None and self.runtime is None:
            return self._run_async(
                self._run_turn_from_wav_async(
                    input_wav_path=input_wav_path,
                    play=play,
                    turn_submitted_at=turn_submitted_at,
                )
            )

        started_at = time.time()
        input_audio = load_wav_pcm(input_wav_path)
        if input_audio.sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Audex vLLM STS input must be {SAMPLE_RATE} Hz PCM WAV, "
                f"got {input_audio.sample_rate} Hz."
            )

        print(
            "Audex STS: transcribing raw input speech with vLLM Metal...",
            flush=True,
        )
        asr_started_at = time.time()
        asr = self._transcribe_audio(
            input_audio.samples,
            sample_rate=input_audio.sample_rate,
        )
        transcript = asr.text
        asr_mlx_clear_cache_seconds = self._clear_mlx_cache()
        print(f"Audex STS: transcript: {transcript}", flush=True)

        print("Audex STS: generating text response with vLLM Metal...", flush=True)
        process_memory_before_text = _process_memory_snapshot()
        text_started_at = time.time()
        pending_messages = [*self.messages, {"role": "user", "content": transcript}]
        pending_messages = self._validate_text_prompt_messages(
            pending_messages,
            max_tokens=self.response_max_tokens,
        )
        text = self._generate_text_response_from_messages(
            pending_messages,
            enable_reasoning=self.thinking_enabled,
            max_tokens=self.response_max_tokens,
            **self._text_conversation_state_kwargs(),
        )
        response_text = scrub_spoken_answer(text.text)
        text_mlx_clear_cache_seconds = self._clear_mlx_cache()
        print(f"Audex STS: response text: {response_text}", flush=True)

        print("Audex STS: generating speech output with vLLM Metal...", flush=True)
        tts_started_at = time.time()
        speech = self.generate_speech_output(
            text=response_text or transcript or "I heard your message.",
            max_tokens=self._speech_max_tokens_for_text(response_text),
            play=play,
        )
        print(f"Audex STS: speech output ready: {speech.wav_path}", flush=True)

        self.messages = [
            *pending_messages,
            {"role": "assistant", "content": response_text},
        ]
        self.turns += 1
        self._persist_conversation()

        elapsed_seconds = round(time.time() - started_at, 3)
        run_log_path = (
            self.output_dir / f"sts-turn-vllm-{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
        semantic_audio = _semantic_audio_diagnostics(
            speech=speech,
            turn_submitted_at=turn_submitted_at,
            play=play,
            cfg_enabled=bool(getattr(self, "tts_cfg_enabled", False)),
            text_to_tts_interleaved=False,
        )
        run_log = {
            "backend": "vllm",
            "selected_model": self.selected_model_repo,
            "input_wav_path": str(input_wav_path),
            "output_wav_path": str(speech.wav_path),
            "speech_output_run_log_path": str(speech.run_log_path),
            "transcript": transcript,
            "response_text": response_text,
            "played": play,
            "semantic_audio": semantic_audio,
            "response_max_tokens": self.response_max_tokens,
            "speech_max_tokens": self._speech_max_tokens_for_text(response_text),
            "thinking_enabled": self.thinking_enabled,
            "turn_index": self.turns,
            "conversation_turns": self.turns,
            "conversation_id": (
                self.conversation.conversation_id
                if self.conversation is not None
                else None
            ),
            "conversation_token_count": (
                self.conversation.token_count
                if self.conversation is not None
                else self._count_messages_tokens()
            ),
            "max_context_tokens": (
                self.conversation.max_context_tokens
                if self.conversation is not None
                else None
            ),
            "persona_id": self.persona.persona_id,
            "persona_path": str(self.persona.path),
            "elapsed_seconds": elapsed_seconds,
            "timings": {
                "elapsed_seconds": elapsed_seconds,
                "session_model_load_seconds": (
                    self._model_runtime_stats().model_load_seconds
                ),
                "session_audio_component_load_seconds": (
                    self.audio_component_load_seconds
                ),
                "session_decoder_load_seconds": self.decoder_load_seconds,
                "asr_elapsed_seconds": asr.elapsed_seconds,
                "asr_wall_seconds": round(text_started_at - asr_started_at, 3),
                "asr_mlx_clear_cache_seconds": asr_mlx_clear_cache_seconds,
                "text_elapsed_seconds": text.elapsed_seconds,
                "text_wall_seconds": round(tts_started_at - text_started_at, 3),
                "text_mlx_clear_cache_seconds": text_mlx_clear_cache_seconds,
                "tts_first_audio_ready_seconds": speech.first_audio_ready_seconds,
                "tts_first_playback_started_seconds": (
                    speech.first_playback_started_seconds
                ),
                "turn_submit_to_first_audio_ready_seconds": semantic_audio[
                    "turn_submit_to_first_audio_ready_seconds"
                ],
                "turn_submit_to_first_device_write_seconds": semantic_audio[
                    "turn_submit_to_first_device_write_seconds"
                ],
                "turn_submit_to_first_estimated_audible_seconds": semantic_audio[
                    "turn_submit_to_first_estimated_audible_seconds"
                ],
            },
            "process_memory": {
                "before_text": process_memory_before_text,
                "after_turn": _process_memory_snapshot(),
                "engine_core": _engine_core_memory_snapshots(),
            },
            "text_context": dict(self._last_text_context_stats),
            "vllm": {
                "engine_class": self._model_runtime_stats().engine_class,
                "asr_finish_reason": asr.finish_reason,
                "text_finish_reason": text.finish_reason,
                "tts_reached_end_token": speech.reached_end_token,
                "tts_hit_max_tokens": speech.hit_max_tokens,
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
        return SpeechToSpeechTurnResult(
            transcript=transcript,
            response_text=response_text,
            input_wav_path=input_wav_path,
            output_wav_path=speech.wav_path,
            run_log_path=run_log_path,
            played=play,
        )

    def run_preemptive_recorded_turn(
        self,
        *,
        recording: _Recording,
        input_wav_path: Path,
        play: bool = True,
        wait_for_submission: Callable[[], float] | None = None,
    ) -> SpeechToSpeechTurnResult:
        """Stage Audex ASR/text/CFG TTS during a stable pause before submit."""

        wait_for_submission = wait_for_submission or _wait_for_turn_submission
        if self.async_runtime is None or self.runtime is not None:
            turn_submitted_at = wait_for_submission()
            samples = recording.stop()
            write_pcm16_wav(input_wav_path, samples, sample_rate=SAMPLE_RATE)
            return self.run_turn_from_wav(
                input_wav_path=input_wav_path,
                play=play,
                turn_submitted_at=turn_submitted_at,
            )
        return self._run_async(
            self._run_preemptive_recorded_turn_async(
                recording=recording,
                input_wav_path=input_wav_path,
                play=play,
                wait_for_submission=wait_for_submission,
            )
        )

    async def _run_preemptive_recorded_turn_async(
        self,
        *,
        recording: _Recording,
        input_wav_path: Path,
        play: bool,
        wait_for_submission: Callable[[], float],
    ) -> SpeechToSpeechTurnResult:
        coordinator: PreemptiveTurnCoordinator[_PreparedSpokenTurn] = (
            PreemptiveTurnCoordinator()
        )

        async def prepare(snapshot, playback_gate):
            return await self._prepare_spoken_turn_async(
                samples=snapshot.samples,
                sample_rate=SAMPLE_RATE,
                play=play,
                playback_start_gate=playback_gate,
                staged_voice_revision=snapshot.voice_revision,
                staged_sample_count=snapshot.sample_count,
            )

        submission = await coordinator.capture(
            recording,
            play=play,
            wait_for_submission=wait_for_submission,
            prepare=prepare,
        )
        write_pcm16_wav(
            input_wav_path,
            submission.samples,
            sample_rate=SAMPLE_RATE,
        )
        if submission.prepared is not None:
            result = self._finish_prepared_spoken_turn(
                prepared=submission.prepared,
                input_wav_path=input_wav_path,
                play=play,
                turn_submitted_at=submission.submitted_at,
                final_voice_revision=submission.final_activity.voice_revision,
                final_sample_count=submission.final_activity.sample_count,
            )
            await self._prime_audio_response_history_async()
            return result
        if submission.preparation_error is not None:
            print(
                "Audex STS: preemptive generation was discarded; "
                "running the submitted turn exactly "
                f"({type(submission.preparation_error).__name__}).",
                flush=True,
            )
        return await self._run_turn_from_wav_async(
            input_wav_path=input_wav_path,
            play=play,
            turn_submitted_at=submission.submitted_at,
        )

    def run_turn_from_text(
        self,
        *,
        user_text: str,
        play: bool = True,
        turn_submitted_at: float | None = None,
    ) -> SpeechToSpeechTurnResult:
        """Skip ASR and run a typed user turn through text generation and TTS."""

        if not user_text.strip():
            raise ValueError("Typed Audex input must not be empty.")
        if self.async_runtime is not None and self.runtime is None:
            return self._run_async(
                self._run_turn_from_text_async(
                    user_text=user_text,
                    play=play,
                    turn_submitted_at=turn_submitted_at,
                )
            )

        started_at = time.time()
        process_memory_before_text = _process_memory_snapshot()
        print("Audex STS: generating text response from typed input...", flush=True)
        text_started_at = time.time()
        pending_messages = [*self.messages, {"role": "user", "content": user_text}]
        pending_messages = self._validate_text_prompt_messages(
            pending_messages,
            max_tokens=self.response_max_tokens,
        )
        text = self._generate_text_response_from_messages(
            pending_messages,
            enable_reasoning=self.thinking_enabled,
            max_tokens=self.response_max_tokens,
            **self._text_conversation_state_kwargs(),
        )
        response_text = scrub_spoken_answer(text.text)
        text_mlx_clear_cache_seconds = self._clear_mlx_cache()
        print(f"Audex STS: response text: {response_text}", flush=True)
        print("Audex STS: generating speech output with vLLM Metal...", flush=True)
        tts_started_at = time.time()
        speech = self.generate_speech_output(
            text=response_text or user_text,
            max_tokens=self._speech_max_tokens_for_text(response_text),
            play=play,
        )
        return self._finish_typed_turn(
            user_text=user_text,
            pending_messages=pending_messages,
            response_text=response_text,
            text=text,
            speech=speech,
            play=play,
            started_at=started_at,
            text_started_at=text_started_at,
            tts_started_at=tts_started_at,
            text_mlx_clear_cache_seconds=text_mlx_clear_cache_seconds,
            process_memory_before_text=process_memory_before_text,
            text_to_tts_interleaved=False,
            tts_prefix_cache_reset=False,
            turn_submitted_at=turn_submitted_at,
        )

    async def _run_turn_from_text_async(
        self,
        *,
        user_text: str,
        play: bool,
        turn_submitted_at: float | None = None,
    ) -> SpeechToSpeechTurnResult:
        if self.async_runtime is None:
            raise RuntimeError("Audex async vLLM runtime is not configured.")

        started_at = time.time()
        process_memory_before_text = _process_memory_snapshot()
        print("Audex STS: generating text response from typed input...", flush=True)
        text_started_at = time.time()
        pending_messages = [*self.messages, {"role": "user", "content": user_text}]
        pending_messages = self._validate_text_prompt_messages(
            pending_messages,
            max_tokens=self.response_max_tokens,
        )
        text_to_tts_interleaved = _text_to_tts_interleaving_enabled(
            thinking_enabled=self.thinking_enabled
        )
        tts_prefix_cache_reset = False
        if text_to_tts_interleaved:
            tts_started_at = text_started_at
            text, speech = await self._generate_response_and_speech_interleaved(
                pending_messages,
                fallback_text=user_text,
                play=play,
            )
            response_text = scrub_spoken_answer(text.text)
            text_mlx_clear_cache_seconds = self._clear_mlx_cache()
            print(f"Audex STS: response text: {response_text}", flush=True)
        else:
            text = await self.async_runtime.generate_text_response_from_messages(
                pending_messages,
                enable_reasoning=self.thinking_enabled,
                max_tokens=self.response_max_tokens,
                **self._text_conversation_state_kwargs(),
            )
            response_text = scrub_spoken_answer(text.text)
            text_mlx_clear_cache_seconds = self._clear_mlx_cache()
            print(f"Audex STS: response text: {response_text}", flush=True)
            if _reset_prefix_cache_before_tts_enabled():
                tts_prefix_cache_reset = await self.async_runtime.reset_prefix_cache()
                if not tts_prefix_cache_reset:
                    raise RuntimeError(
                        "vLLM could not reset its prefix cache before static CFG TTS."
                    )
            tts_started_at = time.time()
            speech = await self.generate_speech_output_streaming_from_async_runtime(
                text=response_text or user_text,
                max_tokens=self._speech_max_tokens_for_text(response_text),
                play=play,
            )
        result = self._finish_typed_turn(
            user_text=user_text,
            pending_messages=pending_messages,
            response_text=response_text,
            text=text,
            speech=speech,
            play=play,
            started_at=started_at,
            text_started_at=text_started_at,
            tts_started_at=tts_started_at,
            text_mlx_clear_cache_seconds=text_mlx_clear_cache_seconds,
            process_memory_before_text=process_memory_before_text,
            text_to_tts_interleaved=text_to_tts_interleaved,
            tts_prefix_cache_reset=tts_prefix_cache_reset,
            turn_submitted_at=turn_submitted_at,
        )
        await self._prime_audio_response_history_async()
        return result

    def _finish_typed_turn(
        self,
        *,
        user_text: str,
        pending_messages: list[dict[str, str]],
        response_text: str,
        text: VllmRequestResult,
        speech: SpeechOutputSmokeResult,
        play: bool,
        started_at: float,
        text_started_at: float,
        tts_started_at: float,
        text_mlx_clear_cache_seconds: float,
        process_memory_before_text: dict[str, object],
        text_to_tts_interleaved: bool,
        tts_prefix_cache_reset: bool,
        turn_submitted_at: float | None,
    ) -> SpeechToSpeechTurnResult:
        print(f"Audex STS: speech output ready: {speech.wav_path}", flush=True)
        self.messages = [
            *pending_messages,
            {"role": "assistant", "content": response_text},
        ]
        self.turns += 1
        self._persist_conversation()

        elapsed_seconds = round(time.time() - started_at, 3)
        run_log_path = (
            self.output_dir / f"sts-turn-vllm-{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
        semantic_audio = _semantic_audio_diagnostics(
            speech=speech,
            turn_submitted_at=turn_submitted_at,
            play=play,
            cfg_enabled=bool(getattr(self, "tts_cfg_enabled", False)),
            text_to_tts_interleaved=text_to_tts_interleaved,
        )
        run_log = {
            "backend": "vllm",
            "selected_model": self.selected_model_repo,
            "input_mode": "text",
            "input_wav_path": None,
            "typed_text": user_text,
            "transcript": None,
            "asr_skipped": True,
            "output_wav_path": str(speech.wav_path),
            "speech_output_run_log_path": str(speech.run_log_path),
            "response_text": response_text,
            "played": play,
            "response_max_tokens": self.response_max_tokens,
            "speech_max_tokens": self._speech_max_tokens_for_text(response_text),
            "thinking_enabled": self.thinking_enabled,
            "text_to_tts_interleaved": text_to_tts_interleaved,
            "tts_prefix_cache_reset": tts_prefix_cache_reset,
            "semantic_audio": semantic_audio,
            "turn_index": self.turns,
            "conversation_turns": self.turns,
            "conversation_id": (
                self.conversation.conversation_id
                if self.conversation is not None
                else None
            ),
            "conversation_token_count": (
                self.conversation.token_count
                if self.conversation is not None
                else self._count_messages_tokens()
            ),
            "max_context_tokens": (
                self.conversation.max_context_tokens
                if self.conversation is not None
                else None
            ),
            "persona_id": self.persona.persona_id,
            "persona_path": str(self.persona.path),
            "elapsed_seconds": elapsed_seconds,
            "timings": {
                "elapsed_seconds": elapsed_seconds,
                "session_model_load_seconds": (
                    self._model_runtime_stats().model_load_seconds
                ),
                "session_audio_component_load_seconds": (
                    self.audio_component_load_seconds
                ),
                "session_decoder_load_seconds": self.decoder_load_seconds,
                "asr_elapsed_seconds": None,
                "asr_wall_seconds": None,
                "text_elapsed_seconds": text.elapsed_seconds,
                "text_wall_seconds": round(tts_started_at - text_started_at, 3),
                "text_mlx_clear_cache_seconds": text_mlx_clear_cache_seconds,
                "tts_first_audio_ready_seconds": speech.first_audio_ready_seconds,
                "tts_first_playback_started_seconds": (
                    speech.first_playback_started_seconds
                ),
                "turn_submit_to_first_audio_ready_seconds": semantic_audio[
                    "turn_submit_to_first_audio_ready_seconds"
                ],
                "turn_submit_to_first_device_write_seconds": semantic_audio[
                    "turn_submit_to_first_device_write_seconds"
                ],
                "turn_submit_to_first_estimated_audible_seconds": semantic_audio[
                    "turn_submit_to_first_estimated_audible_seconds"
                ],
            },
            "process_memory": {
                "before_text": process_memory_before_text,
                "after_turn": _process_memory_snapshot(),
                "engine_core": _engine_core_memory_snapshots(),
            },
            "text_context": dict(self._last_text_context_stats),
            "vllm": {
                "asr_finish_reason": None,
                "text_finish_reason": text.finish_reason,
                "tts_reached_end_token": speech.reached_end_token,
                "tts_hit_max_tokens": speech.hit_max_tokens,
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
        return SpeechToSpeechTurnResult(
            transcript=user_text,
            response_text=response_text,
            input_wav_path=None,
            output_wav_path=speech.wav_path,
            run_log_path=run_log_path,
            played=play,
        )

    async def _prepare_spoken_turn_async(
        self,
        *,
        samples: tuple[float, ...],
        sample_rate: int,
        play: bool,
        playback_start_gate: _PlaybackStartGate | None = None,
        staged_voice_revision: int | None = None,
        staged_sample_count: int | None = None,
    ) -> _PreparedSpokenTurn:
        if self.async_runtime is None:
            raise RuntimeError("Audex async vLLM runtime is not configured.")

        if _direct_audio_response_enabled(thinking_enabled=self.thinking_enabled):
            return await self._prepare_direct_audio_response_turn_async(
                samples=samples,
                sample_rate=sample_rate,
                play=play,
                playback_start_gate=playback_start_gate,
                staged_voice_revision=staged_voice_revision,
                staged_sample_count=staged_sample_count,
            )

        started_at = time.time()
        preemptive = staged_voice_revision is not None
        print(
            (
                "Audex STS: preemptively transcribing live speech with vLLM Metal..."
                if preemptive
                else "Audex STS: transcribing raw input speech with vLLM Metal..."
            ),
            flush=True,
        )
        asr_started_at = time.time()
        asr = await self.async_runtime.transcribe_audio(
            samples,
            sample_rate=sample_rate,
        )
        transcript = asr.text
        asr_mlx_clear_cache_seconds = self._clear_mlx_cache()
        print(
            f"Audex STS: {'staged transcript' if preemptive else 'transcript'}: "
            f"{transcript}",
            flush=True,
        )

        text_to_tts_interleaved = _text_to_tts_interleaving_enabled(
            thinking_enabled=self.thinking_enabled
        )
        tts_prefix_cache_reset = False
        print("Audex STS: generating text response with vLLM Metal...", flush=True)
        process_memory_before_text = _process_memory_snapshot()
        text_started_at = time.time()
        pending_messages = [*self.messages, {"role": "user", "content": transcript}]
        pending_messages = self._validate_text_prompt_messages(
            pending_messages,
            max_tokens=self.response_max_tokens,
        )
        if text_to_tts_interleaved:
            tts_started_at = text_started_at
            text, speech = await self._generate_response_and_speech_interleaved(
                pending_messages,
                fallback_text=transcript or "I heard your message.",
                play=play,
                playback_start_gate=playback_start_gate,
            )
            response_text = scrub_spoken_answer(text.text)
            text_mlx_clear_cache_seconds = self._clear_mlx_cache()
        else:
            text = await self.async_runtime.generate_text_response_from_messages(
                pending_messages,
                enable_reasoning=self.thinking_enabled,
                max_tokens=self.response_max_tokens,
                **self._text_conversation_state_kwargs(),
            )
            response_text = scrub_spoken_answer(text.text)
            text_mlx_clear_cache_seconds = self._clear_mlx_cache()
            if _reset_prefix_cache_before_tts_enabled():
                tts_prefix_cache_reset = await self.async_runtime.reset_prefix_cache()
                if not tts_prefix_cache_reset:
                    raise RuntimeError(
                        "vLLM could not reset its prefix cache before static CFG TTS."
                    )
            tts_started_at = time.time()
            speech = await self.generate_speech_output_streaming_from_async_runtime(
                text=response_text or transcript or "I heard your message.",
                max_tokens=self._speech_max_tokens_for_text(response_text),
                play=play,
                playback_start_gate=playback_start_gate,
            )
        return _PreparedSpokenTurn(
            started_at=started_at,
            asr_started_at=asr_started_at,
            asr_finished_at=text_started_at,
            asr=asr,
            transcript=transcript,
            asr_mlx_clear_cache_seconds=asr_mlx_clear_cache_seconds,
            process_memory_before_text=process_memory_before_text,
            text_started_at=text_started_at,
            pending_messages=pending_messages,
            text_to_tts_interleaved=text_to_tts_interleaved,
            tts_prefix_cache_reset=tts_prefix_cache_reset,
            tts_started_at=tts_started_at,
            text=text,
            response_text=response_text,
            text_mlx_clear_cache_seconds=text_mlx_clear_cache_seconds,
            speech=speech,
            staged_voice_revision=staged_voice_revision,
            staged_sample_count=staged_sample_count,
        )

    async def _prepare_direct_audio_response_turn_async(
        self,
        *,
        samples: tuple[float, ...],
        sample_rate: int,
        play: bool,
        playback_start_gate: _PlaybackStartGate | None,
        staged_voice_revision: int | None,
        staged_sample_count: int | None,
    ) -> _PreparedSpokenTurn:
        """Answer audio first, then use Audex ASR while buffered speech plays."""

        if self.async_runtime is None:
            raise RuntimeError("Audex async vLLM runtime is not configured.")
        started_at = time.time()
        process_memory_before_text = _process_memory_snapshot()
        text_started_at = time.time()
        tts_started_at = text_started_at
        generation_finished = asyncio.Event()
        preemptive = staged_voice_revision is not None
        print(
            "Audex STS: generating a direct response from input speech with "
            "vLLM Metal...",
            flush=True,
        )

        async def transcribe_after_speech_generation() -> (
            tuple[float, float, VllmRequestResult, float]
        ):
            await generation_finished.wait()
            asr_started_at = time.time()
            print(
                (
                    "Audex STS: transcribing staged speech for the conversation..."
                    if preemptive
                    else "Audex STS: transcribing speech for the conversation..."
                ),
                flush=True,
            )
            asr = await self.async_runtime.transcribe_audio(
                samples,
                sample_rate=sample_rate,
            )
            asr_finished_at = time.time()
            clear_seconds = self._clear_mlx_cache()
            print(f"Audex STS: transcript: {asr.text}", flush=True)
            return asr_started_at, asr_finished_at, asr, clear_seconds

        asr_task = asyncio.create_task(transcribe_after_speech_generation())
        response_stream = self.async_runtime.stream_audio_response_from_messages(
            self.messages,
            samples,
            sample_rate=sample_rate,
            enable_reasoning=False,
            max_tokens=self.response_max_tokens,
            **self._audio_conversation_state_kwargs(),
        )
        try:
            text, speech = await self._generate_response_and_speech_interleaved(
                self.messages,
                fallback_text=None,
                play=play,
                playback_start_gate=playback_start_gate,
                generation_finished_event=generation_finished,
                response_stream=response_stream,
            )
            asr_started_at, asr_finished_at, asr, asr_clear_seconds = await asr_task
        except BaseException:
            asr_task.cancel()
            with suppress(BaseException):
                await asr_task
            raise

        transcript = asr.text
        response_text = scrub_spoken_answer(text.text)
        pending_messages = [*self.messages, {"role": "user", "content": transcript}]
        pending_messages = self._validate_text_prompt_messages(
            pending_messages,
            max_tokens=self.response_max_tokens,
        )
        return _PreparedSpokenTurn(
            started_at=started_at,
            asr_started_at=asr_started_at,
            asr_finished_at=asr_finished_at,
            asr=asr,
            transcript=transcript,
            asr_mlx_clear_cache_seconds=asr_clear_seconds,
            process_memory_before_text=process_memory_before_text,
            text_started_at=text_started_at,
            pending_messages=pending_messages,
            text_to_tts_interleaved=True,
            tts_prefix_cache_reset=False,
            tts_started_at=tts_started_at,
            text=text,
            response_text=response_text,
            text_mlx_clear_cache_seconds=self._clear_mlx_cache(),
            speech=speech,
            response_source="audio",
            staged_voice_revision=staged_voice_revision,
            staged_sample_count=staged_sample_count,
        )

    def _finish_prepared_spoken_turn(
        self,
        *,
        prepared: _PreparedSpokenTurn,
        input_wav_path: Path,
        play: bool,
        turn_submitted_at: float,
        final_voice_revision: int | None = None,
        final_sample_count: int | None = None,
    ) -> SpeechToSpeechTurnResult:
        speech = prepared.speech
        print(f"Audex STS: response text: {prepared.response_text}", flush=True)
        print(f"Audex STS: speech output ready: {speech.wav_path}", flush=True)
        self.messages = [
            *prepared.pending_messages,
            {"role": "assistant", "content": prepared.response_text},
        ]
        self.turns += 1
        self._persist_conversation()

        elapsed_seconds = round(time.time() - prepared.started_at, 3)
        run_log_path = (
            self.output_dir / f"sts-turn-vllm-{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
        semantic_audio = _semantic_audio_diagnostics(
            speech=speech,
            turn_submitted_at=turn_submitted_at,
            play=play,
            cfg_enabled=bool(getattr(self, "tts_cfg_enabled", False)),
            text_to_tts_interleaved=prepared.text_to_tts_interleaved,
        )
        run_log = {
            "backend": "vllm",
            "selected_model": self.selected_model_repo,
            "input_wav_path": str(input_wav_path),
            "output_wav_path": str(speech.wav_path),
            "speech_output_run_log_path": str(speech.run_log_path),
            "transcript": prepared.transcript,
            "response_text": prepared.response_text,
            "response_source": prepared.response_source,
            "played": play,
            "semantic_audio": semantic_audio,
            "preemptive_generation": {
                "enabled": prepared.staged_voice_revision is not None,
                "staged_voice_revision": prepared.staged_voice_revision,
                "final_voice_revision": final_voice_revision,
                "staged_sample_count": prepared.staged_sample_count,
                "final_sample_count": final_sample_count,
                "staged_before_submit_seconds": round(
                    max(0.0, turn_submitted_at - prepared.started_at),
                    3,
                ),
            },
            "response_max_tokens": self.response_max_tokens,
            "speech_max_tokens": self._speech_max_tokens_for_text(
                prepared.response_text
            ),
            "thinking_enabled": self.thinking_enabled,
            "text_to_tts_interleaved": prepared.text_to_tts_interleaved,
            "tts_prefix_cache_reset": prepared.tts_prefix_cache_reset,
            "text_to_tts_streaming": (
                dict(self._last_interleaved_text_stream_stats)
                if prepared.text_to_tts_interleaved
                else None
            ),
            "turn_index": self.turns,
            "conversation_turns": self.turns,
            "conversation_id": (
                self.conversation.conversation_id
                if self.conversation is not None
                else None
            ),
            "conversation_token_count": (
                self.conversation.token_count
                if self.conversation is not None
                else self._count_messages_tokens()
            ),
            "max_context_tokens": (
                self.conversation.max_context_tokens
                if self.conversation is not None
                else None
            ),
            "persona_id": self.persona.persona_id,
            "persona_path": str(self.persona.path),
            "elapsed_seconds": elapsed_seconds,
            "timings": {
                "elapsed_seconds": elapsed_seconds,
                "session_model_load_seconds": (
                    self._model_runtime_stats().model_load_seconds
                ),
                "session_audio_component_load_seconds": (
                    self.audio_component_load_seconds
                ),
                "session_decoder_load_seconds": self.decoder_load_seconds,
                "asr_elapsed_seconds": prepared.asr.elapsed_seconds,
                "asr_wall_seconds": round(
                    prepared.asr_finished_at - prepared.asr_started_at,
                    3,
                ),
                "asr_mlx_clear_cache_seconds": (prepared.asr_mlx_clear_cache_seconds),
                "text_elapsed_seconds": prepared.text.elapsed_seconds,
                "text_wall_seconds": round(
                    prepared.tts_started_at - prepared.text_started_at,
                    3,
                ),
                "text_mlx_clear_cache_seconds": (prepared.text_mlx_clear_cache_seconds),
                "tts_first_audio_ready_seconds": speech.first_audio_ready_seconds,
                "tts_first_playback_started_seconds": (
                    speech.first_playback_started_seconds
                ),
                "turn_submit_to_first_audio_ready_seconds": semantic_audio[
                    "turn_submit_to_first_audio_ready_seconds"
                ],
                "turn_submit_to_first_device_write_seconds": semantic_audio[
                    "turn_submit_to_first_device_write_seconds"
                ],
                "turn_submit_to_first_estimated_audible_seconds": semantic_audio[
                    "turn_submit_to_first_estimated_audible_seconds"
                ],
            },
            "process_memory": {
                "before_text": prepared.process_memory_before_text,
                "after_turn": _process_memory_snapshot(),
                "engine_core": _engine_core_memory_snapshots(),
            },
            "text_context": dict(self._last_text_context_stats),
            "vllm": {
                "engine_class": self._model_runtime_stats().engine_class,
                "asr_finish_reason": prepared.asr.finish_reason,
                "text_finish_reason": prepared.text.finish_reason,
                "tts_reached_end_token": speech.reached_end_token,
                "tts_hit_max_tokens": speech.hit_max_tokens,
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
        return SpeechToSpeechTurnResult(
            transcript=prepared.transcript,
            response_text=prepared.response_text,
            input_wav_path=input_wav_path,
            output_wav_path=speech.wav_path,
            run_log_path=run_log_path,
            played=play,
        )

    async def _run_turn_from_wav_async(
        self,
        *,
        input_wav_path: Path,
        play: bool = True,
        turn_submitted_at: float | None = None,
    ) -> SpeechToSpeechTurnResult:
        if self.async_runtime is None:
            raise RuntimeError("Audex async vLLM runtime is not configured.")

        input_audio = load_wav_pcm(input_wav_path)
        if input_audio.sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Audex vLLM STS input must be {SAMPLE_RATE} Hz PCM WAV, "
                f"got {input_audio.sample_rate} Hz."
            )
        prepared = await self._prepare_spoken_turn_async(
            samples=tuple(input_audio.samples),
            sample_rate=input_audio.sample_rate,
            play=play,
        )

        result = self._finish_prepared_spoken_turn(
            prepared=prepared,
            input_wav_path=input_wav_path,
            play=play,
            turn_submitted_at=turn_submitted_at or prepared.started_at,
        )
        await self._prime_audio_response_history_async()
        return result

    async def _generate_response_and_speech_interleaved(
        self,
        messages: list[dict[str, str]],
        *,
        fallback_text: str | None,
        play: bool,
        playback_start_gate: _PlaybackStartGate | None = None,
        generation_finished_event: asyncio.Event | None = None,
        response_stream: AsyncIterator[VllmStreamDelta] | None = None,
    ) -> tuple[VllmRequestResult, SpeechOutputSmokeResult]:
        if self.async_runtime is None:
            raise RuntimeError("Audex async vLLM runtime is not configured.")

        chunk_queue = _QueuedTtsChunkSource()
        final_text = ""
        final_token_ids: tuple[int, ...] = ()
        final_finish_reason: str | None = None
        final_request_debug_name = "text"
        final_elapsed_seconds = 0.0
        stream_started_at = time.time()
        text_stream_event_count = 0
        first_text_delta_seconds: float | None = None
        first_text_delta_wall_seconds: float | None = None
        last_text_delta_seconds: float | None = None
        last_text_delta_wall_seconds: float | None = None
        tts_text_chunk_count = 0
        first_text_to_tts_chunk_seconds: float | None = None
        last_text_to_tts_chunk_seconds: float | None = None
        text_to_tts_chunk_chars: list[int] = []

        async def stream_text_chunks() -> None:
            nonlocal final_text, final_token_ids, final_finish_reason
            nonlocal final_request_debug_name
            nonlocal final_elapsed_seconds
            nonlocal text_stream_event_count, first_text_delta_seconds
            nonlocal first_text_delta_wall_seconds, last_text_delta_seconds
            nonlocal last_text_delta_wall_seconds, tts_text_chunk_count
            nonlocal first_text_to_tts_chunk_seconds, last_text_to_tts_chunk_seconds
            emitted_chars = 0
            stream = response_stream
            if stream is None:
                stream = self.async_runtime.stream_text_response_from_messages(
                    messages,
                    enable_reasoning=self.thinking_enabled,
                    max_tokens=self.response_max_tokens,
                    **self._text_conversation_state_kwargs(),
                )
            async for delta in stream:
                text_stream_event_count += 1
                delta_wall_seconds = round(time.time() - stream_started_at, 3)
                if first_text_delta_seconds is None:
                    first_text_delta_seconds = delta.elapsed_seconds
                    first_text_delta_wall_seconds = delta_wall_seconds
                last_text_delta_seconds = delta.elapsed_seconds
                last_text_delta_wall_seconds = delta_wall_seconds
                clean_text = scrub_spoken_answer(delta.text)
                chunks, emitted_chars = streamed_tts_chunks_from_text(
                    clean_text,
                    emitted_chars,
                    final=delta.finished,
                )
                for chunk in chunks:
                    clean_chunk = scrub_spoken_answer(chunk)
                    if clean_chunk:
                        tts_text_chunk_seconds = round(
                            time.time() - stream_started_at,
                            3,
                        )
                        if first_text_to_tts_chunk_seconds is None:
                            first_text_to_tts_chunk_seconds = tts_text_chunk_seconds
                        last_text_to_tts_chunk_seconds = tts_text_chunk_seconds
                        tts_text_chunk_count += 1
                        text_to_tts_chunk_chars.append(len(clean_chunk))
                        await chunk_queue.put(clean_chunk)
                final_text = clean_text
                final_token_ids = delta.token_ids
                final_finish_reason = delta.finish_reason
                final_request_debug_name = delta.request_debug_name
                final_elapsed_seconds = delta.elapsed_seconds
            if not final_text.strip() and fallback_text:
                await chunk_queue.put(fallback_text)
            await chunk_queue.finish()

        text_task = asyncio.create_task(stream_text_chunks())
        speech_task = asyncio.create_task(
            self.generate_speech_output_streaming_from_async_runtime(
                text=fallback_text or "Response.",
                max_tokens=self._speech_max_tokens_for_text(
                    fallback_text or "Response."
                ),
                play=play,
                tts_chunk_source=chunk_queue,
                text_to_tts_interleaved=True,
                playback_start_gate=playback_start_gate,
                generation_finished_event=generation_finished_event,
            )
        )
        try:
            await text_task
            speech = await speech_task
        except BaseException:
            text_task.cancel()
            speech_task.cancel()
            with suppress(BaseException):
                await text_task
            with suppress(BaseException):
                await speech_task
            raise

        text = final_text.strip() or (fallback_text or "")
        if not text:
            raise RuntimeError("Audex audio-response request produced no spoken text.")
        self._last_interleaved_text_stream_stats = {
            "text_stream_event_count": text_stream_event_count,
            "first_text_delta_seconds": first_text_delta_seconds,
            "first_text_delta_wall_seconds": first_text_delta_wall_seconds,
            "last_text_delta_seconds": last_text_delta_seconds,
            "last_text_delta_wall_seconds": last_text_delta_wall_seconds,
            "tts_text_chunk_count": tts_text_chunk_count,
            "first_text_to_tts_chunk_seconds": first_text_to_tts_chunk_seconds,
            "last_text_to_tts_chunk_seconds": last_text_to_tts_chunk_seconds,
            "text_to_tts_chunk_chars": text_to_tts_chunk_chars,
        }
        return (
            VllmRequestResult(
                text=text,
                token_ids=final_token_ids,
                elapsed_seconds=final_elapsed_seconds,
                finish_reason=final_finish_reason,
                request_debug_name=final_request_debug_name,
            ),
            speech,
        )

    def project_wav_audio(self, input_wav_path: Path):
        self._ensure_audio_projection_components_loaded()
        loaded = load_wav_pcm(input_wav_path)
        if loaded.sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Audex WAV fixtures must be {SAMPLE_RATE} Hz, got {loaded.sample_rate}"
            )
        clips = prepare_audex_wav_clips(input_wav_path)
        features = extract_audex_input_features(
            clips,
            preprocessor_path=self.full_model_path / "audio_preprocessor",
        )
        feature_array = self.mx.array(features.input_features)
        feature_array = feature_array.astype(self.encoder_weights["conv1.weight"].dtype)
        encoder_hidden = encode_audio_features_mlx(
            feature_array,
            self.encoder_weights,
            self.encoder_config,
        )
        projected = project_audio_hidden_states_mlx(
            encoder_hidden,
            self.projector_weights,
            self.projector_config,
        )
        if projected.ndim == 3:
            clips_count, tokens_per_clip, hidden = projected.shape
            return self.mx.reshape(
                projected,
                (int(clips_count) * int(tokens_per_clip), int(hidden)),
            )
        return projected

    def _ensure_audio_projection_components_loaded(self) -> None:
        if (
            self.encoder_config is not None
            and self.encoder_weights is not None
            and self.projector_config is not None
            and self.projector_weights is not None
        ):
            return
        audio_started_at = time.time()
        self.encoder_config = load_audio_encoder_config(self.full_model_path)
        self.encoder_weights = load_audio_encoder_weights_mlx(self.full_model_path)
        self.projector_config = load_audio_projector_config(self.full_model_path)
        self.projector_weights = load_audio_projector_weights_mlx(self.full_model_path)
        self.audio_component_load_seconds = round(time.time() - audio_started_at, 3)

    def generate_speech_output(
        self,
        *,
        text: str,
        max_tokens: int,
        play: bool,
        artifact_prefix: str = "speech-output-vllm",
        decoder_chunk_frames: int = DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES,
        tts_target_segments: int = DEFAULT_VLLM_TTS_TARGET_SEGMENTS,
        tts_segments: tuple[str, ...] | None = None,
    ) -> SpeechOutputSmokeResult:
        if self.async_runtime is not None:
            return self._run_async(
                self.generate_speech_output_streaming_from_async_runtime(
                    text=text,
                    max_tokens=max_tokens,
                    play=play,
                    artifact_prefix=artifact_prefix,
                    decoder_chunk_frames=decoder_chunk_frames,
                    tts_target_segments=tts_target_segments,
                    tts_segments=tts_segments,
                )
            )

        started_at = time.time()
        runtime = self._sync_runtime()
        tts_result = runtime.generate_tts(
            text,
            max_tokens=max_tokens,
        )
        codec = runtime.extract_tts_codec_frames(tts_result)
        token_frames = tuple((frame,) for frame in codec.generated_codec_frames)
        if not token_frames:
            raise RuntimeError("Audex vLLM TTS produced no speech codec frames.")

        decoder_chunk_frames = max(1, int(decoder_chunk_frames))
        decoder_session = AudexSpeechDecoderSession(
            weights=self.decoder_weights,
            config=self.decoder_config,
            chunk_frames=decoder_chunk_frames,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        wav_path = self.output_dir / f"{artifact_prefix}-{timestamp}.wav"
        run_log_path = self.output_dir / f"{artifact_prefix}-{timestamp}.json"
        all_samples: list[float] = []
        chunk_paths: list[Path] = []
        peak_abs = 0.0
        first_audio_ready_seconds: float | None = None
        first_audio_ready_at: float | None = None

        def emit_waveform(waveform) -> None:
            nonlocal first_audio_ready_seconds, first_audio_ready_at, peak_abs
            peak_abs = max(
                peak_abs,
                float(self.mx.max(self.mx.abs(waveform)).item()),
            )
            samples = tuple(float(sample) for sample in waveform.tolist())
            all_samples.extend(samples)
            chunk_path = (
                self.output_dir
                / f"{artifact_prefix}-{timestamp}-chunk-{len(chunk_paths) + 1:04d}.wav"
            )
            write_pcm16_wav(
                chunk_path,
                samples,
                sample_rate=self.decoder_config.sample_rate,
            )
            chunk_paths.append(chunk_path)
            if first_audio_ready_seconds is None:
                first_audio_ready_at = time.time()
                first_audio_ready_seconds = round(
                    first_audio_ready_at - started_at,
                    3,
                )

        for sample_rate, waveform in decoder_session.push(token_frames):
            if sample_rate != self.decoder_config.sample_rate:
                raise ValueError(
                    "Audex decoder emitted unexpected sample rate "
                    f"{sample_rate}; expected {self.decoder_config.sample_rate}"
                )
            emit_waveform(waveform)
        for sample_rate, waveform in decoder_session.flush():
            if sample_rate != self.decoder_config.sample_rate:
                raise ValueError(
                    "Audex decoder emitted unexpected sample rate "
                    f"{sample_rate}; expected {self.decoder_config.sample_rate}"
                )
            emit_waveform(waveform)

        if not all_samples:
            raise RuntimeError("Audex vLLM decoder produced no waveform samples.")

        write_pcm16_wav(
            wav_path,
            all_samples,
            sample_rate=self.decoder_config.sample_rate,
        )
        finite = bool(all(sample == sample for sample in all_samples))
        first_playback_started_seconds = None
        first_playback_started_at: float | None = None
        if play:
            print(f"Audex STS: playing speech output: {wav_path}", flush=True)
            first_playback_started_at = time.time()
            play_wav(wav_path)
            first_playback_started_seconds = first_audio_ready_seconds

        run_log = {
            "backend": "vllm",
            "device": str(self.mx.default_device()),
            "full_model_path": str(self.full_model_path),
            "decoder_path": str(self.decoder_path),
            "streaming": False,
            "vllm_token_streaming": False,
            "decoder_streaming": True,
            "tts_cfg_enabled": False,
            "decoder_chunk_frames": decoder_chunk_frames,
            "chunk_wav_paths": [str(path) for path in chunk_paths],
            "prompt_tokens": 0,
            "generated_token_ids": list(codec.generated_token_ids),
            "generated_codec_frames": list(codec.generated_codec_frames),
            "reached_end_token": codec.reached_end_token,
            "hit_max_tokens": (
                len(codec.generated_token_ids) >= max_tokens
                and not codec.reached_end_token
            ),
            "waveform_shape": [len(all_samples)],
            "sample_rate": self.decoder_config.sample_rate,
            "hop_length": self.decoder_config.hop_length,
            "finite": finite,
            "peak_abs": peak_abs,
            "wav_path": str(wav_path),
            "first_audio_ready_seconds": first_audio_ready_seconds,
            "first_playback_started_seconds": first_playback_started_seconds,
            "first_audio_ready_at": first_audio_ready_at,
            "first_playback_started_at": first_playback_started_at,
        }
        run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
        return SpeechOutputSmokeResult(
            backend="vllm",
            device=str(self.mx.default_device()),
            prompt_tokens=0,
            generated_token_ids=codec.generated_token_ids,
            generated_codec_frames=codec.generated_codec_frames,
            reached_end_token=codec.reached_end_token,
            hit_max_tokens=(
                len(codec.generated_token_ids) >= max_tokens
                and not codec.reached_end_token
            ),
            waveform_shape=(len(all_samples),),
            sample_rate=self.decoder_config.sample_rate,
            hop_length=self.decoder_config.hop_length,
            finite=finite,
            peak_abs=peak_abs,
            wav_path=wav_path,
            run_log_path=run_log_path,
            streaming=False,
            chunk_wav_paths=tuple(chunk_paths),
            first_audio_ready_seconds=first_audio_ready_seconds,
            first_playback_started_seconds=first_playback_started_seconds,
            first_audio_ready_at=first_audio_ready_at,
            first_playback_started_at=first_playback_started_at,
        )

    async def generate_speech_output_streaming_from_async_runtime(
        self,
        *,
        text: str,
        max_tokens: int,
        play: bool,
        artifact_prefix: str = "speech-output-vllm",
        decoder_chunk_frames: int = DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES,
        decoder_steady_chunk_frames: int | None = None,
        tts_target_segments: int = DEFAULT_VLLM_TTS_TARGET_SEGMENTS,
        tts_chunk_source: AsyncIterator[str] | None = None,
        text_to_tts_interleaved: bool = False,
        tts_segments: tuple[str, ...] | None = None,
        playback_start_gate: _PlaybackStartGate | None = None,
        generation_finished_event: asyncio.Event | None = None,
    ) -> SpeechOutputSmokeResult:
        if self.async_runtime is None:
            raise RuntimeError("Audex async vLLM runtime is not configured.")
        configured_tts_cfg = getattr(self, "tts_cfg_enabled", None)
        synthesizer = VllmSpeechSynthesizer(
            async_runtime=self.async_runtime,
            mx=self.mx,
            full_model_path=self.full_model_path,
            decoder_path=self.decoder_path,
            decoder_config=self.decoder_config,
            decoder_weights=self.decoder_weights,
            output_dir=self.output_dir,
            tokenizer=self._model_tokenizer(),
            speech_max_tokens=self.speech_max_tokens,
            tts_cfg_enabled=(
                _vllm_tts_cfg_enabled()
                if configured_tts_cfg is None
                else bool(configured_tts_cfg)
            ),
            decoder_session_factory=self._new_speech_decoder_session,
            player_factory=_ContinuousPcmPlayer,
            decoder_device=getattr(self, "decoder_device", None),
        )
        return await synthesizer.synthesize(
            SpeechSynthesisRequest(
                text=text,
                max_tokens=max_tokens,
                play=play,
                artifact_prefix=artifact_prefix,
                decoder_chunk_frames=decoder_chunk_frames,
                decoder_steady_chunk_frames=decoder_steady_chunk_frames,
                tts_target_segments=tts_target_segments,
                tts_chunk_source=tts_chunk_source,
                text_to_tts_interleaved=text_to_tts_interleaved,
                tts_segments=tts_segments,
                playback_start_gate=playback_start_gate,
                generation_finished_event=generation_finished_event,
            )
        )

    def speak_startup_greeting(
        self,
        *,
        conversation_resumed: bool,
        play: bool,
    ) -> None:
        greeting = startup_greeting_text(
            conversation=self.conversation,
            conversation_resumed=conversation_resumed,
        )
        if conversation_resumed and _has_substantive_conversation_history(
            self.messages
        ):
            greeting_messages = [
                *self.messages,
                {"role": "user", "content": RESUME_STARTUP_GREETING_PROMPT},
            ]
            greeting_messages = self._validate_text_prompt_messages(
                greeting_messages,
                max_tokens=RESUME_STARTUP_GREETING_MAX_TOKENS,
            )
            result = self._generate_text_response_from_messages(
                greeting_messages,
                enable_reasoning=False,
                max_tokens=RESUME_STARTUP_GREETING_MAX_TOKENS,
                **self._text_conversation_state_kwargs(),
            )
            generated_greeting = _limit_spoken_sentences(
                scrub_spoken_answer(result.text),
                max_sentences=2,
            )
            if generated_greeting:
                greeting = generated_greeting
            self._clear_mlx_cache()
        print(f"Audex STS: startup greeting: {greeting}", flush=True)
        if play:
            self.generate_speech_output(
                text=greeting,
                max_tokens=self._speech_max_tokens_for_text(greeting),
                play=True,
                artifact_prefix="startup-greeting-vllm",
                tts_target_segments=1,
            )

    def _speech_max_tokens_for_text(self, text: str) -> int:
        if self.speech_max_tokens is not None:
            return self.speech_max_tokens
        text_tokens = max(1, len(self._model_tokenizer().encode(text)))
        return max(
            DEFAULT_S2S_TTS_MAX_TOKENS,
            text_tokens * DEFAULT_SPEECH_TOKENS_PER_TEXT_TOKEN,
        )

    def _speech_max_tokens_for_tts_chunk(
        self,
        text: str,
        *,
        utterance_max_tokens: int,
        chunk_count: int,
    ) -> int:
        if self.speech_max_tokens is not None or chunk_count <= 1:
            return utterance_max_tokens
        text_tokens = max(1, len(self._model_tokenizer().encode(text)))
        return max(
            DEFAULT_VLLM_TTS_MIN_TOKENS_PER_CHUNK,
            text_tokens * DEFAULT_SPEECH_TOKENS_PER_TEXT_TOKEN,
        )

    def _count_messages_tokens(self) -> int:
        return self._count_prompt_tokens(self.messages)

    def _count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        tokenizer = self._model_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=self.thinking_enabled,
        )
        return len(tokenizer.encode(prompt))

    def _count_text_generation_prompt_tokens(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
    ) -> int:
        tokenizer = self._model_tokenizer()
        request = build_text_messages_response_request(
            tokenizer,
            messages,
            enable_reasoning=self.thinking_enabled,
            max_tokens=max_tokens,
        )
        return len(tokenizer.encode(str(request.prompt)))

    def _validate_text_prompt_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
    ) -> list[dict[str, str]]:
        limit = self._text_context_token_limit()
        response_tokens = max(1, int(max_tokens))
        budget = max(1, limit - response_tokens)
        checked = [dict(message) for message in messages]
        prompt_tokens = self._count_text_generation_prompt_tokens(
            checked,
            max_tokens=response_tokens,
        )
        self._last_text_context_stats = {
            "fits": prompt_tokens <= budget,
            "messages_before": len(messages),
            "prompt_tokens": prompt_tokens,
            "prompt_token_budget": budget,
            "context_token_limit": limit,
            "response_max_tokens": response_tokens,
        }
        if prompt_tokens <= budget:
            return checked
        raise RuntimeError(
            "Audex STS text prompt exceeds the current vLLM Metal engine "
            "context budget. "
            f"prompt_tokens={prompt_tokens}, response_max_tokens={response_tokens}, "
            f"engine_max_model_len={limit}, prompt_budget={budget}. "
            "Audex-Mac will not silently drop conversation history. This is the "
            "configured Mac-demo context limit. Audex-Mac will preserve the "
            "full transcript and will not compact or silently drop conversation "
            "history. Start a new conversation or lower response tokens."
        )

    def _text_context_token_limit(self) -> int:
        cfg_config = getattr(self.async_runtime or self.runtime, "cfg_config", None)
        max_model_len = getattr(cfg_config, "max_model_len", None)
        if isinstance(max_model_len, int) and max_model_len > 0:
            runtime_limit = max_model_len
        else:
            stats = self._model_runtime_stats()
            stats_limit = getattr(stats, "max_model_len", None)
            runtime_limit = (
                stats_limit
                if isinstance(stats_limit, int) and stats_limit > 0
                else DEFAULT_VLLM_TEXT_CONTEXT_TOKENS
            )
        conversation_limit = (
            self.conversation.max_context_tokens
            if self.conversation is not None
            else DEFAULT_VLLM_TEXT_CONTEXT_TOKENS
        )
        return min(
            runtime_limit,
            max(1, int(conversation_limit)),
        )

    def _align_conversation_context_limit(self) -> None:
        """Keep persisted conversation metadata within the loaded model limit."""

        if self.conversation is None:
            return
        runtime_limit = getattr(self._model_runtime_stats(), "max_model_len", None)
        if not isinstance(runtime_limit, int) or runtime_limit <= 0:
            return
        effective_limit = min(self.conversation.max_context_tokens, runtime_limit)
        if effective_limit == self.conversation.max_context_tokens:
            return
        self.conversation.max_context_tokens = effective_limit
        if self.conversation_store is not None:
            self.conversation_store.save(self.conversation)

    def _model_runtime_stats(self):
        if self.runtime is not None:
            return self.runtime.stats
        if self.async_runtime is not None:
            return self.async_runtime.stats
        raise RuntimeError("Audex vLLM runtime is not configured.")

    def _model_tokenizer(self):
        if self.runtime is not None:
            return self.runtime.tokenizer
        if self.async_runtime is not None:
            return self.async_runtime.tokenizer
        raise RuntimeError("Audex vLLM runtime is not configured.")

    def _run_async(self, coroutine):
        if self._async_loop is None:
            return asyncio.run(coroutine)
        if self._async_loop.is_closed():
            raise RuntimeError("Audex async vLLM event loop is already closed.")
        return self._async_loop.run_until_complete(coroutine)

    def run_model_awaitable(self, awaitable: Any) -> Any:
        """Run Sound Lab work on the same loop that owns the shared vLLM engine."""

        return self._run_async(awaitable)

    def _close_async_loop(self) -> None:
        if self._async_loop is None or self._async_loop.is_closed():
            return
        pending = [
            task for task in asyncio.all_tasks(self._async_loop) if not task.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            self._async_loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._async_loop.run_until_complete(self._async_loop.shutdown_asyncgens())
        self._async_loop.close()

    def _sync_runtime(self) -> AudexVllmRuntime:
        if self.runtime is None:
            raise RuntimeError("Audex sync vLLM runtime is not configured.")
        return self.runtime

    def _transcribe_projected_audio(
        self,
        projected_embeddings,
        *,
        num_embeddings: int,
    ):
        if self.runtime is not None:
            return self.runtime.transcribe_projected_audio(
                projected_embeddings,
                num_embeddings=num_embeddings,
            )
        if self.async_runtime is None:
            raise RuntimeError("Audex vLLM runtime is not configured.")
        return self._run_async(
            self.async_runtime.transcribe_projected_audio(
                projected_embeddings,
                num_embeddings=num_embeddings,
            )
        )

    def _transcribe_audio(
        self,
        audio_samples: Any,
        *,
        sample_rate: int,
    ):
        if self.runtime is not None:
            return self.runtime.transcribe_audio(
                audio_samples,
                sample_rate=sample_rate,
            )
        if self.async_runtime is None:
            raise RuntimeError("Audex vLLM runtime is not configured.")
        return self._run_async(
            self.async_runtime.transcribe_audio(
                audio_samples,
                sample_rate=sample_rate,
            )
        )

    def _generate_text_response_from_messages(
        self,
        messages: list[dict[str, str]],
        *,
        enable_reasoning: bool,
        max_tokens: int | None,
        conversation_state_key: str | None = None,
        conversation_state_boundary: str | None = None,
        conversation_state_prefix_token_count: int | None = None,
        conversation_state_prefix_token_hash: str | None = None,
    ):
        if self.runtime is not None:
            return self.runtime.generate_text_response_from_messages(
                messages,
                enable_reasoning=enable_reasoning,
                max_tokens=max_tokens,
                conversation_state_key=conversation_state_key,
                conversation_state_boundary=conversation_state_boundary,
                conversation_state_prefix_token_count=conversation_state_prefix_token_count,
                conversation_state_prefix_token_hash=conversation_state_prefix_token_hash,
            )
        if self.async_runtime is None:
            raise RuntimeError("Audex vLLM runtime is not configured.")
        return self._run_async(
            self.async_runtime.generate_text_response_from_messages(
                messages,
                enable_reasoning=enable_reasoning,
                max_tokens=max_tokens,
                conversation_state_key=conversation_state_key,
                conversation_state_boundary=conversation_state_boundary,
                conversation_state_prefix_token_count=conversation_state_prefix_token_count,
                conversation_state_prefix_token_hash=conversation_state_prefix_token_hash,
            )
        )

    def _text_conversation_state_kwargs(self) -> dict[str, object]:
        if self.conversation is None:
            return {}
        tokens = self._text_history_prompt_tokens()
        return {
            "conversation_state_key": self.conversation.conversation_id,
            "conversation_state_prefix_token_count": len(tokens),
            "conversation_state_prefix_token_hash": _token_hash(tokens),
        }

    def _audio_conversation_state_kwargs(self) -> dict[str, object]:
        if self._audio_conversation_state_cache is not None:
            return dict(self._audio_conversation_state_cache)
        tokens = self._audio_history_prompt_tokens()
        state = {
            "conversation_state_key": self._audio_response_state_key(),
            "conversation_state_prefix_token_count": len(tokens),
            "conversation_state_prefix_token_hash": _token_hash(tokens),
        }
        self._audio_conversation_state_cache = state
        return dict(state)

    def _audio_response_state_key(self) -> str:
        if self.conversation is not None:
            return f"{self.conversation.conversation_id}:audio-response"
        return f"session-{id(self):x}:audio-response"

    def _audio_history_prompt_tokens(self) -> tuple[int, ...]:
        return build_audio_response_prefix_token_ids(
            self._model_tokenizer(),
            self.messages,
        )

    async def _prime_audio_response_history_async(self) -> None:
        if self.async_runtime is None or not _direct_audio_response_enabled(
            thinking_enabled=self.thinking_enabled
        ):
            return
        self._audio_conversation_state_cache = None
        kwargs = self._audio_conversation_state_kwargs()
        await self.async_runtime.prime_audio_response_history(
            self.messages,
            conversation_state_key=str(kwargs["conversation_state_key"]),
            conversation_state_prefix_token_count=int(
                kwargs["conversation_state_prefix_token_count"]
            ),
            conversation_state_prefix_token_hash=str(
                kwargs["conversation_state_prefix_token_hash"]
            ),
        )
        # AsyncLLM can deliver the terminal token just before EngineCore's next
        # cleanup pass publishes the retained prefix snapshot.
        await asyncio.sleep(0.05)

    def _text_history_prompt_tokens(self) -> tuple[int, ...]:
        tokenizer = self._model_tokenizer()
        prompt = build_text_messages_history_prompt(
            tokenizer,
            self.messages,
            enable_reasoning=self.thinking_enabled,
        )
        return tuple(int(token_id) for token_id in tokenizer.encode(prompt))

    def _persist_conversation(
        self,
        *,
        announce: bool = True,
        invalidate_kv_cache: bool = False,
    ) -> None:
        if self.conversation is None or self.conversation_store is None:
            return
        token_count = self._count_messages_tokens()
        self.conversation.messages = list(self.messages)
        self.conversation.token_count = token_count
        if invalidate_kv_cache:
            self._invalidate_conversation_kv_cache()
        self.conversation_store.save(self.conversation)
        if not announce:
            return
        print(
            "Conversation tokens: "
            f"{token_count}/{self.conversation.max_context_tokens}",
            flush=True,
        )
        if token_count >= self.conversation.max_context_tokens:
            print(
                "WARNING: conversation token count has reached the configured "
                "context limit; start a new conversation soon.",
                flush=True,
            )

    def _invalidate_conversation_kv_cache(self) -> None:
        if self.conversation is None:
            return
        kv_cache_path = (
            self.conversation.root
            / f"{self.conversation.conversation_id}.kv.safetensors"
        )
        if not kv_cache_path.is_file():
            return
        with suppress(OSError):
            kv_cache_path.unlink()
            print(
                "Audex STS: invalidated stale conversation KV cache after "
                f"sanitizing resumed history: {kv_cache_path}",
                flush=True,
            )

    def _clear_mlx_cache(self) -> float:
        clear_cache = getattr(self.mx, "clear_cache", None)
        if not callable(clear_cache):
            return 0.0
        started = time.time()
        with suppress(Exception):
            clear_cache()
        gc.collect()
        return round(time.time() - started, 6)

    def _mlx_memory_snapshot(self) -> dict[str, int]:
        snapshot: dict[str, int] = {}
        for key, method_name in (
            ("active_bytes", "get_active_memory"),
            ("cache_bytes", "get_cache_memory"),
            ("peak_bytes", "get_peak_memory"),
        ):
            method = getattr(self.mx, method_name, None)
            if not callable(method):
                continue
            with suppress(Exception):
                snapshot[key] = int(method())
        return snapshot


def _process_memory_snapshot() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "pid": os.getpid(),
        "max_rss_bytes": _ru_maxrss_to_bytes(int(usage.ru_maxrss)),
    }


def _engine_core_memory_snapshots() -> tuple[dict[str, int | str], ...]:
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss=,comm="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    snapshots: list[dict[str, int | str]] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) != 4:
            continue
        pid, ppid, rss_kib, command = parts
        if "VLLM::EngineCore" not in command:
            continue
        with suppress(ValueError):
            snapshots.append(
                {
                    "pid": int(pid),
                    "ppid": int(ppid),
                    "rss_bytes": int(rss_kib) * 1024,
                    "command": command,
                }
            )
    return tuple(snapshots)


def _ru_maxrss_to_bytes(ru_maxrss: int) -> int:
    if os.uname().sysname == "Darwin":
        return ru_maxrss
    return ru_maxrss * 1024


def _vllm_tts_cfg_enabled() -> bool:
    value = os.environ.get(VLLM_TTS_CFG_ENV)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _reset_prefix_cache_before_tts_enabled() -> bool:
    value = os.environ.get(VLLM_RESET_PREFIX_CACHE_BEFORE_TTS_ENV)
    if value is None:
        return _vllm_tts_cfg_enabled()
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _enable_cfg_wiring_if_tts_cfg_requested() -> None:
    if _vllm_tts_cfg_enabled():
        os.environ[VLLM_CFG_WIRING_ENV] = "1"


def _text_to_tts_interleaving_enabled(*, thinking_enabled: bool) -> bool:
    if thinking_enabled:
        return False
    value = os.environ.get("AUDEX_VLLM_STREAM_TEXT_TO_TTS")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _direct_audio_response_enabled(*, thinking_enabled: bool) -> bool:
    if thinking_enabled:
        return False
    value = os.environ.get(VLLM_DIRECT_AUDIO_RESPONSE_ENV)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _preemptive_generation_enabled() -> bool:
    value = os.environ.get("AUDEX_VLLM_PREEMPTIVE_GENERATION")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _semantic_audio_diagnostics(
    *,
    speech: SpeechOutputSmokeResult,
    turn_submitted_at: float | None,
    play: bool,
    cfg_enabled: bool,
    text_to_tts_interleaved: bool,
) -> dict[str, object]:
    first_audio_ready_at = getattr(speech, "first_audio_ready_at", None)
    first_device_write_at = getattr(speech, "first_playback_started_at", None)
    first_estimated_audible_at = getattr(
        speech,
        "first_playback_estimated_audible_at",
        None,
    )
    ready_latency = _elapsed_from_timestamp(turn_submitted_at, first_audio_ready_at)
    device_write_latency = _elapsed_from_timestamp(
        turn_submitted_at,
        first_device_write_at,
    )
    estimated_audible_latency = _elapsed_from_timestamp(
        turn_submitted_at,
        first_estimated_audible_at,
    )
    gate_latency = (
        estimated_audible_latency
        if estimated_audible_latency is not None
        else device_write_latency
    )
    segments = tuple(getattr(speech, "segments", ()) or ())
    first_spoken_chunk_text = segments[0].strip() if segments else None
    gate_eligible = bool(play and first_spoken_chunk_text and gate_latency is not None)
    return {
        "source": "model_response_tts",
        "first_spoken_chunk_text": first_spoken_chunk_text,
        "cfg_enabled": cfg_enabled,
        "text_to_tts_interleaved": text_to_tts_interleaved,
        "measurement_endpoint": (
            "estimated_first_dac_sample"
            if estimated_audible_latency is not None
            else "first_sounddevice_stream_write"
        ),
        "turn_submit_to_first_audio_ready_seconds": ready_latency,
        "turn_submit_to_first_device_write_seconds": device_write_latency,
        "turn_submit_to_first_estimated_audible_seconds": (estimated_audible_latency),
        "gate_seconds": SEMANTIC_AUDIO_GATE_SECONDS,
        "gate_eligible": gate_eligible,
        "gate_passed": (
            gate_latency < SEMANTIC_AUDIO_GATE_SECONDS
            if gate_eligible and gate_latency is not None
            else None
        ),
    }


def _wait_for_turn_submission() -> float:
    with suppress(EOFError):
        input()
    return time.time()


def _elapsed_from_timestamp(
    started_at: float | None,
    finished_at: float | None,
) -> float | None:
    if started_at is None or finished_at is None:
        return None
    return round(max(0.0, finished_at - started_at), 3)


def _sanitize_prompt_history(
    messages: list[dict[str, str]],
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Remove prior prompt leakage before using persisted turns as context."""

    sanitized: list[dict[str, str]] = []
    if system_prompt is not None:
        sanitized.append({"role": "system", "content": system_prompt})
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system" and system_prompt is not None:
            continue
        if role == "assistant":
            content = scrub_spoken_answer(content)
            if not content:
                continue
        sanitized.append({"role": role, "content": content})
    return sanitized


def _token_hash(tokens: tuple[int, ...]) -> str:
    digest = sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()
