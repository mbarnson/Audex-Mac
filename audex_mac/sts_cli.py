"""Push-to-talk Audex speech-to-speech CLI."""

from __future__ import annotations

import json
import math
import queue
import re
import subprocess
import threading
import time
from array import array
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .audio_contract import (
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    NVIDIA_ASR_TEMPERATURE,
    NVIDIA_ASR_TOP_P,
    NVIDIA_TEXT_TEMPERATURE,
    NVIDIA_TEXT_TOP_P,
    NVIDIA_TTS_CFG_SCALE,
    NVIDIA_TTS_TEMPERATURE,
    NVIDIA_TTS_TOP_K,
    NVIDIA_TTS_TOP_P,
    SPEECHGEN_START_TOKEN,
    build_audio_chat_prompt,
    build_audio_prompt_plan,
    build_codec_token_map,
    build_tts_null_prompt,
    build_tts_prompt,
    expand_sound_placeholder,
    tokenize_tts_cfg_pair,
)
from .audio_encoder import (
    encode_audio_features_mlx,
    load_audio_encoder_config,
    load_audio_encoder_weights_mlx,
)
from .audio_features import extract_audex_input_features
from .audio_pcm import SAMPLE_RATE, prepare_audex_wav_clips
from .audio_projector import (
    load_audio_projector_config,
    load_audio_projector_weights_mlx,
    project_audio_hidden_states_mlx,
)
from .audio_splice import splice_audio_embeddings_mlx, validate_audio_splice_plan
from .conversations import (
    DEFAULT_DEMO_CONTEXT_TOKENS,
    Conversation,
    ConversationStore,
)
from .interactive_input import InputKind, read_turn_input
from .patches import apply_audex_runtime_patches
from .personas import DEFAULT_PERSONA_NAME, Persona, load_persona
from .speech_decoder import (
    AudexSpeechDecoderSession,
    decode_speech_token_frames_mlx,
    load_speech_decoder_config,
    load_speech_decoder_weights_mlx,
)
from .speech_generation import (
    SpeechTokenGenerationSmokeResult,
    _generate_tts_cfg_token_ids,
)
from .speech_output import (
    RUNS_DIR,
    float_samples_to_pcm16_bytes,
    run_speech_output_smoke,
    write_pcm16_wav,
)
from .text_generation import STOP_MARKERS, clean_generation

TEXT_VOCAB = 131_072
SOUND_TOKEN_ID = 29
SOUND_START_TOKEN_ID = 30
SOUND_END_TOKEN_ID = 31
DEFAULT_RESPONSE_MAX_TOKENS = 4096
DEFAULT_S2S_TTS_MAX_TOKENS = 2400
DEFAULT_SPEECH_TOKENS_PER_TEXT_TOKEN = 64
DEFAULT_ASR_MAX_TOKENS = 2048
DEFAULT_MAX_CONTEXT_TOKENS = DEFAULT_DEMO_CONTEXT_TOKENS
DEFAULT_STREAM_DECODER_CHUNK_FRAMES = 5
DEFAULT_PLAYBACK_PREBUFFER_SECONDS = 0.8
DEFAULT_PLAYBACK_LATENCY = "low"
DEFAULT_PLAYBACK_QUEUE_OVERRUN_SECONDS = 2.0
DEFAULT_PLAYBACK_ADAPTIVE_MIN_PREBUFFER_SECONDS = 0.08
DEFAULT_PLAYBACK_ADAPTIVE_MAX_PREBUFFER_SECONDS = 8.0
DEFAULT_PLAYBACK_ARRIVAL_RATE_WINDOW_SECONDS = 0.25
DEFAULT_PLAYBACK_ARRIVAL_RATE_EMA_ALPHA = 0.25
PLAYBACK_QUEUE_UNDERRUN_GRACE_SECONDS = 0.02
STREAMING_TTS_START_SENTENCES = 3
TTS_SEGMENT_SILENCE_SECONDS = 0.2
TTS_WARMUP_TEXT = "Warm up."
FIRST_STARTUP_GREETING_TEXT = (
    "Hi! I'm Audex. What should I call you, and what should we talk about today?"
)
RESUME_UNKNOWN_USER_GREETING_TEXT = (
    "Hi! Nice to hear from you again. What do you want to talk about today?"
)


@dataclass(frozen=True, slots=True)
class SpeechToSpeechTurnResult:
    transcript: str
    response_text: str
    input_wav_path: Path | None
    output_wav_path: Path
    run_log_path: Path
    played: bool


@dataclass(frozen=True, slots=True)
class AudexSpeechToSpeechSessionStats:
    model_load_seconds: float
    audio_component_load_seconds: float
    decoder_load_seconds: float
    speech_warmup_seconds: float
    turns: int


class AudexSpeechToSpeechSession:
    """Reusable MLX Audex state for conversational speech-to-speech turns."""

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
        enable_kv_cache: bool = True,
    ) -> None:
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.generate import generate_step
            from mlx_lm.models import cache
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError(
                "Audex STS session requires mlx, mlx_lm, and transformers."
            ) from exc

        self.full_model_path = full_model_path
        self.decoder_path = decoder_path
        self.selected_model_repo = selected_model_repo
        self.output_dir = output_dir
        self.thinking_enabled = thinking_enabled
        self.response_max_tokens = response_max_tokens
        self.speech_max_tokens = speech_max_tokens
        self.persona = persona or load_persona(DEFAULT_PERSONA_NAME)
        self.conversation_store = conversation_store
        self.conversation = conversation
        self.enable_kv_cache = enable_kv_cache
        self.loaded_conversation_kv_cache = False
        self.text_prompt_cache = None
        self.text_prompt_cache_token_count: int | None = None
        self.text_prompt_cache_token_hash: str | None = None
        if conversation is not None:
            self.messages = list(conversation.messages)
        else:
            self.messages = [{"role": "system", "content": self.persona.system_prompt}]
        self.turns = 0

        mx.set_default_device(mx.gpu)
        apply_audex_runtime_patches()
        self.mx = mx
        self.generate_step = generate_step
        self.cache_module = cache
        self.make_sampler = make_sampler

        model_started_at = time.time()
        self.model, self.tokenizer = load(
            str(full_model_path),
            tokenizer_config={"trust_remote_code": True},
        )
        template_path = full_model_path / "chat_template.jinja"
        if template_path.is_file():
            self.tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        self.model_load_seconds = round(time.time() - model_started_at, 3)
        self._load_conversation_prompt_cache()

        audio_started_at = time.time()
        self.encoder_config = load_audio_encoder_config(full_model_path)
        self.encoder_weights = load_audio_encoder_weights_mlx(full_model_path)
        self.projector_config = load_audio_projector_config(full_model_path)
        self.projector_weights = load_audio_projector_weights_mlx(full_model_path)
        self.audio_component_load_seconds = round(time.time() - audio_started_at, 3)

        decoder_started_at = time.time()
        self.decoder_config = load_speech_decoder_config(decoder_path)
        self.decoder_weights = load_speech_decoder_weights_mlx(decoder_path)
        self.decoder_load_seconds = round(time.time() - decoder_started_at, 3)

        self.token_map = build_codec_token_map(self.tokenizer.get_vocab())
        speechgen_start_id = int(
            self.tokenizer.convert_tokens_to_ids(SPEECHGEN_START_TOKEN)
        )
        if speechgen_start_id != self.token_map.speechgen_start:
            raise ValueError(
                f"Tokenizer mismatch for {SPEECHGEN_START_TOKEN}: "
                f"encode={speechgen_start_id} vocab={self.token_map.speechgen_start}"
            )
        warmup_started_at = time.time()
        self._warm_speech_output_path()
        self.speech_warmup_seconds = round(time.time() - warmup_started_at, 3)

    @property
    def stats(self) -> AudexSpeechToSpeechSessionStats:
        return AudexSpeechToSpeechSessionStats(
            model_load_seconds=self.model_load_seconds,
            audio_component_load_seconds=self.audio_component_load_seconds,
            decoder_load_seconds=self.decoder_load_seconds,
            speech_warmup_seconds=self.speech_warmup_seconds,
            turns=self.turns,
        )

    def run_turn_from_wav(
        self,
        *,
        input_wav_path: Path,
        play: bool = True,
    ) -> SpeechToSpeechTurnResult:
        started_at = time.time()
        print("Audex STS: transcribing input speech...", flush=True)
        transcript = self.transcribe_wav(input_wav_path)
        print(f"Audex STS: transcript: {transcript}", flush=True)
        print("Audex STS: generating text response...", flush=True)
        tts_started_at = time.time()
        if play:
            response_text, speech = self._generate_response_and_stream_speech(
                transcript,
            )
        else:
            response_text = self.generate_text_response(transcript)
            print("Audex STS: streaming speech output...", flush=True)
            speech = self.generate_speech_output_streaming(
                text=response_text or transcript or "I heard your message.",
                max_tokens_per_segment=self._speech_max_tokens_for_text(response_text),
                play=False,
                first_chunk_label="Audex STS: first answer chunk ready",
            )
        speech_max_tokens = self._speech_max_tokens_for_text(response_text)
        print(f"Audex STS: speech output ready: {speech.wav_path}", flush=True)
        played = play

        self.turns += 1
        run_log_path = (
            self.output_dir / f"sts-turn-{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
        elapsed_seconds = round(time.time() - started_at, 3)
        first_audio_ready_seconds = (
            round(tts_started_at - started_at + speech.first_audio_ready_seconds, 3)
            if speech.first_audio_ready_seconds is not None
            else None
        )
        first_playback_started_seconds = (
            round(
                tts_started_at - started_at + speech.first_playback_started_seconds,
                3,
            )
            if speech.first_playback_started_seconds is not None
            else None
        )
        run_log = {
            "selected_model": self.selected_model_repo,
            "input_wav_path": str(input_wav_path),
            "output_wav_path": str(speech.wav_path),
            "speech_output_run_log_path": str(speech.run_log_path),
            "acknowledgement_output_wav_path": None,
            "acknowledgement_output_run_log_path": None,
            "transcript": transcript,
            "response_text": response_text,
            "played": played,
            "response_max_tokens": self.response_max_tokens,
            "speech_max_tokens": speech_max_tokens,
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
                else self._count_messages_tokens(self.messages)
            ),
            "max_context_tokens": (
                self.conversation.max_context_tokens
                if self.conversation is not None
                else DEFAULT_MAX_CONTEXT_TOKENS
            ),
            "persona_id": self.persona.persona_id,
            "persona_path": str(self.persona.path),
            "kv_cache_path": (
                str(self._conversation_kv_cache_path())
                if self.conversation is not None and self.enable_kv_cache
                else None
            ),
            "elapsed_seconds": elapsed_seconds,
            "timings": {
                "elapsed_seconds": elapsed_seconds,
                "session_model_load_seconds": self.model_load_seconds,
                "session_audio_component_load_seconds": (
                    self.audio_component_load_seconds
                ),
                "session_decoder_load_seconds": self.decoder_load_seconds,
                "session_speech_warmup_seconds": self.speech_warmup_seconds,
                "first_audio_ready_seconds": first_audio_ready_seconds,
                "first_playback_started_seconds": first_playback_started_seconds,
                "ack_first_audio_ready_seconds": None,
                "ack_first_playback_started_seconds": None,
                "tts_first_audio_ready_seconds": speech.first_audio_ready_seconds,
                "tts_first_playback_started_seconds": (
                    speech.first_playback_started_seconds
                ),
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
            played=played,
        )

    def run_turn_from_text(
        self,
        *,
        user_text: str,
        play: bool = True,
    ) -> SpeechToSpeechTurnResult:
        """Skip ASR and run typed text through response generation and TTS."""

        if not user_text.strip():
            raise ValueError("Typed Audex input must not be empty.")
        started_at = time.time()
        print("Audex STS: generating text response from typed input...", flush=True)
        tts_started_at = time.time()
        if play:
            response_text, speech = self._generate_response_and_stream_speech(user_text)
        else:
            response_text = self.generate_text_response(user_text)
            print("Audex STS: streaming speech output...", flush=True)
            speech = self.generate_speech_output_streaming(
                text=response_text or user_text,
                max_tokens_per_segment=self._speech_max_tokens_for_text(response_text),
                play=False,
                first_chunk_label="Audex STS: first answer chunk ready",
            )
        print(f"Audex STS: speech output ready: {speech.wav_path}", flush=True)

        self.turns += 1
        elapsed_seconds = round(time.time() - started_at, 3)
        run_log_path = (
            self.output_dir / f"sts-turn-{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
        run_log = {
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
                else self._count_messages_tokens(self.messages)
            ),
            "max_context_tokens": (
                self.conversation.max_context_tokens
                if self.conversation is not None
                else DEFAULT_MAX_CONTEXT_TOKENS
            ),
            "persona_id": self.persona.persona_id,
            "persona_path": str(self.persona.path),
            "elapsed_seconds": elapsed_seconds,
            "timings": {
                "elapsed_seconds": elapsed_seconds,
                "session_model_load_seconds": self.model_load_seconds,
                "session_audio_component_load_seconds": (
                    self.audio_component_load_seconds
                ),
                "session_decoder_load_seconds": self.decoder_load_seconds,
                "session_speech_warmup_seconds": self.speech_warmup_seconds,
                "asr_elapsed_seconds": None,
                "tts_first_audio_ready_seconds": speech.first_audio_ready_seconds,
                "tts_first_playback_started_seconds": (
                    speech.first_playback_started_seconds
                ),
                "text_to_tts_wall_seconds": round(time.time() - tts_started_at, 3),
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

    def _generate_response_and_stream_speech(self, transcript: str):
        text_queue: queue.Queue[str | None] = queue.Queue()
        speech_result: dict[str, Any] = {}
        speech_error: list[BaseException] = []
        spoken_prefix = ""

        def text_segments() -> Iterable[str]:
            while True:
                segment = text_queue.get()
                if segment is None:
                    return
                if segment.strip():
                    yield segment

        def speech_worker() -> None:
            try:
                speech_result["speech"] = (
                    self.generate_speech_output_streaming_from_text_segments(
                        text_segments=text_segments(),
                        max_tokens_per_segment=DEFAULT_S2S_TTS_MAX_TOKENS,
                        play=True,
                        first_chunk_label="Audex STS: first answer chunk ready",
                    )
                )
            except BaseException as exc:
                speech_error.append(exc)

        worker = threading.Thread(target=speech_worker, daemon=True)
        worker.start()

        def response_text_callback(text: str) -> None:
            nonlocal spoken_prefix
            if spoken_prefix:
                return
            prefix = _first_complete_sentence_batch(
                text,
                min_sentences=STREAMING_TTS_START_SENTENCES,
            )
            if not prefix:
                return
            spoken_prefix = prefix
            print("Audex STS: streaming speech output...", flush=True)
            text_queue.put(prefix)

        try:
            response_text = self.generate_text_response(
                transcript,
                response_text_callback=response_text_callback,
            )
            speakable_response = response_text or transcript or "I heard your message."
            remaining = _remaining_after_prefix(speakable_response, spoken_prefix)
            if remaining:
                if not spoken_prefix:
                    print("Audex STS: streaming speech output...", flush=True)
                text_queue.put(remaining)
        except TypeError as exc:
            if "response_text_callback" not in str(exc):
                raise
            response_text = self.generate_text_response(transcript)
            print("Audex STS: streaming speech output...", flush=True)
            text_queue.put(response_text or transcript or "I heard your message.")
        finally:
            text_queue.put(None)
            worker.join()

        if speech_error:
            raise speech_error[0]
        return response_text, speech_result["speech"]

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
        print(f"Audex STS: startup greeting: {greeting}", flush=True)
        if not play:
            return
        self.generate_speech_output_streaming(
            text=greeting,
            max_tokens_per_segment=self._speech_max_tokens_for_text(greeting),
            play=True,
            artifact_prefix="startup-greeting",
            first_chunk_label="Audex STS: first startup greeting chunk ready",
        )

    def transcribe_wav(
        self,
        wav_path: Path,
        *,
        max_tokens: int = DEFAULT_ASR_MAX_TOKENS,
    ) -> str:
        clips = prepare_audex_wav_clips(wav_path)
        features = extract_audex_input_features(
            clips,
            preprocessor_path=self.full_model_path / "audio_preprocessor",
        )
        feature_array = _features_to_mlx(features.input_features, self.mx)
        feature_array = feature_array.astype(self.encoder_weights["conv1.weight"].dtype)
        encoder_hidden = encode_audio_features_mlx(
            feature_array,
            self.encoder_weights,
            self.encoder_config,
        )
        audio_embeddings = project_audio_hidden_states_mlx(
            encoder_hidden,
            self.projector_weights,
            self.projector_config,
        )
        audio_embeddings = self.mx.reshape(
            audio_embeddings,
            (
                int(audio_embeddings.shape[0]) * int(audio_embeddings.shape[1]),
                int(audio_embeddings.shape[2]),
            ),
        )
        stream_printer = _TextStreamPrinter(
            "Audex ASR text",
            _clean_streaming_transcription,
        )

        prompt = self._build_audio_chat_prompt_from_template(
            DEFAULT_AUDIO_PROMPT,
            sample_count=clips.original_sample_count,
        )
        prompt_tokens = tuple(
            int(token_id) for token_id in self.tokenizer.encode(prompt)
        )
        validate_audio_splice_plan(
            prompt_tokens,
            tuple(int(part) for part in audio_embeddings.shape),
            sound_token_id=SOUND_TOKEN_ID,
        )
        token_ids = self.mx.array(prompt_tokens, dtype=self.mx.int32)
        input_embeddings = self.model.model.embed_tokens(token_ids[None])[0]
        spliced = splice_audio_embeddings_mlx(
            token_ids,
            input_embeddings,
            audio_embeddings,
            sound_token_id=SOUND_TOKEN_ID,
        )
        generated_ids = _generate_token_ids(
            mx=self.mx,
            generate_step=self.generate_step,
            make_sampler=self.make_sampler,
            model=self.model,
            prompt=token_ids,
            max_tokens=max_tokens,
            temperature=NVIDIA_ASR_TEMPERATURE,
            top_p=NVIDIA_ASR_TOP_P,
            input_embeddings=spliced,
            decode_text=self.tokenizer.decode,
            text_callback=stream_printer.update,
        )
        transcript = _clean_transcription(self.tokenizer.decode(generated_ids))
        stream_printer.finish(transcript)
        return transcript

    def generate_text_response(
        self,
        transcript: str,
        *,
        response_text_callback: Callable[[str], None] | None = None,
    ) -> str:
        user_message = {"role": "user", "content": transcript or "[inaudible]"}
        turn_messages = self.messages + [user_message]
        stream_printer = _TextStreamPrinter("Audex response text", _clean_response_text)
        prompt = self.tokenizer.apply_chat_template(
            turn_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking_enabled,
        )
        prompt_tokens = tuple(
            int(token_id) for token_id in self.tokenizer.encode(prompt)
        )
        prompt_cache = None
        generation_tokens = prompt_tokens
        cache_prefix_tokens = self._history_prompt_tokens()
        if (
            self.enable_kv_cache
            and _tokens_start_with(prompt_tokens, cache_prefix_tokens)
            and self._can_use_prompt_cache(cache_prefix_tokens)
        ):
            prompt_cache = self.text_prompt_cache
            generation_tokens = prompt_tokens[len(cache_prefix_tokens) :]
            self.text_prompt_cache = None
            print(
                "Audex STS: resumed conversation KV cache "
                f"({len(cache_prefix_tokens)} cached tokens, "
                f"{len(generation_tokens)} suffix tokens).",
                flush=True,
            )
        elif self.enable_kv_cache and self.conversation is not None:
            print("Audex STS: conversation KV cache miss; prefilling text.", flush=True)

        def on_text_update(text: str) -> None:
            stream_printer.update(text)
            if response_text_callback is not None:
                response_text_callback(_clean_response_text(text))

        generated_ids = _generate_token_ids(
            mx=self.mx,
            generate_step=self.generate_step,
            make_sampler=self.make_sampler,
            model=self.model,
            prompt=self.mx.array(generation_tokens, dtype=self.mx.int32),
            max_tokens=self.response_max_tokens,
            temperature=NVIDIA_TEXT_TEMPERATURE,
            top_p=NVIDIA_TEXT_TOP_P,
            prompt_cache=prompt_cache,
            decode_text=self.tokenizer.decode,
            text_callback=on_text_update,
        )
        full_response = clean_generation(self.tokenizer.decode(generated_ids))
        response = _clean_response_text(full_response)
        stream_printer.finish(response)
        self.messages.append(user_message)
        self.messages.append({"role": "assistant", "content": response})
        self._persist_conversation()
        return response

    def generate_speech_output(
        self,
        *,
        text: str,
        max_tokens: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        speech = self.generate_speech_tokens(
            text=text,
            max_tokens=max_tokens,
            progress_callback=progress_callback,
        )
        token_frames = tuple((frame,) for frame in speech.generated_codec_frames)
        if not token_frames:
            raise RuntimeError(
                "Audex speech-token generation produced no codec frames."
            )

        waveform = decode_speech_token_frames_mlx(
            token_frames,
            self.decoder_weights,
            self.decoder_config,
        )
        finite = bool(self.mx.all(self.mx.isfinite(waveform)).item())
        peak_abs = float(self.mx.max(self.mx.abs(waveform)).item())
        samples = tuple(float(sample) for sample in waveform.tolist())

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        wav_path = self.output_dir / f"speech-output-{timestamp}.wav"
        run_log_path = self.output_dir / f"speech-output-{timestamp}.json"
        write_pcm16_wav(
            wav_path,
            samples,
            sample_rate=self.decoder_config.sample_rate,
        )
        run_log = {
            "backend": "mlx",
            "device": str(self.mx.default_device()),
            "full_model_path": str(self.full_model_path),
            "decoder_path": str(self.decoder_path),
            "prompt_tokens": speech.prompt_tokens,
            "tts_sampler": {
                "temperature": speech.temperature,
                "top_p": speech.top_p,
                "top_k": speech.top_k,
                "cfg_scale": speech.cfg_scale_reference,
                "cfg_applied": speech.cfg_applied,
            },
            "generated_token_ids": list(speech.generated_token_ids),
            "generated_token_text": list(speech.generated_token_text),
            "generated_codec_frames": list(speech.generated_codec_frames),
            "reached_end_token": speech.reached_end_token,
            "hit_max_tokens": speech.hit_max_tokens,
            "waveform_shape": list(int(part) for part in waveform.shape),
            "sample_rate": self.decoder_config.sample_rate,
            "hop_length": self.decoder_config.hop_length,
            "finite": finite,
            "peak_abs": peak_abs,
            "wav_path": str(wav_path),
        }
        run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")

        from .speech_output import SpeechOutputSmokeResult

        return SpeechOutputSmokeResult(
            backend="mlx",
            device=str(self.mx.default_device()),
            prompt_tokens=speech.prompt_tokens,
            generated_token_ids=speech.generated_token_ids,
            generated_codec_frames=speech.generated_codec_frames,
            reached_end_token=speech.reached_end_token,
            hit_max_tokens=speech.hit_max_tokens,
            waveform_shape=tuple(int(part) for part in waveform.shape),
            sample_rate=self.decoder_config.sample_rate,
            hop_length=self.decoder_config.hop_length,
            finite=finite,
            peak_abs=peak_abs,
            wav_path=wav_path,
            run_log_path=run_log_path,
        )

    def _warm_speech_output_path(self) -> None:
        pending_frames: list[int] = []

        def on_token(token_id: int) -> None:
            if token_id in self.token_map.speech_codec:
                pending_frames.append(self.token_map.speech_codec[token_id])

        speech = self.generate_speech_tokens(
            text=TTS_WARMUP_TEXT,
            max_tokens=DEFAULT_STREAM_DECODER_CHUNK_FRAMES,
            token_callback=on_token,
        )
        frames = pending_frames or list(speech.generated_codec_frames) or [0]
        decode_speech_token_frames_mlx(
            tuple((frame,) for frame in frames[:DEFAULT_STREAM_DECODER_CHUNK_FRAMES]),
            self.decoder_weights,
            self.decoder_config,
        )

    def generate_speech_output_streaming(
        self,
        *,
        text: str,
        max_tokens_per_segment: int,
        play: bool,
        decoder_chunk_frames: int = DEFAULT_STREAM_DECODER_CHUNK_FRAMES,
        artifact_prefix: str = "speech-output",
        first_chunk_label: str = "Audex STS: first speech chunk ready",
    ):
        from .speech_output import SpeechOutputSmokeResult

        if max_tokens_per_segment <= 0:
            raise ValueError(
                f"max_tokens_per_segment must be positive, got {max_tokens_per_segment}"
            )
        decoder_chunk_frames = max(1, int(decoder_chunk_frames))

        started_at = time.time()
        segments = _split_tts_segments(text)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        wav_path = self.output_dir / f"{artifact_prefix}-{timestamp}.wav"
        run_log_path = self.output_dir / f"{artifact_prefix}-{timestamp}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        player = (
            _ContinuousPcmPlayer(
                started_at=started_at,
                sample_rate=self.decoder_config.sample_rate,
            )
            if play
            else None
        )
        if player is not None:
            player.start()
        decoder_session = AudexSpeechDecoderSession(
            weights=self.decoder_weights,
            config=self.decoder_config,
            chunk_frames=decoder_chunk_frames,
        )

        all_token_ids: list[int] = []
        all_token_text: list[str] = []
        all_codec_frames: list[int] = []
        all_samples: list[float] = []
        chunk_paths: list[Path] = []
        prompt_tokens_total = 0
        reached_end_token = True
        hit_max_tokens = False
        peak_abs = 0.0
        first_audio_ready_seconds: float | None = None

        playback_diagnostics: dict[str, object] | None = None
        try:
            for segment_index, segment in enumerate(segments, start=1):
                pending_frames: list[int] = []

                def emit_waveform(waveform: Any) -> None:
                    nonlocal first_audio_ready_seconds, peak_abs
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
                        first_audio_ready_seconds = round(time.time() - started_at, 3)
                        print(
                            f"{first_chunk_label} ({first_audio_ready_seconds:.3f}s).",
                            flush=True,
                        )
                    if player is not None:
                        player.enqueue_samples(samples)

                def push_frames(pending_frames: list[int]) -> None:
                    if not pending_frames:
                        return
                    frames = tuple((frame,) for frame in pending_frames)
                    pending_frames.clear()
                    for sample_rate, waveform in decoder_session.push(frames):
                        if sample_rate != self.decoder_config.sample_rate:
                            raise ValueError(
                                "Audex decoder emitted unexpected sample rate "
                                f"{sample_rate}; expected {self.decoder_config.sample_rate}"
                            )
                        emit_waveform(waveform)

                def on_token(
                    token_id: int,
                    pending_frames: list[int] = pending_frames,
                ) -> None:
                    if token_id in self.token_map.speech_codec:
                        frame = self.token_map.speech_codec[token_id]
                        pending_frames.append(frame)
                        if len(pending_frames) >= decoder_chunk_frames:
                            push_frames(pending_frames)

                speech = self.generate_speech_tokens(
                    text=segment,
                    max_tokens=max_tokens_per_segment,
                    token_callback=on_token,
                )
                prompt_tokens_total += speech.prompt_tokens
                all_token_ids.extend(speech.generated_token_ids)
                all_token_text.extend(speech.generated_token_text)
                all_codec_frames.extend(speech.generated_codec_frames)
                reached_end_token = reached_end_token and speech.reached_end_token
                hit_max_tokens = hit_max_tokens or speech.hit_max_tokens
                push_frames(pending_frames)
                for sample_rate, waveform in decoder_session.flush():
                    if sample_rate != self.decoder_config.sample_rate:
                        raise ValueError(
                            "Audex decoder emitted unexpected sample rate "
                            f"{sample_rate}; expected {self.decoder_config.sample_rate}"
                        )
                    emit_waveform(waveform)
                decoder_session.reset()

                if segment_index < len(segments):
                    silence_samples = [0.0] * int(
                        round(
                            self.decoder_config.sample_rate
                            * TTS_SEGMENT_SILENCE_SECONDS
                        )
                    )
                    all_samples.extend(silence_samples)
                    silence_path = (
                        self.output_dir
                        / f"{artifact_prefix}-{timestamp}-chunk-{len(chunk_paths) + 1:04d}.wav"
                    )
                    write_pcm16_wav(
                        silence_path,
                        silence_samples,
                        sample_rate=self.decoder_config.sample_rate,
                    )
                    chunk_paths.append(silence_path)
                    if player is not None:
                        player.enqueue_samples(silence_samples)
        finally:
            if player is not None:
                player.close()
                playback_diagnostics = player.diagnostics()

        if not all_codec_frames:
            raise RuntimeError(
                "Audex speech-token generation produced no codec frames."
            )

        write_pcm16_wav(
            wav_path,
            all_samples,
            sample_rate=self.decoder_config.sample_rate,
        )
        finite = bool(all(math.isfinite(sample) for sample in all_samples))
        run_log = {
            "backend": "mlx",
            "device": str(self.mx.default_device()),
            "full_model_path": str(self.full_model_path),
            "decoder_path": str(self.decoder_path),
            "streaming": True,
            "artifact_prefix": artifact_prefix,
            "segments": segments,
            "decoder_chunk_frames": decoder_chunk_frames,
            "chunk_wav_paths": [str(path) for path in chunk_paths],
            "playback_transport": (
                "sounddevice_raw_output_stream" if player is not None else None
            ),
            "playback_prebuffer_seconds": (
                DEFAULT_PLAYBACK_PREBUFFER_SECONDS if player is not None else None
            ),
            "prompt_tokens": prompt_tokens_total,
            "tts_sampler": {
                "temperature": NVIDIA_TTS_TEMPERATURE,
                "top_p": NVIDIA_TTS_TOP_P,
                "top_k": NVIDIA_TTS_TOP_K,
                "cfg_scale": NVIDIA_TTS_CFG_SCALE,
                "cfg_applied": True,
            },
            "generated_token_ids": all_token_ids,
            "generated_token_text": all_token_text,
            "generated_codec_frames": all_codec_frames,
            "reached_end_token": reached_end_token,
            "hit_max_tokens": hit_max_tokens,
            "waveform_shape": [len(all_samples)],
            "sample_rate": self.decoder_config.sample_rate,
            "hop_length": self.decoder_config.hop_length,
            "finite": finite,
            "peak_abs": peak_abs,
            "wav_path": str(wav_path),
            "first_audio_ready_seconds": first_audio_ready_seconds,
            "first_playback_started_seconds": (
                player.first_playback_started_seconds if player is not None else None
            ),
            "playback_diagnostics": playback_diagnostics,
        }
        run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")

        return SpeechOutputSmokeResult(
            backend="mlx",
            device=str(self.mx.default_device()),
            prompt_tokens=prompt_tokens_total,
            generated_token_ids=tuple(all_token_ids),
            generated_codec_frames=tuple(all_codec_frames),
            reached_end_token=reached_end_token,
            hit_max_tokens=hit_max_tokens,
            waveform_shape=(len(all_samples),),
            sample_rate=self.decoder_config.sample_rate,
            hop_length=self.decoder_config.hop_length,
            finite=finite,
            peak_abs=peak_abs,
            wav_path=wav_path,
            run_log_path=run_log_path,
            streaming=True,
            segments=tuple(segments),
            chunk_wav_paths=tuple(chunk_paths),
            first_audio_ready_seconds=first_audio_ready_seconds,
            first_playback_started_seconds=(
                player.first_playback_started_seconds if player is not None else None
            ),
            playback_diagnostics=playback_diagnostics,
        )

    def generate_speech_output_streaming_from_text_segments(
        self,
        *,
        text_segments: Iterable[str],
        max_tokens_per_segment: int,
        play: bool,
        decoder_chunk_frames: int = DEFAULT_STREAM_DECODER_CHUNK_FRAMES,
        artifact_prefix: str = "speech-output",
        first_chunk_label: str = "Audex STS: first speech chunk ready",
    ):
        from .speech_output import SpeechOutputSmokeResult

        if max_tokens_per_segment <= 0:
            raise ValueError(
                f"max_tokens_per_segment must be positive, got {max_tokens_per_segment}"
            )
        decoder_chunk_frames = max(1, int(decoder_chunk_frames))

        started_at = time.time()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        wav_path = self.output_dir / f"{artifact_prefix}-{timestamp}.wav"
        run_log_path = self.output_dir / f"{artifact_prefix}-{timestamp}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        player = (
            _ContinuousPcmPlayer(
                started_at=started_at,
                sample_rate=self.decoder_config.sample_rate,
            )
            if play
            else None
        )
        if player is not None:
            player.start()
        decoder_session = AudexSpeechDecoderSession(
            weights=self.decoder_weights,
            config=self.decoder_config,
            chunk_frames=decoder_chunk_frames,
        )

        segments: list[str] = []
        all_token_ids: list[int] = []
        all_token_text: list[str] = []
        all_codec_frames: list[int] = []
        all_samples: list[float] = []
        chunk_paths: list[Path] = []
        prompt_tokens_total = 0
        reached_end_token = True
        hit_max_tokens = False
        peak_abs = 0.0
        first_audio_ready_seconds: float | None = None

        def emit_waveform(waveform: Any) -> None:
            nonlocal first_audio_ready_seconds, peak_abs
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
                first_audio_ready_seconds = round(time.time() - started_at, 3)
                print(
                    f"{first_chunk_label} ({first_audio_ready_seconds:.3f}s).",
                    flush=True,
                )
            if player is not None:
                player.enqueue_samples(samples)

        def emit_silence() -> None:
            silence_samples = [0.0] * int(
                round(self.decoder_config.sample_rate * TTS_SEGMENT_SILENCE_SECONDS)
            )
            all_samples.extend(silence_samples)
            silence_path = (
                self.output_dir
                / f"{artifact_prefix}-{timestamp}-chunk-{len(chunk_paths) + 1:04d}.wav"
            )
            write_pcm16_wav(
                silence_path,
                silence_samples,
                sample_rate=self.decoder_config.sample_rate,
            )
            chunk_paths.append(silence_path)
            if player is not None:
                player.enqueue_samples(silence_samples)

        def push_frames(pending_frames: list[int]) -> None:
            if not pending_frames:
                return
            frames = tuple((frame,) for frame in pending_frames)
            pending_frames.clear()
            for sample_rate, waveform in decoder_session.push(frames):
                if sample_rate != self.decoder_config.sample_rate:
                    raise ValueError(
                        "Audex decoder emitted unexpected sample rate "
                        f"{sample_rate}; expected {self.decoder_config.sample_rate}"
                    )
                emit_waveform(waveform)

        playback_diagnostics: dict[str, object] | None = None
        try:
            for text_batch in text_segments:
                for segment in _split_tts_segments(text_batch):
                    if segments:
                        emit_silence()
                    segments.append(segment)
                    pending_frames: list[int] = []

                    def on_token(
                        token_id: int,
                        pending_frames: list[int] = pending_frames,
                    ) -> None:
                        if token_id in self.token_map.speech_codec:
                            frame = self.token_map.speech_codec[token_id]
                            pending_frames.append(frame)
                            if len(pending_frames) >= decoder_chunk_frames:
                                push_frames(pending_frames)

                    speech = self.generate_speech_tokens(
                        text=segment,
                        max_tokens=max_tokens_per_segment,
                        token_callback=on_token,
                    )
                    prompt_tokens_total += speech.prompt_tokens
                    all_token_ids.extend(speech.generated_token_ids)
                    all_token_text.extend(speech.generated_token_text)
                    all_codec_frames.extend(speech.generated_codec_frames)
                    reached_end_token = reached_end_token and speech.reached_end_token
                    hit_max_tokens = hit_max_tokens or speech.hit_max_tokens
                    push_frames(pending_frames)
                    for sample_rate, waveform in decoder_session.flush():
                        if sample_rate != self.decoder_config.sample_rate:
                            raise ValueError(
                                "Audex decoder emitted unexpected sample rate "
                                f"{sample_rate}; expected {self.decoder_config.sample_rate}"
                            )
                        emit_waveform(waveform)
                    decoder_session.reset()
        finally:
            if player is not None:
                player.close()
                playback_diagnostics = player.diagnostics()

        if not all_codec_frames:
            raise RuntimeError(
                "Audex speech-token generation produced no codec frames."
            )

        write_pcm16_wav(
            wav_path,
            all_samples,
            sample_rate=self.decoder_config.sample_rate,
        )
        finite = bool(all(math.isfinite(sample) for sample in all_samples))
        run_log = {
            "backend": "mlx",
            "device": str(self.mx.default_device()),
            "full_model_path": str(self.full_model_path),
            "decoder_path": str(self.decoder_path),
            "streaming": True,
            "artifact_prefix": artifact_prefix,
            "segments": segments,
            "decoder_chunk_frames": decoder_chunk_frames,
            "chunk_wav_paths": [str(path) for path in chunk_paths],
            "playback_transport": (
                "sounddevice_raw_output_stream" if player is not None else None
            ),
            "playback_prebuffer_seconds": (
                DEFAULT_PLAYBACK_PREBUFFER_SECONDS if player is not None else None
            ),
            "prompt_tokens": prompt_tokens_total,
            "tts_sampler": {
                "temperature": NVIDIA_TTS_TEMPERATURE,
                "top_p": NVIDIA_TTS_TOP_P,
                "top_k": NVIDIA_TTS_TOP_K,
                "cfg_scale": NVIDIA_TTS_CFG_SCALE,
                "cfg_applied": True,
            },
            "generated_token_ids": all_token_ids,
            "generated_token_text": all_token_text,
            "generated_codec_frames": all_codec_frames,
            "reached_end_token": reached_end_token,
            "hit_max_tokens": hit_max_tokens,
            "waveform_shape": [len(all_samples)],
            "sample_rate": self.decoder_config.sample_rate,
            "hop_length": self.decoder_config.hop_length,
            "finite": finite,
            "peak_abs": peak_abs,
            "wav_path": str(wav_path),
            "first_audio_ready_seconds": first_audio_ready_seconds,
            "first_playback_started_seconds": (
                player.first_playback_started_seconds if player is not None else None
            ),
            "playback_diagnostics": playback_diagnostics,
        }
        run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")

        return SpeechOutputSmokeResult(
            backend="mlx",
            device=str(self.mx.default_device()),
            prompt_tokens=prompt_tokens_total,
            generated_token_ids=tuple(all_token_ids),
            generated_codec_frames=tuple(all_codec_frames),
            reached_end_token=reached_end_token,
            hit_max_tokens=hit_max_tokens,
            waveform_shape=(len(all_samples),),
            sample_rate=self.decoder_config.sample_rate,
            hop_length=self.decoder_config.hop_length,
            finite=finite,
            peak_abs=peak_abs,
            wav_path=wav_path,
            run_log_path=run_log_path,
            streaming=True,
            segments=tuple(segments),
            chunk_wav_paths=tuple(chunk_paths),
            first_audio_ready_seconds=first_audio_ready_seconds,
            first_playback_started_seconds=(
                player.first_playback_started_seconds if player is not None else None
            ),
            playback_diagnostics=playback_diagnostics,
        )

    def generate_speech_tokens(
        self,
        *,
        text: str,
        max_tokens: int,
        progress_callback: Callable[[int, int], None] | None = None,
        token_callback: Callable[[int], None] | None = None,
    ) -> SpeechTokenGenerationSmokeResult:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        prompt = build_tts_prompt(text, self.tokenizer)
        null_prompt = build_tts_null_prompt(prompt, self.tokenizer)
        prompt_tokens, null_prompt_tokens = tokenize_tts_cfg_pair(
            prompt,
            null_prompt,
            self.tokenizer,
        )
        vocab_size = int(self.model.args.vocab_size)
        prompt_max_token_id = max(prompt_tokens)
        if prompt_max_token_id >= vocab_size:
            raise ValueError(
                "Audex TTS prompt token exceeds full model vocabulary: "
                f"token={prompt_max_token_id} vocab_size={vocab_size}"
            )
        if self.token_map.speechgen_end >= vocab_size:
            raise ValueError(
                "Audex speech tokenizer exceeds full model vocabulary: "
                f"speechgen_end={self.token_map.speechgen_end} vocab_size={vocab_size}"
            )

        sampler = self.make_sampler(
            temp=NVIDIA_TTS_TEMPERATURE,
            top_p=NVIDIA_TTS_TOP_P,
            top_k=NVIDIA_TTS_TOP_K,
        )
        generated_token_ids, logprobs_shape = _generate_tts_cfg_token_ids(
            mx=self.mx,
            cache_module=self.cache_module,
            model=self.model,
            max_tokens=max_tokens,
            sampler=sampler,
            cond_prompt_tokens=prompt_tokens,
            uncond_prompt_tokens=null_prompt_tokens,
            cfg_scale=NVIDIA_TTS_CFG_SCALE,
            codec_min_id=min(self.token_map.speech_codec),
            codec_max_id=max(self.token_map.speech_codec),
            speechgen_end_id=self.token_map.speechgen_end,
            vocab_size=vocab_size,
            progress_callback=progress_callback,
            token_callback=token_callback,
        )
        generated_token_text: list[str] = []
        generated_codec_frames: list[int] = []
        for token_id in generated_token_ids:
            generated_token_text.append(
                str(self.tokenizer.convert_ids_to_tokens(token_id))
            )
            if token_id in self.token_map.speech_codec:
                generated_codec_frames.append(self.token_map.speech_codec[token_id])
        reached_end_token = bool(
            generated_token_ids
            and generated_token_ids[-1] == self.token_map.speechgen_end
        )

        return SpeechTokenGenerationSmokeResult(
            backend="mlx_lm",
            device=str(self.mx.default_device()),
            model_type=str(self.model.args.model_type),
            vocab_size=vocab_size,
            prompt_tokens=len(prompt_tokens),
            prompt_max_token_id=prompt_max_token_id,
            speechgen_start_id=self.token_map.speechgen_start,
            speechgen_end_id=self.token_map.speechgen_end,
            codec_token_count=len(self.token_map.speech_codec),
            generated_token_ids=tuple(generated_token_ids),
            generated_token_text=tuple(generated_token_text),
            generated_codec_frames=tuple(generated_codec_frames),
            logprobs_shape=logprobs_shape,
            reached_end_token=reached_end_token,
            hit_max_tokens=len(generated_token_ids) >= max_tokens
            and not reached_end_token,
            temperature=NVIDIA_TTS_TEMPERATURE,
            top_p=NVIDIA_TTS_TOP_P,
            top_k=NVIDIA_TTS_TOP_K,
            cfg_scale_reference=NVIDIA_TTS_CFG_SCALE,
            cfg_applied=True,
        )

    def _build_audio_chat_prompt_from_template(
        self,
        prompt: str,
        *,
        sample_count: int,
    ) -> str:
        prompt_plan = build_audio_prompt_plan(prompt, sample_count=sample_count)
        messages = [
            {"role": "system", "content": self.messages[0]["content"]},
            {"role": "user", "content": f"{prompt}\n<sound>"},
        ]
        template_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return _expand_sound_placeholder_for_prompt(
            template_prompt,
            prompt_plan.num_embeddings,
        )

    def _speech_max_tokens_for_text(self, text: str) -> int:
        if self.speech_max_tokens is not None:
            return self.speech_max_tokens
        text_tokens = max(1, len(self.tokenizer.encode(text)))
        return max(
            DEFAULT_S2S_TTS_MAX_TOKENS,
            text_tokens * DEFAULT_SPEECH_TOKENS_PER_TEXT_TOKEN,
        )

    def _history_prompt_tokens(self) -> tuple[int, ...]:
        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=self.thinking_enabled,
        )
        return tuple(int(token_id) for token_id in self.tokenizer.encode(prompt))

    def _count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=self.thinking_enabled,
        )
        return len(self.tokenizer.encode(prompt))

    def _persist_conversation(self) -> None:
        if self.conversation is None or self.conversation_store is None:
            return
        token_count = self._count_messages_tokens(self.messages)
        self.conversation.messages = list(self.messages)
        self.conversation.token_count = token_count
        self.conversation_store.save(self.conversation)
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
        if self.enable_kv_cache:
            self._save_conversation_prompt_cache()

    def _conversation_kv_cache_path(self) -> Path | None:
        if self.conversation is None:
            return None
        return (
            self.conversation.root
            / f"{self.conversation.conversation_id}.kv.safetensors"
        )

    def _load_conversation_prompt_cache(self) -> None:
        cache_path = self._conversation_kv_cache_path()
        if not self.enable_kv_cache or cache_path is None or not cache_path.is_file():
            return
        try:
            prompt_cache, metadata = self.cache_module.load_prompt_cache(
                str(cache_path),
                return_metadata=True,
            )
        except Exception as exc:
            print(f"Audex STS: failed to load KV cache: {exc}", flush=True)
            return
        expected_tokens = self._history_prompt_tokens()
        expected_hash = _token_hash(expected_tokens)
        if (
            metadata.get("conversation_id")
            != (self.conversation.conversation_id if self.conversation else None)
            or metadata.get("prompt_token_count") != str(len(expected_tokens))
            or metadata.get("prompt_token_hash") != expected_hash
            or metadata.get("selected_model") != str(self.selected_model_repo)
        ):
            print("Audex STS: KV cache metadata mismatch; ignoring cache.", flush=True)
            return
        self.text_prompt_cache = prompt_cache
        self.loaded_conversation_kv_cache = True
        self.text_prompt_cache_token_count = len(expected_tokens)
        self.text_prompt_cache_token_hash = expected_hash
        print(
            "Audex STS: loaded conversation KV cache "
            f"({len(expected_tokens)} tokens).",
            flush=True,
        )

    def _can_use_prompt_cache(self, prefix_tokens: tuple[int, ...]) -> bool:
        if self.text_prompt_cache is None:
            return False
        return self.text_prompt_cache_token_count == len(
            prefix_tokens
        ) and self.text_prompt_cache_token_hash == _token_hash(prefix_tokens)

    def _save_conversation_prompt_cache(self) -> None:
        cache_path = self._conversation_kv_cache_path()
        if cache_path is None:
            return
        tokens = self._history_prompt_tokens()
        if not tokens:
            return
        prompt_cache = self.cache_module.make_prompt_cache(self.model)
        token_array = self.mx.array(tokens, dtype=self.mx.int32)
        self.model(token_array[None], cache=prompt_cache)
        self.mx.eval([cache.state for cache in prompt_cache])
        metadata = {
            "format": "audex-mac-conversation-kv-v1",
            "conversation_id": (
                self.conversation.conversation_id if self.conversation else ""
            ),
            "selected_model": str(self.selected_model_repo),
            "persona_id": self.persona.persona_id,
            "prompt_token_count": str(len(tokens)),
            "prompt_token_hash": _token_hash(tokens),
            "thinking_enabled": str(self.thinking_enabled).lower(),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_module.save_prompt_cache(str(cache_path), prompt_cache, metadata)
        self.text_prompt_cache = prompt_cache
        self.text_prompt_cache_token_count = len(tokens)
        self.text_prompt_cache_token_hash = metadata["prompt_token_hash"]
        print(f"Audex STS: saved conversation KV cache: {cache_path}", flush=True)


def run_fixture_turn(
    *,
    full_model_path: Path,
    decoder_path: Path,
    input_wav_path: Path,
    selected_model_repo: str | None = None,
    output_dir: Path = RUNS_DIR,
    play: bool = True,
    response_max_tokens: int = DEFAULT_RESPONSE_MAX_TOKENS,
    speech_max_tokens: int | None = None,
    thinking_enabled: bool = False,
    session: AudexSpeechToSpeechSession | None = None,
) -> SpeechToSpeechTurnResult:
    """Run one Audex speech-to-speech turn from a local WAV fixture."""

    if session is not None:
        return session.run_turn_from_wav(input_wav_path=input_wav_path, play=play)

    started_at = time.time()
    print("Audex STS: transcribing input speech...", flush=True)
    transcript = transcribe_wav_with_audex(
        full_model_path=full_model_path,
        wav_path=input_wav_path,
    )
    print(f"Audex STS: transcript: {transcript}", flush=True)
    print("Audex STS: generating text response...", flush=True)
    response_text = generate_audex_text_response(
        full_model_path=full_model_path,
        transcript=transcript,
        max_tokens=response_max_tokens,
        thinking_enabled=thinking_enabled,
    )
    print(f"Audex STS: response text: {response_text}", flush=True)
    print("Audex STS: generating speech output...", flush=True)
    speech = run_speech_output_smoke(
        full_model_path=full_model_path,
        decoder_path=decoder_path,
        output_dir=output_dir,
        text=response_text or transcript or "I heard your message.",
        max_tokens=(
            speech_max_tokens
            if speech_max_tokens is not None
            else DEFAULT_S2S_TTS_MAX_TOKENS
        ),
        progress_callback=_print_speech_progress,
    )
    print(f"Audex STS: speech output ready: {speech.wav_path}", flush=True)
    played = False
    if play:
        print(f"Audex STS: playing speech output: {speech.wav_path}", flush=True)
        play_wav(speech.wav_path)
        played = True

    run_log_path = output_dir / f"sts-turn-{time.strftime('%Y%m%d-%H%M%S')}.json"
    elapsed_seconds = round(time.time() - started_at, 3)
    run_log = {
        "selected_model": selected_model_repo,
        "input_wav_path": str(input_wav_path),
        "output_wav_path": str(speech.wav_path),
        "speech_output_run_log_path": str(speech.run_log_path),
        "transcript": transcript,
        "response_text": response_text,
        "played": played,
        "response_max_tokens": response_max_tokens,
        "speech_max_tokens": (
            speech_max_tokens
            if speech_max_tokens is not None
            else DEFAULT_S2S_TTS_MAX_TOKENS
        ),
        "thinking_enabled": thinking_enabled,
        "elapsed_seconds": elapsed_seconds,
        "timings": {"elapsed_seconds": elapsed_seconds},
    }
    run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
    return SpeechToSpeechTurnResult(
        transcript=transcript,
        response_text=response_text,
        input_wav_path=input_wav_path,
        output_wav_path=speech.wav_path,
        run_log_path=run_log_path,
        played=played,
    )


def _print_speech_progress(generated_tokens: int, max_tokens: int) -> None:
    if generated_tokens not in {1, max_tokens} and generated_tokens % 256 != 0:
        return
    print(
        f"Audex TTS: generated {generated_tokens}/{max_tokens} speech tokens...",
        flush=True,
    )


def run_interactive_ptt(
    *,
    full_model_path: Path,
    decoder_path: Path,
    selected_model_repo: str | None = None,
    output_dir: Path = RUNS_DIR,
    play: bool = True,
    response_max_tokens: int = DEFAULT_RESPONSE_MAX_TOKENS,
    speech_max_tokens: int | None = None,
    thinking_enabled: bool = False,
    conversation: Conversation | None = None,
    conversation_store: ConversationStore | None = None,
    persona: Persona | None = None,
    enable_kv_cache: bool = True,
    conversation_resumed: bool = False,
) -> SpeechToSpeechTurnResult | None:
    """Run typed or push-to-talk turns in one persistent Audex session."""

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Audex STS: loading persistent MLX session...", flush=True)
    session = AudexSpeechToSpeechSession(
        full_model_path=full_model_path,
        decoder_path=decoder_path,
        selected_model_repo=selected_model_repo,
        output_dir=output_dir,
        thinking_enabled=thinking_enabled,
        response_max_tokens=response_max_tokens,
        speech_max_tokens=speech_max_tokens,
        conversation=conversation,
        conversation_store=conversation_store,
        persona=persona,
        enable_kv_cache=enable_kv_cache,
    )
    stats = session.stats
    print(
        "Audex STS: session ready "
        f"(model={stats.model_load_seconds:.3f}s, "
        f"audio={stats.audio_component_load_seconds:.3f}s, "
        f"decoder={stats.decoder_load_seconds:.3f}s, "
        f"speech_warmup={stats.speech_warmup_seconds:.3f}s).",
        flush=True,
    )
    session.speak_startup_greeting(
        conversation_resumed=conversation_resumed,
        play=play,
    )

    last_turn: SpeechToSpeechTurnResult | None = None
    while True:
        turn_input = read_turn_input()
        if turn_input.kind is InputKind.QUIT:
            break
        if turn_input.kind is InputKind.TEXT:
            last_turn = session.run_turn_from_text(
                user_text=turn_input.text,
                play=play,
            )
            print(f"Response: {last_turn.response_text}")
            print(f"Speech output WAV: {last_turn.output_wav_path}")
            print(f"Typed-turn run log: {last_turn.run_log_path}")
            continue

        input_wav_path = output_dir / f"ptt-input-{time.strftime('%Y%m%d-%H%M%S')}.wav"
        print("Recording. Press Enter to stop.")
        recording = _start_recording()
        with suppress(EOFError):
            input()
        samples = recording.stop()
        write_pcm16_wav(input_wav_path, samples, sample_rate=SAMPLE_RATE)
        print(f"Captured input WAV: {input_wav_path}")
        last_turn = session.run_turn_from_wav(input_wav_path=input_wav_path, play=play)
        print(f"Transcript: {last_turn.transcript}")
        print(f"Response: {last_turn.response_text}")
        print(f"Speech output WAV: {last_turn.output_wav_path}")
        print(f"Speech-to-speech run log: {last_turn.run_log_path}")

    if last_turn is None:
        print("Audex STS: no turns completed.")
    return last_turn


def transcribe_wav_with_audex(
    *,
    full_model_path: Path,
    wav_path: Path,
    max_tokens: int = DEFAULT_ASR_MAX_TOKENS,
) -> str:
    """Use Audex audio input to transcribe a 16 kHz WAV fixture."""

    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError(
            "Audex ASR path requires mlx, mlx_lm, and transformers."
        ) from exc

    mx.set_default_device(mx.gpu)
    apply_audex_runtime_patches()
    clips = prepare_audex_wav_clips(wav_path)
    features = extract_audex_input_features(
        clips,
        preprocessor_path=full_model_path / "audio_preprocessor",
    )
    feature_array = _features_to_mlx(features.input_features, mx)
    encoder_config = load_audio_encoder_config(full_model_path)
    encoder_weights = load_audio_encoder_weights_mlx(full_model_path)
    feature_array = feature_array.astype(encoder_weights["conv1.weight"].dtype)
    encoder_hidden = encode_audio_features_mlx(
        feature_array,
        encoder_weights,
        encoder_config,
    )
    projector_config = load_audio_projector_config(full_model_path)
    projector_weights = load_audio_projector_weights_mlx(full_model_path)
    audio_embeddings = project_audio_hidden_states_mlx(
        encoder_hidden,
        projector_weights,
        projector_config,
    )
    audio_embeddings = mx.reshape(
        audio_embeddings,
        (
            int(audio_embeddings.shape[0]) * int(audio_embeddings.shape[1]),
            int(audio_embeddings.shape[2]),
        ),
    )

    model, tokenizer = load(
        str(full_model_path), tokenizer_config={"trust_remote_code": True}
    )
    template_path = full_model_path / "chat_template.jinja"
    if template_path.is_file():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    stream_printer = _TextStreamPrinter(
        "Audex ASR text",
        _clean_streaming_transcription,
    )

    prompt_plan = build_audio_prompt_plan(
        DEFAULT_AUDIO_PROMPT,
        sample_count=clips.original_sample_count,
    )
    prompt = build_audio_chat_prompt(prompt_plan, thinking_enabled=False)
    prompt_tokens = tuple(int(token_id) for token_id in tokenizer.encode(prompt))
    validate_audio_splice_plan(
        prompt_tokens,
        tuple(int(part) for part in audio_embeddings.shape),
        sound_token_id=SOUND_TOKEN_ID,
    )
    token_ids = mx.array(prompt_tokens, dtype=mx.int32)
    input_embeddings = model.model.embed_tokens(token_ids[None])[0]
    spliced = splice_audio_embeddings_mlx(
        token_ids,
        input_embeddings,
        audio_embeddings,
        sound_token_id=SOUND_TOKEN_ID,
    )
    generated_ids = _generate_token_ids(
        mx=mx,
        generate_step=generate_step,
        make_sampler=make_sampler,
        model=model,
        prompt=token_ids,
        max_tokens=max_tokens,
        temperature=NVIDIA_ASR_TEMPERATURE,
        top_p=NVIDIA_ASR_TOP_P,
        input_embeddings=spliced,
        decode_text=tokenizer.decode,
        text_callback=stream_printer.update,
    )
    transcript = _clean_transcription(tokenizer.decode(generated_ids))
    stream_printer.finish(transcript)
    return transcript


def generate_audex_text_response(
    *,
    full_model_path: Path,
    transcript: str,
    max_tokens: int = DEFAULT_RESPONSE_MAX_TOKENS,
    thinking_enabled: bool = False,
) -> str:
    """Use Audex text mode to produce the spoken response text."""

    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError("Audex text response path requires mlx_lm.") from exc

    mx.set_default_device(mx.gpu)
    apply_audex_runtime_patches()
    model, tokenizer = load(
        str(full_model_path), tokenizer_config={"trust_remote_code": True}
    )
    template_path = full_model_path / "chat_template.jinja"
    if template_path.is_file():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    stream_printer = _TextStreamPrinter("Audex response text", clean_generation)
    messages = [
        {
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": transcript or "[inaudible]",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking_enabled,
    )
    prompt_tokens = tuple(int(token_id) for token_id in tokenizer.encode(prompt))
    generated_ids = _generate_token_ids(
        mx=mx,
        generate_step=generate_step,
        make_sampler=make_sampler,
        model=model,
        prompt=mx.array(prompt_tokens, dtype=mx.int32),
        max_tokens=max_tokens,
        temperature=NVIDIA_TEXT_TEMPERATURE,
        top_p=NVIDIA_TEXT_TOP_P,
        decode_text=tokenizer.decode,
        text_callback=stream_printer.update,
    )
    response = _clean_response_text(tokenizer.decode(generated_ids))
    stream_printer.finish(response)
    return response


def play_wav(path: Path) -> None:
    subprocess.run(["afplay", str(path)], check=True)


def _generate_token_ids(
    *,
    mx: Any,
    generate_step: Any,
    make_sampler: Any,
    model: Any,
    prompt: Any,
    max_tokens: int,
    temperature: float,
    top_p: float,
    input_embeddings: Any | None = None,
    prompt_cache: Any | None = None,
    decode_text: Callable[[list[int]], str] | None = None,
    text_callback: Callable[[str], None] | None = None,
) -> list[int]:
    processor = _text_only_logits_processor(mx, int(model.args.vocab_size))
    generator = generate_step(
        prompt,
        model,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature, top_p=top_p),
        logits_processors=[processor],
        input_embeddings=input_embeddings,
        prompt_cache=prompt_cache,
    )
    generated: list[int] = []
    for token, _ in generator:
        token_id = int(token)
        generated.append(token_id)
        if decode_text is not None and text_callback is not None:
            text_callback(decode_text(generated))
        if token_id in (2, 11):
            break
    return generated


def startup_greeting_text(
    *,
    conversation: Conversation | None,
    conversation_resumed: bool,
) -> str:
    if not conversation_resumed:
        return FIRST_STARTUP_GREETING_TEXT
    user_name = conversation.user_name if conversation is not None else None
    if user_name:
        return (
            f"Hi, {user_name}! Nice to hear from you again. "
            "What do you want to talk about today?"
        )
    return RESUME_UNKNOWN_USER_GREETING_TEXT


def _token_hash(tokens: tuple[int, ...]) -> str:
    digest = sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _tokens_start_with(tokens: tuple[int, ...], prefix: tuple[int, ...]) -> bool:
    return len(tokens) >= len(prefix) and tokens[: len(prefix)] == prefix


class _TextStreamPrinter:
    def __init__(
        self,
        label: str,
        cleaner: Callable[[str], str],
        *,
        min_delta_chars: int = 12,
    ) -> None:
        self.label = label
        self.cleaner = cleaner
        self.min_delta_chars = min_delta_chars
        self.last_text = ""
        self.started = False

    def update(self, text: str) -> None:
        cleaned = self.cleaner(text)
        if not cleaned or cleaned == self.last_text:
            return
        if len(cleaned) - len(self.last_text) < self.min_delta_chars:
            return
        self._print_delta(cleaned, final=False)

    def finish(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned:
            self._print_delta(cleaned, final=True)
        elif self.started:
            print("", flush=True)

    def _print_delta(self, text: str, *, final: bool) -> None:
        if not self.started:
            print(f"{self.label}: ", end="", flush=True)
            self.started = True
        if text.startswith(self.last_text):
            print(text[len(self.last_text) :], end="", flush=True)
        elif text != self.last_text:
            print(f"\n{self.label}: {text}", end="", flush=True)
        self.last_text = text
        if final:
            print("", flush=True)


def _text_only_logits_processor(mx: Any, vocab_size: int):
    allowed = mx.arange(vocab_size) < min(TEXT_VOCAB, vocab_size)
    for token_id in (SOUND_TOKEN_ID, SOUND_START_TOKEN_ID, SOUND_END_TOKEN_ID):
        if token_id < vocab_size:
            allowed = mx.logical_and(allowed, mx.arange(vocab_size) != token_id)

    def processor(_tokens, logits):
        return mx.where(allowed[None, :], logits, mx.array(-float("inf"), logits.dtype))

    return processor


def _clean_transcription(text: str) -> str:
    cleaned = clean_generation(text)
    first_quote = cleaned.find("'")
    last_quote = cleaned.rfind("'")
    if first_quote != -1 and last_quote > first_quote:
        return cleaned[first_quote + 1 : last_quote].strip()
    for marker in STOP_MARKERS:
        cleaned = cleaned.replace(marker, "")
    return cleaned.strip()


def _clean_response_text(text: str) -> str:
    cleaned = clean_generation(text)
    if "</think>" in cleaned:
        spoken = cleaned.rsplit("</think>", 1)[-1].strip()
        if spoken:
            return spoken
    if "<think>" in cleaned:
        return cleaned.split("<think>", 1)[0].strip()
    return cleaned.strip()


def _clean_streaming_transcription(text: str) -> str:
    cleaned = clean_generation(text)
    first_quote = cleaned.find("'")
    if first_quote != -1:
        tail = cleaned[first_quote + 1 :]
        last_quote = tail.rfind("'")
        if last_quote != -1:
            tail = tail[:last_quote]
        return tail.strip()
    if _is_asr_wrapper_fragment(cleaned):
        return ""
    cleaned = _strip_asr_wrapper_prefixes(cleaned)
    for marker in STOP_MARKERS:
        cleaned = cleaned.replace(marker, "")
    cleaned = cleaned.strip(" :")
    if _is_asr_wrapper_fragment(cleaned):
        return ""
    return cleaned


def _strip_asr_wrapper_prefixes(text: str) -> str:
    cleaned = text
    patterns = (
        r"^\s*Language\s*:\s*[A-Za-z][A-Za-z -]*\.?\s*",
        r"^\s*The\s+content\s+of\s+the\s+input\s+audio(?:\s+is)?\.?\s*",
        r"^\s*The\s+spoken\s+content\s+of\s+the\s+audio(?:\s+is)?\.?\s*",
        r"^\s*The\s+transcription\s+is\.?\s*",
    )
    previous = None
    while cleaned != previous:
        previous = cleaned
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _is_asr_wrapper_fragment(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip(" .:").lower()
    if not normalized:
        return False
    wrapper_prefixes = (
        "language",
        "the content",
        "the content of",
        "the content of the",
        "the content of the input",
        "the content of the input audio",
        "the content of the input audio is",
        "the spoken content",
        "the spoken content of",
        "the spoken content of the",
        "the spoken content of the audio",
        "the spoken content of the audio is",
        "the transcription",
        "the transcription is",
    )
    return normalized in wrapper_prefixes


def _expand_sound_placeholder_for_prompt(prompt: str, num_embeddings: int) -> str:
    return expand_sound_placeholder(prompt, num_embeddings)


def _split_tts_segments(text: str, *, max_chars: int = 360) -> tuple[str, ...]:
    spoken = _clean_response_text(text)
    if not spoken:
        return ("I heard your message.",)

    segments: list[str] = []
    for line in spoken.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.findall(r"[^.!?]+[.!?]+[\"')\]]*|[^.!?]+$", line)
        for part in parts:
            part = re.sub(r"[*_`#]+", "", part).strip()
            if not part:
                continue
            if not re.search(r"[A-Za-z0-9]", part):
                continue
            while len(part) > max_chars:
                split_at = part.rfind(" ", 0, max_chars)
                if split_at <= max_chars // 2:
                    split_at = max_chars
                segments.append(part[:split_at].strip())
                part = part[split_at:].strip()
            if part:
                segments.append(part)
    if (
        len(segments) > 1
        and not re.search(r"[.!?][\"')\]]*$", segments[-1])
        and len(segments[-1]) < 16
    ):
        segments.pop()
    return tuple(segments or (spoken.strip(),))


def _first_complete_sentence_batch(text: str, *, min_sentences: int) -> str:
    spoken = _clean_response_text(text)
    if not spoken:
        return ""
    sentences = [
        re.sub(r"[*_`#]+", "", part).strip()
        for part in re.findall(r"[^.!?]+[.!?]+[\"')\]]*", spoken)
    ]
    sentences = [
        sentence for sentence in sentences if re.search(r"[A-Za-z0-9]", sentence)
    ]
    if len(sentences) < min_sentences:
        return ""
    return " ".join(sentences[:min_sentences]).strip()


def _remaining_after_prefix(text: str, prefix: str) -> str:
    cleaned = _clean_response_text(text)
    prefix = prefix.strip()
    if not cleaned:
        return ""
    if prefix and cleaned.startswith(prefix):
        return cleaned[len(prefix) :].strip()
    return cleaned


class _PlaybackStartGate:
    """Thread-safe release/cancel gate for pre-generated semantic PCM."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._released = False
        self._cancelled = False
        self._released_at: float | None = None

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def released_at(self) -> float | None:
        with self._lock:
            return self._released_at

    def release(self, *, released_at: float | None = None) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._released = True
            self._released_at = released_at if released_at is not None else time.time()
            self._event.set()

    def cancel(self) -> None:
        with self._lock:
            if self._released:
                return
            self._cancelled = True
            self._event.set()

    def wait(self) -> bool:
        self._event.wait()
        with self._lock:
            return self._released and not self._cancelled


class _ContinuousPcmPlayer:
    def __init__(
        self,
        *,
        started_at: float,
        sample_rate: int,
        output_sample_rate: int | None = None,
        output_blocksize: int = 0,
        prebuffer_seconds: float = DEFAULT_PLAYBACK_PREBUFFER_SECONDS,
        latency: str = DEFAULT_PLAYBACK_LATENCY,
        adaptive_prebuffer: bool = True,
        start_gate: _PlaybackStartGate | None = None,
        first_playback_callback: Callable[[float, float | None], None] | None = None,
    ) -> None:
        self.started_at = started_at
        self.input_sample_rate = int(sample_rate)
        self.sample_rate = int(output_sample_rate or sample_rate)
        if self.sample_rate < self.input_sample_rate or (
            self.sample_rate % self.input_sample_rate != 0
        ):
            raise ValueError(
                "Continuous PCM output rate must be an integer multiple of the "
                f"decoder rate; got input={self.input_sample_rate}, "
                f"output={self.sample_rate}."
            )
        self._resample_factor = self.sample_rate // self.input_sample_rate
        self._resample_previous_sample: int | None = None
        self.output_blocksize = max(0, int(output_blocksize))
        self.prebuffer_seconds = max(0.0, float(prebuffer_seconds))
        self.latency = latency
        self.adaptive_prebuffer = adaptive_prebuffer
        self.start_gate = start_gate
        self.first_playback_callback = first_playback_callback
        self.first_playback_started_seconds: float | None = None
        self.first_playback_started_at: float | None = None
        self.first_playback_estimated_audible_at: float | None = None
        self.first_playback_estimated_audible_seconds: float | None = None
        self.first_enqueue_seconds: float | None = None
        self.prebuffer_ready_seconds: float | None = None
        self.gate_ready_seconds: float | None = None
        self.stream_enter_started_seconds: float | None = None
        self.stream_enter_finished_seconds: float | None = None
        self._output_latency_seconds: float | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._queued_bytes = 0
        self._queue_high_water_bytes = 0
        self._queue_overrun_count = 0
        self._device_underflow_count = 0
        self._initial_device_underflow = False
        self._queue_underrun_count = 0
        self._chunks_written = 0
        self._bytes_written = 0
        self._playback_started = False
        self._min_write_available_frames: int | None = None
        self._playback_buffered_until: float | None = None
        self._playback_drain_seconds = 0.0
        self._arrival_rate_bytes_per_second = float(self.sample_rate * 2)
        self._arrival_window_bytes = 0
        self._arrival_window_started_at = time.monotonic()
        self._prebuffer_target_bytes = 0
        self._actual_prebuffer_bytes = 0

    def start(self) -> None:
        self._thread.start()

    def enqueue_samples(self, samples: list[float] | tuple[float, ...]) -> None:
        pcm = float_samples_to_pcm16_bytes(samples)
        self.enqueue_pcm(pcm)

    def enqueue_pcm(self, pcm: bytes) -> None:
        if self._resample_factor > 1:
            pcm, self._resample_previous_sample = _upsample_pcm16_linear(
                pcm,
                factor=self._resample_factor,
                previous_sample=self._resample_previous_sample,
            )
        now = time.monotonic()
        with self._lock:
            if self.first_enqueue_seconds is None:
                self.first_enqueue_seconds = round(time.time() - self.started_at, 3)
            self._record_arrival_locked(len(pcm), now)
            self._queued_bytes += len(pcm)
            self._queue_high_water_bytes = max(
                self._queue_high_water_bytes,
                self._queued_bytes,
            )
            overrun_threshold_seconds = (
                self.prebuffer_seconds + DEFAULT_PLAYBACK_QUEUE_OVERRUN_SECONDS
            )
            if (
                self._playback_started
                and self._seconds_for_bytes(self._queued_bytes)
                > overrun_threshold_seconds
            ):
                self._queue_overrun_count += 1
        self._queue.put(pcm)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error

    def diagnostics(self) -> dict[str, object]:
        with self._lock:
            return {
                "device_underflow_count": self._device_underflow_count,
                "initial_device_underflow": self._initial_device_underflow,
                "queue_underrun_count": self._queue_underrun_count,
                "queue_overrun_count": self._queue_overrun_count,
                "queue_high_water_seconds": round(
                    self._seconds_for_bytes(self._queue_high_water_bytes),
                    3,
                ),
                "chunks_written": self._chunks_written,
                "bytes_written": self._bytes_written,
                "min_write_available_frames": self._min_write_available_frames,
                "latency": self.latency,
                "input_sample_rate": self.input_sample_rate,
                "output_sample_rate": self.sample_rate,
                "output_blocksize": self.output_blocksize,
                "output_latency_seconds": self._output_latency_seconds,
                "first_enqueue_seconds": self.first_enqueue_seconds,
                "prebuffer_ready_seconds": self.prebuffer_ready_seconds,
                "gate_ready_seconds": self.gate_ready_seconds,
                "stream_enter_started_seconds": self.stream_enter_started_seconds,
                "stream_enter_finished_seconds": self.stream_enter_finished_seconds,
                "playback_drain_seconds": round(self._playback_drain_seconds, 3),
                "adaptive_prebuffer": self.adaptive_prebuffer,
                "start_gate_released": (
                    self.start_gate.released if self.start_gate is not None else None
                ),
                "start_gate_cancelled": (
                    self.start_gate.cancelled if self.start_gate is not None else None
                ),
                "prebuffer_target_seconds": round(
                    self._seconds_for_bytes(self._prebuffer_target_bytes),
                    3,
                ),
                "actual_prebuffer_seconds": round(
                    self._seconds_for_bytes(self._actual_prebuffer_bytes),
                    3,
                ),
                "arrival_rate_audio_realtime_ratio": round(
                    self._arrival_rate_bytes_per_second / (self.sample_rate * 2),
                    3,
                ),
            }

    def _run(self) -> None:
        try:
            try:
                import sounddevice as sd
            except ImportError as exc:
                raise RuntimeError(
                    "Continuous speech playback requires sounddevice in the "
                    "active runtime. Run ./start.sh --refresh-deps to install "
                    "Audex-Mac dependencies into the vLLM Metal environment."
                ) from exc
            stream_kwargs: dict[str, object] = {
                "samplerate": self.sample_rate,
                "channels": 1,
                "dtype": "int16",
                "latency": self.latency,
            }
            if self.output_blocksize:
                stream_kwargs["blocksize"] = self.output_blocksize
            stream_context = sd.RawOutputStream(
                **stream_kwargs,
            )
            pending: list[bytes] = []
            pending_bytes = 0
            closed_during_prebuffer = False
            while True:
                target_prebuffer_bytes = self._target_prebuffer_bytes()
                if pending_bytes >= target_prebuffer_bytes:
                    break
                pcm = self._queue.get()
                if pcm is None:
                    closed_during_prebuffer = True
                    break
                pending.append(pcm)
                pending_bytes += len(pcm)
            with self._lock:
                self._prebuffer_target_bytes = target_prebuffer_bytes
                self._actual_prebuffer_bytes = pending_bytes
                self.prebuffer_ready_seconds = round(
                    time.time() - self.started_at,
                    3,
                )

            if self.start_gate is not None and not self.start_gate.wait():
                return
            self.gate_ready_seconds = round(time.time() - self.started_at, 3)

            self.stream_enter_started_seconds = round(
                time.time() - self.started_at,
                3,
            )
            with stream_context as stream:
                self.stream_enter_finished_seconds = round(
                    time.time() - self.started_at,
                    3,
                )
                for pcm in pending:
                    self._mark_dequeued(pcm)
                    self._write_stream_chunk(stream, pcm)
                if closed_during_prebuffer:
                    self._drain_playback_buffer()
                    return
                while True:
                    wait_started = time.monotonic()
                    pcm = self._queue.get()
                    if pcm is None:
                        self._drain_playback_buffer()
                        return
                    self._record_queue_wait(wait_started, time.monotonic())
                    self._mark_dequeued(pcm)
                    self._write_stream_chunk(stream, pcm)
        except BaseException as exc:
            self._error = exc

    def _write_stream_chunk(self, stream: Any, pcm: bytes) -> None:
        if self.first_playback_started_seconds is None:
            self.first_playback_started_at = time.time()
            self.first_playback_started_seconds = round(
                self.first_playback_started_at - self.started_at,
                3,
            )
            output_latency = getattr(stream, "latency", 0.0)
            if isinstance(output_latency, tuple):
                output_latency = output_latency[-1] if output_latency else 0.0
            with suppress(TypeError, ValueError):
                self._output_latency_seconds = max(0.0, float(output_latency))
            if self._output_latency_seconds is not None:
                self.first_playback_estimated_audible_at = (
                    self.first_playback_started_at + self._output_latency_seconds
                )
                self.first_playback_estimated_audible_seconds = round(
                    self.first_playback_estimated_audible_at - self.started_at,
                    3,
                )
            if self.first_playback_callback is not None:
                with suppress(Exception):
                    self.first_playback_callback(
                        self.first_playback_started_at,
                        self.first_playback_estimated_audible_at,
                    )
        write_available = getattr(stream, "write_available", None)
        if isinstance(write_available, int):
            with self._lock:
                if self._min_write_available_frames is None:
                    self._min_write_available_frames = write_available
                else:
                    self._min_write_available_frames = min(
                        self._min_write_available_frames,
                        write_available,
                    )
        first_write = self._chunks_written == 0
        underflowed = stream.write(pcm)
        frames = len(pcm) // 2
        with self._lock:
            self._playback_started = True
            self._chunks_written += 1
            self._bytes_written += len(pcm)
            if underflowed:
                if first_write:
                    self._initial_device_underflow = True
                else:
                    self._device_underflow_count += 1
            now = time.monotonic()
            buffered_from = max(now, self._playback_buffered_until or now)
            self._playback_buffered_until = buffered_from + frames / self.sample_rate

    def _drain_playback_buffer(self) -> None:
        buffered_until = self._playback_buffered_until
        if buffered_until is None:
            return
        remaining = max(0.0, buffered_until - time.monotonic())
        if remaining <= 0.0:
            return
        time.sleep(remaining)
        with self._lock:
            self._playback_drain_seconds += remaining

    def _mark_dequeued(self, pcm: bytes) -> None:
        with self._lock:
            self._queued_bytes = max(0, self._queued_bytes - len(pcm))

    def _record_queue_wait(self, wait_started: float, wait_finished: float) -> None:
        with self._lock:
            buffered_until = self._playback_buffered_until
            if (
                buffered_until is not None
                and wait_finished
                > buffered_until + PLAYBACK_QUEUE_UNDERRUN_GRACE_SECONDS
            ):
                self._queue_underrun_count += 1

    def _seconds_for_bytes(self, byte_count: int) -> float:
        return byte_count / (self.sample_rate * 2)

    def _record_arrival_locked(self, byte_count: int, now: float) -> None:
        self._arrival_window_bytes += byte_count
        elapsed = now - self._arrival_window_started_at
        if elapsed < DEFAULT_PLAYBACK_ARRIVAL_RATE_WINDOW_SECONDS:
            return
        if elapsed > 0:
            instant_rate = self._arrival_window_bytes / elapsed
            alpha = DEFAULT_PLAYBACK_ARRIVAL_RATE_EMA_ALPHA
            self._arrival_rate_bytes_per_second = (
                alpha * instant_rate
                + (1.0 - alpha) * self._arrival_rate_bytes_per_second
            )
        self._arrival_window_bytes = 0
        self._arrival_window_started_at = now

    def _target_prebuffer_bytes(self) -> int:
        configured = int(round(self.sample_rate * self.prebuffer_seconds)) * 2
        if not self.adaptive_prebuffer or configured <= 0:
            return configured
        realtime_bytes_per_second = self.sample_rate * 2
        arrival_ratio = max(
            0.001,
            self._arrival_rate_bytes_per_second / realtime_bytes_per_second,
        )
        adaptive_seconds = self.prebuffer_seconds / min(1.0, arrival_ratio)
        floor = (
            int(
                round(
                    self.sample_rate * DEFAULT_PLAYBACK_ADAPTIVE_MIN_PREBUFFER_SECONDS
                )
            )
            * 2
        )
        ceiling = (
            int(
                round(
                    self.sample_rate * DEFAULT_PLAYBACK_ADAPTIVE_MAX_PREBUFFER_SECONDS
                )
            )
            * 2
        )
        adaptive = int(round(self.sample_rate * adaptive_seconds)) * 2
        return max(floor, min(ceiling, adaptive))


def _upsample_pcm16_linear(
    pcm: bytes,
    *,
    factor: int,
    previous_sample: int | None = None,
) -> tuple[bytes, int | None]:
    """Upsample native-endian PCM16 without adding a NumPy hot-loop dependency."""

    if factor < 1:
        raise ValueError(f"PCM upsample factor must be positive, got {factor}.")
    if not pcm:
        return b"", previous_sample
    samples = array("h")
    samples.frombytes(pcm)
    if factor == 1:
        return pcm, int(samples[-1])
    output = array("h")
    prior = previous_sample
    for raw_sample in samples:
        sample = int(raw_sample)
        if prior is None:
            output.extend([sample] * factor)
        else:
            delta = sample - prior
            output.extend(
                prior + round(delta * step / factor) for step in range(1, factor + 1)
            )
        prior = sample
    return output.tobytes(), prior


def _features_to_mlx(input_features: Any, mx: Any) -> Any:
    return mx.array(input_features)


@dataclass(frozen=True, slots=True)
class _RecordingSnapshot:
    samples: tuple[float, ...]
    sample_count: int
    voice_revision: int
    last_voice_at: float | None
    quiet_sample_count: int


class _Recording:
    def __init__(self, stream: Any | None = None):
        self.stream = stream
        self.chunks: list[Any] = []
        self._lock = threading.Lock()
        self._voice_revision = 0
        self._last_voice_at: float | None = None
        self._sample_count = 0
        self._quiet_sample_count = 0

    def attach_stream(self, stream: Any) -> None:
        self.stream = stream

    def append_chunk(self, chunk: Any) -> None:
        samples = _mono_samples_from_capture_chunk(chunk)
        last_voiced_sample = max(
            (index for index, sample in enumerate(samples) if abs(sample) >= 0.012),
            default=None,
        )
        voiced = last_voiced_sample is not None
        with self._lock:
            self.chunks.append(chunk)
            self._sample_count += len(samples)
            if voiced:
                self._voice_revision += 1
                self._last_voice_at = time.monotonic()
                self._quiet_sample_count = len(samples) - int(last_voiced_sample) - 1
            else:
                self._quiet_sample_count += len(samples)

    def snapshot(self) -> _RecordingSnapshot:
        with self._lock:
            chunks = tuple(self.chunks)
            voice_revision = self._voice_revision
            last_voice_at = self._last_voice_at
            quiet_sample_count = self._quiet_sample_count
        samples = tuple(
            sample
            for chunk in chunks
            for sample in _mono_samples_from_capture_chunk(chunk)
        )
        return _RecordingSnapshot(
            samples=samples,
            sample_count=len(samples),
            voice_revision=voice_revision,
            last_voice_at=last_voice_at,
            quiet_sample_count=quiet_sample_count,
        )

    def activity(self) -> _RecordingSnapshot:
        with self._lock:
            return _RecordingSnapshot(
                samples=(),
                sample_count=self._sample_count,
                voice_revision=self._voice_revision,
                last_voice_at=self._last_voice_at,
                quiet_sample_count=self._quiet_sample_count,
            )

    def stop(self) -> list[float]:
        if self.stream is None:
            raise RuntimeError("Audio recording stream was not attached.")
        self.stream.stop()
        self.stream.close()
        samples = list(self.snapshot().samples)
        return samples or [0.0]


def _mono_samples_from_capture_chunk(chunk: Any) -> tuple[float, ...]:
    samples: list[float] = []
    for frame in chunk:
        if hasattr(frame, "__len__"):
            samples.append(float(sum(frame) / len(frame)))
        else:
            samples.append(float(frame))
    return tuple(samples)


def _start_recording() -> _Recording:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Microphone capture requires sounddevice in the active runtime. "
            "Use --input-wav for fixture mode if audio capture is unavailable."
        ) from exc

    recording = _Recording()

    def callback(indata, _frames, _time, status):
        if status:
            print(f"Audio capture status: {status}")
        recording.append_chunk(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    )
    recording.attach_stream(stream)
    stream.start()
    return recording
