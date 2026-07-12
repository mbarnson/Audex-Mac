"""Streaming CFG/no-CFG speech synthesis for the vLLM Audex runtime."""

from __future__ import annotations

import asyncio
import gc
import json
import os
import time
import wave
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .speech_decoder import AudexSpeechDecoderSession
from .speech_output import (
    SpeechOutputSmokeResult,
    float_samples_to_pcm16_bytes,
    mlx_waveform_to_pcm16_bytes,
    pcm16_bytes_peak_abs,
)
from .sts_cli import (
    DEFAULT_PLAYBACK_LATENCY,
    DEFAULT_SPEECH_TOKENS_PER_TEXT_TOKEN,
    TTS_SEGMENT_SILENCE_SECONDS,
    _ContinuousPcmPlayer,
    _PlaybackStartGate,
)
from .tts_text import (
    DEFAULT_CFG_TTS_ATOM_MAX_CHARS as DEFAULT_VLLM_CFG_TTS_ATOM_MAX_CHARS,
)
from .tts_text import (
    DEFAULT_CFG_TTS_MIN_TAIL_CHARS as DEFAULT_VLLM_CFG_TTS_MIN_TAIL_CHARS,
)
from .tts_text import (
    DEFAULT_TTS_MAX_CHARS_PER_CHUNK as DEFAULT_VLLM_TTS_MAX_CHARS_PER_CHUNK,
)
from .tts_text import (
    DEFAULT_TTS_SENTENCES_PER_CHUNK as DEFAULT_VLLM_TTS_SENTENCES_PER_CHUNK,
)
from .tts_text import (
    DEFAULT_TTS_STREAM_MIN_CHARS_PER_CHUNK as DEFAULT_VLLM_TTS_STREAM_MIN_CHARS_PER_CHUNK,
)
from .tts_text import (
    DEFAULT_TTS_TARGET_SEGMENTS as DEFAULT_VLLM_TTS_TARGET_SEGMENTS,
)
from .tts_text import (
    prepare_text_for_tts,
    split_cfg_spoken_tts_chunks,
    split_spoken_tts_chunks,
)
from .vllm_runtime import AudexAsyncVllmRuntime

DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES = 1
DEFAULT_VLLM_STREAM_DECODER_STEADY_CHUNK_FRAMES = 8
DEFAULT_VLLM_PLAYBACK_PREBUFFER_SECONDS = 0.8
DEFAULT_VLLM_CFG_PLAYBACK_PREBUFFER_SECONDS = 2.0
DEFAULT_VLLM_INTERLEAVED_PLAYBACK_PREBUFFER_SECONDS = 0.08
DEFAULT_VLLM_INTERLEAVED_PLAYBACK_LATENCY = 0.05
DEFAULT_VLLM_PLAYBACK_OUTPUT_SAMPLE_RATE = 192_000
DEFAULT_VLLM_PLAYBACK_OUTPUT_BLOCKSIZE = 512
DEFAULT_VLLM_INTERLEAVED_READY_BATCH_WINDOW_SECONDS = 0.05
DEFAULT_VLLM_INTERLEAVED_TAIL_BATCH_WINDOW_SECONDS = 0.05
DEFAULT_VLLM_INTERLEAVED_TAIL_MAX_BATCH_CHUNKS = 4
DEFAULT_VLLM_TTS_MIN_TOKENS_PER_CHUNK = 512
VLLM_TTS_CFG_ENV = "AUDEX_VLLM_TTS_CFG"
VLLM_TTS_CFG_PRIME_FIRST_SEGMENT_ENV = "AUDEX_VLLM_TTS_CFG_PRIME_FIRST_SEGMENT"
VLLM_CFG_TTS_TARGET_SEGMENTS_ENV = "AUDEX_VLLM_CFG_TTS_TARGET_SEGMENTS"
VLLM_NONPAGED_KV_CAPACITY_SEQS_ENV = "AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS"
VLLM_SKIP_SPEECH_DECODER_ENV = "AUDEX_VLLM_SKIP_SPEECH_DECODER"
VLLM_DECODER_PRIMER_FRAMES_ENV = "AUDEX_VLLM_DECODER_PRIMER_FRAMES"


@dataclass(frozen=True, slots=True)
class SpeechSynthesisRequest:
    text: str
    max_tokens: int
    play: bool
    artifact_prefix: str = "speech-output-vllm"
    decoder_chunk_frames: int = DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES
    decoder_steady_chunk_frames: int | None = None
    tts_target_segments: int = DEFAULT_VLLM_TTS_TARGET_SEGMENTS
    tts_chunk_source: AsyncIterator[str] | None = None
    text_to_tts_interleaved: bool = False
    tts_segments: tuple[str, ...] | None = None
    playback_start_gate: _PlaybackStartGate | None = None
    generation_finished_event: asyncio.Event | None = None
    pcm_chunk_sink: Callable[[int, bytes], None] | None = None


async def _static_async_iter(chunks: tuple[str, ...]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


class _QueuedTtsChunkSource:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._buffer: deque[str] = deque()

    async def put(self, chunk: str) -> None:
        await self._queue.put(chunk)

    async def finish(self) -> None:
        await self._queue.put(None)

    def __aiter__(self) -> _QueuedTtsChunkSource:
        return self

    async def __anext__(self) -> str:
        if self._buffer:
            return self._buffer.popleft()
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                raise StopAsyncIteration
            clean_chunk = chunk.strip()
            if clean_chunk:
                return clean_chunk

    def drain_finished_nowait(self, first_chunk: str) -> tuple[str, ...] | None:
        drained: list[str] = []
        finished = False
        while True:
            try:
                chunk = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if chunk is None:
                finished = True
                break
            clean_chunk = chunk.strip()
            if clean_chunk:
                drained.append(clean_chunk)
        if not finished:
            self._buffer.extendleft(reversed(drained))
            return None
        return (first_chunk, *drained)

    def drain_ready_nowait(self, first_chunk: str) -> tuple[tuple[str, ...], bool]:
        drained: list[str] = []
        finished = False
        while True:
            try:
                chunk = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if chunk is None:
                finished = True
                break
            clean_chunk = chunk.strip()
            if clean_chunk:
                drained.append(clean_chunk)
        return (first_chunk, *drained), finished


def _playback_prebuffer_seconds(
    tts_chunks: tuple[str, ...],
    *,
    tts_cfg_enabled: bool = False,
) -> float:
    total_chars = sum(len(chunk) for chunk in tts_chunks)
    base_prebuffer = DEFAULT_VLLM_PLAYBACK_PREBUFFER_SECONDS
    if tts_cfg_enabled:
        base_prebuffer = DEFAULT_VLLM_CFG_PLAYBACK_PREBUFFER_SECONDS
        if len(tts_chunks) <= 2 and total_chars <= (
            DEFAULT_VLLM_TTS_MAX_CHARS_PER_CHUNK * 2
        ):
            return base_prebuffer
        if total_chars >= DEFAULT_VLLM_TTS_MAX_CHARS_PER_CHUNK * 4:
            return 5.0
        if len(tts_chunks) >= 3:
            return 4.0
        return 3.0
    if len(tts_chunks) <= 1 and total_chars <= DEFAULT_VLLM_TTS_MAX_CHARS_PER_CHUNK:
        return base_prebuffer
    if total_chars >= DEFAULT_VLLM_TTS_MAX_CHARS_PER_CHUNK * 4:
        return max(base_prebuffer, 5.0)
    if len(tts_chunks) >= 3:
        return max(base_prebuffer, 4.0)
    return max(base_prebuffer, 3.0)


def _concurrent_tts_chunks_enabled() -> bool:
    return os.environ.get("AUDEX_VLLM_CONCURRENT_TTS_CHUNKS") == "1"


def _vllm_tts_cfg_enabled() -> bool:
    value = os.environ.get(VLLM_TTS_CFG_ENV)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _vllm_tts_cfg_prime_first_segment_enabled() -> bool:
    value = os.environ.get(VLLM_TTS_CFG_PRIME_FIRST_SEGMENT_ENV)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _effective_cfg_tts_target_segments(default: int) -> int:
    requested = _positive_int_env(VLLM_CFG_TTS_TARGET_SEGMENTS_ENV)
    target = requested if requested is not None else max(1, int(default))
    capacity = _positive_int_env(VLLM_NONPAGED_KV_CAPACITY_SEQS_ENV)
    if capacity is not None:
        target = min(target, capacity)
    return max(1, target)


def _positive_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _speech_decoder_skip_enabled() -> bool:
    value = os.environ.get(VLLM_SKIP_SPEECH_DECODER_ENV)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _interleaved_tail_batching_enabled() -> bool:
    value = os.environ.get("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL")
    if value is None:
        return False
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _interleaved_initial_ready_batching_enabled() -> bool:
    value = os.environ.get("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_INITIAL_READY")
    if value is None:
        return False
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _retain_tts_artifacts_enabled() -> bool:
    value = os.environ.get("AUDEX_VLLM_RETAIN_TTS_ARTIFACTS")
    if value is None:
        return False
    return value.strip().lower() not in {"0", "false", "no", "off"}


class VllmSpeechSynthesizer:
    """Own streamed token scheduling, codec decoding, playback, and artifacts."""

    def __init__(
        self,
        *,
        async_runtime: AudexAsyncVllmRuntime,
        mx: Any,
        full_model_path: Path,
        decoder_path: Path,
        decoder_config: Any,
        decoder_weights: Any,
        output_dir: Path,
        tokenizer: Any,
        speech_max_tokens: int | None,
        tts_cfg_enabled: bool,
        decoder_session_factory: Any = AudexSpeechDecoderSession,
        player_factory: Any = _ContinuousPcmPlayer,
        decoder_device: Any | None = None,
    ) -> None:
        self.async_runtime = async_runtime
        self.mx = mx
        self.full_model_path = full_model_path
        self.decoder_path = decoder_path
        self.decoder_config = decoder_config
        self.decoder_weights = decoder_weights
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.speech_max_tokens = speech_max_tokens
        self.tts_cfg_enabled = tts_cfg_enabled
        self.decoder_session_factory = decoder_session_factory
        self.player_factory = player_factory
        self.decoder_device = decoder_device

    def _speech_max_tokens_for_tts_chunk(
        self,
        text: str,
        *,
        utterance_max_tokens: int,
        chunk_count: int,
    ) -> int:
        if self.speech_max_tokens is not None or chunk_count <= 1:
            return utterance_max_tokens
        text_tokens = max(1, len(self.tokenizer.encode(text)))
        return max(
            DEFAULT_VLLM_TTS_MIN_TOKENS_PER_CHUNK,
            text_tokens * DEFAULT_SPEECH_TOKENS_PER_TEXT_TOKEN,
        )

    def _mlx_memory_snapshot(self) -> dict[str, int]:
        snapshot: dict[str, int] = {}
        for method_name, key in (
            ("get_active_memory", "active_bytes"),
            ("get_cache_memory", "cache_bytes"),
            ("get_peak_memory", "peak_bytes"),
        ):
            method = getattr(self.mx, method_name, None)
            if not callable(method):
                continue
            with suppress(Exception):
                snapshot[key] = int(method())
        return snapshot

    async def synthesize(
        self,
        request: SpeechSynthesisRequest,
    ) -> SpeechOutputSmokeResult:
        text = request.text
        max_tokens = request.max_tokens
        play = request.play
        artifact_prefix = request.artifact_prefix
        decoder_chunk_frames = request.decoder_chunk_frames
        decoder_steady_chunk_frames = request.decoder_steady_chunk_frames
        tts_target_segments = request.tts_target_segments
        tts_chunk_source = request.tts_chunk_source
        text_to_tts_interleaved = request.text_to_tts_interleaved
        tts_segments = request.tts_segments
        playback_start_gate = request.playback_start_gate
        generation_finished_event = request.generation_finished_event
        pcm_chunk_sink = request.pcm_chunk_sink
        if self.async_runtime is None:
            raise RuntimeError("Audex async vLLM runtime is not configured.")

        started_at = time.time()
        decoder_chunk_frames = max(
            1,
            int(
                _positive_int_env(VLLM_DECODER_PRIMER_FRAMES_ENV)
                or decoder_chunk_frames
            ),
        )
        decoder_steady_chunk_frames = max(
            1,
            int(
                decoder_steady_chunk_frames
                if decoder_steady_chunk_frames is not None
                else (
                    DEFAULT_VLLM_STREAM_DECODER_STEADY_CHUNK_FRAMES
                    if decoder_chunk_frames == DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES
                    else decoder_chunk_frames
                )
            ),
        )
        if tts_chunk_source is not None and tts_segments is not None:
            raise ValueError("tts_chunk_source and tts_segments are mutually exclusive")
        static_tts_chunks = tts_chunk_source is None
        tts_input_text = prepare_text_for_tts(text)
        configured_tts_cfg = getattr(self, "tts_cfg_enabled", None)
        tts_cfg_enabled = (
            _vllm_tts_cfg_enabled()
            if configured_tts_cfg is None
            else bool(configured_tts_cfg)
        )
        effective_tts_target_segments = (
            _effective_cfg_tts_target_segments(tts_target_segments)
            if tts_cfg_enabled
            else tts_target_segments
        )
        if tts_segments is not None:
            tts_chunks = [prepare_text_for_tts(segment) for segment in tts_segments]
        else:
            tts_chunks = (
                list(
                    split_cfg_spoken_tts_chunks(
                        tts_input_text,
                        target_segments=effective_tts_target_segments,
                    )
                )
                if tts_cfg_enabled
                else (
                    list(split_spoken_tts_chunks(tts_input_text))
                    if static_tts_chunks
                    else []
                )
            )
        decoder_session = self.decoder_session_factory(
            weights=self.decoder_weights,
            config=self.decoder_config,
            chunk_frames=decoder_chunk_frames,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        wav_path = self.output_dir / f"{artifact_prefix}-{timestamp}.wav"
        run_log_path = self.output_dir / f"{artifact_prefix}-{timestamp}.json"
        live_milestone_path = (
            self.output_dir / f"{artifact_prefix}-{timestamp}.live.json"
        )
        sample_count = 0
        all_token_ids: list[int] = []
        all_codec_frames: list[int] = []
        decoded_chunk_count = 0
        peak_abs = 0.0
        finite = True
        pcm_pack_seconds = 0.0
        pcm_pack_fast_path_count = 0
        pcm_pack_fallback_count = 0
        player_enqueue_seconds = 0.0
        wav_write_seconds = 0.0
        decoder_push_seconds = 0.0
        decoder_flush_seconds = 0.0
        decoder_reset_seconds = 0.0
        mlx_clear_cache_seconds = 0.0
        mlx_clear_cache_count = 0
        reached_end_token = False
        all_segments_reached_end = True
        first_audio_ready_seconds: float | None = None
        first_audio_ready_at: float | None = None
        playback_diagnostics: dict[str, object] | None = None
        stream_event_count = 0
        tail_decode_start_event = asyncio.Event()
        first_token_event_seconds: float | None = None
        last_token_event_seconds: float | None = None
        first_tts_chunk_ready_seconds: float | None = None
        first_codec_frame_seconds: float | None = None
        last_codec_frame_seconds: float | None = None
        first_codec_frame_wall_seconds: float | None = None
        last_codec_frame_wall_seconds: float | None = None
        first_decoder_push_started_seconds: float | None = None
        first_decoder_push_finished_seconds: float | None = None
        first_decoder_push_frame_count: int | None = None
        stream_finished_seconds: float | None = None
        stream_finished_raw_seconds: float | None = None
        playback_close_seconds: float | None = None
        segment_codec_frame_counts: dict[int, int] = {}
        segment_finished: dict[int, bool] = {}
        segment_max_tokens: dict[int, int] = {}
        segment_texts: dict[int, str] = {}
        segment_token_counts: dict[int, int] = {}
        segment_ready_seconds: dict[int, float] = {}
        segment_wall_seconds: dict[int, float] = {}
        segment_mlx_memory_after_clear: dict[int, dict[str, int]] = {}
        pending_codec_frames: list[int] = []
        retain_heavy_tts_artifacts = _retain_tts_artifacts_enabled()
        generated_token_id_count = 0
        generated_codec_frame_count = 0
        decoder_primer_pushed = False
        player: _ContinuousPcmPlayer | None = None
        playback_prebuffer_seconds: float | None = None
        speech_decoder_skipped = _speech_decoder_skip_enabled()
        mlx_memory_start = self._mlx_memory_snapshot()
        mlx_memory_after_stream: dict[str, int] | None = None
        cfg_concurrent_tts_chunks = (
            static_tts_chunks and tts_cfg_enabled and len(tts_chunks) > 1
        )
        cfg_prime_first_segment = (
            cfg_concurrent_tts_chunks and _vllm_tts_cfg_prime_first_segment_enabled()
        )
        no_cfg_concurrent_tts_chunks = (
            static_tts_chunks
            and not tts_cfg_enabled
            and _concurrent_tts_chunks_enabled()
            and len(tts_chunks) > 1
        )
        concurrent_tts_chunks = (
            cfg_concurrent_tts_chunks or no_cfg_concurrent_tts_chunks
        )
        interleaved_tail_batched = False
        interleaved_all_ready_batched = False
        interleaved_initial_ready_batched = False
        tail_task: asyncio.Task[None] | None = None
        wav_stream = wave.open(str(wav_path), "wb")  # noqa: SIM115
        wav_stream.setnchannels(1)
        wav_stream.setsampwidth(2)
        wav_stream.setframerate(self.decoder_config.sample_rate)

        def emit_pcm(pcm: bytes, *, samples_len: int, finite_value: bool) -> None:
            nonlocal sample_count, finite, peak_abs
            nonlocal player_enqueue_seconds, wav_write_seconds
            sample_count += samples_len
            finite = finite and finite_value
            peak_abs = max(peak_abs, pcm16_bytes_peak_abs(pcm))
            wav_write_started = time.time()
            wav_stream.writeframesraw(pcm)
            wav_write_seconds += time.time() - wav_write_started
            if pcm_chunk_sink is not None:
                pcm_chunk_sink(self.decoder_config.sample_rate, pcm)
            if player is not None:
                enqueue_started = time.time()
                player.enqueue_pcm(pcm)
                player_enqueue_seconds += time.time() - enqueue_started

        def emit_samples(samples: tuple[float, ...]) -> None:
            nonlocal pcm_pack_seconds, pcm_pack_fallback_count
            pack_started = time.time()
            pcm = float_samples_to_pcm16_bytes(samples)
            pcm_pack_seconds += time.time() - pack_started
            pcm_pack_fallback_count += 1
            emit_pcm(
                pcm,
                samples_len=len(samples),
                finite_value=all(sample == sample for sample in samples),
            )

        def emit_waveform(waveform) -> None:
            nonlocal pcm_pack_seconds
            nonlocal decoded_chunk_count, first_audio_ready_seconds
            nonlocal first_audio_ready_at
            nonlocal pcm_pack_fast_path_count, pcm_pack_fallback_count
            pack_started = time.time()
            pcm = mlx_waveform_to_pcm16_bytes(waveform, self.mx)
            pcm_pack_seconds += time.time() - pack_started
            if pcm is not None:
                pcm_pack_fast_path_count += 1
                emit_pcm(pcm, samples_len=len(pcm) // 2, finite_value=True)
            else:
                fallback_started = time.time()
                samples = tuple(float(sample) for sample in waveform.tolist())
                pcm = float_samples_to_pcm16_bytes(samples)
                pcm_pack_seconds += time.time() - fallback_started
                pcm_pack_fallback_count += 1
                emit_pcm(
                    pcm,
                    samples_len=len(samples),
                    finite_value=all(sample == sample for sample in samples),
                )
            decoded_chunk_count += 1
            if first_audio_ready_seconds is None:
                first_audio_ready_at = time.time()
                first_audio_ready_seconds = round(
                    first_audio_ready_at - started_at,
                    3,
                )
                tail_decode_start_event.set()

        def emit_silence() -> None:
            if speech_decoder_skipped:
                return
            silence_samples = (0.0,) * int(
                round(self.decoder_config.sample_rate * TTS_SEGMENT_SILENCE_SECONDS)
            )
            emit_samples(silence_samples)

        def push_decoder_frames(
            frames: tuple[tuple[int, ...], ...],
            *,
            chunk_frames: int,
        ) -> None:
            nonlocal decoder_push_seconds
            nonlocal first_decoder_push_started_seconds
            nonlocal first_decoder_push_finished_seconds
            nonlocal first_decoder_push_frame_count
            if speech_decoder_skipped:
                return
            push_started = time.time()
            decoder_session.chunk_frames = chunk_frames
            if first_decoder_push_started_seconds is None:
                first_decoder_push_started_seconds = round(push_started - started_at, 3)
                first_decoder_push_frame_count = len(frames)
            for sample_rate, waveform in decoder_session.push(frames):
                if sample_rate != self.decoder_config.sample_rate:
                    raise ValueError(
                        "Audex decoder emitted unexpected sample rate "
                        f"{sample_rate}; expected {self.decoder_config.sample_rate}"
                    )
                emit_waveform(waveform)
            if first_decoder_push_finished_seconds is None:
                first_decoder_push_finished_seconds = round(time.time() - started_at, 3)
            decoder_push_seconds += time.time() - push_started

        def active_decoder_chunk_frames() -> int:
            if decoder_primer_pushed:
                return decoder_steady_chunk_frames
            return decoder_chunk_frames

        def decoder_push_threshold() -> int:
            threshold = active_decoder_chunk_frames()
            if decoder_primer_pushed:
                return threshold
            lookahead_steps = int(getattr(self.decoder_config, "lookahead_steps", 0))
            return threshold + max(0, lookahead_steps)

        def flush_decoder() -> None:
            nonlocal decoder_flush_seconds
            if speech_decoder_skipped:
                pending_codec_frames.clear()
                return
            if pending_codec_frames:
                push_decoder_frames(
                    tuple((frame,) for frame in pending_codec_frames),
                    chunk_frames=active_decoder_chunk_frames(),
                )
                pending_codec_frames.clear()
            flush_started = time.time()
            for sample_rate, waveform in decoder_session.flush():
                if sample_rate != self.decoder_config.sample_rate:
                    raise ValueError(
                        "Audex decoder emitted unexpected sample rate "
                        f"{sample_rate}; expected {self.decoder_config.sample_rate}"
                    )
                emit_waveform(waveform)
            decoder_flush_seconds += time.time() - flush_started

        def clear_mlx_cache() -> None:
            nonlocal mlx_clear_cache_count, mlx_clear_cache_seconds
            clear_cache = getattr(self.mx, "clear_cache", None)
            if callable(clear_cache):
                with suppress(Exception):
                    clear_started = time.time()
                    clear_cache()
                    mlx_clear_cache_seconds += time.time() - clear_started
                    mlx_clear_cache_count += 1
            gc.collect()

        def consume_event(event, *, segment_index: int | None = None) -> None:
            nonlocal decoder_primer_pushed
            nonlocal stream_event_count, first_token_event_seconds
            nonlocal last_token_event_seconds, first_codec_frame_seconds
            nonlocal last_codec_frame_seconds, first_codec_frame_wall_seconds
            nonlocal last_codec_frame_wall_seconds
            nonlocal generated_token_id_count, generated_codec_frame_count
            if segment_index is None:
                segment_index = int(getattr(event, "segment_index", 0) or 0)
            stream_event_count += 1
            if first_token_event_seconds is None:
                first_token_event_seconds = event.elapsed_seconds
            last_token_event_seconds = event.elapsed_seconds
            if event.generated_token_ids:
                segment_token_counts[segment_index] = len(event.generated_token_ids)
                generated_token_id_count = sum(segment_token_counts.values())
                segment_texts.setdefault(segment_index, tts_chunks[segment_index])
                segment_finished[segment_index] = event.reached_end_token
                segment_max_tokens.setdefault(
                    segment_index,
                    self._speech_max_tokens_for_tts_chunk(
                        tts_chunks[segment_index],
                        utterance_max_tokens=max_tokens,
                        chunk_count=len(tts_chunks),
                    ),
                )
                if retain_heavy_tts_artifacts:
                    all_token_ids_by_segment[segment_index] = event.generated_token_ids
            if event.new_codec_frames:
                generated_codec_frame_count += len(event.new_codec_frames)
                codec_frame_wall_seconds = round(time.time() - started_at, 3)
                if first_codec_frame_seconds is None:
                    first_codec_frame_seconds = event.elapsed_seconds
                    first_codec_frame_wall_seconds = codec_frame_wall_seconds
                last_codec_frame_seconds = event.elapsed_seconds
                last_codec_frame_wall_seconds = codec_frame_wall_seconds
                if retain_heavy_tts_artifacts:
                    all_codec_frames.extend(event.new_codec_frames)
                segment_codec_frame_counts[segment_index] = (
                    segment_codec_frame_counts.get(segment_index, 0)
                    + len(event.new_codec_frames)
                )
                pending_codec_frames.extend(event.new_codec_frames)
                if speech_decoder_skipped:
                    tail_decode_start_event.set()
                while len(pending_codec_frames) >= decoder_push_threshold():
                    threshold = decoder_push_threshold()
                    chunk_frames = active_decoder_chunk_frames()
                    ready_frames = pending_codec_frames[:threshold]
                    del pending_codec_frames[:threshold]
                    decoder_primer_pushed = True
                    push_decoder_frames(
                        tuple((frame,) for frame in ready_frames),
                        chunk_frames=chunk_frames,
                    )

        all_token_ids_by_segment: dict[int, tuple[int, ...]] = {}

        def finalize_segment(
            segment_index: int,
            *,
            segment_started: float,
            segment_reached_end: bool,
            emit_trailing_silence: bool,
        ) -> None:
            nonlocal decoder_reset_seconds, all_segments_reached_end
            flush_decoder()
            reset_decoder = getattr(decoder_session, "reset", None)
            if callable(reset_decoder) and not speech_decoder_skipped:
                reset_started = time.time()
                reset_decoder()
                decoder_reset_seconds += time.time() - reset_started
            if not speech_decoder_skipped:
                clear_mlx_cache()
            segment_mlx_memory_after_clear[segment_index] = self._mlx_memory_snapshot()
            if emit_trailing_silence:
                emit_silence()
            if retain_heavy_tts_artifacts:
                all_token_ids.extend(all_token_ids_by_segment.get(segment_index, ()))
            segment_codec_frame_counts.setdefault(segment_index, 0)
            segment_finished[segment_index] = segment_reached_end
            all_segments_reached_end = all_segments_reached_end and segment_reached_end
            segment_wall_seconds[segment_index] = round(
                time.time() - segment_started, 3
            )
            tail_decode_start_event.set()

        async def stream_interleaved_tail_chunks(
            chunk_iterator: AsyncIterator[str],
            *,
            tail_offset: int,
            queue: asyncio.Queue[object],
        ) -> None:
            async def collect_tail_chunks(
                output: asyncio.Queue[str | None],
            ) -> None:
                try:
                    async for tail_chunk in chunk_iterator:
                        clean_tail_chunk = prepare_text_for_tts(tail_chunk)
                        if clean_tail_chunk:
                            await output.put(clean_tail_chunk)
                    await output.put(None)
                except BaseException as exc:
                    await output.put(None)
                    raise exc

            async def drain_tail_batch(
                input_queue: asyncio.Queue[str | None],
                first_chunk: str,
            ) -> tuple[tuple[str, ...], bool]:
                chunks = [first_chunk]
                finished = False
                await asyncio.sleep(DEFAULT_VLLM_INTERLEAVED_TAIL_BATCH_WINDOW_SECONDS)
                while len(chunks) < DEFAULT_VLLM_INTERLEAVED_TAIL_MAX_BATCH_CHUNKS:
                    try:
                        tail_item = input_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if tail_item is None:
                        finished = True
                        break
                    chunks.append(tail_item)
                return tuple(chunks), finished

            tail_chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()
            collector_task = asyncio.create_task(collect_tail_chunks(tail_chunk_queue))
            try:
                next_tail_offset = tail_offset
                finished = False
                while not finished:
                    tail_item = await tail_chunk_queue.get()
                    if tail_item is None:
                        finished = True
                        break
                    tail_chunks, finished = await drain_tail_batch(
                        tail_chunk_queue,
                        tail_item,
                    )
                    tail_started = time.time()
                    tail_max_tokens = tuple(
                        self._speech_max_tokens_for_tts_chunk(
                            tail_chunk,
                            utterance_max_tokens=max_tokens,
                            chunk_count=len(tail_chunks) + next_tail_offset,
                        )
                        for tail_chunk in tail_chunks
                    )
                    await queue.put(
                        (
                            "meta",
                            next_tail_offset,
                            tail_chunks,
                            tail_max_tokens,
                            tail_started,
                        )
                    )
                    if tts_cfg_enabled:
                        event_source = (
                            self.async_runtime.stream_tts_cfg_segments_codec_frames(
                                tail_chunks,
                                max_tokens_per_segment=tail_max_tokens,
                            )
                        )
                    else:
                        event_source = (
                            self.async_runtime.stream_tts_segmented_codec_frames(
                                tail_chunks,
                                max_tokens_per_segment=tail_max_tokens,
                            )
                        )
                    await tail_decode_start_event.wait()
                    async for event in event_source:
                        await queue.put(("event", event))
                    next_tail_offset += len(tail_chunks)
                await collector_task
                await queue.put(None)
            except BaseException as exc:
                if not collector_task.done():
                    collector_task.cancel()
                    with suppress(BaseException):
                        await collector_task
                await queue.put(("error", exc))

        async def consume_interleaved_tail_queue(
            *,
            queue: asyncio.Queue[object],
            task: asyncio.Task[None],
            default_tail_offset: int,
        ) -> int:
            nonlocal interleaved_tail_batched
            tail_started = time.time()
            tail_offset = default_tail_offset
            while True:
                tail_item = await queue.get()
                if tail_item is None:
                    break
                tail_kind = tail_item[0]
                if tail_kind == "error":
                    raise tail_item[1]
                if tail_kind == "meta":
                    (
                        _kind,
                        tail_offset,
                        tail_chunks,
                        tail_max_tokens,
                        tail_started,
                    ) = tail_item
                    interleaved_tail_batched = True
                    emit_silence()
                    for tail_index, tail_chunk in enumerate(tail_chunks):
                        actual_index = tail_offset + tail_index
                        tts_chunks.append(tail_chunk)
                        segment_texts[actual_index] = tail_chunk
                        segment_ready_seconds[actual_index] = round(
                            tail_started - started_at,
                            3,
                        )
                        segment_max_tokens[actual_index] = tail_max_tokens[tail_index]
                    continue
                _kind, event = tail_item
                actual_index = tail_offset + int(event.segment_index)
                consume_event(event, segment_index=actual_index)
                segment_reached_end_by_index[actual_index] = event.reached_end_token
                if event.segment_finished:
                    finalize_segment(
                        actual_index,
                        segment_started=tail_started,
                        segment_reached_end=segment_reached_end_by_index.get(
                            actual_index,
                            False,
                        ),
                        emit_trailing_silence=actual_index < len(tts_chunks) - 1,
                    )
            await task
            return len(tts_chunks)

        try:
            if play and not speech_decoder_skipped:
                playback_prebuffer_seconds = (
                    DEFAULT_VLLM_INTERLEAVED_PLAYBACK_PREBUFFER_SECONDS
                    if text_to_tts_interleaved
                    else _playback_prebuffer_seconds(
                        tuple(tts_chunks),
                        tts_cfg_enabled=tts_cfg_enabled,
                    )
                )
                player_kwargs: dict[str, object] = {
                    "started_at": started_at,
                    "sample_rate": self.decoder_config.sample_rate,
                    "output_sample_rate": DEFAULT_VLLM_PLAYBACK_OUTPUT_SAMPLE_RATE,
                    "output_blocksize": DEFAULT_VLLM_PLAYBACK_OUTPUT_BLOCKSIZE,
                    "prebuffer_seconds": playback_prebuffer_seconds,
                    "latency": (
                        DEFAULT_VLLM_INTERLEAVED_PLAYBACK_LATENCY
                        if text_to_tts_interleaved
                        else DEFAULT_PLAYBACK_LATENCY
                    ),
                }
                if playback_start_gate is not None:

                    def record_first_playback(
                        device_write_at: float,
                        estimated_audible_at: float | None,
                    ) -> None:
                        submitted_at = playback_start_gate.released_at
                        onset_at = estimated_audible_at or device_write_at
                        submit_to_onset = (
                            max(0.0, onset_at - submitted_at)
                            if submitted_at is not None
                            else None
                        )
                        live_milestone_path.write_text(
                            json.dumps(
                                {
                                    "schema_version": 1,
                                    "source": "model_response_tts",
                                    "cfg_enabled": tts_cfg_enabled,
                                    "text_to_tts_interleaved": (
                                        text_to_tts_interleaved
                                    ),
                                    "submitted_at": submitted_at,
                                    "first_device_write_at": device_write_at,
                                    "first_estimated_audible_at": (
                                        estimated_audible_at
                                    ),
                                    "submit_to_first_estimated_audible_seconds": (
                                        round(submit_to_onset, 3)
                                        if submit_to_onset is not None
                                        else None
                                    ),
                                },
                                indent=2,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        if submit_to_onset is not None:
                            print(
                                "Audex STS: first semantic audio estimated at "
                                f"{submit_to_onset:.3f}s after submit.",
                                flush=True,
                            )

                    player_kwargs["start_gate"] = playback_start_gate
                    player_kwargs["first_playback_callback"] = record_first_playback
                player = self.player_factory(
                    **player_kwargs,
                )
                player.start()
            segment_started_at: dict[int, float] = {}
            segment_reached_end_by_index: dict[int, bool] = {}
            for segment_index, tts_chunk in enumerate(tts_chunks):
                segment_texts[segment_index] = tts_chunk
                segment_max_tokens[segment_index] = (
                    self._speech_max_tokens_for_tts_chunk(
                        tts_chunk,
                        utterance_max_tokens=max_tokens,
                        chunk_count=len(tts_chunks),
                    )
                )
            if concurrent_tts_chunks:
                shared_segment_start = time.time()
                first_tts_chunk_ready_seconds = 0.0
                for segment_index in range(len(tts_chunks)):
                    segment_ready_seconds[segment_index] = 0.0
                max_tokens_per_segment = tuple(
                    segment_max_tokens[index] for index in range(len(tts_chunks))
                )
                if tts_cfg_enabled:
                    event_source = (
                        self.async_runtime.stream_tts_cfg_segments_codec_frames(
                            tuple(tts_chunks),
                            max_tokens_per_segment=max_tokens_per_segment,
                            prime_first_segment=cfg_prime_first_segment,
                        )
                    )
                else:
                    event_source = self.async_runtime.stream_tts_segmented_codec_frames(
                        tuple(tts_chunks),
                        max_tokens_per_segment=max_tokens_per_segment,
                    )
                async for event in event_source:
                    segment_index = int(event.segment_index)
                    segment_started_at.setdefault(segment_index, shared_segment_start)
                    consume_event(event)
                    segment_reached_end_by_index[segment_index] = (
                        event.reached_end_token
                    )
                    if event.segment_finished:
                        finalize_segment(
                            segment_index,
                            segment_started=segment_started_at[segment_index],
                            segment_reached_end=segment_reached_end_by_index.get(
                                segment_index,
                                False,
                            ),
                            emit_trailing_silence=segment_index < len(tts_chunks) - 1,
                        )
            else:
                if static_tts_chunks:
                    chunk_iterator = _static_async_iter(tuple(tts_chunks))
                elif tts_chunk_source is not None:
                    chunk_iterator = tts_chunk_source
                else:
                    chunk_iterator = _static_async_iter(())

                segment_index = 0
                tail_queue: asyncio.Queue[object] | None = None
                async for tts_chunk in chunk_iterator:
                    chunk_ready_seconds = round(time.time() - started_at, 3)
                    segment_ready_seconds[segment_index] = chunk_ready_seconds
                    if first_tts_chunk_ready_seconds is None:
                        first_tts_chunk_ready_seconds = chunk_ready_seconds
                    initial_batch_chunks: tuple[str, ...] | None = None
                    source_finished_after_first_chunk = False
                    if (
                        segment_index == 0
                        and not static_tts_chunks
                        and text_to_tts_interleaved
                        and not tts_cfg_enabled
                        and _interleaved_tail_batching_enabled()
                        and _interleaved_initial_ready_batching_enabled()
                    ):
                        drain_ready = getattr(
                            chunk_iterator,
                            "drain_ready_nowait",
                            None,
                        )
                        if callable(drain_ready):
                            await asyncio.sleep(
                                DEFAULT_VLLM_INTERLEAVED_READY_BATCH_WINDOW_SECONDS
                            )
                            (
                                initial_batch_chunks,
                                source_finished_after_first_chunk,
                            ) = drain_ready(tts_chunk)
                        else:
                            drain_finished = getattr(
                                chunk_iterator,
                                "drain_finished_nowait",
                                None,
                            )
                            if callable(drain_finished):
                                await asyncio.sleep(
                                    DEFAULT_VLLM_INTERLEAVED_READY_BATCH_WINDOW_SECONDS
                                )
                                initial_batch_chunks = drain_finished(tts_chunk)
                                source_finished_after_first_chunk = (
                                    initial_batch_chunks is not None
                                )
                    if (
                        initial_batch_chunks is not None
                        and len(initial_batch_chunks) > 1
                    ):
                        if source_finished_after_first_chunk:
                            interleaved_all_ready_batched = True
                        else:
                            interleaved_initial_ready_batched = True
                        shared_segment_start = time.time()
                        initial_max_tokens = tuple(
                            self._speech_max_tokens_for_tts_chunk(
                                initial_chunk,
                                utterance_max_tokens=max_tokens,
                                chunk_count=len(initial_batch_chunks),
                            )
                            for initial_chunk in initial_batch_chunks
                        )
                        for initial_index, initial_chunk in enumerate(
                            initial_batch_chunks
                        ):
                            tts_chunks.append(initial_chunk)
                            segment_texts[initial_index] = initial_chunk
                            segment_ready_seconds[initial_index] = chunk_ready_seconds
                            segment_max_tokens[initial_index] = initial_max_tokens[
                                initial_index
                            ]
                        if not source_finished_after_first_chunk:
                            tail_queue = asyncio.Queue()
                            tail_task = asyncio.create_task(
                                stream_interleaved_tail_chunks(
                                    chunk_iterator,
                                    tail_offset=len(tts_chunks),
                                    queue=tail_queue,
                                )
                            )
                        if tts_cfg_enabled:
                            initial_event_source = (
                                self.async_runtime.stream_tts_cfg_segments_codec_frames(
                                    initial_batch_chunks,
                                    max_tokens_per_segment=initial_max_tokens,
                                )
                            )
                        else:
                            initial_event_source = (
                                self.async_runtime.stream_tts_segmented_codec_frames(
                                    initial_batch_chunks,
                                    max_tokens_per_segment=initial_max_tokens,
                                )
                            )
                        async for event in initial_event_source:
                            actual_index = int(event.segment_index)
                            segment_started_at.setdefault(
                                actual_index,
                                shared_segment_start,
                            )
                            consume_event(event, segment_index=actual_index)
                            segment_reached_end_by_index[actual_index] = (
                                event.reached_end_token
                            )
                            if event.segment_finished:
                                finalize_segment(
                                    actual_index,
                                    segment_started=segment_started_at[actual_index],
                                    segment_reached_end=(
                                        segment_reached_end_by_index.get(
                                            actual_index,
                                            False,
                                        )
                                    ),
                                    emit_trailing_silence=actual_index
                                    < len(tts_chunks) - 1,
                                )
                        segment_index = len(tts_chunks)
                        if source_finished_after_first_chunk:
                            break
                        if tail_queue is None or tail_task is None:
                            raise RuntimeError(
                                "Audex interleaved TTS tail task was not started."
                            )
                        segment_index = await consume_interleaved_tail_queue(
                            queue=tail_queue,
                            task=tail_task,
                            default_tail_offset=segment_index,
                        )
                        break
                    tts_chunk = prepare_text_for_tts(tts_chunk)
                    if not tts_chunk:
                        continue
                    if not static_tts_chunks:
                        tts_chunks.append(tts_chunk)
                    if segment_index > 0 and not static_tts_chunks:
                        emit_silence()
                    segment_texts[segment_index] = tts_chunk
                    segment_max_tokens[segment_index] = (
                        self._speech_max_tokens_for_tts_chunk(
                            tts_chunk,
                            utterance_max_tokens=max_tokens,
                            chunk_count=(
                                len(tts_chunks)
                                if static_tts_chunks
                                else (1 if source_finished_after_first_chunk else 2)
                            ),
                        )
                    )
                    segment_started = time.time()
                    segment_reached_end = False
                    if (
                        segment_index == 0
                        and not static_tts_chunks
                        and text_to_tts_interleaved
                        and _interleaved_tail_batching_enabled()
                        and not source_finished_after_first_chunk
                    ):
                        tail_queue = asyncio.Queue()
                        tail_task = asyncio.create_task(
                            stream_interleaved_tail_chunks(
                                chunk_iterator,
                                tail_offset=segment_index + 1,
                                queue=tail_queue,
                            )
                        )
                    if tts_cfg_enabled:
                        event_source = self.async_runtime.stream_tts_cfg_codec_frames(
                            tts_chunk,
                            max_tokens=segment_max_tokens[segment_index],
                        )
                    else:
                        event_source = self.async_runtime.stream_tts_codec_frames(
                            tts_chunk,
                            max_tokens=segment_max_tokens[segment_index],
                        )
                    async for event in event_source:
                        consume_event(event, segment_index=segment_index)
                        segment_reached_end = event.reached_end_token
                    finalize_segment(
                        segment_index,
                        segment_started=segment_started,
                        segment_reached_end=segment_reached_end,
                        emit_trailing_silence=(
                            static_tts_chunks and segment_index < len(tts_chunks) - 1
                        ),
                    )
                    segment_index += 1
                    if source_finished_after_first_chunk:
                        break
                    if tail_task is not None and tail_queue is not None:
                        segment_index = await consume_interleaved_tail_queue(
                            queue=tail_queue,
                            task=tail_task,
                            default_tail_offset=segment_index,
                        )
                        break
            reached_end_token = all_segments_reached_end
        finally:
            mlx_memory_after_stream = self._mlx_memory_snapshot()
            if tail_task is not None and not tail_task.done():
                tail_task.cancel()
                with suppress(BaseException):
                    await tail_task
            if generation_finished_event is not None:
                generation_finished_event.set()
            if player is not None:
                close_started = time.time()
                await asyncio.to_thread(player.close)
                playback_close_seconds = round(time.time() - close_started, 3)
                playback_diagnostics = player.diagnostics()
            if not speech_decoder_skipped:
                clear_mlx_cache()
            stream_finished_raw_seconds = time.time() - started_at
            stream_finished_seconds = round(stream_finished_raw_seconds, 3)
            wav_finalize_started = time.time()
            wav_stream.writeframes(b"")
            wav_stream.close()
            wav_write_seconds += time.time() - wav_finalize_started

        if generated_codec_frame_count <= 0:
            raise RuntimeError("Audex async vLLM TTS produced no speech codec frames.")
        if sample_count <= 0 and not speech_decoder_skipped:
            raise RuntimeError("Audex async vLLM decoder produced no waveform samples.")
        first_playback_started_seconds = (
            player.first_playback_started_seconds if player is not None else None
        )
        first_playback_started_at = (
            getattr(player, "first_playback_started_at", None)
            if player is not None
            else None
        )
        first_playback_estimated_audible_at = (
            getattr(player, "first_playback_estimated_audible_at", None)
            if player is not None
            else None
        )
        audio_duration_seconds = sample_count / self.decoder_config.sample_rate
        stream_realtime_ratio = (
            audio_duration_seconds / last_codec_frame_seconds
            if last_codec_frame_seconds and last_codec_frame_seconds > 0
            else None
        )
        codec_frames_per_second = (
            generated_codec_frame_count / last_codec_frame_seconds
            if last_codec_frame_seconds and last_codec_frame_seconds > 0
            else None
        )
        first_playback_after_audio_seconds = (
            first_playback_started_seconds - first_audio_ready_seconds
            if (
                first_playback_started_seconds is not None
                and first_audio_ready_seconds is not None
            )
            else None
        )
        first_decoder_wait_after_codec_seconds = (
            first_decoder_push_started_seconds - first_codec_frame_wall_seconds
            if (
                first_decoder_push_started_seconds is not None
                and first_codec_frame_wall_seconds is not None
            )
            else None
        )

        segment_hit_max_tokens = {
            index: (
                segment_token_counts.get(index, 0) >= segment_max_tokens.get(index, 0)
                and not segment_finished.get(index, False)
            )
            for index in segment_max_tokens
        }
        hit_max_tokens = any(segment_hit_max_tokens.values())
        run_log = {
            "backend": "vllm",
            "device": str(self.decoder_device or self.mx.default_device()),
            "decoder_device": str(self.decoder_device or self.mx.default_device()),
            "full_model_path": str(self.full_model_path),
            "decoder_path": str(self.decoder_path),
            "streaming": True,
            "vllm_token_streaming": True,
            "decoder_streaming": True,
            "decoder_after_token_stream": False,
            "speech_decoder_skipped": speech_decoder_skipped,
            "text_to_tts_interleaved": text_to_tts_interleaved,
            "tts_cfg_enabled": tts_cfg_enabled,
            "tts_cfg_prime_first_segment": cfg_prime_first_segment,
            "tts_concurrent_segments": concurrent_tts_chunks,
            "tts_interleaved_tail_batched": interleaved_tail_batched,
            "tts_interleaved_all_ready_batched": interleaved_all_ready_batched,
            "tts_interleaved_initial_ready_batched": (
                interleaved_initial_ready_batched
            ),
            "decoder_chunk_frames": decoder_chunk_frames,
            "decoder_steady_chunk_frames": decoder_steady_chunk_frames,
            "decoded_chunk_count": decoded_chunk_count,
            "chunk_wav_paths": [],
            "playback_transport": (
                "sounddevice_raw_output_stream" if player is not None else None
            ),
            "playback_start_gated": playback_start_gate is not None,
            "playback_gate_released": (
                playback_start_gate.released
                if playback_start_gate is not None
                else None
            ),
            "playback_gate_cancelled": (
                playback_start_gate.cancelled
                if playback_start_gate is not None
                else None
            ),
            "playback_prebuffer_seconds": (
                playback_prebuffer_seconds if player is not None else None
            ),
            "prompt_tokens": 0,
            "heavy_tts_artifacts_retained": retain_heavy_tts_artifacts,
            "generated_token_id_count": generated_token_id_count,
            "generated_codec_frame_count": generated_codec_frame_count,
            "generated_token_ids": all_token_ids,
            "generated_codec_frames": all_codec_frames,
            "reached_end_token": reached_end_token,
            "hit_max_tokens": hit_max_tokens,
            "waveform_shape": [sample_count],
            "audio_duration_seconds": round(audio_duration_seconds, 3),
            "audio_realtime_ratio": (
                round(stream_realtime_ratio, 3)
                if stream_realtime_ratio is not None
                else None
            ),
            "codec_frames_per_second": (
                round(codec_frames_per_second, 3)
                if codec_frames_per_second is not None
                else None
            ),
            "sample_rate": self.decoder_config.sample_rate,
            "hop_length": self.decoder_config.hop_length,
            "finite": finite,
            "peak_abs": peak_abs,
            "wav_path": str(wav_path),
            "live_milestone_path": str(live_milestone_path),
            "first_audio_ready_seconds": first_audio_ready_seconds,
            "first_playback_started_seconds": first_playback_started_seconds,
            "first_audio_ready_at": first_audio_ready_at,
            "first_playback_started_at": first_playback_started_at,
            "first_playback_estimated_audible_at": (
                first_playback_estimated_audible_at
            ),
            "first_playback_after_audio_seconds": (
                round(first_playback_after_audio_seconds, 3)
                if first_playback_after_audio_seconds is not None
                else None
            ),
            "playback_diagnostics": playback_diagnostics,
            "playback_written_audio_seconds": (
                round(
                    int(playback_diagnostics.get("bytes_written", 0))
                    / (self.decoder_config.sample_rate * 2),
                    3,
                )
                if playback_diagnostics is not None
                else None
            ),
            "stream_event_count": stream_event_count,
            "first_token_event_seconds": first_token_event_seconds,
            "last_token_event_seconds": last_token_event_seconds,
            "first_tts_chunk_ready_seconds": first_tts_chunk_ready_seconds,
            "first_codec_frame_seconds": first_codec_frame_seconds,
            "last_codec_frame_seconds": last_codec_frame_seconds,
            "first_codec_frame_wall_seconds": first_codec_frame_wall_seconds,
            "last_codec_frame_wall_seconds": last_codec_frame_wall_seconds,
            "first_decoder_push_started_seconds": (first_decoder_push_started_seconds),
            "first_decoder_push_finished_seconds": (
                first_decoder_push_finished_seconds
            ),
            "first_decoder_push_frame_count": first_decoder_push_frame_count,
            "first_decoder_wait_after_codec_seconds": (
                round(first_decoder_wait_after_codec_seconds, 3)
                if first_decoder_wait_after_codec_seconds is not None
                else None
            ),
            "stream_finished_seconds": stream_finished_seconds,
            "playback_close_seconds": playback_close_seconds,
            "pcm_pack_seconds": round(pcm_pack_seconds, 6),
            "pcm_pack_fast_path_count": pcm_pack_fast_path_count,
            "pcm_pack_fallback_count": pcm_pack_fallback_count,
            "player_enqueue_seconds": round(player_enqueue_seconds, 6),
            "wav_write_seconds": round(wav_write_seconds, 6),
            "decoder_push_seconds": round(decoder_push_seconds, 6),
            "decoder_flush_seconds": round(decoder_flush_seconds, 6),
            "decoder_reset_seconds": round(decoder_reset_seconds, 6),
            "mlx_clear_cache_seconds": round(mlx_clear_cache_seconds, 6),
            "mlx_clear_cache_count": mlx_clear_cache_count,
            "mlx_memory_start": mlx_memory_start,
            "mlx_memory_after_stream": mlx_memory_after_stream,
            "mlx_memory_after_clear": self._mlx_memory_snapshot(),
            "tts_target_segments": effective_tts_target_segments,
            "tts_requested_target_segments": tts_target_segments,
            "tts_observed_segments": len(segment_codec_frame_counts),
            "tts_sentence_chunk_size": DEFAULT_VLLM_TTS_SENTENCES_PER_CHUNK,
            "tts_cfg_partition_cost": "chars",
            "tts_cfg_atom_max_chars": DEFAULT_VLLM_CFG_TTS_ATOM_MAX_CHARS,
            "tts_cfg_min_tail_chars": DEFAULT_VLLM_CFG_TTS_MIN_TAIL_CHARS,
            "tts_max_chars_per_chunk": DEFAULT_VLLM_TTS_MAX_CHARS_PER_CHUNK,
            "tts_stream_min_chars_per_chunk": (
                DEFAULT_VLLM_TTS_STREAM_MIN_CHARS_PER_CHUNK
            ),
            "tts_segment_texts": {
                str(index): text for index, text in sorted(segment_texts.items())
            },
            "tts_segment_codec_frame_counts": {
                str(index): count
                for index, count in sorted(segment_codec_frame_counts.items())
            },
            "tts_segment_token_counts": {
                str(index): count
                for index, count in sorted(segment_token_counts.items())
            },
            "tts_segment_ready_seconds": {
                str(index): seconds
                for index, seconds in sorted(segment_ready_seconds.items())
            },
            "tts_segment_max_tokens": {
                str(index): count for index, count in sorted(segment_max_tokens.items())
            },
            "tts_segment_wall_seconds": {
                str(index): seconds
                for index, seconds in sorted(segment_wall_seconds.items())
            },
            "tts_segment_mlx_memory_after_clear": {
                str(index): snapshot
                for index, snapshot in sorted(segment_mlx_memory_after_clear.items())
            },
            "tts_segment_finished": {
                str(index): finished
                for index, finished in sorted(segment_finished.items())
            },
            "tts_segment_hit_max_tokens": {
                str(index): hit for index, hit in sorted(segment_hit_max_tokens.items())
            },
        }
        run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
        return SpeechOutputSmokeResult(
            backend="vllm",
            device=str(self.decoder_device or self.mx.default_device()),
            prompt_tokens=0,
            generated_token_ids=tuple(all_token_ids),
            generated_codec_frames=tuple(all_codec_frames),
            reached_end_token=reached_end_token,
            hit_max_tokens=hit_max_tokens,
            waveform_shape=(sample_count,),
            sample_rate=self.decoder_config.sample_rate,
            hop_length=self.decoder_config.hop_length,
            finite=finite,
            peak_abs=peak_abs,
            wav_path=wav_path,
            run_log_path=run_log_path,
            streaming=True,
            segments=tuple(
                segment_text for _index, segment_text in sorted(segment_texts.items())
            ),
            chunk_wav_paths=(),
            first_audio_ready_seconds=first_audio_ready_seconds,
            first_playback_started_seconds=first_playback_started_seconds,
            first_audio_ready_at=first_audio_ready_at,
            first_playback_started_at=first_playback_started_at,
            first_playback_estimated_audible_at=(first_playback_estimated_audible_at),
        )
