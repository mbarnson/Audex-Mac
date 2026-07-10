"""Persistent vLLM runtime skeleton for Audex speech-to-speech."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .audio_contract import (
    NVIDIA_TTS_CFG_SCALE,
    NVIDIA_TTS_CFG_TEMPERATURE,
    NVIDIA_TTS_CFG_TOP_K,
    NVIDIA_TTS_CFG_TOP_P,
    NVIDIA_TTS_TEMPERATURE,
    NVIDIA_TTS_TOP_K,
    NVIDIA_TTS_TOP_P,
    build_codec_token_map,
)
from .conversations import DEFAULT_DEMO_CONTEXT_TOKENS
from .patches.runtime import apply_audex_runtime_patches
from .vllm_cfg import (
    AudexVllmCfgConfig,
    configure_audex_vllm_cfg,
    require_audex_vllm_cfg_ready,
)
from .vllm_sts_requests import (
    VllmGenerationRequest,
    VllmTtsSamplingConfig,
    build_asr_projected_embeddings_request,
    build_asr_request,
    build_audio_history_prime_request,
    build_audio_messages_response_request,
    build_text_messages_response_request,
    build_text_response_request,
    build_tts_cfg_requests,
    build_tts_request,
)

STOP_TEXTS = ("<|im_end|>", "<|end_of_text|>", "<|eot_id|>")
DEFAULT_GPU_MEMORY_UTILIZATION = 0.60
GPU_MEMORY_UTILIZATION_ENV = "AUDEX_VLLM_GPU_MEMORY_UTILIZATION"
CFG_WIRING_ENV = "AUDEX_VLLM_ENABLE_CFG_WIRING"
DIRECT_AUDIO_TRIM_PADDED_EMBEDDINGS_ENV = (
    "AUDEX_VLLM_DIRECT_AUDIO_TRIM_PADDED_EMBEDDINGS"
)


@dataclass(frozen=True, slots=True)
class VllmRequestResult:
    text: str
    token_ids: tuple[int, ...]
    elapsed_seconds: float
    finish_reason: str | None
    request_debug_name: str


@dataclass(frozen=True, slots=True)
class VllmSpeechCodecResult:
    generated_token_ids: tuple[int, ...]
    generated_codec_frames: tuple[int, ...]
    reached_end_token: bool


@dataclass(frozen=True, slots=True)
class VllmStreamDelta:
    text: str
    token_ids: tuple[int, ...]
    new_token_ids: tuple[int, ...]
    elapsed_seconds: float
    finished: bool
    finish_reason: str | None
    request_debug_name: str
    request_id: str


@dataclass(frozen=True, slots=True)
class VllmTtsCodecStreamEvent:
    generated_token_ids: tuple[int, ...]
    new_codec_frames: tuple[int, ...]
    reached_end_token: bool
    finished: bool
    elapsed_seconds: float
    segment_index: int = 0
    segment_finished: bool = False


@dataclass(frozen=True, slots=True)
class AudexVllmRuntimeStats:
    model_load_seconds: float
    engine_class: str
    model_path: Path
    cfg_enabled: bool
    cfg_script_dir: Path | None
    cfg_scheduler_reserve_full_isl: bool | None
    nonpaged_kv_capacity_seqs: int | None
    max_model_len: int


class AudexVllmRuntime:
    """Own one vLLM engine and execute NVIDIA-shaped Audex STS requests."""

    def __init__(
        self,
        *,
        model_path: Path,
        tokenizer: Any,
        engine: Any,
        sampling_params_cls: type[Any],
        model_load_seconds: float,
        max_model_len: int = DEFAULT_DEMO_CONTEXT_TOKENS,
        cfg_config: AudexVllmCfgConfig | None = None,
        tts_sampling_config: VllmTtsSamplingConfig | None = None,
    ) -> None:
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.engine = engine
        self.sampling_params_cls = sampling_params_cls
        self.model_load_seconds = model_load_seconds
        self.max_model_len = max_model_len
        self.cfg_config = cfg_config
        self.tts_sampling_config = tts_sampling_config
        self.token_map = build_codec_token_map(tokenizer.get_vocab())

    @classmethod
    def from_model_path(
        cls,
        model_path: Path,
        *,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float | None = DEFAULT_GPU_MEMORY_UTILIZATION,
        tts_sampling_config: VllmTtsSamplingConfig | None = None,
    ) -> AudexVllmRuntime:
        """Load a persistent vLLM engine for the full Audex checkpoint."""

        apply_audex_runtime_patches()
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
        template_path = model_path / "chat_template.jinja"
        if template_path.is_file():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")

        engine_kwargs = _base_engine_kwargs(
            model_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        cfg_config = configure_audex_vllm_cfg(
            engine_kwargs,
            model_path,
            cfg_scale=(
                tts_sampling_config.cfg_scale
                if tts_sampling_config is not None
                else _runtime_cfg_scale()
            ),
        )
        require_audex_vllm_cfg_ready(cfg_config)

        started = time.time()
        engine = LLM(**engine_kwargs)
        model_load_seconds = round(time.time() - started, 3)
        return cls(
            model_path=model_path,
            tokenizer=tokenizer,
            engine=engine,
            sampling_params_cls=SamplingParams,
            model_load_seconds=model_load_seconds,
            max_model_len=int(engine_kwargs["max_model_len"]),
            cfg_config=cfg_config,
            tts_sampling_config=tts_sampling_config,
        )

    @property
    def stats(self) -> AudexVllmRuntimeStats:
        return AudexVllmRuntimeStats(
            model_load_seconds=self.model_load_seconds,
            engine_class=f"{type(self.engine).__module__}.{type(self.engine).__name__}",
            model_path=self.model_path,
            cfg_enabled=bool(self.cfg_config and self.cfg_config.enabled),
            cfg_script_dir=(
                self.cfg_config.script_dir if self.cfg_config is not None else None
            ),
            cfg_scheduler_reserve_full_isl=(
                self.cfg_config.scheduler_reserve_full_isl
                if self.cfg_config is not None
                else None
            ),
            nonpaged_kv_capacity_seqs=_effective_nonpaged_kv_capacity_seqs(),
            max_model_len=self.max_model_len,
        )

    def transcribe_audio(
        self,
        audio_samples: Any,
        *,
        sample_rate: int = 16000,
    ) -> VllmRequestResult:
        request = build_asr_request(
            self.tokenizer,
            audio_samples,
            sample_rate=sample_rate,
        )
        result = self.generate_one(request)
        return _replace_text(result, clean_transcription(result.text))

    def transcribe_projected_audio(
        self,
        projected_embeddings: Any,
        *,
        num_embeddings: int | None = None,
    ) -> VllmRequestResult:
        request = build_asr_projected_embeddings_request(
            self.tokenizer,
            projected_embeddings,
            num_embeddings=num_embeddings,
        )
        result = self.generate_one(request)
        return _replace_text(result, clean_transcription(result.text))

    def generate_text_response(
        self,
        transcript: str,
        *,
        enable_reasoning: bool = False,
        max_tokens: int | None = None,
    ) -> VllmRequestResult:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        request = build_text_response_request(
            self.tokenizer,
            transcript,
            enable_reasoning=enable_reasoning,
            **kwargs,
        )
        result = self.generate_one(request)
        return _replace_text(result, extract_spoken_answer(result.text))

    def generate_text_response_from_messages(
        self,
        messages: list[dict[str, str]],
        *,
        enable_reasoning: bool = False,
        max_tokens: int | None = None,
        conversation_state_key: str | None = None,
        conversation_state_boundary: str | None = None,
        conversation_state_prefix_token_count: int | None = None,
        conversation_state_prefix_token_hash: str | None = None,
    ) -> VllmRequestResult:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        request = build_text_messages_response_request(
            self.tokenizer,
            messages,
            enable_reasoning=enable_reasoning,
            conversation_state_key=conversation_state_key,
            conversation_state_boundary=conversation_state_boundary,
            conversation_state_prefix_token_count=conversation_state_prefix_token_count,
            conversation_state_prefix_token_hash=conversation_state_prefix_token_hash,
            **kwargs,
        )
        result = self.generate_one(request)
        return _replace_text(result, extract_spoken_answer(result.text))

    def generate_tts_cfg_pair(
        self,
        text: str,
        *,
        pair_id: str,
        max_tokens: int | None = None,
    ) -> tuple[VllmRequestResult, VllmRequestResult]:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        cond, uncond = build_tts_cfg_requests(
            self.tokenizer,
            text,
            speechgen_end_id=self.token_map.speechgen_end,
            eos_token_id=self.tokenizer.eos_token_id,
            pair_id=pair_id,
            codec_min_id=min(self.token_map.speech_codec),
            codec_max_id=max(self.token_map.speech_codec),
            **_tts_sampling_kwargs(self.tts_sampling_config, cfg_enabled=True),
            **kwargs,
        )
        return self.generate_many((cond, uncond))

    def generate_tts(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
    ) -> VllmRequestResult:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        request = build_tts_request(
            self.tokenizer,
            text,
            speechgen_end_id=self.token_map.speechgen_end,
            eos_token_id=self.tokenizer.eos_token_id,
            codec_min_id=min(self.token_map.speech_codec),
            codec_max_id=max(self.token_map.speech_codec),
            skip_paged_logits_eval=True,
            **_tts_sampling_kwargs(self.tts_sampling_config, cfg_enabled=False),
            **kwargs,
        )
        return self.generate_one(request)

    def extract_tts_codec_frames(
        self,
        result: VllmRequestResult,
    ) -> VllmSpeechCodecResult:
        return extract_tts_codec_frames(result.token_ids, self.token_map)

    def generate_one(self, request: VllmGenerationRequest) -> VllmRequestResult:
        return self.generate_many((request,))[0]

    def generate_many(
        self,
        requests: tuple[VllmGenerationRequest, ...],
    ) -> tuple[VllmRequestResult, ...]:
        prompts = [request.prompt for request in requests]
        sampling_params = [
            self.sampling_params_cls(**_sampling_kwargs(request.sampling))
            for request in requests
        ]
        started = time.time()
        outputs = self.engine.generate(prompts, sampling_params)
        elapsed_seconds = round(time.time() - started, 3)
        results: list[VllmRequestResult] = []
        for request, output in zip(requests, outputs, strict=True):
            completion = output.outputs[0]
            results.append(
                VllmRequestResult(
                    text=completion.text or "",
                    token_ids=tuple(int(token_id) for token_id in completion.token_ids),
                    elapsed_seconds=elapsed_seconds,
                    finish_reason=getattr(completion, "finish_reason", None),
                    request_debug_name=request.debug_name,
                )
            )
        return tuple(results)


class AudexAsyncVllmRuntime:
    """Async vLLM engine wrapper for token-streamed Audex requests."""

    def __init__(
        self,
        *,
        model_path: Path,
        tokenizer: Any,
        engine: Any,
        sampling_params_cls: type[Any],
        model_load_seconds: float,
        max_model_len: int = DEFAULT_DEMO_CONTEXT_TOKENS,
        cfg_config: AudexVllmCfgConfig | None = None,
        tts_sampling_config: VllmTtsSamplingConfig | None = None,
    ) -> None:
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.engine = engine
        self.sampling_params_cls = sampling_params_cls
        self.model_load_seconds = model_load_seconds
        self.max_model_len = max_model_len
        self.cfg_config = cfg_config
        self.tts_sampling_config = tts_sampling_config
        self.token_map = build_codec_token_map(tokenizer.get_vocab())

    @property
    def stats(self) -> AudexVllmRuntimeStats:
        return AudexVllmRuntimeStats(
            model_load_seconds=self.model_load_seconds,
            engine_class=f"{type(self.engine).__module__}.{type(self.engine).__name__}",
            model_path=self.model_path,
            cfg_enabled=bool(self.cfg_config and self.cfg_config.enabled),
            cfg_script_dir=(
                self.cfg_config.script_dir if self.cfg_config is not None else None
            ),
            cfg_scheduler_reserve_full_isl=(
                self.cfg_config.scheduler_reserve_full_isl
                if self.cfg_config is not None
                else None
            ),
            nonpaged_kv_capacity_seqs=_effective_nonpaged_kv_capacity_seqs(),
            max_model_len=self.max_model_len,
        )

    @classmethod
    def from_model_path(
        cls,
        model_path: Path,
        *,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float | None = DEFAULT_GPU_MEMORY_UTILIZATION,
        tts_sampling_config: VllmTtsSamplingConfig | None = None,
    ) -> AudexAsyncVllmRuntime:
        """Load an async vLLM engine for streamed Audex generation."""

        apply_audex_runtime_patches()
        from transformers import AutoTokenizer
        from vllm import AsyncLLMEngine, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
        template_path = model_path / "chat_template.jinja"
        if template_path.is_file():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")

        engine_kwargs = _base_engine_kwargs(
            model_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        cfg_config = configure_audex_vllm_cfg(
            engine_kwargs,
            model_path,
            cfg_scale=(
                tts_sampling_config.cfg_scale
                if tts_sampling_config is not None
                else _runtime_cfg_scale()
            ),
        )
        require_audex_vllm_cfg_ready(cfg_config)

        started = time.time()
        engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))
        model_load_seconds = round(time.time() - started, 3)
        return cls(
            model_path=model_path,
            tokenizer=tokenizer,
            engine=engine,
            sampling_params_cls=SamplingParams,
            model_load_seconds=model_load_seconds,
            max_model_len=int(engine_kwargs["max_model_len"]),
            cfg_config=cfg_config,
            tts_sampling_config=tts_sampling_config,
        )

    def build_tts_cfg_pair(
        self,
        text: str,
        *,
        pair_id: str,
        max_tokens: int | None = None,
    ) -> tuple[VllmGenerationRequest, VllmGenerationRequest]:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return build_tts_cfg_requests(
            self.tokenizer,
            text,
            speechgen_end_id=self.token_map.speechgen_end,
            eos_token_id=self.tokenizer.eos_token_id,
            pair_id=pair_id,
            codec_min_id=min(self.token_map.speech_codec),
            codec_max_id=max(self.token_map.speech_codec),
            **_tts_sampling_kwargs(self.tts_sampling_config, cfg_enabled=True),
            **kwargs,
        )

    def build_tts_request(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
    ) -> VllmGenerationRequest:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return build_tts_request(
            self.tokenizer,
            text,
            speechgen_end_id=self.token_map.speechgen_end,
            eos_token_id=self.tokenizer.eos_token_id,
            codec_min_id=min(self.token_map.speech_codec),
            codec_max_id=max(self.token_map.speech_codec),
            skip_paged_logits_eval=True,
            **_tts_sampling_kwargs(self.tts_sampling_config, cfg_enabled=False),
            **kwargs,
        )

    async def stream_many(
        self,
        requests: tuple[VllmGenerationRequest, ...],
        *,
        include_cumulative_token_ids: bool = True,
    ) -> AsyncIterator[VllmStreamDelta]:
        queue: asyncio.Queue[VllmStreamDelta | BaseException | None] = asyncio.Queue()
        started = time.time()

        async def worker(request: VllmGenerationRequest) -> None:
            last_len = 0
            cumulative_token_ids: list[int] = []
            delta_output = _is_delta_output_kind(request.sampling.output_kind)
            try:
                sampling_params = self.sampling_params_cls(
                    **_sampling_kwargs(request.sampling)
                )
                request_id = _request_id(request)
                async for output in self.engine.generate(
                    request.prompt,
                    sampling_params,
                    request_id,
                ):
                    completion = output.outputs[0]
                    raw_token_ids = tuple(
                        int(token_id) for token_id in completion.token_ids
                    )
                    if delta_output:
                        new_token_ids = raw_token_ids
                        if include_cumulative_token_ids:
                            cumulative_token_ids.extend(new_token_ids)
                            token_ids = tuple(cumulative_token_ids)
                        else:
                            token_ids = ()
                    else:
                        token_ids = raw_token_ids
                        new_token_ids = token_ids[last_len:]
                    delta = VllmStreamDelta(
                        text=completion.text or "",
                        token_ids=token_ids,
                        new_token_ids=new_token_ids,
                        elapsed_seconds=round(time.time() - started, 3),
                        finished=bool(getattr(output, "finished", False)),
                        finish_reason=getattr(completion, "finish_reason", None),
                        request_debug_name=request.debug_name,
                        request_id=request_id,
                    )
                    if include_cumulative_token_ids or not delta_output:
                        last_len = len(token_ids)
                    await queue.put(delta)
            except BaseException as exc:
                await queue.put(exc)
            finally:
                await queue.put(None)

        tasks = [asyncio.create_task(worker(request)) for request in requests]
        finished_workers = 0
        try:
            while finished_workers < len(tasks):
                item = await queue.get()
                if item is None:
                    finished_workers += 1
                    continue
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def generate_many_final(
        self,
        requests: tuple[VllmGenerationRequest, ...],
    ) -> tuple[VllmRequestResult, ...]:
        async def worker(request: VllmGenerationRequest) -> VllmRequestResult:
            sampling_params = self.sampling_params_cls(
                **_sampling_kwargs(request.sampling)
            )
            request_id = _request_id(request)
            started = time.time()
            latest_output = None
            async for output in self.engine.generate(
                request.prompt,
                sampling_params,
                request_id,
            ):
                latest_output = output
            if latest_output is None:
                raise RuntimeError(
                    f"Async vLLM request produced no output: {request.debug_name}"
                )
            completion = latest_output.outputs[0]
            return VllmRequestResult(
                text=completion.text or "",
                token_ids=tuple(int(token_id) for token_id in completion.token_ids),
                elapsed_seconds=round(time.time() - started, 3),
                finish_reason=getattr(completion, "finish_reason", None),
                request_debug_name=request.debug_name,
            )

        return tuple(await asyncio.gather(*(worker(request) for request in requests)))

    async def generate_one_final(
        self,
        request: VllmGenerationRequest,
    ) -> VllmRequestResult:
        return (await self.generate_many_final((request,)))[0]

    async def reset_prefix_cache(self) -> bool:
        """Reset vLLM scheduler-side KV and prefix-cache bookkeeping."""

        return bool(await self.engine.reset_prefix_cache())

    async def transcribe_audio(
        self,
        audio_samples: Any,
        *,
        sample_rate: int = 16000,
    ) -> VllmRequestResult:
        request = build_asr_request(
            self.tokenizer,
            audio_samples,
            sample_rate=sample_rate,
        )
        result = await self.generate_one_final(request)
        return _replace_text(result, clean_transcription(result.text))

    async def transcribe_projected_audio(
        self,
        projected_embeddings: Any,
        *,
        num_embeddings: int | None = None,
    ) -> VllmRequestResult:
        request = build_asr_projected_embeddings_request(
            self.tokenizer,
            projected_embeddings,
            num_embeddings=num_embeddings,
        )
        result = await self.generate_one_final(request)
        return _replace_text(result, clean_transcription(result.text))

    async def generate_text_response_from_messages(
        self,
        messages: list[dict[str, str]],
        *,
        enable_reasoning: bool = False,
        max_tokens: int | None = None,
        conversation_state_key: str | None = None,
        conversation_state_boundary: str | None = None,
        conversation_state_prefix_token_count: int | None = None,
        conversation_state_prefix_token_hash: str | None = None,
    ) -> VllmRequestResult:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        request = build_text_messages_response_request(
            self.tokenizer,
            messages,
            enable_reasoning=enable_reasoning,
            conversation_state_key=conversation_state_key,
            conversation_state_boundary=conversation_state_boundary,
            conversation_state_prefix_token_count=conversation_state_prefix_token_count,
            conversation_state_prefix_token_hash=conversation_state_prefix_token_hash,
            **kwargs,
        )
        result = await self.generate_one_final(request)
        return _replace_text(result, extract_spoken_answer(result.text))

    async def stream_text_response_from_messages(
        self,
        messages: list[dict[str, str]],
        *,
        enable_reasoning: bool = False,
        max_tokens: int | None = None,
        conversation_state_key: str | None = None,
        conversation_state_boundary: str | None = None,
        conversation_state_prefix_token_count: int | None = None,
        conversation_state_prefix_token_hash: str | None = None,
    ) -> AsyncIterator[VllmStreamDelta]:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        request = build_text_messages_response_request(
            self.tokenizer,
            messages,
            enable_reasoning=enable_reasoning,
            conversation_state_key=conversation_state_key,
            conversation_state_boundary=conversation_state_boundary,
            conversation_state_prefix_token_count=conversation_state_prefix_token_count,
            conversation_state_prefix_token_hash=conversation_state_prefix_token_hash,
            **kwargs,
        )
        request = replace(
            request,
            sampling=replace(request.sampling, output_kind="DELTA"),
        )
        cumulative_text = ""
        async for delta in self.stream_many((request,)):
            cumulative_text += delta.text
            yield replace(delta, text=extract_spoken_answer(cumulative_text))

    async def stream_audio_response_from_messages(
        self,
        messages: list[dict[str, str]],
        audio_samples: Any,
        *,
        sample_rate: int = 16000,
        enable_reasoning: bool = False,
        max_tokens: int | None = None,
        trim_padded_audio_embeddings: bool | None = None,
        conversation_state_key: str | None = None,
        conversation_state_prefix_token_count: int | None = None,
        conversation_state_prefix_token_hash: str | None = None,
    ) -> AsyncIterator[VllmStreamDelta]:
        kwargs: dict[str, Any] = {}
        if trim_padded_audio_embeddings is None:
            value = os.environ.get(DIRECT_AUDIO_TRIM_PADDED_EMBEDDINGS_ENV)
            trim_padded_audio_embeddings = (
                value is None
                or value.strip().lower()
                not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
            )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        request = build_audio_messages_response_request(
            self.tokenizer,
            messages,
            audio_samples,
            sample_rate=sample_rate,
            enable_reasoning=enable_reasoning,
            trim_padded_audio_embeddings=trim_padded_audio_embeddings,
            conversation_state_key=conversation_state_key,
            conversation_state_prefix_token_count=(
                conversation_state_prefix_token_count
            ),
            conversation_state_prefix_token_hash=conversation_state_prefix_token_hash,
            **kwargs,
        )
        request = replace(
            request,
            sampling=replace(request.sampling, output_kind="DELTA"),
        )
        cumulative_text = ""
        async for delta in self.stream_many((request,)):
            cumulative_text += delta.text
            yield replace(delta, text=extract_spoken_answer(cumulative_text))

    async def prime_audio_response_history(
        self,
        messages: list[dict[str, str]],
        *,
        conversation_state_key: str,
        conversation_state_prefix_token_count: int,
        conversation_state_prefix_token_hash: str,
    ) -> VllmRequestResult:
        request = build_audio_history_prime_request(
            self.tokenizer,
            messages,
            conversation_state_key=conversation_state_key,
            conversation_state_prefix_token_count=conversation_state_prefix_token_count,
            conversation_state_prefix_token_hash=conversation_state_prefix_token_hash,
        )
        return await self.generate_one_final(request)

    async def generate_tts_cfg_pair(
        self,
        text: str,
        *,
        pair_id: str,
        max_tokens: int | None = None,
    ) -> tuple[VllmRequestResult, VllmRequestResult]:
        return await self.generate_many_final(
            self.build_tts_cfg_pair(text, pair_id=pair_id, max_tokens=max_tokens)
        )

    async def generate_tts(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
    ) -> VllmRequestResult:
        return await self.generate_one_final(
            self.build_tts_request(text, max_tokens=max_tokens)
        )

    async def stream_tts_codec_frames(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
    ) -> AsyncIterator[VllmTtsCodecStreamEvent]:
        request = self.build_tts_request(text, max_tokens=max_tokens)
        request = replace(
            request,
            sampling=replace(request.sampling, output_kind="DELTA"),
        )
        sampling_params = self.sampling_params_cls(**_sampling_kwargs(request.sampling))
        request_id = _request_id(request)
        started = time.time()
        generated_token_ids: list[int] = []
        reached_end_token = False
        async for output in self.engine.generate(
            request.prompt,
            sampling_params,
            request_id,
        ):
            completion = output.outputs[0]
            raw_token_ids = completion.token_ids
            new_raw_token_ids = raw_token_ids
            new_codec_frames: list[int] = []
            for raw_token_id in new_raw_token_ids:
                token_id = int(raw_token_id)
                generated_token_ids.append(token_id)
                if token_id == self.token_map.speechgen_end:
                    reached_end_token = True
                    continue
                codec_frame = self.token_map.speech_codec.get(token_id)
                if codec_frame is None:
                    continue
                new_codec_frames.append(codec_frame)

            finished = bool(getattr(output, "finished", False)) or reached_end_token
            # Avoid copying the full generated-token history on every streamed
            # audio-token event. The caller accumulates codec-frame deltas for
            # playback; the complete token list is only needed for final logs.
            event_token_ids = tuple(generated_token_ids) if finished else ()
            yield VllmTtsCodecStreamEvent(
                generated_token_ids=event_token_ids,
                new_codec_frames=tuple(new_codec_frames),
                reached_end_token=reached_end_token,
                finished=finished,
                elapsed_seconds=round(time.time() - started, 3),
                segment_finished=finished,
            )
            if finished:
                break

    async def stream_tts_segmented_codec_frames(
        self,
        segments: tuple[str, ...],
        *,
        max_tokens_per_segment: tuple[int | None, ...],
    ) -> AsyncIterator[VllmTtsCodecStreamEvent]:
        """Stream ordered no-CFG TTS frames while keeping chunks batched."""

        if len(segments) != len(max_tokens_per_segment):
            raise ValueError("segments and max_tokens_per_segment must match")
        if len(segments) <= 1:
            segment = segments[0] if segments else ""
            max_tokens = max_tokens_per_segment[0] if max_tokens_per_segment else None
            async for event in self.stream_tts_codec_frames(
                segment,
                max_tokens=max_tokens,
            ):
                yield event
            return

        requests: list[VllmGenerationRequest] = []
        debug_to_index: dict[str, int] = {}
        for index, (segment, segment_max_tokens) in enumerate(
            zip(segments, max_tokens_per_segment, strict=True)
        ):
            request = self.build_tts_request(segment, max_tokens=segment_max_tokens)
            debug_name = f"tts-segment-{index}"
            requests.append(
                replace(
                    request,
                    debug_name=debug_name,
                    request_id_suffix=f"seg-{index}",
                    sampling=replace(request.sampling, output_kind="DELTA"),
                )
            )
            debug_to_index[debug_name] = index

        buffers: list[list[int]] = [[] for _ in segments]
        token_ids_by_segment: list[list[int]] = [[] for _ in segments]
        reached_by_segment = [False for _ in segments]
        finished_by_segment = [False for _ in segments]
        closed_by_segment = [False for _ in segments]
        active_segment = 0

        async for delta in self.stream_many(
            tuple(requests),
            include_cumulative_token_ids=False,
        ):
            segment_index = debug_to_index.get(delta.request_debug_name)
            if segment_index is None:
                continue

            new_codec_frames: list[int] = []
            for token_id in delta.new_token_ids:
                token_ids_by_segment[segment_index].append(token_id)
                if token_id == self.token_map.speechgen_end:
                    reached_by_segment[segment_index] = True
                    continue
                codec_frame = self.token_map.speech_codec.get(token_id)
                if codec_frame is not None:
                    new_codec_frames.append(codec_frame)
            finished_by_segment[segment_index] = delta.finished
            buffers[segment_index].extend(new_codec_frames)

            while active_segment < len(segments):
                if buffers[active_segment]:
                    new_frames = tuple(buffers[active_segment])
                    buffers[active_segment].clear()
                    segment_finished = finished_by_segment[active_segment]
                    yield VllmTtsCodecStreamEvent(
                        generated_token_ids=(
                            tuple(token_ids_by_segment[active_segment])
                            if segment_finished
                            else ()
                        ),
                        new_codec_frames=new_frames,
                        reached_end_token=reached_by_segment[active_segment],
                        finished=all(finished_by_segment),
                        elapsed_seconds=delta.elapsed_seconds,
                        segment_index=active_segment,
                        segment_finished=segment_finished,
                    )
                    continue
                if finished_by_segment[active_segment]:
                    boundary_segment = active_segment
                    closed_by_segment[boundary_segment] = True
                    active_segment += 1
                    yield VllmTtsCodecStreamEvent(
                        generated_token_ids=tuple(
                            token_ids_by_segment[boundary_segment]
                        ),
                        new_codec_frames=(),
                        reached_end_token=reached_by_segment[boundary_segment],
                        finished=all(finished_by_segment),
                        elapsed_seconds=delta.elapsed_seconds,
                        segment_index=boundary_segment,
                        segment_finished=True,
                    )
                    continue
                break

        for segment_index in range(active_segment, len(segments)):
            if buffers[segment_index]:
                segment_finished = finished_by_segment[segment_index]
                yield VllmTtsCodecStreamEvent(
                    generated_token_ids=(
                        tuple(token_ids_by_segment[segment_index])
                        if segment_finished
                        else ()
                    ),
                    new_codec_frames=tuple(buffers[segment_index]),
                    reached_end_token=reached_by_segment[segment_index],
                    finished=all(finished_by_segment),
                    elapsed_seconds=0.0,
                    segment_index=segment_index,
                    segment_finished=segment_finished,
                )
            if (
                finished_by_segment[segment_index]
                and not closed_by_segment[segment_index]
            ):
                yield VllmTtsCodecStreamEvent(
                    generated_token_ids=tuple(token_ids_by_segment[segment_index]),
                    new_codec_frames=(),
                    reached_end_token=reached_by_segment[segment_index],
                    finished=all(finished_by_segment),
                    elapsed_seconds=0.0,
                    segment_index=segment_index,
                    segment_finished=True,
                )

    async def stream_tts_cfg_codec_frames(
        self,
        text: str,
        *,
        pair_id: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[VllmTtsCodecStreamEvent]:
        pair_id = pair_id or f"tts-{uuid.uuid4().hex}"
        cond, uncond = self.build_tts_cfg_pair(
            text,
            pair_id=pair_id,
            max_tokens=max_tokens,
        )
        cond = replace(
            cond,
            sampling=replace(cond.sampling, output_kind="DELTA"),
        )
        uncond = replace(
            uncond,
            sampling=replace(uncond.sampling, output_kind="DELTA"),
        )
        generated_token_ids: list[int] = []
        reached_end_token = False
        async for delta in self.stream_many(
            (cond, uncond),
            include_cumulative_token_ids=False,
        ):
            if delta.request_debug_name != "tts-cond":
                continue
            new_codec = extract_tts_codec_frames(delta.new_token_ids, self.token_map)
            generated_token_ids.extend(delta.new_token_ids)
            reached_end_token = reached_end_token or new_codec.reached_end_token
            finished = delta.finished or reached_end_token
            yield VllmTtsCodecStreamEvent(
                generated_token_ids=tuple(generated_token_ids) if finished else (),
                new_codec_frames=new_codec.generated_codec_frames,
                reached_end_token=reached_end_token,
                finished=finished,
                elapsed_seconds=delta.elapsed_seconds,
                segment_finished=finished,
            )
            if finished:
                break

    async def stream_tts_cfg_segmented_codec_frames(
        self,
        text: str,
        *,
        pair_id: str | None = None,
        max_tokens: int | None = None,
        target_segments: int = 16,
    ) -> AsyncIterator[VllmTtsCodecStreamEvent]:
        """Stream ordered TTS frames while keeping multiple CFG pairs active."""

        segments = split_tts_text_segments(text, target_segments=target_segments)
        segment_token_budgets = _segment_token_budgets(segments, max_tokens)
        async for event in self.stream_tts_cfg_segments_codec_frames(
            segments,
            pair_id=pair_id,
            max_tokens_per_segment=segment_token_budgets,
        ):
            yield event

    async def stream_tts_cfg_segments_codec_frames(
        self,
        segments: tuple[str, ...],
        *,
        pair_id: str | None = None,
        max_tokens_per_segment: tuple[int | None, ...],
        prime_first_segment: bool = False,
    ) -> AsyncIterator[VllmTtsCodecStreamEvent]:
        """Stream ordered TTS frames for explicit CFG text segments."""

        if len(segments) != len(max_tokens_per_segment):
            raise ValueError("segments and max_tokens_per_segment must match")
        if len(segments) <= 1:
            segment = segments[0] if segments else ""
            max_tokens = max_tokens_per_segment[0] if max_tokens_per_segment else None
            async for event in self.stream_tts_cfg_codec_frames(
                segment,
                pair_id=pair_id,
                max_tokens=max_tokens,
            ):
                yield event
            return

        base_pair_id = pair_id or f"tts-{uuid.uuid4().hex}"
        request_groups: list[tuple[VllmGenerationRequest, VllmGenerationRequest]] = []
        cond_debug_to_index: dict[str, int] = {}
        for index, (segment, segment_max_tokens) in enumerate(
            zip(segments, max_tokens_per_segment, strict=True)
        ):
            cond, uncond = self.build_tts_cfg_pair(
                segment,
                pair_id=f"{base_pair_id}-seg-{index}",
                max_tokens=segment_max_tokens,
            )
            cond_name = f"tts-segment-{index}-cond"
            uncond_name = f"tts-segment-{index}-uncond"
            cond_request = replace(
                cond,
                debug_name=cond_name,
                request_id_suffix=f"seg-{index}-cond",
                sampling=replace(cond.sampling, output_kind="DELTA"),
            )
            uncond_request = replace(
                uncond,
                debug_name=uncond_name,
                request_id_suffix=f"seg-{index}-uncond",
                sampling=replace(uncond.sampling, output_kind="DELTA"),
            )
            request_groups.append((cond_request, uncond_request))
            cond_debug_to_index[cond_name] = index

        requests = tuple(request for group in request_groups for request in group)
        if not prime_first_segment:
            delta_source = self.stream_many(
                requests,
                include_cumulative_token_ids=False,
            )
        else:
            delta_source = self._stream_many_after_first_delta(
                initial_requests=request_groups[0],
                deferred_requests=tuple(
                    request for group in request_groups[1:] for request in group
                ),
                trigger_debug_name="tts-segment-0-cond",
                include_cumulative_token_ids=False,
            )

        buffers: list[list[int]] = [[] for _ in segments]
        token_ids_by_segment: list[list[int]] = [[] for _ in segments]
        reached_by_segment = [False for _ in segments]
        finished_by_segment = [False for _ in segments]
        closed_by_segment = [False for _ in segments]
        active_segment = 0

        async for delta in delta_source:
            segment_index = cond_debug_to_index.get(delta.request_debug_name)
            if segment_index is None:
                continue

            new_codec_frames: list[int] = []
            for token_id in delta.new_token_ids:
                token_ids_by_segment[segment_index].append(token_id)
                if token_id == self.token_map.speechgen_end:
                    reached_by_segment[segment_index] = True
                    continue
                codec_frame = self.token_map.speech_codec.get(token_id)
                if codec_frame is not None:
                    new_codec_frames.append(codec_frame)
            finished_by_segment[segment_index] = (
                delta.finished or reached_by_segment[segment_index]
            )
            buffers[segment_index].extend(new_codec_frames)

            while active_segment < len(segments):
                if buffers[active_segment]:
                    new_frames = tuple(buffers[active_segment])
                    buffers[active_segment].clear()
                    segment_finished = finished_by_segment[active_segment]
                    if segment_finished:
                        closed_by_segment[active_segment] = True
                    yield VllmTtsCodecStreamEvent(
                        generated_token_ids=(
                            tuple(token_ids_by_segment[active_segment])
                            if segment_finished
                            else ()
                        ),
                        new_codec_frames=new_frames,
                        reached_end_token=reached_by_segment[active_segment],
                        finished=all(finished_by_segment),
                        elapsed_seconds=delta.elapsed_seconds,
                        segment_index=active_segment,
                        segment_finished=segment_finished,
                    )
                    continue
                if finished_by_segment[active_segment]:
                    should_emit_boundary = not closed_by_segment[active_segment]
                    closed_by_segment[active_segment] = True
                    boundary_segment = active_segment
                    active_segment += 1
                    if should_emit_boundary:
                        yield VllmTtsCodecStreamEvent(
                            generated_token_ids=tuple(
                                token_ids_by_segment[boundary_segment]
                            ),
                            new_codec_frames=(),
                            reached_end_token=reached_by_segment[boundary_segment],
                            finished=all(finished_by_segment),
                            elapsed_seconds=delta.elapsed_seconds,
                            segment_index=boundary_segment,
                            segment_finished=True,
                        )
                    continue
                break

        for segment_index in range(active_segment, len(segments)):
            if buffers[segment_index]:
                new_frames = tuple(buffers[segment_index])
                buffers[segment_index].clear()
                segment_finished = finished_by_segment[segment_index]
                if segment_finished:
                    closed_by_segment[segment_index] = True
                yield VllmTtsCodecStreamEvent(
                    generated_token_ids=(
                        tuple(token_ids_by_segment[segment_index])
                        if segment_finished
                        else ()
                    ),
                    new_codec_frames=new_frames,
                    reached_end_token=reached_by_segment[segment_index],
                    finished=all(finished_by_segment),
                    elapsed_seconds=0.0,
                    segment_index=segment_index,
                    segment_finished=segment_finished,
                )
            if (
                finished_by_segment[segment_index]
                and not closed_by_segment[segment_index]
            ):
                closed_by_segment[segment_index] = True
                yield VllmTtsCodecStreamEvent(
                    generated_token_ids=tuple(token_ids_by_segment[segment_index]),
                    new_codec_frames=(),
                    reached_end_token=reached_by_segment[segment_index],
                    finished=all(finished_by_segment),
                    elapsed_seconds=0.0,
                    segment_index=segment_index,
                    segment_finished=True,
                )

        final_segment_index = len(segments) - 1
        if not closed_by_segment[final_segment_index]:
            closed_by_segment[final_segment_index] = True
            yield VllmTtsCodecStreamEvent(
                generated_token_ids=tuple(token_ids_by_segment[final_segment_index]),
                new_codec_frames=(),
                reached_end_token=reached_by_segment[final_segment_index],
                finished=all(finished_by_segment),
                elapsed_seconds=0.0,
                segment_index=final_segment_index,
                segment_finished=True,
            )

    async def _stream_many_after_first_delta(
        self,
        *,
        initial_requests: tuple[VllmGenerationRequest, ...],
        deferred_requests: tuple[VllmGenerationRequest, ...],
        trigger_debug_name: str,
        include_cumulative_token_ids: bool,
    ) -> AsyncIterator[VllmStreamDelta]:
        queue: asyncio.Queue[VllmStreamDelta | BaseException | None] = asyncio.Queue()
        started = time.time()
        tasks: list[asyncio.Task[None]] = []
        deferred_started = False
        finished_workers = 0

        async def worker(request: VllmGenerationRequest) -> None:
            last_len = 0
            cumulative_token_ids: list[int] = []
            delta_output = _is_delta_output_kind(request.sampling.output_kind)
            try:
                sampling_params = self.sampling_params_cls(
                    **_sampling_kwargs(request.sampling)
                )
                request_id = _request_id(request)
                async for output in self.engine.generate(
                    request.prompt,
                    sampling_params,
                    request_id,
                ):
                    completion = output.outputs[0]
                    raw_token_ids = tuple(
                        int(token_id) for token_id in completion.token_ids
                    )
                    if delta_output:
                        new_token_ids = raw_token_ids
                        if include_cumulative_token_ids:
                            cumulative_token_ids.extend(new_token_ids)
                            token_ids = tuple(cumulative_token_ids)
                        else:
                            token_ids = ()
                    else:
                        token_ids = raw_token_ids
                        new_token_ids = token_ids[last_len:]
                    delta = VllmStreamDelta(
                        text=completion.text or "",
                        token_ids=token_ids,
                        new_token_ids=new_token_ids,
                        elapsed_seconds=round(time.time() - started, 3),
                        finished=bool(getattr(output, "finished", False)),
                        finish_reason=getattr(completion, "finish_reason", None),
                        request_debug_name=request.debug_name,
                        request_id=request_id,
                    )
                    if include_cumulative_token_ids or not delta_output:
                        last_len = len(token_ids)
                    await queue.put(delta)
            except BaseException as exc:
                await queue.put(exc)
            finally:
                await queue.put(None)

        def start_requests(requests: tuple[VllmGenerationRequest, ...]) -> None:
            for request in requests:
                tasks.append(asyncio.create_task(worker(request)))

        start_requests(initial_requests)
        try:
            while finished_workers < len(tasks):
                item = await queue.get()
                if item is None:
                    finished_workers += 1
                    continue
                if isinstance(item, BaseException):
                    raise item
                if (
                    not deferred_started
                    and deferred_requests
                    and item.request_debug_name == trigger_debug_name
                ):
                    deferred_started = True
                    start_requests(deferred_requests)
                yield item
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def _base_engine_kwargs(
    model_path: Path,
    *,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float | None,
) -> dict[str, Any]:
    max_model_len = _checkpoint_max_model_len(model_path)
    engine_kwargs: dict[str, Any] = {
        "model": str(model_path),
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "enable_prefix_caching": True,
        "max_model_len": max_model_len,
        "limit_mm_per_prompt": {"audio": 1},
    }
    effective_gpu_memory_utilization = _effective_gpu_memory_utilization(
        gpu_memory_utilization
    )
    if effective_gpu_memory_utilization is not None:
        engine_kwargs["gpu_memory_utilization"] = effective_gpu_memory_utilization
    return engine_kwargs


def _checkpoint_max_model_len(model_path: Path) -> int:
    """Clamp the demo context reservation to the checkpoint's declared limit."""

    config_path = model_path / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_DEMO_CONTEXT_TOKENS
    declared_limit = payload.get("max_position_embeddings")
    if not isinstance(declared_limit, int) or isinstance(declared_limit, bool):
        return DEFAULT_DEMO_CONTEXT_TOKENS
    if declared_limit <= 0:
        return DEFAULT_DEMO_CONTEXT_TOKENS
    return min(DEFAULT_DEMO_CONTEXT_TOKENS, declared_limit)


def _effective_gpu_memory_utilization(value: float | None) -> float | None:
    if value is None:
        return None
    raw_value = os.environ.get(GPU_MEMORY_UTILIZATION_ENV)
    if raw_value is None:
        return value
    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{GPU_MEMORY_UTILIZATION_ENV} must be a float between 0 and 1, "
            f"got {raw_value!r}"
        ) from exc
    if not 0.0 < parsed <= 1.0:
        raise ValueError(
            f"{GPU_MEMORY_UTILIZATION_ENV} must be a float between 0 and 1, "
            f"got {raw_value!r}"
        )
    return parsed


def _effective_nonpaged_kv_capacity_seqs() -> int | None:
    raw_value = os.environ.get("AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS")
    if raw_value is None or not raw_value.strip():
        return None
    try:
        parsed = int(raw_value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _runtime_cfg_scale() -> float:
    value = os.environ.get(CFG_WIRING_ENV)
    if value is None:
        return 0.0
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        return NVIDIA_TTS_CFG_SCALE
    return 0.0


def _tts_sampling_kwargs(
    config: VllmTtsSamplingConfig | None,
    *,
    cfg_enabled: bool,
) -> dict[str, Any]:
    if config is not None:
        kwargs: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "require_compact_window_decode": (config.require_compact_window_decode),
        }
        if cfg_enabled:
            kwargs["cfg_scale"] = config.cfg_scale
        if config.seed is not None:
            kwargs["seed"] = config.seed
        return kwargs
    kwargs: dict[str, Any] = {
        "temperature": (
            NVIDIA_TTS_CFG_TEMPERATURE if cfg_enabled else NVIDIA_TTS_TEMPERATURE
        ),
        "top_p": NVIDIA_TTS_CFG_TOP_P if cfg_enabled else NVIDIA_TTS_TOP_P,
        "top_k": NVIDIA_TTS_CFG_TOP_K if cfg_enabled else NVIDIA_TTS_TOP_K,
    }
    if cfg_enabled:
        kwargs["cfg_scale"] = NVIDIA_TTS_CFG_SCALE
    return kwargs


def _request_id(request: VllmGenerationRequest) -> str:
    suffix = f"-{request.request_id_suffix}" if request.request_id_suffix else ""
    return f"audex-{request.debug_name}{suffix}-{uuid.uuid4().hex}"


def split_tts_text_segments(text: str, *, target_segments: int = 4) -> tuple[str, ...]:
    """Split response text into ordered TTS work units for vLLM batching."""

    cleaned = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if not cleaned:
        return ()
    target_segments = max(1, int(target_segments))
    sentence_segments = tuple(
        segment.strip()
        for segment in re.findall(r"[^.!?]+(?:[.!?]+|$)", cleaned)
        if segment.strip()
    )
    segments = list(sentence_segments or (cleaned,))
    while len(segments) < target_segments:
        split_index = _longest_splittable_segment_index(segments)
        if split_index is None:
            break
        left, right = _split_segment_near_middle(segments[split_index])
        segments[split_index : split_index + 1] = [left, right]
    return tuple(segments)


def _longest_splittable_segment_index(segments: list[str]) -> int | None:
    candidates = [
        (len(segment.split()), index)
        for index, segment in enumerate(segments)
        if len(segment.split()) >= 3
    ]
    if not candidates:
        return None
    return max(candidates)[1]


def _split_segment_near_middle(segment: str) -> tuple[str, str]:
    words = segment.split()
    midpoint = max(1, len(words) // 2)
    left = " ".join(words[:midpoint]).strip()
    right = " ".join(words[midpoint:]).strip()
    if not left or not right:
        return segment, ""
    if left[-1] not in ",;:.!?":
        left += ","
    return left, right


def _segment_token_budgets(
    segments: tuple[str, ...],
    max_tokens: int | None,
) -> tuple[int | None, ...]:
    if max_tokens is None:
        return tuple(None for _ in segments)
    total_chars = max(1, sum(len(segment) for segment in segments))
    budgets = []
    for segment in segments:
        share = int(max_tokens * (len(segment) / total_chars))
        budgets.append(max(16, share + 16))
    return tuple(budgets)


def _flatten_completed_tts_tokens(
    token_ids_by_segment: list[tuple[int, ...]],
    through_index: int,
) -> list[int]:
    tokens: list[int] = []
    for index in range(through_index + 1):
        tokens.extend(token_ids_by_segment[index])
    return tokens


def clean_stop_text(text: str) -> str:
    for stop in STOP_TEXTS:
        if stop in text:
            text = text[: text.index(stop)]
    return text.strip()


def clean_transcription(text: str) -> str:
    text = clean_stop_text(text)
    first_quote = text.find("'")
    last_quote = text.rfind("'")
    if first_quote != -1 and last_quote > first_quote:
        return text[first_quote + 1 : last_quote].strip()
    return text


def extract_spoken_answer(full_output: str) -> str:
    text = clean_stop_text(full_output)
    if "</think>" in text:
        answer = text.rsplit("</think>", 1)[-1].strip()
        if answer:
            return scrub_spoken_answer(answer)
    return scrub_spoken_answer(text)


_BOILERPLATE_LINE_PATTERNS = (
    re.compile(
        r"^Audex (?:is|was) (?:a conversational partner )?"
        r"(?:created|built)(?: by NVIDIA)?\b",
        re.IGNORECASE,
    ),
    re.compile(r"^Audex is a conversational partner\b", re.IGNORECASE),
    re.compile(r"^It (?:is ready|can help you|was created|is likely)\b", re.IGNORECASE),
    re.compile(r"^NVIDIA is a company\b", re.IGNORECASE),
    re.compile(r"^The Nemotron-Cascade-\d+ model\b", re.IGNORECASE),
    re.compile(r"^\[CRITICAL\]", re.IGNORECASE),
    re.compile(r"^Your turn\b", re.IGNORECASE),
)


def scrub_spoken_answer(text: str) -> str:
    """Remove prompt/template leakage before text is persisted or spoken."""

    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if kept_lines and kept_lines[-1]:
                kept_lines.append("")
            continue
        if any(pattern.search(stripped) for pattern in _BOILERPLATE_LINE_PATTERNS):
            continue
        if "Do not write literal keyboard-action words" in stripped:
            continue
        kept_lines.append(stripped)
    return "\n".join(kept_lines).strip()


def extract_tts_codec_frames(
    token_ids: tuple[int, ...],
    token_map: Any,
) -> VllmSpeechCodecResult:
    codec_frames: list[int] = []
    reached_end_token = False
    for token_id in token_ids:
        if token_id == token_map.speechgen_end:
            reached_end_token = True
            break
        if token_id == token_map.speechgen_start:
            continue
        if token_id in token_map.speech_codec:
            codec_frames.append(token_map.speech_codec[token_id])
    return VllmSpeechCodecResult(
        generated_token_ids=token_ids,
        generated_codec_frames=tuple(codec_frames),
        reached_end_token=reached_end_token,
    )


def _sampling_kwargs(sampling: Any) -> dict[str, Any]:
    kwargs = sampling.as_kwargs()
    output_kind = kwargs.get("output_kind")
    if isinstance(output_kind, str):
        try:
            from vllm.sampling_params import RequestOutputKind

            kwargs["output_kind"] = RequestOutputKind[output_kind]
        except Exception:
            pass
    return kwargs


def _is_delta_output_kind(output_kind: Any) -> bool:
    if output_kind is None:
        return False
    if isinstance(output_kind, str):
        return output_kind == "DELTA"
    return getattr(output_kind, "name", None) == "DELTA"


def _replace_text(result: VllmRequestResult, text: str) -> VllmRequestResult:
    return VllmRequestResult(
        text=text,
        token_ids=result.token_ids,
        elapsed_seconds=result.elapsed_seconds,
        finish_reason=result.finish_reason,
        request_debug_name=result.request_debug_name,
    )
