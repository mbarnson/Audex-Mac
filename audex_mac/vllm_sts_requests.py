"""vLLM request builders for NVIDIA-shaped Audex speech-to-speech."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from .audio_contract import (
    DEFAULT_SYSTEM_PROMPT,
    NVIDIA_ASR_TEMPERATURE,
    NVIDIA_ASR_TOP_P,
    NVIDIA_TEXT_TEMPERATURE,
    NVIDIA_TEXT_TOP_P,
    NVIDIA_TTS_CFG_SCALE,
    NVIDIA_TTS_CFG_TEMPERATURE,
    NVIDIA_TTS_CFG_TOP_K,
    NVIDIA_TTS_CFG_TOP_P,
    NVIDIA_TTS_TEMPERATURE,
    NVIDIA_TTS_TOP_K,
    NVIDIA_TTS_TOP_P,
    SAMPLE_RATE,
    SOUND_END_TOKEN,
    SOUND_PLACEHOLDER,
    SOUND_START_TOKEN,
    SOUND_TOKEN,
    SPEECHGEN_END_TOKEN,
    SPEECHGEN_START_TOKEN,
    build_tts_null_prompt,
    build_tts_prompt,
    tokenize_tts_cfg_pair,
)
from .patches.vllm_metal_audex_adapter import (
    projected_audio_embedding_count,
    raw_audio_num_embeddings,
)

DEFAULT_VLLM_TEXT_PROMPT = (
    "Answer the user's latest message for a spoken conversation. Keep the "
    "answer natural, concise, and directly on topic. Only write words meant to "
    "be spoken aloud. Use ordinary sentence punctuation and capitalization. "
    "Do not introduce yourself, talk about yourself, describe how you work, or "
    "repeat these instructions. Do not use markdown, bullet points, headers, "
    "citations, keyboard-action words, or placeholder syntax."
)
DEFAULT_VLLM_AUDIO_RESPONSE_PROMPT = (
    "The audio contains a user speaking to you. Answer the question that user "
    "asks; do not transcribe, quote, or describe the audio."
)
AUDEX_TEXT_STATE_KEY_ARG = "audex_text_state_key"
AUDEX_TEXT_STATE_MODE_ARG = "audex_text_state_mode"
AUDEX_TEXT_STATE_BOUNDARY_ARG = "audex_text_state_boundary"
AUDEX_TEXT_STATE_PREFIX_TOKEN_COUNT_ARG = "audex_text_state_prefix_token_count"
AUDEX_TEXT_STATE_PREFIX_TOKEN_HASH_ARG = "audex_text_state_prefix_token_hash"
AUDEX_TEXT_STATE_APPEND_MODE = "append"
AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY = "committed_history_prefill"
DEFAULT_ASR_PROMPT = "Transcribe the input speech."
DEFAULT_ASR_MAX_TOKENS = 2048
DEFAULT_TEXT_MAX_TOKENS = 4096
DEFAULT_TTS_MAX_TOKENS = 2400
_TEXT_DISALLOWED_MODAL_MARKERS = (
    SOUND_PLACEHOLDER,
    SOUND_TOKEN,
    SOUND_START_TOKEN,
    SOUND_END_TOKEN,
    SPEECHGEN_START_TOKEN,
    SPEECHGEN_END_TOKEN,
    "<audiogen_start>",
    "<audiogen_end>",
    "<audiogen>",
)
_TEXT_DISALLOWED_CODEC_RE = re.compile(r"<(?:speech|audio)codec_\d+>")


@dataclass(frozen=True, slots=True)
class VllmSamplingPlan:
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int | None = None
    seed: int | None = None
    detokenize: bool | None = None
    output_kind: str | None = None
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    extra_args: dict[str, Any] | None = None

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.detokenize is not None:
            kwargs["detokenize"] = self.detokenize
        if self.output_kind is not None:
            kwargs["output_kind"] = self.output_kind
        if self.stop:
            kwargs["stop"] = list(self.stop)
        if self.stop_token_ids:
            kwargs["stop_token_ids"] = list(self.stop_token_ids)
        if self.extra_args is not None:
            kwargs["extra_args"] = dict(self.extra_args)
        return kwargs


@dataclass(frozen=True, slots=True)
class VllmTtsSamplingConfig:
    temperature: float
    top_p: float
    top_k: int
    cfg_scale: float
    seed: int | None = None
    require_compact_window_decode: bool = False

    @property
    def cfg_enabled(self) -> bool:
        return self.cfg_scale > 1.0


@dataclass(frozen=True, slots=True)
class VllmGenerationRequest:
    prompt: str | dict[str, Any]
    sampling: VllmSamplingPlan
    debug_name: str
    request_id_suffix: str | None = None


def compose_text_input(prompt: str, text: str) -> str:
    prompt = prompt.strip()
    text = text.strip()
    return f"{prompt}\n\n{text}" if prompt else text


def compose_system_prompt(system_prompt: str, response_policy: str) -> str:
    system_prompt = system_prompt.strip()
    response_policy = response_policy.strip()
    if not response_policy:
        return system_prompt
    if not system_prompt:
        return response_policy
    return f"{system_prompt}\n\n{response_policy}"


def build_chat_prompt(
    tokenizer: Any,
    user_text: str,
    *,
    enable_thinking: bool,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_asr_request(
    tokenizer: Any,
    audio_samples: Any,
    *,
    audio_placeholder: str = SOUND_TOKEN,
    sample_rate: int = SAMPLE_RATE,
    max_tokens: int = DEFAULT_ASR_MAX_TOKENS,
) -> VllmGenerationRequest:
    prompt = build_chat_prompt(
        tokenizer,
        f"{DEFAULT_ASR_PROMPT}\n{audio_placeholder}",
        enable_thinking=False,
    )
    return VllmGenerationRequest(
        prompt={
            "prompt": prompt,
            "multi_modal_data": {"audio": [(audio_samples, sample_rate)]},
        },
        sampling=VllmSamplingPlan(
            max_tokens=max_tokens,
            temperature=NVIDIA_ASR_TEMPERATURE,
            top_p=NVIDIA_ASR_TOP_P,
            extra_args=_text_modality_guard_extra_args(tokenizer),
        ),
        debug_name="asr",
    )


def build_audio_messages_response_request(
    tokenizer: Any,
    messages: list[dict[str, str]],
    audio_samples: Any,
    *,
    audio_placeholder: str = SOUND_TOKEN,
    sample_rate: int = SAMPLE_RATE,
    prompt_text: str = DEFAULT_VLLM_AUDIO_RESPONSE_PROMPT,
    enable_reasoning: bool = False,
    max_tokens: int = DEFAULT_TEXT_MAX_TOKENS,
    trim_padded_audio_embeddings: bool = False,
    conversation_state_key: str | None = None,
    conversation_state_prefix_token_count: int | None = None,
    conversation_state_prefix_token_hash: str | None = None,
    conversation_state_boundary: str | None = None,
) -> VllmGenerationRequest:
    """Build one Audex request that answers the current spoken user turn."""

    prompt_messages = _audio_response_prompt_messages(
        messages,
        prompt_text=prompt_text,
        audio_placeholder=audio_placeholder,
    )
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_reasoning,
    )
    audio_payload: Any = (audio_samples, sample_rate)
    if trim_padded_audio_embeddings:
        audio_payload = {
            "audex_raw_audio_samples": audio_samples,
            "sample_rate": sample_rate,
            "audex_raw_audio_num_embeddings": raw_audio_num_embeddings(
                audio_samples,
                sample_rate=sample_rate,
                trim_padded=True,
            ),
        }
    return VllmGenerationRequest(
        prompt={
            "prompt": prompt,
            "multi_modal_data": {"audio": [audio_payload]},
        },
        sampling=VllmSamplingPlan(
            max_tokens=max_tokens,
            temperature=NVIDIA_TEXT_TEMPERATURE,
            top_p=NVIDIA_TEXT_TOP_P,
            extra_args=_with_text_conversation_state_extra_args(
                _text_modality_guard_extra_args(tokenizer),
                conversation_state_key=conversation_state_key,
                boundary=conversation_state_boundary,
                prefix_token_count=conversation_state_prefix_token_count,
                prefix_token_hash=conversation_state_prefix_token_hash,
            ),
        ),
        debug_name="audio-response",
    )


def build_audio_response_prefix_token_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    prompt_text: str = DEFAULT_VLLM_AUDIO_RESPONSE_PROMPT,
    audio_placeholder: str = SOUND_TOKEN,
) -> tuple[int, ...]:
    """Tokenize the exact direct prompt prefix before its first audio token."""

    prompt = tokenizer.apply_chat_template(
        _audio_response_prompt_messages(
            messages,
            prompt_text=prompt_text,
            audio_placeholder=audio_placeholder,
        ),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_ids = tuple(int(token_id) for token_id in tokenizer.encode(prompt))
    sound_token_id = int(tokenizer.get_vocab()[audio_placeholder])
    try:
        audio_offset = token_ids.index(sound_token_id)
    except ValueError as exc:
        raise ValueError("Direct Audex prompt has no audio placeholder token.") from exc
    if audio_offset <= 0:
        raise ValueError("Direct Audex audio placeholder cannot start the prompt.")
    return token_ids[:audio_offset]


def build_audio_history_prime_request(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    conversation_state_key: str,
    conversation_state_prefix_token_count: int,
    conversation_state_prefix_token_hash: str,
) -> VllmGenerationRequest:
    """Prefill and retain the exact history prefix without user-visible output."""

    request = build_audio_messages_response_request(
        tokenizer,
        messages,
        (0.0,),
        max_tokens=1,
        trim_padded_audio_embeddings=True,
        conversation_state_key=conversation_state_key,
        conversation_state_prefix_token_count=conversation_state_prefix_token_count,
        conversation_state_prefix_token_hash=conversation_state_prefix_token_hash,
        conversation_state_boundary=AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY,
    )
    return VllmGenerationRequest(
        prompt=request.prompt,
        sampling=replace(request.sampling, temperature=0.0, top_p=1.0),
        debug_name="audio-history-prime",
    )


def _audio_response_prompt_messages(
    messages: list[dict[str, str]],
    *,
    prompt_text: str,
    audio_placeholder: str,
) -> list[dict[str, str]]:
    prompt_messages = [dict(message) for message in messages]
    prompt_messages.append(
        {
            "role": "user",
            "content": f"{prompt_text.strip()}\n{audio_placeholder}",
        }
    )
    return prompt_messages


def build_asr_projected_embeddings_request(
    tokenizer: Any,
    projected_embeddings: Any,
    *,
    audio_placeholder: str = SOUND_TOKEN,
    num_embeddings: int | None = None,
    max_tokens: int = DEFAULT_ASR_MAX_TOKENS,
) -> VllmGenerationRequest:
    """Build Audex ASR request using precomputed projected audio embeddings."""

    embedding_count = (
        num_embeddings
        if num_embeddings is not None
        else projected_audio_embedding_count(projected_embeddings)
    )
    if embedding_count <= 0:
        raise ValueError(f"num_embeddings must be positive, got {embedding_count}")
    transport_embeddings = make_projected_embeddings_vllm_serializable(
        projected_embeddings
    )
    prompt = build_chat_prompt(
        tokenizer,
        f"{DEFAULT_ASR_PROMPT}\n{audio_placeholder * embedding_count}",
        enable_thinking=False,
    )
    return VllmGenerationRequest(
        prompt={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": [
                    {
                        "audex_projected_embeddings": transport_embeddings,
                    }
                ]
            },
        },
        sampling=VllmSamplingPlan(
            max_tokens=max_tokens,
            temperature=NVIDIA_ASR_TEMPERATURE,
            top_p=NVIDIA_ASR_TOP_P,
            extra_args=_text_modality_guard_extra_args(tokenizer),
        ),
        debug_name="asr-projected",
    )


def make_projected_embeddings_vllm_serializable(projected_embeddings: Any) -> Any:
    """Convert MLX arrays to vLLM's supported IPC tensor transport type."""

    if not _is_mlx_array(projected_embeddings):
        return projected_embeddings
    from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

    return mlx_to_torch(projected_embeddings, device="cpu")


def _is_mlx_array(value: Any) -> bool:
    value_type = type(value)
    return value_type.__module__ == "mlx.core" and value_type.__name__ == "array"


def build_text_response_request(
    tokenizer: Any,
    transcript: str,
    *,
    prompt_text: str = DEFAULT_VLLM_TEXT_PROMPT,
    enable_reasoning: bool = False,
    max_tokens: int = DEFAULT_TEXT_MAX_TOKENS,
) -> VllmGenerationRequest:
    prompt = build_chat_prompt(
        tokenizer,
        transcript,
        enable_thinking=enable_reasoning,
        system_prompt=compose_system_prompt(DEFAULT_SYSTEM_PROMPT, prompt_text),
    )
    return VllmGenerationRequest(
        prompt=prompt,
        sampling=VllmSamplingPlan(
            max_tokens=max_tokens,
            temperature=NVIDIA_TEXT_TEMPERATURE,
            top_p=NVIDIA_TEXT_TOP_P,
            extra_args=_text_modality_guard_extra_args(tokenizer),
        ),
        debug_name="text",
    )


def build_text_messages_response_request(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    prompt_text: str = DEFAULT_VLLM_TEXT_PROMPT,
    enable_reasoning: bool = False,
    max_tokens: int = DEFAULT_TEXT_MAX_TOKENS,
    conversation_state_key: str | None = None,
    conversation_state_boundary: str | None = None,
    conversation_state_prefix_token_count: int | None = None,
    conversation_state_prefix_token_hash: str | None = None,
) -> VllmGenerationRequest:
    prompt = build_text_messages_generation_prompt(
        tokenizer,
        messages,
        prompt_text=prompt_text,
        enable_reasoning=enable_reasoning,
    )
    return VllmGenerationRequest(
        prompt=prompt,
        sampling=VllmSamplingPlan(
            max_tokens=max_tokens,
            temperature=NVIDIA_TEXT_TEMPERATURE,
            top_p=NVIDIA_TEXT_TOP_P,
            extra_args=_with_text_conversation_state_extra_args(
                _text_modality_guard_extra_args(tokenizer),
                conversation_state_key=conversation_state_key,
                boundary=conversation_state_boundary,
                prefix_token_count=conversation_state_prefix_token_count,
                prefix_token_hash=conversation_state_prefix_token_hash,
            ),
        ),
        debug_name="text",
    )


def build_text_messages_generation_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    prompt_text: str = DEFAULT_VLLM_TEXT_PROMPT,
    enable_reasoning: bool = False,
) -> str:
    prompt_messages = _messages_with_current_user_instruction(messages, prompt_text)
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_reasoning,
    )


def build_text_messages_history_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    prompt_text: str = DEFAULT_VLLM_TEXT_PROMPT,
    enable_reasoning: bool = False,
) -> str:
    prompt_messages = _messages_with_current_user_instruction(messages, prompt_text)
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_reasoning,
    )


def _messages_with_current_user_instruction(
    messages: list[dict[str, str]],
    prompt_text: str,
) -> list[dict[str, str]]:
    if not messages or not prompt_text.strip():
        return list(messages)

    prompt_messages = [dict(message) for message in messages]
    for index, message in enumerate(prompt_messages):
        if message.get("role") != "system":
            continue
        prompt_messages[index]["content"] = compose_system_prompt(
            message.get("content", ""),
            prompt_text,
        )
        break
    else:
        prompt_messages.insert(0, {"role": "system", "content": prompt_text.strip()})
    return prompt_messages


def _with_text_conversation_state_extra_args(
    extra_args: dict[str, Any] | None,
    *,
    conversation_state_key: str | None,
    boundary: str | None,
    prefix_token_count: int | None,
    prefix_token_hash: str | None,
) -> dict[str, Any] | None:
    if not conversation_state_key:
        return extra_args
    merged = dict(extra_args or {})
    merged.update(
        {
            AUDEX_TEXT_STATE_KEY_ARG: conversation_state_key,
            AUDEX_TEXT_STATE_MODE_ARG: AUDEX_TEXT_STATE_APPEND_MODE,
            AUDEX_TEXT_STATE_PREFIX_TOKEN_COUNT_ARG: int(prefix_token_count or 0),
            AUDEX_TEXT_STATE_PREFIX_TOKEN_HASH_ARG: prefix_token_hash or "",
        }
    )
    if boundary:
        merged[AUDEX_TEXT_STATE_BOUNDARY_ARG] = boundary
    return merged


def build_tts_request(
    tokenizer: Any,
    text: str,
    *,
    speechgen_end_id: int,
    eos_token_id: int | None,
    codec_min_id: int | None = None,
    codec_max_id: int | None = None,
    skip_paged_logits_eval: bool = False,
    max_tokens: int = DEFAULT_TTS_MAX_TOKENS,
    temperature: float = NVIDIA_TTS_TEMPERATURE,
    top_p: float = NVIDIA_TTS_TOP_P,
    top_k: int = NVIDIA_TTS_TOP_K,
    seed: int | None = None,
    require_compact_window_decode: bool = False,
) -> VllmGenerationRequest:
    return VllmGenerationRequest(
        prompt=build_tts_prompt(text, tokenizer),
        sampling=_tts_sampling_plan(
            speechgen_end_id=speechgen_end_id,
            eos_token_id=eos_token_id,
            codec_min_id=codec_min_id,
            codec_max_id=codec_max_id,
            skip_paged_logits_eval=skip_paged_logits_eval,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            require_compact_window_decode=require_compact_window_decode,
        ),
        debug_name="tts",
    )


def build_tts_cfg_requests(
    tokenizer: Any,
    text: str,
    *,
    speechgen_end_id: int,
    eos_token_id: int | None,
    pair_id: str,
    codec_min_id: int | None = None,
    codec_max_id: int | None = None,
    max_tokens: int = DEFAULT_TTS_MAX_TOKENS,
    cfg_scale: float = NVIDIA_TTS_CFG_SCALE,
    temperature: float = NVIDIA_TTS_CFG_TEMPERATURE,
    top_p: float = NVIDIA_TTS_CFG_TOP_P,
    top_k: int = NVIDIA_TTS_CFG_TOP_K,
    seed: int | None = None,
    require_compact_window_decode: bool = False,
) -> tuple[VllmGenerationRequest, VllmGenerationRequest]:
    cond_prompt = build_tts_prompt(text, tokenizer)
    uncond_prompt = build_tts_null_prompt(cond_prompt, tokenizer)
    cond_ids, uncond_ids = tokenize_tts_cfg_pair(cond_prompt, uncond_prompt, tokenizer)
    base_sampling = _tts_sampling_plan(
        speechgen_end_id=speechgen_end_id,
        eos_token_id=eos_token_id,
        codec_min_id=codec_min_id,
        codec_max_id=codec_max_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        require_compact_window_decode=require_compact_window_decode,
        max_tokens=max_tokens,
    )
    cond_sampling = _with_cfg_extra_args(
        base_sampling,
        cfg_scale=cfg_scale,
        cfg_role="cond",
        pair_id=pair_id,
    )
    uncond_sampling = _with_cfg_extra_args(
        base_sampling,
        cfg_scale=cfg_scale,
        cfg_role="uncond",
        pair_id=pair_id,
    )
    return (
        VllmGenerationRequest(
            prompt={"prompt_token_ids": list(cond_ids)},
            sampling=cond_sampling,
            debug_name="tts-cond",
            request_id_suffix="cond",
        ),
        VllmGenerationRequest(
            prompt={"prompt_token_ids": list(uncond_ids)},
            sampling=uncond_sampling,
            debug_name="tts-uncond",
            request_id_suffix="uncond",
        ),
    )


def _tts_sampling_plan(
    *,
    speechgen_end_id: int,
    eos_token_id: int | None,
    codec_min_id: int | None = None,
    codec_max_id: int | None = None,
    skip_paged_logits_eval: bool = False,
    temperature: float = NVIDIA_TTS_TEMPERATURE,
    top_p: float = NVIDIA_TTS_TOP_P,
    top_k: int = NVIDIA_TTS_TOP_K,
    seed: int | None = None,
    require_compact_window_decode: bool = False,
    max_tokens: int,
) -> VllmSamplingPlan:
    stop_token_ids = [speechgen_end_id]
    if eos_token_id is not None:
        stop_token_ids.append(eos_token_id)
    extra_args = None
    if codec_min_id is not None and codec_max_id is not None:
        extra_args = {
            "audex_tts_codec_min_id": int(codec_min_id),
            "audex_tts_codec_max_id": int(codec_max_id),
            "audex_tts_speechgen_end_id": int(speechgen_end_id),
        }
    if skip_paged_logits_eval:
        extra_args = dict(extra_args or {})
        extra_args["audex_tts_skip_paged_logits_eval"] = True
    if require_compact_window_decode:
        extra_args = dict(extra_args or {})
        extra_args["audex_tts_require_compact_window_decode"] = True
    return VllmSamplingPlan(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else None,
        seed=seed,
        detokenize=False,
        stop_token_ids=tuple(stop_token_ids),
        extra_args=extra_args,
    )


def _text_modality_guard_extra_args(tokenizer: Any) -> dict[str, Any] | None:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        return None
    vocab = get_vocab()
    if not isinstance(vocab, dict):
        return None

    codec_token_ids = [
        int(token_id)
        for token, token_id in vocab.items()
        if _TEXT_DISALLOWED_CODEC_RE.fullmatch(str(token))
    ]
    marker_token_ids = [
        int(vocab[token]) for token in _TEXT_DISALLOWED_MODAL_MARKERS if token in vocab
    ]
    ranges = _compact_token_id_ranges(codec_token_ids)
    marker_token_ids = sorted(set(marker_token_ids))
    if not ranges and not marker_token_ids:
        return None
    return {
        "audex_disallow_token_ranges": [list(token_range) for token_range in ranges],
        "audex_disallow_token_ids": marker_token_ids,
    }


def _compact_token_id_ranges(token_ids: list[int]) -> list[tuple[int, int]]:
    if not token_ids:
        return []
    sorted_ids = sorted(set(int(token_id) for token_id in token_ids))
    ranges: list[tuple[int, int]] = []
    start = previous = sorted_ids[0]
    for token_id in sorted_ids[1:]:
        if token_id == previous + 1:
            previous = token_id
            continue
        ranges.append((start, previous))
        start = previous = token_id
    ranges.append((start, previous))
    return ranges


def _with_cfg_extra_args(
    sampling: VllmSamplingPlan,
    *,
    cfg_scale: float,
    cfg_role: str,
    pair_id: str,
) -> VllmSamplingPlan:
    extra_args = dict(sampling.extra_args or {})
    extra_args.update(
        {
            "cfg_scale": cfg_scale,
            "cfg_role": cfg_role,
            "cfg_pair_id": pair_id,
        }
    )
    return VllmSamplingPlan(
        max_tokens=sampling.max_tokens,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        seed=sampling.seed,
        detokenize=sampling.detokenize,
        stop=sampling.stop,
        stop_token_ids=sampling.stop_token_ids,
        extra_args=extra_args,
    )
