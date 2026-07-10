"""Audex speech-token generation preflights for the MLX path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .audio_contract import (
    NVIDIA_TTS_CFG_SCALE,
    NVIDIA_TTS_CFG_TEMPERATURE,
    NVIDIA_TTS_CFG_TOP_K,
    NVIDIA_TTS_CFG_TOP_P,
    SPEECHGEN_START_TOKEN,
    build_codec_token_map,
    build_tts_null_prompt,
    build_tts_prompt,
    tokenize_tts_cfg_pair,
)
from .patches import apply_audex_runtime_patches


@dataclass(frozen=True, slots=True)
class SpeechTokenGenerationSmokeResult:
    backend: str
    device: str
    model_type: str
    vocab_size: int
    prompt_tokens: int
    prompt_max_token_id: int
    speechgen_start_id: int
    speechgen_end_id: int
    codec_token_count: int
    generated_token_ids: tuple[int, ...]
    generated_token_text: tuple[str, ...]
    generated_codec_frames: tuple[int, ...]
    logprobs_shape: tuple[int, ...]
    reached_end_token: bool
    hit_max_tokens: bool
    temperature: float
    top_p: float
    top_k: int
    cfg_scale_reference: float
    cfg_applied: bool

    @property
    def ready(self) -> bool:
        return (
            self.vocab_size > self.speechgen_end_id
            and self.prompt_max_token_id < self.vocab_size
            and len(self.generated_token_ids) > 0
            and len(self.generated_codec_frames) == len(self.generated_token_ids)
        )


def run_speech_token_generation_smoke(
    *,
    full_model_path: Path,
    text: str = "Say: hello from Audex on Mac.",
    max_tokens: int = 8,
    seed: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SpeechTokenGenerationSmokeResult:
    """Generate a short run of Audex speech-codec tokens from the full LM head."""

    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models import cache
        from mlx_lm.sample_utils import make_sampler
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Audex speech-token generation preflight requires mlx, mlx_lm, and "
            "transformers in the active runtime."
        ) from exc

    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")

    mx.set_default_device(mx.gpu)
    mx.random.seed(seed)
    apply_audex_runtime_patches()

    model, _ = load(
        str(full_model_path),
        tokenizer_config={"trust_remote_code": True},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(full_model_path),
        trust_remote_code=True,
    )
    template_path = full_model_path / "chat_template.jinja"
    if template_path.is_file():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")

    token_map = build_codec_token_map(tokenizer.get_vocab())
    speechgen_start_id = int(tokenizer.convert_tokens_to_ids(SPEECHGEN_START_TOKEN))
    if speechgen_start_id != token_map.speechgen_start:
        raise ValueError(
            f"Tokenizer mismatch for {SPEECHGEN_START_TOKEN}: "
            f"encode={speechgen_start_id} vocab={token_map.speechgen_start}"
        )

    prompt = build_tts_prompt(text, tokenizer)
    null_prompt = build_tts_null_prompt(prompt, tokenizer)
    prompt_tokens, null_prompt_tokens = tokenize_tts_cfg_pair(
        prompt,
        null_prompt,
        tokenizer,
    )
    vocab_size = int(model.args.vocab_size)
    prompt_max_token_id = max(prompt_tokens)
    if prompt_max_token_id >= vocab_size:
        raise ValueError(
            "Audex TTS prompt token exceeds full model vocabulary: "
            f"token={prompt_max_token_id} vocab_size={vocab_size}"
        )
    if token_map.speechgen_end >= vocab_size:
        raise ValueError(
            "Audex speech tokenizer exceeds full model vocabulary: "
            f"speechgen_end={token_map.speechgen_end} vocab_size={vocab_size}"
        )

    sampler = make_sampler(
        temp=NVIDIA_TTS_CFG_TEMPERATURE,
        top_p=NVIDIA_TTS_CFG_TOP_P,
        top_k=NVIDIA_TTS_CFG_TOP_K,
    )
    generated_token_ids, logprobs_shape = _generate_tts_cfg_token_ids(
        mx=mx,
        cache_module=cache,
        model=model,
        max_tokens=max_tokens,
        sampler=sampler,
        cond_prompt_tokens=prompt_tokens,
        uncond_prompt_tokens=null_prompt_tokens,
        cfg_scale=NVIDIA_TTS_CFG_SCALE,
        codec_min_id=min(token_map.speech_codec),
        codec_max_id=max(token_map.speech_codec),
        speechgen_end_id=token_map.speechgen_end,
        vocab_size=vocab_size,
        progress_callback=progress_callback,
    )
    generated_token_text: list[str] = []
    generated_codec_frames: list[int] = []
    for token_id in generated_token_ids:
        generated_token_text.append(str(tokenizer.convert_ids_to_tokens(token_id)))
        if token_id in token_map.speech_codec:
            generated_codec_frames.append(token_map.speech_codec[token_id])
    reached_end_token = bool(
        generated_token_ids and generated_token_ids[-1] == token_map.speechgen_end
    )

    return SpeechTokenGenerationSmokeResult(
        backend="mlx_lm",
        device=str(mx.default_device()),
        model_type=str(model.args.model_type),
        vocab_size=vocab_size,
        prompt_tokens=len(prompt_tokens),
        prompt_max_token_id=prompt_max_token_id,
        speechgen_start_id=token_map.speechgen_start,
        speechgen_end_id=token_map.speechgen_end,
        codec_token_count=len(token_map.speech_codec),
        generated_token_ids=tuple(generated_token_ids),
        generated_token_text=tuple(generated_token_text),
        generated_codec_frames=tuple(generated_codec_frames),
        logprobs_shape=logprobs_shape,
        reached_end_token=reached_end_token,
        hit_max_tokens=len(generated_token_ids) >= max_tokens and not reached_end_token,
        temperature=NVIDIA_TTS_CFG_TEMPERATURE,
        top_p=NVIDIA_TTS_CFG_TOP_P,
        top_k=NVIDIA_TTS_CFG_TOP_K,
        cfg_scale_reference=NVIDIA_TTS_CFG_SCALE,
        cfg_applied=True,
    )


def _generate_tts_cfg_token_ids(
    *,
    mx,
    cache_module,
    model,
    max_tokens: int,
    sampler,
    cond_prompt_tokens: tuple[int, ...],
    uncond_prompt_tokens: tuple[int, ...],
    cfg_scale: float,
    codec_min_id: int,
    codec_max_id: int,
    speechgen_end_id: int,
    vocab_size: int,
    progress_callback: Callable[[int, int], None] | None,
    token_callback: Callable[[int], None] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(cond_prompt_tokens) != len(uncond_prompt_tokens):
        raise ValueError(
            "TTS CFG prompts must be the same length: "
            f"cond={len(cond_prompt_tokens)} uncond={len(uncond_prompt_tokens)}"
        )
    if len(cond_prompt_tokens) == 0:
        raise ValueError("TTS CFG prompt must not be empty.")
    if cfg_scale < 1.0:
        raise ValueError(f"cfg_scale must be >= 1.0, got {cfg_scale}")

    cond_cache = cache_module.make_prompt_cache(model)
    uncond_cache = cache_module.make_prompt_cache(model)
    cond_tokens = mx.array(cond_prompt_tokens, dtype=mx.int32)
    uncond_tokens = mx.array(uncond_prompt_tokens, dtype=mx.int32)

    if len(cond_prompt_tokens) > 1:
        model(cond_tokens[:-1][None], cache=cond_cache)
        model(uncond_tokens[:-1][None], cache=uncond_cache)
        mx.eval([state.state for state in cond_cache])
        mx.eval([state.state for state in uncond_cache])

    next_cond = cond_tokens[-1:]
    next_uncond = uncond_tokens[-1:]
    generated: list[int] = []
    logprobs_shape: tuple[int, ...] = (vocab_size,)
    candidate_ids = mx.arange(vocab_size)
    allowed = (
        ((candidate_ids >= codec_min_id) & (candidate_ids <= codec_max_id))
        | (candidate_ids == speechgen_end_id)
    )[None, :]

    for _ in range(max_tokens):
        cond_logits = model(next_cond[None], cache=cond_cache)[:, -1, :]
        uncond_logits = model(next_uncond[None], cache=uncond_cache)[:, -1, :]
        blended_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
        blended_logits = mx.where(
            allowed,
            blended_logits,
            mx.array(-float("inf"), blended_logits.dtype),
        )
        logprobs = blended_logits - mx.logsumexp(blended_logits, keepdims=True)
        sampled = sampler(logprobs)
        mx.eval(sampled)
        token_id = int(sampled.item())
        generated.append(token_id)
        if token_callback is not None:
            token_callback(token_id)
        if progress_callback is not None and (
            len(generated) == 1
            or len(generated) % 16 == 0
            or token_id == speechgen_end_id
        ):
            progress_callback(len(generated), max_tokens)
        if token_id == speechgen_end_id:
            break
        next_cond = mx.array([token_id], dtype=mx.int32)
        next_uncond = next_cond

    return tuple(generated), logprobs_shape
