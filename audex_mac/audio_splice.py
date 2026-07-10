"""Audex audio-embedding splice helpers for MLX generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_contract import (
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_SOUND_EMBEDDING_SIZE,
    SOUND_TOKEN,
    build_audio_chat_prompt,
    build_audio_prompt_plan,
)
from .audio_encoder import project_zero_audio_features_mlx
from .patches import apply_audex_runtime_patches


@dataclass(frozen=True, slots=True)
class AudioEmbeddingSplicePlan:
    sound_token_id: int
    prompt_tokens: tuple[int, ...]
    sound_positions: tuple[int, ...]
    audio_embedding_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AudioEmbeddingSpliceSmokeResult:
    backend: str
    device: str
    prompt_tokens: int
    sound_tokens: int
    audio_embedding_shape: tuple[int, ...]
    input_embedding_shape: tuple[int, ...]
    spliced_embedding_shape: tuple[int, ...]
    generated_token_id: int
    logprobs_shape: tuple[int, ...]


def find_sound_token_positions(
    token_ids: list[int] | tuple[int, ...],
    *,
    sound_token_id: int,
) -> tuple[int, ...]:
    """Return all `<so_embedding>` placeholder positions in token order."""

    return tuple(
        index for index, token_id in enumerate(token_ids) if token_id == sound_token_id
    )


def validate_audio_splice_plan(
    token_ids: list[int] | tuple[int, ...],
    audio_embedding_shape: tuple[int, ...],
    *,
    sound_token_id: int,
) -> AudioEmbeddingSplicePlan:
    """Validate token/audio counts before mutating prompt embeddings."""

    positions = find_sound_token_positions(token_ids, sound_token_id=sound_token_id)
    if len(audio_embedding_shape) != 2:
        raise ValueError(
            "Audio embeddings must be flattened to (tokens, hidden), got "
            f"{audio_embedding_shape}"
        )
    audio_tokens = int(audio_embedding_shape[0])
    if len(positions) != audio_tokens:
        raise ValueError(
            "Mismatch between <so_embedding> token count and projected audio tokens: "
            f"placeholders={len(positions)} audio_tokens={audio_tokens}"
        )
    return AudioEmbeddingSplicePlan(
        sound_token_id=sound_token_id,
        prompt_tokens=tuple(int(token_id) for token_id in token_ids),
        sound_positions=positions,
        audio_embedding_shape=audio_embedding_shape,
    )


def splice_audio_embeddings_mlx(
    token_ids: Any,
    input_embeddings: Any,
    audio_embeddings: Any,
    *,
    sound_token_id: int,
) -> Any:
    """Replace Audex `<so_embedding>` token embeddings with audio embeddings."""

    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "Audex audio embedding splice requires mlx in the active runtime."
        ) from exc

    if input_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D input embeddings, got {input_embeddings.shape}")
    if audio_embeddings.ndim == 3:
        clips, tokens_per_clip, hidden = audio_embeddings.shape
        audio_embeddings = mx.reshape(
            audio_embeddings,
            (int(clips) * int(tokens_per_clip), int(hidden)),
        )
    if audio_embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D or 3D audio embeddings, got {audio_embeddings.shape}"
        )
    if int(input_embeddings.shape[-1]) != int(audio_embeddings.shape[-1]):
        raise ValueError(
            "Input/audio hidden-size mismatch: "
            f"input={input_embeddings.shape[-1]} audio={audio_embeddings.shape[-1]}"
        )

    mask = token_ids == sound_token_id
    placeholder_count = int(mx.sum(mask.astype(mx.int32)).item())
    audio_token_count = int(audio_embeddings.shape[0])
    if placeholder_count != audio_token_count:
        raise ValueError(
            "Mismatch between <so_embedding> token count and projected audio tokens: "
            f"placeholders={placeholder_count} audio_tokens={audio_token_count}"
        )

    feature_indices = mx.where(
        mask,
        mx.cumsum(mask.astype(mx.int32)) - 1,
        mx.zeros_like(token_ids),
    )
    gathered_audio = mx.take(audio_embeddings, feature_indices, axis=0).astype(
        input_embeddings.dtype
    )
    spliced = mx.where(mx.expand_dims(mask, axis=-1), gathered_audio, input_embeddings)
    mx.eval(spliced)
    return spliced


def run_audio_embedding_splice_smoke(
    *,
    full_model_path: Path,
    text_model_path: Path,
    sample_count: int = 1,
) -> AudioEmbeddingSpliceSmokeResult:
    """Run a one-token MLX-LM generation smoke with Audex audio embeddings."""

    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Audex audio embedding splice preflight requires mlx, mlx_lm, and "
            "transformers in the active runtime."
        ) from exc

    mx.set_default_device(mx.gpu)
    apply_audex_runtime_patches()
    projected = project_zero_audio_features_mlx(full_model_path)
    audio_embeddings = projected.projected_embeddings
    audio_embeddings_flat = mx.reshape(
        audio_embeddings,
        (
            int(audio_embeddings.shape[0]) * int(audio_embeddings.shape[1]),
            int(audio_embeddings.shape[2]),
        ),
    )

    full_tokenizer = AutoTokenizer.from_pretrained(
        str(full_model_path),
        trust_remote_code=True,
    )
    sound_token_id = int(full_tokenizer.convert_tokens_to_ids(SOUND_TOKEN))
    encoded_sound_token = full_tokenizer.encode(SOUND_TOKEN, add_special_tokens=False)
    if encoded_sound_token != [sound_token_id]:
        raise ValueError(
            f"Full Audex tokenizer must encode {SOUND_TOKEN} as one token, got "
            f"{encoded_sound_token}"
        )
    prompt_plan = build_audio_prompt_plan(
        DEFAULT_AUDIO_PROMPT,
        sample_count=sample_count,
        embeddings_per_clip=DEFAULT_SOUND_EMBEDDING_SIZE,
    )
    prompt = build_audio_chat_prompt(prompt_plan, thinking_enabled=False)
    prompt_tokens = tuple(
        int(token_id)
        for token_id in full_tokenizer.encode(prompt, add_special_tokens=False)
    )
    validate_audio_splice_plan(
        prompt_tokens,
        tuple(int(part) for part in audio_embeddings_flat.shape),
        sound_token_id=sound_token_id,
    )

    model, _ = load(
        str(text_model_path),
        tokenizer_config={"trust_remote_code": True},
    )
    text_vocab_size = int(model.args.vocab_size)
    max_prompt_token = max(prompt_tokens)
    if max_prompt_token >= text_vocab_size:
        raise ValueError(
            "Audex audio prompt token exceeds text model vocabulary: "
            f"token={max_prompt_token} vocab_size={text_vocab_size}"
        )
    token_ids = mx.array(prompt_tokens, dtype=mx.int32)
    input_embeddings = model.model.embed_tokens(token_ids[None])[0]
    spliced = splice_audio_embeddings_mlx(
        token_ids,
        input_embeddings,
        audio_embeddings_flat,
        sound_token_id=sound_token_id,
    )
    generator = generate_step(
        token_ids,
        model,
        max_tokens=1,
        input_embeddings=spliced,
    )
    generated_token, logprobs = next(generator)
    return AudioEmbeddingSpliceSmokeResult(
        backend="mlx_lm",
        device=str(mx.default_device()),
        prompt_tokens=len(prompt_tokens),
        sound_tokens=prompt_tokens.count(sound_token_id),
        audio_embedding_shape=tuple(int(part) for part in audio_embeddings_flat.shape),
        input_embedding_shape=tuple(int(part) for part in input_embeddings.shape),
        spliced_embedding_shape=tuple(int(part) for part in spliced.shape),
        generated_token_id=int(generated_token),
        logprobs_shape=tuple(int(part) for part in logprobs.shape),
    )
