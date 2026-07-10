"""MLX execution for NVIDIA's Audex causal speech decoder."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPEECH_DECODER_DEVICE_ENV = "AUDEX_SPEECH_DECODER_DEVICE"
DEFAULT_SPEECH_DECODER_DEVICE = "cpu"

DECODER_WEIGHT_KEYS = (
    "audex_speech_token_embedder.project_out.bias",
    "audex_speech_token_embedder.project_out.weight",
    "module.wav_proj.weight",
    "module.fc_post_a.weight",
    "module.lookahead_conv.weight",
    "module.lookahead_proj.weight",
    "module.backbone.final_layer_norm.weight",
    "module.head.proj.weight",
)

DECODER_LAYER_WEIGHT_SUFFIXES = (
    "att_norm.weight",
    "ffn_norm.weight",
    "att.c_attn.weight",
    "att.c_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)


@dataclass(frozen=True, slots=True)
class AudexSpeechDecoderConfig:
    hidden_dim: int
    depth: int
    heads: int
    pos_meb_dim: int
    hop_length: int
    vq_dim: int
    lookahead_steps: int
    sample_rate: int
    codebook_levels: tuple[int, ...]
    codebook_size: int
    token_embed_dim: int

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.heads


@dataclass(frozen=True, slots=True)
class SpeechDecoderSmokeResult:
    backend: str
    device: str
    frame_count: int
    input_shape: tuple[int, ...]
    vq_embedding_shape: tuple[int, ...]
    waveform_shape: tuple[int, ...]
    waveform_dtype: str
    sample_rate: int
    hop_length: int
    lookahead_steps: int
    finite: bool
    peak_abs: float

    @property
    def ready(self) -> bool:
        expected_samples = self.frame_count * self.hop_length
        return (
            self.finite
            and self.waveform_shape == (expected_samples,)
            and self.sample_rate == 16_000
            and self.peak_abs <= 1.0
        )


class AudexSpeechDecoderCache:
    def __init__(self) -> None:
        self.key_values: dict[int, tuple[Any, Any]] = {}
        self.position = 0

    def input_positions(self, mx: Any, length: int) -> Any:
        return mx.arange(self.position, self.position + length)[None, :]

    def update(self, mx: Any, layer_idx: int, key: Any, value: Any) -> tuple[Any, Any]:
        if layer_idx in self.key_values:
            prev_key, prev_value = self.key_values[layer_idx]
            key = mx.concatenate((prev_key, key), axis=2)
            value = mx.concatenate((prev_value, value), axis=2)
        self.key_values[layer_idx] = (key, value)
        return key, value

    def advance(self, length: int) -> None:
        self.position += length

    def reset(self) -> None:
        self.key_values.clear()
        self.position = 0


class AudexSpeechDecoderSession:
    def __init__(
        self,
        *,
        weights: dict[str, Any],
        config: AudexSpeechDecoderConfig,
        chunk_frames: int,
        device: Any | None = None,
    ) -> None:
        if chunk_frames <= 0:
            raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
        self.weights = weights
        self.config = config
        self.chunk_frames = chunk_frames
        self.device = device
        self.cache = AudexSpeechDecoderCache()
        self.buffer: list[list[int]] = []

    def reset(self) -> None:
        self.cache.reset()
        self.buffer.clear()

    def push(self, token_frames: Sequence[Sequence[int]]) -> list[tuple[int, Any]]:
        self.buffer.extend(list(frame) for frame in token_frames)
        return self._drain(flush=False)

    def flush(self) -> list[tuple[int, Any]]:
        return self._drain(flush=True)

    def _drain(self, *, flush: bool) -> list[tuple[int, Any]]:
        if self.device is not None:
            try:
                import mlx.core as mx
            except ImportError as exc:
                raise RuntimeError(
                    "Audex speech decoder execution requires mlx."
                ) from exc
            with mx.stream(self.device):
                return self._drain_on_current_stream(flush=flush)
        return self._drain_on_current_stream(flush=flush)

    def _drain_on_current_stream(self, *, flush: bool) -> list[tuple[int, Any]]:
        chunks: list[tuple[int, Any]] = []
        ready_frames = len(self.buffer) - self.config.lookahead_steps
        while self.buffer and (flush or ready_frames >= self.chunk_frames):
            emit_frames = (
                min(self.chunk_frames, len(self.buffer)) if flush else self.chunk_frames
            )
            waveform = self._decode_buffered_frames(emit_frames, flush=flush)
            del self.buffer[:emit_frames]
            ready_frames = len(self.buffer) - self.config.lookahead_steps
            chunks.append((self.config.sample_rate, waveform))
        return chunks

    def _decode_buffered_frames(self, emit_frames: int, *, flush: bool) -> Any:
        vq_emb = embed_speech_token_frames_mlx(
            self.buffer[:emit_frames],
            self.weights,
            self.config,
        )
        lookahead_vq_emb = None
        if self.config.lookahead_steps > 0:
            future_frames = self.buffer[
                emit_frames : emit_frames + self.config.lookahead_steps
            ]
            future_parts = []
            if future_frames:
                future_parts.append(
                    embed_speech_token_frames_mlx(
                        future_frames,
                        self.weights,
                        self.config,
                    )
                )
            missing_frames = (
                self.config.lookahead_steps - len(future_frames) if flush else 0
            )
            if missing_frames > 0:
                try:
                    import mlx.core as mx
                except ImportError as exc:
                    raise RuntimeError(
                        "Audex speech decoder execution requires mlx."
                    ) from exc
                future_parts.append(
                    mx.zeros(
                        (1, missing_frames, self.config.vq_dim),
                        dtype=vq_emb.dtype,
                    )
                )
            if len(future_parts) == 1:
                lookahead_vq_emb = future_parts[0]
            elif future_parts:
                try:
                    import mlx.core as mx
                except ImportError as exc:
                    raise RuntimeError(
                        "Audex speech decoder execution requires mlx."
                    ) from exc
                lookahead_vq_emb = mx.concatenate(future_parts, axis=1)
        return decode_speech_embeddings_cached_mlx(
            vq_emb,
            self.weights,
            self.config,
            self.cache,
            lookahead_vq_emb=lookahead_vq_emb,
        )


def configured_speech_decoder_device(mx: Any) -> Any:
    """Resolve the device dedicated to causal speech decoding."""

    configured = (
        os.environ.get(
            SPEECH_DECODER_DEVICE_ENV,
            DEFAULT_SPEECH_DECODER_DEVICE,
        )
        .strip()
        .lower()
    )
    if configured not in {"cpu", "gpu"}:
        raise ValueError(
            f"{SPEECH_DECODER_DEVICE_ENV} must be 'cpu' or 'gpu', "
            f"got {configured!r}."
        )
    device = getattr(mx, configured, None)
    if device is None and configured == "cpu":
        device = getattr(mx, "gpu", None)
    if device is None:
        raise RuntimeError(f"MLX device is unavailable: {configured}")
    return device


def load_speech_decoder_config(decoder_path: Path) -> AudexSpeechDecoderConfig:
    raw = json.loads((decoder_path / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config.json must be an object: {decoder_path}")

    return AudexSpeechDecoderConfig(
        hidden_dim=_required_int(raw, "hidden_dim"),
        depth=_required_int(raw, "depth"),
        heads=_required_int(raw, "heads"),
        pos_meb_dim=_required_int(raw, "pos_meb_dim"),
        hop_length=_required_int(raw, "hop_length"),
        vq_dim=_required_int(raw, "vq_dim"),
        lookahead_steps=_required_int(raw, "lookahead_steps"),
        sample_rate=_required_int(raw, "sample_rate"),
        codebook_levels=tuple(int(level) for level in raw["codebook_levels"]),
        codebook_size=_required_int(raw, "codebook_size"),
        token_embed_dim=_required_int(raw, "token_embed_dim"),
    )


def expected_speech_decoder_weight_keys(
    config: AudexSpeechDecoderConfig,
) -> tuple[str, ...]:
    layer_keys = tuple(
        f"module.backbone.transformers.{layer_idx}.{suffix}"
        for layer_idx in range(config.depth)
        for suffix in DECODER_LAYER_WEIGHT_SUFFIXES
    )
    if config.lookahead_steps <= 0:
        return tuple(
            key
            for key in DECODER_WEIGHT_KEYS + layer_keys
            if not key.startswith("module.lookahead_")
        )
    return DECODER_WEIGHT_KEYS + layer_keys


def load_speech_decoder_weights_mlx(decoder_path: Path) -> dict[str, Any]:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "Audex speech decoder preflight requires mlx in the active runtime."
        ) from exc

    config = load_speech_decoder_config(decoder_path)
    required = expected_speech_decoder_weight_keys(config)
    shard_path = decoder_path / "model.safetensors"
    if not shard_path.is_file():
        raise FileNotFoundError(f"Missing decoder safetensors: {shard_path}")

    shard = mx.load(str(shard_path), stream=mx.cpu)
    missing = tuple(key for key in required if key not in shard)
    if missing:
        raise FileNotFoundError(
            "Missing Audex speech decoder tensors: " + ", ".join(missing)
        )

    weights = {key: shard[key] for key in required}
    mx.eval(*weights.values())
    return weights


def embed_speech_token_frames_mlx(
    token_frames: Sequence[Sequence[int]],
    weights: dict[str, Any],
    config: AudexSpeechDecoderConfig,
) -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("Audex speech decoder embedding requires mlx.") from exc

    frames = _normalize_token_frames(token_frames, config)
    indices = mx.array(frames, dtype=mx.int32)[None, :, None]
    levels = mx.array(config.codebook_levels, dtype=mx.int32)
    basis_values: list[int] = []
    value = 1
    for level in config.codebook_levels:
        basis_values.append(value)
        value *= int(level)
    basis = mx.array(basis_values, dtype=mx.int32)

    level_indices = (indices // basis) % levels
    codes = level_indices.astype(
        weights["audex_speech_token_embedder.project_out.weight"].dtype
    )
    levels_float = levels.astype(codes.dtype)
    codes = codes * (2.0 / (levels_float - 1.0)) - 1.0
    return _linear(
        mx,
        codes,
        weights["audex_speech_token_embedder.project_out.weight"],
        weights["audex_speech_token_embedder.project_out.bias"],
    )


def decode_speech_token_frames_mlx(
    token_frames: Sequence[Sequence[int]],
    weights: dict[str, Any],
    config: AudexSpeechDecoderConfig,
) -> Any:
    """Decode Audex speech-codec frames to waveform samples with MLX."""

    vq_emb = embed_speech_token_frames_mlx(token_frames, weights, config)
    return decode_speech_embeddings_mlx(vq_emb, weights, config)


def decode_speech_embeddings_mlx(
    vq_emb: Any,
    weights: dict[str, Any],
    config: AudexSpeechDecoderConfig,
) -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("Audex speech decoder execution requires mlx.") from exc

    x = _linear(mx, vq_emb, weights["module.fc_post_a.weight"])
    x = _apply_lookahead(mx, x, weights, config)
    return _decode_projected_frames_mlx(mx, x, weights, config)


def decode_speech_embeddings_cached_mlx(
    vq_emb: Any,
    weights: dict[str, Any],
    config: AudexSpeechDecoderConfig,
    cache: AudexSpeechDecoderCache,
    *,
    lookahead_vq_emb: Any | None = None,
) -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("Audex speech decoder execution requires mlx.") from exc

    x = _linear(mx, vq_emb, weights["module.fc_post_a.weight"])
    if config.lookahead_steps > 0:
        if lookahead_vq_emb is None:
            lookahead_vq_emb = mx.zeros(
                (int(vq_emb.shape[0]), config.lookahead_steps, int(vq_emb.shape[-1])),
                dtype=vq_emb.dtype,
            )
        if int(lookahead_vq_emb.shape[1]) != config.lookahead_steps:
            raise ValueError(
                "lookahead_vq_emb must have "
                f"{config.lookahead_steps} frames, got {int(lookahead_vq_emb.shape[1])}"
            )
        lookahead_x = _linear(mx, lookahead_vq_emb, weights["module.fc_post_a.weight"])
        x = _apply_lookahead_window(mx, x, lookahead_x, weights, config)
    return _decode_projected_frames_mlx(mx, x, weights, config, cache=cache)


def _decode_projected_frames_mlx(
    mx: Any,
    x: Any,
    weights: dict[str, Any],
    config: AudexSpeechDecoderConfig,
    *,
    cache: AudexSpeechDecoderCache | None = None,
) -> Any:
    for layer_idx in range(config.depth):
        x = _decoder_layer(mx, x, weights, config, layer_idx, cache=cache)
    if cache is not None:
        cache.advance(int(x.shape[1]))
    x = _rms_norm(mx, x, weights["module.backbone.final_layer_norm.weight"])
    x = mx.tanh(_linear(mx, x, weights["module.head.proj.weight"]))
    x = mx.reshape(x, (int(x.shape[0]), 1, -1))[0, 0]
    mx.eval(x)
    return x


def run_speech_decoder_smoke(
    *,
    decoder_path: Path,
    token_frames: Sequence[Sequence[int]] | None = None,
) -> SpeechDecoderSmokeResult:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "Audex speech decoder preflight requires mlx in the active runtime."
        ) from exc

    mx.set_default_device(mx.gpu)
    config = load_speech_decoder_config(decoder_path)
    frames = token_frames or tuple((idx,) for idx in range(config.lookahead_steps + 4))
    weights = load_speech_decoder_weights_mlx(decoder_path)
    vq_emb = embed_speech_token_frames_mlx(frames, weights, config)
    wav = decode_speech_token_frames_mlx(frames, weights, config)
    finite = bool(mx.all(mx.isfinite(wav)).item())
    peak_abs = float(mx.max(mx.abs(wav)).item())
    return SpeechDecoderSmokeResult(
        backend="mlx",
        device=str(mx.default_device()),
        frame_count=len(frames),
        input_shape=(len(frames), 1),
        vq_embedding_shape=tuple(int(part) for part in vq_emb.shape),
        waveform_shape=tuple(int(part) for part in wav.shape),
        waveform_dtype=str(wav.dtype),
        sample_rate=config.sample_rate,
        hop_length=config.hop_length,
        lookahead_steps=config.lookahead_steps,
        finite=finite,
        peak_abs=peak_abs,
    )


def _decoder_layer(
    mx: Any,
    hidden_states: Any,
    weights: dict[str, Any],
    config: AudexSpeechDecoderConfig,
    layer_idx: int,
    *,
    cache: AudexSpeechDecoderCache | None = None,
) -> Any:
    prefix = f"module.backbone.transformers.{layer_idx}."
    x = hidden_states + _self_attention(
        mx,
        _rms_norm(mx, hidden_states, weights[f"{prefix}att_norm.weight"]),
        weights,
        config,
        prefix,
        layer_idx,
        cache=cache,
    )
    return x + _mlp(
        mx, _rms_norm(mx, x, weights[f"{prefix}ffn_norm.weight"]), weights, prefix
    )


def _self_attention(
    mx: Any,
    hidden_states: Any,
    weights: dict[str, Any],
    config: AudexSpeechDecoderConfig,
    prefix: str,
    layer_idx: int,
    *,
    cache: AudexSpeechDecoderCache | None = None,
) -> Any:
    batch, sequence, _ = hidden_states.shape
    input_positions = (
        cache.input_positions(mx, int(sequence)) if cache is not None else None
    )
    qkv = _linear(mx, hidden_states, weights[f"{prefix}att.c_attn.weight"])
    qkv = mx.reshape(
        qkv,
        (batch, sequence, 3, config.heads, config.head_dim),
    )
    qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
    query, key, value = qkv[0], qkv[1], qkv[2]
    query = _apply_rope(mx, query, input_positions=input_positions)
    key = _apply_rope(mx, key, input_positions=input_positions)
    if cache is None:
        mask = "causal" if int(sequence) > 1 else None
    else:
        key, value = cache.update(mx, layer_idx, key, value)
        key_positions = mx.arange(int(key.shape[2])).reshape(1, 1, 1, -1)
        valid = key_positions <= mx.reshape(
            input_positions, (batch, 1, int(sequence), 1)
        )
        fill = mx.array(mx.finfo(query.dtype).min, query.dtype)
        mask = mx.where(valid, mx.array(0, query.dtype), fill)
    attended = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=config.head_dim**-0.5,
        mask=mask,
    )
    attended = mx.reshape(
        mx.transpose(attended, (0, 2, 1, 3)),
        (batch, sequence, config.hidden_dim),
    )
    return _linear(mx, attended, weights[f"{prefix}att.c_proj.weight"])


def _apply_rope(mx: Any, x: Any, *, input_positions: Any | None = None) -> Any:
    *prefix, sequence, dim = x.shape
    half_dim = int(dim) // 2
    theta = 1.0 / (
        10_000 ** (mx.arange(0, int(dim), 2, dtype=mx.float32)[:half_dim] / int(dim))
    )
    positions = (
        mx.arange(int(sequence), dtype=mx.float32)
        if input_positions is None
        else input_positions[0].astype(mx.float32)
    )
    angles = positions[:, None] * theta[None, :]
    cos = mx.cos(angles)
    sin = mx.sin(angles)
    x_float = x.astype(mx.float32)
    pairs = mx.reshape(x_float, (*prefix, int(sequence), half_dim, 2))
    cos = mx.reshape(cos, (1, 1, int(sequence), half_dim))
    sin = mx.reshape(sin, (1, 1, int(sequence), half_dim))
    first = pairs[..., 0] * cos - pairs[..., 1] * sin
    second = pairs[..., 1] * cos + pairs[..., 0] * sin
    rotated = mx.stack((first, second), axis=-1)
    return mx.reshape(rotated, x.shape).astype(x.dtype)


def _mlp(mx: Any, x: Any, weights: dict[str, Any], prefix: str) -> Any:
    x = _linear(mx, x, weights[f"{prefix}mlp.fc1.weight"])
    x = x * mx.sigmoid(x)
    return _linear(mx, x, weights[f"{prefix}mlp.fc2.weight"])


def _apply_lookahead(
    mx: Any,
    x: Any,
    weights: dict[str, Any],
    config: AudexSpeechDecoderConfig,
) -> Any:
    if config.lookahead_steps <= 0:
        return x
    padded = mx.pad(x, [(0, 0), (0, config.lookahead_steps), (0, 0)])
    conv = _conv1d_from_checkpoint_weight(
        mx,
        padded,
        weights["module.lookahead_conv.weight"],
        groups=config.hidden_dim,
    )
    conv = conv * mx.sigmoid(conv)
    projected = _conv1d_from_checkpoint_weight(
        mx,
        conv,
        weights["module.lookahead_proj.weight"],
    )
    return x + projected


def _apply_lookahead_window(
    mx: Any,
    x: Any,
    lookahead_x: Any,
    weights: dict[str, Any],
    config: AudexSpeechDecoderConfig,
) -> Any:
    if config.lookahead_steps <= 0:
        return x
    if int(lookahead_x.shape[1]) != config.lookahead_steps:
        raise ValueError(
            f"lookahead_x must have {config.lookahead_steps} frames, "
            f"got {int(lookahead_x.shape[1])}"
        )
    window = mx.concatenate((x, lookahead_x), axis=1)
    conv = _conv1d_from_checkpoint_weight(
        mx,
        window,
        weights["module.lookahead_conv.weight"],
        groups=config.hidden_dim,
    )
    conv = conv * mx.sigmoid(conv)
    projected = _conv1d_from_checkpoint_weight(
        mx,
        conv,
        weights["module.lookahead_proj.weight"],
    )
    return x + projected


def _conv1d_from_checkpoint_weight(
    mx: Any,
    x: Any,
    weight: Any,
    *,
    groups: int = 1,
) -> Any:
    return mx.conv1d(x, mx.transpose(weight, (0, 2, 1)), groups=groups)


def _linear(mx: Any, x: Any, weight: Any, bias: Any | None = None) -> Any:
    output = x @ mx.transpose(weight)
    if bias is not None:
        output = output + bias
    return output


def _rms_norm(mx: Any, x: Any, weight: Any, eps: float = 1e-6) -> Any:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps) * weight


def _normalize_token_frames(
    token_frames: Sequence[Sequence[int]],
    config: AudexSpeechDecoderConfig,
) -> tuple[int, ...]:
    frames: list[int] = []
    for frame in token_frames:
        if len(frame) != 1:
            raise ValueError(f"Audex decoder expects one code per frame, got {frame}")
        code = int(frame[0])
        if code < 0 or code >= config.codebook_size:
            raise ValueError(
                f"Speech codec frame {code} outside codebook_size={config.codebook_size}"
            )
        frames.append(code)
    if not frames:
        raise ValueError("At least one speech codec frame is required")
    return tuple(frames)


def _required_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if value is None:
        raise ValueError(f"Missing decoder config key: {key}")
    return int(value)
