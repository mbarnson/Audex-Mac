"""Audex NV-Whisper/Qwen2Audio encoder execution on MLX."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_projector import (
    load_audio_projector_config,
    load_audio_projector_weights_mlx,
    project_audio_hidden_states_mlx,
)

ENCODER_WEIGHT_PREFIX = "audio_encoder."
ENCODER_STEM_KEYS = (
    "conv1.weight",
    "conv1.bias",
    "conv2.weight",
    "conv2.bias",
    "embed_positions.weight",
    "layer_norm.weight",
    "layer_norm.bias",
)
ENCODER_LAYER_KEYS = (
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.out_proj.weight",
    "self_attn.out_proj.bias",
    "self_attn_layer_norm.weight",
    "self_attn_layer_norm.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "final_layer_norm.weight",
    "final_layer_norm.bias",
)


@dataclass(frozen=True, slots=True)
class AudexAudioEncoderConfig:
    d_model: int
    encoder_attention_heads: int
    encoder_ffn_dim: int
    encoder_layers: int
    max_source_positions: int
    num_mel_bins: int
    activation_function: str
    scale_embedding: bool
    layer_norm_eps: float = 1e-5

    @property
    def expected_feature_frames(self) -> int:
        return self.max_source_positions * 2

    @property
    def head_dim(self) -> int:
        return self.d_model // self.encoder_attention_heads


@dataclass(frozen=True, slots=True)
class AudexAudioEncoderSmokeResult:
    backend: str
    device: str
    input_shape: tuple[int, ...]
    encoder_shape: tuple[int, ...]
    projected_shape: tuple[int, ...]
    encoder_dtype: str
    projected_dtype: str
    encoder_layers: int


@dataclass(frozen=True, slots=True)
class AudexProjectedAudioResult:
    input_features: Any
    encoder_hidden: Any
    projected_embeddings: Any
    encoder_layers: int


def load_audio_encoder_config(model_path: Path) -> AudexAudioEncoderConfig:
    """Read Audex's Qwen2Audio/NV-Whisper encoder config."""

    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config.json must be a JSON object")
    audio_config = raw.get("audio_config", {})
    if not isinstance(audio_config, dict):
        raise ValueError("config.json audio_config must be a JSON object")

    return AudexAudioEncoderConfig(
        d_model=_required_int(audio_config, "d_model"),
        encoder_attention_heads=_required_int(
            audio_config,
            "encoder_attention_heads",
        ),
        encoder_ffn_dim=_required_int(audio_config, "encoder_ffn_dim"),
        encoder_layers=_required_int(audio_config, "encoder_layers"),
        max_source_positions=_required_int(audio_config, "max_source_positions"),
        num_mel_bins=_required_int(audio_config, "num_mel_bins"),
        activation_function=str(audio_config.get("activation_function", "gelu")),
        scale_embedding=bool(audio_config.get("scale_embedding", False)),
    )


def expected_audio_encoder_weight_keys(
    config: AudexAudioEncoderConfig,
) -> tuple[str, ...]:
    layer_keys = tuple(
        f"layers.{layer_idx}.{suffix}"
        for layer_idx in range(config.encoder_layers)
        for suffix in ENCODER_LAYER_KEYS
    )
    return ENCODER_STEM_KEYS + layer_keys


def resolve_audio_encoder_shards(model_path: Path) -> dict[str, Path]:
    """Map each required audio_encoder tensor to its safetensors shard."""

    config = load_audio_encoder_config(model_path)
    required = expected_audio_encoder_weight_keys(config)
    raw = json.loads(
        (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = raw.get("weight_map", {}) if isinstance(raw, dict) else {}
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json weight_map must be an object")

    missing = [
        f"{ENCODER_WEIGHT_PREFIX}{key}"
        for key in required
        if f"{ENCODER_WEIGHT_PREFIX}{key}" not in weight_map
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Audex audio encoder tensors: " + ", ".join(missing)
        )

    return {
        key: model_path / str(weight_map[f"{ENCODER_WEIGHT_PREFIX}{key}"])
        for key in required
    }


def load_audio_encoder_weights_mlx(model_path: Path) -> dict[str, Any]:
    """Load Audex's required audio_encoder tensors as MLX arrays."""

    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "Audex audio encoder preflight requires mlx in the active runtime. "
            "Run through ./start.sh after vLLM Metal dependencies are ready."
        ) from exc

    weights: dict[str, Any] = {}
    keys_by_shard: dict[Path, list[str]] = {}
    for key, shard_path in resolve_audio_encoder_shards(model_path).items():
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
        keys_by_shard.setdefault(shard_path, []).append(key)

    for shard_path, stripped_keys in keys_by_shard.items():
        shard = mx.load(str(shard_path), stream=mx.cpu)
        for stripped_key in stripped_keys:
            full_key = f"{ENCODER_WEIGHT_PREFIX}{stripped_key}"
            array = shard[full_key]
            mx.eval(array)
            weights[stripped_key] = array
    return weights


def encode_audio_features_mlx(
    input_features: Any,
    weights: dict[str, Any],
    config: AudexAudioEncoderConfig,
) -> Any:
    """Run Audex's Qwen2Audio encoder over prepared Whisper input features."""

    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "Audex audio encoder execution requires mlx in the active runtime."
        ) from exc

    _validate_encoder_inputs(input_features, config)
    _validate_encoder_weights(weights, config)

    x = mx.transpose(input_features, (0, 2, 1))
    x = _conv1d_from_checkpoint_weight(
        mx,
        x,
        weights["conv1.weight"],
        weights["conv1.bias"],
    )
    x = _gelu(mx, x)
    x = _conv1d_from_checkpoint_weight(
        mx,
        x,
        weights["conv2.weight"],
        weights["conv2.bias"],
        stride=2,
    )
    x = _gelu(mx, x)

    if config.scale_embedding:
        x = x * math.sqrt(config.d_model)
    sequence = int(x.shape[1])
    x = x + weights["embed_positions.weight"][:sequence]

    for layer_idx in range(config.encoder_layers):
        x = _encoder_layer(mx, x, weights, config, layer_idx)

    batch, sequence, hidden = x.shape
    if int(sequence) % 2 != 0:
        raise ValueError(f"Expected even encoder sequence length, got {sequence}")
    x = mx.reshape(x, (batch, int(sequence) // 2, 2, hidden))
    x = mx.mean(x, axis=2)
    x = _layer_norm(
        mx,
        x,
        weights["layer_norm.weight"],
        weights["layer_norm.bias"],
        config.layer_norm_eps,
    )
    mx.eval(x)
    return x


def run_audio_encoder_smoke(
    model_path: Path,
    *,
    num_clips: int = 1,
) -> AudexAudioEncoderSmokeResult:
    """Run real Audex encoder and projector weights on one zero feature batch."""

    result = project_zero_audio_features_mlx(model_path, num_clips=num_clips)
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "Audex audio encoder preflight requires mlx in the active runtime. "
            "Run through ./start.sh after vLLM Metal dependencies are ready."
        ) from exc

    return AudexAudioEncoderSmokeResult(
        backend="mlx",
        device=str(mx.default_device()),
        input_shape=tuple(int(part) for part in result.input_features.shape),
        encoder_shape=tuple(int(part) for part in result.encoder_hidden.shape),
        projected_shape=tuple(int(part) for part in result.projected_embeddings.shape),
        encoder_dtype=str(result.encoder_hidden.dtype),
        projected_dtype=str(result.projected_embeddings.dtype),
        encoder_layers=result.encoder_layers,
    )


def project_zero_audio_features_mlx(
    model_path: Path,
    *,
    num_clips: int = 1,
) -> AudexProjectedAudioResult:
    """Project one or more zero Audex feature clips into LLM embedding space."""

    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "Audex audio encoder preflight requires mlx in the active runtime. "
            "Run through ./start.sh after vLLM Metal dependencies are ready."
        ) from exc

    mx.set_default_device(mx.gpu)
    config = load_audio_encoder_config(model_path)
    encoder_weights = load_audio_encoder_weights_mlx(model_path)
    features = mx.zeros(
        (num_clips, config.num_mel_bins, config.expected_feature_frames),
        dtype=encoder_weights["conv1.weight"].dtype,
    )
    encoder_hidden = encode_audio_features_mlx(features, encoder_weights, config)
    projector_config = load_audio_projector_config(model_path)
    projector_weights = load_audio_projector_weights_mlx(model_path)
    projected = project_audio_hidden_states_mlx(
        encoder_hidden,
        projector_weights,
        projector_config,
    )
    return AudexProjectedAudioResult(
        input_features=features,
        encoder_hidden=encoder_hidden,
        projected_embeddings=projected,
        encoder_layers=config.encoder_layers,
    )


def _encoder_layer(
    mx: Any,
    hidden_states: Any,
    weights: dict[str, Any],
    config: AudexAudioEncoderConfig,
    layer_idx: int,
) -> Any:
    prefix = f"layers.{layer_idx}."
    residual = hidden_states
    x = _layer_norm(
        mx,
        hidden_states,
        weights[f"{prefix}self_attn_layer_norm.weight"],
        weights[f"{prefix}self_attn_layer_norm.bias"],
        config.layer_norm_eps,
    )
    x = _self_attention(mx, x, weights, config, prefix)
    x = residual + x

    residual = x
    x = _layer_norm(
        mx,
        x,
        weights[f"{prefix}final_layer_norm.weight"],
        weights[f"{prefix}final_layer_norm.bias"],
        config.layer_norm_eps,
    )
    x = _linear(mx, x, weights[f"{prefix}fc1.weight"], weights[f"{prefix}fc1.bias"])
    if config.activation_function != "gelu":
        raise ValueError(
            f"Unsupported Audex audio encoder activation: {config.activation_function}"
        )
    x = _gelu(mx, x)
    x = _linear(mx, x, weights[f"{prefix}fc2.weight"], weights[f"{prefix}fc2.bias"])
    return residual + x


def _self_attention(
    mx: Any,
    hidden_states: Any,
    weights: dict[str, Any],
    config: AudexAudioEncoderConfig,
    prefix: str,
) -> Any:
    batch, sequence, _ = hidden_states.shape
    query = _linear(
        mx,
        hidden_states,
        weights[f"{prefix}self_attn.q_proj.weight"],
        weights[f"{prefix}self_attn.q_proj.bias"],
    )
    key = _linear(
        mx,
        hidden_states,
        weights[f"{prefix}self_attn.k_proj.weight"],
        None,
    )
    value = _linear(
        mx,
        hidden_states,
        weights[f"{prefix}self_attn.v_proj.weight"],
        weights[f"{prefix}self_attn.v_proj.bias"],
    )
    query = _split_heads(mx, query, batch, sequence, config) * (config.head_dim**-0.5)
    key = _split_heads(mx, key, batch, sequence, config)
    value = _split_heads(mx, value, batch, sequence, config)
    attended = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=1.0,
    )
    attended = mx.transpose(attended, (0, 2, 1, 3))
    attended = mx.reshape(attended, (batch, sequence, config.d_model))
    return _linear(
        mx,
        attended,
        weights[f"{prefix}self_attn.out_proj.weight"],
        weights[f"{prefix}self_attn.out_proj.bias"],
    )


def _split_heads(
    mx: Any,
    x: Any,
    batch: int,
    sequence: int,
    config: AudexAudioEncoderConfig,
) -> Any:
    x = mx.reshape(
        x,
        (batch, sequence, config.encoder_attention_heads, config.head_dim),
    )
    return mx.transpose(x, (0, 2, 1, 3))


def _linear(mx: Any, x: Any, weight: Any, bias: Any | None) -> Any:
    out = mx.matmul(x, mx.transpose(weight))
    if bias is not None:
        out = out + bias
    return out


def _conv1d_from_checkpoint_weight(
    mx: Any,
    x: Any,
    weight: Any,
    bias: Any,
    *,
    stride: int = 1,
) -> Any:
    weight = mx.transpose(weight, (0, 2, 1))
    return mx.conv1d(x, weight, stride=stride, padding=1) + bias


def _layer_norm(mx: Any, x: Any, weight: Any, bias: Any, eps: float) -> Any:
    mean = mx.mean(x.astype(mx.float32), axis=-1, keepdims=True)
    centered = x.astype(mx.float32) - mean
    variance = mx.mean(mx.square(centered), axis=-1, keepdims=True)
    x = centered * mx.rsqrt(variance + eps)
    return x.astype(weight.dtype) * weight + bias


def _gelu(mx: Any, x: Any) -> Any:
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def _validate_encoder_inputs(
    input_features: Any,
    config: AudexAudioEncoderConfig,
) -> None:
    shape = tuple(int(part) for part in input_features.shape)
    valid_frames = (
        len(shape) == 3
        and shape[1] == config.num_mel_bins
        and 4 <= shape[2] <= config.expected_feature_frames
        and shape[2] % 4 == 0
    )
    if not valid_frames:
        raise ValueError(
            "Expected Audex input features shaped "
            f"(clips, {config.num_mel_bins}, frames), where frames is a "
            f"multiple of 4 from 4 through {config.expected_feature_frames}; "
            f"got {shape}"
        )


def _validate_encoder_weights(
    weights: dict[str, Any],
    config: AudexAudioEncoderConfig,
) -> None:
    missing = [
        key for key in expected_audio_encoder_weight_keys(config) if key not in weights
    ]
    if missing:
        raise ValueError("Missing loaded audio encoder weights: " + ", ".join(missing))

    expected_shapes = {
        "conv1.weight": (config.d_model, config.num_mel_bins, 3),
        "conv1.bias": (config.d_model,),
        "conv2.weight": (config.d_model, config.d_model, 3),
        "conv2.bias": (config.d_model,),
        "embed_positions.weight": (config.max_source_positions, config.d_model),
        "layer_norm.weight": (config.d_model,),
        "layer_norm.bias": (config.d_model,),
    }
    for layer_idx in range(config.encoder_layers):
        prefix = f"layers.{layer_idx}."
        expected_shapes.update(
            {
                f"{prefix}self_attn.k_proj.weight": (config.d_model, config.d_model),
                f"{prefix}self_attn.v_proj.weight": (config.d_model, config.d_model),
                f"{prefix}self_attn.v_proj.bias": (config.d_model,),
                f"{prefix}self_attn.q_proj.weight": (config.d_model, config.d_model),
                f"{prefix}self_attn.q_proj.bias": (config.d_model,),
                f"{prefix}self_attn.out_proj.weight": (
                    config.d_model,
                    config.d_model,
                ),
                f"{prefix}self_attn.out_proj.bias": (config.d_model,),
                f"{prefix}self_attn_layer_norm.weight": (config.d_model,),
                f"{prefix}self_attn_layer_norm.bias": (config.d_model,),
                f"{prefix}fc1.weight": (config.encoder_ffn_dim, config.d_model),
                f"{prefix}fc1.bias": (config.encoder_ffn_dim,),
                f"{prefix}fc2.weight": (config.d_model, config.encoder_ffn_dim),
                f"{prefix}fc2.bias": (config.d_model,),
                f"{prefix}final_layer_norm.weight": (config.d_model,),
                f"{prefix}final_layer_norm.bias": (config.d_model,),
            }
        )

    for key, expected in expected_shapes.items():
        shape = tuple(int(part) for part in weights[key].shape)
        if shape != expected:
            raise ValueError(f"Expected {key} shape {expected}, got {shape}")


def _required_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if value is None:
        raise KeyError(f"config.json audio_config missing {key}")
    return int(value)
