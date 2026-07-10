"""Audex audio projector execution on MLX."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_contract import DEFAULT_SOUND_EMBEDDING_SIZE

PROJECTOR_WEIGHT_KEYS = (
    "audio_projector.norm.weight",
    "audio_projector.fc1.weight",
    "audio_projector.fc2.weight",
)

STRIPPED_PROJECTOR_WEIGHT_KEYS = (
    "norm.weight",
    "fc1.weight",
    "fc2.weight",
)


@dataclass(frozen=True, slots=True)
class AudexProjectorConfig:
    audio_hidden_size: int
    intermediate_size: int
    text_hidden_size: int
    sound_embeddings_per_clip: int
    norm_eps: float
    activation: str


@dataclass(frozen=True, slots=True)
class AudexProjectorSmokeResult:
    backend: str
    device: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_dtype: str
    weight_dtype: str
    activation: str


def load_audio_projector_config(model_path: Path) -> AudexProjectorConfig:
    """Read the Audex projector dimensions from the full checkpoint config."""

    config_path = model_path / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config.json must be a JSON object")

    return AudexProjectorConfig(
        audio_hidden_size=_required_int(raw, "audio_encoder_hidden_size"),
        intermediate_size=_required_int(raw, "audio_projector_intermediate_size"),
        text_hidden_size=_required_int(raw, "hidden_size"),
        sound_embeddings_per_clip=_required_int(raw, "sound_embedding_size"),
        norm_eps=float(raw.get("audio_projector_norm_eps", 1e-5)),
        activation=str(raw.get("audio_projector_activation", "relu2")),
    )


def resolve_audio_projector_shards(model_path: Path) -> dict[str, Path]:
    """Map each required projector tensor to its local safetensors shard."""

    index_path = model_path / "model.safetensors.index.json"
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = raw.get("weight_map", {}) if isinstance(raw, dict) else {}
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json weight_map must be an object")

    missing = [key for key in PROJECTOR_WEIGHT_KEYS if key not in weight_map]
    if missing:
        raise FileNotFoundError(
            "Missing Audex audio projector tensors: " + ", ".join(missing)
        )

    return {key: model_path / str(weight_map[key]) for key in PROJECTOR_WEIGHT_KEYS}


def load_audio_projector_weights_mlx(
    model_path: Path,
    *,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Load Audex's required audio_projector tensors as MLX arrays."""

    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "Audex audio projector preflight requires mlx in the active runtime. "
            "Run through ./start.sh after vLLM Metal dependencies are ready."
        ) from exc

    target_dtype = _mlx_dtype(mx, dtype)
    weights: dict[str, Any] = {}
    keys_by_shard: dict[Path, list[str]] = {}
    for key, shard_path in resolve_audio_projector_shards(model_path).items():
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
        keys_by_shard.setdefault(shard_path, []).append(key)

    for shard_path, keys in keys_by_shard.items():
        # MLX safetensors Load nodes cannot eval directly on GPU in this
        # runtime. Materialize from disk on CPU, then let projector math run on
        # the GPU default device.
        shard = mx.load(str(shard_path), stream=mx.cpu)
        for key in keys:
            array = shard[key]
            mx.eval(array)
            weights[strip_audio_projector_prefix(key)] = array

    if dtype != "bfloat16":
        weights = {key: value.astype(target_dtype) for key, value in weights.items()}
    return weights


def project_audio_hidden_states_mlx(
    hidden_states: Any,
    weights: dict[str, Any],
    config: AudexProjectorConfig,
) -> Any:
    """Apply Audex RMSNorm -> fc1 -> relu^2 -> fc2 projector in MLX."""

    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "Audex audio projector execution requires mlx in the active runtime."
        ) from exc

    _validate_projector_weights(weights, config)
    if config.activation != "relu2":
        raise ValueError(
            f"Unsupported Audex audio projector activation: {config.activation}"
        )

    x = hidden_states
    input_dtype = x.dtype
    variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
    x = x * mx.rsqrt(variance + config.norm_eps)
    x = x.astype(input_dtype) * weights["norm.weight"]
    x = mx.matmul(x, mx.transpose(weights["fc1.weight"]))
    x = mx.square(mx.maximum(x, 0))
    x = mx.matmul(x, mx.transpose(weights["fc2.weight"]))
    mx.eval(x)
    return x


def run_audio_projector_smoke(
    model_path: Path,
    *,
    num_clips: int = 1,
) -> AudexProjectorSmokeResult:
    """Load real projector tensors and run one MLX shape-forcing projection."""

    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "Audex audio projector preflight requires mlx in the active runtime. "
            "Run through ./start.sh after vLLM Metal dependencies are ready."
        ) from exc

    config = load_audio_projector_config(model_path)
    mx.set_default_device(mx.gpu)
    if config.sound_embeddings_per_clip != DEFAULT_SOUND_EMBEDDING_SIZE:
        raise ValueError(
            "Audex projector smoke expects "
            f"{DEFAULT_SOUND_EMBEDDING_SIZE} sound embeddings per clip, got "
            f"{config.sound_embeddings_per_clip}"
        )

    weights = load_audio_projector_weights_mlx(model_path)
    hidden_states = mx.zeros(
        (
            num_clips,
            config.sound_embeddings_per_clip,
            config.audio_hidden_size,
        ),
        dtype=weights["fc1.weight"].dtype,
    )
    output = project_audio_hidden_states_mlx(hidden_states, weights, config)
    return AudexProjectorSmokeResult(
        backend="mlx",
        device=str(mx.default_device()),
        input_shape=tuple(int(part) for part in hidden_states.shape),
        output_shape=tuple(int(part) for part in output.shape),
        output_dtype=str(output.dtype),
        weight_dtype=str(weights["fc1.weight"].dtype),
        activation=config.activation,
    )


def strip_audio_projector_prefix(key: str) -> str:
    if not key.startswith("audio_projector."):
        raise ValueError(f"Expected audio_projector.* key, got {key}")
    stripped = key.removeprefix("audio_projector.")
    if stripped not in STRIPPED_PROJECTOR_WEIGHT_KEYS:
        raise ValueError(f"Unexpected Audex audio projector key: {key}")
    return stripped


def _validate_projector_weights(
    weights: dict[str, Any],
    config: AudexProjectorConfig,
) -> None:
    missing = [key for key in STRIPPED_PROJECTOR_WEIGHT_KEYS if key not in weights]
    if missing:
        raise ValueError("Missing loaded projector weights: " + ", ".join(missing))

    expected_shapes = {
        "norm.weight": (config.audio_hidden_size,),
        "fc1.weight": (config.intermediate_size, config.audio_hidden_size),
        "fc2.weight": (config.text_hidden_size, config.intermediate_size),
    }
    for key, expected in expected_shapes.items():
        shape = tuple(int(part) for part in weights[key].shape)
        if shape != expected:
            raise ValueError(f"Expected {key} shape {expected}, got {shape}")


def _required_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if value is None:
        raise KeyError(f"config.json missing {key}")
    return int(value)


def _mlx_dtype(mx: Any, dtype: str) -> Any:
    if dtype == "bfloat16":
        return mx.bfloat16
    if dtype == "float16":
        return mx.float16
    if dtype == "float32":
        return mx.float32
    raise ValueError(f"Unsupported MLX dtype: {dtype}")
