"""Audex full-checkpoint audio component validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_contract import (
    DEFAULT_SOUND_CLIP_DURATION,
    DEFAULT_SOUND_EMBEDDING_SIZE,
    SAMPLE_RATE,
    SOUND_END_TOKEN,
    SOUND_START_TOKEN,
    SOUND_TOKEN,
)
from .config_values import optional_int, optional_str

REQUIRED_AUDIO_PROJECTOR_KEYS = (
    "audio_projector.norm.weight",
    "audio_projector.fc1.weight",
    "audio_projector.fc2.weight",
)


@dataclass(frozen=True, slots=True)
class AudioComponentPreflight:
    model_path: Path
    ready: bool
    architecture: tuple[str, ...]
    model_type: str | None
    audio_model_type: str | None
    audio_encoder_layers: int | None
    audio_encoder_hidden_size: int | None
    audio_mel_bins: int | None
    audio_max_source_positions: int | None
    sound_token_id: int | None
    sound_embeddings_per_clip: int | None
    audio_encoder_weight_count: int
    audio_weight_shards: tuple[str, ...]
    missing_items: tuple[str, ...]


def preflight_audio_components(model_path: Path) -> AudioComponentPreflight:
    """Validate Audex native audio-input config and tensor index metadata."""

    missing: list[str] = []
    config = _read_json(model_path / "config.json", missing)
    index = _read_json(model_path / "model.safetensors.index.json", missing)
    weight_map = index.get("weight_map", {}) if isinstance(index, dict) else {}
    if not isinstance(weight_map, dict):
        missing.append("model.safetensors.index.json weight_map")
        weight_map = {}

    audio_config = config.get("audio_config", {}) if isinstance(config, dict) else {}
    if not isinstance(audio_config, dict):
        missing.append("config.json audio_config")
        audio_config = {}

    architecture = tuple(str(item) for item in config.get("architectures", ()))
    model_type = optional_str(config.get("model_type"))
    audio_model_type = optional_str(config.get("audio_model_type"))
    encoder_layers = optional_int(audio_config.get("encoder_layers"))
    encoder_hidden_size = optional_int(config.get("audio_encoder_hidden_size"))
    mel_bins = optional_int(audio_config.get("num_mel_bins"))
    max_source_positions = optional_int(audio_config.get("max_source_positions"))
    sound_token_id = optional_int(config.get("sound_token_id"))
    embeddings_per_clip = optional_int(config.get("sound_embedding_size"))

    if not any("Audex" in item for item in architecture):
        missing.append("config.json architectures Audex entry")
    if model_type is None or "audex" not in model_type:
        missing.append("config.json model_type audex")
    if audio_model_type != "NV-Whisper":
        missing.append("config.json audio_model_type=NV-Whisper")
    if encoder_layers is None or encoder_layers <= 0:
        missing.append("config.json audio_config.encoder_layers")
    if encoder_hidden_size is None or encoder_hidden_size <= 0:
        missing.append("config.json audio_encoder_hidden_size")
    if mel_bins != 128:
        missing.append("config.json audio_config.num_mel_bins=128")
    if max_source_positions is None or max_source_positions <= 0:
        missing.append("config.json audio_config.max_source_positions")
    elif max_source_positions // 2 != DEFAULT_SOUND_EMBEDDING_SIZE:
        missing.append("audio encoder output token count=750")
    if embeddings_per_clip != DEFAULT_SOUND_EMBEDDING_SIZE:
        missing.append("config.json sound_embedding_size=750")

    _expect_config_value(
        config,
        "sound_target_rate",
        SAMPLE_RATE,
        missing,
    )
    _expect_config_value(
        config,
        "sound_clip_duration",
        DEFAULT_SOUND_CLIP_DURATION,
        missing,
    )
    _expect_config_value(config, "sound_token", SOUND_TOKEN, missing)
    _expect_config_value(config, "sound_start_token", SOUND_START_TOKEN, missing)
    _expect_config_value(config, "sound_end_token", SOUND_END_TOKEN, missing)

    audio_encoder_keys = tuple(
        key for key in weight_map if key.startswith("audio_encoder.")
    )
    if not audio_encoder_keys:
        missing.append("audio_encoder.* weights")

    for key in REQUIRED_AUDIO_PROJECTOR_KEYS:
        if key not in weight_map:
            missing.append(key)

    audio_weight_shards = tuple(
        sorted(
            {
                str(shard)
                for key, shard in weight_map.items()
                if key.startswith("audio_encoder.")
                or key.startswith("audio_projector.")
            }
        )
    )
    if not audio_weight_shards:
        missing.append("audio component safetensors shards")

    return AudioComponentPreflight(
        model_path=model_path,
        ready=not missing,
        architecture=architecture,
        model_type=model_type,
        audio_model_type=audio_model_type,
        audio_encoder_layers=encoder_layers,
        audio_encoder_hidden_size=encoder_hidden_size,
        audio_mel_bins=mel_bins,
        audio_max_source_positions=max_source_positions,
        sound_token_id=sound_token_id,
        sound_embeddings_per_clip=embeddings_per_clip,
        audio_encoder_weight_count=len(audio_encoder_keys),
        audio_weight_shards=audio_weight_shards,
        missing_items=tuple(dict.fromkeys(missing)),
    )


def _read_json(path: Path, missing: list[str]) -> dict[str, Any]:
    if not path.is_file():
        missing.append(str(path.name))
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        missing.append(f"{path.name} valid JSON")
        return {}
    if not isinstance(raw, dict):
        missing.append(f"{path.name} JSON object")
        return {}
    return raw


def _expect_config_value(
    config: dict[str, Any],
    name: str,
    expected: Any,
    missing: list[str],
) -> None:
    if config.get(name) != expected:
        missing.append(f"config.json {name}={expected}")
