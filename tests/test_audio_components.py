from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.audio_components import preflight_audio_components

pytestmark = pytest.mark.fast


def test_audio_components_preflight_accepts_audex_native_audio_metadata(
    tmp_path: Path,
) -> None:
    write_audio_component_metadata(tmp_path)

    result = preflight_audio_components(tmp_path)

    assert result.ready is True
    assert result.architecture == ("NemotronDenseAudexForConditionalGeneration",)
    assert result.model_type == "nemotron_dense_audex"
    assert result.audio_model_type == "NV-Whisper"
    assert result.audio_encoder_layers == 32
    assert result.audio_encoder_hidden_size == 1280
    assert result.audio_mel_bins == 128
    assert result.audio_max_source_positions == 1500
    assert result.sound_token_id == 29
    assert result.sound_embeddings_per_clip == 750
    assert result.audio_encoder_weight_count == 1
    assert result.audio_weight_shards == ("model-00002-of-00002.safetensors",)


def test_audio_components_preflight_requires_projector_weights(
    tmp_path: Path,
) -> None:
    write_audio_component_metadata(tmp_path, include_projector=False)

    result = preflight_audio_components(tmp_path)

    assert result.ready is False
    assert "audio_projector.fc1.weight" in result.missing_items
    assert "audio_projector.fc2.weight" in result.missing_items
    assert "audio_projector.norm.weight" in result.missing_items


def test_audio_components_preflight_rejects_wrong_sound_token_contract(
    tmp_path: Path,
) -> None:
    config = audex_audio_config()
    config["sound_embedding_size"] = 749
    write_audio_component_metadata(tmp_path, config=config)

    result = preflight_audio_components(tmp_path)

    assert result.ready is False
    assert "config.json sound_embedding_size=750" in result.missing_items


def write_audio_component_metadata(
    model_path: Path,
    *,
    config: dict | None = None,
    include_projector: bool = True,
) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "config.json").write_text(
        json.dumps(config or audex_audio_config()),
        encoding="utf-8",
    )
    weight_map = {
        "audio_encoder.conv1.weight": "model-00002-of-00002.safetensors",
    }
    if include_projector:
        weight_map.update(
            {
                "audio_projector.norm.weight": "model-00002-of-00002.safetensors",
                "audio_projector.fc1.weight": "model-00002-of-00002.safetensors",
                "audio_projector.fc2.weight": "model-00002-of-00002.safetensors",
            }
        )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


def audex_audio_config() -> dict:
    return {
        "architectures": ["NemotronDenseAudexForConditionalGeneration"],
        "model_type": "nemotron_dense_audex",
        "audio_model_type": "NV-Whisper",
        "audio_encoder_hidden_size": 1280,
        "audio_config": {
            "encoder_layers": 32,
            "num_mel_bins": 128,
            "max_source_positions": 1500,
        },
        "sound_token": "<so_embedding>",
        "sound_token_id": 29,
        "sound_start_token": "<so_start>",
        "sound_end_token": "<so_end>",
        "sound_target_rate": 16000,
        "sound_clip_duration": 30.0,
        "sound_embedding_size": 750,
    }
