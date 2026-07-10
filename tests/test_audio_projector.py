from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from audex_mac.audio_projector import (
    AudexProjectorConfig,
    load_audio_projector_config,
    project_audio_hidden_states_mlx,
    resolve_audio_projector_shards,
    strip_audio_projector_prefix,
)

pytestmark = pytest.mark.fast


def test_load_audio_projector_config_reads_audex_dimensions(tmp_path: Path) -> None:
    write_projector_config(tmp_path)

    config = load_audio_projector_config(tmp_path)

    assert config.audio_hidden_size == 1280
    assert config.intermediate_size == 4096
    assert config.text_hidden_size == 2048
    assert config.sound_embeddings_per_clip == 750
    assert config.norm_eps == 1e-5
    assert config.activation == "relu2"


def test_resolve_audio_projector_shards_requires_all_projector_tensors(
    tmp_path: Path,
) -> None:
    write_projector_index(tmp_path, include_fc2=False)

    with pytest.raises(FileNotFoundError, match="audio_projector.fc2.weight"):
        resolve_audio_projector_shards(tmp_path)


def test_resolve_audio_projector_shards_maps_required_tensors(
    tmp_path: Path,
) -> None:
    write_projector_index(tmp_path)

    shards = resolve_audio_projector_shards(tmp_path)

    assert shards == {
        "audio_projector.norm.weight": tmp_path / "model-00002-of-00002.safetensors",
        "audio_projector.fc1.weight": tmp_path / "model-00002-of-00002.safetensors",
        "audio_projector.fc2.weight": tmp_path / "model-00002-of-00002.safetensors",
    }


def test_strip_audio_projector_prefix_rejects_non_projector_keys() -> None:
    with pytest.raises(ValueError, match="audio_projector"):
        strip_audio_projector_prefix("audio_encoder.layers.0.weight")


@pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="MLX is installed in the vLLM Metal runtime, not the fast test venv",
)
def test_project_audio_hidden_states_mlx_uses_expected_shape() -> None:
    import mlx.core as mx

    config = AudexProjectorConfig(
        audio_hidden_size=2,
        intermediate_size=2,
        text_hidden_size=2,
        sound_embeddings_per_clip=1,
        norm_eps=1e-5,
        activation="relu2",
    )
    weights = {
        "norm.weight": mx.ones((2,), dtype=mx.float32),
        "fc1.weight": mx.eye(2, dtype=mx.float32),
        "fc2.weight": mx.eye(2, dtype=mx.float32),
    }
    hidden = mx.array([[[3.0, 4.0]]], dtype=mx.float32)

    output = project_audio_hidden_states_mlx(hidden, weights, config)

    assert tuple(int(part) for part in output.shape) == (1, 1, 2)


def write_projector_config(model_path: Path) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "audio_encoder_hidden_size": 1280,
                "audio_projector_activation": "relu2",
                "audio_projector_intermediate_size": 4096,
                "audio_projector_norm_eps": 1e-5,
                "hidden_size": 2048,
                "sound_embedding_size": 750,
            }
        ),
        encoding="utf-8",
    )


def write_projector_index(
    model_path: Path,
    *,
    include_fc2: bool = True,
) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    weight_map = {
        "audio_projector.norm.weight": "model-00002-of-00002.safetensors",
        "audio_projector.fc1.weight": "model-00002-of-00002.safetensors",
    }
    if include_fc2:
        weight_map["audio_projector.fc2.weight"] = "model-00002-of-00002.safetensors"
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
