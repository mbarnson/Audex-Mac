from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from audex_mac.audio_encoder import (
    AudexAudioEncoderConfig,
    encode_audio_features_mlx,
    expected_audio_encoder_weight_keys,
    load_audio_encoder_config,
    resolve_audio_encoder_shards,
)

pytestmark = pytest.mark.fast


def test_load_audio_encoder_config_reads_qwen2_audio_dimensions(
    tmp_path: Path,
) -> None:
    write_encoder_config(tmp_path)

    config = load_audio_encoder_config(tmp_path)

    assert config.d_model == 1280
    assert config.encoder_attention_heads == 20
    assert config.encoder_ffn_dim == 5120
    assert config.encoder_layers == 32
    assert config.max_source_positions == 1500
    assert config.num_mel_bins == 128
    assert config.activation_function == "gelu"
    assert config.expected_feature_frames == 3000
    assert config.head_dim == 64


def test_expected_audio_encoder_weight_keys_include_all_layer_tensors() -> None:
    config = AudexAudioEncoderConfig(
        d_model=4,
        encoder_attention_heads=2,
        encoder_ffn_dim=8,
        encoder_layers=2,
        max_source_positions=4,
        num_mel_bins=2,
        activation_function="gelu",
        scale_embedding=False,
    )

    keys = expected_audio_encoder_weight_keys(config)

    assert "conv1.weight" in keys
    assert "layers.0.self_attn.q_proj.weight" in keys
    assert "layers.1.final_layer_norm.bias" in keys
    assert len(keys) == 7 + 2 * 15


def test_resolve_audio_encoder_shards_requires_all_layer_tensors(
    tmp_path: Path,
) -> None:
    write_encoder_config(tmp_path, encoder_layers=1)
    write_encoder_index(tmp_path, include_fc2=False)

    with pytest.raises(FileNotFoundError, match="audio_encoder.layers.0.fc2.weight"):
        resolve_audio_encoder_shards(tmp_path)


@pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="MLX is installed in the vLLM Metal runtime, not the fast test venv",
)
def test_encode_audio_features_mlx_accepts_tiny_zero_layer_encoder() -> None:
    import mlx.core as mx

    config = AudexAudioEncoderConfig(
        d_model=2,
        encoder_attention_heads=1,
        encoder_ffn_dim=4,
        encoder_layers=0,
        max_source_positions=2,
        num_mel_bins=2,
        activation_function="gelu",
        scale_embedding=False,
    )
    weights = {
        "conv1.weight": mx.zeros((2, 2, 3), dtype=mx.float32),
        "conv1.bias": mx.zeros((2,), dtype=mx.float32),
        "conv2.weight": mx.zeros((2, 2, 3), dtype=mx.float32),
        "conv2.bias": mx.zeros((2,), dtype=mx.float32),
        "embed_positions.weight": mx.zeros((2, 2), dtype=mx.float32),
        "layer_norm.weight": mx.ones((2,), dtype=mx.float32),
        "layer_norm.bias": mx.zeros((2,), dtype=mx.float32),
    }
    features = mx.zeros((1, 2, 4), dtype=mx.float32)

    output = encode_audio_features_mlx(features, weights, config)

    assert tuple(int(part) for part in output.shape) == (1, 1, 2)


def write_encoder_config(
    model_path: Path,
    *,
    encoder_layers: int = 32,
) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "audio_config": {
                    "activation_function": "gelu",
                    "d_model": 1280,
                    "encoder_attention_heads": 20,
                    "encoder_ffn_dim": 5120,
                    "encoder_layers": encoder_layers,
                    "max_source_positions": 1500,
                    "model_type": "qwen2_audio_encoder",
                    "num_mel_bins": 128,
                    "scale_embedding": False,
                }
            }
        ),
        encoding="utf-8",
    )


def write_encoder_index(
    model_path: Path,
    *,
    include_fc2: bool = True,
) -> None:
    config = load_audio_encoder_config(model_path)
    weight_map = {
        f"audio_encoder.{key}": "model-00002-of-00002.safetensors"
        for key in expected_audio_encoder_weight_keys(config)
    }
    if not include_fc2:
        weight_map.pop("audio_encoder.layers.0.fc2.weight")
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
