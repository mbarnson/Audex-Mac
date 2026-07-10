from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.audio_features import (
    extract_audex_input_features,
    preflight_audio_preprocessor,
)
from audex_mac.audio_pcm import prepare_audex_pcm_clips

pytestmark = pytest.mark.fast


def test_audio_preprocessor_preflight_accepts_audex_whisper_config(
    tmp_path: Path,
) -> None:
    write_audio_preprocessor(tmp_path)

    result = preflight_audio_preprocessor(tmp_path)

    assert result.ready is True
    assert result.feature_extractor_type == "WhisperFeatureExtractor"
    assert result.feature_size == 128
    assert result.n_samples == 480_000
    assert result.nb_max_frames == 3000
    assert result.sampling_rate == 16_000
    assert result.chunk_length == 30


def test_audio_preprocessor_preflight_rejects_wrong_feature_shape(
    tmp_path: Path,
) -> None:
    write_audio_preprocessor(tmp_path, feature_size=80)

    result = preflight_audio_preprocessor(tmp_path)

    assert result.ready is False
    assert "audio_preprocessor feature_size=128" in result.missing_items


def test_extract_audex_input_features_validates_expected_shape() -> None:
    clips = prepare_audex_pcm_clips([0.0], sample_rate=16_000)

    result = extract_audex_input_features(
        clips,
        feature_extractor=FakeFeatureExtractor((1, 128, 3000)),
    )

    assert result.num_clips == 1
    assert result.feature_shape == (1, 128, 3000)
    assert result.feature_dtype == "float32"
    assert result.feature_extractor_type == "FakeFeatureExtractor"


def test_extract_audex_input_features_rejects_wrong_shape() -> None:
    clips = prepare_audex_pcm_clips([0.0], sample_rate=16_000)

    with pytest.raises(ValueError, match="Expected Audex input features"):
        extract_audex_input_features(
            clips,
            feature_extractor=FakeFeatureExtractor((1, 80, 3000)),
        )


class FakeFeatureExtractor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def __call__(self, clips, **kwargs):
        assert len(clips) == self.shape[0]
        assert kwargs["sampling_rate"] == 16_000
        assert kwargs["return_tensors"] == "np"
        assert kwargs["padding"] == "max_length"
        assert kwargs["return_attention_mask"] is False
        return FakeFeatures(self.shape)


class FakeFeatures:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.input_features = FakeTensor(shape)


class FakeTensor:
    dtype = "float32"

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def write_audio_preprocessor(
    model_path: Path,
    *,
    feature_size: int = 128,
) -> None:
    preprocessor = model_path / "audio_preprocessor"
    preprocessor.mkdir(parents=True)
    (preprocessor / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "chunk_length": 30,
                "feature_extractor_type": "WhisperFeatureExtractor",
                "feature_size": feature_size,
                "hop_length": 160,
                "n_fft": 400,
                "n_samples": 480000,
                "nb_max_frames": 3000,
                "padding_side": "right",
                "padding_value": 0.0,
                "processor_class": "Qwen2AudioProcessor",
                "return_attention_mask": True,
                "sampling_rate": 16000,
            }
        ),
        encoding="utf-8",
    )
