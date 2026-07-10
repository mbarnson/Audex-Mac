"""Audex NV-Whisper feature extraction helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_contract import DEFAULT_SOUND_CLIP_DURATION, SAMPLE_RATE
from .audio_pcm import PreparedAudioClips
from .config_values import optional_int, optional_str

EXPECTED_FEATURE_SIZE = 128
EXPECTED_N_SAMPLES = int(SAMPLE_RATE * DEFAULT_SOUND_CLIP_DURATION)
EXPECTED_MAX_FRAMES = 3000


@dataclass(frozen=True, slots=True)
class AudioPreprocessorPreflight:
    preprocessor_path: Path
    ready: bool
    feature_extractor_type: str | None
    feature_size: int | None
    n_samples: int | None
    nb_max_frames: int | None
    sampling_rate: int | None
    chunk_length: int | None
    missing_items: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AudexFeatureExtractionResult:
    num_clips: int
    feature_shape: tuple[int, ...]
    feature_dtype: str
    feature_extractor_type: str
    input_features: Any


def preflight_audio_preprocessor(model_path: Path) -> AudioPreprocessorPreflight:
    preprocessor_path = model_path / "audio_preprocessor"
    config_path = preprocessor_path / "preprocessor_config.json"
    missing: list[str] = []
    config: dict[str, Any] = {}
    if not config_path.is_file():
        missing.append("audio_preprocessor/preprocessor_config.json")
    else:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            missing.append("audio_preprocessor/preprocessor_config.json object")
        else:
            config = raw

    feature_extractor_type = optional_str(config.get("feature_extractor_type"))
    feature_size = optional_int(config.get("feature_size"))
    n_samples = optional_int(config.get("n_samples"))
    nb_max_frames = optional_int(config.get("nb_max_frames"))
    sampling_rate = optional_int(config.get("sampling_rate"))
    chunk_length = optional_int(config.get("chunk_length"))

    if feature_extractor_type != "WhisperFeatureExtractor":
        missing.append(
            "audio_preprocessor feature_extractor_type=WhisperFeatureExtractor"
        )
    if feature_size != EXPECTED_FEATURE_SIZE:
        missing.append(f"audio_preprocessor feature_size={EXPECTED_FEATURE_SIZE}")
    if n_samples != EXPECTED_N_SAMPLES:
        missing.append(f"audio_preprocessor n_samples={EXPECTED_N_SAMPLES}")
    if nb_max_frames != EXPECTED_MAX_FRAMES:
        missing.append(f"audio_preprocessor nb_max_frames={EXPECTED_MAX_FRAMES}")
    if sampling_rate != SAMPLE_RATE:
        missing.append(f"audio_preprocessor sampling_rate={SAMPLE_RATE}")
    if chunk_length != int(DEFAULT_SOUND_CLIP_DURATION):
        missing.append(
            f"audio_preprocessor chunk_length={int(DEFAULT_SOUND_CLIP_DURATION)}"
        )

    return AudioPreprocessorPreflight(
        preprocessor_path=preprocessor_path,
        ready=not missing,
        feature_extractor_type=feature_extractor_type,
        feature_size=feature_size,
        n_samples=n_samples,
        nb_max_frames=nb_max_frames,
        sampling_rate=sampling_rate,
        chunk_length=chunk_length,
        missing_items=tuple(dict.fromkeys(missing)),
    )


def extract_audex_input_features(
    clips: PreparedAudioClips,
    *,
    preprocessor_path: Path | None = None,
    feature_extractor: Any | None = None,
) -> AudexFeatureExtractionResult:
    """Convert prepared PCM clips to Audex/NV-Whisper input features."""

    if feature_extractor is None:
        if preprocessor_path is None:
            raise ValueError("preprocessor_path is required without feature_extractor")
        feature_extractor = _load_feature_extractor(preprocessor_path)

    features = feature_extractor(
        [list(clip) for clip in clips.clips],
        sampling_rate=clips.sample_rate,
        return_tensors="np",
        padding="max_length",
        return_attention_mask=False,
    )
    input_features = features.input_features
    shape = tuple(int(part) for part in input_features.shape)
    expected_shape = (clips.num_clips, EXPECTED_FEATURE_SIZE, EXPECTED_MAX_FRAMES)
    if shape != expected_shape:
        raise ValueError(f"Expected Audex input features {expected_shape}, got {shape}")

    return AudexFeatureExtractionResult(
        num_clips=clips.num_clips,
        feature_shape=shape,
        feature_dtype=str(getattr(input_features, "dtype", "unknown")),
        feature_extractor_type=type(feature_extractor).__name__,
        input_features=input_features,
    )


def _load_feature_extractor(preprocessor_path: Path) -> Any:
    try:
        from transformers import AutoFeatureExtractor
    except ImportError as exc:
        raise RuntimeError(
            "Audex audio feature extraction requires transformers in the active "
            "runtime. Run through ./start.sh after vLLM Metal dependencies are ready."
        ) from exc

    return AutoFeatureExtractor.from_pretrained(
        str(preprocessor_path),
        trust_remote_code=True,
    )
