"""Audex speech runtime preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio_components import AudioComponentPreflight, preflight_audio_components
from .audio_contract import (
    DecoderPreflight,
    SpeechTokenizerPreflight,
    preflight_decoder,
    preflight_speech_tokenizer,
)
from .audio_features import AudioPreprocessorPreflight, preflight_audio_preprocessor
from .checkpoints import HF_CACHE_ROOT, SnapshotCheck, verify_snapshot
from .models import AudexModel


@dataclass(frozen=True, slots=True)
class AudioRuntimePreflight:
    model: AudexModel
    snapshot_check: SnapshotCheck
    audio_components: AudioComponentPreflight | None
    audio_preprocessor: AudioPreprocessorPreflight | None
    decoder: DecoderPreflight | None
    speech_tokenizer: SpeechTokenizerPreflight | None

    @property
    def ready(self) -> bool:
        return (
            self.snapshot_check.complete
            and self.audio_components is not None
            and self.audio_components.ready
            and self.audio_preprocessor is not None
            and self.audio_preprocessor.ready
            and self.decoder is not None
            and self.decoder.ready
            and self.speech_tokenizer is not None
            and self.speech_tokenizer.ready
        )

    @property
    def model_path(self) -> Path | None:
        if self.snapshot_check.snapshot_path is None:
            return None
        return self.snapshot_check.snapshot_path / "checkpoint_folder_full"

    @property
    def decoder_path(self) -> Path | None:
        if self.snapshot_check.snapshot_path is None:
            return None
        return self.snapshot_check.snapshot_path / "audex_causal_speech_decoder"

    @property
    def missing_items(self) -> tuple[str, ...]:
        missing = list(self.snapshot_check.missing_summary)
        if self.audio_components is None:
            missing.append("checkpoint_folder_full audio components")
        else:
            missing.extend(
                f"checkpoint_folder_full/{item}"
                for item in self.audio_components.missing_items
            )
        if self.audio_preprocessor is None:
            missing.append("checkpoint_folder_full/audio_preprocessor")
        else:
            missing.extend(
                f"checkpoint_folder_full/{item}"
                for item in self.audio_preprocessor.missing_items
            )
        if self.decoder is None:
            missing.append("audex_causal_speech_decoder")
        else:
            missing.extend(
                f"audex_causal_speech_decoder/{name}"
                for name in self.decoder.missing_files
            )
            if self.decoder.sample_rate != 16000:
                missing.append(
                    "audex_causal_speech_decoder/config.json sample_rate=16000"
                )
        if self.speech_tokenizer is None:
            missing.append("checkpoint_folder_full/tokenizer.json")
        elif not self.speech_tokenizer.ready:
            missing.append(self.speech_tokenizer.error or "speech tokenizer tokens")
        return tuple(dict.fromkeys(missing))


def preflight_audio_runtime(
    model: AudexModel,
    cache_root: Path = HF_CACHE_ROOT,
) -> AudioRuntimePreflight:
    """Check whether the selected Audex speech snapshot has native STS assets."""

    snapshot_check = verify_snapshot(
        model,
        required_files=model.speech_required_files,
        checkpoint_dirs=model.speech_checkpoint_dirs,
        cache_root=cache_root,
    )
    if snapshot_check.snapshot_path is None:
        return AudioRuntimePreflight(
            model=model,
            snapshot_check=snapshot_check,
            audio_components=None,
            audio_preprocessor=None,
            decoder=None,
            speech_tokenizer=None,
        )

    snapshot = snapshot_check.snapshot_path
    return AudioRuntimePreflight(
        model=model,
        snapshot_check=snapshot_check,
        audio_components=preflight_audio_components(
            snapshot / "checkpoint_folder_full"
        ),
        audio_preprocessor=preflight_audio_preprocessor(
            snapshot / "checkpoint_folder_full"
        ),
        decoder=preflight_decoder(snapshot / "audex_causal_speech_decoder"),
        speech_tokenizer=preflight_speech_tokenizer(
            snapshot / "checkpoint_folder_full" / "tokenizer.json"
        ),
    )
