"""Deterministic PCM preparation for Audex native audio input."""

from __future__ import annotations

import math
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .audio_contract import (
    DEFAULT_SOUND_CLIP_DURATION,
    MAX_AUDIO_CLIPS,
    SAMPLE_RATE,
)

Number = int | float
AudioInput = Sequence[Number] | Sequence[Sequence[Number]]


@dataclass(frozen=True, slots=True)
class PreparedAudioClips:
    sample_rate: int
    clip_samples: int
    original_sample_count: int
    clips: tuple[tuple[float, ...], ...]

    @property
    def num_clips(self) -> int:
        return len(self.clips)

    @property
    def padded_sample_count(self) -> int:
        return self.num_clips * self.clip_samples


@dataclass(frozen=True, slots=True)
class LoadedWavPcm:
    path: Path
    sample_rate: int
    channels: int
    sample_width: int
    samples: tuple[float, ...]


def prepare_audex_wav_clips(path: Path) -> PreparedAudioClips:
    loaded = load_wav_pcm(path)
    if loaded.sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Audex WAV fixtures must be {SAMPLE_RATE} Hz, got {loaded.sample_rate}"
        )
    return prepare_audex_pcm_clips(loaded.samples, sample_rate=loaded.sample_rate)


def load_wav_pcm(path: Path) -> LoadedWavPcm:
    """Load 16-bit PCM WAV into normalized mono float samples."""

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if channels <= 0:
        raise ValueError(f"WAV must have at least one channel: {path}")
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got width={sample_width}")

    integers = tuple(
        int.from_bytes(raw[index : index + sample_width], "little", signed=True)
        for index in range(0, len(raw), sample_width)
    )
    if channels == 1:
        audio: AudioInput = integers
    else:
        audio = tuple(
            integers[index : index + channels]
            for index in range(0, len(integers), channels)
        )
    return LoadedWavPcm(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        samples=normalize_audio_samples(audio),
    )


def prepare_audex_pcm_clips(
    audio: AudioInput,
    *,
    sample_rate: int = SAMPLE_RATE,
    clip_duration: float = DEFAULT_SOUND_CLIP_DURATION,
) -> PreparedAudioClips:
    """Normalize mono PCM and split/pad it into NVIDIA-style Audex clips."""

    normalized = normalize_audio_samples(audio)
    clip_samples = int(round(sample_rate * clip_duration))
    if clip_samples <= 0:
        raise ValueError(f"Invalid clip_samples: {clip_samples}")
    if not normalized:
        normalized = (0.0,)

    num_clips = max(1, math.ceil(len(normalized) / clip_samples))
    if num_clips > MAX_AUDIO_CLIPS:
        raise ValueError(
            f"Audio needs {num_clips} clips > MAX_AUDIO_CLIPS={MAX_AUDIO_CLIPS}"
        )

    clips = []
    for index in range(num_clips):
        start = index * clip_samples
        clip = normalized[start : start + clip_samples]
        if len(clip) < clip_samples:
            clip = clip + (0.0,) * (clip_samples - len(clip))
        clips.append(clip)

    return PreparedAudioClips(
        sample_rate=sample_rate,
        clip_samples=clip_samples,
        original_sample_count=len(normalized),
        clips=tuple(clips),
    )


def normalize_audio_samples(audio: AudioInput) -> tuple[float, ...]:
    """Return mono float PCM in [-1, 1], matching NVIDIA's release behavior."""

    mono = _to_mono_samples(audio)
    if not mono:
        return ()

    values = tuple(float(sample) for sample in mono)
    if all(isinstance(sample, int) and not isinstance(sample, bool) for sample in mono):
        values = tuple(sample / 32768.0 for sample in values)

    max_abs = max((abs(sample) for sample in values), default=0.0)
    if max_abs > 1.0:
        values = tuple(sample / max_abs for sample in values)

    return values


def _to_mono_samples(audio: AudioInput) -> tuple[Number, ...]:
    if not audio:
        return ()

    first = audio[0]  # type: ignore[index]
    if not _is_sequence(first):
        return tuple(audio)  # type: ignore[arg-type]

    rows = [tuple(row) for row in audio]  # type: ignore[union-attr]
    if not rows:
        return ()

    row_lengths = {len(row) for row in rows}
    if len(row_lengths) != 1:
        raise ValueError("Audio channel rows must have consistent lengths")

    row_count = len(rows)
    column_count = next(iter(row_lengths))
    if column_count == 0:
        return ()

    if column_count <= 2:
        return tuple(sum(frame) / column_count for frame in rows)

    if row_count <= 2:
        return tuple(sum(frame) / row_count for frame in zip(*rows, strict=True))

    raise ValueError(
        "Unsupported audio shape; expected mono, frame-major stereo, or "
        "channel-major stereo PCM"
    )


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)
