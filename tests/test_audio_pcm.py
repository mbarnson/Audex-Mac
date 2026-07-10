from __future__ import annotations

import wave
from pathlib import Path

import pytest

from audex_mac.audio_pcm import (
    load_wav_pcm,
    normalize_audio_samples,
    prepare_audex_pcm_clips,
    prepare_audex_wav_clips,
)

pytestmark = pytest.mark.fast


def test_normalize_audio_samples_scales_int16_pcm() -> None:
    result = normalize_audio_samples([-32768, 0, 32767])

    assert result == pytest.approx((-1.0, 0.0, 32767 / 32768.0))


def test_normalize_audio_samples_peak_normalizes_large_float_pcm() -> None:
    result = normalize_audio_samples([-2.0, 0.0, 1.0])

    assert result == pytest.approx((-1.0, 0.0, 0.5))


def test_normalize_audio_samples_averages_frame_major_stereo() -> None:
    result = normalize_audio_samples([(1.0, -1.0), (0.5, 0.25)])

    assert result == pytest.approx((0.0, 0.375))


def test_normalize_audio_samples_averages_channel_major_stereo() -> None:
    result = normalize_audio_samples([[1.0, 0.5, 0.0], [-1.0, 0.25, 1.0]])

    assert result == pytest.approx((0.0, 0.375, 0.5))


def test_prepare_audex_pcm_clips_pads_short_audio_to_one_30_second_clip() -> None:
    result = prepare_audex_pcm_clips([0.25, -0.25], sample_rate=16_000)

    assert result.sample_rate == 16_000
    assert result.clip_samples == 480_000
    assert result.original_sample_count == 2
    assert result.num_clips == 1
    assert len(result.clips[0]) == 480_000
    assert result.clips[0][:4] == pytest.approx((0.25, -0.25, 0.0, 0.0))


def test_prepare_audex_pcm_clips_splits_multiple_30_second_clips() -> None:
    result = prepare_audex_pcm_clips([0.0] * 480_001, sample_rate=16_000)

    assert result.num_clips == 2
    assert result.padded_sample_count == 960_000


def test_prepare_audex_pcm_clips_rejects_more_than_30_clips() -> None:
    with pytest.raises(ValueError, match="MAX_AUDIO_CLIPS"):
        prepare_audex_pcm_clips([0.0] * 31, sample_rate=1, clip_duration=1.0)


def test_normalize_audio_samples_rejects_unsupported_multichannel_shape() -> None:
    with pytest.raises(ValueError, match="Unsupported audio shape"):
        normalize_audio_samples([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]])


def test_load_wav_pcm_reads_16_bit_mono_samples(tmp_path: Path) -> None:
    wav_path = tmp_path / "mono.wav"
    write_pcm16_wav(wav_path, sample_rate=16_000, channels=1, samples=(-32768, 32767))

    loaded = load_wav_pcm(wav_path)

    assert loaded.sample_rate == 16_000
    assert loaded.channels == 1
    assert loaded.sample_width == 2
    assert loaded.samples == pytest.approx((-1.0, 32767 / 32768.0))


def test_prepare_audex_wav_clips_loads_native_test_fixture_shape(
    tmp_path: Path,
) -> None:
    wav_path = tmp_path / "stereo.wav"
    write_pcm16_wav(
        wav_path,
        sample_rate=16_000,
        channels=2,
        samples=(-32768, 32767, 0, 32767),
    )

    clips = prepare_audex_wav_clips(wav_path)

    assert clips.num_clips == 1
    assert clips.clip_samples == 480_000
    assert clips.clips[0][0] == pytest.approx(-0.5 / 16383.5)
    assert clips.clips[0][1] == pytest.approx(1.0)


def test_prepare_audex_wav_clips_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    wav_path = tmp_path / "wrong-rate.wav"
    write_pcm16_wav(wav_path, sample_rate=8_000, channels=1, samples=(0,))

    with pytest.raises(ValueError, match="16000 Hz"):
        prepare_audex_wav_clips(wav_path)


def write_pcm16_wav(
    path: Path,
    *,
    sample_rate: int,
    channels: int,
    samples: tuple[int, ...],
) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(
            b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)
        )
