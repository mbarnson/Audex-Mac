from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from audex_mac.speech_output import (
    SpeechOutputSmokeResult,
    float_samples_to_pcm16_bytes,
    mlx_waveform_to_pcm16_bytes,
    pcm16_bytes_peak_abs,
    write_pcm16_wav,
    write_pcm16_wav_bytes,
)

pytestmark = pytest.mark.fast


def test_write_pcm16_wav_writes_mono_16khz_audio(tmp_path: Path) -> None:
    wav_path = tmp_path / "out.wav"

    write_pcm16_wav(wav_path, (-2.0, -0.5, 0.0, 0.5, 2.0), sample_rate=16_000)

    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == 5


def test_float_samples_to_pcm16_bytes_clips_and_packs() -> None:
    assert float_samples_to_pcm16_bytes([-1.5, 0.0, 1.5]) == struct.pack(
        "<hhh",
        -32767,
        0,
        32767,
    )


def test_pcm16_bytes_peak_abs_reads_little_endian_pcm() -> None:
    pcm = struct.pack("<hhh", -32767, 0, 16384)

    assert pcm16_bytes_peak_abs(pcm) == pytest.approx(1.0)


def test_pcm16_bytes_peak_abs_ignores_partial_sample() -> None:
    pcm = struct.pack("<h", 8192) + b"x"

    assert pcm16_bytes_peak_abs(pcm) == pytest.approx(8192 / 32767.0)


def test_mlx_waveform_to_pcm16_bytes_uses_vectorized_bytes_path() -> None:
    class FakePcm:
        def __init__(self, source) -> None:
            self.source = source

        def astype(self, dtype):
            self.dtype = dtype
            return self

        def tobytes(self) -> bytes:
            return b"pcm"

    class FakeWaveform:
        def __mul__(self, scale: float):
            self.scale = scale
            return FakePcm(self)

    class FakeMx:
        int16 = "int16"

        def __init__(self) -> None:
            self.evals = []

        def clip(self, waveform, low: float, high: float):
            self.clip_args = (waveform, low, high)
            return waveform

        def eval(self, value) -> None:
            self.evals.append(value)

    waveform = FakeWaveform()
    mx = FakeMx()

    assert mlx_waveform_to_pcm16_bytes(waveform, mx) == b"pcm"
    assert mx.clip_args == (waveform, -1.0, 1.0)
    assert waveform.scale == 32767.0
    assert mx.evals


def test_mlx_waveform_to_pcm16_bytes_returns_none_without_bytes_path() -> None:
    class FakeMx:
        int16 = "int16"

        def clip(self, waveform, low: float, high: float):
            return waveform

    class FakeWaveform:
        def __mul__(self, scale: float):
            return self

        def astype(self, dtype):
            return self

    assert mlx_waveform_to_pcm16_bytes(FakeWaveform(), FakeMx()) is None


def test_mlx_waveform_to_pcm16_bytes_uses_buffer_protocol_without_tobytes() -> None:
    class FakePcm(bytes):
        def astype(self, dtype):
            self.dtype = dtype
            return self

    class FakeWaveform:
        def __mul__(self, scale: float):
            self.scale = scale
            return FakePcm(b"pcm")

    class FakeMx:
        int16 = "int16"

        def __init__(self) -> None:
            self.evals = []

        def clip(self, waveform, low: float, high: float):
            self.clip_args = (waveform, low, high)
            return waveform

        def eval(self, value) -> None:
            self.evals.append(value)

    waveform = FakeWaveform()
    mx = FakeMx()

    assert mlx_waveform_to_pcm16_bytes(waveform, mx) == b"pcm"
    assert mx.clip_args == (waveform, -1.0, 1.0)
    assert waveform.scale == 32767.0
    assert mx.evals


def test_write_pcm16_wav_bytes_writes_already_packed_audio(tmp_path: Path) -> None:
    wav_path = tmp_path / "packed.wav"
    pcm = struct.pack("<hhh", -32767, 0, 32767)

    write_pcm16_wav_bytes(wav_path, pcm, sample_rate=16_000)

    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == 3
        assert wav.readframes(3) == pcm


def test_speech_output_smoke_result_requires_wav_and_run_log(tmp_path: Path) -> None:
    wav_path = tmp_path / "speech.wav"
    log_path = tmp_path / "speech.json"
    wav_path.write_bytes(b"RIFF")
    log_path.write_text("{}\n", encoding="utf-8")

    result = SpeechOutputSmokeResult(
        backend="mlx",
        device="Device(gpu, 0)",
        prompt_tokens=57,
        generated_token_ids=(166944, 149775),
        generated_codec_frames=(35867, 18698),
        reached_end_token=False,
        hit_max_tokens=True,
        waveform_shape=(640,),
        sample_rate=16_000,
        hop_length=320,
        finite=True,
        peak_abs=0.05,
        wav_path=wav_path,
        run_log_path=log_path,
    )

    assert result.ready is True


def test_speech_output_smoke_result_rejects_missing_artifacts(tmp_path: Path) -> None:
    result = SpeechOutputSmokeResult(
        backend="mlx",
        device="Device(gpu, 0)",
        prompt_tokens=57,
        generated_token_ids=(166944,),
        generated_codec_frames=(35867,),
        reached_end_token=False,
        hit_max_tokens=True,
        waveform_shape=(320,),
        sample_rate=16_000,
        hop_length=320,
        finite=True,
        peak_abs=0.05,
        wav_path=tmp_path / "missing.wav",
        run_log_path=tmp_path / "missing.json",
    )

    assert result.ready is False
