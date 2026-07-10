from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from audex_mac.audio_evaluation_generation import TtaOutputInspection
from audex_mac.audio_evaluation_xcodec import (
    XCODEC1_REPO_ID,
    XCodec1Config,
    XCodec1WavDecoder,
    choose_torch_device,
    decode_xcodec1_inspection,
    resolve_xcodec1_config,
)

pytestmark = pytest.mark.fast


def test_resolve_xcodec1_config_requires_explicit_or_environment_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="XCodec1 is required"):
        resolve_xcodec1_config(env={})

    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="XCodec1 path does not exist"):
        resolve_xcodec1_config(explicit_path=missing, env={})

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    with pytest.raises(FileNotFoundError, match="config.json"):
        resolve_xcodec1_config(explicit_path=invalid, env={})

    valid = tmp_path / "xcodec"
    valid.mkdir()
    (valid / "config.json").write_text("{}", encoding="utf-8")

    config = resolve_xcodec1_config(env={"XCODEC1_PATH": str(valid)})

    assert config == XCodec1Config(path=valid, device="auto", repo_id=XCODEC1_REPO_ID)
    assert resolve_xcodec1_config(
        explicit_path=valid,
        env={},
        device="cpu",
    ) == XCodec1Config(path=valid, device="cpu", repo_id=XCODEC1_REPO_ID)


def test_choose_torch_device_prefers_cuda_then_mps_then_cpu() -> None:
    assert choose_torch_device(_torch(cuda=True, mps=True)) == "cuda"
    assert choose_torch_device(_torch(cuda=False, mps=True)) == "mps"
    assert choose_torch_device(_torch(cuda=False, mps=False)) == "cpu"
    assert choose_torch_device(_torch(cuda=False, mps=True), requested="auto") == "mps"
    assert choose_torch_device(_torch(cuda=False, mps=False), requested="mps") == "mps"


def test_decode_xcodec1_inspection_offsets_interleaved_rvq_codes() -> None:
    fake_torch = FakeTorch()
    codec = FakeCodec()
    inspection = TtaOutputInspection(
        codec_ids=(0, 1025, 2050, 3075, 1, 1026, 2051, 3076),
        codec_token_count=8,
        frame_count=2,
        duration_seconds=0.04,
        reached_end_token=True,
        first_phase_mismatch=None,
        unexpected_token_ids=(),
        failures=(),
    )

    waveform = decode_xcodec1_inspection(
        codec,
        inspection,
        torch_module=fake_torch,
        device="mps",
    )

    assert fake_torch.tensor_calls == [
        {"data": list(inspection.codec_ids), "dtype": fake_torch.long, "device": "mps"}
    ]
    assert codec.seen_codes.tolist() == [[[0, 1], [1, 2], [2, 3], [3, 4]]]
    assert waveform == (0.25, -0.25)


def test_decode_xcodec1_inspection_fails_on_invalid_structure() -> None:
    inspection = TtaOutputInspection(
        codec_ids=(),
        codec_token_count=0,
        frame_count=0,
        duration_seconds=0.0,
        reached_end_token=False,
        first_phase_mismatch=None,
        unexpected_token_ids=(),
        failures=("missing_end_token",),
    )

    with pytest.raises(ValueError, match="invalid TTA structure"):
        decode_xcodec1_inspection(FakeCodec(), inspection, torch_module=FakeTorch())


def test_xcodec1_wav_decoder_writes_pcm16_wav(tmp_path: Path) -> None:
    config = XCodec1Config(path=tmp_path, device="mps")
    inspection = TtaOutputInspection(
        codec_ids=(0, 1025, 2050, 3075),
        codec_token_count=4,
        frame_count=1,
        duration_seconds=0.02,
        reached_end_token=True,
        first_phase_mismatch=None,
        unexpected_token_ids=(),
        failures=(),
    )
    destination = tmp_path / "generated.wav"

    decoder = XCodec1WavDecoder(
        config,
        codec_loader=lambda _config: FakeCodec(),
        torch_module=FakeTorch(),
    )

    decoder(inspection, destination, case=None)

    assert destination.is_file()
    import soundfile as sf

    info = sf.info(destination)
    assert info.samplerate == 16_000
    assert info.channels == 1


def test_xcodec1_wav_decoder_peak_normalizes_before_pcm16_write(tmp_path: Path) -> None:
    config = XCodec1Config(path=tmp_path, device="mps")
    inspection = TtaOutputInspection(
        codec_ids=(0, 1025, 2050, 3075),
        codec_token_count=4,
        frame_count=1,
        duration_seconds=0.02,
        reached_end_token=True,
        first_phase_mismatch=None,
        unexpected_token_ids=(),
        failures=(),
    )
    destination = tmp_path / "generated.wav"

    decoder = XCodec1WavDecoder(
        config,
        codec_loader=lambda _config: FakeCodec(waveform=[2.0, -2.0]),
        torch_module=FakeTorch(),
    )

    decoder(inspection, destination, case=None)

    import soundfile as sf

    samples, _sample_rate = sf.read(destination, dtype="float32")
    assert max(abs(float(sample)) for sample in samples) < 1.0


def _torch(*, cuda: bool, mps: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


class FakeTorch:
    long = "long"

    def __init__(self) -> None:
        self.tensor_calls: list[dict[str, Any]] = []

    def tensor(self, data: list[int], *, dtype: str, device: str) -> FakeTensor:
        self.tensor_calls.append({"data": data, "dtype": dtype, "device": device})
        return FakeTensor(data, shape=(len(data),))

    def arange(self, count: int, *, device: str, dtype: str) -> FakeTensor:
        del device, dtype
        return FakeTensor(list(range(count)), shape=(count,))

    class no_grad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None


class FakeTensor:
    def __init__(self, data: list[int] | list[Any], *, shape: tuple[int, ...]) -> None:
        self.data = data
        self.shape = shape
        self.device = "fake"
        self.dtype = "long"

    def view(self, *shape: int) -> FakeTensor:
        if shape == (1, -1, 1):
            return FakeTensor([self.data], shape=(1, len(self.data), 1))
        assert len(shape) == 3
        assert shape[0] == 1
        assert shape[2] == 4
        frame_count = shape[1]
        frames = [
            self.data[index : index + 4] for index in range(0, frame_count * 4, 4)
        ]
        return FakeTensor([frames], shape=shape)

    def transpose(self, first: int, second: int) -> FakeTensor:
        assert (first, second) == (1, 2)
        frames = self.data[0]
        codebooks = [[frame[index] for frame in frames] for index in range(4)]
        return FakeTensor([codebooks], shape=(1, 4, 2))

    def __getitem__(self, key: object) -> FakeTensor:
        return self

    def __mul__(self, value: int) -> FakeTensor:
        return FakeTensor([item * value for item in self.data], shape=self.shape)

    def __sub__(self, other: FakeTensor) -> FakeTensor:
        offsets = other.data[0]
        adjusted = [
            [value - offsets[codebook_index] for value in codebook]
            for codebook_index, codebook in enumerate(self.data[0])
        ]
        return FakeTensor([adjusted], shape=self.shape)

    def clamp(self, minimum: int, maximum: int) -> FakeTensor:
        adjusted = [
            [max(minimum, min(maximum, value)) for value in codebook]
            for codebook in self.data[0]
        ]
        return FakeTensor([adjusted], shape=self.shape)

    def tolist(self) -> list[Any]:
        return self.data


class FakeCodec:
    def __init__(self, waveform: list[float] | None = None) -> None:
        self.seen_codes: FakeTensor | None = None
        self.waveform = waveform or [0.25, -0.25]

    def decode(self, codes: FakeTensor) -> Any:
        self.seen_codes = codes
        return SimpleNamespace(audio_values=FakeAudioValues(self.waveform))


class FakeAudioValues:
    def __init__(self, waveform: list[float]) -> None:
        self.waveform = waveform

    def __getitem__(self, key: object) -> FakeAudioValues:
        del key
        return self

    def detach(self) -> FakeAudioValues:
        return self

    def cpu(self) -> FakeAudioValues:
        return self

    def numpy(self) -> list[float]:
        return self.waveform
