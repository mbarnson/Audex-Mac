"""External XCodec1 decoding for Audex text-to-audio evaluation."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_evaluation_generation import (
    XCODEC1_CODEBOOK_SIZE,
    XCODEC1_GENERATED_CODEBOOKS,
    TtaOutputInspection,
)

XCODEC1_REPO_ID = "hf-audio/xcodec-hubert-general-balanced"
XCODEC1_ENV = "XCODEC1_PATH"
MAX_XCODEC1_FRAMES = 500


@dataclass(frozen=True, slots=True)
class XCodec1Config:
    path: Path
    device: str = "auto"
    repo_id: str = XCODEC1_REPO_ID


class XCodec1WavDecoder:
    """Callable adapter that decodes one valid TTA stream to a local WAV."""

    def __init__(
        self,
        config: XCodec1Config,
        *,
        codec_loader: Callable[[XCodec1Config], Any] | None = None,
        torch_module: Any | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        self.config = config
        self._codec_loader = codec_loader or load_xcodec1_model
        self._torch_module = torch_module
        self._codec: Any | None = None
        self.sample_rate = sample_rate

    def __call__(
        self,
        inspection: TtaOutputInspection,
        destination: Path,
        case: Any,
    ) -> None:
        del case
        waveform = decode_xcodec1_inspection(
            self._load_codec(),
            inspection,
            torch_module=self._torch_module,
            device=self.config.device,
        )
        _write_pcm16_wav(destination, waveform, sample_rate=self.sample_rate)

    def _load_codec(self) -> Any:
        if self._codec is None:
            self._codec = self._codec_loader(self.config)
        return self._codec


def resolve_xcodec1_config(
    explicit_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    device: str | None = None,
) -> XCodec1Config:
    """Resolve the external XCodec1 model directory and fail loudly if absent."""

    active_env = os.environ if env is None else env
    raw_path = (
        explicit_path if explicit_path is not None else active_env.get(XCODEC1_ENV)
    )
    if raw_path is None or not str(raw_path).strip():
        raise RuntimeError(
            "XCodec1 is required for Audex text-to-audio decoding. Set "
            f"{XCODEC1_ENV} or pass an explicit path to a local download of "
            f"{XCODEC1_REPO_ID}."
        )
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"XCodec1 path does not exist: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"XCodec1 path is missing config.json: {path}. Expected a local "
            f"snapshot of {XCODEC1_REPO_ID}."
        )
    selected_device = device or "auto"
    return XCodec1Config(path=path, device=selected_device)


def choose_torch_device(torch_module: Any, *, requested: str | None = None) -> str:
    if requested and requested != "auto":
        return requested
    cuda = getattr(torch_module, "cuda", None)
    if (
        cuda is not None
        and callable(getattr(cuda, "is_available", None))
        and bool(cuda.is_available())
    ):
        return "cuda"
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if (
        mps_backend is not None
        and callable(getattr(mps_backend, "is_available", None))
        and bool(mps_backend.is_available())
    ):
        return "mps"
    raise RuntimeError(
        "XCodec1 device=auto requested accelerator execution, but no accelerator "
        "is available. Pass --xcodec-device cpu explicitly to allow CPU decode."
    )


def load_xcodec1_model(config: XCodec1Config) -> Any:
    """Load XCodec1 with Transformers without making it a runtime import."""

    try:
        transformers = importlib.import_module("transformers")
        torch_module = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "XCodec1 decoding requires evaluator-only dependencies: torch and "
            "transformers. Install the audio-eval optional dependency group and "
            f"download {XCODEC1_REPO_ID}."
        ) from exc

    model = transformers.AutoModel.from_pretrained(
        str(config.path),
        trust_remote_code=True,
    )
    resolved_device = choose_torch_device(torch_module, requested=config.device)
    return model.to(resolved_device).eval()


def decode_xcodec1_inspection(
    codec: Any,
    inspection: TtaOutputInspection,
    *,
    torch_module: Any | None = None,
    device: str = "auto",
) -> tuple[float, ...]:
    """Decode valid flat interleaved RVQ codec IDs to mono waveform samples."""

    if not inspection.valid:
        raise ValueError(f"cannot decode invalid TTA structure: {inspection.failures}")
    if not inspection.codec_ids:
        raise ValueError("cannot decode empty TTA codec stream")

    torch = torch_module or importlib.import_module("torch")
    resolved_device = choose_torch_device(torch, requested=device)
    usable = len(inspection.codec_ids) - (
        len(inspection.codec_ids) % XCODEC1_GENERATED_CODEBOOKS
    )
    if usable <= 0:
        raise ValueError("TTA codec stream contains no complete RVQ frames")
    codec_ids = list(inspection.codec_ids[:usable])
    num_frames = usable // XCODEC1_GENERATED_CODEBOOKS

    with torch.no_grad():
        codes = torch.tensor(codec_ids, dtype=torch.long, device=resolved_device)
        codes = codes.view(1, num_frames, XCODEC1_GENERATED_CODEBOOKS)[
            :, :MAX_XCODEC1_FRAMES
        ].transpose(1, 2)
        layer_offsets = (
            torch.arange(
                XCODEC1_GENERATED_CODEBOOKS,
                device=getattr(codes, "device", resolved_device),
                dtype=getattr(codes, "dtype", torch.long),
            )
            * XCODEC1_CODEBOOK_SIZE
        )
        codes = (codes - layer_offsets.view(1, -1, 1)).clamp(
            0,
            XCODEC1_CODEBOOK_SIZE - 1,
        )
        decoder_outputs = codec.decode(codes)
        audio_values = decoder_outputs.audio_values[0, 0, :].detach()
        host_values = audio_values.cpu()
        waveform = host_values.numpy()
    return tuple(float(sample) for sample in waveform)


def _write_pcm16_wav(
    destination: Path,
    waveform: tuple[float, ...],
    *,
    sample_rate: int,
) -> None:
    if not waveform:
        raise ValueError("XCodec1 decoder produced an empty waveform")
    try:
        sf = importlib.import_module("soundfile")
    except ImportError as exc:
        raise RuntimeError(
            "XCodec1 WAV output requires the audio-eval dependency soundfile"
        ) from exc
    peak = max(abs(sample) for sample in waveform)
    output = (
        [sample / peak * 0.99 for sample in waveform] if peak > 0.99 else list(waveform)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, output, sample_rate, subtype="PCM_16")
