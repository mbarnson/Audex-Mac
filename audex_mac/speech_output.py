"""Combined Audex speech-token to waveform output smoke."""

from __future__ import annotations

import json
import sys
import time
import wave
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .speech_decoder import (
    decode_speech_token_frames_mlx,
    load_speech_decoder_config,
    load_speech_decoder_weights_mlx,
)
from .speech_generation import run_speech_token_generation_smoke

RUNS_DIR = Path(__file__).resolve().parents[1] / ".audex" / "runs"


@dataclass(frozen=True, slots=True)
class SpeechOutputSmokeResult:
    backend: str
    device: str
    prompt_tokens: int
    generated_token_ids: tuple[int, ...]
    generated_codec_frames: tuple[int, ...]
    reached_end_token: bool
    hit_max_tokens: bool
    waveform_shape: tuple[int, ...]
    sample_rate: int
    hop_length: int
    finite: bool
    peak_abs: float
    wav_path: Path
    run_log_path: Path
    streaming: bool = False
    segments: tuple[str, ...] = ()
    chunk_wav_paths: tuple[Path, ...] = ()
    first_audio_ready_seconds: float | None = None
    first_playback_started_seconds: float | None = None
    first_audio_ready_at: float | None = None
    first_playback_started_at: float | None = None
    first_playback_estimated_audible_at: float | None = None
    playback_diagnostics: dict[str, object] | None = None

    @property
    def ready(self) -> bool:
        expected_samples = len(self.generated_codec_frames) * self.hop_length
        waveform_ok = (
            self.waveform_shape[0] >= expected_samples
            if self.streaming and len(self.waveform_shape) == 1
            else self.waveform_shape == (expected_samples,)
        )
        return (
            self.finite
            and waveform_ok
            and self.sample_rate == 16_000
            and self.wav_path.is_file()
            and self.run_log_path.is_file()
        )


def run_speech_output_smoke(
    *,
    full_model_path: Path,
    decoder_path: Path,
    output_dir: Path = RUNS_DIR,
    text: str = "Say: hello from Audex on Mac.",
    max_tokens: int = 8,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SpeechOutputSmokeResult:
    """Generate Audex speech tokens, decode them with MLX, and write a WAV."""

    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("Audex speech output smoke requires mlx.") from exc

    speech = run_speech_token_generation_smoke(
        full_model_path=full_model_path,
        text=text,
        max_tokens=max_tokens,
        progress_callback=progress_callback,
    )
    token_frames = tuple((frame,) for frame in speech.generated_codec_frames)
    if not token_frames:
        raise RuntimeError("Audex speech-token generation produced no codec frames.")

    config = load_speech_decoder_config(decoder_path)
    weights = load_speech_decoder_weights_mlx(decoder_path)
    waveform = decode_speech_token_frames_mlx(token_frames, weights, config)
    finite = bool(mx.all(mx.isfinite(waveform)).item())
    peak_abs = float(mx.max(mx.abs(waveform)).item())
    samples = tuple(float(sample) for sample in waveform.tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    wav_path = output_dir / f"speech-output-{timestamp}.wav"
    run_log_path = output_dir / f"speech-output-{timestamp}.json"
    write_pcm16_wav(wav_path, samples, sample_rate=config.sample_rate)
    run_log = {
        "backend": "mlx",
        "device": str(mx.default_device()),
        "full_model_path": str(full_model_path),
        "decoder_path": str(decoder_path),
        "prompt_tokens": speech.prompt_tokens,
        "tts_sampler": {
            "temperature": speech.temperature,
            "top_p": speech.top_p,
            "top_k": speech.top_k,
            "cfg_scale": speech.cfg_scale_reference,
            "cfg_applied": speech.cfg_applied,
        },
        "generated_token_ids": list(speech.generated_token_ids),
        "generated_token_text": list(speech.generated_token_text),
        "generated_codec_frames": list(speech.generated_codec_frames),
        "reached_end_token": speech.reached_end_token,
        "hit_max_tokens": speech.hit_max_tokens,
        "waveform_shape": list(int(part) for part in waveform.shape),
        "sample_rate": config.sample_rate,
        "hop_length": config.hop_length,
        "finite": finite,
        "peak_abs": peak_abs,
        "wav_path": str(wav_path),
    }
    run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")

    return SpeechOutputSmokeResult(
        backend="mlx",
        device=str(mx.default_device()),
        prompt_tokens=speech.prompt_tokens,
        generated_token_ids=speech.generated_token_ids,
        generated_codec_frames=speech.generated_codec_frames,
        reached_end_token=speech.reached_end_token,
        hit_max_tokens=speech.hit_max_tokens,
        waveform_shape=tuple(int(part) for part in waveform.shape),
        sample_rate=config.sample_rate,
        hop_length=config.hop_length,
        finite=finite,
        peak_abs=peak_abs,
        wav_path=wav_path,
        run_log_path=run_log_path,
    )


def write_pcm16_wav(
    path: Path,
    samples: Sequence[float],
    *,
    sample_rate: int,
) -> None:
    """Write mono PCM16 WAV audio from normalized float samples."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframesraw(float_samples_to_pcm16_bytes(samples))
        wav.writeframes(b"")


def write_pcm16_wav_bytes(
    path: Path,
    pcm: bytes,
    *,
    sample_rate: int,
) -> None:
    """Write mono little-endian PCM16 WAV audio from already-packed frames."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframesraw(pcm)
        wav.writeframes(b"")


def mlx_waveform_to_pcm16_bytes(waveform: object, mx: object) -> bytes | None:
    """Pack an MLX waveform as PCM16 bytes without materializing Python floats."""

    clip = getattr(mx, "clip", None)
    int16 = getattr(mx, "int16", None)
    if not callable(clip) or int16 is None:
        return None
    try:
        pcm = (clip(waveform, -1.0, 1.0) * 32767.0).astype(int16)
        eval_fn = getattr(mx, "eval", None)
        if callable(eval_fn):
            eval_fn(pcm)
        tobytes = getattr(pcm, "tobytes", None)
        data = tobytes() if callable(tobytes) else memoryview(pcm).tobytes()
    except Exception:
        return None
    return data if isinstance(data, bytes) else None


def pcm16_bytes_peak_abs(pcm: bytes) -> float:
    """Return normalized peak amplitude from little-endian signed PCM16 bytes."""

    if not pcm:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    return max(abs(sample) for sample in samples) / 32767.0


def float_samples_to_pcm16_bytes(samples: Sequence[float]) -> bytes:
    """Pack normalized float samples as little-endian signed PCM16 bytes."""

    pcm = array(
        "h",
        (
            int(round(max(-1.0, min(1.0, float(sample))) * 32767.0))
            for sample in samples
        ),
    )
    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm.tobytes()
