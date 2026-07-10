"""Pinned Transformers CLAP embedding backend for the evaluator worker."""

from __future__ import annotations

import importlib
import math
import time
from pathlib import Path
from typing import Any


def require_torch_device(torch_module: Any, device: str) -> str:
    """Validate an explicit worker device without silently falling back to CPU."""

    if device == "cpu":
        return device
    if device == "mps":
        backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        available = getattr(backend, "is_available", None)
        if backend is None or not callable(available) or not bool(available()):
            raise RuntimeError("MPS was requested for CLAP but is not available")
        return device
    if device == "cuda":
        cuda = getattr(torch_module, "cuda", None)
        available = getattr(cuda, "is_available", None)
        if cuda is None or not callable(available) or not bool(available()):
            raise RuntimeError("CUDA was requested for CLAP but is not available")
        return device
    raise ValueError(f"unsupported CLAP device: {device}")


class TransformersClapBackend:
    """Load pinned CLAP weights and expose batched text/audio embeddings."""

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str,
        device: str,
        text_batch_size: int = 64,
        audio_batch_size: int = 8,
    ) -> None:
        if text_batch_size <= 0 or audio_batch_size <= 0:
            raise ValueError("CLAP batch sizes must be positive")
        self._torch = importlib.import_module("torch")
        self._transformers = importlib.import_module("transformers")
        self._soundfile = importlib.import_module("soundfile")
        self._numpy = importlib.import_module("numpy")
        self._scipy_signal = importlib.import_module("scipy.signal")
        self.device = require_torch_device(self._torch, device)
        self.text_batch_size = text_batch_size
        self.audio_batch_size = audio_batch_size

        started = time.perf_counter()
        self._processor = self._transformers.AutoProcessor.from_pretrained(
            repo_id,
            revision=revision,
        )
        self._model = (
            self._transformers.ClapModel.from_pretrained(
                repo_id,
                revision=revision,
            )
            .to(self.device)
            .eval()
        )
        self.model_load_seconds = time.perf_counter() - started
        feature_extractor = getattr(self._processor, "feature_extractor", None)
        self.sample_rate = int(getattr(feature_extractor, "sampling_rate", 48_000))

    def embed_text(self, texts: list[str]) -> tuple[list[list[float]], float, float]:
        if not texts:
            raise ValueError("CLAP text embedding requires at least one caption")
        vectors: list[list[float]] = []
        preprocessing_seconds = 0.0
        inference_seconds = 0.0
        for start in range(0, len(texts), self.text_batch_size):
            batch = texts[start : start + self.text_batch_size]
            preprocess_started = time.perf_counter()
            inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding=True,
            )
            preprocessing_seconds += time.perf_counter() - preprocess_started

            inference_started = time.perf_counter()
            inputs = _move_to_device(inputs, self.device)
            with self._torch.inference_mode():
                features = self._model.get_text_features(**inputs)
            vectors.extend(_tensor_rows(features))
            inference_seconds += time.perf_counter() - inference_started
        return vectors, preprocessing_seconds, inference_seconds

    def embed_audio(self, paths: list[Path]) -> tuple[list[list[float]], float, float]:
        if not paths:
            raise ValueError("CLAP audio embedding requires at least one WAV")
        vectors: list[list[float]] = []
        preprocessing_seconds = 0.0
        inference_seconds = 0.0
        for start in range(0, len(paths), self.audio_batch_size):
            batch_paths = paths[start : start + self.audio_batch_size]
            preprocess_started = time.perf_counter()
            waveforms = [self._load_audio(path) for path in batch_paths]
            inputs = self._processor(
                audio=waveforms,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            preprocessing_seconds += time.perf_counter() - preprocess_started

            inference_started = time.perf_counter()
            inputs = _move_to_device(inputs, self.device)
            with self._torch.inference_mode():
                features = self._model.get_audio_features(**inputs)
            vectors.extend(_tensor_rows(features))
            inference_seconds += time.perf_counter() - inference_started
        return vectors, preprocessing_seconds, inference_seconds

    def _load_audio(self, path: Path) -> Any:
        waveform, sample_rate = self._soundfile.read(
            str(path),
            dtype="float32",
            always_2d=True,
        )
        if waveform.size == 0:
            raise ValueError(f"CLAP input WAV is empty: {path}")
        mono = self._numpy.mean(waveform, axis=1, dtype=self._numpy.float32)
        if not bool(self._numpy.isfinite(mono).all()):
            raise ValueError(f"CLAP input WAV contains nonfinite samples: {path}")
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"CLAP input WAV has invalid sample rate: {path}")
        if sample_rate != self.sample_rate:
            divisor = math.gcd(sample_rate, self.sample_rate)
            mono = self._scipy_signal.resample_poly(
                mono,
                self.sample_rate // divisor,
                sample_rate // divisor,
            ).astype(self._numpy.float32, copy=False)
        return mono


def _move_to_device(inputs: Any, device: str) -> Any:
    move = getattr(inputs, "to", None)
    if callable(move):
        return move(device)
    return {
        key: value.to(device) if callable(getattr(value, "to", None)) else value
        for key, value in dict(inputs).items()
    }


def _tensor_rows(features: Any) -> list[list[float]]:
    tensor = features
    for attribute in ("text_embeds", "audio_embeds", "pooler_output"):
        candidate = getattr(features, attribute, None)
        if candidate is not None:
            tensor = candidate
            break
    detach = getattr(tensor, "detach", None)
    if not callable(detach):
        raise TypeError("CLAP model did not return an embedding tensor")
    return detach().float().cpu().tolist()
