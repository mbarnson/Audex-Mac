"""Pinned Transformers AST backend for evaluator worker diagnostics."""

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
            raise RuntimeError("MPS was requested for AST but is not available")
        return device
    if device == "cuda":
        cuda = getattr(torch_module, "cuda", None)
        available = getattr(cuda, "is_available", None)
        if cuda is None or not callable(available) or not bool(available()):
            raise RuntimeError("CUDA was requested for AST but is not available")
        return device
    raise ValueError(f"unsupported AST device: {device}")


class TransformersAstBackend:
    """Load pinned AST weights and expose batched AudioSet probabilities."""

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str,
        device: str,
        audio_batch_size: int = 8,
    ) -> None:
        if audio_batch_size <= 0:
            raise ValueError("AST audio batch size must be positive")
        self._torch = importlib.import_module("torch")
        self._transformers = importlib.import_module("transformers")
        self._soundfile = importlib.import_module("soundfile")
        self._numpy = importlib.import_module("numpy")
        self._scipy_signal = importlib.import_module("scipy.signal")
        self.device = require_torch_device(self._torch, device)
        self.audio_batch_size = audio_batch_size

        started = time.perf_counter()
        self._processor = self._transformers.AutoProcessor.from_pretrained(
            repo_id,
            revision=revision,
        )
        self._model = (
            self._transformers.AutoModelForAudioClassification.from_pretrained(
                repo_id,
                revision=revision,
            )
            .to(self.device)
            .eval()
        )
        self.model_load_seconds = time.perf_counter() - started
        feature_extractor = getattr(self._processor, "feature_extractor", None)
        self.sample_rate = int(getattr(feature_extractor, "sampling_rate", 16_000))
        id2label = getattr(getattr(self._model, "config", None), "id2label", None)
        if not isinstance(id2label, dict) or not id2label:
            raise ValueError("AST model config must expose non-empty id2label")
        self.id2label = {
            int(index): str(label)
            for index, label in id2label.items()
            if str(label).strip()
        }
        if not self.id2label:
            raise ValueError("AST id2label map must contain non-empty labels")
        self.labels = frozenset(self.id2label.values())

    def classify_audio(
        self, paths: list[Path]
    ) -> tuple[list[dict[str, float]], float, float]:
        if not paths:
            raise ValueError("AST classification requires at least one WAV")
        rows: list[dict[str, float]] = []
        preprocessing_seconds = 0.0
        inference_seconds = 0.0
        for start in range(0, len(paths), self.audio_batch_size):
            batch_paths = paths[start : start + self.audio_batch_size]
            preprocess_started = time.perf_counter()
            waveforms = [self._load_audio(path) for path in batch_paths]
            inputs = self._processor(
                waveforms,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            preprocessing_seconds += time.perf_counter() - preprocess_started

            inference_started = time.perf_counter()
            inputs = _move_to_device(inputs, self.device)
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
                probabilities = self._torch.sigmoid(outputs.logits)
            rows.extend(_probability_rows(probabilities, self.id2label))
            inference_seconds += time.perf_counter() - inference_started
        return rows, preprocessing_seconds, inference_seconds

    def _load_audio(self, path: Path) -> Any:
        waveform, sample_rate = self._soundfile.read(
            str(path),
            dtype="float32",
            always_2d=True,
        )
        if waveform.size == 0:
            raise ValueError(f"AST input WAV is empty: {path}")
        mono = self._numpy.mean(waveform, axis=1, dtype=self._numpy.float32)
        if not bool(self._numpy.isfinite(mono).all()):
            raise ValueError(f"AST input WAV contains nonfinite samples: {path}")
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"AST input WAV has invalid sample rate: {path}")
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


def _probability_rows(
    probabilities: Any,
    id2label: dict[int, str],
) -> list[dict[str, float]]:
    tensor = probabilities.detach().float().cpu()
    rows = tensor.tolist()
    expected_width = max(id2label) + 1
    output: list[dict[str, float]] = []
    for row in rows:
        if len(row) < expected_width:
            raise ValueError("AST probability vector is narrower than id2label")
        output.append(
            {
                label: float(row[index])
                for index, label in sorted(id2label.items())
                if index < len(row)
            }
        )
    return output
