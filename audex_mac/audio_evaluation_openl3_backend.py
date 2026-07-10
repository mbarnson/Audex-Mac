"""Adapter for the pinned Stability AI FD_OpenL3 implementation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .audio_evaluation_openl3 import STABLE_AUDIO_METRICS_SOURCE_SHA256

IDENTICAL_FD_MAX_ABS = 1e-6
PERMUTATION_FD_MAX_ABS = 1e-6
UNRELATED_FD_MIN = 100.0
FIXED_VECTOR_FD_EXPECTED = 30.0
FIXED_VECTOR_FD_TOLERANCE = 1e-5


class OfficialOpenL3Backend:
    """Run the exact pinned source file with fail-loud self-qualification."""

    def __init__(self, *, implementation_file: Path) -> None:
        actual_sha256 = hashlib.sha256(implementation_file.read_bytes()).hexdigest()
        if actual_sha256 != STABLE_AUDIO_METRICS_SOURCE_SHA256:
            raise ValueError(
                "stable-audio-metrics source SHA-256 mismatch: "
                f"expected {STABLE_AUDIO_METRICS_SOURCE_SHA256}, got {actual_sha256}"
            )
        spec = importlib.util.spec_from_file_location(
            "audex_pinned_stable_audio_metrics_openl3_fd",
            implementation_file,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"cannot load OpenL3 implementation: {implementation_file}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._metric = module
        self._numpy = importlib.import_module("numpy")

    def qualify(self) -> dict[str, Any]:
        numpy = self._numpy
        base = numpy.concatenate(
            (numpy.eye(4, dtype=numpy.float64), -numpy.eye(4, dtype=numpy.float64))
        )
        identical_fd = self._distance(base, base.copy())
        permuted_fd = self._distance(base, base[::-1].copy())
        unrelated_fd = self._distance(base, base + 10.0)
        fixed_vector_fd = self._distance(
            base,
            base + numpy.asarray((1.0, 2.0, 3.0, 4.0)),
        )
        qualified = (
            abs(identical_fd) <= IDENTICAL_FD_MAX_ABS
            and abs(permuted_fd) <= PERMUTATION_FD_MAX_ABS
            and unrelated_fd >= UNRELATED_FD_MIN
            and abs(fixed_vector_fd - FIXED_VECTOR_FD_EXPECTED)
            <= FIXED_VECTOR_FD_TOLERANCE
        )
        return {
            "qualified": qualified,
            "identical_fd": identical_fd,
            "permuted_fd": permuted_fd,
            "unrelated_fd": unrelated_fd,
            "fixed_vector_fd": fixed_vector_fd,
            "thresholds": {
                "identical_fd_max_abs": IDENTICAL_FD_MAX_ABS,
                "permutation_fd_max_abs": PERMUTATION_FD_MAX_ABS,
                "unrelated_fd_min": UNRELATED_FD_MIN,
                "fixed_vector_fd_expected": FIXED_VECTOR_FD_EXPECTED,
                "fixed_vector_fd_tolerance": FIXED_VECTOR_FD_TOLERANCE,
            },
        }

    def score(self, request: Mapping[str, Any]) -> tuple[float, float]:
        started = time.perf_counter()
        score = self._metric.openl3_fd(
            channels=int(request["channels"]),
            samplingrate=int(request["sample_rate"]),
            content_type=str(request["content_type"]),
            openl3_hop_size=float(request["hop_seconds"]),
            eval_path=str(request["generated_dir"]),
            eval_files_extension=".wav",
            load_ref_embeddings=str(request["reference_statistics_path"]),
            batching=int(request["batch_size"]),
        )
        elapsed = time.perf_counter() - started
        value = float(score)
        if not math.isfinite(value):
            raise ValueError(f"FD_OpenL3 must be finite, got {score!r}")
        return value, elapsed

    def _distance(self, left: Any, right: Any) -> float:
        left_mu, left_sigma = self._metric.calculate_embd_statistics(left)
        right_mu, right_sigma = self._metric.calculate_embd_statistics(right)
        return float(
            self._metric.calculate_frechet_distance(
                left_mu,
                left_sigma,
                right_mu,
                right_sigma,
            )
        )
