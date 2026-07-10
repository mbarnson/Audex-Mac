"""Local generation oracle suites for autonomous audio evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .audio_evaluation import AudioEvaluationCase
from .audio_evaluation_runner import (
    GenerationAttempt,
    OracleQualification,
)


@dataclass(frozen=True, slots=True)
class SignalSanityConfig:
    min_duration_seconds: float = 0.25
    max_duration_seconds: float = 10.5
    min_peak: float = 0.001
    max_peak: float = 0.999
    min_rms: float = 0.0005
    max_abs_dc_offset: float = 0.25
    min_sample_delta_peak: float = 0.00001


@dataclass(frozen=True, slots=True)
class SignalSanityOracleSuite:
    """Smoke-tier local oracle for waveform-level generation sanity."""

    config: SignalSanityConfig = SignalSanityConfig()

    def qualify(self) -> OracleQualification:
        passing = self._score_metrics(
            {
                "finite": True,
                "nonempty": True,
                "duration_seconds": 5.0,
                "peak": 0.5,
                "rms": 0.1,
                "dc_offset": 0.0,
                "sample_delta_peak": 0.1,
                "clipped": False,
            }
        )
        silent = self._score_metrics(
            {
                "finite": True,
                "nonempty": True,
                "duration_seconds": 5.0,
                "peak": 0.0,
                "rms": 0.0,
                "dc_offset": 0.0,
                "sample_delta_peak": 0.0,
                "clipped": False,
            }
        )
        clipped = self._score_metrics(
            {
                "finite": True,
                "nonempty": True,
                "duration_seconds": 5.0,
                "peak": 1.0,
                "rms": 0.8,
                "dc_offset": 0.0,
                "sample_delta_peak": 0.1,
                "clipped": True,
            }
        )
        dc_biased = self._score_metrics(
            {
                "finite": True,
                "nonempty": True,
                "duration_seconds": 5.0,
                "peak": 0.5,
                "rms": 0.3,
                "dc_offset": 0.4,
                "sample_delta_peak": 0.1,
                "clipped": False,
            }
        )
        flat = self._score_metrics(
            {
                "finite": True,
                "nonempty": True,
                "duration_seconds": 5.0,
                "peak": 0.5,
                "rms": 0.5,
                "dc_offset": 0.0,
                "sample_delta_peak": 0.0,
                "clipped": False,
            }
        )
        qualified = bool(
            passing["verdict"] == "PASS"
            and silent["verdict"] == "FAIL"
            and clipped["verdict"] == "FAIL"
            and dc_biased["verdict"] == "FAIL"
            and flat["verdict"] == "FAIL"
        )
        failures = () if qualified else ("signal_sanity_self_test_failed",)
        return OracleQualification(
            qualified=qualified,
            oracle_results={
                "signal_sanity": {
                    "qualified": qualified,
                    "config": self._config_payload(),
                    "self_tests": {
                        "passing": passing,
                        "silent": silent,
                        "clipped": clipped,
                        "dc_biased": dc_biased,
                        "flat": flat,
                    },
                }
            },
            failures=failures,
        )

    def score(
        self,
        case: AudioEvaluationCase,
        attempt: GenerationAttempt,
    ) -> Mapping[str, Any]:
        del case
        score = self._score_metrics(attempt.signal_metrics)
        return {
            "oracle": "signal_sanity",
            "config": self._config_payload(),
            "verdict": score["verdict"],
            "checks": score["checks"],
            "protocol_failures": score["protocol_failures"],
        }

    def _score_metrics(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        duration = _float_metric(metrics, "duration_seconds")
        peak = _float_metric(metrics, "peak")
        rms = _float_metric(metrics, "rms")
        dc_offset = _float_metric(metrics, "dc_offset")
        sample_delta_peak = _float_metric(metrics, "sample_delta_peak")
        checks = {
            "finite": bool(metrics.get("finite", False)),
            "nonempty": bool(metrics.get("nonempty", False)),
            "duration_in_range": (
                duration is not None
                and self.config.min_duration_seconds
                <= duration
                <= self.config.max_duration_seconds
            ),
            "audible_peak": peak is not None and peak >= self.config.min_peak,
            "audible_rms": rms is not None and rms >= self.config.min_rms,
            "dc_offset_in_range": (
                dc_offset is not None
                and abs(dc_offset) <= self.config.max_abs_dc_offset
            ),
            "sample_variation": (
                sample_delta_peak is not None
                and sample_delta_peak >= self.config.min_sample_delta_peak
            ),
            "not_clipped": (
                peak is not None
                and peak <= self.config.max_peak
                and not bool(metrics.get("clipped", False))
            ),
        }
        failures = tuple(name for name, passed in checks.items() if not passed)
        return {
            "verdict": "PASS" if not failures else "FAIL",
            "checks": checks,
            "protocol_failures": failures,
        }

    def _config_payload(self) -> dict[str, float]:
        return {
            "min_duration_seconds": self.config.min_duration_seconds,
            "max_duration_seconds": self.config.max_duration_seconds,
            "min_peak": self.config.min_peak,
            "max_peak": self.config.max_peak,
            "min_rms": self.config.min_rms,
            "max_abs_dc_offset": self.config.max_abs_dc_offset,
            "min_sample_delta_peak": self.config.min_sample_delta_peak,
        }


def _float_metric(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
