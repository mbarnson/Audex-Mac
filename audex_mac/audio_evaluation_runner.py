"""Track-agnostic orchestration for one autonomous audio evaluation run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .audio_evaluation import (
    AudioEvaluationCase,
    AudioEvaluationRun,
    AudioEvaluationSummary,
    EvaluationTrack,
    derive_case_seed,
    score_constrained_answer,
)
from .audio_evaluation_generation import (
    NVIDIA_TTA_CFG_PAIRS_PER_BATCH,
    TtaOutputInspection,
)


@dataclass(frozen=True, slots=True)
class UnderstandingAttempt:
    raw_answer: str
    elapsed_seconds: float
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class GenerationAttempt:
    raw_wav_path: Path
    enhanced_wav_path: Path | None
    structure: TtaOutputInspection
    signal_metrics: Mapping[str, Any]
    elapsed_seconds: float
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class OracleQualification:
    qualified: bool
    oracle_results: Mapping[str, Any]
    failures: tuple[str, ...]


class UnderstandingAdapter(Protocol):
    def answer(
        self, case: AudioEvaluationCase, *, seed: int
    ) -> UnderstandingAttempt: ...


class GenerationAdapter(Protocol):
    def generate_many(
        self,
        cases: tuple[tuple[AudioEvaluationCase, int], ...],
    ) -> tuple[GenerationAttempt, ...]: ...


class OracleSuite(Protocol):
    def qualify(self) -> OracleQualification: ...

    def score(
        self, case: AudioEvaluationCase, attempt: GenerationAttempt
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class UnqualifiedOracleSuite:
    """Generation oracle placeholder that fails closed until local oracles exist."""

    reason: str = "generation_oracles_not_qualified"

    def qualify(self) -> OracleQualification:
        return OracleQualification(
            qualified=False,
            oracle_results={"status": self.reason},
            failures=(self.reason,),
        )

    def score(
        self, case: AudioEvaluationCase, attempt: GenerationAttempt
    ) -> Mapping[str, Any]:
        del case, attempt
        return {"verdict": "UNSCORED", "reason": self.reason}


class AudioEvaluationRunner:
    """Execute isolated cases while keeping model/oracle implementations hidden."""

    def __init__(
        self,
        *,
        understanding: UnderstandingAdapter,
        generation: GenerationAdapter,
        oracles: OracleSuite,
    ) -> None:
        self._understanding = understanding
        self._generation = generation
        self._oracles = oracles

    def run(
        self,
        run: AudioEvaluationRun,
        *,
        master_seed: int,
        capability_targets: Mapping[str, float] | None = None,
    ) -> AudioEvaluationSummary:
        generation_cases = tuple(
            case for case in run.cases if case.track is EvaluationTrack.GENERATION
        )
        qualification = (
            self._oracles.qualify()
            if generation_cases
            else OracleQualification(qualified=True, oracle_results={}, failures=())
        )
        if generation_cases:
            run.record_oracle_qualification(
                {
                    "qualified": qualification.qualified,
                    "oracle_results": dict(qualification.oracle_results),
                    "failures": list(qualification.failures),
                }
            )
        protocol_failures = list(qualification.failures)
        generation_results = self._generate_in_reference_waves(
            generation_cases,
            master_seed=master_seed,
        )
        for case in run.cases:
            seed = derive_case_seed(master_seed, case.case_id)
            try:
                if case.track is EvaluationTrack.UNDERSTANDING:
                    self._run_understanding_case(run, case, seed=seed)
                else:
                    generation_result = generation_results[case.case_id]
                    if isinstance(generation_result, Exception):
                        raise generation_result
                    case_failures = self._record_generation_case(
                        run,
                        case,
                        seed=seed,
                        attempt=generation_result,
                        oracles_qualified=qualification.qualified,
                    )
                    protocol_failures.extend(case_failures)
            except Exception as exc:
                failure = f"{case.case_id}: {type(exc).__name__}: {exc}"
                protocol_failures.append(failure)
                run.record_output(
                    case_id=case.case_id,
                    payload={
                        "attempt": 1,
                        "seed": seed,
                        "technical_failure": failure,
                    },
                )
        return run.finalize(
            required_oracles_qualified=qualification.qualified,
            protocol_failures=tuple(protocol_failures),
            capability_targets=capability_targets,
        )

    def _generate_in_reference_waves(
        self,
        cases: tuple[AudioEvaluationCase, ...],
        *,
        master_seed: int,
    ) -> dict[str, GenerationAttempt | Exception]:
        results: dict[str, GenerationAttempt | Exception] = {}
        for start in range(0, len(cases), NVIDIA_TTA_CFG_PAIRS_PER_BATCH):
            wave = cases[start : start + NVIDIA_TTA_CFG_PAIRS_PER_BATCH]
            seeded = tuple(
                (case, derive_case_seed(master_seed, case.case_id)) for case in wave
            )
            try:
                attempts = self._generation.generate_many(seeded)
                if len(attempts) != len(wave):
                    raise ValueError(
                        "generation adapter returned "
                        f"{len(attempts)} attempts for {len(wave)} cases"
                    )
            except Exception as exc:
                for case in wave:
                    results[case.case_id] = exc
                continue
            for case, attempt in zip(wave, attempts, strict=True):
                results[case.case_id] = attempt
        return results

    def _run_understanding_case(
        self,
        run: AudioEvaluationRun,
        case: AudioEvaluationCase,
        *,
        seed: int,
    ) -> None:
        attempt = self._understanding.answer(case, seed=seed)
        assert case.expected_answer is not None
        score = score_constrained_answer(
            attempt.raw_answer,
            choices=case.choices,
            expected=case.expected_answer,
        )
        run.record_output(
            case_id=case.case_id,
            payload={
                "attempt": 1,
                "seed": seed,
                "raw_answer": attempt.raw_answer,
                "normalized_answer": score.normalized_answer,
                "valid": score.valid,
                "correct": score.correct,
                "elapsed_seconds": attempt.elapsed_seconds,
                "finish_reason": attempt.finish_reason,
            },
        )

    def _record_generation_case(
        self,
        run: AudioEvaluationRun,
        case: AudioEvaluationCase,
        *,
        seed: int,
        attempt: GenerationAttempt,
        oracles_qualified: bool,
    ) -> tuple[str, ...]:
        signal_finite = bool(attempt.signal_metrics.get("finite", False))
        signal_nonempty = bool(attempt.signal_metrics.get("nonempty", False))
        waveform_exists = attempt.raw_wav_path.is_file()
        failures = [
            f"{case.case_id}: {failure}" for failure in attempt.structure.failures
        ]
        if not signal_finite:
            failures.append(f"{case.case_id}: nonfinite_waveform")
        if not signal_nonempty:
            failures.append(f"{case.case_id}: empty_waveform")
        if not waveform_exists:
            failures.append(f"{case.case_id}: missing_raw_wav")
        run.record_output(
            case_id=case.case_id,
            payload={
                "attempt": 1,
                "seed": seed,
                "raw_wav_path": str(attempt.raw_wav_path),
                "enhanced_wav_path": (
                    str(attempt.enhanced_wav_path)
                    if attempt.enhanced_wav_path is not None
                    else None
                ),
                "codec_token_count": attempt.structure.codec_token_count,
                "frame_count": attempt.structure.frame_count,
                "duration_seconds": attempt.structure.duration_seconds,
                "reached_end_token": attempt.structure.reached_end_token,
                "structurally_valid": attempt.structure.valid,
                "structure_failures": list(attempt.structure.failures),
                "signal_metrics": dict(attempt.signal_metrics),
                "elapsed_seconds": attempt.elapsed_seconds,
                "finish_reason": attempt.finish_reason,
            },
        )
        metric_failures: tuple[str, ...] = ()
        if oracles_qualified and not failures:
            metrics = dict(self._oracles.score(case, attempt))
            metric_failures = _metric_protocol_failures(metrics)
        else:
            metrics = {
                "verdict": "UNSCORED",
                "reason": (
                    "oracle_not_qualified"
                    if not oracles_qualified
                    else "invalid_output"
                ),
            }
        run.record_generation_metrics(case_id=case.case_id, payload=metrics)
        failures.extend(f"{case.case_id}: {failure}" for failure in metric_failures)
        return tuple(failures)


def _metric_protocol_failures(metrics: Mapping[str, Any]) -> tuple[str, ...]:
    raw_failures = metrics.get("protocol_failures", ())
    if not isinstance(raw_failures, (list, tuple)):
        return ()
    return tuple(str(failure) for failure in raw_failures if str(failure).strip())
