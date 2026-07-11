"""Core contracts and local artifacts for autonomous audio evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import resource
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class EvaluationTrack(StrEnum):
    UNDERSTANDING = "understanding"
    GENERATION = "generation"


class RunVerdict(StrEnum):
    PASS = "PASS"
    CAPABILITY_FAIL = "CAPABILITY_FAIL"
    PROTOCOL_FAIL = "PROTOCOL_FAIL"
    CHARACTERIZED = "CHARACTERIZED"
    UNSCORED = "UNSCORED"


@dataclass(frozen=True, slots=True)
class AudioEvaluationCase:
    case_id: str
    track: EvaluationTrack
    dataset_id: str
    dataset_revision: str
    dataset_config: str
    dataset_split: str
    source_row_id: str
    source_row_hash: str
    license: str
    category: str
    prompt: str
    expected_answer: str | None = None
    audio_path: str | None = None
    caption: str | None = None
    hard_foil_caption: str | None = None
    choices: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = {
            "case_id": self.case_id,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "dataset_config": self.dataset_config,
            "dataset_split": self.dataset_split,
            "source_row_id": self.source_row_id,
            "source_row_hash": self.source_row_hash,
            "license": self.license,
            "category": self.category,
            "prompt": self.prompt,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"audio evaluation case has empty fields: {missing}")
        empty_tags = [tag for tag in self.tags if not str(tag).strip()]
        if empty_tags:
            raise ValueError("audio evaluation case has empty tags")
        if self.track is EvaluationTrack.UNDERSTANDING:
            if not self.audio_path:
                raise ValueError("understanding cases require audio_path")
            if not self.expected_answer:
                raise ValueError("understanding cases require expected_answer")
            if not self.choices:
                raise ValueError("understanding cases require choices")
        elif not self.caption:
            raise ValueError("generation cases require caption")


@dataclass(frozen=True, slots=True)
class ConstrainedAnswerScore:
    normalized_answer: str | None
    valid: bool
    correct: bool


@dataclass(frozen=True, slots=True)
class AudioEvaluationSummary:
    verdict: RunVerdict
    total_cases: int
    completed_cases: int
    missing_case_ids: tuple[str, ...]
    case_completeness: float
    accuracy: float | None
    invalid_response_rate: float | None
    confidence_intervals: Mapping[str, Any]
    understanding_by_category: Mapping[str, Mapping[str, Any]]
    balanced_accuracy: float | None
    binary_rates: Mapping[str, Any]
    generation: Mapping[str, Any]
    by_tag: Mapping[str, Mapping[str, Any]]
    technical_failures: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    protocol_failures: tuple[str, ...]
    capability_targets: Mapping[str, float]
    capability_failures: tuple[str, ...]


def derive_case_seed(master_seed: int, case_id: str) -> int:
    """Derive a stable unsigned 64-bit seed without depending on Python hash."""

    digest = hashlib.sha256(f"{int(master_seed)}\0{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def select_stratified_cases(
    cases: Iterable[AudioEvaluationCase],
    *,
    count: int,
    master_seed: int,
) -> tuple[AudioEvaluationCase, ...]:
    """Select a deterministic equal share from every case category."""

    all_cases = tuple(cases)
    if count <= 0:
        raise ValueError("selection count must be positive")
    by_category: dict[str, list[AudioEvaluationCase]] = {}
    seen_ids: set[str] = set()
    for case in all_cases:
        if case.case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case.case_id}")
        seen_ids.add(case.case_id)
        by_category.setdefault(case.category, []).append(case)
    if not by_category:
        raise ValueError("cannot select from an empty case set")
    if count % len(by_category):
        raise ValueError(
            f"count {count} cannot be balanced across {len(by_category)} strata"
        )

    share = count // len(by_category)
    selected: list[AudioEvaluationCase] = []
    for category in sorted(by_category):
        available = by_category[category]
        if len(available) < share:
            raise ValueError(
                f"stratum {category!r} requires {share} cases but has {len(available)}"
            )
        ranked = sorted(
            available,
            key=lambda case: (
                hashlib.sha256(
                    f"{int(master_seed)}\0{case.case_id}".encode()
                ).hexdigest(),
                case.case_id,
            ),
        )
        selected.extend(ranked[:share])
    return tuple(sorted(selected, key=lambda case: case.case_id))


def score_constrained_answer(
    raw_answer: str,
    *,
    choices: tuple[str, ...],
    expected: str,
) -> ConstrainedAnswerScore:
    """Score one exact constrained answer; prose and multiple answers fail closed."""

    allowed = {choice.strip().upper() for choice in choices}
    match = re.fullmatch(
        r"\s*(?:answer\s*:\s*)?([A-Za-z0-9]+)[.!]?\s*",
        raw_answer,
        flags=re.IGNORECASE,
    )
    normalized = match.group(1).upper() if match is not None else None
    valid = normalized in allowed
    return ConstrainedAnswerScore(
        normalized_answer=normalized if valid else None,
        valid=valid,
        correct=bool(valid and normalized == expected.strip().upper()),
    )


class AudioEvaluationRun:
    """Own one immutable case manifest and its append-only local outputs."""

    def __init__(
        self,
        *,
        run_dir: Path,
        tier: str,
        cases: tuple[AudioEvaluationCase, ...],
    ) -> None:
        self.run_dir = run_dir
        self.tier = tier
        self.cases = cases
        self.manifest_path = run_dir / "manifest.json"
        self.environment_path = run_dir / "environment.json"
        self.summary_path = run_dir / "summary.json"
        self.oracle_qualification_path = (
            run_dir / "generation" / "oracle_qualification.json"
        )
        self._case_by_id = {case.case_id: case for case in cases}
        self._completed_case_ids: set[str] = set()
        self._started_at_utc = _utc_now()
        self._started_monotonic = time.monotonic()

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        run_id: str,
        tier: str,
        master_seed: int,
        cases: tuple[AudioEvaluationCase, ...],
        manifest_metadata: Mapping[str, Any],
        environment: Mapping[str, Any] | None = None,
    ) -> AudioEvaluationRun:
        if not run_id.strip() or Path(run_id).name != run_id:
            raise ValueError("run_id must be one non-empty path component")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("evaluation cases must have unique case ids")
        if not cases:
            raise ValueError("evaluation run requires at least one case")
        _reject_credentials(manifest_metadata)
        _reject_credentials(environment or {})

        run_dir = root / run_id
        if run_dir.exists():
            raise ValueError(f"evaluation run already exists: {run_dir}")
        for relative in (
            "understanding",
            "generation",
            "media/raw",
            "media/enhanced",
        ):
            (run_dir / relative).mkdir(parents=True, exist_ok=True)

        run = cls(run_dir=run_dir, tier=tier, cases=cases)
        cases_by_track = {
            track: tuple(case for case in cases if case.track is track)
            for track in EvaluationTrack
        }
        for track, track_cases in cases_by_track.items():
            case_path = run_dir / track.value / "cases.jsonl"
            _write_jsonl(case_path, (_case_payload(case) for case in track_cases))
            (run_dir / track.value / "outputs.jsonl").touch()
        (run_dir / "generation" / "metrics.jsonl").touch()

        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "tier": tier,
            "master_seed": int(master_seed),
            "case_count": len(cases),
            "case_ids": [case.case_id for case in cases],
            **dict(manifest_metadata),
        }
        run.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run.environment_path.write_text(
            json.dumps(dict(environment or {}), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return run

    def record_output(self, *, case_id: str, payload: Mapping[str, Any]) -> None:
        case = self._case_by_id.get(case_id)
        if case is None:
            raise ValueError(f"output references unknown case: {case_id}")
        if case_id in self._completed_case_ids:
            raise ValueError(f"output already recorded for case: {case_id}")
        _reject_credentials(payload)
        output_path = self.run_dir / case.track.value / "outputs.jsonl"
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"case_id": case_id, **dict(payload)},
                    sort_keys=True,
                )
                + "\n"
            )
        self._completed_case_ids.add(case_id)

    def record_generation_metrics(
        self, *, case_id: str, payload: Mapping[str, Any]
    ) -> None:
        case = self._case_by_id.get(case_id)
        if case is None:
            raise ValueError(f"metrics reference unknown case: {case_id}")
        if case.track is not EvaluationTrack.GENERATION:
            raise ValueError(f"metrics require a generation case: {case_id}")
        _reject_credentials(payload)
        path = self.run_dir / "generation" / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"case_id": case_id, **dict(payload)}, sort_keys=True) + "\n"
            )

    def record_oracle_qualification(self, payload: Mapping[str, Any]) -> None:
        _reject_credentials(payload)
        self.oracle_qualification_path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def finalize(
        self,
        *,
        required_oracles_qualified: bool,
        protocol_failures: tuple[str, ...] = (),
        capability_targets: Mapping[str, float] | None = None,
    ) -> AudioEvaluationSummary:
        missing = tuple(
            case.case_id
            for case in self.cases
            if case.case_id not in self._completed_case_ids
        )
        outputs = self._load_outputs()
        outputs_by_case_id = {
            str(output["case_id"]): output
            for output in outputs
            if str(output.get("case_id", "")).strip()
        }
        scored = [output for output in outputs if "correct" in output]
        accuracy = (
            sum(bool(output["correct"]) for output in scored) / len(scored)
            if scored
            else None
        )
        invalid_rate = (
            sum(not bool(output.get("valid", False)) for output in scored) / len(scored)
            if scored
            else None
        )
        effective_failures = list(protocol_failures)
        if not required_oracles_qualified:
            effective_failures.append("required_oracle_qualification_failed")
        if missing:
            effective_failures.append("incomplete_cases")
        understanding_by_category = _understanding_by_category(
            self.cases,
            outputs_by_case_id,
        )
        balanced_accuracy = _balanced_accuracy(understanding_by_category)
        generation_metrics = self._load_generation_metrics()
        generation = _generation_summary(
            self.cases,
            outputs_by_case_id,
            generation_metrics,
        )
        technical_failures = _technical_failure_summary(
            self.cases,
            outputs_by_case_id,
        )
        by_tag = _by_tag_summary(
            self.cases,
            outputs_by_case_id,
            generation_metrics,
        )
        target_payload = dict(capability_targets or {})
        capability_failures = (
            ()
            if effective_failures
            else _evaluate_capability_targets(
                targets=target_payload,
                metrics={
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_accuracy,
                    "invalid_response_rate": invalid_rate,
                    "technical_failure_rate": technical_failures.get(
                        "technical_failure_rate"
                    ),
                    "generation_structural_failure_rate": (
                        (
                            int(generation["completed_cases"])
                            - int(generation["structurally_valid"])
                        )
                        / int(generation["completed_cases"])
                        if int(generation["completed_cases"])
                        else None
                    ),
                },
            )
        )
        if effective_failures:
            verdict = RunVerdict.PROTOCOL_FAIL
        elif target_payload and capability_failures:
            verdict = RunVerdict.CAPABILITY_FAIL
        elif target_payload:
            verdict = RunVerdict.PASS
        else:
            verdict = RunVerdict.CHARACTERIZED
        summary = AudioEvaluationSummary(
            verdict=verdict,
            total_cases=len(self.cases),
            completed_cases=len(self._completed_case_ids),
            missing_case_ids=missing,
            case_completeness=len(self._completed_case_ids) / len(self.cases),
            accuracy=accuracy,
            invalid_response_rate=invalid_rate,
            confidence_intervals=_confidence_intervals(scored),
            understanding_by_category=understanding_by_category,
            balanced_accuracy=balanced_accuracy,
            binary_rates=_binary_rates(self.cases, outputs_by_case_id),
            generation=generation,
            by_tag=by_tag,
            technical_failures=technical_failures,
            diagnostics=_diagnostics_summary(
                self.cases,
                outputs_by_case_id,
                started_at_utc=self._started_at_utc,
                wall_clock_seconds=time.monotonic() - self._started_monotonic,
            ),
            protocol_failures=tuple(dict.fromkeys(effective_failures)),
            capability_targets=target_payload,
            capability_failures=capability_failures,
        )
        self.summary_path.write_text(
            json.dumps(_summary_payload(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def _load_outputs(self) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for track in EvaluationTrack:
            path = self.run_dir / track.value / "outputs.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    outputs.append(json.loads(line))
        return outputs

    def _load_generation_metrics(self) -> list[dict[str, Any]]:
        path = self.run_dir / "generation" / "metrics.jsonl"
        metrics: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                metrics.append(json.loads(line))
        return metrics


def _case_payload(case: AudioEvaluationCase) -> dict[str, Any]:
    payload = asdict(case)
    payload["track"] = case.track.value
    payload["choices"] = list(case.choices)
    payload["tags"] = list(case.tags)
    return payload


def _summary_payload(summary: AudioEvaluationSummary) -> dict[str, Any]:
    payload = asdict(summary)
    payload["verdict"] = summary.verdict.value
    payload["missing_case_ids"] = list(summary.missing_case_ids)
    payload["protocol_failures"] = list(summary.protocol_failures)
    return payload


def _understanding_by_category(
    cases: tuple[AudioEvaluationCase, ...],
    outputs_by_case_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_category: dict[str, dict[str, Any]] = {}
    for case in cases:
        if case.track is not EvaluationTrack.UNDERSTANDING:
            continue
        bucket = by_category.setdefault(
            case.category,
            {
                "total_cases": 0,
                "completed_cases": 0,
                "valid_responses": 0,
                "correct": 0,
                "accuracy": None,
                "invalid_response_rate": None,
            },
        )
        bucket["total_cases"] += 1
        output = outputs_by_case_id.get(case.case_id)
        if output is None or "correct" not in output:
            continue
        bucket["completed_cases"] += 1
        if bool(output.get("valid", False)):
            bucket["valid_responses"] += 1
        if bool(output.get("correct", False)):
            bucket["correct"] += 1

    for bucket in by_category.values():
        completed = int(bucket["completed_cases"])
        if completed:
            bucket["accuracy"] = int(bucket["correct"]) / completed
            bucket["invalid_response_rate"] = (
                completed - int(bucket["valid_responses"])
            ) / completed
    return dict(sorted(by_category.items()))


def _balanced_accuracy(
    category_summary: Mapping[str, Mapping[str, Any]],
) -> float | None:
    accuracies = [
        float(summary["accuracy"])
        for summary in category_summary.values()
        if summary.get("accuracy") is not None
    ]
    return sum(accuracies) / len(accuracies) if accuracies else None


def _confidence_intervals(scored_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy_ci = _bootstrap_accuracy_ci(scored_outputs)
    return {"accuracy": accuracy_ci} if accuracy_ci is not None else {}


def _binary_rates(
    cases: tuple[AudioEvaluationCase, ...],
    outputs_by_case_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    positive_cases = 0
    negative_cases = 0
    false_positives = 0
    false_negatives = 0
    completed_cases = 0
    for case in cases:
        choices = {choice.upper() for choice in case.choices}
        if (
            case.track is not EvaluationTrack.UNDERSTANDING
            or choices != {"YES", "NO"}
            or case.expected_answer is None
        ):
            continue
        expected = case.expected_answer.strip().upper()
        if expected not in {"YES", "NO"}:
            continue
        output = outputs_by_case_id.get(case.case_id)
        if output is None:
            continue
        completed_cases += 1
        normalized = output.get("normalized_answer")
        answer = str(normalized).strip().upper() if normalized is not None else None
        if expected == "YES":
            positive_cases += 1
            if answer != "YES":
                false_negatives += 1
        else:
            negative_cases += 1
            if answer == "YES":
                false_positives += 1
    return {
        "completed_cases": completed_cases,
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "false_positive_rate": (
            false_positives / negative_cases if negative_cases else None
        ),
        "false_negative_rate": (
            false_negatives / positive_cases if positive_cases else None
        ),
    }


def _bootstrap_accuracy_ci(
    scored_outputs: list[dict[str, Any]],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
) -> dict[str, Any] | None:
    values = [
        1.0 if bool(output.get("correct", False)) else 0.0 for output in scored_outputs
    ]
    if not values:
        return None
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    seed_material = "|".join(
        f"{output.get('case_id')}={int(bool(output.get('correct', False)))}"
        for output in sorted(scored_outputs, key=lambda item: str(item.get("case_id")))
    )
    n = len(values)
    estimates: list[float] = []
    for sample_index in range(samples):
        total = 0.0
        for draw_index in range(n):
            digest = hashlib.sha256(
                f"{seed_material}\0{sample_index}\0{draw_index}".encode()
            ).digest()
            total += values[int.from_bytes(digest[:8], "big") % n]
        estimates.append(total / n)
    estimates.sort()
    alpha = 1.0 - confidence
    lower_index = int((alpha / 2.0) * (samples - 1))
    upper_index = int((1.0 - alpha / 2.0) * (samples - 1))
    return {
        "method": "deterministic_nonparametric_bootstrap",
        "confidence": confidence,
        "samples": samples,
        "lower": estimates[lower_index],
        "upper": estimates[upper_index],
    }


def _generation_summary(
    cases: tuple[AudioEvaluationCase, ...],
    outputs_by_case_id: Mapping[str, Mapping[str, Any]],
    metric_outputs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    generation_cases = [
        case for case in cases if case.track is EvaluationTrack.GENERATION
    ]
    completed_outputs = [
        outputs_by_case_id[case.case_id]
        for case in generation_cases
        if case.case_id in outputs_by_case_id
    ]
    structural_failures: dict[str, int] = {}
    signal_failures: dict[str, int] = {}
    for output in completed_outputs:
        for failure in output.get("structure_failures", ()):
            key = str(failure)
            structural_failures[key] = structural_failures.get(key, 0) + 1
        signal_metrics = output.get("signal_metrics", {})
        if isinstance(signal_metrics, Mapping):
            if not bool(signal_metrics.get("finite", True)):
                signal_failures["nonfinite_waveform"] = (
                    signal_failures.get("nonfinite_waveform", 0) + 1
                )
            if not bool(signal_metrics.get("nonempty", True)):
                signal_failures["empty_waveform"] = (
                    signal_failures.get("empty_waveform", 0) + 1
                )
            if bool(signal_metrics.get("clipped", False)):
                signal_failures["clipped_waveform"] = (
                    signal_failures.get("clipped_waveform", 0) + 1
                )
    return {
        "total_cases": len(generation_cases),
        "completed_cases": len(completed_outputs),
        "structurally_valid": sum(
            bool(output.get("structurally_valid", False))
            for output in completed_outputs
        ),
        "structural_failures": dict(sorted(structural_failures.items())),
        "signal_failures": dict(sorted(signal_failures.items())),
        "semantic_metrics": _semantic_generation_metrics(metric_outputs),
    }


def _semantic_generation_metrics(
    metric_outputs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = tuple(metric_outputs)
    caption_similarity = _numeric_values(metrics, "caption_similarity")
    hard_foil_wins = _bool_values(metrics, "hard_foil_win")
    hard_foil_margins = _numeric_values(metrics, "hard_foil_margin")
    retrieval_top1 = _retrieval_top1_values(metrics)
    ast_expected_hits = _bool_values(metrics, "expected_label_hit")
    ast_forbidden_fps = _bool_values(metrics, "forbidden_label_false_positive")
    return {
        "clap": {
            "scored_cases": len(caption_similarity),
            "mean_caption_similarity": _mean(caption_similarity),
            "hard_foil_cases": len(hard_foil_wins),
            "hard_foil_win_rate": _mean(hard_foil_wins),
            "mean_hard_foil_margin": _mean(hard_foil_margins),
            "retrieval_top1_cases": len(retrieval_top1),
            "retrieval_top1_rate": _mean(retrieval_top1),
        },
        "ast": {
            "expected_label_cases": len(ast_expected_hits),
            "expected_label_hit_rate": _mean(ast_expected_hits),
            "forbidden_label_cases": len(ast_forbidden_fps),
            "forbidden_label_false_positive_rate": _mean(ast_forbidden_fps),
        },
        "openl3": {
            "fd_openl3_by_dataset": _fd_openl3_by_dataset(metrics),
        },
    }


def _numeric_values(
    metrics: Iterable[Mapping[str, Any]],
    key: str,
) -> tuple[float, ...]:
    values: list[float] = []
    for metric in metrics:
        value = metric.get(key)
        if _is_number(value):
            values.append(float(value))
    return tuple(values)


def _bool_values(
    metrics: Iterable[Mapping[str, Any]],
    key: str,
) -> tuple[float, ...]:
    values: list[float] = []
    for metric in metrics:
        value = metric.get(key)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
    return tuple(values)


def _retrieval_top1_values(metrics: Iterable[Mapping[str, Any]]) -> tuple[float, ...]:
    values: list[float] = []
    for metric in metrics:
        rank = metric.get("retrieval_rank")
        if _is_number(rank):
            values.append(1.0 if int(float(rank)) == 1 else 0.0)
    return tuple(values)


def _fd_openl3_by_dataset(metrics: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in metrics:
        nested = metric.get("fd_openl3_by_dataset")
        if isinstance(nested, Mapping):
            for dataset, value in nested.items():
                if _is_number(value):
                    values[str(dataset)] = float(value)
        dataset = metric.get("dataset")
        fd_openl3 = metric.get("fd_openl3")
        if dataset is not None and _is_number(fd_openl3):
            values[str(dataset)] = float(fd_openl3)
    return dict(sorted(values.items()))


def _mean(values: tuple[float, ...]) -> float | None:
    return sum(values) / len(values) if values else None


def _by_tag_summary(
    cases: tuple[AudioEvaluationCase, ...],
    outputs_by_case_id: Mapping[str, Mapping[str, Any]],
    metric_outputs: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    metrics_by_case_id: dict[str, list[Mapping[str, Any]]] = {}
    for metric in metric_outputs:
        case_id = str(metric.get("case_id", "")).strip()
        if case_id:
            metrics_by_case_id.setdefault(case_id, []).append(metric)

    summaries: dict[str, dict[str, Any]] = {}
    for tag in sorted({tag for case in cases for tag in case.tags}):
        tag_cases = tuple(case for case in cases if tag in case.tags)
        tag_outputs_by_case_id = {
            case.case_id: outputs_by_case_id[case.case_id]
            for case in tag_cases
            if case.case_id in outputs_by_case_id
        }
        tag_metrics: list[Mapping[str, Any]] = []
        for case in tag_cases:
            tag_metrics.extend(metrics_by_case_id.get(case.case_id, ()))

        scored_outputs = [
            output for output in tag_outputs_by_case_id.values() if "correct" in output
        ]
        generation_cases = tuple(
            case for case in tag_cases if case.track is EvaluationTrack.GENERATION
        )
        generation_outputs = [
            tag_outputs_by_case_id[case.case_id]
            for case in generation_cases
            if case.case_id in tag_outputs_by_case_id
        ]
        technical = _technical_failure_summary(tag_cases, tag_outputs_by_case_id)
        summaries[tag] = {
            "total_cases": len(tag_cases),
            "completed_cases": len(tag_outputs_by_case_id),
            "understanding_cases": sum(
                1 for case in tag_cases if case.track is EvaluationTrack.UNDERSTANDING
            ),
            "generation_cases": len(generation_cases),
            "understanding_accuracy": (
                sum(bool(output["correct"]) for output in scored_outputs)
                / len(scored_outputs)
                if scored_outputs
                else None
            ),
            "invalid_response_rate": (
                sum(not bool(output.get("valid", False)) for output in scored_outputs)
                / len(scored_outputs)
                if scored_outputs
                else None
            ),
            "generation_structural_failure_rate": (
                sum(
                    not bool(output.get("structurally_valid", False))
                    for output in generation_outputs
                )
                / len(generation_outputs)
                if generation_outputs
                else None
            ),
            "semantic_metrics": _semantic_generation_metrics(tag_metrics),
            "technical_failure_rate": technical["technical_failure_rate"],
        }
    return summaries


def _technical_failure_summary(
    cases: tuple[AudioEvaluationCase, ...],
    outputs_by_case_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_track = {
        track.value: {
            "completed_cases": 0,
            "technical_failures": 0,
            "technical_failure_rate": None,
        }
        for track in EvaluationTrack
    }
    failures: dict[str, str] = {}
    completed_cases = 0
    for case in cases:
        output = outputs_by_case_id.get(case.case_id)
        if output is None:
            continue
        completed_cases += 1
        track_summary = by_track[case.track.value]
        track_summary["completed_cases"] += 1
        failure = output.get("technical_failure")
        if failure is None:
            continue
        failures[case.case_id] = str(failure)
        track_summary["technical_failures"] += 1
    for track_summary in by_track.values():
        track_completed = int(track_summary["completed_cases"])
        if track_completed:
            track_summary["technical_failure_rate"] = (
                int(track_summary["technical_failures"]) / track_completed
            )
    return {
        "completed_cases": completed_cases,
        "technical_failures": len(failures),
        "technical_failure_rate": (
            len(failures) / completed_cases if completed_cases else None
        ),
        "by_track": by_track,
        "failures": dict(sorted(failures.items())),
    }


def _diagnostics_summary(
    cases: tuple[AudioEvaluationCase, ...],
    outputs_by_case_id: Mapping[str, Mapping[str, Any]],
    *,
    started_at_utc: str,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    by_track: dict[str, Any] = {}
    overall_elapsed = 0.0
    overall_completed = 0
    for track in EvaluationTrack:
        track_cases = [case for case in cases if case.track is track]
        outputs = [
            outputs_by_case_id[case.case_id]
            for case in track_cases
            if case.case_id in outputs_by_case_id
        ]
        elapsed_values = [
            float(output["elapsed_seconds"])
            for output in outputs
            if _is_number(output.get("elapsed_seconds"))
        ]
        elapsed_total = sum(elapsed_values)
        overall_elapsed += elapsed_total
        overall_completed += len(outputs)
        payload: dict[str, Any] = {
            "total_cases": len(track_cases),
            "completed_cases": len(outputs),
            "elapsed_seconds_total": elapsed_total,
            "elapsed_seconds_mean": (
                elapsed_total / len(elapsed_values) if elapsed_values else None
            ),
            "cases_per_second": (
                len(elapsed_values) / elapsed_total if elapsed_total > 0 else None
            ),
        }
        if track is EvaluationTrack.GENERATION:
            generated_audio_seconds = sum(
                float(output["duration_seconds"])
                for output in outputs
                if _is_number(output.get("duration_seconds"))
            )
            payload["generated_audio_seconds"] = generated_audio_seconds
            payload["audio_realtime_ratio"] = (
                generated_audio_seconds / elapsed_total if elapsed_total > 0 else None
            )
        by_track[track.value] = payload
    return {
        "by_track": by_track,
        "completed_cases": overall_completed,
        "elapsed_seconds_total": overall_elapsed,
        "cases_per_second": (
            overall_completed / overall_elapsed if overall_elapsed > 0 else None
        ),
        "started_at_utc": started_at_utc,
        "finalized_at_utc": _utc_now(),
        "wall_clock_seconds": wall_clock_seconds,
        "process_peak_rss": _process_peak_rss(),
    }


def _evaluate_capability_targets(
    *,
    targets: Mapping[str, float],
    metrics: Mapping[str, float | None],
) -> tuple[str, ...]:
    failures: list[str] = []
    for target_name, raw_threshold in sorted(targets.items()):
        threshold = float(raw_threshold)
        if target_name.endswith("_min"):
            metric_name = target_name[: -len("_min")]
            observed = metrics.get(metric_name)
            if observed is None:
                failures.append(f"{target_name}:missing_metric")
            elif float(observed) < threshold:
                failures.append(f"{target_name}:{observed:.6g}<{threshold:.6g}")
        elif target_name.endswith("_max"):
            metric_name = target_name[: -len("_max")]
            observed = metrics.get(metric_name)
            if observed is None:
                failures.append(f"{target_name}:missing_metric")
            elif float(observed) > threshold:
                failures.append(f"{target_name}:{observed:.6g}>{threshold:.6g}")
        else:
            failures.append(f"{target_name}:unsupported_target_suffix")
    return tuple(failures)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _process_peak_rss() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "value": int(usage.ru_maxrss),
        "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "unit": "bytes_on_macos_kib_on_linux",
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _reject_credentials(value: Any, *, path: str = "manifest") -> None:
    credential_keys = {
        "api_key",
        "credential",
        "credentials",
        "hf_token",
        "openai_api_key",
        "openrouter_api_key",
        "password",
        "secret",
        "token",
    }
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if key in credential_keys or key.endswith(
                ("_api_key", "_password", "_secret")
            ):
                raise ValueError(
                    f"credential-like manifest key is forbidden: {path}.{key}"
                )
            _reject_credentials(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_credentials(child, path=f"{path}[{index}]")
