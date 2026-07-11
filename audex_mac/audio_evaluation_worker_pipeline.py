"""Fail-closed semantic worker pipeline for generated audio metrics."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_evaluation import AudioEvaluationRun
from .audio_evaluation_ast import build_ast_worker_command
from .audio_evaluation_clap import build_clap_worker_command
from .audio_evaluation_openl3 import build_openl3_worker_command

CommandRunner = Callable[[tuple[str, ...]], int]


@dataclass(frozen=True, slots=True)
class GenerationWorkerConfig:
    semantic_python: Path
    semantic_device: str
    openl3_python: Path | None = None
    openl3_implementation_file: Path | None = None

    def __post_init__(self) -> None:
        if self.semantic_device not in {"cpu", "mps", "cuda"}:
            raise ValueError("semantic worker device must be one of cpu, mps, or cuda")
        if (self.openl3_python is None) != (self.openl3_implementation_file is None):
            raise ValueError(
                "OpenL3 worker Python and implementation file must be configured together"
            )


@dataclass(frozen=True, slots=True)
class GenerationWorkerPipelineResult:
    qualified: bool
    oracle_results: Mapping[str, Any]
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _WorkerSpec:
    oracle: str
    failure_prefix: str
    request_name: str
    result_name: str
    command_builder: Callable[..., tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _OpenL3Result:
    payload: Mapping[str, Any]
    metrics: dict[str, float] | None
    failures: tuple[str, ...]


_WORKERS = (
    _WorkerSpec(
        oracle="clap",
        failure_prefix="clap_worker",
        request_name="clap-request.json",
        result_name="clap-result.json",
        command_builder=build_clap_worker_command,
    ),
    _WorkerSpec(
        oracle="ast",
        failure_prefix="ast_worker",
        request_name="ast-request.json",
        result_name="ast-result.json",
        command_builder=build_ast_worker_command,
    ),
)


def run_generation_worker_pipeline(
    run: AudioEvaluationRun,
    *,
    config: GenerationWorkerConfig,
    command_runner: CommandRunner | None = None,
) -> GenerationWorkerPipelineResult:
    """Run generated-audio semantic workers and ingest metrics only if all pass."""

    command_runner = command_runner or _run_subprocess
    generation_dir = run.run_dir / "generation"
    failures: list[str] = []
    worker_payloads: dict[str, Mapping[str, Any]] = {}
    expected_case_ids_by_oracle: dict[str, tuple[str, ...]] = {}
    openl3_metrics: dict[str, float] | None = None

    for spec in _WORKERS:
        request_path = generation_dir / spec.request_name
        if not request_path.is_file():
            failures.append(f"{spec.failure_prefix}_request_missing:{request_path}")
            continue
        expected_case_ids_by_oracle[spec.oracle] = _request_case_ids(request_path)
        output_path = generation_dir / spec.result_name
        command = spec.command_builder(
            python=config.semantic_python,
            request_path=request_path,
            output_path=output_path,
            device=config.semantic_device,
        )
        try:
            exit_code = command_runner(command)
        except Exception as exc:
            failures.append(
                f"{spec.failure_prefix}_launch_failed:" f"{type(exc).__name__}: {exc}"
            )
            continue
        payload = _read_worker_payload(output_path)
        worker_payloads[spec.oracle] = payload
        status = str(payload.get("status", "<missing>"))
        reason = str(payload.get("reason", "<none>"))
        if exit_code != 0 or status != "PASS" or not _qualified(payload):
            failures.append(
                f"{spec.failure_prefix}_failed:"
                f"exit={exit_code}:status={status}:reason={reason}"
            )
            continue
        actual_case_ids = _per_case_ids(payload)
        expected_case_ids = expected_case_ids_by_oracle[spec.oracle]
        if len(actual_case_ids) != len(set(actual_case_ids)) or not actual_case_ids:
            failures.append(f"{spec.failure_prefix}_result_has_invalid_case_ids")
            continue
        missing = sorted(set(expected_case_ids) - set(actual_case_ids))
        unexpected = sorted(set(actual_case_ids) - set(expected_case_ids))
        if missing or unexpected:
            failures.append(
                f"{spec.failure_prefix}_case_coverage_mismatch:"
                f"missing={missing}:unexpected={unexpected}"
            )

    if config.openl3_python is not None:
        assert config.openl3_implementation_file is not None
        openl3_result = _run_openl3(
            generation_dir=generation_dir,
            python=config.openl3_python,
            implementation_file=config.openl3_implementation_file,
            command_runner=command_runner,
        )
        worker_payloads["openl3"] = openl3_result.payload
        failures.extend(openl3_result.failures)
        openl3_metrics = openl3_result.metrics

    if failures:
        result = GenerationWorkerPipelineResult(
            qualified=False,
            oracle_results={
                oracle: _qualification_payload(payload)
                for oracle, payload in sorted(worker_payloads.items())
            },
            failures=tuple(failures),
        )
        _record_pipeline_qualification(run, result)
        return result

    for spec in _WORKERS:
        payload = worker_payloads.get(spec.oracle)
        if payload is None:
            continue
        for per_case in payload.get("per_case", ()):
            if not isinstance(per_case, Mapping):
                continue
            case_id = str(per_case.get("case_id", "")).strip()
            if not case_id:
                continue
            metric_payload = {
                key: value for key, value in per_case.items() if key != "case_id"
            }
            run.record_generation_metrics(
                case_id=case_id,
                payload={"oracle": spec.oracle, **metric_payload},
            )
    if openl3_metrics is not None:
        run.record_generation_aggregate_metrics(
            {
                "oracle": "openl3",
                "fd_openl3_by_dataset": openl3_metrics,
            }
        )

    result = GenerationWorkerPipelineResult(
        qualified=True,
        oracle_results={
            oracle: _qualification_payload(payload)
            for oracle, payload in sorted(worker_payloads.items())
        },
        failures=(),
    )
    _record_pipeline_qualification(run, result)
    return result


def _request_case_ids(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests", ())
    if not isinstance(requests, list):
        return ()
    return tuple(
        str(request.get("case_id", "")).strip()
        for request in requests
        if isinstance(request, Mapping) and str(request.get("case_id", "")).strip()
    )


def _read_worker_payload(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {"status": "PROTOCOL_FAIL", "reason": "worker_result_missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "PROTOCOL_FAIL",
            "reason": "worker_result_invalid",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, Mapping):
        return {"status": "PROTOCOL_FAIL", "reason": "worker_result_invalid"}
    return payload


def _qualified(payload: Mapping[str, Any]) -> bool:
    qualification = payload.get("qualification")
    return isinstance(qualification, Mapping) and bool(qualification.get("qualified"))


def _qualification_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    qualification = payload.get("qualification")
    if isinstance(qualification, Mapping):
        return dict(qualification)
    return {"qualified": False, "status": "MISSING"}


def _per_case_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    per_case = payload.get("per_case", ())
    if not isinstance(per_case, list):
        return ()
    return tuple(
        str(item.get("case_id", "")).strip()
        for item in per_case
        if isinstance(item, Mapping) and str(item.get("case_id", "")).strip()
    )


def _run_openl3(
    *,
    generation_dir: Path,
    python: Path,
    implementation_file: Path,
    command_runner: CommandRunner,
) -> _OpenL3Result:
    request_path = generation_dir / "openl3-request.json"
    output_path = generation_dir / "openl3-result.json"
    if not request_path.is_file():
        return _OpenL3Result(
            payload={"qualified": False},
            metrics=None,
            failures=(f"openl3_worker_request_missing:{request_path}",),
        )
    command = build_openl3_worker_command(
        python=python,
        request_path=request_path,
        output_path=output_path,
        implementation_file=implementation_file,
    )
    try:
        exit_code = command_runner(command)
    except Exception as exc:
        return _OpenL3Result(
            payload={"qualified": False},
            metrics=None,
            failures=("openl3_worker_launch_failed:" f"{type(exc).__name__}: {exc}",),
        )
    payload = _read_worker_payload(output_path)
    status = str(payload.get("status", "<missing>"))
    reason = str(payload.get("reason", "<none>"))
    if exit_code != 0 or status != "PASS" or not _qualified(payload):
        return _OpenL3Result(
            payload=payload,
            metrics=None,
            failures=(
                "openl3_worker_failed:"
                f"exit={exit_code}:status={status}:reason={reason}",
            ),
        )
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    expected = {
        str(request.get("dataset", "")).strip()
        for request in request_payload.get("requests", ())
        if isinstance(request, Mapping)
    }
    raw_metrics = payload.get("fd_openl3_by_dataset")
    if not isinstance(raw_metrics, Mapping):
        return _OpenL3Result(
            payload=payload,
            metrics=None,
            failures=("openl3_worker_result_missing_dataset_metrics",),
        )
    actual = {str(dataset).strip() for dataset in raw_metrics}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if "" in expected or missing or unexpected:
        return _OpenL3Result(
            payload=payload,
            metrics=None,
            failures=(
                "openl3_worker_dataset_coverage_mismatch:"
                f"missing={missing}:unexpected={unexpected}",
            ),
        )
    metrics = {str(dataset): float(value) for dataset, value in raw_metrics.items()}
    return _OpenL3Result(payload=payload, metrics=metrics, failures=())


def _record_pipeline_qualification(
    run: AudioEvaluationRun,
    result: GenerationWorkerPipelineResult,
) -> None:
    run.record_oracle_qualification(
        {
            "qualified": result.qualified,
            "oracle_results": dict(result.oracle_results),
            "failures": list(result.failures),
        }
    )


def _run_subprocess(command: tuple[str, ...]) -> int:
    return subprocess.run(command, check=False).returncode
