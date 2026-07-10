"""Pinned Stability FD_OpenL3 worker for full audio evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .audio_evaluation_openl3 import (
    OPENL3_BATCH_SIZE,
    OPENL3_CHANNELS,
    OPENL3_EMBEDDING_SIZE,
    OPENL3_HOP_SECONDS,
    OPENL3_INPUT_REPR,
    OPENL3_REQUEST_SCHEMA,
    OPENL3_SAMPLE_RATE,
    STABLE_AUDIO_METRICS_REPO,
    STABLE_AUDIO_METRICS_REVISION,
    STABLE_AUDIO_METRICS_SOURCE_SHA256,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audex OpenL3 metric worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-file", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_worker(
        request_path=args.request,
        output_path=args.output,
        implementation_file=args.implementation_file,
        version_info=sys.version_info,
    )


def run_worker(
    *,
    request_path: Path,
    output_path: Path,
    version_info: tuple[int, ...] | sys.version_info,
    implementation_file: Path | None = None,
    backend_factory: Callable[..., Any] | None = None,
) -> int:
    if tuple(version_info[:2]) != (3, 11):
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="python_version_unsupported",
            detail=(
                "OpenL3 worker must run under Python 3.11; got "
                f"{version_info[0]}.{version_info[1]}"
            ),
        )
        return 2
    missing = _missing_modules(
        ("openl3", "librosa", "numpy", "scipy", "soxr", "pyloudnorm")
    )
    if missing:
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="missing_openl3_worker_dependencies",
            detail=f"Missing worker modules: {', '.join(missing)}",
        )
        return 2
    if not request_path.is_file():
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="request_not_found",
            detail=str(request_path),
        )
        return 2
    validation_error = _request_validation_error(request_path)
    if validation_error is not None:
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="invalid_openl3_request",
            detail=validation_error,
        )
        return 2
    if implementation_file is None or not implementation_file.is_file():
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="openl3_implementation_not_found",
            detail=str(implementation_file),
        )
        return 2

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    runtime_error = _runtime_input_error(payload)
    if runtime_error is not None:
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="invalid_openl3_runtime_inputs",
            detail=runtime_error,
        )
        return 2
    factory = backend_factory or _default_backend_factory
    try:
        backend = factory(implementation_file=implementation_file)
        qualification = dict(backend.qualify())
        if not bool(qualification.get("qualified", False)):
            _write_payload(
                output_path,
                {
                    "schema_version": OPENL3_REQUEST_SCHEMA,
                    "status": "UNSCORED",
                    "reason": "openl3_oracle_not_qualified",
                    "qualification": qualification,
                },
            )
            return 2
        per_dataset: list[dict[str, Any]] = []
        for request in payload["requests"]:
            fd_openl3, metric_seconds = backend.score(request)
            per_dataset.append(
                {
                    "dataset": str(request["dataset"]),
                    "fd_openl3": float(fd_openl3),
                    "metric_seconds": float(metric_seconds),
                }
            )
    except Exception as exc:
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="openl3_scoring_failed",
            detail=f"{type(exc).__name__}: {exc}",
        )
        return 2

    _write_payload(
        output_path,
        {
            "schema_version": OPENL3_REQUEST_SCHEMA,
            "status": "PASS",
            "implementation": dict(payload["implementation"]),
            "qualification": qualification,
            "fd_openl3_by_dataset": {
                result["dataset"]: result["fd_openl3"] for result in per_dataset
            },
            "per_dataset": per_dataset,
        },
    )
    return 0


def _request_validation_error(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return str(exc)
    if not isinstance(payload, dict):
        return "request must be a JSON object"
    if payload.get("schema_version") != OPENL3_REQUEST_SCHEMA:
        return f"schema_version must be {OPENL3_REQUEST_SCHEMA}"
    if not str(payload.get("run_id", "")).strip():
        return "run_id must be non-empty"
    implementation = payload.get("implementation")
    expected_implementation = {
        "repo_id": STABLE_AUDIO_METRICS_REPO,
        "revision": STABLE_AUDIO_METRICS_REVISION,
        "source_sha256": STABLE_AUDIO_METRICS_SOURCE_SHA256,
    }
    if implementation != expected_implementation:
        return "implementation does not match pinned stable-audio-metrics source"
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        return "requests must be a non-empty list"
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            return f"request {index} must be an object"
        for field in (
            "dataset",
            "content_type",
            "generated_dir",
            "reference_statistics_path",
        ):
            if not str(request.get(field, "")).strip():
                return f"request {index} is missing {field}"
        if request.get("content_type") not in {"env", "music"}:
            return f"request {index} content_type must be env or music"
        required_values = {
            "embedding_size": OPENL3_EMBEDDING_SIZE,
            "input_repr": OPENL3_INPUT_REPR,
            "hop_seconds": OPENL3_HOP_SECONDS,
            "channels": OPENL3_CHANNELS,
            "sample_rate": OPENL3_SAMPLE_RATE,
            "batch_size": OPENL3_BATCH_SIZE,
        }
        for field, expected in required_values.items():
            if request.get(field) != expected:
                return f"request {index} {field} must be {expected}"
        expected_file_count = request.get("expected_file_count")
        if not isinstance(expected_file_count, int) or expected_file_count <= 0:
            return f"request {index} expected_file_count must be a positive integer"
    return None


def _runtime_input_error(payload: Mapping[str, Any]) -> str | None:
    for request in payload["requests"]:
        dataset = str(request["dataset"])
        generated_dir = Path(str(request["generated_dir"]))
        reference_stats = Path(str(request["reference_statistics_path"]))
        if not generated_dir.is_dir():
            return f"{dataset} generated_dir does not exist: {generated_dir}"
        if not reference_stats.is_file():
            return f"{dataset} reference statistics do not exist: {reference_stats}"
        wav_count = sum(1 for _path in generated_dir.glob("*.wav"))
        expected_count = int(request["expected_file_count"])
        if wav_count != expected_count:
            return (
                f"{dataset} expected {expected_count} WAV files but found {wav_count}"
            )
    return None


def _default_backend_factory(**kwargs: Any) -> Any:
    from .audio_evaluation_openl3_backend import OfficialOpenL3Backend

    return OfficialOpenL3Backend(**kwargs)


def _missing_modules(module_names: tuple[str, ...]) -> tuple[str, ...]:
    missing: list[str] = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
    return tuple(missing)


def _write_result(
    path: Path,
    *,
    status: str,
    reason: str,
    detail: str,
) -> None:
    _write_payload(
        path,
        {
            "schema_version": OPENL3_REQUEST_SCHEMA,
            "status": status,
            "reason": reason,
            "detail": detail,
        },
    )


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
