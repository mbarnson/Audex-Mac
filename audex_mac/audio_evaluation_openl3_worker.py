"""Fail-loud OpenL3 worker entry point for full audio evaluation.

This module is intentionally separate from the main evaluator runtime.  It must
run under a Python 3.11 environment with OpenL3/stable-audio-metrics installed.
The metric implementation is not vendored into Audex-Mac's interactive runtime.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audex OpenL3 metric worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_worker(
        request_path=args.request,
        output_path=args.output,
        version_info=sys.version_info,
    )


def run_worker(
    *,
    request_path: Path,
    output_path: Path,
    version_info: tuple[int, ...] | sys.version_info,
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
    missing = _missing_modules(("openl3", "stable_audio_metrics"))
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
    _write_result(
        output_path,
        status="UNSCORED",
        reason="openl3_metric_not_implemented",
        detail=(
            "Worker dependency boundary is present; FD_OpenL3 computation still "
            "needs the pinned stable-audio-metrics implementation."
        ),
    )
    return 2


def _request_validation_error(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return str(exc)
    if not isinstance(payload, dict):
        return "request must be a JSON object"
    if payload.get("schema_version") != 1:
        return "schema_version must be 1"
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        return "requests must be a non-empty list"
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            return f"request {index} must be an object"
        for field in ("dataset", "content_type", "generated_dir"):
            if not str(request.get(field, "")).strip():
                return f"request {index} is missing {field}"
        if request.get("content_type") not in {"env", "music"}:
            return f"request {index} content_type must be env or music"
        if request.get("embedding_size") != 512:
            return f"request {index} embedding_size must be 512"
        if request.get("input_repr") != "mel256":
            return f"request {index} input_repr must be mel256"
        try:
            hop_seconds = float(request.get("hop_seconds"))
        except (TypeError, ValueError):
            return f"request {index} hop_seconds must be numeric"
        if hop_seconds != 0.5:
            return f"request {index} hop_seconds must be 0.5"
        reference_dir = request.get("reference_dir")
        if reference_dir is not None and not str(reference_dir).strip():
            return f"request {index} reference_dir must be null or non-empty"
    return None


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
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "detail": detail,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
