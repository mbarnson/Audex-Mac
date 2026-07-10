"""Fail-loud AST worker entry point for audio generation evaluation.

This module is intentionally separate from the live conversational runtime.
It will eventually host pinned Audio Spectrogram Transformer inference over
generated WAVs; until then it validates the worker boundary and returns
`UNSCORED` rather than pretending AudioSet event metrics exist.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audex AST metric worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    args = parser.parse_args(argv)
    return run_worker(
        request_path=args.request,
        output_path=args.output,
        device=args.device,
    )


def run_worker(
    *,
    request_path: Path,
    output_path: Path,
    device: str,
) -> int:
    if device not in {"cpu", "mps", "cuda"}:
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="device_unsupported",
            detail=f"Unsupported AST device: {device}",
        )
        return 2
    missing = _missing_modules(("torch", "transformers", "soundfile"))
    if missing:
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="missing_ast_worker_dependencies",
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
            reason="invalid_ast_request",
            detail=validation_error,
        )
        return 2
    _write_result(
        output_path,
        status="UNSCORED",
        reason="ast_metric_not_implemented",
        detail=(
            "AST worker dependency boundary is present; pinned AudioSet event "
            "sanity scoring remains future work."
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
        for field in ("case_id", "generated_wav_path"):
            if not str(request.get(field, "")).strip():
                return f"request {index} is missing {field}"
        expected_labels = request.get("expected_labels")
        forbidden_labels = request.get("forbidden_labels", [])
        if not isinstance(expected_labels, list) or not expected_labels:
            return f"request {index} expected_labels must be a non-empty list"
        if not isinstance(forbidden_labels, list):
            return f"request {index} forbidden_labels must be a list"
        if any(not str(label).strip() for label in expected_labels):
            return f"request {index} expected_labels must not contain empty labels"
        if any(not str(label).strip() for label in forbidden_labels):
            return f"request {index} forbidden_labels must not contain empty labels"
        if set(expected_labels) & set(forbidden_labels):
            return f"request {index} expected/forbidden labels must not overlap"
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
