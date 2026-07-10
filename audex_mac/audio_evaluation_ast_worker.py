"""AST worker entry point for audio generation event-sanity diagnostics."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .audio_evaluation_ast import AST_REPO_ID, AST_REVISION

AST_EXPECTED_LABEL_MIN_PROBABILITY = 0.10
AST_FORBIDDEN_LABEL_MIN_PROBABILITY = 0.10
AST_TOP_K = 10


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
    backend_factory: Callable[..., Any] | None = None,
) -> int:
    if device not in {"cpu", "mps", "cuda"}:
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="device_unsupported",
            detail=f"Unsupported AST device: {device}",
        )
        return 2
    missing = _missing_modules(
        ("torch", "transformers", "soundfile", "numpy", "scipy.signal")
    )
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
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    factory = backend_factory or _default_backend_factory
    try:
        backend = factory(
            repo_id=AST_REPO_ID,
            revision=AST_REVISION,
            device=device,
        )
        result = _evaluate(payload, backend=backend)
    except Exception as exc:
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="ast_scoring_failed",
            detail=f"{type(exc).__name__}: {exc}",
        )
        return 2
    _write_payload(
        output_path,
        {
            "schema_version": 1,
            "status": "UNSCORED",
            "reason": "ast_oracle_not_qualified",
            "qualification": {
                "qualified": False,
                "status": "NOT_RUN",
            },
            "model": {
                "repo_id": AST_REPO_ID,
                "revision": AST_REVISION,
                "device": device,
            },
            "logit_policy": "sigmoid_raw_logits",
            "thresholds": {
                "expected_label_min_probability": AST_EXPECTED_LABEL_MIN_PROBABILITY,
                "forbidden_label_min_probability": AST_FORBIDDEN_LABEL_MIN_PROBABILITY,
                "top_k": AST_TOP_K,
            },
            **result,
        },
    )
    return 2


def _evaluate(payload: Mapping[str, Any], *, backend: Any) -> dict[str, Any]:
    requests = list(payload["requests"])
    paths = [Path(str(request["generated_wav_path"])) for request in requests]
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"generated WAV paths do not exist: {', '.join(missing_paths)}"
        )

    probabilities, preprocessing_seconds, inference_seconds = backend.classify_audio(
        paths
    )
    if len(probabilities) != len(requests):
        raise ValueError(
            f"AST output row mismatch: expected {len(requests)}, got {len(probabilities)}"
        )
    backend_labels = getattr(backend, "labels", frozenset())
    if not isinstance(backend_labels, frozenset):
        backend_labels = frozenset(backend_labels)

    per_case: list[dict[str, Any]] = []
    expected_hits: list[float] = []
    forbidden_false_positives: list[float] = []
    for request, label_scores in zip(requests, probabilities, strict=True):
        _ensure_requested_labels_known(request, backend_labels)
        expected_labels = tuple(str(label) for label in request["expected_labels"])
        forbidden_labels = tuple(
            str(label) for label in request.get("forbidden_labels", ())
        )
        expected_label_scores = {
            label: _finite_probability(label_scores[label]) for label in expected_labels
        }
        forbidden_label_scores = {
            label: _finite_probability(label_scores[label])
            for label in forbidden_labels
        }
        expected_label_hit = any(
            score >= AST_EXPECTED_LABEL_MIN_PROBABILITY
            for score in expected_label_scores.values()
        )
        forbidden_label_false_positive = any(
            score >= AST_FORBIDDEN_LABEL_MIN_PROBABILITY
            for score in forbidden_label_scores.values()
        )
        expected_hits.append(1.0 if expected_label_hit else 0.0)
        if forbidden_label_scores:
            forbidden_false_positives.append(
                1.0 if forbidden_label_false_positive else 0.0
            )
        per_case.append(
            {
                "case_id": str(request["case_id"]),
                "expected_label_scores": expected_label_scores,
                "forbidden_label_scores": forbidden_label_scores,
                "expected_label_hit": expected_label_hit,
                "forbidden_label_false_positive": forbidden_label_false_positive,
                "top_labels": _top_labels(label_scores, top_k=AST_TOP_K),
            }
        )
    return {
        "metrics": {
            "expected_label_cases": len(expected_hits),
            "expected_label_hit_rate": _mean(expected_hits),
            "forbidden_label_cases": len(forbidden_false_positives),
            "forbidden_label_false_positive_rate": (
                _mean(forbidden_false_positives) if forbidden_false_positives else None
            ),
        },
        "per_case": per_case,
        "timings": {
            "model_load_seconds": float(backend.model_load_seconds),
            "preprocessing_seconds": round(float(preprocessing_seconds), 9),
            "inference_seconds": round(float(inference_seconds), 9),
        },
    }


def _ensure_requested_labels_known(
    request: Mapping[str, Any],
    backend_labels: frozenset[str],
) -> None:
    requested = tuple(request["expected_labels"]) + tuple(
        request.get("forbidden_labels", ())
    )
    unknown = sorted(
        str(label) for label in requested if str(label) not in backend_labels
    )
    if unknown:
        raise ValueError(
            f"AST request {request['case_id']} contains unknown labels: {unknown}"
        )


def _finite_probability(value: Any) -> float:
    probability = float(value)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"AST probability must be finite in [0, 1]: {value!r}")
    return probability


def _top_labels(scores: Mapping[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(
        ((str(label), _finite_probability(score)) for label, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return [
        {"label": label, "probability": probability}
        for label, probability in ranked[:top_k]
    ]


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize empty AST scores")
    return sum(values) / len(values)


def _default_backend_factory(**kwargs: Any) -> Any:
    from .audio_evaluation_ast_backend import TransformersAstBackend

    return TransformersAstBackend(**kwargs)


def _request_validation_error(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return str(exc)
    if not isinstance(payload, dict):
        return "request must be a JSON object"
    if payload.get("schema_version") != 1:
        return "schema_version must be 1"
    if not str(payload.get("run_id", "")).strip():
        return "run_id must be non-empty"
    model = payload.get("model")
    if not isinstance(model, dict):
        return "model must be an object"
    if model.get("repo_id") != AST_REPO_ID:
        return f"model repo_id must be {AST_REPO_ID}"
    if model.get("revision") != AST_REVISION:
        return f"model revision must be {AST_REVISION}"
    if payload.get("logit_policy") != "sigmoid_raw_logits":
        return "logit_policy must be sigmoid_raw_logits"
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
    _write_payload(path, payload)


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
