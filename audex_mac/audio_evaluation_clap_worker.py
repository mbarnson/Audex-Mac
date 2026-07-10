"""Fail-loud CLAP worker entry point for audio generation evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .audio_evaluation_clap import CLAP_REPO_ID, CLAP_REVISION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audex CLAP metric worker")
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
            detail=f"Unsupported CLAP device: {device}",
        )
        return 2
    missing = _missing_modules(("torch", "transformers", "soundfile", "numpy", "scipy"))
    if missing:
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="missing_clap_worker_dependencies",
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
            reason="invalid_clap_request",
            detail=validation_error,
        )
        return 2
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    factory = backend_factory or _default_backend_factory
    try:
        backend = factory(
            repo_id=CLAP_REPO_ID,
            revision=CLAP_REVISION,
            device=device,
        )
        result = _evaluate(payload, backend=backend)
    except Exception as exc:
        _write_result(
            output_path,
            status="PROTOCOL_FAIL",
            reason="clap_scoring_failed",
            detail=f"{type(exc).__name__}: {exc}",
        )
        return 2
    _write_payload(
        output_path,
        {
            "schema_version": 1,
            "status": "UNSCORED",
            "reason": "clap_oracle_not_qualified",
            "qualification": {
                "qualified": False,
                "status": "NOT_RUN",
            },
            "model": {
                "repo_id": CLAP_REPO_ID,
                "revision": CLAP_REVISION,
                "device": device,
            },
            **result,
        },
    )
    return 2


def _evaluate(payload: Mapping[str, Any], *, backend: Any) -> dict[str, Any]:
    requests = list(payload["requests"])
    captions = [str(request["caption"]) for request in requests]
    hard_foils = [str(request["hard_foil_caption"]) for request in requests]
    paths = [Path(str(request["generated_wav_path"])) for request in requests]
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"generated WAV paths do not exist: {', '.join(missing_paths)}"
        )

    text_vectors, text_preprocess, text_inference = backend.embed_text(
        captions + hard_foils
    )
    audio_vectors, audio_preprocess, audio_inference = backend.embed_audio(paths)
    normalized_text = _normalized_matrix(text_vectors, expected_rows=len(paths) * 2)
    normalized_audio = _normalized_matrix(audio_vectors, expected_rows=len(paths))
    if len(normalized_text[0]) != len(normalized_audio[0]):
        raise ValueError("CLAP text/audio embedding dimensions differ")

    caption_vectors = normalized_text[: len(paths)]
    foil_vectors = normalized_text[len(paths) :]
    caption_scores = _paired_dot(normalized_audio, caption_vectors)
    foil_scores = _paired_dot(normalized_audio, foil_vectors)
    ranks = _retrieval_ranks(normalized_audio, caption_vectors)
    margins = [
        caption - foil
        for caption, foil in zip(caption_scores, foil_scores, strict=True)
    ]
    wins = [margin > 0.0 for margin in margins]
    per_case = [
        {
            "case_id": str(request["case_id"]),
            "caption_similarity": caption_scores[index],
            "hard_foil_similarity": foil_scores[index],
            "hard_foil_margin": margins[index],
            "hard_foil_win": wins[index],
            "retrieval_rank": ranks[index],
        }
        for index, request in enumerate(requests)
    ]
    return {
        "metrics": {
            "caption_similarity_mean": _mean(caption_scores),
            "hard_foil_margin_mean": _mean(margins),
            "hard_foil_win_rate": _mean([float(win) for win in wins]),
            "retrieval_recall_at_1": _mean([float(rank == 1) for rank in ranks]),
        },
        "per_case": per_case,
        "timings": {
            "model_load_seconds": float(backend.model_load_seconds),
            "preprocessing_seconds": round(
                float(text_preprocess + audio_preprocess), 9
            ),
            "inference_seconds": round(float(text_inference + audio_inference), 9),
        },
    }


def _normalized_matrix(
    vectors: Sequence[Sequence[float]], *, expected_rows: int
) -> Any:
    numpy = importlib.import_module("numpy")
    matrix = numpy.asarray(vectors, dtype=numpy.float32)
    actual_rows = int(matrix.shape[0]) if matrix.ndim >= 1 else 0
    if matrix.ndim != 2 or actual_rows != expected_rows or matrix.shape[1] <= 0:
        raise ValueError(
            f"CLAP embedding rows mismatch: expected {expected_rows}, got {actual_rows}"
        )
    if not bool(numpy.isfinite(matrix).all()):
        raise ValueError("CLAP embeddings must be finite and rectangular")
    norms = numpy.linalg.norm(matrix, axis=1, keepdims=True)
    if bool((norms <= 0.0).any()):
        raise ValueError("CLAP embeddings must have nonzero norm")
    return matrix / norms


def _paired_dot(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[float]:
    numpy = importlib.import_module("numpy")
    return [float(value) for value in numpy.einsum("ij,ij->i", left, right)]


def _retrieval_ranks(
    audio_vectors: Sequence[Sequence[float]],
    caption_vectors: Sequence[Sequence[float]],
    *,
    chunk_size: int = 128,
) -> list[int]:
    if chunk_size <= 0:
        raise ValueError("CLAP retrieval chunk size must be positive")
    numpy = importlib.import_module("numpy")
    ranks: list[int] = []
    for start in range(0, len(audio_vectors), chunk_size):
        stop = min(start + chunk_size, len(audio_vectors))
        similarities = audio_vectors[start:stop] @ caption_vectors.T
        local_rows = numpy.arange(stop - start)
        target_columns = numpy.arange(start, stop)
        targets = similarities[local_rows, target_columns]
        chunk_ranks = 1 + numpy.sum(similarities > targets[:, None], axis=1)
        ranks.extend(int(rank) for rank in chunk_ranks)
    return ranks


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize empty CLAP scores")
    return sum(values) / len(values)


def _default_backend_factory(**kwargs: Any) -> Any:
    from .audio_evaluation_clap_backend import TransformersClapBackend

    return TransformersClapBackend(**kwargs)


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
    if model.get("repo_id") != CLAP_REPO_ID:
        return f"model repo_id must be {CLAP_REPO_ID}"
    if model.get("revision") != CLAP_REVISION:
        return f"model revision must be {CLAP_REVISION}"
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        return "requests must be a non-empty list"
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            return f"request {index} must be an object"
        caption = str(request.get("caption", "")).strip()
        hard_foil = str(request.get("hard_foil_caption", "")).strip()
        for field in ("case_id", "generated_wav_path", "caption", "hard_foil_caption"):
            if not str(request.get(field, "")).strip():
                return f"request {index} is missing {field}"
        if caption == hard_foil:
            return f"request {index} hard_foil_caption must differ from caption"
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
