"""AST worker request contracts for AudioSet-style event sanity evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audio_evaluation import AudioEvaluationCase, EvaluationTrack

AST_WORKER_MODULE = "audex_mac.audio_evaluation_ast_worker"
AST_REQUEST_SCHEMA = 1
AST_REPO_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"


@dataclass(frozen=True, slots=True)
class AstCaseRequest:
    case_id: str
    generated_wav_path: str
    expected_labels: tuple[str, ...]
    forbidden_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("AST request case_id must not be empty")
        if not self.generated_wav_path.strip():
            raise ValueError("AST request generated_wav_path must not be empty")
        if not self.expected_labels:
            raise ValueError("AST request requires expected labels")
        empty_expected = [label for label in self.expected_labels if not label.strip()]
        empty_forbidden = [
            label for label in self.forbidden_labels if not label.strip()
        ]
        if empty_expected or empty_forbidden:
            raise ValueError("AST request labels must not be empty")
        overlap = set(self.expected_labels) & set(self.forbidden_labels)
        if overlap:
            raise ValueError(
                f"AST expected/forbidden labels overlap: {sorted(overlap)}"
            )


def build_ast_case_requests(
    cases: Iterable[AudioEvaluationCase],
    *,
    generated_wav_by_case_id: Mapping[str, str | Path],
    expected_labels_by_case_id: Mapping[str, Sequence[str]],
    forbidden_labels_by_case_id: Mapping[str, Sequence[str]] | None = None,
) -> tuple[AstCaseRequest, ...]:
    """Build event-classifier requests from explicit case-label metadata."""

    forbidden_labels_by_case_id = forbidden_labels_by_case_id or {}
    requests: list[AstCaseRequest] = []
    for case in cases:
        if case.track is not EvaluationTrack.GENERATION:
            continue
        generated_wav = generated_wav_by_case_id.get(case.case_id)
        if generated_wav is None:
            raise ValueError(f"generation case {case.case_id} has no generated WAV")
        expected_labels = tuple(expected_labels_by_case_id.get(case.case_id, ()))
        if not expected_labels:
            raise ValueError(f"generation case {case.case_id} has no AST labels")
        forbidden_labels = tuple(forbidden_labels_by_case_id.get(case.case_id, ()))
        requests.append(
            AstCaseRequest(
                case_id=case.case_id,
                generated_wav_path=str(generated_wav),
                expected_labels=expected_labels,
                forbidden_labels=forbidden_labels,
            )
        )
    return tuple(requests)


def write_ast_worker_request(
    path: Path,
    *,
    run_id: str,
    requests: Iterable[AstCaseRequest],
    model_repo_id: str = AST_REPO_ID,
    model_revision: str = AST_REVISION,
) -> None:
    payload = {
        "schema_version": AST_REQUEST_SCHEMA,
        "run_id": run_id,
        "model": {
            "repo_id": model_repo_id,
            "revision": model_revision,
        },
        "logit_policy": "sigmoid_raw_logits",
        "requests": [asdict(request) for request in requests],
        "metrics": {
            "expected_label_hits": True,
            "forbidden_label_false_positives": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_ast_worker_command(
    *,
    python: str | Path,
    request_path: Path,
    output_path: Path,
    device: str,
) -> tuple[str, ...]:
    return (
        str(python),
        "-m",
        AST_WORKER_MODULE,
        "--request",
        str(request_path),
        "--output",
        str(output_path),
        "--device",
        device,
    )


def load_ast_worker_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"AST worker result must be a JSON object: {path}")
    if payload.get("status") != "PASS":
        raise RuntimeError(
            "AST worker did not produce a passing result: "
            f"{payload.get('reason', '<missing reason>')}"
        )
    return payload
