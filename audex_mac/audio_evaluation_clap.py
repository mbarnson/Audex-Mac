"""CLAP worker request contracts for caption-alignment audio evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audio_evaluation import AudioEvaluationCase, EvaluationTrack
from .audio_evaluation_esc50 import ESC50_HARD_NEGATIVES

ESC50_DATASET_ID = "ashraq/esc50"

CLAP_WORKER_MODULE = "audex_mac.audio_evaluation_clap_worker"
CLAP_REQUEST_SCHEMA = 1
CLAP_REPO_ID = "laion/clap-htsat-unfused"
CLAP_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"


@dataclass(frozen=True, slots=True)
class ClapCaseRequest:
    case_id: str
    generated_wav_path: str
    caption: str
    hard_foil_caption: str

    def __post_init__(self) -> None:
        missing = [
            name
            for name, value in (
                ("case_id", self.case_id),
                ("generated_wav_path", self.generated_wav_path),
                ("caption", self.caption),
                ("hard_foil_caption", self.hard_foil_caption),
            )
            if not str(value).strip()
        ]
        if missing:
            raise ValueError(f"CLAP request has empty fields: {missing}")
        if self.caption == self.hard_foil_caption:
            raise ValueError("CLAP hard foil caption must differ from caption")


@dataclass(frozen=True, slots=True)
class ClapQualificationRequest:
    case_id: str
    audio_path: str
    expected_caption: str
    hard_negative_captions: tuple[str, ...]

    def __post_init__(self) -> None:
        missing = [
            name
            for name, value in (
                ("case_id", self.case_id),
                ("audio_path", self.audio_path),
                ("expected_caption", self.expected_caption),
            )
            if not str(value).strip()
        ]
        if missing:
            raise ValueError(f"CLAP qualification request has empty fields: {missing}")
        if len(self.hard_negative_captions) != 3:
            raise ValueError("CLAP qualification requires exactly three hard negatives")
        captions = (
            self.expected_caption,
            *self.hard_negative_captions,
        )
        if any(not caption.strip() for caption in captions):
            raise ValueError("CLAP qualification captions must not be empty")
        if len(set(captions)) != len(captions):
            raise ValueError("CLAP qualification captions must be distinct")


def build_clap_case_requests(
    cases: Iterable[AudioEvaluationCase],
    *,
    generated_wav_by_case_id: Mapping[str, str | Path],
) -> tuple[ClapCaseRequest, ...]:
    """Build caption-alignment requests for completed generation cases."""

    requests: list[ClapCaseRequest] = []
    for case in cases:
        if case.track is not EvaluationTrack.GENERATION:
            continue
        if not case.caption:
            raise ValueError(f"generation case {case.case_id} has no caption")
        if not case.hard_foil_caption:
            raise ValueError(f"generation case {case.case_id} has no hard foil")
        generated_wav = generated_wav_by_case_id.get(case.case_id)
        if generated_wav is None:
            raise ValueError(f"generation case {case.case_id} has no generated WAV")
        requests.append(
            ClapCaseRequest(
                case_id=case.case_id,
                generated_wav_path=str(generated_wav),
                caption=case.caption,
                hard_foil_caption=case.hard_foil_caption,
            )
        )
    return tuple(requests)


def build_clap_qualification_requests(
    cases: Iterable[AudioEvaluationCase],
) -> tuple[ClapQualificationRequest, ...]:
    """Build fixed four-way CLAP calibration requests from pinned ESC-50 cases."""

    requests: list[ClapQualificationRequest] = []
    for case in cases:
        if (
            case.track is not EvaluationTrack.UNDERSTANDING
            or case.dataset_id != ESC50_DATASET_ID
        ):
            continue
        if not case.audio_path:
            raise ValueError(f"ESC-50 case {case.case_id} has no audio path")
        negatives = ESC50_HARD_NEGATIVES.get(case.category)
        if negatives is None:
            raise ValueError(f"ESC-50 case has unknown category: {case.category}")
        requests.append(
            ClapQualificationRequest(
                case_id=case.case_id,
                audio_path=case.audio_path,
                expected_caption=_esc50_caption(case.category),
                hard_negative_captions=tuple(
                    _esc50_caption(category) for category in negatives
                ),
            )
        )
    return tuple(requests)


def _esc50_caption(category: str) -> str:
    return f"The sound of {category.replace('_', ' ')}."


def write_clap_worker_request(
    path: Path,
    *,
    run_id: str,
    requests: Iterable[ClapCaseRequest],
    qualification_requests: Iterable[ClapQualificationRequest] = (),
    model_repo_id: str = CLAP_REPO_ID,
    model_revision: str = CLAP_REVISION,
) -> None:
    payload = {
        "schema_version": CLAP_REQUEST_SCHEMA,
        "run_id": run_id,
        "model": {
            "repo_id": model_repo_id,
            "revision": model_revision,
        },
        "requests": [asdict(request) for request in requests],
        "qualification_requests": [
            asdict(request) for request in qualification_requests
        ],
        "metrics": {
            "caption_similarity": True,
            "hard_foil_win": True,
            "hard_foil_margin": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_clap_worker_command(
    *,
    python: str | Path,
    request_path: Path,
    output_path: Path,
    device: str,
) -> tuple[str, ...]:
    return (
        str(python),
        "-m",
        CLAP_WORKER_MODULE,
        "--request",
        str(request_path),
        "--output",
        str(output_path),
        "--device",
        device,
    )


def load_clap_worker_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"CLAP worker result must be a JSON object: {path}")
    if payload.get("status") != "PASS":
        raise RuntimeError(
            "CLAP worker did not produce a passing result: "
            f"{payload.get('reason', '<missing reason>')}"
        )
    return payload
