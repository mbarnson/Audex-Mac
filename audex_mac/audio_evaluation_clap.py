"""CLAP worker request contracts for caption-alignment audio evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audio_evaluation import AudioEvaluationCase, EvaluationTrack

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


def write_clap_worker_request(
    path: Path,
    *,
    run_id: str,
    requests: Iterable[ClapCaseRequest],
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
