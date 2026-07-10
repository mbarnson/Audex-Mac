"""OpenL3 worker request contracts for full audio-capability evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OPENL3_WORKER_MODULE = "audex_mac.audio_evaluation_openl3_worker"
OPENL3_REQUEST_SCHEMA = 1
OPENL3_EMBEDDING_SIZE = 512
OPENL3_HOP_SECONDS = 0.5
OPENL3_INPUT_REPR = "mel256"


@dataclass(frozen=True, slots=True)
class OpenL3DatasetRequest:
    dataset: str
    content_type: str
    generated_dir: str
    reference_dir: str | None = None
    embedding_size: int = OPENL3_EMBEDDING_SIZE
    input_repr: str = OPENL3_INPUT_REPR
    hop_seconds: float = OPENL3_HOP_SECONDS

    def __post_init__(self) -> None:
        if self.content_type not in {"env", "music"}:
            raise ValueError(f"unsupported OpenL3 content_type: {self.content_type}")
        if self.embedding_size != OPENL3_EMBEDDING_SIZE:
            raise ValueError("Audex OpenL3 parity requires 512-dimensional embeddings")
        if self.input_repr != OPENL3_INPUT_REPR:
            raise ValueError("Audex OpenL3 parity requires mel256 input")
        if self.hop_seconds != OPENL3_HOP_SECONDS:
            raise ValueError("Audex OpenL3 parity requires 0.5-second hop")


def default_full_openl3_requests(run_dir: Path) -> tuple[OpenL3DatasetRequest, ...]:
    """Return the paper-style OpenL3 requests for a completed full run."""

    enhanced_dir = run_dir / "media" / "enhanced"
    return (
        OpenL3DatasetRequest(
            dataset="audiocaps",
            content_type="env",
            generated_dir=str(enhanced_dir),
        ),
        OpenL3DatasetRequest(
            dataset="song-describer",
            content_type="music",
            generated_dir=str(enhanced_dir),
        ),
    )


def write_openl3_worker_request(
    path: Path,
    *,
    run_id: str,
    requests: Iterable[OpenL3DatasetRequest],
) -> None:
    payload = {
        "schema_version": OPENL3_REQUEST_SCHEMA,
        "run_id": run_id,
        "requests": [asdict(request) for request in requests],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_openl3_worker_command(
    *,
    python: str | Path,
    request_path: Path,
    output_path: Path,
) -> tuple[str, ...]:
    return (
        str(python),
        "-m",
        OPENL3_WORKER_MODULE,
        "--request",
        str(request_path),
        "--output",
        str(output_path),
    )


def load_openl3_worker_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"OpenL3 worker result must be a JSON object: {path}")
    if payload.get("status") != "PASS":
        raise RuntimeError(
            "OpenL3 worker did not produce a passing result: "
            f"{payload.get('reason', '<missing reason>')}"
        )
    return payload
