"""OpenL3 worker request contracts for full audio-capability evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OPENL3_WORKER_MODULE = "audex_mac.audio_evaluation_openl3_worker"
OPENL3_REQUEST_SCHEMA = 2
OPENL3_EMBEDDING_SIZE = 512
OPENL3_HOP_SECONDS = 0.5
OPENL3_INPUT_REPR = "mel256"
OPENL3_CHANNELS = 2
OPENL3_SAMPLE_RATE = 44_100
OPENL3_BATCH_SIZE = 4
STABLE_AUDIO_METRICS_REPO = "Stability-AI/stable-audio-metrics"
STABLE_AUDIO_METRICS_REVISION = "fd55536cc812c460ecc421220864993c7f168184"
STABLE_AUDIO_METRICS_SOURCE_SHA256 = (
    "03cbfa6c524ad5af992c390f23c5384dfda242de09931f9a7162132fa7095f21"
)
AUDIOCAPS_REFERENCE_STATS_FILENAME = (
    "audiocaps-test__channels2__44100__openl3env__openl3hopsize0.5__batch4.npz"
)
SONG_DESCRIBER_REFERENCE_STATS_FILENAME = (
    "song_describer__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz"
)


@dataclass(frozen=True, slots=True)
class OpenL3DatasetRequest:
    dataset: str
    content_type: str
    generated_dir: str
    reference_statistics_path: str
    expected_file_count: int
    embedding_size: int = OPENL3_EMBEDDING_SIZE
    input_repr: str = OPENL3_INPUT_REPR
    hop_seconds: float = OPENL3_HOP_SECONDS
    channels: int = OPENL3_CHANNELS
    sample_rate: int = OPENL3_SAMPLE_RATE
    batch_size: int = OPENL3_BATCH_SIZE

    def __post_init__(self) -> None:
        if self.content_type not in {"env", "music"}:
            raise ValueError(f"unsupported OpenL3 content_type: {self.content_type}")
        if self.embedding_size != OPENL3_EMBEDDING_SIZE:
            raise ValueError("Audex OpenL3 parity requires 512-dimensional embeddings")
        if self.input_repr != OPENL3_INPUT_REPR:
            raise ValueError("Audex OpenL3 parity requires mel256 input")
        if self.hop_seconds != OPENL3_HOP_SECONDS:
            raise ValueError("Audex OpenL3 parity requires 0.5-second hop")
        if self.channels != OPENL3_CHANNELS:
            raise ValueError("Audex OpenL3 parity requires stereo channel handling")
        if self.sample_rate != OPENL3_SAMPLE_RATE:
            raise ValueError("Audex OpenL3 parity requires 44.1 kHz metric bandwidth")
        if self.batch_size != OPENL3_BATCH_SIZE:
            raise ValueError("Audex OpenL3 parity requires batch size 4")
        if not self.reference_statistics_path.strip():
            raise ValueError("OpenL3 request requires reference statistics")
        if self.expected_file_count <= 0:
            raise ValueError("OpenL3 expected_file_count must be positive")


def default_full_openl3_requests(
    run_dir: Path,
    *,
    reference_stats_root: Path,
) -> tuple[OpenL3DatasetRequest, ...]:
    """Return the paper-style OpenL3 requests for a completed full run."""

    metric_root = run_dir / "media" / "openl3"
    return (
        OpenL3DatasetRequest(
            dataset="audiocaps",
            content_type="env",
            generated_dir=str(metric_root / "audiocaps"),
            reference_statistics_path=str(
                reference_stats_root / AUDIOCAPS_REFERENCE_STATS_FILENAME
            ),
            expected_file_count=4_875,
        ),
        OpenL3DatasetRequest(
            dataset="song-describer",
            content_type="music",
            generated_dir=str(metric_root / "song-describer"),
            reference_statistics_path=str(
                reference_stats_root / SONG_DESCRIBER_REFERENCE_STATS_FILENAME
            ),
            expected_file_count=746,
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
        "implementation": {
            "repo_id": STABLE_AUDIO_METRICS_REPO,
            "revision": STABLE_AUDIO_METRICS_REVISION,
            "source_sha256": STABLE_AUDIO_METRICS_SOURCE_SHA256,
        },
        "requests": [asdict(request) for request in requests],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_openl3_worker_command(
    *,
    python: str | Path,
    request_path: Path,
    output_path: Path,
    implementation_file: str | Path,
) -> tuple[str, ...]:
    return (
        str(python),
        "-m",
        OPENL3_WORKER_MODULE,
        "--request",
        str(request_path),
        "--output",
        str(output_path),
        "--implementation-file",
        str(implementation_file),
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
