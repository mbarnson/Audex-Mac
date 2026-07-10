from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from audex_mac.audio_evaluation_openl3 import (
    OPENL3_REQUEST_SCHEMA,
    STABLE_AUDIO_METRICS_REPO,
    STABLE_AUDIO_METRICS_REVISION,
    STABLE_AUDIO_METRICS_SOURCE_SHA256,
    OpenL3DatasetRequest,
    build_openl3_worker_command,
    default_full_openl3_requests,
    load_openl3_worker_result,
    write_openl3_worker_request,
)
from audex_mac.audio_evaluation_openl3_backend import OfficialOpenL3Backend
from audex_mac.audio_evaluation_openl3_worker import run_worker

pytestmark = pytest.mark.fast


def test_openl3_worker_request_contract_matches_full_tier_parameters(
    tmp_path: Path,
) -> None:
    requests = default_full_openl3_requests(
        tmp_path / "run",
        reference_stats_root=tmp_path / "reference-stats",
    )
    request_path = tmp_path / "request.json"

    write_openl3_worker_request(
        request_path,
        run_id="full-run",
        requests=requests,
    )

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": OPENL3_REQUEST_SCHEMA,
        "run_id": "full-run",
        "implementation": {
            "repo_id": STABLE_AUDIO_METRICS_REPO,
            "revision": STABLE_AUDIO_METRICS_REVISION,
            "source_sha256": STABLE_AUDIO_METRICS_SOURCE_SHA256,
        },
        "requests": [
            {
                "batch_size": 4,
                "channels": 2,
                "content_type": "env",
                "dataset": "audiocaps",
                "embedding_size": 512,
                "expected_file_count": 4875,
                "generated_dir": str(
                    tmp_path / "run" / "media" / "openl3" / "audiocaps"
                ),
                "hop_seconds": 0.5,
                "input_repr": "mel256",
                "reference_statistics_path": str(
                    tmp_path
                    / "reference-stats"
                    / "audiocaps-test__channels2__44100__openl3env__openl3hopsize0.5__batch4.npz"
                ),
                "sample_rate": 44100,
            },
            {
                "batch_size": 4,
                "channels": 2,
                "content_type": "music",
                "dataset": "song-describer",
                "embedding_size": 512,
                "expected_file_count": 746,
                "generated_dir": str(
                    tmp_path / "run" / "media" / "openl3" / "song-describer"
                ),
                "hop_seconds": 0.5,
                "input_repr": "mel256",
                "reference_statistics_path": str(
                    tmp_path
                    / "reference-stats"
                    / "song_describer__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz"
                ),
                "sample_rate": 44100,
            },
        ],
    }


def test_openl3_request_rejects_non_paper_parameters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="content_type"):
        OpenL3DatasetRequest(
            dataset="bad",
            content_type="speech",
            generated_dir=str(tmp_path),
            reference_statistics_path=str(tmp_path / "reference.npz"),
            expected_file_count=1,
        )
    with pytest.raises(ValueError, match="512"):
        OpenL3DatasetRequest(
            dataset="bad",
            content_type="env",
            generated_dir=str(tmp_path),
            reference_statistics_path=str(tmp_path / "reference.npz"),
            expected_file_count=1,
            embedding_size=6144,
        )


def test_openl3_worker_command_and_result_contract(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"

    command = build_openl3_worker_command(
        python="/opt/openl3/bin/python",
        request_path=request_path,
        output_path=output_path,
        implementation_file="/opt/stable-audio-metrics/src/openl3_fd.py",
    )

    assert command == (
        "/opt/openl3/bin/python",
        "-m",
        "audex_mac.audio_evaluation_openl3_worker",
        "--request",
        str(request_path),
        "--output",
        str(output_path),
        "--implementation-file",
        "/opt/stable-audio-metrics/src/openl3_fd.py",
    )
    output_path.write_text(
        json.dumps({"schema_version": 1, "status": "PASS", "fd_openl3": 12.3}),
        encoding="utf-8",
    )
    assert load_openl3_worker_result(output_path)["fd_openl3"] == 12.3
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PROTOCOL_FAIL",
                "reason": "missing_openl3_worker_dependencies",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="missing_openl3_worker_dependencies"):
        load_openl3_worker_result(output_path)


def test_openl3_worker_fails_loud_outside_python_311(tmp_path: Path) -> None:
    exit_code = run_worker(
        request_path=tmp_path / "request.json",
        output_path=tmp_path / "result.json",
        version_info=(3, 12, 0),
    )

    assert exit_code == 2
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PROTOCOL_FAIL"
    assert payload["reason"] == "python_version_unsupported"


def test_openl3_worker_fails_loud_when_dependencies_are_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "audex_mac.audio_evaluation_openl3_worker._missing_modules",
        lambda _names: ("openl3",),
    )

    exit_code = run_worker(
        request_path=request_path,
        output_path=tmp_path / "result.json",
        version_info=(3, 11, 9),
    )

    assert exit_code == 2
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PROTOCOL_FAIL"
    assert payload["reason"] == "missing_openl3_worker_dependencies"


def test_openl3_worker_rejects_invalid_request_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": OPENL3_REQUEST_SCHEMA,
                "run_id": "invalid-run",
                "implementation": {
                    "repo_id": STABLE_AUDIO_METRICS_REPO,
                    "revision": STABLE_AUDIO_METRICS_REVISION,
                    "source_sha256": STABLE_AUDIO_METRICS_SOURCE_SHA256,
                },
                "requests": [
                    {
                        "batch_size": 4,
                        "channels": 2,
                        "dataset": "audiocaps",
                        "content_type": "speech",
                        "generated_dir": str(tmp_path / "generated"),
                        "expected_file_count": 1,
                        "embedding_size": 512,
                        "input_repr": "mel256",
                        "hop_seconds": 0.5,
                        "reference_statistics_path": str(tmp_path / "reference.npz"),
                        "sample_rate": 44100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "audex_mac.audio_evaluation_openl3_worker._missing_modules",
        lambda _names: (),
    )

    exit_code = run_worker(
        request_path=request_path,
        output_path=tmp_path / "result.json",
        version_info=(3, 11, 9),
    )

    assert exit_code == 2
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PROTOCOL_FAIL"
    assert payload["reason"] == "invalid_openl3_request"
    assert "content_type" in payload["detail"]


def test_openl3_worker_qualifies_and_scores_pinned_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation_file = tmp_path / "openl3_fd.py"
    implementation_file.write_text("# pinned fixture\n", encoding="utf-8")
    generated_dir = tmp_path / "generated" / "audiocaps"
    generated_dir.mkdir(parents=True)
    (generated_dir / "103012.wav").write_bytes(b"RIFF fixture")
    reference_stats = tmp_path / "audiocaps-reference.npz"
    reference_stats.write_bytes(b"NPZ fixture")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": OPENL3_REQUEST_SCHEMA,
                "run_id": "full-fixture",
                "implementation": {
                    "repo_id": STABLE_AUDIO_METRICS_REPO,
                    "revision": STABLE_AUDIO_METRICS_REVISION,
                    "source_sha256": STABLE_AUDIO_METRICS_SOURCE_SHA256,
                },
                "requests": [
                    {
                        "batch_size": 4,
                        "channels": 2,
                        "content_type": "env",
                        "dataset": "audiocaps",
                        "embedding_size": 512,
                        "expected_file_count": 1,
                        "generated_dir": str(generated_dir),
                        "hop_seconds": 0.5,
                        "input_repr": "mel256",
                        "reference_statistics_path": str(reference_stats),
                        "sample_rate": 44100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeBackend:
        def qualify(self) -> dict[str, object]:
            return {
                "qualified": True,
                "identical_fd": 0.0,
                "permuted_fd": 0.0,
                "unrelated_fd": 400.0,
                "fixed_vector_fd": 30.0,
            }

        def score(self, request: dict[str, object]) -> tuple[float, float]:
            assert request["dataset"] == "audiocaps"
            return 66.9, 1.25

    monkeypatch.setattr(
        "audex_mac.audio_evaluation_openl3_worker._missing_modules",
        lambda _names: (),
    )

    exit_code = run_worker(
        request_path=request_path,
        output_path=tmp_path / "result.json",
        implementation_file=implementation_file,
        version_info=(3, 11, 9),
        backend_factory=lambda **_kwargs: FakeBackend(),
    )

    assert exit_code == 0
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["qualification"]["qualified"] is True
    assert payload["fd_openl3_by_dataset"] == {"audiocaps": 66.9}
    assert payload["per_dataset"] == [
        {
            "dataset": "audiocaps",
            "fd_openl3": 66.9,
            "metric_seconds": 1.25,
        }
    ]


def test_official_openl3_backend_qualifies_fixed_fd_vectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation_file = tmp_path / "openl3_fd.py"
    implementation_file.write_text(
        """
import numpy as np
from scipy import linalg

def calculate_embd_statistics(values):
    return np.mean(values, axis=0), np.cov(values, rowvar=False)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

def openl3_fd(**kwargs):
    assert kwargs["channels"] == 2
    assert kwargs["samplingrate"] == 44100
    assert kwargs["batching"] == 4
    return 12.3
""".lstrip(),
        encoding="utf-8",
    )
    source_sha = hashlib.sha256(implementation_file.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "audex_mac.audio_evaluation_openl3_backend.STABLE_AUDIO_METRICS_SOURCE_SHA256",
        source_sha,
    )
    backend = OfficialOpenL3Backend(implementation_file=implementation_file)

    qualification = backend.qualify()
    score, elapsed = backend.score(
        {
            "batch_size": 4,
            "channels": 2,
            "content_type": "env",
            "generated_dir": str(tmp_path / "generated"),
            "hop_seconds": 0.5,
            "reference_statistics_path": str(tmp_path / "reference.npz"),
            "sample_rate": 44100,
        }
    )

    assert qualification["qualified"] is True
    assert qualification["identical_fd"] == pytest.approx(0.0, abs=1e-6)
    assert qualification["permuted_fd"] == pytest.approx(0.0, abs=1e-6)
    assert qualification["unrelated_fd"] == pytest.approx(400.0)
    assert qualification["fixed_vector_fd"] == pytest.approx(30.0)
    assert score == 12.3
    assert elapsed >= 0.0
