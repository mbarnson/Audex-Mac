from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.audio_evaluation_openl3 import (
    OpenL3DatasetRequest,
    build_openl3_worker_command,
    default_full_openl3_requests,
    load_openl3_worker_result,
    write_openl3_worker_request,
)
from audex_mac.audio_evaluation_openl3_worker import run_worker

pytestmark = pytest.mark.fast


def test_openl3_worker_request_contract_matches_full_tier_parameters(
    tmp_path: Path,
) -> None:
    requests = default_full_openl3_requests(tmp_path / "run")
    request_path = tmp_path / "request.json"

    write_openl3_worker_request(
        request_path,
        run_id="full-run",
        requests=requests,
    )

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "run_id": "full-run",
        "requests": [
            {
                "content_type": "env",
                "dataset": "audiocaps",
                "embedding_size": 512,
                "generated_dir": str(tmp_path / "run" / "media" / "enhanced"),
                "hop_seconds": 0.5,
                "input_repr": "mel256",
                "reference_dir": None,
            },
            {
                "content_type": "music",
                "dataset": "song-describer",
                "embedding_size": 512,
                "generated_dir": str(tmp_path / "run" / "media" / "enhanced"),
                "hop_seconds": 0.5,
                "input_repr": "mel256",
                "reference_dir": None,
            },
        ],
    }


def test_openl3_request_rejects_non_paper_parameters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="content_type"):
        OpenL3DatasetRequest(
            dataset="bad",
            content_type="speech",
            generated_dir=str(tmp_path),
        )
    with pytest.raises(ValueError, match="512"):
        OpenL3DatasetRequest(
            dataset="bad",
            content_type="env",
            generated_dir=str(tmp_path),
            embedding_size=6144,
        )


def test_openl3_worker_command_and_result_contract(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"

    command = build_openl3_worker_command(
        python="/opt/openl3/bin/python",
        request_path=request_path,
        output_path=output_path,
    )

    assert command == (
        "/opt/openl3/bin/python",
        "-m",
        "audex_mac.audio_evaluation_openl3_worker",
        "--request",
        str(request_path),
        "--output",
        str(output_path),
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
