from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.audio_evaluation import AudioEvaluationCase, EvaluationTrack
from audex_mac.audio_evaluation_clap import (
    CLAP_REVISION,
    ClapCaseRequest,
    build_clap_case_requests,
    build_clap_worker_command,
    load_clap_worker_result,
    write_clap_worker_request,
)
from audex_mac.audio_evaluation_clap_worker import run_worker

pytestmark = pytest.mark.fast


def _generation_case(
    *,
    case_id: str = "audiocaps-1",
    caption: str = "A dog barks twice.",
    hard_foil_caption: str | None = "A bell rings in a hallway.",
) -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id=case_id,
        track=EvaluationTrack.GENERATION,
        dataset_id="fixture/audiocaps",
        dataset_revision="rev1",
        dataset_config="default",
        dataset_split="test",
        source_row_id=case_id,
        source_row_hash=f"hash-{case_id}",
        license="CC0",
        category="audiocaps",
        prompt=caption,
        caption=caption,
        hard_foil_caption=hard_foil_caption,
    )


def test_clap_worker_request_records_caption_and_hard_foil_contract(
    tmp_path: Path,
) -> None:
    case = _generation_case()
    requests = build_clap_case_requests(
        (case,),
        generated_wav_by_case_id={case.case_id: tmp_path / "generated.wav"},
    )
    request_path = tmp_path / "request.json"

    write_clap_worker_request(
        request_path,
        run_id="smoke-run",
        requests=requests,
    )

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert payload == {
        "metrics": {
            "caption_similarity": True,
            "hard_foil_margin": True,
            "hard_foil_win": True,
        },
        "model": {
            "repo_id": "laion/clap-htsat-unfused",
            "revision": CLAP_REVISION,
        },
        "requests": [
            {
                "caption": "A dog barks twice.",
                "case_id": "audiocaps-1",
                "generated_wav_path": str(tmp_path / "generated.wav"),
                "hard_foil_caption": "A bell rings in a hallway.",
            }
        ],
        "run_id": "smoke-run",
        "schema_version": 1,
    }


def test_clap_request_rejects_missing_or_self_foil(tmp_path: Path) -> None:
    missing_foil = _generation_case(hard_foil_caption=None)
    with pytest.raises(ValueError, match="hard foil"):
        build_clap_case_requests(
            (missing_foil,),
            generated_wav_by_case_id={missing_foil.case_id: tmp_path / "one.wav"},
        )

    with pytest.raises(ValueError, match="differ"):
        ClapCaseRequest(
            case_id="bad",
            generated_wav_path=str(tmp_path / "bad.wav"),
            caption="A dog barks.",
            hard_foil_caption="A dog barks.",
        )


def test_clap_request_rejects_missing_generated_wav_mapping(tmp_path: Path) -> None:
    del tmp_path
    case = _generation_case()

    with pytest.raises(ValueError, match="generated WAV"):
        build_clap_case_requests((case,), generated_wav_by_case_id={})


def test_clap_worker_command_and_result_contract(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"

    command = build_clap_worker_command(
        python="/opt/audio-eval/bin/python",
        request_path=request_path,
        output_path=output_path,
        device="mps",
    )

    assert command == (
        "/opt/audio-eval/bin/python",
        "-m",
        "audex_mac.audio_evaluation_clap_worker",
        "--request",
        str(request_path),
        "--output",
        str(output_path),
        "--device",
        "mps",
    )
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "metrics": {"hard_foil_win_rate": 0.91},
            }
        ),
        encoding="utf-8",
    )
    assert load_clap_worker_result(output_path)["metrics"]["hard_foil_win_rate"] == 0.91
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PROTOCOL_FAIL",
                "reason": "missing_clap_worker_dependencies",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="missing_clap_worker_dependencies"):
        load_clap_worker_result(output_path)


def test_clap_worker_fails_loud_when_dependencies_are_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "audex_mac.audio_evaluation_clap_worker._missing_modules",
        lambda _names: ("transformers",),
    )

    exit_code = run_worker(
        request_path=request_path,
        output_path=tmp_path / "result.json",
        device="mps",
    )

    assert exit_code == 2
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PROTOCOL_FAIL"
    assert payload["reason"] == "missing_clap_worker_dependencies"
