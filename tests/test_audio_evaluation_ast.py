from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.audio_evaluation import AudioEvaluationCase, EvaluationTrack
from audex_mac.audio_evaluation_ast import (
    AST_REVISION,
    AstCaseRequest,
    build_ast_case_requests,
    build_ast_worker_command,
    load_ast_worker_result,
    write_ast_worker_request,
)
from audex_mac.audio_evaluation_ast_labels import (
    PINNED_AST_LABEL_FIXTURE_VOCABULARY,
    STRUCTURED_CONTROL_AST_LABELS,
    explicit_ast_label_maps,
)
from audex_mac.audio_evaluation_ast_worker import run_worker

pytestmark = pytest.mark.fast


def _generation_case(
    *,
    case_id: str = "control-quantity-01",
    source_row_id: str | None = None,
    caption: str = "A dog barks twice.",
) -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id=case_id,
        track=EvaluationTrack.GENERATION,
        dataset_id="fixture/audiocaps",
        dataset_revision="rev1",
        dataset_config="default",
        dataset_split="test",
        source_row_id=source_row_id or case_id,
        source_row_hash=f"hash-{case_id}",
        license="CC0",
        category="structured-control",
        prompt=caption,
        caption=caption,
        tags=("control:quantity", "generation:structured-control"),
    )


def test_structured_control_ast_labels_match_the_pinned_checkpoint_vocabulary() -> None:
    used_labels = {
        label
        for expected_and_forbidden in STRUCTURED_CONTROL_AST_LABELS.values()
        for labels in expected_and_forbidden
        for label in labels
    }

    assert used_labels <= PINNED_AST_LABEL_FIXTURE_VOCABULARY
    assert "Liquid" in used_labels
    assert "Wind noise (microphone)" in used_labels


def test_explicit_ast_label_maps_cover_only_hand_authored_control_cases() -> None:
    labeled = _generation_case(source_row_id="quantity-01")
    unlabeled = _generation_case(
        case_id="audiocaps-1",
        source_row_id="audiocaps-1",
    )

    expected, forbidden = explicit_ast_label_maps((labeled, unlabeled))

    assert expected == {labeled.case_id: ("Dog", "Bark")}
    assert forbidden == {labeled.case_id: ("Speech", "Music")}


def test_ast_worker_request_requires_explicit_expected_labels(
    tmp_path: Path,
) -> None:
    case = _generation_case()
    requests = build_ast_case_requests(
        (case,),
        generated_wav_by_case_id={case.case_id: tmp_path / "generated.wav"},
        expected_labels_by_case_id={case.case_id: ("Dog", "Bark")},
        forbidden_labels_by_case_id={case.case_id: ("Speech",)},
    )
    request_path = tmp_path / "request.json"

    write_ast_worker_request(
        request_path,
        run_id="standard-run",
        requests=requests,
    )

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert payload == {
        "logit_policy": "sigmoid_raw_logits",
        "metrics": {
            "expected_label_hits": True,
            "forbidden_label_false_positives": True,
        },
        "model": {
            "repo_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "revision": AST_REVISION,
        },
        "requests": [
            {
                "case_id": "control-quantity-01",
                "expected_labels": ["Dog", "Bark"],
                "forbidden_labels": ["Speech"],
                "generated_wav_path": str(tmp_path / "generated.wav"),
            }
        ],
        "run_id": "standard-run",
        "schema_version": 1,
    }


def test_ast_request_rejects_missing_labels_or_overlapping_labels(
    tmp_path: Path,
) -> None:
    case = _generation_case()
    with pytest.raises(ValueError, match="AST labels"):
        build_ast_case_requests(
            (case,),
            generated_wav_by_case_id={case.case_id: tmp_path / "generated.wav"},
            expected_labels_by_case_id={},
        )

    with pytest.raises(ValueError, match="overlap"):
        AstCaseRequest(
            case_id="bad",
            generated_wav_path=str(tmp_path / "bad.wav"),
            expected_labels=("Dog",),
            forbidden_labels=("Dog",),
        )


def test_ast_request_rejects_missing_generated_wav_mapping() -> None:
    case = _generation_case()

    with pytest.raises(ValueError, match="generated WAV"):
        build_ast_case_requests(
            (case,),
            generated_wav_by_case_id={},
            expected_labels_by_case_id={case.case_id: ("Dog",)},
        )


def test_ast_worker_command_and_result_contract(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"

    command = build_ast_worker_command(
        python="/opt/audio-eval/bin/python",
        request_path=request_path,
        output_path=output_path,
        device="mps",
    )

    assert command == (
        "/opt/audio-eval/bin/python",
        "-m",
        "audex_mac.audio_evaluation_ast_worker",
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
                "metrics": {"expected_label_hit_rate": 0.88},
            }
        ),
        encoding="utf-8",
    )
    assert (
        load_ast_worker_result(output_path)["metrics"]["expected_label_hit_rate"]
        == 0.88
    )
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PROTOCOL_FAIL",
                "reason": "missing_ast_worker_dependencies",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="missing_ast_worker_dependencies"):
        load_ast_worker_result(output_path)


def test_ast_worker_fails_loud_when_dependencies_are_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "audex_mac.audio_evaluation_ast_worker._missing_modules",
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
    assert payload["reason"] == "missing_ast_worker_dependencies"


def test_ast_worker_rejects_invalid_request_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "requests": [
                    {
                        "case_id": "bad",
                        "generated_wav_path": "/tmp/bad.wav",
                        "expected_labels": ["Dog"],
                        "forbidden_labels": ["Dog"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "audex_mac.audio_evaluation_ast_worker._missing_modules",
        lambda _names: (),
    )

    exit_code = run_worker(
        request_path=request_path,
        output_path=tmp_path / "result.json",
        device="mps",
    )

    assert exit_code == 2
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PROTOCOL_FAIL"
    assert payload["reason"] == "invalid_ast_request"
    assert "overlap" in payload["detail"]
