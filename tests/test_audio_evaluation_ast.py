from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.audio_evaluation import AudioEvaluationCase, EvaluationTrack
from audex_mac.audio_evaluation_ast import (
    AST_REVISION,
    AstCaseRequest,
    AstQualificationRequest,
    build_ast_case_requests,
    build_ast_qualification_requests,
    build_ast_worker_command,
    load_ast_worker_result,
    write_ast_worker_request,
)
from audex_mac.audio_evaluation_ast_backend import require_torch_device
from audex_mac.audio_evaluation_ast_labels import (
    ESC50_AST_EXPECTED_LABELS,
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


def _esc50_case(*, case_id: str, category: str) -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id=case_id,
        track=EvaluationTrack.UNDERSTANDING,
        dataset_id="ashraq/esc50",
        dataset_revision="rev1",
        dataset_config="default",
        dataset_split="train",
        source_row_id=f"{case_id}.wav",
        source_row_hash=f"hash-{case_id}",
        license="CC-BY-NC-3.0",
        category=category,
        prompt="Return YES or NO.",
        expected_answer="YES",
        audio_path=f"/cache/{case_id}.wav",
        choices=("YES", "NO"),
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


def test_esc50_ast_label_map_covers_all_fixed_categories() -> None:
    assert len(ESC50_AST_EXPECTED_LABELS) == 50
    assert ESC50_AST_EXPECTED_LABELS["dog"] == ("Dog",)
    assert ESC50_AST_EXPECTED_LABELS["mouse_click"] == ("Mouse", "Clicking")
    assert ESC50_AST_EXPECTED_LABELS["washing_machine"] == (
        "Mechanical fan",
        "Hum",
    )


def test_explicit_ast_label_maps_cover_only_hand_authored_control_cases() -> None:
    labeled = _generation_case(source_row_id="quantity-01")
    unlabeled = _generation_case(
        case_id="audiocaps-1",
        source_row_id="audiocaps-1",
    )

    expected, forbidden = explicit_ast_label_maps((labeled, unlabeled))

    assert expected == {labeled.case_id: ("Dog", "Bark")}
    assert forbidden == {labeled.case_id: ("Speech", "Music")}


def test_ast_qualification_uses_one_fixed_mapping_per_esc50_class() -> None:
    requests = build_ast_qualification_requests(
        (
            _esc50_case(case_id="dog-b", category="dog"),
            _esc50_case(case_id="dog-a", category="dog"),
            _generation_case(),
        )
    )

    assert requests == (
        AstQualificationRequest(
            case_id="dog-a",
            audio_path="/cache/dog-a.wav",
            expected_labels=("Dog",),
            forbidden_labels=("Chicken, rooster", "Pig", "Cattle, bovinae"),
        ),
    )


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
        "qualification_requests": [],
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
    with pytest.raises(ValueError, match="forbidden"):
        AstQualificationRequest(
            case_id="bad-calibration",
            audio_path=str(tmp_path / "bad-calibration.wav"),
            expected_labels=("Dog",),
            forbidden_labels=(),
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
                "run_id": "invalid-run",
                "model": {
                    "repo_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
                    "revision": AST_REVISION,
                },
                "logit_policy": "sigmoid_raw_logits",
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


def test_ast_worker_scores_expected_and_forbidden_event_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    dog_wav = tmp_path / "dog.wav"
    speech_wav = tmp_path / "speech.wav"
    dog_wav.write_bytes(b"RIFF dog")
    speech_wav.write_bytes(b"RIFF speech")
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "fixture-run",
                "model": {
                    "repo_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
                    "revision": AST_REVISION,
                },
                "logit_policy": "sigmoid_raw_logits",
                "requests": [
                    {
                        "case_id": "dog",
                        "generated_wav_path": str(dog_wav),
                        "expected_labels": ["Dog", "Bark"],
                        "forbidden_labels": ["Speech"],
                    },
                    {
                        "case_id": "speech",
                        "generated_wav_path": str(speech_wav),
                        "expected_labels": ["Dog"],
                        "forbidden_labels": ["Speech"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeBackend:
        labels = frozenset(("Dog", "Bark", "Speech", "Music"))
        model_load_seconds = 0.25

        def classify_audio(
            self,
            paths: list[Path],
        ) -> tuple[list[dict[str, float]], float, float]:
            assert paths == [dog_wav, speech_wav]
            return (
                [
                    {"Dog": 0.91, "Bark": 0.72, "Speech": 0.02, "Music": 0.01},
                    {"Dog": 0.03, "Bark": 0.01, "Speech": 0.88, "Music": 0.02},
                ],
                0.30,
                0.40,
            )

    monkeypatch.setattr(
        "audex_mac.audio_evaluation_ast_worker._missing_modules",
        lambda _names: (),
    )

    exit_code = run_worker(
        request_path=request_path,
        output_path=output_path,
        device="mps",
        backend_factory=lambda **_kwargs: FakeBackend(),
    )

    assert exit_code == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "UNSCORED"
    assert payload["reason"] == "ast_oracle_not_qualified"
    assert payload["qualification"] == {
        "qualified": False,
        "status": "NOT_RUN",
        "thresholds": {
            "max_forbidden_label_false_positive_rate": 0.15,
            "min_case_count": 50,
            "min_expected_label_hit_rate": 0.85,
        },
    }
    assert payload["model"] == {
        "device": "mps",
        "repo_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "revision": AST_REVISION,
    }
    assert payload["logit_policy"] == "sigmoid_raw_logits"
    assert payload["metrics"] == {
        "expected_label_cases": 2,
        "expected_label_hit_rate": 0.5,
        "forbidden_label_cases": 2,
        "forbidden_label_false_positive_rate": 0.5,
    }
    assert payload["per_case"][0] == {
        "case_id": "dog",
        "expected_label_hit": True,
        "expected_label_scores": {"Bark": 0.72, "Dog": 0.91},
        "forbidden_label_false_positive": False,
        "forbidden_label_scores": {"Speech": 0.02},
        "top_labels": [
            {"label": "Dog", "probability": 0.91},
            {"label": "Bark", "probability": 0.72},
            {"label": "Speech", "probability": 0.02},
            {"label": "Music", "probability": 0.01},
        ],
    }
    assert payload["per_case"][1]["expected_label_hit"] is False
    assert payload["per_case"][1]["forbidden_label_false_positive"] is True
    assert payload["timings"] == {
        "inference_seconds": 0.4,
        "model_load_seconds": 0.25,
        "preprocessing_seconds": 0.3,
    }


def test_ast_worker_qualifies_with_fixed_label_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    generated_wav = tmp_path / "generated.wav"
    dog_wav = tmp_path / "dog.wav"
    speech_wav = tmp_path / "speech.wav"
    generated_wav.write_bytes(b"RIFF generated")
    dog_wav.write_bytes(b"RIFF dog")
    speech_wav.write_bytes(b"RIFF speech")
    write_ast_worker_request(
        request_path,
        run_id="qualified-run",
        requests=(
            AstCaseRequest(
                case_id="generated",
                generated_wav_path=str(generated_wav),
                expected_labels=("Dog", "Bark"),
                forbidden_labels=("Speech",),
            ),
        ),
        qualification_requests=(
            AstQualificationRequest(
                case_id="dog-calibration",
                audio_path=str(dog_wav),
                expected_labels=("Dog", "Bark"),
                forbidden_labels=("Speech",),
            ),
            AstQualificationRequest(
                case_id="speech-calibration",
                audio_path=str(speech_wav),
                expected_labels=("Speech",),
                forbidden_labels=("Dog", "Bark"),
            ),
        ),
    )

    class FakeBackend:
        labels = frozenset(("Dog", "Bark", "Speech", "Music"))
        model_load_seconds = 0.25

        def classify_audio(
            self,
            paths: list[Path],
        ) -> tuple[list[dict[str, float]], float, float]:
            scores = {
                generated_wav: {
                    "Dog": 0.91,
                    "Bark": 0.72,
                    "Speech": 0.02,
                    "Music": 0.01,
                },
                dog_wav: {
                    "Dog": 0.93,
                    "Bark": 0.74,
                    "Speech": 0.01,
                    "Music": 0.01,
                },
                speech_wav: {
                    "Dog": 0.01,
                    "Bark": 0.02,
                    "Speech": 0.92,
                    "Music": 0.03,
                },
            }
            return ([scores[path] for path in paths], 0.30, 0.40)

    monkeypatch.setattr(
        "audex_mac.audio_evaluation_ast_worker._missing_modules",
        lambda _names: (),
    )
    monkeypatch.setattr(
        "audex_mac.audio_evaluation_ast_worker.AST_MIN_QUALIFICATION_CASES",
        2,
    )

    exit_code = run_worker(
        request_path=request_path,
        output_path=output_path,
        device="mps",
        backend_factory=lambda **_kwargs: FakeBackend(),
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert "reason" not in payload
    assert payload["qualification"]["qualified"] is True
    assert payload["qualification"]["expected_label_hit_rate"] == 1.0
    assert payload["qualification"]["forbidden_label_false_positive_rate"] == 0.0
    assert payload["metrics"]["expected_label_hit_rate"] == 1.0


def test_ast_backend_requires_the_explicit_accelerator() -> None:
    class Availability:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = Availability()

        class backends:
            mps = Availability()

    with pytest.raises(RuntimeError, match="MPS.*not available"):
        require_torch_device(FakeTorch(), "mps")
    with pytest.raises(RuntimeError, match="CUDA.*not available"):
        require_torch_device(FakeTorch(), "cuda")
    assert require_torch_device(FakeTorch(), "cpu") == "cpu"
