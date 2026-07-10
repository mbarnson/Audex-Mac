from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.audio_evaluation import (
    AudioEvaluationCase,
    AudioEvaluationRun,
    EvaluationTrack,
    RunVerdict,
    derive_case_seed,
    score_constrained_answer,
    select_stratified_cases,
)


def _case(case_id: str, category: str) -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id=case_id,
        track=EvaluationTrack.UNDERSTANDING,
        dataset_id="fixture/audio",
        dataset_revision="abc123",
        dataset_config="default",
        dataset_split="test",
        source_row_id=case_id,
        source_row_hash=f"hash-{case_id}",
        license="CC0-1.0",
        category=category,
        prompt="Which sound is present? A. rain B. dog",
        expected_answer="B",
        audio_path=f"/cache/{case_id}.wav",
        choices=("A", "B"),
    )


def _generation_case(case_id: str, category: str = "audiocaps") -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id=case_id,
        track=EvaluationTrack.GENERATION,
        dataset_id="fixture/captions",
        dataset_revision="def456",
        dataset_config="default",
        dataset_split="test",
        source_row_id=case_id,
        source_row_hash=f"hash-{case_id}",
        license="CC-BY-4.0",
        category=category,
        prompt="A dog barks twice.",
        caption="A dog barks twice.",
        tags=("generation:caption",),
    )


def _yes_no_case(case_id: str, expected: str) -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id=case_id,
        track=EvaluationTrack.UNDERSTANDING,
        dataset_id="fixture/binary",
        dataset_revision="abc123",
        dataset_config="default",
        dataset_split="test",
        source_row_id=case_id,
        source_row_hash=f"hash-{case_id}",
        license="CC0-1.0",
        category="binary",
        prompt="Does this recording contain a dog? Return only YES or NO.",
        expected_answer=expected,
        audio_path=f"/cache/{case_id}.wav",
        choices=("YES", "NO"),
    )


@pytest.mark.fast
def test_stratified_selection_is_balanced_stable_and_order_independent() -> None:
    cases = tuple(
        _case(f"{category}-{index}", category)
        for category in ("sound", "music")
        for index in range(10)
    )

    selected = select_stratified_cases(cases, count=8, master_seed=20260710)
    reversed_selected = select_stratified_cases(
        tuple(reversed(cases)), count=8, master_seed=20260710
    )

    assert [case.case_id for case in selected] == [
        case.case_id for case in reversed_selected
    ]
    assert sum(case.category == "sound" for case in selected) == 4
    assert sum(case.category == "music" for case in selected) == 4
    assert len({derive_case_seed(20260710, case.case_id) for case in selected}) == 8


@pytest.mark.fast
def test_stratified_selection_fails_when_a_stratum_cannot_supply_its_share() -> None:
    cases = (
        *(_case(f"sound-{i}", "sound") for i in range(8)),
        _case("music-0", "music"),
    )

    with pytest.raises(ValueError, match="music.*requires 4.*has 1"):
        select_stratified_cases(cases, count=8, master_seed=1)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("raw", "expected_valid", "expected_correct"),
    [
        ("B", True, True),
        ("Answer: b.", True, True),
        ("A", True, False),
        ("A or B", False, False),
        ("The dog is barking", False, False),
    ],
)
def test_constrained_answer_scoring_is_fail_closed(
    raw: str, expected_valid: bool, expected_correct: bool
) -> None:
    score = score_constrained_answer(raw, choices=("A", "B"), expected="B")

    assert score.valid is expected_valid
    assert score.correct is expected_correct


@pytest.mark.fast
def test_run_artifacts_require_complete_outputs_before_characterizing(
    tmp_path: Path,
) -> None:
    cases = (_case("sound-1", "sound"), _case("music-1", "music"))
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="run-1",
        tier="smoke",
        master_seed=17,
        cases=cases,
        manifest_metadata={"model": {"repository": "nvidia/audex"}},
    )

    run.record_output(
        case_id="sound-1",
        payload={"raw_answer": "B", "valid": True, "correct": True},
    )
    incomplete = run.finalize(required_oracles_qualified=True)

    assert incomplete.verdict is RunVerdict.PROTOCOL_FAIL
    assert incomplete.missing_case_ids == ("music-1",)

    run.record_output(
        case_id="music-1",
        payload={"raw_answer": "B", "valid": True, "correct": True},
    )
    complete = run.finalize(required_oracles_qualified=True)

    assert complete.verdict is RunVerdict.CHARACTERIZED
    assert complete.missing_case_ids == ()
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert summary["case_completeness"] == 1.0
    assert summary["verdict"] == "CHARACTERIZED"
    assert (run.run_dir / "understanding" / "cases.jsonl").is_file()
    assert (run.run_dir / "understanding" / "outputs.jsonl").is_file()


@pytest.mark.fast
def test_run_artifacts_record_case_tags(tmp_path: Path) -> None:
    case = AudioEvaluationCase(
        case_id="tagged-generation-1",
        track=EvaluationTrack.GENERATION,
        dataset_id="fixture/captions",
        dataset_revision="def456",
        dataset_config="default",
        dataset_split="test",
        source_row_id="tagged-generation-1",
        source_row_hash="hash-tagged-generation-1",
        license="CC-BY-4.0",
        category="structured-control",
        prompt="A timer beeps twice.",
        caption="A timer beeps twice.",
        tags=("control:quantity", "generation:structured-control"),
    )
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="tagged-run",
        tier="smoke",
        master_seed=17,
        cases=(case,),
        manifest_metadata={"model": {"repository": "nvidia/audex"}},
    )

    payload = json.loads(
        (run.run_dir / "generation" / "cases.jsonl").read_text(encoding="utf-8")
    )
    assert payload["tags"] == ["control:quantity", "generation:structured-control"]


@pytest.mark.fast
def test_run_summary_reports_category_and_generation_breakdowns(
    tmp_path: Path,
) -> None:
    cases = (
        _case("sound-1", "sound"),
        _case("sound-2", "sound"),
        _case("music-1", "music"),
        _yes_no_case("binary-positive", "YES"),
        _yes_no_case("binary-negative", "NO"),
        _generation_case("audiocaps-1"),
    )
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="breakdowns",
        tier="smoke",
        master_seed=17,
        cases=cases,
        manifest_metadata={"model": {"repository": "nvidia/audex"}},
    )

    run.record_output(
        case_id="sound-1",
        payload={
            "raw_answer": "B",
            "valid": True,
            "correct": True,
            "elapsed_seconds": 0.5,
        },
    )
    run.record_output(
        case_id="sound-2",
        payload={
            "raw_answer": "A",
            "valid": True,
            "correct": False,
            "elapsed_seconds": 1.5,
        },
    )
    run.record_output(
        case_id="music-1",
        payload={
            "raw_answer": "dog",
            "valid": False,
            "correct": False,
            "elapsed_seconds": 1.0,
        },
    )
    run.record_output(
        case_id="binary-positive",
        payload={
            "raw_answer": "NO",
            "normalized_answer": "NO",
            "valid": True,
            "correct": False,
            "elapsed_seconds": 1.0,
        },
    )
    run.record_output(
        case_id="binary-negative",
        payload={
            "raw_answer": "YES",
            "normalized_answer": "YES",
            "valid": True,
            "correct": False,
            "elapsed_seconds": 1.0,
        },
    )
    run.record_output(
        case_id="audiocaps-1",
        payload={
            "structurally_valid": False,
            "structure_failures": ["missing_end_token"],
            "duration_seconds": 10.0,
            "elapsed_seconds": 5.0,
            "signal_metrics": {
                "finite": True,
                "nonempty": True,
                "clipped": True,
            },
        },
    )

    summary = run.finalize(required_oracles_qualified=True)

    assert summary.accuracy == pytest.approx(1 / 5)
    assert summary.balanced_accuracy == pytest.approx(1 / 6)
    payload = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert payload["understanding_by_category"]["sound"] == {
        "accuracy": 0.5,
        "completed_cases": 2,
        "correct": 1,
        "invalid_response_rate": 0.0,
        "total_cases": 2,
        "valid_responses": 2,
    }
    assert payload["understanding_by_category"]["music"]["invalid_response_rate"] == 1.0
    accuracy_ci = payload["confidence_intervals"]["accuracy"]
    assert accuracy_ci["method"] == "deterministic_nonparametric_bootstrap"
    assert accuracy_ci["samples"] == 2000
    assert (
        0.0 <= accuracy_ci["lower"] <= summary.accuracy <= accuracy_ci["upper"] <= 1.0
    )
    assert payload["binary_rates"] == {
        "completed_cases": 2,
        "false_negative_rate": 1.0,
        "false_negatives": 1,
        "false_positive_rate": 1.0,
        "false_positives": 1,
        "negative_cases": 1,
        "positive_cases": 1,
    }
    assert payload["generation"] == {
        "completed_cases": 1,
        "signal_failures": {"clipped_waveform": 1},
        "structural_failures": {"missing_end_token": 1},
        "structurally_valid": 0,
        "total_cases": 1,
    }
    assert payload["technical_failures"] == {
        "by_track": {
            "generation": {
                "completed_cases": 1,
                "technical_failure_rate": 0.0,
                "technical_failures": 0,
            },
            "understanding": {
                "completed_cases": 5,
                "technical_failure_rate": 0.0,
                "technical_failures": 0,
            },
        },
        "completed_cases": 6,
        "failures": {},
        "technical_failure_rate": 0.0,
        "technical_failures": 0,
    }
    assert payload["diagnostics"]["elapsed_seconds_total"] == 10.0
    assert payload["diagnostics"]["cases_per_second"] == 0.6
    assert (
        payload["diagnostics"]["by_track"]["understanding"]["elapsed_seconds_mean"]
        == 1.0
    )
    assert (
        payload["diagnostics"]["by_track"]["generation"]["audio_realtime_ratio"] == 2.0
    )
    assert payload["diagnostics"]["started_at_utc"].endswith("Z")
    assert payload["diagnostics"]["finalized_at_utc"].endswith("Z")
    assert payload["diagnostics"]["wall_clock_seconds"] >= 0.0
    assert payload["diagnostics"]["process_peak_rss"]["value"] > 0
    assert (
        payload["diagnostics"]["process_peak_rss"]["source"]
        == "resource.getrusage(RUSAGE_SELF).ru_maxrss"
    )


@pytest.mark.fast
def test_run_summary_applies_explicit_capability_targets(tmp_path: Path) -> None:
    cases = (_case("sound-1", "sound"), _case("sound-2", "sound"))
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="target-pass",
        tier="standard",
        master_seed=17,
        cases=cases,
        manifest_metadata={"model": {"repository": "nvidia/audex"}},
    )
    for case in cases:
        run.record_output(
            case_id=case.case_id,
            payload={"raw_answer": "B", "valid": True, "correct": True},
        )

    summary = run.finalize(
        required_oracles_qualified=True,
        capability_targets={
            "accuracy_min": 1.0,
            "invalid_response_rate_max": 0.0,
        },
    )

    assert summary.verdict is RunVerdict.PASS
    assert summary.capability_failures == ()
    payload = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert payload["capability_targets"] == {
        "accuracy_min": 1.0,
        "invalid_response_rate_max": 0.0,
    }


@pytest.mark.fast
def test_run_summary_reports_capability_failures_against_named_targets(
    tmp_path: Path,
) -> None:
    cases = (_case("sound-1", "sound"), _case("sound-2", "sound"))
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="target-fail",
        tier="standard",
        master_seed=17,
        cases=cases,
        manifest_metadata={"model": {"repository": "nvidia/audex"}},
    )
    run.record_output(
        case_id="sound-1",
        payload={"raw_answer": "B", "valid": True, "correct": True},
    )
    run.record_output(
        case_id="sound-2",
        payload={"raw_answer": "A", "valid": True, "correct": False},
    )

    summary = run.finalize(
        required_oracles_qualified=True,
        capability_targets={"accuracy_min": 0.75},
    )

    assert summary.verdict is RunVerdict.CAPABILITY_FAIL
    assert summary.capability_failures == ("accuracy_min:0.5<0.75",)


@pytest.mark.fast
def test_protocol_failures_dominate_capability_targets(tmp_path: Path) -> None:
    case = _case("sound-1", "sound")
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="target-protocol-fail",
        tier="standard",
        master_seed=17,
        cases=(case,),
        manifest_metadata={"model": {"repository": "nvidia/audex"}},
    )

    summary = run.finalize(
        required_oracles_qualified=True,
        capability_targets={"accuracy_min": 1.0},
    )

    assert summary.verdict is RunVerdict.PROTOCOL_FAIL
    assert summary.protocol_failures == ("incomplete_cases",)
    assert summary.capability_failures == ()


@pytest.mark.fast
def test_run_artifacts_reject_secrets_and_duplicate_case_outputs(
    tmp_path: Path,
) -> None:
    case = _case("sound-1", "sound")
    with pytest.raises(ValueError, match="credential-like manifest key"):
        AudioEvaluationRun.create(
            root=tmp_path,
            run_id="bad",
            tier="smoke",
            master_seed=17,
            cases=(case,),
            manifest_metadata={"hf_token": "secret"},
        )

    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="ok",
        tier="smoke",
        master_seed=17,
        cases=(case,),
        manifest_metadata={
            "sampling": {"max_tokens": 2048},
            "model": {"tokenizer_hash": "abc123"},
        },
    )
    run.record_output(case_id=case.case_id, payload={"valid": True})
    with pytest.raises(ValueError, match="already recorded"):
        run.record_output(case_id=case.case_id, payload={"valid": True})
