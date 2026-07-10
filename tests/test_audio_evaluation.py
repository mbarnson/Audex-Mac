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
