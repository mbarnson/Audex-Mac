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
)
from audex_mac.audio_evaluation_generation import TtaOutputInspection
from audex_mac.audio_evaluation_runner import (
    AudioEvaluationRunner,
    GenerationAttempt,
    OracleQualification,
    UnderstandingAttempt,
    UnqualifiedOracleSuite,
)


class FakeUnderstandingAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def answer(self, case: AudioEvaluationCase, *, seed: int) -> UnderstandingAttempt:
        self.calls.append((case.case_id, seed))
        return UnderstandingAttempt(
            raw_answer="B",
            elapsed_seconds=0.2,
            finish_reason="stop",
        )


class FakeGenerationAdapter:
    def __init__(self, raw_wav_path: Path) -> None:
        self.raw_wav_path = raw_wav_path
        self.calls: list[tuple[str, int]] = []

    def generate(self, case: AudioEvaluationCase, *, seed: int) -> GenerationAttempt:
        self.calls.append((case.case_id, seed))
        return GenerationAttempt(
            raw_wav_path=self.raw_wav_path,
            enhanced_wav_path=None,
            structure=TtaOutputInspection(
                codec_ids=tuple(range(2000)),
                codec_token_count=2000,
                frame_count=500,
                duration_seconds=10.0,
                reached_end_token=True,
                first_phase_mismatch=None,
                unexpected_token_ids=(),
                failures=(),
            ),
            signal_metrics={"finite": True, "nonempty": True, "peak": 0.8},
            elapsed_seconds=1.5,
            finish_reason="stop",
        )


class FakeOracleSuite:
    def qualify(self) -> OracleQualification:
        return OracleQualification(
            qualified=True,
            oracle_results={"clap": {"qualified": True}},
            failures=(),
        )

    def score(
        self, case: AudioEvaluationCase, attempt: GenerationAttempt
    ) -> dict[str, object]:
        return {"clap_score": 0.71, "hard_foil_won": True}


class FailingMetricOracleSuite(FakeOracleSuite):
    def score(
        self, case: AudioEvaluationCase, attempt: GenerationAttempt
    ) -> dict[str, object]:
        del case, attempt
        return {
            "verdict": "FAIL",
            "protocol_failures": ["audible_peak"],
        }


def _understanding_case() -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id="mmau-1",
        track=EvaluationTrack.UNDERSTANDING,
        dataset_id="fixture/mmau",
        dataset_revision="rev1",
        dataset_config="default",
        dataset_split="test",
        source_row_id="1",
        source_row_hash="hash1",
        license="CC0",
        category="sound",
        prompt="Choose A or B.",
        expected_answer="B",
        audio_path="/cache/one.wav",
        choices=("A", "B"),
    )


def _generation_case() -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id="audiocaps-1",
        track=EvaluationTrack.GENERATION,
        dataset_id="fixture/audiocaps",
        dataset_revision="rev1",
        dataset_config="default",
        dataset_split="test",
        source_row_id="1",
        source_row_hash="hash2",
        license="CC0",
        category="audiocaps",
        prompt="A dog barks twice.",
        caption="A dog barks twice.",
    )


@pytest.mark.fast
def test_runner_executes_mixed_smoke_run_and_records_scores(tmp_path: Path) -> None:
    cases = (_understanding_case(), _generation_case())
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="smoke-1",
        tier="smoke",
        master_seed=55,
        cases=cases,
        manifest_metadata={"model": {"repository": "fixture/audex"}},
    )
    raw_wav = tmp_path / "raw.wav"
    raw_wav.write_bytes(b"RIFF-fixture")
    understanding = FakeUnderstandingAdapter()
    generation = FakeGenerationAdapter(raw_wav)

    summary = AudioEvaluationRunner(
        understanding=understanding,
        generation=generation,
        oracles=FakeOracleSuite(),
    ).run(run, master_seed=55)

    assert summary.verdict is RunVerdict.CHARACTERIZED
    assert understanding.calls == [("mmau-1", derive_case_seed(55, "mmau-1"))]
    assert generation.calls == [("audiocaps-1", derive_case_seed(55, "audiocaps-1"))]
    understanding_output = json.loads(
        (run.run_dir / "understanding" / "outputs.jsonl").read_text()
    )
    assert understanding_output["normalized_answer"] == "B"
    assert understanding_output["correct"] is True
    generation_output = json.loads(
        (run.run_dir / "generation" / "outputs.jsonl").read_text()
    )
    assert generation_output["codec_token_count"] == 2000
    assert generation_output["structurally_valid"] is True
    generation_metric = json.loads(
        (run.run_dir / "generation" / "metrics.jsonl").read_text()
    )
    assert generation_metric["clap_score"] == 0.71
    assert generation_metric["hard_foil_won"] is True
    qualification = json.loads(run.oracle_qualification_path.read_text())
    assert qualification == {
        "failures": [],
        "oracle_results": {"clap": {"qualified": True}},
        "qualified": True,
    }


@pytest.mark.fast
def test_runner_marks_structural_failure_as_protocol_failure(tmp_path: Path) -> None:
    case = _generation_case()
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="bad-generation",
        tier="smoke",
        master_seed=55,
        cases=(case,),
        manifest_metadata={},
    )
    generation = FakeGenerationAdapter(tmp_path / "missing.wav")
    valid = generation.generate

    def invalid_generate(
        selected_case: AudioEvaluationCase, *, seed: int
    ) -> GenerationAttempt:
        attempt = valid(selected_case, seed=seed)
        return GenerationAttempt(
            raw_wav_path=attempt.raw_wav_path,
            enhanced_wav_path=None,
            structure=TtaOutputInspection(
                codec_ids=(),
                codec_token_count=0,
                frame_count=0,
                duration_seconds=0.0,
                reached_end_token=False,
                first_phase_mismatch=None,
                unexpected_token_ids=(),
                failures=("missing_end_token", "incomplete_target"),
            ),
            signal_metrics={"finite": True, "nonempty": False},
            elapsed_seconds=1.5,
            finish_reason="length",
        )

    generation.generate = invalid_generate  # type: ignore[method-assign]
    summary = AudioEvaluationRunner(
        understanding=FakeUnderstandingAdapter(),
        generation=generation,
        oracles=FakeOracleSuite(),
    ).run(run, master_seed=55)

    assert summary.verdict is RunVerdict.PROTOCOL_FAIL


@pytest.mark.fast
def test_unqualified_generation_oracle_fails_closed(tmp_path: Path) -> None:
    case = _generation_case()
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="unqualified-generation",
        tier="smoke",
        master_seed=55,
        cases=(case,),
        manifest_metadata={},
    )
    raw_wav = tmp_path / "raw.wav"
    raw_wav.write_bytes(b"RIFF-fixture")

    summary = AudioEvaluationRunner(
        understanding=FakeUnderstandingAdapter(),
        generation=FakeGenerationAdapter(raw_wav),
        oracles=UnqualifiedOracleSuite(),
    ).run(run, master_seed=55)

    assert summary.verdict is RunVerdict.PROTOCOL_FAIL
    assert "generation_oracles_not_qualified" in summary.protocol_failures
    assert "required_oracle_qualification_failed" in summary.protocol_failures
    generation_metric = json.loads(
        (run.run_dir / "generation" / "metrics.jsonl").read_text()
    )
    assert generation_metric == {
        "case_id": "audiocaps-1",
        "reason": "oracle_not_qualified",
        "verdict": "UNSCORED",
    }
    qualification = json.loads(run.oracle_qualification_path.read_text())
    assert qualification == {
        "failures": ["generation_oracles_not_qualified"],
        "oracle_results": {"status": "generation_oracles_not_qualified"},
        "qualified": False,
    }


@pytest.mark.fast
def test_runner_promotes_oracle_protocol_failures(tmp_path: Path) -> None:
    case = _generation_case()
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="oracle-failure",
        tier="smoke",
        master_seed=55,
        cases=(case,),
        manifest_metadata={},
    )
    raw_wav = tmp_path / "raw.wav"
    raw_wav.write_bytes(b"RIFF-fixture")

    summary = AudioEvaluationRunner(
        understanding=FakeUnderstandingAdapter(),
        generation=FakeGenerationAdapter(raw_wav),
        oracles=FailingMetricOracleSuite(),
    ).run(run, master_seed=55)

    assert summary.verdict is RunVerdict.PROTOCOL_FAIL
    assert "audiocaps-1: audible_peak" in summary.protocol_failures
