from __future__ import annotations

import pytest

from audex_mac.audio_evaluation import AudioEvaluationCase, EvaluationTrack
from audex_mac.audio_evaluation_generation import TtaOutputInspection
from audex_mac.audio_evaluation_oracles import SignalSanityOracleSuite
from audex_mac.audio_evaluation_runner import GenerationAttempt

pytestmark = pytest.mark.fast


def test_signal_sanity_oracle_qualifies_against_self_tests() -> None:
    qualification = SignalSanityOracleSuite().qualify()

    assert qualification.qualified is True
    assert qualification.failures == ()
    assert qualification.oracle_results["signal_sanity"]["qualified"] is True


def test_signal_sanity_oracle_flags_silence_as_protocol_failure(tmp_path) -> None:
    attempt = GenerationAttempt(
        raw_wav_path=tmp_path / "silent.wav",
        enhanced_wav_path=None,
        structure=TtaOutputInspection(
            codec_ids=tuple(range(4)),
            codec_token_count=4,
            frame_count=1,
            duration_seconds=0.02,
            reached_end_token=True,
            first_phase_mismatch=None,
            unexpected_token_ids=(),
            failures=(),
        ),
        signal_metrics={
            "finite": True,
            "nonempty": True,
            "duration_seconds": 5.0,
            "peak": 0.0,
            "rms": 0.0,
            "dc_offset": 0.0,
            "sample_delta_peak": 0.0,
            "zero_crossing_rate": 0.0,
            "clipped": False,
        },
        elapsed_seconds=1.0,
        finish_reason="stop",
    )

    metrics = SignalSanityOracleSuite().score(_generation_case(), attempt)

    assert metrics["verdict"] == "FAIL"
    assert "audible_peak" in metrics["protocol_failures"]


@pytest.mark.fast
def test_signal_sanity_oracle_flags_dc_bias_and_flat_waveforms(tmp_path) -> None:
    oracle = SignalSanityOracleSuite()

    dc_biased = oracle.score(
        _generation_case(),
        _attempt(
            tmp_path,
            {
                "finite": True,
                "nonempty": True,
                "duration_seconds": 5.0,
                "peak": 0.5,
                "rms": 0.3,
                "dc_offset": 0.4,
                "sample_delta_peak": 0.1,
                "zero_crossing_rate": 0.01,
                "clipped": False,
            },
        ),
    )
    flat = oracle.score(
        _generation_case(),
        _attempt(
            tmp_path,
            {
                "finite": True,
                "nonempty": True,
                "duration_seconds": 5.0,
                "peak": 0.5,
                "rms": 0.3,
                "dc_offset": 0.0,
                "sample_delta_peak": 0.0,
                "zero_crossing_rate": 0.0,
                "clipped": False,
            },
        ),
    )

    assert "dc_offset_in_range" in dc_biased["protocol_failures"]
    assert "sample_variation" in flat["protocol_failures"]
    assert "zero_crossing_activity" in flat["protocol_failures"]


def _attempt(
    tmp_path,
    signal_metrics: dict[str, object],
) -> GenerationAttempt:
    return GenerationAttempt(
        raw_wav_path=tmp_path / "generated.wav",
        enhanced_wav_path=None,
        structure=TtaOutputInspection(
            codec_ids=tuple(range(4)),
            codec_token_count=4,
            frame_count=1,
            duration_seconds=0.02,
            reached_end_token=True,
            first_phase_mismatch=None,
            unexpected_token_ids=(),
            failures=(),
        ),
        signal_metrics=signal_metrics,
        elapsed_seconds=1.0,
        finish_reason="stop",
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
