from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.audio_evaluation_targets import (
    baseline_target_profile,
    merge_capability_targets,
    paper_reproduction_targets,
)

pytestmark = pytest.mark.fast


def test_baseline_target_profile_builds_documented_regression_thresholds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "verdict": "CHARACTERIZED",
                "case_completeness": 1.0,
                "protocol_failures": [],
                "accuracy": 0.80,
                "generation": {
                    "completed_cases": 10,
                    "structurally_valid": 9,
                    "semantic_metrics": {"clap": {"hard_foil_win_rate": 0.75}},
                },
            }
        ),
        encoding="utf-8",
    )

    profile = baseline_target_profile(name="30b-bf16-20260710", summary_path=path)

    assert profile.name == "30b-bf16-20260710"
    assert profile.targets == {
        "accuracy_min": pytest.approx(0.78),
        "generation_structural_failures_max": 1.0,
        "hard_foil_win_rate_min": pytest.approx(0.70),
    }
    assert len(profile.summary_sha256) == 64


def test_baseline_target_profile_rejects_invalid_or_incomplete_baseline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "verdict": "PROTOCOL_FAIL",
                "case_completeness": 0.5,
                "protocol_failures": ["incomplete_cases"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="complete protocol-valid run"):
        baseline_target_profile(name="bad", summary_path=path)


def test_paper_reproduction_targets_are_model_specific_and_bf16_only() -> None:
    assert paper_reproduction_targets(tier="full", model="30b", profile="bf16") == {
        "fd_openl3_audiocaps_max": pytest.approx(73.59),
        "fd_openl3_song_describer_max": pytest.approx(68.97),
        "understanding_music_accuracy_min": pytest.approx(0.745),
        "understanding_sound_accuracy_min": pytest.approx(0.785),
    }
    assert (
        paper_reproduction_targets(tier="standard", model="30b", profile="bf16") == {}
    )
    assert paper_reproduction_targets(tier="full", model="30b", profile="nvfp4") == {}


def test_merge_capability_targets_keeps_the_stricter_threshold() -> None:
    assert merge_capability_targets(
        {"accuracy_min": 0.7, "technical_failure_rate_max": 0.1},
        {"accuracy_min": 0.8, "technical_failure_rate_max": 0.2},
    ) == {"accuracy_min": 0.8, "technical_failure_rate_max": 0.1}
