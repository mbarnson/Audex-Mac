from __future__ import annotations

import pytest

from audex_mac.audio_evaluation import EvaluationTrack
from audex_mac.audio_evaluation_datasets import (
    MaterializedAudio,
    build_caption_cases,
    build_esc50_cases,
    build_mmau_cases,
)


def _audio(row_id: str) -> MaterializedAudio:
    return MaterializedAudio(
        path=f"/cache/{row_id}.wav",
        sha256=f"sha256-{row_id}",
        sample_rate=16_000,
        duration_seconds=5.0,
    )


@pytest.mark.fast
def test_mmau_rows_become_pinned_multiple_choice_audio_cases() -> None:
    rows = [
        {
            "id": "mmau-1",
            "question": "What is heard first?",
            "choices": ["A dog", "A bell", "Rain", "Music"],
            "answer": "B",
            "dataset": "fixture",
            "task": "sound",
            "split": "test",
            "category": "Reasoning",
            "sub_category": "events",
            "difficulty": "medium",
        }
    ]

    cases = build_mmau_cases(
        rows,
        dataset_revision="deadbeef",
        license="fixture-license",
        materialize_audio=lambda _row: _audio("mmau-1"),
    )

    assert len(cases) == 1
    case = cases[0]
    assert case.track is EvaluationTrack.UNDERSTANDING
    assert case.case_id == "mmau-mmau-1"
    assert case.dataset_id == "TwinkStart/MMAU"
    assert case.dataset_revision == "deadbeef"
    assert case.dataset_split == "v05.15.25"
    assert case.category == "sound"
    assert case.choices == ("A", "B", "C", "D")
    assert case.expected_answer == "B"
    assert "A. A dog" in case.prompt
    assert "Return only the single choice letter." in case.prompt
    assert case.audio_path == "/cache/mmau-1.wav"
    assert case.tags == ("dataset:mmau", "task:sound", "sub-category:events")
    assert len(case.source_row_hash) == 64


@pytest.mark.fast
def test_mmau_builder_rejects_speech_rows_and_unknown_answers() -> None:
    speech = {
        "id": "speech-1",
        "question": "What was said?",
        "choices": ["one", "two"],
        "answer": "A",
        "task": "speech",
        "category": "Reasoning",
    }
    with pytest.raises(ValueError, match="non-speech"):
        build_mmau_cases(
            [speech],
            dataset_revision="deadbeef",
            license="fixture-license",
            materialize_audio=lambda _row: _audio("speech-1"),
        )

    invalid = {**speech, "id": "sound-1", "task": "sound", "answer": "Z"}
    with pytest.raises(ValueError, match="answer.*choices"):
        build_mmau_cases(
            [invalid],
            dataset_revision="deadbeef",
            license="fixture-license",
            materialize_audio=lambda _row: _audio("sound-1"),
        )


@pytest.mark.fast
def test_esc50_rows_become_balanced_entailment_probes_with_hard_foils() -> None:
    rows = [
        {"filename": "dog-1.wav", "category": "dog", "fold": 1, "target": 0},
        {
            "filename": "rooster-1.wav",
            "category": "rooster",
            "fold": 1,
            "target": 1,
        },
    ]

    cases = build_esc50_cases(
        rows,
        dataset_revision="cafebabe",
        materialize_audio=lambda row: _audio(str(row["filename"])),
        foil_by_category={"dog": "rooster", "rooster": "dog"},
    )

    assert [case.expected_answer for case in cases] == ["YES", "NO"]
    assert cases[0].choices == ("YES", "NO")
    assert "Does this recording contain the sound of a dog?" in cases[0].prompt
    assert "Does this recording contain the sound of a dog?" in cases[1].prompt
    assert cases[0].category == "dog"
    assert cases[1].category == "rooster"
    assert cases[0].tags == (
        "dataset:esc50",
        "class:dog",
        "queried-class:dog",
        "query:positive",
    )
    assert cases[1].tags == (
        "dataset:esc50",
        "class:rooster",
        "queried-class:dog",
        "query:hard-foil",
    )


@pytest.mark.fast
def test_caption_rows_are_pinned_without_reference_audio_in_the_prompt() -> None:
    cases = build_caption_cases(
        [
            {
                "audiocap_id": 17,
                "caption": "A dog barks twice beside a passing train.",
                "youtube_id": "secret-source",
            }
        ],
        dataset_id="OpenSound/AudioCaps",
        dataset_revision="1234abcd",
        dataset_config="default",
        dataset_split="test",
        license="CC-BY-4.0",
        id_field="audiocap_id",
        caption_field="caption",
        category="audiocaps",
    )

    assert len(cases) == 1
    case = cases[0]
    assert case.track is EvaluationTrack.GENERATION
    assert case.caption == "A dog barks twice beside a passing train."
    assert case.audio_path is None
    assert case.prompt == case.caption
    assert case.hard_foil_caption is None
    assert case.tags == ("generation:caption", "dataset:opensound-audiocaps")
    assert "secret-source" not in case.prompt
    assert len(case.source_row_hash) == 64


@pytest.mark.fast
def test_caption_rows_get_deterministic_hard_foil_captions() -> None:
    cases = build_caption_cases(
        [
            {"audiocap_id": 2, "caption": "A cat meows."},
            {"audiocap_id": 1, "caption": "A train horn sounds."},
            {"audiocap_id": 3, "caption": "Rain falls on leaves."},
        ],
        dataset_id="OpenSound/AudioCaps",
        dataset_revision="1234abcd",
        dataset_config="default",
        dataset_split="test",
        license="CC-BY-4.0",
        id_field="audiocap_id",
        caption_field="caption",
        category="audiocaps",
    )

    by_id = {case.case_id: case for case in cases}
    assert by_id["audiocaps-1"].hard_foil_caption == "A cat meows."
    assert by_id["audiocaps-2"].hard_foil_caption == "Rain falls on leaves."
    assert by_id["audiocaps-3"].hard_foil_caption == "A train horn sounds."
    assert all(case.hard_foil_caption != case.caption for case in cases)


@pytest.mark.fast
def test_caption_hard_foils_skip_duplicate_captions() -> None:
    cases = build_caption_cases(
        [
            {"audiocap_id": 1, "caption": "A dog barks."},
            {"audiocap_id": 2, "caption": "A dog barks."},
            {"audiocap_id": 3, "caption": "A bell rings."},
        ],
        dataset_id="OpenSound/AudioCaps",
        dataset_revision="1234abcd",
        dataset_config="default",
        dataset_split="test",
        license="CC-BY-4.0",
        id_field="audiocap_id",
        caption_field="caption",
        category="audiocaps",
    )

    assert all(case.hard_foil_caption for case in cases)
    assert all(case.hard_foil_caption != case.caption for case in cases)
