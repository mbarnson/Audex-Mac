from __future__ import annotations

from typing import Any

import pytest

from audex_mac.audio_evaluation import EvaluationTrack
from audex_mac.audio_evaluation_datasets import MaterializedAudio
from audex_mac.audio_evaluation_suite import (
    AUDIOCAPS_CAPTION_PIN,
    MMAU_PIN,
    SONG_DESCRIBER_PIN,
    build_smoke_cases_from_rows,
)


def _mmau_row(row_id: str, task: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "audio": [{"src": f"https://example.invalid/{row_id}.wav"}],
        "question": f"What is heard in {row_id}?",
        "choices": ["A dog", "A bell"],
        "answer": "A dog",
        "dataset": "MMAU",
        "task": task,
        "split": "test",
        "category": "Reasoning",
        "sub_category": "events",
        "difficulty": "medium",
    }


def _esc_row(filename: str, category: str) -> dict[str, Any]:
    return {
        "filename": filename,
        "category": category,
        "fold": 1,
        "target": 0,
    }


def _caption_row(row_id: int, caption: str) -> dict[str, Any]:
    return {
        "audiocap_id": row_id,
        "caption": caption,
        "youtube_id": f"yt-{row_id}",
    }


def _song_row(row_id: int, caption: str) -> dict[str, Any]:
    return {
        "caption_id": row_id,
        "track_id": row_id + 1000,
        "caption": caption,
    }


@pytest.mark.fast
def test_smoke_suite_selects_pinned_cases_before_materializing_audio() -> None:
    mmau_rows = tuple(
        _mmau_row(f"{task}-{index}", task)
        for task in ("sound", "music")
        for index in range(12)
    )
    esc_rows = tuple(
        _esc_row(f"{category}-{index}.wav", category)
        for category in ("dog", "rooster", "rain", "clock")
        for index in range(4)
    )
    audiocaps_rows = tuple(
        _caption_row(index, f"AudioCaps caption {index}") for index in range(8)
    )
    song_rows = tuple(
        _song_row(index, f"SongDescriber caption {index}") for index in range(8)
    )
    materialized: list[str] = []

    def materialize(row: dict[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        materialized.append(row_id)
        return MaterializedAudio(
            path=f"/cache/{row_id}.wav",
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=5.0,
        )

    cases = build_smoke_cases_from_rows(
        mmau_rows=mmau_rows,
        esc50_rows=esc_rows,
        audiocaps_rows=audiocaps_rows,
        song_describer_rows=song_rows,
        master_seed=20260710,
        materialize_audio=materialize,
    )

    assert len(cases) == 32
    assert len(materialized) == 24
    assert sum(case.category == "sound" for case in cases) == 8
    assert sum(case.category == "music" for case in cases) == 8
    assert sum(case.dataset_id == "ashraq/esc50" for case in cases) == 8
    assert sum(case.track is EvaluationTrack.GENERATION for case in cases) == 8
    assert {case.license for case in cases if case.dataset_id == MMAU_PIN.repo_id} == {
        "Apache-2.0"
    }
    assert {
        case.license
        for case in cases
        if case.dataset_id == AUDIOCAPS_CAPTION_PIN.repo_id
    } == {"MIT"}
    assert {
        case.license for case in cases if case.dataset_id == SONG_DESCRIBER_PIN.repo_id
    } == {"CC-BY-SA-4.0"}


@pytest.mark.fast
def test_smoke_suite_keeps_song_describer_optional_when_rows_are_absent() -> None:
    mmau_rows = tuple(
        _mmau_row(f"{task}-{index}", task)
        for task in ("sound", "music")
        for index in range(8)
    )
    esc_rows = tuple(
        _esc_row(f"{category}-{index}.wav", category)
        for category in ("dog", "rooster", "rain", "clock")
        for index in range(4)
    )
    audiocaps_rows = tuple(
        _caption_row(index, f"AudioCaps caption {index}") for index in range(4)
    )

    cases = build_smoke_cases_from_rows(
        mmau_rows=mmau_rows,
        esc50_rows=esc_rows,
        audiocaps_rows=audiocaps_rows,
        song_describer_rows=(),
        master_seed=20260710,
        materialize_audio=lambda row: MaterializedAudio(
            path=f"/cache/{row.get('id') or row.get('filename')}.wav",
            sha256=f"sha-{row.get('id') or row.get('filename')}",
            sample_rate=16_000,
            duration_seconds=5.0,
        ),
    )

    assert len(cases) == 28
    assert not any(case.dataset_id == SONG_DESCRIBER_PIN.repo_id for case in cases)


@pytest.mark.fast
def test_smoke_suite_allows_explicitly_empty_esc50_rows() -> None:
    mmau_rows = tuple(
        _mmau_row(f"{task}-{index}", task)
        for task in ("sound", "music")
        for index in range(8)
    )
    audiocaps_rows = tuple(
        _caption_row(index, f"AudioCaps caption {index}") for index in range(4)
    )
    song_rows = tuple(
        _song_row(index, f"SongDescriber caption {index}") for index in range(4)
    )

    cases = build_smoke_cases_from_rows(
        mmau_rows=mmau_rows,
        esc50_rows=(),
        audiocaps_rows=audiocaps_rows,
        song_describer_rows=song_rows,
        master_seed=20260710,
        materialize_audio=lambda row: MaterializedAudio(
            path=f"/cache/{row.get('id') or row.get('filename')}.wav",
            sha256=f"sha-{row.get('id') or row.get('filename')}",
            sample_rate=16_000,
            duration_seconds=5.0,
        ),
    )

    assert len(cases) == 24
    assert not any(case.dataset_id == "ashraq/esc50" for case in cases)
