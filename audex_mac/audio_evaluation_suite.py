"""Pinned dataset suite planning for autonomous Audex audio evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from .audio_evaluation import AudioEvaluationCase, EvaluationTrack
from .audio_evaluation_datasets import (
    MaterializedAudio,
    build_caption_cases,
    build_esc50_cases,
    build_mmau_cases,
)
from .audio_evaluation_hf import DatasetPin, select_stratified_rows

MMAU_PIN = DatasetPin(
    repo_id="TwinkStart/MMAU",
    revision="42bd874593a0beed966e505411e896a808f9931f",
    config="default",
    split="v05.15.25",
    license="Apache-2.0",
    expected_rows=1000,
)
ESC50_PIN = DatasetPin(
    repo_id="ashraq/esc50",
    revision="e3e2a63ffff66b9a9735524551e3818e96af03ee",
    config="default",
    split="train",
    license="CC-BY-NC-3.0",
    expected_rows=2000,
)
AUDIOCAPS_CAPTION_PIN = DatasetPin(
    repo_id="d0rj/audiocaps",
    revision="54887eb2a01bf806cdbec0aca41fd85628dac0e4",
    config="default",
    split="test",
    license="MIT",
    expected_rows=4875,
)
AUDIOCAPS_AUDIO_PIN = DatasetPin(
    repo_id="OpenSound/AudioCaps",
    revision="b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded",
    config="default",
    split="test",
    license="CC-BY-NC-4.0",
    expected_rows=4411,
)
SONG_DESCRIBER_PIN = DatasetPin(
    repo_id="renumics/song-describer-dataset",
    revision="dc39062efec7515add304b98a54da2948709a808",
    config="default",
    split="train",
    license="CC-BY-SA-4.0",
    expected_rows=746,
)

SMOKE_MMAU_PER_DOMAIN = 8
SMOKE_ESC50_CASES = 8
SMOKE_AUDIOCAPS_CASES = 4
SMOKE_SONG_DESCRIBER_CASES = 4
STANDARD_MMAU_PER_DOMAIN = 125
STANDARD_ESC50_CASES = 250
STANDARD_AUDIOCAPS_CASES = 64
STANDARD_SONG_DESCRIBER_CASES = 64
STANDARD_CONTROL_DATASET_ID = "audex-mac/ualm-inspired-controls"
STANDARD_CONTROL_REVISION = "2026-07-10"
STANDARD_CONTROL_LICENSE = "local-synthetic-evaluation-prompts"
STANDARD_CONTROL_PROMPTS = (
    ("quantity-01", "Three dogs bark one after another in a small room."),
    ("quantity-02", "A single church bell rings exactly twice, then silence."),
    ("quantity-03", "Five short camera shutter clicks occur at an even pace."),
    ("quantity-04", "Two people clap four times in sync in a quiet studio."),
    ("distance-01", "A thunderclap rumbles far away behind steady light rain."),
    ("distance-02", "A motorcycle passes very close from left to right."),
    ("distance-03", "Children shout faintly from across a large playground."),
    ("distance-04", "A train horn sounds distant, then grows slightly nearer."),
    ("temporal-01", "Glass breaks first, then a dog barks twice."),
    ("temporal-02", "A door opens, footsteps cross the room, then the door shuts."),
    ("temporal-03", "A phone vibrates, pauses, then rings with a bright melody."),
    ("temporal-04", "Rain starts softly, intensifies, and ends with one thunder roll."),
    ("quality-01", "A scratchy old radio plays quiet jazz under heavy static."),
    (
        "quality-02",
        "Clean studio recording of a solo acoustic guitar chord progression.",
    ),
    ("quality-03", "Muffled underwater bubbling with low-frequency resonance."),
    (
        "quality-04",
        "A clipped distorted microphone captures loud cheering in a stadium.",
    ),
    ("negative-01", "Ten seconds of near silence in an empty recording booth."),
    ("negative-02", "White noise fades in slowly and then fades out."),
    ("negative-03", "A quiet room tone with a distant air conditioner hum."),
    ("negative-04", "Soft wind noise with no voices, music, or animal sounds."),
    ("trap-01", "Generate the sound of a piano, not a spoken description of one."),
    ("trap-02", "Create rain and thunder audio; do not include speech."),
    ("trap-03", "Make a kitchen timer beep repeatedly without any narration."),
    ("trap-04", "Produce crowd applause only, with no words or singing."),
)


def build_smoke_cases_from_rows(
    *,
    mmau_rows: tuple[Mapping[str, Any], ...],
    esc50_rows: tuple[Mapping[str, Any], ...],
    audiocaps_rows: tuple[Mapping[str, Any], ...],
    song_describer_rows: tuple[Mapping[str, Any], ...],
    master_seed: int,
    materialize_audio: Callable[[Mapping[str, Any]], MaterializedAudio],
) -> tuple[AudioEvaluationCase, ...]:
    """Build the smoke-tier case manifest from already fetched pinned rows."""

    mmau_selected = select_stratified_rows(
        (
            row
            for row in mmau_rows
            if str(row.get("task", "")).lower() in {"sound", "music"}
        ),
        count=SMOKE_MMAU_PER_DOMAIN * 2,
        master_seed=master_seed,
        row_id=lambda row: str(row.get("id", "")),
        stratum=lambda row: str(row.get("task", "")),
    )
    esc_selected: tuple[Mapping[str, Any], ...] = ()
    if esc50_rows:
        esc_selected = select_stratified_rows(
            esc50_rows,
            count=SMOKE_ESC50_CASES,
            master_seed=master_seed,
            row_id=lambda row: str(row.get("filename", row.get("file", ""))),
            stratum=lambda row: str(row.get("category", "")),
        )
    audiocaps_selected = _select_rows(
        audiocaps_rows,
        count=SMOKE_AUDIOCAPS_CASES,
        master_seed=master_seed,
        row_id=lambda row: str(row.get("audiocap_id", "")),
    )
    song_selected: tuple[Mapping[str, Any], ...] = ()
    if song_describer_rows:
        song_selected = _select_rows(
            song_describer_rows,
            count=SMOKE_SONG_DESCRIBER_CASES,
            master_seed=master_seed,
            row_id=lambda row: str(row.get("caption_id", row.get("track_id", ""))),
        )

    cases: list[AudioEvaluationCase] = []
    cases.extend(
        build_mmau_cases(
            mmau_selected,
            dataset_revision=MMAU_PIN.revision,
            license=MMAU_PIN.license,
            materialize_audio=materialize_audio,
        )
    )
    if esc_selected:
        esc_foils = _placeholder_foil_by_category(esc50_rows)
        cases.extend(
            build_esc50_cases(
                esc_selected,
                dataset_revision=ESC50_PIN.revision,
                materialize_audio=materialize_audio,
                foil_by_category=esc_foils,
            )
        )
    cases.extend(
        build_caption_cases(
            audiocaps_selected,
            dataset_id=AUDIOCAPS_CAPTION_PIN.repo_id,
            dataset_revision=AUDIOCAPS_CAPTION_PIN.revision,
            dataset_config=AUDIOCAPS_CAPTION_PIN.config,
            dataset_split=AUDIOCAPS_CAPTION_PIN.split,
            license=AUDIOCAPS_CAPTION_PIN.license,
            id_field="audiocap_id",
            caption_field="caption",
            category="audiocaps",
        )
    )
    if song_selected:
        cases.extend(
            build_caption_cases(
                song_selected,
                dataset_id=SONG_DESCRIBER_PIN.repo_id,
                dataset_revision=SONG_DESCRIBER_PIN.revision,
                dataset_config=SONG_DESCRIBER_PIN.config,
                dataset_split=SONG_DESCRIBER_PIN.split,
                license=SONG_DESCRIBER_PIN.license,
                id_field="caption_id",
                caption_field="caption",
                category="song-describer",
            )
        )
    return tuple(cases)


def build_standard_cases_from_rows(
    *,
    mmau_rows: tuple[Mapping[str, Any], ...],
    esc50_rows: tuple[Mapping[str, Any], ...],
    audiocaps_rows: tuple[Mapping[str, Any], ...],
    song_describer_rows: tuple[Mapping[str, Any], ...],
    master_seed: int,
    materialize_audio: Callable[[Mapping[str, Any]], MaterializedAudio],
) -> tuple[AudioEvaluationCase, ...]:
    """Build the standard-tier local regression manifest from pinned rows."""

    mmau_selected = select_stratified_rows(
        (
            row
            for row in mmau_rows
            if str(row.get("task", "")).lower() in {"sound", "music"}
        ),
        count=STANDARD_MMAU_PER_DOMAIN * 2,
        master_seed=master_seed,
        row_id=lambda row: str(row.get("id", "")),
        stratum=lambda row: str(row.get("task", "")),
    )
    esc_selected: tuple[Mapping[str, Any], ...] = ()
    if esc50_rows:
        esc_selected = select_stratified_rows(
            esc50_rows,
            count=STANDARD_ESC50_CASES,
            master_seed=master_seed,
            row_id=lambda row: str(row.get("filename", row.get("file", ""))),
            stratum=lambda row: str(row.get("category", "")),
        )
    audiocaps_selected = _select_rows(
        audiocaps_rows,
        count=STANDARD_AUDIOCAPS_CASES,
        master_seed=master_seed,
        row_id=lambda row: str(row.get("audiocap_id", "")),
    )
    song_selected: tuple[Mapping[str, Any], ...] = ()
    if song_describer_rows:
        song_selected = _select_rows(
            song_describer_rows,
            count=STANDARD_SONG_DESCRIBER_CASES,
            master_seed=master_seed,
            row_id=lambda row: str(row.get("caption_id", row.get("track_id", ""))),
        )

    cases: list[AudioEvaluationCase] = []
    cases.extend(
        build_mmau_cases(
            mmau_selected,
            dataset_revision=MMAU_PIN.revision,
            license=MMAU_PIN.license,
            materialize_audio=materialize_audio,
        )
    )
    if esc_selected:
        esc_foils = _placeholder_foil_by_category(esc50_rows)
        cases.extend(
            build_esc50_cases(
                esc_selected,
                dataset_revision=ESC50_PIN.revision,
                materialize_audio=materialize_audio,
                foil_by_category=esc_foils,
            )
        )
    cases.extend(
        build_caption_cases(
            audiocaps_selected,
            dataset_id=AUDIOCAPS_CAPTION_PIN.repo_id,
            dataset_revision=AUDIOCAPS_CAPTION_PIN.revision,
            dataset_config=AUDIOCAPS_CAPTION_PIN.config,
            dataset_split=AUDIOCAPS_CAPTION_PIN.split,
            license=AUDIOCAPS_CAPTION_PIN.license,
            id_field="audiocap_id",
            caption_field="caption",
            category="audiocaps",
        )
    )
    if song_selected:
        cases.extend(
            build_caption_cases(
                song_selected,
                dataset_id=SONG_DESCRIBER_PIN.repo_id,
                dataset_revision=SONG_DESCRIBER_PIN.revision,
                dataset_config=SONG_DESCRIBER_PIN.config,
                dataset_split=SONG_DESCRIBER_PIN.split,
                license=SONG_DESCRIBER_PIN.license,
                id_field="caption_id",
                caption_field="caption",
                category="song-describer",
            )
        )
    cases.extend(_build_standard_control_cases())
    return tuple(cases)


def build_full_cases_from_rows(
    *,
    mmau_rows: tuple[Mapping[str, Any], ...],
    esc50_rows: tuple[Mapping[str, Any], ...],
    audiocaps_rows: tuple[Mapping[str, Any], ...],
    song_describer_rows: tuple[Mapping[str, Any], ...],
    materialize_audio: Callable[[Mapping[str, Any]], MaterializedAudio],
) -> tuple[AudioEvaluationCase, ...]:
    """Build the full paper-style manifest from all supplied pinned rows."""

    mmau_selected = tuple(
        row
        for row in mmau_rows
        if str(row.get("task", "")).lower() in {"sound", "music"}
    )
    cases: list[AudioEvaluationCase] = []
    cases.extend(
        build_mmau_cases(
            mmau_selected,
            dataset_revision=MMAU_PIN.revision,
            license=MMAU_PIN.license,
            materialize_audio=materialize_audio,
        )
    )
    if esc50_rows:
        esc_foils = _placeholder_foil_by_category(esc50_rows)
        cases.extend(
            build_esc50_cases(
                esc50_rows,
                dataset_revision=ESC50_PIN.revision,
                materialize_audio=materialize_audio,
                foil_by_category=esc_foils,
            )
        )
    cases.extend(
        build_caption_cases(
            audiocaps_rows,
            dataset_id=AUDIOCAPS_CAPTION_PIN.repo_id,
            dataset_revision=AUDIOCAPS_CAPTION_PIN.revision,
            dataset_config=AUDIOCAPS_CAPTION_PIN.config,
            dataset_split=AUDIOCAPS_CAPTION_PIN.split,
            license=AUDIOCAPS_CAPTION_PIN.license,
            id_field="audiocap_id",
            caption_field="caption",
            category="audiocaps",
        )
    )
    if song_describer_rows:
        cases.extend(
            build_caption_cases(
                song_describer_rows,
                dataset_id=SONG_DESCRIBER_PIN.repo_id,
                dataset_revision=SONG_DESCRIBER_PIN.revision,
                dataset_config=SONG_DESCRIBER_PIN.config,
                dataset_split=SONG_DESCRIBER_PIN.split,
                license=SONG_DESCRIBER_PIN.license,
                id_field="caption_id",
                caption_field="caption",
                category="song-describer",
            )
        )
    cases.extend(_build_standard_control_cases())
    return tuple(cases)


def _build_standard_control_cases() -> tuple[AudioEvaluationCase, ...]:
    cases: list[AudioEvaluationCase] = []
    for row_id, caption in STANDARD_CONTROL_PROMPTS:
        row = {"id": row_id, "caption": caption}
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        cases.append(
            AudioEvaluationCase(
                case_id=f"control-{row_id}",
                track=EvaluationTrack.GENERATION,
                dataset_id=STANDARD_CONTROL_DATASET_ID,
                dataset_revision=STANDARD_CONTROL_REVISION,
                dataset_config="ualm-inspired",
                dataset_split="standard",
                source_row_id=row_id,
                source_row_hash=hashlib.sha256(canonical.encode()).hexdigest(),
                license=STANDARD_CONTROL_LICENSE,
                category="structured-control",
                prompt=caption,
                caption=caption,
            )
        )
    return tuple(cases)


def _select_rows(
    rows: tuple[Mapping[str, Any], ...],
    *,
    count: int,
    master_seed: int,
    row_id: Callable[[Mapping[str, Any]], str],
) -> tuple[Mapping[str, Any], ...]:
    if len(rows) < count:
        raise ValueError(f"requires {count} rows but has {len(rows)}")
    return select_stratified_rows(
        rows,
        count=count,
        master_seed=master_seed,
        row_id=row_id,
        stratum=lambda _row: "all",
    )


def _placeholder_foil_by_category(
    rows: tuple[Mapping[str, Any], ...],
) -> dict[str, str]:
    """Return a deterministic smoke foil map; standard/full need semantic foils."""

    categories = sorted(
        {
            str(row.get("category", "")).strip().lower()
            for row in rows
            if str(row.get("category", "")).strip()
        }
    )
    if len(categories) < 2:
        raise ValueError("ESC-50 foil mapping requires at least two categories")
    return {
        category: categories[(index + 1) % len(categories)]
        for index, category in enumerate(categories)
    }
