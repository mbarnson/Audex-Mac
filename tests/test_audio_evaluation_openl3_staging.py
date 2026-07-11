from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from audex_mac.audio_evaluation import (
    AudioEvaluationCase,
    AudioEvaluationRun,
    EvaluationTrack,
)
from audex_mac.audio_evaluation_openl3_staging import stage_openl3_corpora

pytestmark = pytest.mark.fast


def _case(case_id: str, *, category: str, source_row_id: str) -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id=case_id,
        track=EvaluationTrack.GENERATION,
        dataset_id=f"fixture/{category}",
        dataset_revision="revision",
        dataset_config="default",
        dataset_split="test",
        source_row_id=source_row_id,
        source_row_hash=f"hash-{case_id}",
        license="fixture",
        category=category,
        prompt=f"Caption {case_id}",
        caption=f"Caption {case_id}",
        hard_foil_caption=f"Foil {case_id}",
    )


def test_stage_openl3_corpora_hardlinks_exact_dataset_filenames(tmp_path: Path) -> None:
    cases = (
        _case("audiocaps-123", category="audiocaps", source_row_id="123"),
        _case(
            "song-describer-abc",
            category="song-describer",
            source_row_id="abc",
        ),
        _case("control", category="structured-control", source_row_id="quantity-01"),
    )
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="openl3-stage",
        tier="standard",
        master_seed=7,
        cases=cases,
        manifest_metadata={},
    )
    for case in cases:
        enhanced = run.run_dir / "media" / "enhanced" / f"{case.case_id}.wav"
        enhanced.parent.mkdir(parents=True, exist_ok=True)
        enhanced.write_bytes(f"wav-{case.case_id}".encode())
        run.record_output(
            case_id=case.case_id,
            payload={"enhanced_wav_path": str(enhanced)},
        )

    staged = stage_openl3_corpora(run)

    assert staged == {"audiocaps": 1, "song-describer": 1}
    audiocaps = run.run_dir / "media" / "openl3" / "audiocaps" / "123.wav"
    song = run.run_dir / "media" / "openl3" / "song-describer" / "abc.wav"
    assert audiocaps.read_bytes() == b"wav-audiocaps-123"
    assert song.read_bytes() == b"wav-song-describer-abc"
    assert (
        os.stat(audiocaps).st_ino
        == os.stat(run.run_dir / "media" / "enhanced" / "audiocaps-123.wav").st_ino
    )
    staging = json.loads(
        (run.run_dir / "generation" / "openl3-staging.json").read_text(encoding="utf-8")
    )
    assert staging["counts"] == staged
    assert len(staging["files"]) == 2


def test_stage_openl3_corpora_fails_on_missing_enhanced_metric_view(
    tmp_path: Path,
) -> None:
    case = _case("audiocaps-123", category="audiocaps", source_row_id="123")
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="openl3-stage",
        tier="standard",
        master_seed=7,
        cases=(case,),
        manifest_metadata={},
    )
    run.record_output(case_id=case.case_id, payload={"raw_wav_path": "raw.wav"})

    with pytest.raises(RuntimeError, match="enhanced metric WAV"):
        stage_openl3_corpora(run)


def test_stage_openl3_corpora_rejects_unsafe_dataset_filename(tmp_path: Path) -> None:
    case = _case(
        "audiocaps-unsafe",
        category="audiocaps",
        source_row_id="../unsafe",
    )
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="openl3-stage",
        tier="standard",
        master_seed=7,
        cases=(case,),
        manifest_metadata={},
    )
    enhanced = run.run_dir / "media" / "enhanced" / "unsafe.wav"
    enhanced.parent.mkdir(parents=True, exist_ok=True)
    enhanced.write_bytes(b"wav")
    run.record_output(
        case_id=case.case_id,
        payload={"enhanced_wav_path": str(enhanced)},
    )

    with pytest.raises(ValueError, match="unsafe OpenL3 filename"):
        stage_openl3_corpora(run)
