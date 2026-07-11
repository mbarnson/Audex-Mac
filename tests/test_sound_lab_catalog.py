from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from audex_mac.sound_lab.catalog import SoundLabCatalog


@pytest.mark.fast
def test_catalog_persists_blind_candidates_preferences_and_reveal(
    tmp_path: Path,
) -> None:
    database = tmp_path / "catalog.sqlite3"
    catalog = SoundLabCatalog(database)
    catalog.create_job(
        job_id="job-1",
        requested_brief="Five explosions",
        requested_count=2,
        model_repo="audex-fixture",
    )
    catalog.record_design_attempts(
        "job-1",
        raw_attempts=("```json\n{...}\n```", '{"variants": []}'),
        repair_used=True,
    )
    catalog.add_candidate(
        asset_id="asset-secret-1",
        job_id="job-1",
        blind_label="B",
        caption="A close dry blast",
        difference="close and dry",
        seed=101,
        recipe="cfg3",
    )
    catalog.add_candidate(
        asset_id="asset-secret-2",
        job_id="job-1",
        blind_label="A",
        caption="A distant canyon blast",
        difference="distant with a long tail",
        seed=202,
        recipe="cfg3",
    )
    wav_path = tmp_path / "asset-secret-1.wav"
    wav_path.write_bytes(b"RIFFfixture")
    catalog.mark_candidate_ready(
        "asset-secret-1",
        wav_path=wav_path,
        duration_seconds=10.0,
        elapsed_seconds=3.5,
    )
    catalog.record_preference(
        job_id="job-1",
        selected_label="B",
        rejected_labels=("A",),
        note="B has the better impact.",
    )

    blind = SoundLabCatalog(database).public_snapshot()

    assert blind["jobs"][0]["job_id"] == "job-1"
    assert [item["label"] for item in blind["jobs"][0]["candidates"]] == ["A", "B"]
    assert "caption" not in blind["jobs"][0]["candidates"][0]
    assert "seed" not in blind["jobs"][0]["candidates"][0]
    assert blind["jobs"][0]["preference"] == {
        "selected_label": "B",
        "rejected_labels": ["A"],
        "note": "B has the better impact.",
    }
    assert catalog.audio_path("asset-secret-1") == wav_path
    assert SoundLabCatalog(database).job_diagnostics("job-1") == {
        "designer_raw_attempts": ["```json\n{...}\n```", '{"variants": []}'],
        "designer_repair_used": True,
        "failure": None,
    }

    with pytest.raises(ValueError, match="preference"):
        other_catalog = SoundLabCatalog(tmp_path / "other.sqlite3")
        other_catalog.create_job(
            job_id="unrated",
            requested_brief="One sound",
            requested_count=1,
            model_repo="fixture",
        )
        other_catalog.reveal_job("unrated")

    catalog.reveal_job("job-1")
    revealed = SoundLabCatalog(database).public_snapshot()
    candidates = revealed["jobs"][0]["candidates"]
    assert candidates[0]["caption"] == "A distant canyon blast"
    assert candidates[0]["seed"] == 202
    assert candidates[0]["recipe"] == "cfg3"
    assert candidates[1]["caption"] == "A close dry blast"
    assert candidates[1]["seed"] == 101


@pytest.mark.fast
def test_catalog_migrates_existing_jobs_for_designer_attempt_diagnostics(
    tmp_path: Path,
) -> None:
    database = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("""
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                requested_brief TEXT NOT NULL,
                requested_count INTEGER NOT NULL,
                model_repo TEXT NOT NULL,
                state TEXT NOT NULL,
                failure TEXT,
                revealed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)

    catalog = SoundLabCatalog(database)
    catalog.create_job(
        job_id="new-job",
        requested_brief="A dog barking",
        requested_count=1,
        model_repo="fixture",
    )
    catalog.record_design_attempts(
        "new-job",
        raw_attempts=("raw",),
        repair_used=False,
    )

    assert catalog.job_diagnostics("new-job")["designer_raw_attempts"] == ["raw"]
