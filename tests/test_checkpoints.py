from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.checkpoints import (
    local_snapshot_path,
    verify_indexed_checkpoint,
    verify_snapshot,
)
from audex_mac.models import DEFAULT_MODEL

pytestmark = pytest.mark.fast


def test_indexed_checkpoint_detects_missing_shards(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_folder_textonly"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"present")

    result = verify_indexed_checkpoint(checkpoint)

    assert result.complete is False
    assert result.missing_shards == ("model-00002-of-00002.safetensors",)


def test_snapshot_verification_requires_ref_and_exact_files(tmp_path: Path) -> None:
    result = verify_snapshot(
        DEFAULT_MODEL,
        required_files=("checkpoint_folder_full/config.json",),
        checkpoint_dirs=("checkpoint_folder_full",),
        cache_root=tmp_path,
    )

    assert result.complete is False
    assert result.snapshot_path is None
    assert result.missing_summary == ("checkpoint_folder_full/config.json",)


def test_snapshot_verification_checks_indexed_checkpoint(tmp_path: Path) -> None:
    repo_dir = tmp_path / "models--nvidia--Nemotron-Labs-Audex-2B"
    snapshot = repo_dir / "snapshots" / "rev"
    checkpoint = snapshot / "checkpoint_folder_textonly"
    checkpoint.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("rev", encoding="utf-8")
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )

    result = verify_snapshot(
        DEFAULT_MODEL,
        required_files=("checkpoint_folder_textonly/config.json",),
        checkpoint_dirs=("checkpoint_folder_textonly",),
        cache_root=tmp_path,
    )

    assert result.complete is False
    assert result.missing_summary == ("checkpoint_folder_textonly/missing.safetensors",)


def test_local_snapshot_falls_back_when_ref_points_to_absent_revision(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "models--nvidia--Nemotron-Labs-Audex-2B"
    snapshot = repo_dir / "snapshots" / "present"
    snapshot.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("absent", encoding="utf-8")

    assert local_snapshot_path(DEFAULT_MODEL.repo_id, cache_root=tmp_path) == snapshot


def test_snapshot_verification_prefers_complete_older_snapshot_over_partial_ref(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "models--nvidia--Nemotron-Labs-Audex-2B"
    partial = repo_dir / "snapshots" / "new-partial"
    complete = repo_dir / "snapshots" / "older-complete"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("new-partial", encoding="utf-8")
    for snapshot in (partial, complete):
        checkpoint = snapshot / "checkpoint_folder_textonly"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"lm_head.weight": "model.safetensors"}}),
            encoding="utf-8",
        )
    (complete / "checkpoint_folder_textonly" / "model.safetensors").write_bytes(
        b"locally converted weights"
    )

    result = verify_snapshot(
        DEFAULT_MODEL,
        required_files=("checkpoint_folder_textonly/config.json",),
        checkpoint_dirs=("checkpoint_folder_textonly",),
        cache_root=tmp_path,
    )

    assert result.complete is True
    assert result.snapshot_path == complete
