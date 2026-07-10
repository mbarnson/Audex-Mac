from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from audex_mac import audio_evaluation_cli, cli
from audex_mac.audio_evaluation_datasets import MaterializedAudio
from audex_mac.audio_evaluation_hf import DatasetPin

pytestmark = pytest.mark.fast


def _mmau_row(row_id: str, task: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "audio": [{"src": f"https://example.invalid/{row_id}.wav"}],
        "question": f"What is heard in {row_id}?",
        "choices": ["A dog", "A bell"],
        "answer": "A dog",
        "task": task,
        "category": "Reasoning",
    }


def _esc_row(filename: str, category: str) -> dict[str, Any]:
    return {"filename": filename, "category": category, "fold": 1, "target": 0}


def _caption_row(row_id: int, caption: str) -> dict[str, Any]:
    return {"audiocap_id": row_id, "caption": caption}


def _song_row(row_id: int, caption: str) -> dict[str, Any]:
    return {"caption_id": row_id, "track_id": row_id + 1000, "caption": caption}


@pytest.mark.parametrize("first_arg", ["eval-audio-capabilities"])
def test_top_level_cli_delegates_audio_evaluation_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    first_arg: str,
) -> None:
    calls: list[list[str]] = []

    def fake_eval_main(argv: list[str]) -> int:
        calls.append(argv)
        return 7

    monkeypatch.setattr(cli, "run_audio_evaluation_cli", fake_eval_main)

    assert cli.main([first_arg, "--tier", "smoke"]) == 7
    assert calls == [["--tier", "smoke"]]


def test_audio_evaluation_cli_materializes_smoke_manifest_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows_by_repo = {
        "TwinkStart/MMAU": tuple(
            _mmau_row(f"{task}-{index}", task)
            for task in ("sound", "music")
            for index in range(8)
        ),
        "ashraq/esc50": tuple(
            _esc_row(f"{category}-{index}.wav", category)
            for category in ("dog", "rooster", "rain", "clock")
            for index in range(4)
        ),
        "d0rj/audiocaps": tuple(
            _caption_row(index, f"AudioCaps caption {index}") for index in range(4)
        ),
        "renumics/song-describer-dataset": tuple(
            _song_row(index, f"SongDescriber caption {index}") for index in range(4)
        ),
    }

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        del client
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF fixture")
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=5.0,
        )

    monkeypatch.setenv("HF_TOKEN", "hf_should_not_be_recorded")

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "smoke",
            "--materialize-only",
            "--run-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-id",
            "smoke-test",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
    )

    assert exit_code == 0
    assert "smoke-test" in capsys.readouterr().out
    run_dir = tmp_path / "runs" / "smoke-test"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert manifest["tier"] == "smoke"
    assert manifest["case_count"] == 32
    assert manifest["model"]["size"] == "30b"
    assert environment["hf_token_present"] is True
    assert "hf_should_not_be_recorded" not in json.dumps(manifest)
    assert "hf_should_not_be_recorded" not in json.dumps(environment)
