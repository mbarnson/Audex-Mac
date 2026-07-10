from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from audex_mac.audio_evaluation_datasets import MaterializedAudio, build_mmau_cases
from audex_mac.audio_evaluation_hf import (
    DatasetPin,
    HfAudioMaterializer,
    HfDatasetClient,
    fetch_verified_rows,
    select_stratified_rows,
)


class FakeTransport:
    def __init__(self, payloads: Mapping[str, Mapping[str, Any] | bytes]) -> None:
        self.payloads = dict(payloads)
        self.requested_urls: list[str] = []

    def get_json(self, url: str, *, headers: Mapping[str, str]) -> Mapping[str, Any]:
        self.requested_urls.append(url)
        payload = self.payloads[url]
        assert isinstance(payload, Mapping)
        return payload

    def get_bytes(self, url: str, *, headers: Mapping[str, str]) -> bytes:
        self.requested_urls.append(url)
        payload = self.payloads[url]
        assert isinstance(payload, bytes)
        return payload


def _pin() -> DatasetPin:
    return DatasetPin(
        repo_id="TwinkStart/MMAU",
        revision="42bd874593a0beed966e505411e896a808f9931f",
        config="default",
        split="v05.15.25",
        license="Apache-2.0",
        expected_rows=4,
    )


def _row(row_id: str, task: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "audio": [
            {
                "src": f"https://datasets-server.local/assets/{row_id}?Expires=123",
                "type": "audio/wav",
            }
        ],
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


@pytest.mark.fast
def test_fetch_verified_rows_rejects_revision_drift_and_truncated_cells() -> None:
    pin = _pin()
    good_base = "https://huggingface.co/api/datasets/TwinkStart/MMAU"
    row_url = (
        "https://datasets-server.huggingface.co/rows?"
        "dataset=TwinkStart%2FMMAU&config=default&split=v05.15.25&offset=0&length=100"
    )
    drift = FakeTransport(
        {
            good_base: {"sha": "wrong"},
            row_url: {},
        }
    )

    with pytest.raises(ValueError, match="revision drift"):
        fetch_verified_rows(pin, client=HfDatasetClient(transport=drift))

    truncated = FakeTransport(
        {
            good_base: {"sha": pin.revision},
            row_url: {
                "num_rows_total": 4,
                "rows": [
                    {"row": _row("sound-1", "sound"), "truncated_cells": ["audio"]}
                ],
            },
        }
    )

    with pytest.raises(ValueError, match="truncated"):
        fetch_verified_rows(pin, client=HfDatasetClient(transport=truncated))


@pytest.mark.fast
def test_fetch_verified_rows_paginates_and_keeps_ephemeral_audio_assets() -> None:
    pin = _pin()
    base = "https://huggingface.co/api/datasets/TwinkStart/MMAU"
    first_page = (
        "https://datasets-server.huggingface.co/rows?"
        "dataset=TwinkStart%2FMMAU&config=default&split=v05.15.25&offset=0&length=2"
    )
    second_page = (
        "https://datasets-server.huggingface.co/rows?"
        "dataset=TwinkStart%2FMMAU&config=default&split=v05.15.25&offset=2&length=2"
    )
    transport = FakeTransport(
        {
            base: {"sha": pin.revision},
            first_page: {
                "num_rows_total": 4,
                "rows": [
                    {"row": _row("sound-1", "sound"), "truncated_cells": []},
                    {"row": _row("music-1", "music"), "truncated_cells": []},
                ],
            },
            second_page: {
                "num_rows_total": 4,
                "rows": [
                    {"row": _row("sound-2", "sound"), "truncated_cells": []},
                    {"row": _row("music-2", "music"), "truncated_cells": []},
                ],
            },
        }
    )

    rows = fetch_verified_rows(
        pin,
        client=HfDatasetClient(transport=transport),
        page_size=2,
    )

    assert [row["id"] for row in rows] == ["sound-1", "music-1", "sound-2", "music-2"]
    serialized = json.dumps(rows, sort_keys=True)
    assert "datasets-server.local/assets" in serialized
    assert rows[0]["audio"][0]["src"].startswith("https://datasets-server.local")


@pytest.mark.fast
def test_stratified_row_selection_happens_before_audio_materialization(
    tmp_path: Path,
) -> None:
    rows = tuple(
        _row(f"{task}-{index}", task)
        for task in ("sound", "music")
        for index in range(5)
    )
    selected = select_stratified_rows(
        rows,
        count=4,
        master_seed=20260710,
        row_id=lambda row: str(row["id"]),
        stratum=lambda row: str(row["task"]),
    )
    materialized_ids: list[str] = []

    def materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row["id"])
        materialized_ids.append(row_id)
        return MaterializedAudio(
            path=str(tmp_path / f"{row_id}.wav"),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=5.0,
        )

    cases = build_mmau_cases(
        selected,
        dataset_revision=_pin().revision,
        license=_pin().license,
        materialize_audio=materialize,
    )

    assert len(cases) == 4
    assert len(materialized_ids) == 4
    assert {case.category for case in cases} == {"sound", "music"}


@pytest.mark.fast
def test_hf_audio_materializer_downloads_asset_to_decoder(tmp_path: Path) -> None:
    row = _row("sound-1", "sound")
    asset_url = row["audio"][0]["src"]
    transport = FakeTransport({asset_url: b"wav-bytes"})
    decoded: list[tuple[bytes, Path]] = []

    def decoder(raw_bytes: bytes, destination: Path) -> MaterializedAudio:
        decoded.append((raw_bytes, destination))
        destination.write_bytes(b"RIFF decoded")
        return MaterializedAudio(
            path=str(destination),
            sha256="decoded-sha",
            sample_rate=16_000,
            duration_seconds=5.0,
        )

    materializer = HfAudioMaterializer(
        client=HfDatasetClient(transport=transport),
        cache_dir=tmp_path,
        decoder=decoder,
    )

    audio = materializer.materialize(row)

    assert audio.path.endswith(".wav")
    assert decoded == [(b"wav-bytes", Path(audio.path))]
    assert Path(audio.path).read_bytes() == b"RIFF decoded"
