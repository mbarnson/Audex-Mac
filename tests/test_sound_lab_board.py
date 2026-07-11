from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from audex_mac.sound_lab.board import SoundLabBoard
from audex_mac.sound_lab.catalog import SoundLabCatalog


@pytest.mark.fast
def test_board_serves_blind_audio_and_records_audition(tmp_path: Path) -> None:
    catalog = SoundLabCatalog(tmp_path / "catalog.sqlite3")
    catalog.create_job(
        job_id="job-1",
        requested_brief="Two impacts",
        requested_count=2,
        model_repo="fixture",
    )
    for asset_id, label in (("asset-1", "A"), ("asset-2", "B")):
        catalog.add_candidate(
            asset_id=asset_id,
            job_id="job-1",
            blind_label=label,
            caption=f"secret caption {label}",
            difference=f"secret difference {label}",
            seed=1,
            recipe="cfg3",
        )
    wav = tmp_path / "asset-1.wav"
    wav.write_bytes(b"RIFFblind-audio")
    catalog.mark_candidate_ready(
        "asset-1", wav_path=wav, duration_seconds=10.0, elapsed_seconds=2.0
    )
    opened: list[str] = []

    with SoundLabBoard(catalog, opener=opened.append) as board:
        assert opened == [board.url]
        snapshot = _json_request(f"{board.url}/api/state")
        assert snapshot["jobs"][0]["candidates"][0]["label"] == "A"
        assert "caption" not in snapshot["jobs"][0]["candidates"][0]
        with urllib.request.urlopen(
            f"{board.url}/audio/asset-1", timeout=2
        ) as response:
            assert response.headers["Content-Type"] == "audio/wav"
            assert response.read() == b"RIFFblind-audio"

        saved = _json_request(
            f"{board.url}/api/preferences",
            method="POST",
            payload={
                "job_id": "job-1",
                "selected_label": "A",
                "rejected_labels": ["B"],
                "note": "A wins.",
            },
        )
        assert saved == {"ok": True}
        revealed = _json_request(
            f"{board.url}/api/reveal",
            method="POST",
            payload={"job_id": "job-1"},
        )
        assert revealed == {"ok": True}
        snapshot = _json_request(f"{board.url}/api/state")
        assert snapshot["jobs"][0]["candidates"][0]["caption"] == "secret caption A"


def _json_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))
