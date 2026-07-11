from __future__ import annotations

import wave
from pathlib import Path

import pytest

from audex_mac.sound_lab.catalog import SoundLabCatalog
from audex_mac.sound_lab.session import (
    GeneratedSound,
    SoundLabSession,
    VariantBrief,
)
from audex_mac.sound_lab.tools import RenderSoundsCall


class FakePlanner:
    def plan(self, user_text: str) -> RenderSoundsCall | str:
        assert user_text == "Make two very different thunderclaps."
        return RenderSoundsCall(
            brief=user_text,
            count=2,
            constraints={"avoid": ["speech"]},
            parent_asset_ids=(),
            preamble="I'll design two distinct thunderclaps.",
        )


class FakeDesigner:
    def design(
        self, call: RenderSoundsCall, *, job_id: str
    ) -> tuple[VariantBrief, ...]:
        assert call.count == 2
        assert job_id == "job-fixed"
        return (
            VariantBrief("A sharp nearby thunder crack.", "near and dry", 101),
            VariantBrief("Distant rolling thunder in a valley.", "far and long", 202),
        )


class FakeGenerator:
    def __init__(self) -> None:
        self.captions: list[str] = []

    def generate(
        self,
        variant: VariantBrief,
        *,
        asset_id: str,
        output_dir: Path,
    ) -> GeneratedSound:
        self.captions.append(variant.caption)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{asset_id}.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(b"\0\0" * 160)
        return GeneratedSound(path, duration_seconds=0.01, elapsed_seconds=0.25)


@pytest.mark.fast
def test_session_turn_plans_renders_and_publishes_each_blind_candidate(
    tmp_path: Path,
) -> None:
    catalog = SoundLabCatalog(tmp_path / "catalog.sqlite3")
    generator = FakeGenerator()
    identifiers = iter(("job-fixed", "asset-one", "asset-two"))
    session = SoundLabSession(
        catalog=catalog,
        planner=FakePlanner(),
        designer=FakeDesigner(),
        generator=generator,
        asset_root=tmp_path / "assets",
        model_repo="audex-fixture",
        id_factory=lambda _prefix: next(identifiers),
    )

    turn = session.handle("Make two very different thunderclaps.")

    assert turn.job_id == "job-fixed"
    assert turn.message == "I'll design two distinct thunderclaps."
    assert turn.ready_count == 2
    assert turn.failed_count == 0
    assert generator.captions == [
        "A sharp nearby thunder crack.",
        "Distant rolling thunder in a valley.",
    ]
    snapshot = catalog.public_snapshot()
    assert snapshot["jobs"][0]["state"] == "complete"
    assert {item["label"] for item in snapshot["jobs"][0]["candidates"]} == {
        "A",
        "B",
    }
    assert all(item["state"] == "ready" for item in snapshot["jobs"][0]["candidates"])
    assert all("caption" not in item for item in snapshot["jobs"][0]["candidates"])
