from __future__ import annotations

import wave
from pathlib import Path

import pytest

from audex_mac.sound_lab.catalog import SoundLabCatalog
from audex_mac.sound_lab.session import (
    GeneratedSound,
    SoundGenerationOutcome,
    SoundGenerationRequest,
    SoundLabSession,
    VariantBrief,
    VariantDesignError,
    VariantDesignResult,
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
    def design(self, call: RenderSoundsCall, *, job_id: str) -> VariantDesignResult:
        assert call.count == 2
        assert job_id == "job-fixed"
        return VariantDesignResult(
            variants=(
                VariantBrief("A sharp nearby thunder crack.", "near and dry", 101),
                VariantBrief(
                    "Distant rolling thunder in a valley.", "far and long", 202
                ),
            ),
            raw_attempts=("designer raw response",),
            repair_used=True,
        )


class FakeGenerator:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def generate_many(
        self,
        requests: tuple[SoundGenerationRequest, ...],
        *,
        output_dir: Path,
    ) -> tuple[SoundGenerationOutcome, ...]:
        self.batches.append([request.variant.caption for request in requests])
        outcomes: list[SoundGenerationOutcome] = []
        for request in requests:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{request.asset_id}.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(b"\0\0" * 160)
            outcomes.append(
                SoundGenerationOutcome(
                    asset_id=request.asset_id,
                    generated=GeneratedSound(
                        path,
                        duration_seconds=0.01,
                        elapsed_seconds=0.25,
                        seed_used=request.variant.seed,
                    ),
                )
            )
        return tuple(outcomes)


class FailingDesigner:
    def design(self, call: RenderSoundsCall, *, job_id: str) -> VariantDesignResult:
        del call, job_id
        raise VariantDesignError(
            ("first invalid", "repair invalid"),
            ("raw first", "raw repair"),
        )


class PartialThenExplodingGenerator:
    def generate_many(
        self,
        requests: tuple[SoundGenerationRequest, ...],
        *,
        output_dir: Path,
    ) -> tuple[SoundGenerationOutcome, ...]:
        def outcomes():
            first = requests[0]
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{first.asset_id}.wav"
            path.write_bytes(b"RIFFfixture")
            yield SoundGenerationOutcome(
                asset_id=first.asset_id,
                generated=GeneratedSound(path, 1.0, 0.5, first.variant.seed),
            )
            raise RuntimeError("retry batch exploded")

        return outcomes()


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
    assert generator.batches == [
        [
            "A sharp nearby thunder crack.",
            "Distant rolling thunder in a valley.",
        ]
    ]
    snapshot = catalog.public_snapshot()
    assert snapshot["jobs"][0]["state"] == "complete"
    assert {item["label"] for item in snapshot["jobs"][0]["candidates"]} == {
        "A",
        "B",
    }
    assert all(item["state"] == "ready" for item in snapshot["jobs"][0]["candidates"])
    assert all("caption" not in item for item in snapshot["jobs"][0]["candidates"])
    assert catalog.job_diagnostics("job-fixed") == {
        "designer_raw_attempts": ["designer raw response"],
        "designer_repair_used": True,
        "failure": None,
    }


@pytest.mark.fast
def test_session_persists_failed_designer_attempts(tmp_path: Path) -> None:
    catalog = SoundLabCatalog(tmp_path / "catalog.sqlite3")
    session = SoundLabSession(
        catalog=catalog,
        planner=FakePlanner(),
        designer=FailingDesigner(),
        generator=FakeGenerator(),
        asset_root=tmp_path / "assets",
        model_repo="audex-fixture",
        id_factory=lambda _prefix: "job-fixed",
    )

    with pytest.raises(VariantDesignError):
        session.handle("Make two very different thunderclaps.")

    diagnostics = catalog.job_diagnostics("job-fixed")
    assert diagnostics["designer_raw_attempts"] == ["raw first", "raw repair"]
    assert diagnostics["designer_repair_used"] is True
    assert "first invalid" in str(diagnostics["failure"])


@pytest.mark.fast
def test_session_preserves_ready_outcomes_if_a_later_retry_batch_raises(
    tmp_path: Path,
) -> None:
    catalog = SoundLabCatalog(tmp_path / "catalog.sqlite3")
    identifiers = iter(("job-fixed", "asset-one", "asset-two"))
    session = SoundLabSession(
        catalog=catalog,
        planner=FakePlanner(),
        designer=FakeDesigner(),
        generator=PartialThenExplodingGenerator(),
        asset_root=tmp_path / "assets",
        model_repo="audex-fixture",
        id_factory=lambda _prefix: next(identifiers),
    )

    turn = session.handle("Make two very different thunderclaps.")

    assert turn.ready_count == 1
    assert turn.failed_count == 1
    states = {
        candidate["asset_id"]: candidate["state"]
        for candidate in catalog.public_snapshot()["jobs"][0]["candidates"]
    }
    assert states == {"asset-one": "ready", "asset-two": "failed"}
