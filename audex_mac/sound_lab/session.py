"""Deep orchestration module for one Audex Sound Lab session."""

from __future__ import annotations

import hashlib
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .catalog import SoundLabCatalog
from .tools import RenderSoundsCall


@dataclass(frozen=True, slots=True)
class VariantBrief:
    caption: str
    difference: str
    seed: int


@dataclass(frozen=True, slots=True)
class GeneratedSound:
    wav_path: Path
    duration_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SoundLabTurn:
    message: str
    job_id: str | None = None
    ready_count: int = 0
    failed_count: int = 0


class SoundLabPlanner(Protocol):
    def plan(self, user_text: str) -> RenderSoundsCall | str: ...


class SoundVariantDesigner(Protocol):
    def design(
        self,
        call: RenderSoundsCall,
        *,
        job_id: str,
    ) -> tuple[VariantBrief, ...]: ...


class SoundGenerator(Protocol):
    def generate(
        self,
        variant: VariantBrief,
        *,
        asset_id: str,
        output_dir: Path,
    ) -> GeneratedSound: ...


class SoundLabSession:
    """Turn one user request into durable, blind, incrementally playable sounds."""

    def __init__(
        self,
        *,
        catalog: SoundLabCatalog,
        planner: SoundLabPlanner,
        designer: SoundVariantDesigner,
        generator: SoundGenerator,
        asset_root: Path,
        model_repo: str,
        recipe: str = "nvidia-tta-cfg3-xcodec1",
        id_factory: Callable[[str], str] | None = None,
    ) -> None:
        self._catalog = catalog
        self._planner = planner
        self._designer = designer
        self._generator = generator
        self._asset_root = Path(asset_root)
        self._model_repo = model_repo
        self._recipe = recipe
        self._id_factory = id_factory or _opaque_id

    def handle(self, user_text: str) -> SoundLabTurn:
        text = user_text.strip()
        if not text:
            raise ValueError("Sound Lab request must not be empty")
        planned = self._planner.plan(text)
        if isinstance(planned, str):
            response = planned.strip()
            if not response:
                raise ValueError("Sound Lab planner returned an empty response")
            return SoundLabTurn(message=response)

        job_id = self._id_factory("job")
        self._catalog.create_job(
            job_id=job_id,
            requested_brief=planned.brief,
            requested_count=planned.count,
            model_repo=self._model_repo,
        )
        try:
            variants = self._validate_variants(
                self._designer.design(planned, job_id=job_id),
                expected_count=planned.count,
            )
        except Exception as exc:
            self._catalog.finish_job(
                job_id,
                failed=True,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        labels = _blind_labels(job_id, len(variants))
        candidates: list[tuple[str, str, VariantBrief]] = []
        for label, variant in zip(labels, variants, strict=True):
            asset_id = self._id_factory("asset")
            candidates.append((asset_id, label, variant))
            self._catalog.add_candidate(
                asset_id=asset_id,
                job_id=job_id,
                blind_label=label,
                caption=variant.caption,
                difference=variant.difference,
                seed=variant.seed,
                recipe=self._recipe,
            )

        ready_count = 0
        failures: list[str] = []
        output_dir = self._asset_root / job_id
        for asset_id, label, variant in candidates:
            self._catalog.mark_candidate_generating(asset_id)
            try:
                generated = self._generator.generate(
                    variant,
                    asset_id=asset_id,
                    output_dir=output_dir,
                )
                if not generated.wav_path.is_file():
                    raise FileNotFoundError(
                        f"generator did not create WAV: {generated.wav_path}"
                    )
                self._catalog.mark_candidate_ready(
                    asset_id,
                    wav_path=generated.wav_path,
                    duration_seconds=generated.duration_seconds,
                    elapsed_seconds=generated.elapsed_seconds,
                )
                ready_count += 1
            except Exception as exc:
                failure = f"{label}: {type(exc).__name__}: {exc}"
                failures.append(failure)
                self._catalog.mark_candidate_failed(asset_id, failure)

        self._catalog.finish_job(
            job_id,
            failed=ready_count == 0,
            error="; ".join(failures),
        )
        message = planned.preamble or (
            f"I designed {len(variants)} distinct sounds for blind audition."
        )
        return SoundLabTurn(
            message=message,
            job_id=job_id,
            ready_count=ready_count,
            failed_count=len(failures),
        )

    @staticmethod
    def _validate_variants(
        variants: tuple[VariantBrief, ...],
        *,
        expected_count: int,
    ) -> tuple[VariantBrief, ...]:
        if len(variants) != expected_count:
            raise ValueError(
                "Sound Lab designer returned "
                f"{len(variants)} variants; expected {expected_count}"
            )
        normalized: set[str] = set()
        for variant in variants:
            caption = " ".join(variant.caption.split())
            difference = " ".join(variant.difference.split())
            if not caption or not difference:
                raise ValueError("Sound Lab variants require caption and difference")
            key = caption.casefold()
            if key in normalized:
                raise ValueError("Sound Lab designer returned duplicate captions")
            normalized.add(key)
        return variants


def _opaque_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _blind_labels(job_id: str, count: int) -> tuple[str, ...]:
    seed = int.from_bytes(hashlib.sha256(job_id.encode("utf-8")).digest()[:8], "big")
    return tuple(random.Random(seed).sample(tuple("ABCDE"), count))
