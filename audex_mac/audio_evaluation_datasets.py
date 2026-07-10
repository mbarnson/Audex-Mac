"""Dataset-to-case adapters for autonomous audio evaluation."""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .audio_evaluation import AudioEvaluationCase, EvaluationTrack

MMAU_DATASET_ID = "TwinkStart/MMAU"
MMAU_CONFIG = "default"
MMAU_SPLIT = "v05.15.25"
ESC50_DATASET_ID = "ashraq/esc50"
ESC50_CONFIG = "default"
ESC50_SPLIT = "train"
ESC50_LICENSE = "CC-BY-NC-3.0"


@dataclass(frozen=True, slots=True)
class MaterializedAudio:
    """One evaluator-ready local audio view plus its provenance hash."""

    path: str
    sha256: str
    sample_rate: int
    duration_seconds: float

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("materialized audio path must not be empty")
        if not self.sha256.strip():
            raise ValueError("materialized audio sha256 must not be empty")
        if self.sample_rate != 16_000:
            raise ValueError(
                f"Audex evaluation audio must be 16000 Hz, got {self.sample_rate}"
            )
        if not 0.0 < self.duration_seconds <= 30.0:
            raise ValueError("Audex evaluation audio must be between 0 and 30 seconds")


def build_mmau_cases(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_revision: str,
    license: str,
    materialize_audio: Callable[[Mapping[str, Any]], MaterializedAudio],
) -> tuple[AudioEvaluationCase, ...]:
    """Convert pinned MMAU sound/music rows into exact-choice cases."""

    cases: list[AudioEvaluationCase] = []
    for index, row in enumerate(rows):
        category = str(row.get("task", "")).strip().lower()
        if category not in {"sound", "music"}:
            raise ValueError(
                f"MMAU autonomous evaluation accepts non-speech sound/music rows; "
                f"row {index} has task {category!r}"
            )
        row_id = str(row.get("id", "")).strip()
        question = str(row.get("question", "")).strip()
        raw_choices = row.get("choices")
        if not row_id or not question:
            raise ValueError(f"MMAU row {index} is missing id or question")
        if (
            not isinstance(raw_choices, (list, tuple))
            or not 2 <= len(raw_choices) <= 26
        ):
            raise ValueError(f"MMAU row {row_id} has invalid choices")
        choice_texts = tuple(str(choice).strip() for choice in raw_choices)
        if any(not choice for choice in choice_texts):
            raise ValueError(f"MMAU row {row_id} has an empty choice")
        choice_labels = tuple(string.ascii_uppercase[: len(choice_texts)])
        expected = _normalize_choice_answer(
            row.get("answer"),
            labels=choice_labels,
            texts=choice_texts,
        )
        if expected is None:
            raise ValueError(f"MMAU row {row_id} answer is not among its choices")
        audio = materialize_audio(row)
        prompt = _multiple_choice_prompt(question, choice_labels, choice_texts)
        cases.append(
            AudioEvaluationCase(
                case_id=f"mmau-{row_id}",
                track=EvaluationTrack.UNDERSTANDING,
                dataset_id=MMAU_DATASET_ID,
                dataset_revision=dataset_revision,
                dataset_config=MMAU_CONFIG,
                dataset_split=MMAU_SPLIT,
                source_row_id=row_id,
                source_row_hash=_source_row_hash(row, audio.sha256),
                license=license,
                category=category,
                prompt=prompt,
                expected_answer=expected,
                audio_path=audio.path,
                choices=choice_labels,
            )
        )
    return tuple(cases)


def build_esc50_cases(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_revision: str,
    materialize_audio: Callable[[Mapping[str, Any]], MaterializedAudio],
    foil_by_category: Mapping[str, str],
) -> tuple[AudioEvaluationCase, ...]:
    """Convert ESC-50 rows into alternating positive/hard-negative probes."""

    cases: list[AudioEvaluationCase] = []
    for index, row in enumerate(rows):
        filename = str(row.get("filename", row.get("file", ""))).strip()
        category = str(row.get("category", "")).strip().lower()
        if not filename or not category:
            raise ValueError(f"ESC-50 row {index} is missing filename or category")
        foil = str(foil_by_category.get(category, "")).strip().lower()
        if not foil or foil == category:
            raise ValueError(f"ESC-50 category {category!r} has no distinct hard foil")
        positive = index % 2 == 0
        queried_category = category if positive else foil
        expected = "YES" if positive else "NO"
        audio = materialize_audio(row)
        prompt = (
            f"Does this recording contain the sound of a {queried_category}? "
            "Return only YES or NO."
        )
        cases.append(
            AudioEvaluationCase(
                case_id=f"esc50-{_safe_id(filename)}-{'positive' if positive else 'foil'}",
                track=EvaluationTrack.UNDERSTANDING,
                dataset_id=ESC50_DATASET_ID,
                dataset_revision=dataset_revision,
                dataset_config=ESC50_CONFIG,
                dataset_split=ESC50_SPLIT,
                source_row_id=filename,
                source_row_hash=_source_row_hash(row, audio.sha256),
                license=ESC50_LICENSE,
                category=category,
                prompt=prompt,
                expected_answer=expected,
                audio_path=audio.path,
                choices=("YES", "NO"),
            )
        )
    return tuple(cases)


def build_caption_cases(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    dataset_revision: str,
    dataset_config: str,
    dataset_split: str,
    license: str,
    id_field: str,
    caption_field: str,
    category: str,
) -> tuple[AudioEvaluationCase, ...]:
    """Convert pinned caption rows into reference-audio-blind generation cases."""

    cases: list[AudioEvaluationCase] = []
    for index, row in enumerate(rows):
        row_id = str(row.get(id_field, "")).strip()
        caption = str(row.get(caption_field, "")).strip()
        if not row_id or not caption:
            raise ValueError(
                f"caption row {index} is missing {id_field!r} or {caption_field!r}"
            )
        cases.append(
            AudioEvaluationCase(
                case_id=f"{_safe_id(category)}-{_safe_id(row_id)}",
                track=EvaluationTrack.GENERATION,
                dataset_id=dataset_id,
                dataset_revision=dataset_revision,
                dataset_config=dataset_config,
                dataset_split=dataset_split,
                source_row_id=row_id,
                source_row_hash=_source_row_hash(row),
                license=license,
                category=category,
                prompt=caption,
                caption=caption,
            )
        )
    return attach_caption_hard_foils(tuple(cases))


def attach_caption_hard_foils(
    cases: tuple[AudioEvaluationCase, ...],
) -> tuple[AudioEvaluationCase, ...]:
    """Attach deterministic caption foils for generation semantic metrics.

    Foils are drawn from the same generated-audio manifest so CLAP-style
    retrieval can later ask whether the output is closer to the requested
    caption than to a plausible in-suite negative without leaking reference
    audio into the prompt.
    """

    if len(cases) < 2:
        return cases
    ranked = tuple(sorted(cases, key=lambda case: case.case_id))
    foil_by_id: dict[str, str] = {}
    for index, case in enumerate(ranked):
        if case.track is not EvaluationTrack.GENERATION or not case.caption:
            continue
        for offset in range(1, len(ranked)):
            candidate = ranked[(index + offset) % len(ranked)]
            if (
                candidate.track is EvaluationTrack.GENERATION
                and candidate.caption
                and candidate.caption != case.caption
            ):
                foil_by_id[case.case_id] = candidate.caption
                break
    if not foil_by_id:
        return cases
    return tuple(
        (
            replace(case, hard_foil_caption=foil_by_id[case.case_id])
            if case.case_id in foil_by_id
            else case
        )
        for case in cases
    )


def _multiple_choice_prompt(
    question: str,
    labels: tuple[str, ...],
    texts: tuple[str, ...],
) -> str:
    choices = "\n".join(
        f"{label}. {text}" for label, text in zip(labels, texts, strict=True)
    )
    return f"{question}\n{choices}\nReturn only the single choice letter."


def _normalize_choice_answer(
    raw_answer: Any,
    *,
    labels: tuple[str, ...],
    texts: tuple[str, ...],
) -> str | None:
    if isinstance(raw_answer, int) and 0 <= raw_answer < len(labels):
        return labels[raw_answer]
    answer = str(raw_answer).strip()
    upper = answer.upper()
    if upper in labels:
        return upper
    for label, text in zip(labels, texts, strict=True):
        if answer.casefold() == text.casefold():
            return label
    return None


def _source_row_hash(row: Mapping[str, Any], audio_sha256: str | None = None) -> str:
    metadata = {
        str(key): value
        for key, value in row.items()
        if str(key) != "audio" and _json_scalar_or_container(value)
    }
    if audio_sha256 is not None:
        metadata["audio_sha256"] = audio_sha256
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _json_scalar_or_container(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _safe_id(value: str) -> str:
    safe = "".join(
        character.lower() if character.isalnum() else "-" for character in value
    )
    return "-".join(part for part in safe.split("-") if part)
