"""Contracts for repeatable, blind TTS quality experiments."""

from __future__ import annotations

import json
import os
import random
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vllm_sts_requests import VllmTtsSamplingConfig

MIN_LISTENING_PASSAGE_WORDS = 40
QUALITY_MATRIX_CASE_COUNT = 6
_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True, slots=True)
class TtsQualityRecipe:
    recipe_id: str
    sampling: VllmTtsSamplingConfig
    compact_window_decode: bool = True


TTS_QUALITY_RECIPES = {
    recipe.recipe_id: recipe
    for recipe in (
        TtsQualityRecipe(
            recipe_id="plain-reference",
            sampling=VllmTtsSamplingConfig(
                temperature=0.8,
                top_p=1.0,
                top_k=0,
                cfg_scale=1.0,
            ),
        ),
        TtsQualityRecipe(
            recipe_id="nvidia-tts-cfg",
            sampling=VllmTtsSamplingConfig(
                temperature=0.8,
                top_p=1.0,
                top_k=0,
                cfg_scale=2.0,
            ),
        ),
        TtsQualityRecipe(
            recipe_id="audex-cfg3",
            sampling=VllmTtsSamplingConfig(
                temperature=1.0,
                top_p=1.0,
                top_k=80,
                cfg_scale=3.0,
            ),
        ),
    )
}


def tts_quality_recipe(recipe_id: str, *, seed: int) -> TtsQualityRecipe:
    try:
        recipe = TTS_QUALITY_RECIPES[recipe_id]
    except KeyError as exc:
        raise ValueError(f"unknown TTS quality recipe: {recipe_id}") from exc
    return TtsQualityRecipe(
        recipe_id=recipe.recipe_id,
        sampling=VllmTtsSamplingConfig(
            temperature=recipe.sampling.temperature,
            top_p=recipe.sampling.top_p,
            top_k=recipe.sampling.top_k,
            cfg_scale=recipe.sampling.cfg_scale,
            seed=seed,
            require_compact_window_decode=recipe.compact_window_decode,
        ),
        compact_window_decode=recipe.compact_window_decode,
    )


@dataclass(frozen=True, slots=True)
class TtsQualityCase:
    case_id: str
    category: str
    text: str
    required_terms: tuple[str, ...]

    @property
    def word_count(self) -> int:
        return len(_WORD_RE.findall(self.text))


@dataclass(frozen=True, slots=True)
class TtsQualityCorpus:
    version: int
    cases: tuple[TtsQualityCase, ...]


@dataclass(frozen=True, slots=True)
class BlindListeningSet:
    output_dir: Path
    listening_path: Path
    sample_paths: tuple[Path, ...]
    key_path: Path


def load_tts_quality_corpus(path: Path) -> TtsQualityCorpus:
    """Load and validate long-form passages used by the quality matrix."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("quality corpus must be a JSON object")
    version = int(payload.get("version", 0))
    if version != 1:
        raise ValueError(f"unsupported quality corpus version: {version}")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("quality corpus must contain a non-empty cases list")

    cases: list[TtsQualityCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"quality case {index} must be a JSON object")
        case = _parse_case(raw_case, index=index)
        if case.case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case.case_id}")
        seen_ids.add(case.case_id)
        cases.append(case)
    return TtsQualityCorpus(version=version, cases=tuple(cases))


def create_blind_listening_set(
    *,
    manifest_paths: tuple[Path, ...],
    output_dir: Path,
    key_path: Path,
    random_seed: int,
) -> BlindListeningSet:
    """Copy a recipe matrix into an opaque listener set and private key."""

    expected_recipe_ids = set(TTS_QUALITY_RECIPES)
    if len(manifest_paths) != len(expected_recipe_ids):
        raise ValueError(
            "blind listening requires exactly one manifest for every quality recipe"
        )
    output_resolved = output_dir.resolve()
    if key_path.resolve().is_relative_to(output_resolved):
        raise ValueError("blind decoding key must be outside the listener directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"blind listener directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    recipes: dict[str, dict[str, dict[str, Any]]] = {}
    case_order: list[str] | None = None
    case_contract: dict[str, tuple[object, ...]] | None = None
    shared_seed: int | None = None
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        recipe_id = str(payload.get("recipe_id", "")).strip()
        if recipe_id not in expected_recipe_ids or recipe_id in recipes:
            raise ValueError(f"invalid or duplicate recipe id: {recipe_id!r}")
        _validate_manifest_recipe(payload, recipe_id=recipe_id)
        manifest_seed = payload.get("seed")
        if not isinstance(manifest_seed, int):
            raise ValueError(f"recipe {recipe_id} must record an integer seed")
        if shared_seed is None:
            shared_seed = manifest_seed
        elif manifest_seed != shared_seed:
            raise ValueError("recipe manifests must use the same seed")
        raw_samples = payload.get("samples")
        if (
            not isinstance(raw_samples, list)
            or len(raw_samples) != QUALITY_MATRIX_CASE_COUNT
        ):
            raise ValueError(
                f"recipe {recipe_id} must contain exactly "
                f"{QUALITY_MATRIX_CASE_COUNT} samples"
            )
        samples_by_case: dict[str, dict[str, Any]] = {}
        current_contract: dict[str, tuple[object, ...]] = {}
        for raw_sample in raw_samples:
            if not isinstance(raw_sample, dict):
                raise ValueError(f"recipe {recipe_id} has an invalid sample")
            case_id = str(raw_sample.get("case_id", "")).strip()
            wav_path = Path(str(raw_sample.get("wav_path", "")))
            if not case_id or case_id in samples_by_case:
                raise ValueError(f"recipe {recipe_id} has invalid case id {case_id!r}")
            if not wav_path.is_file():
                raise ValueError(f"recipe {recipe_id} WAV does not exist: {wav_path}")
            _validate_sample_run_log(raw_sample, recipe_id=recipe_id)
            samples_by_case[case_id] = dict(raw_sample)
            current_contract[case_id] = (
                raw_sample.get("category"),
                raw_sample.get("text"),
                tuple(raw_sample.get("required_terms", ())),
                raw_sample.get("word_count"),
            )
        current_order = list(samples_by_case)
        if case_order is None:
            case_order = current_order
            case_contract = current_contract
        elif set(current_order) != set(case_order):
            raise ValueError("recipe manifests do not contain identical case ids")
        elif current_contract != case_contract:
            raise ValueError("recipe manifests do not contain identical case content")
        recipes[recipe_id] = samples_by_case

    if set(recipes) != expected_recipe_ids:
        raise ValueError("blind listening matrix is missing a quality recipe")

    assert case_order is not None
    rng = random.Random(random_seed)
    records = [
        (case_id, recipe_id, recipes[recipe_id][case_id])
        for case_id in case_order
        for recipe_id in recipes
    ]
    rng.shuffle(records)
    copied_by_case: dict[str, list[Path]] = {case_id: [] for case_id in case_order}
    key_samples: list[dict[str, object]] = []
    sample_paths: list[Path] = []
    for sample_number, (case_id, recipe_id, raw_sample) in enumerate(records, start=1):
        sample_path = output_dir / f"sample-{sample_number:02d}.wav"
        source_path = Path(str(raw_sample["wav_path"]))
        shutil.copyfile(source_path, sample_path)
        copied_by_case[case_id].append(sample_path)
        sample_paths.append(sample_path)
        key_samples.append(
            {
                "sample": sample_path.name,
                "case_id": case_id,
                "category": raw_sample.get("category"),
                "recipe_id": recipe_id,
                "source_wav_path": str(source_path),
                "source_run_log_path": raw_sample.get("run_log_path"),
            }
        )

    group_case_ids = list(case_order)
    rng.shuffle(group_case_ids)
    group_key: list[dict[str, object]] = []
    sheet_lines = [
        "# Blind TTS Listening Set",
        "",
        "For each group, listen to all three files and record a winner plus any notes.",
        "Replay freely; the order and filenames are randomized.",
        "",
    ]
    for group_number, case_id in enumerate(group_case_ids, start=1):
        group_paths = list(copied_by_case[case_id])
        rng.shuffle(group_paths)
        sheet_lines.extend(
            [
                f"## Group {group_number}",
                "",
                *[
                    f"- `afplay {shlex.quote(str(path.resolve()))}`"
                    for path in group_paths
                ],
                "",
                "Winner: __________",
                "",
                "Notes: ________________________________________________",
                "",
            ]
        )
        group_key.append(
            {
                "group": group_number,
                "case_id": case_id,
                "samples": [path.name for path in group_paths],
            }
        )
    listening_path = output_dir / "LISTENING.md"
    listening_path.write_text("\n".join(sheet_lines), encoding="utf-8")

    shared_mtime = 946684800
    for sample_path in sample_paths:
        sample_path.touch()
        sample_path.chmod(0o644)
        os.utime(sample_path, (shared_mtime, shared_mtime))

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "random_seed": random_seed,
                "groups": group_key,
                "samples": key_samples,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return BlindListeningSet(
        output_dir=output_dir,
        listening_path=listening_path,
        sample_paths=tuple(sorted(sample_paths)),
        key_path=key_path,
    )


def _validate_manifest_recipe(payload: dict[str, Any], *, recipe_id: str) -> None:
    recipe = TTS_QUALITY_RECIPES[recipe_id]
    expected_sampling = {
        "temperature": recipe.sampling.temperature,
        "top_p": recipe.sampling.top_p,
        "top_k": recipe.sampling.top_k,
        "cfg_scale": recipe.sampling.cfg_scale,
    }
    if payload.get("sampling") != expected_sampling:
        raise ValueError(f"recipe {recipe_id} does not match its canonical sampler")
    if payload.get("cfg_enabled") is not recipe.sampling.cfg_enabled:
        raise ValueError(f"recipe {recipe_id} has inconsistent CFG metadata")
    if payload.get("controlled_segments_per_case") != 1:
        raise ValueError(
            f"recipe {recipe_id} is not controlled to one segment per case"
        )
    if payload.get("compact_window_decode") is not True:
        raise ValueError(f"recipe {recipe_id} did not request compact-window decode")
    if payload.get("compact_window_decode_required") is not True:
        raise ValueError(f"recipe {recipe_id} did not require compact-window decode")


def _validate_sample_run_log(
    raw_sample: dict[str, Any],
    *,
    recipe_id: str,
) -> None:
    run_log_path = Path(str(raw_sample.get("run_log_path", "")))
    if not run_log_path.is_file():
        raise ValueError(f"recipe {recipe_id} run log does not exist: {run_log_path}")
    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    if run_log.get("tts_observed_segments") != 1:
        raise ValueError(f"recipe {recipe_id} sample did not use exactly one segment")
    expected_cfg = TTS_QUALITY_RECIPES[recipe_id].sampling.cfg_enabled
    if run_log.get("tts_cfg_enabled") is not expected_cfg:
        raise ValueError(f"recipe {recipe_id} sample has inconsistent CFG execution")
    if run_log.get("reached_end_token") is not True or run_log.get("hit_max_tokens"):
        raise ValueError(f"recipe {recipe_id} sample did not terminate cleanly")


def _parse_case(raw_case: dict[str, Any], *, index: int) -> TtsQualityCase:
    case_id = str(raw_case.get("id", "")).strip()
    if not _CASE_ID_RE.fullmatch(case_id):
        raise ValueError(f"quality case {index} has invalid id: {case_id!r}")
    category = str(raw_case.get("category", "")).strip()
    if not category:
        raise ValueError(f"quality case {case_id} has no category")
    text = str(raw_case.get("text", "")).strip()
    required_terms_raw = raw_case.get("required_terms", [])
    if not isinstance(required_terms_raw, list) or not all(
        isinstance(term, str) and term.strip() for term in required_terms_raw
    ):
        raise ValueError(f"quality case {case_id} has invalid required_terms")
    case = TtsQualityCase(
        case_id=case_id,
        category=category,
        text=text,
        required_terms=tuple(term.strip() for term in required_terms_raw),
    )
    if case.word_count < MIN_LISTENING_PASSAGE_WORDS:
        raise ValueError(
            f"quality case {case_id} must contain at least "
            f"{MIN_LISTENING_PASSAGE_WORDS} words; got {case.word_count}"
        )
    return case
