from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from audex_mac.tts_quality import (
    TTS_QUALITY_RECIPES,
    create_blind_listening_set,
    load_tts_quality_corpus,
    tts_quality_recipe,
)

pytestmark = pytest.mark.fast


def load_manifest_evaluator_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = scripts_dir / "evaluate_tts_quality_manifest.py"
    sys.path.insert(0, str(scripts_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "evaluate_tts_quality_manifest",
            script_path,
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_quality_recipes_are_named_enforced_and_seeded() -> None:
    plain = tts_quality_recipe("plain-reference", seed=17)
    matched_cfg = tts_quality_recipe("nvidia-tts-cfg", seed=17)
    cfg3 = tts_quality_recipe("audex-cfg3", seed=17)

    assert (
        plain.sampling.temperature,
        plain.sampling.top_k,
        plain.sampling.cfg_scale,
        plain.sampling.seed,
    ) == (0.8, 0, 1.0, 17)
    assert (
        matched_cfg.sampling.temperature,
        matched_cfg.sampling.top_k,
        matched_cfg.sampling.cfg_scale,
        matched_cfg.sampling.seed,
    ) == (0.8, 0, 2.0, 17)
    assert (
        cfg3.sampling.temperature,
        cfg3.sampling.top_k,
        cfg3.sampling.cfg_scale,
        cfg3.sampling.seed,
    ) == (1.0, 80, 3.0, 17)
    assert all(recipe.compact_window_decode for recipe in (plain, matched_cfg, cfg3))

    with pytest.raises(ValueError, match="unknown TTS quality recipe"):
        tts_quality_recipe("mislabeled", seed=17)


def test_quality_gate_requires_every_deliberate_term() -> None:
    evaluator = load_manifest_evaluator_module()

    passed = evaluator.quality_sample_passed(
        word_errors={"word_error_rate": 0.01},
        repetition={"excessive": False},
        required_terms={"missing": ["Worcestershire"]},
        terminated_cleanly=True,
        max_word_error_rate=0.45,
    )

    assert passed is False


def test_quality_corpus_requires_long_unique_blind_listening_passages(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "conversational",
                        "category": "baseline",
                        "text": " ".join(f"word{index}" for index in range(45)),
                        "required_terms": ["word17", "word31"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    corpus = load_tts_quality_corpus(path)

    assert corpus.version == 1
    assert corpus.cases[0].case_id == "conversational"
    assert corpus.cases[0].word_count == 45
    assert corpus.cases[0].required_terms == ("word17", "word31")


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        (
            [
                {
                    "id": "short",
                    "category": "baseline",
                    "text": "This is much too short to judge prosody.",
                }
            ],
            "at least 40 words",
        ),
        (
            [
                {
                    "id": "duplicate",
                    "category": "baseline",
                    "text": " ".join("first" for _ in range(40)),
                },
                {
                    "id": "duplicate",
                    "category": "technical",
                    "text": " ".join("second" for _ in range(40)),
                },
            ],
            "duplicate case id",
        ),
    ],
)
def test_quality_corpus_rejects_unusable_cases(
    tmp_path: Path,
    cases: list[dict[str, object]],
    message: str,
) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps({"version": 1, "cases": cases}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_tts_quality_corpus(path)


def _quality_manifest_paths(tmp_path: Path) -> list[Path]:
    manifest_paths: list[Path] = []
    for recipe_id, recipe in TTS_QUALITY_RECIPES.items():
        samples = []
        for case_number in range(1, 7):
            case_id = f"case_{case_number}"
            wav_path = tmp_path / f"raw-{recipe_id}-{case_id}.wav"
            wav_path.write_bytes(f"{recipe_id}:{case_id}".encode())
            run_log_path = wav_path.with_suffix(".json")
            run_log_path.write_text(
                json.dumps(
                    {
                        "tts_observed_segments": 1,
                        "tts_cfg_enabled": recipe.sampling.cfg_enabled,
                        "reached_end_token": True,
                        "hit_max_tokens": False,
                    }
                ),
                encoding="utf-8",
            )
            samples.append(
                {
                    "case_id": case_id,
                    "category": "test",
                    "text": f"Long controlled text for {case_id}",
                    "word_count": 50,
                    "required_terms": [f"term {case_number}"],
                    "wav_path": str(wav_path),
                    "run_log_path": str(run_log_path),
                }
            )
        manifest_path = tmp_path / f"{recipe_id}.manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "recipe_id": recipe_id,
                    "cfg_enabled": recipe.sampling.cfg_enabled,
                    "seed": 20260709,
                    "sampling": {
                        "temperature": recipe.sampling.temperature,
                        "top_p": recipe.sampling.top_p,
                        "top_k": recipe.sampling.top_k,
                        "cfg_scale": recipe.sampling.cfg_scale,
                    },
                    "controlled_segments_per_case": 1,
                    "compact_window_decode": True,
                    "compact_window_decode_required": True,
                    "samples": samples,
                }
            ),
            encoding="utf-8",
        )
        manifest_paths.append(manifest_path)
    return manifest_paths


def test_blind_listening_set_hides_recipe_and_case_names(
    tmp_path: Path,
) -> None:
    manifest_paths = _quality_manifest_paths(tmp_path)

    output_dir = tmp_path / "blind"
    key_path = tmp_path / "private" / "key.json"
    listening_set = create_blind_listening_set(
        manifest_paths=tuple(manifest_paths),
        output_dir=output_dir,
        key_path=key_path,
        random_seed=8675309,
    )

    assert len(listening_set.sample_paths) == 18
    assert {path.name for path in listening_set.sample_paths} == {
        f"sample-{index:02d}.wav" for index in range(1, 19)
    }
    sheet = listening_set.listening_path.read_text(encoding="utf-8")
    assert all(recipe_id not in sheet for recipe_id in TTS_QUALITY_RECIPES)
    assert all(f"case_{case_number}" not in sheet for case_number in range(1, 7))
    assert key_path.parent != output_dir
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert len(key["samples"]) == 18
    assert {item["recipe_id"] for item in key["samples"]} == set(TTS_QUALITY_RECIPES)


def test_blind_listening_set_rejects_a_confounded_matrix(tmp_path: Path) -> None:
    manifest_paths = _quality_manifest_paths(tmp_path)
    manifest = json.loads(manifest_paths[1].read_text(encoding="utf-8"))
    manifest["samples"][0]["text"] = "Different segment boundaries and content"
    manifest_paths[1].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="identical case content"):
        create_blind_listening_set(
            manifest_paths=tuple(manifest_paths),
            output_dir=tmp_path / "blind",
            key_path=tmp_path / "private" / "key.json",
            random_seed=8675309,
        )
