from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.models import DEFAULT_MODEL
from audex_mac.text_benchmark import TextBenchmark
from audex_mac.text_runtime import preflight_text_runtime

pytestmark = pytest.mark.fast


def benchmark() -> TextBenchmark:
    return TextBenchmark(
        name="test benchmark",
        system="system",
        generation={
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 100,
            "max_tokens": 4096,
            "thinking_enabled": False,
        },
        turns=[{"user": "q", "expected_shape": "a"} for _ in range(10)],
        pass_criteria=["coherent"],
        sampler_reference="test",
    )


def make_snapshot(cache_root: Path) -> Path:
    repo_dir = cache_root / "models--nvidia--Nemotron-Labs-Audex-2B"
    snapshot = repo_dir / "snapshots" / "rev"
    checkpoint = snapshot / "checkpoint_folder_textonly"
    script = snapshot / "inference_scripts_vllm" / "textonly_scripts"
    checkpoint.mkdir(parents=True)
    script.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("rev", encoding="utf-8")
    for rel in DEFAULT_MODEL.text_required_files:
        path = snapshot / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    return snapshot


def test_text_preflight_reports_missing_checkpoint_shards(tmp_path: Path) -> None:
    make_snapshot(tmp_path)

    result = preflight_text_runtime(
        DEFAULT_MODEL,
        benchmark(),
        cache_root=tmp_path,
        apply_patches=False,
    )

    assert result.ready is False
    assert "checkpoint_folder_textonly/model.safetensors" in result.missing_items
