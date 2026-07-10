from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from audex_mac.model_select import select_model
from audex_mac.models import AUDEX_30B_NVFP4_REPO, AUDEX_30B_REPO, SUPPORTED_MODELS
from audex_mac.nvfp4_conversion import (
    RECIPE_ID,
    _model_card,
    _strip_model_card_metadata,
    local_revision,
    quantize_routed_expert,
    recipe_manifest,
)
from audex_mac.text_runtime import TextRuntimePreflight

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    "path",
    [
        "backbone.layers.1.mixer.switch_mlp.fc1",
        "backbone.layers.51.mixer.switch_mlp.fc2",
        "model.backbone.layers.8.mixer.switch_mlp.fc1",
    ],
)
def test_nvfp4_recipe_quantizes_only_fused_routed_experts(path: str) -> None:
    assert quantize_routed_expert(path, object()) is True


@pytest.mark.parametrize(
    "path",
    [
        "backbone.layers.1.mixer.gate",
        "backbone.layers.1.mixer.shared_experts.up_proj",
        "backbone.layers.4.mixer.in_proj",
        "backbone.layers.5.mixer.o_proj",
        "backbone.embeddings",
        "lm_head",
        "audio_encoder.layers.0.fc1",
    ],
)
def test_nvfp4_recipe_preserves_non_expert_modules(path: str) -> None:
    assert quantize_routed_expert(path, object()) is False


def test_nvfp4_recipe_has_stable_hugging_face_identity() -> None:
    revision = local_revision("source-revision")
    manifest = recipe_manifest("source-revision")

    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert revision == local_revision("source-revision")
    assert revision != local_revision("other-revision")
    assert manifest["recipe_id"] == RECIPE_ID
    assert manifest["base_model"] == AUDEX_30B_REPO
    assert manifest["base_revision"] == "source-revision"
    assert manifest["output_repo"] == AUDEX_30B_NVFP4_REPO
    assert manifest["oracle_calibration"] is None


def test_nvfp4_snapshot_is_preferred_over_bf16_when_both_are_cached() -> None:
    class Cached30BProbe:
        def is_cached(self, model, readiness: str = "speech") -> bool:
            return model.repo_id in {AUDEX_30B_NVFP4_REPO, AUDEX_30B_REPO}

    selection = select_model(Cached30BProbe())

    assert selection.selected.repo_id == AUDEX_30B_NVFP4_REPO
    assert selection.cached is True


def test_nvfp4_text_runtime_uses_full_multimodal_checkpoint() -> None:
    model = next(
        model for model in SUPPORTED_MODELS if model.repo_id == AUDEX_30B_NVFP4_REPO
    )
    preflight = TextRuntimePreflight(
        model=model,
        benchmark=SimpleNamespace(),
        snapshot_check=SimpleNamespace(snapshot_path=Path("/snapshot")),
        dependency_checks=(),
    )

    assert preflight.model_path == Path("/snapshot/checkpoint_folder_full")


def test_nvfp4_publication_card_is_hf_native_and_preserves_upstream_card() -> None:
    upstream = """---
library_name: transformers
license: other
---

# Upstream Audex card

[License](license/NVIDIA-OneWay-Noncommercial-License.docx)
"""

    card = _model_card("source-revision", upstream)

    assert "base_model: nvidia/Nemotron-Labs-Audex-30B-A3B" in card
    assert "base_model_relation: quantized" in card
    assert "library_name: mlx" in card
    assert "pipeline_tag: text-generation" in card
    assert "# Audex-30B-A3B NVFP4 — Quality-First MLX Quant" in card
    assert "preserve Audex's reasoning, routing, ASR, TTS" in card
    assert "[The Audex-Mac demonstration repository]" in card
    assert "one persistent vLLM Metal\nengine backed by MLX" in card
    assert "Audex's own audio encoder, language backbone" in card
    assert "M3,\nM4, M5, or newer Mac with at least 48 GB of RAM" in card
    assert "# Original NVIDIA model card" in card
    assert "# Upstream Audex card" in card
    assert "library_name: transformers" not in card
    assert "## Why this quant is different" not in card
    assert "## Near-real-time voice-to-voice on Apple Silicon" not in card
    assert "## Estimated Apple Silicon memory requirements" not in card
    assert "## Reproduce this conversion" not in card
    assert "## Quality and limitations" not in card
    assert (
        "https://huggingface.co/nvidia/Nemotron-Labs-Audex-30B-A3B/"
        "blob/source-revision/license/NVIDIA-OneWay-Noncommercial-License.docx" in card
    )


def test_strip_model_card_metadata_keeps_only_upstream_body() -> None:
    upstream = "---\ntags:\n  - audio\n---\n\n# Original\n"

    assert _strip_model_card_metadata(upstream) == "# Original\n"
