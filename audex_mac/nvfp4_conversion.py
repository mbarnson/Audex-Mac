"""Reproducible MLX NVFP4 conversion for Audex-30B-A3B.

The first quality trial deliberately quantizes only the routed MoE expert
matrices. Every attention, Mamba, router, embedding, output, audio, and speech
decoder weight remains in its source precision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .checkpoints import HF_CACHE_ROOT, local_snapshot_path, repo_cache_dir
from .models import AUDEX_30B_NVFP4_REPO, AUDEX_30B_REPO

RECIPE_ID = "audex-30b-a3b-nvfp4-experts-v1"
SOURCE_CHECKPOINT_DIR = "checkpoint_folder_full"
EXPECTED_EXPERT_PROJECTIONS = 46
AUDEX_MAC_URL = "https://github.com/mbarnson/Audex-Mac"
_EXPERT_MODULE = re.compile(
    r"^(?:model\.)?backbone\.layers\.\d+\.mixer\.switch_mlp\.fc[12]$"
)


def quantize_routed_expert(path: str, _module: Any) -> bool:
    """Return true only for fused routed-expert projections."""

    return _EXPERT_MODULE.fullmatch(path) is not None


def local_revision(source_revision: str) -> str:
    """Build a deterministic, Hugging Face-shaped local snapshot revision."""

    identity = f"{AUDEX_30B_NVFP4_REPO}\n{source_revision}\n{RECIPE_ID}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:40]


def recipe_manifest(source_revision: str) -> dict[str, Any]:
    return {
        "recipe_id": RECIPE_ID,
        "base_model": AUDEX_30B_REPO,
        "base_revision": source_revision,
        "output_repo": AUDEX_30B_NVFP4_REPO,
        "weight_format": {
            "mode": "nvfp4",
            "bits": 4,
            "group_size": 16,
            "activation_quantization": False,
            "mlx_layout": True,
        },
        "module_policy": {
            "nvfp4": [
                "backbone.layers.*.mixer.switch_mlp.fc1",
                "backbone.layers.*.mixer.switch_mlp.fc2",
            ],
            "source_precision": [
                "audio_encoder.*",
                "audio_projector.*",
                "audex_causal_speech_decoder/*",
                "backbone.embeddings",
                "backbone.layers.*.mixer.gate",
                "backbone.layers.*.mixer.shared_experts.*",
                "backbone.layers.*.mixer.in_proj",
                "backbone.layers.*.mixer.out_proj",
                "backbone.layers.*.mixer.{q,k,v,o}_proj",
                "backbone.layers.*.mixer.{A_log,D,dt_bias,conv1d,norm}*",
                "backbone.norm_f",
                "lm_head",
            ],
        },
        "oracle_calibration": None,
        "notes": [
            "Conservative first trial: only routed experts are quantized.",
            "The MLX runtime fuses all 128 experts within each MoE projection.",
            "No oQe claim is made because current oQe weighting is affine-only.",
        ],
    }


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_size))


def _safetensors_metadata(path: Path) -> tuple[int, int]:
    parameters = 0
    data_bytes = 0
    for name, value in _read_safetensors_header(path).items():
        if name == "__metadata__":
            continue
        shape = value["shape"]
        count = 1
        for dimension in shape:
            count *= int(dimension)
        parameters += count
        offsets = value["data_offsets"]
        data_bytes += int(offsets[1]) - int(offsets[0])
    return parameters, data_bytes


def _hardlink_or_copy(source: str | Path, destination: str | Path) -> str:
    source_path = Path(source).resolve()
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)
    return str(destination_path)


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=_hardlink_or_copy)


def _merge_audio_weights(source_checkpoint: Path, output_checkpoint: Path) -> None:
    audio_name = "model-audio.safetensors"
    source_audio = source_checkpoint / audio_name
    _hardlink_or_copy(source_audio, output_checkpoint / audio_name)

    source_index_path = source_checkpoint / "model.safetensors.index.json"
    output_index_path = output_checkpoint / "model.safetensors.index.json"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    output_index = json.loads(output_index_path.read_text(encoding="utf-8"))
    audio_weights = {
        key: shard
        for key, shard in source_index["weight_map"].items()
        if key.startswith(("audio_encoder.", "audio_projector."))
    }
    if not audio_weights or set(audio_weights.values()) != {audio_name}:
        raise ValueError("Unexpected Audex audio shard layout in source checkpoint")

    output_index["weight_map"].update(audio_weights)
    output_index["weight_map"] = dict(sorted(output_index["weight_map"].items()))
    parameters, data_bytes = _safetensors_metadata(source_audio)
    metadata = output_index.setdefault("metadata", {})
    metadata["total_parameters"] = int(metadata.get("total_parameters", 0)) + parameters
    metadata["total_size"] = int(metadata.get("total_size", 0)) + data_bytes
    output_index_path.write_text(
        json.dumps(output_index, indent=4) + "\n",
        encoding="utf-8",
    )


def _validate_quantized_checkpoint(checkpoint: Path) -> None:
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization", {})
    expected = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    if any(quantization.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected quantization config: {quantization!r}")

    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    scale_bases = sorted(
        key.removesuffix(".scales")
        for key in index["weight_map"]
        if key.endswith(".scales")
    )
    invalid = [base for base in scale_bases if not quantize_routed_expert(base, None)]
    if invalid:
        raise ValueError(f"Non-expert modules were quantized: {invalid[:5]!r}")
    if len(scale_bases) != EXPECTED_EXPERT_PROJECTIONS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_EXPERT_PROJECTIONS} quantized expert projections, "
            f"found {len(scale_bases)}"
        )
    if not any(key.startswith("audio_encoder.") for key in index["weight_map"]):
        raise ValueError("Quantized checkpoint is missing BF16 audio encoder weights")


def _strip_model_card_metadata(model_card: str) -> str:
    """Remove one leading Hugging Face YAML block from an upstream card."""

    normalized = model_card.lstrip("\ufeff")
    if not normalized.startswith("---\n"):
        return normalized.lstrip("\n")
    marker = normalized.find("\n---\n", 4)
    if marker < 0:
        raise ValueError("Upstream model card has an unterminated metadata block")
    return normalized[marker + len("\n---\n") :].lstrip("\n")


def _pin_upstream_card_links(model_card: str, source_revision: str) -> str:
    base = f"https://huggingface.co/{AUDEX_30B_REPO}/blob/{source_revision}"
    return model_card.replace(
        "(license/NVIDIA-OneWay-Noncommercial-License.docx)",
        f"({base}/license/NVIDIA-OneWay-Noncommercial-License.docx)",
    )


def _model_card(source_revision: str, upstream_model_card: str) -> str:
    upstream_body = _pin_upstream_card_links(
        _strip_model_card_metadata(upstream_model_card),
        source_revision,
    )
    return f"""---
base_model: {AUDEX_30B_REPO}
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
license: other
license_name: nvidia-oneway-noncommercial-license
license_link: https://huggingface.co/{AUDEX_30B_REPO}/blob/{source_revision}/license/NVIDIA-OneWay-Noncommercial-License.docx
language:
  - en
tags:
  - mlx
  - nvfp4
  - 4-bit
  - quantized
  - mixture-of-experts
  - apple-silicon
  - custom_code
  - audio
  - audio-language-modeling
  - audio-understanding
  - text-to-speech
  - text-to-audio
  - speech-recognition
  - speech-translation
  - long-context
---

# Audex-30B-A3B NVFP4 — Quality-First MLX Quant

An MLX-native, selective mixed-precision conversion of
[`{AUDEX_30B_REPO}`](https://huggingface.co/{AUDEX_30B_REPO}), designed to
preserve Audex's reasoning, routing, ASR, TTS, and general-audio capabilities.
[The Audex-Mac demonstration repository]({AUDEX_MAC_URL}) provides
typed-to-voice and voice-to-voice conversation using one persistent vLLM Metal
engine backed by MLX. It uses Audex's own audio encoder, language backbone,
speech-token generation, and NVIDIA causal speech decoder. We recommend an M3,
M4, M5, or newer Mac with at least 48 GB of RAM for this model.

---

# Original NVIDIA model card

> Preserved from the pinned upstream revision `{source_revision}`.

{upstream_body}
"""


def _compose_snapshot(
    source_snapshot: Path,
    output_snapshot: Path,
    source_revision: str,
) -> None:
    source_checkpoint = source_snapshot / SOURCE_CHECKPOINT_DIR
    output_checkpoint = output_snapshot / SOURCE_CHECKPOINT_DIR

    from audex_mac.patches.runtime import apply_audex_runtime_patches

    apply_audex_runtime_patches()
    from mlx_lm.convert import convert

    convert(
        hf_path=str(source_checkpoint),
        mlx_path=str(output_checkpoint),
        quantize=True,
        q_group_size=16,
        q_bits=4,
        q_mode="nvfp4",
        dtype="bfloat16",
        quant_predicate=quantize_routed_expert,
        trust_remote_code=True,
    )

    _merge_audio_weights(source_checkpoint, output_checkpoint)
    audio_preprocessor = source_checkpoint / "audio_preprocessor"
    if audio_preprocessor.is_dir():
        _copy_tree(audio_preprocessor, output_checkpoint / "audio_preprocessor")

    for dirname in (
        "audex_causal_speech_decoder",
        "nv-whisper",
        "inference_scripts_vllm",
    ):
        _copy_tree(source_snapshot / dirname, output_snapshot / dirname)
    assets = source_snapshot / "assets"
    if assets.is_dir():
        _copy_tree(assets, output_snapshot / "assets")
    for filename in ("LICENSE", ".gitattributes"):
        source_file = source_snapshot / filename
        if source_file.is_file():
            _hardlink_or_copy(source_file, output_snapshot / filename)
    upstream_model_card = source_snapshot / "MODELCARD.md"
    if upstream_model_card.is_file():
        _hardlink_or_copy(
            upstream_model_card,
            output_snapshot / "UPSTREAM_MODELCARD.md",
        )

    upstream_readme = (source_snapshot / "README.md").read_text(encoding="utf-8")
    (output_snapshot / "README.md").write_text(
        _model_card(source_revision, upstream_readme), encoding="utf-8"
    )
    (output_snapshot / "quantization_recipe.json").write_text(
        json.dumps(recipe_manifest(source_revision), indent=2) + "\n",
        encoding="utf-8",
    )
    _validate_quantized_checkpoint(output_checkpoint)


def convert_to_cache(
    *,
    cache_root: Path = HF_CACHE_ROOT,
    source_snapshot: Path | None = None,
    replace: bool = False,
) -> Path:
    source_snapshot = source_snapshot or local_snapshot_path(
        AUDEX_30B_REPO, cache_root=cache_root
    )
    if source_snapshot is None:
        raise FileNotFoundError(
            f"No cached snapshot found for {AUDEX_30B_REPO}; run "
            "./start.sh --model audex-30b-a3b --yes-download "
            "--preflight-audio-runtime first"
        )
    source_snapshot = source_snapshot.resolve()
    source_revision = source_snapshot.name
    revision = local_revision(source_revision)
    output_root = repo_cache_dir(AUDEX_30B_NVFP4_REPO, cache_root)
    output_snapshot = output_root / "snapshots" / revision
    work_snapshot = output_root / "snapshots" / f".{revision}.incomplete"

    if output_snapshot.exists():
        if not replace:
            raise FileExistsError(
                f"Output snapshot already exists: {output_snapshot}. Use --replace to rebuild."
            )
        shutil.rmtree(output_snapshot)
    if work_snapshot.exists():
        shutil.rmtree(work_snapshot)
    work_snapshot.mkdir(parents=True)

    try:
        _compose_snapshot(source_snapshot, work_snapshot, source_revision)
        work_snapshot.rename(output_snapshot)
    except BaseException:
        shutil.rmtree(work_snapshot, ignore_errors=True)
        raise

    refs = output_root / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    ref_tmp = refs / ".main.tmp"
    ref_tmp.write_text(revision, encoding="utf-8")
    ref_tmp.replace(refs / "main")
    return output_snapshot


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the quality-first Audex-30B NVFP4 MLX snapshot"
    )
    parser.add_argument("--cache-root", type=Path, default=HF_CACHE_ROOT)
    parser.add_argument("--source-snapshot", type=Path, default=None)
    parser.add_argument(
        "--replace", action="store_true", help="replace this recipe's existing snapshot"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = convert_to_cache(
        cache_root=args.cache_root,
        source_snapshot=args.source_snapshot,
        replace=args.replace,
    )
    print(f"NVFP4 snapshot ready: {output}")
    print(f"./start.sh will now prefer {AUDEX_30B_NVFP4_REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
