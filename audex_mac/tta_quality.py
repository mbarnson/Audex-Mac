"""Controlled, blind BF16-versus-NVFP4 text-to-audio quality contracts."""

from __future__ import annotations

import json
import random
import re
import secrets
import shutil
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_evaluation import AudioEvaluationCase, EvaluationTrack, derive_case_seed

NVIDIA_TTA_REFERENCE_RECIPE = "nvidia-tta-cfg3-topk80-temp1-xcodec1-vae-v1"
NVIDIA_CFG_PAIRS_PER_BATCH = 2


def configure_nvidia_tta_environment(env: MutableMapping[str, str]) -> None:
    env.update(
        {
            "AUDEX_VLLM_TTS_CFG": "1",
            "AUDEX_VLLM_ENABLE_CFG_WIRING": "1",
            "AUDEX_VLLM_CFG_MAX_MODEL_LEN": "8192",
            "AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS": "4",
        }
    )


@dataclass(frozen=True, slots=True)
class TtaQualityCase:
    case_id: str
    caption: str


@dataclass(frozen=True, slots=True)
class TtaQualityCorpus:
    master_seed: int
    cases: tuple[TtaQualityCase, ...]


@dataclass(frozen=True, slots=True)
class BlindQuantListeningSet:
    listening_path: Path
    key_path: Path
    sample_paths: tuple[Path, ...]


def load_tta_quality_corpus(path: Path) -> TtaQualityCorpus:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("unsupported TTA quality corpus")
    master_seed = payload.get("master_seed")
    items = payload.get("cases")
    if not isinstance(master_seed, int) or not isinstance(items, list) or not items:
        raise ValueError("TTA quality corpus requires master_seed and cases")
    cases: list[TtaQualityCase] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("TTA quality case must be an object")
        case_id = str(item.get("case_id", "")).strip()
        caption = " ".join(str(item.get("caption", "")).split())
        if not case_id or not _literal_caption(caption):
            raise ValueError(f"invalid literal TTA quality case: {case_id!r}")
        cases.append(TtaQualityCase(case_id=case_id, caption=caption))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("TTA quality case ids must be unique")
    if len({case.caption.casefold() for case in cases}) != len(cases):
        raise ValueError("TTA quality captions must be unique")
    return TtaQualityCorpus(master_seed=master_seed, cases=tuple(cases))


def render_tta_quality_manifest(
    *,
    corpus: TtaQualityCorpus,
    generation: Any,
    profile: str,
    model_repo: str,
    model_revision: str,
    model_file_hashes: dict[str, str],
    output_dir: Path,
    xcodec_identity: str,
    enhancement_identity: str,
) -> Path:
    if profile not in {"bf16", "nvfp4"}:
        raise ValueError(f"unsupported TTA quant profile: {profile}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    seeded_cases = tuple(
        (
            AudioEvaluationCase(
                case_id=case.case_id,
                track=EvaluationTrack.GENERATION,
                dataset_id="audex-tta-quant-listening",
                dataset_revision="v1",
                dataset_config="nvidia-reference",
                dataset_split="listening",
                source_row_id=case.case_id,
                source_row_hash=case.case_id,
                license="local-evaluation-manifest",
                category="quantization-quality",
                prompt=case.caption,
                caption=case.caption,
            ),
            derive_case_seed(corpus.master_seed, case.case_id),
        )
        for case in corpus.cases
    )
    for start in range(0, len(seeded_cases), NVIDIA_CFG_PAIRS_PER_BATCH):
        wave = seeded_cases[start : start + NVIDIA_CFG_PAIRS_PER_BATCH]
        attempts = generation.generate_many(wave)
        for (case, seed), attempt in zip(wave, attempts, strict=True):
            if not attempt.structure.nvidia_reference_decodable:
                raise RuntimeError(
                    f"TTA quality case {case.case_id} failed structure: "
                    f"{attempt.structure.failures}"
                )
            if (
                attempt.enhanced_wav_path is None
                or not attempt.enhanced_wav_path.is_file()
            ):
                raise RuntimeError(
                    f"TTA quality case lacks NVIDIA-enhanced WAV: {case.case_id}"
                )
            samples.append(
                {
                    "case_id": case.case_id,
                    "caption": case.caption,
                    "seed": seed,
                    "raw_wav_path": str(attempt.raw_wav_path.resolve()),
                    "wav_path": str(attempt.enhanced_wav_path.resolve()),
                    "frame_count": attempt.structure.frame_count,
                    "codec_duration_seconds": attempt.structure.duration_seconds,
                    "duration_seconds": 10.0,
                    "elapsed_seconds": attempt.elapsed_seconds,
                    "finish_reason": attempt.finish_reason,
                }
            )
    manifest_path = output_dir / f"tta-quant-{profile}.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "profile": profile,
                "model_size": "30b",
                "model_repo": model_repo,
                "model_revision": model_revision,
                "model_file_hashes": model_file_hashes,
                "engine_max_model_len": 8192,
                "recipe": NVIDIA_TTA_REFERENCE_RECIPE,
                "master_seed": corpus.master_seed,
                "xcodec_identity": xcodec_identity,
                "enhancement_identity": enhancement_identity,
                "enhancement_seed": 0,
                "samples": samples,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def create_blind_quant_listening_set(
    *,
    manifest_paths: tuple[Path, Path],
    output_dir: Path,
    key_path: Path,
) -> BlindQuantListeningSet:
    manifests = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in manifest_paths
    ]
    profiles = {manifest.get("profile") for manifest in manifests}
    if profiles != {"bf16", "nvfp4"}:
        raise ValueError("quant listening requires one BF16 and one NVFP4 manifest")
    _validate_matched_manifests(manifests)
    output_dir = Path(output_dir)
    key_path = Path(key_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"blind listener directory must be empty: {output_dir}")
    try:
        key_path.resolve().relative_to(output_dir.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("blind decoding key must be outside listener directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    by_profile = {str(item["profile"]): item for item in manifests}
    case_ids = [str(item["case_id"]) for item in manifests[0]["samples"]]
    sample_rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        for profile in ("bf16", "nvfp4"):
            sample = next(
                item
                for item in by_profile[profile]["samples"]
                if item["case_id"] == case_id
            )
            sample_rows.append({"profile": profile, **sample})
    packaging_nonce = secrets.token_hex(32)
    rng = random.Random(int(packaging_nonce, 16))
    rng.shuffle(sample_rows)

    copied: list[Path] = []
    key_samples: list[dict[str, Any]] = []
    names_by_case: dict[str, list[str]] = {case_id: [] for case_id in case_ids}
    for index, row in enumerate(sample_rows, start=1):
        name = f"sample-{index:02d}.wav"
        destination = output_dir / name
        shutil.copyfile(Path(str(row["wav_path"])), destination)
        copied.append(destination)
        names_by_case[str(row["case_id"])].append(name)
        key_samples.append({"sample": name, **row})

    lines = [
        "# Blind Audex TTA Quantization Listening Set",
        "",
        "Judge prompt alignment, event structure, artifacts, and overall realism.",
        "Record a winner for every group before opening the private key.",
        "",
    ]
    first_manifest = manifests[0]
    for group_index, case_id in enumerate(case_ids, start=1):
        sample = next(
            item for item in first_manifest["samples"] if item["case_id"] == case_id
        )
        names = sorted(names_by_case[case_id])
        lines.extend(
            [
                f"## G{group_index}: {sample['caption']}",
                "",
                f"- `{names[0]}`",
                f"- `{names[1]}`",
                "- Winner: ",
                "- Notes: ",
                "",
            ]
        )
    listening_path = output_dir / "LISTENING.md"
    listening_path.write_text("\n".join(lines), encoding="utf-8")
    key_path.write_text(
        json.dumps(
            {
                "version": 1,
                "master_seed": first_manifest["master_seed"],
                "packaging_nonce": packaging_nonce,
                "recipes": [
                    {
                        field: manifest[field]
                        for field in (
                            "profile",
                            "model_repo",
                            "model_revision",
                            "model_file_hashes",
                            "engine_max_model_len",
                            "recipe",
                            "xcodec_identity",
                            "enhancement_identity",
                            "enhancement_seed",
                        )
                    }
                    for manifest in manifests
                ],
                "samples": key_samples,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return BlindQuantListeningSet(
        listening_path=listening_path,
        key_path=key_path,
        sample_paths=tuple(copied),
    )


def _validate_matched_manifests(manifests: list[dict[str, Any]]) -> None:
    first, second = manifests
    for field in (
        "version",
        "model_size",
        "engine_max_model_len",
        "recipe",
        "master_seed",
        "xcodec_identity",
        "enhancement_identity",
        "enhancement_seed",
    ):
        if first.get(field) != second.get(field):
            raise ValueError(f"quant manifests differ in {field}")
    left = [
        (item.get("case_id"), item.get("caption"), item.get("seed"))
        for item in first.get("samples", [])
    ]
    right = [
        (item.get("case_id"), item.get("caption"), item.get("seed"))
        for item in second.get("samples", [])
    ]
    if not left or left != right:
        raise ValueError("quant manifests do not contain identical cases and seeds")
    for manifest in manifests:
        for sample in manifest["samples"]:
            path = Path(str(sample.get("wav_path", "")))
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"quant sample WAV is missing: {path}")


def _literal_caption(caption: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", caption)
    if not 3 <= len(words) <= 24:
        return False
    lowered = caption.casefold()
    return not any(
        phrase in lowered
        for phrase in ("create ", "generate ", "sound effect", "cinematic")
    )
