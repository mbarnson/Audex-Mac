"""One-shot NVIDIA-reference TTA rendering and blind quantization packaging."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

from .audio_evaluation_adapters import build_nvidia_tta_generation_adapter
from .audio_evaluation_enhancement import (
    NvidiaEnhancementVae,
    enhancement_vae_artifact_identity,
)
from .audio_evaluation_generation import configure_nvidia_tta_engine_environment
from .audio_evaluation_xcodec import (
    build_nvidia_tta_wav_decoder,
    xcodec1_artifact_identity,
)
from .audio_model_resolver import (
    audio_model_repo,
    load_audio_vllm_runtime,
    resolve_cached_audio_model,
)
from .sound_lab.cli import (
    _resolve_or_download_enhancement_vae,
    _resolve_or_download_xcodec1,
)
from .tta_quality import (
    create_blind_quant_listening_set,
    load_tta_quality_corpus,
    render_tta_quality_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audex TTA quantization quality gate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render")
    render.add_argument("--profile", choices=("bf16", "nvfp4"), required=True)
    render.add_argument("--corpus", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--model-path", type=Path, default=None)
    render.add_argument("--xcodec1-path", type=Path, default=None)
    render.add_argument("--enhancement-vae-path", type=Path, default=None)
    render.add_argument(
        "--device", choices=("auto", "mps", "cpu", "cuda"), default="auto"
    )
    package = subparsers.add_parser("package")
    package.add_argument("manifest", type=Path, nargs=2)
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--key-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "package":
        listening = create_blind_quant_listening_set(
            manifest_paths=tuple(args.manifest),
            output_dir=args.output_dir,
            key_path=args.key_out,
        )
        print(f"Blind listening sheet: {listening.listening_path}")
        print(f"Blind sample count: {len(listening.sample_paths)}")
        print(f"Private decoding key: {listening.key_path}")
        return 0
    return _render(args)


def _render(args: argparse.Namespace) -> int:
    configure_nvidia_tta_engine_environment(os.environ)
    model_repo = audio_model_repo("30b", args.profile)
    model_path = args.model_path
    if model_path is None:
        model_path, model_repo = resolve_cached_audio_model("30b", args.profile)
    xcodec = _resolve_or_download_xcodec1(args.xcodec1_path, device=args.device)
    enhancement = _resolve_or_download_enhancement_vae(
        args.enhancement_vae_path,
        model="30b",
        device=args.device,
    )
    print(f"Audex TTA quality: loading {model_repo}...", flush=True)
    runtime = load_audio_vllm_runtime(model_path, args.profile)
    generation = build_nvidia_tta_generation_adapter(
        runtime=runtime,
        raw_dir=args.output_dir / "raw",
        enhanced_dir=args.output_dir / "enhanced",
        decode_to_wav=build_nvidia_tta_wav_decoder(xcodec),
        enhance_wav=NvidiaEnhancementVae(enhancement),
    )
    model_revision, model_file_hashes = _model_provenance(model_path)
    manifest = render_tta_quality_manifest(
        corpus=load_tta_quality_corpus(args.corpus),
        generation=generation,
        profile=args.profile,
        model_repo=model_repo,
        model_revision=model_revision,
        model_file_hashes=model_file_hashes,
        output_dir=args.output_dir,
        xcodec_identity=xcodec1_artifact_identity(xcodec.path),
        enhancement_identity=enhancement_vae_artifact_identity(enhancement.root),
    )
    print(f"TTA quant manifest: {manifest}")
    return 0


def _model_provenance(model_path: Path) -> tuple[str, dict[str, str]]:
    resolved = Path(model_path).resolve()
    parts = resolved.parts
    try:
        snapshot_index = parts.index("snapshots")
        revision = parts[snapshot_index + 1]
    except (ValueError, IndexError):
        revision = ""
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        candidates = tuple(
            path
            for path in resolved.rglob("*")
            if path.is_file()
            and (path.suffix == ".json" or path.name.endswith(".index.json"))
        )
        return revision, _sha256_files(candidates, root=resolved)
    artifacts = tuple(path for path in resolved.rglob("*") if path.is_file())
    if not artifacts:
        raise ValueError(f"local model path contains no hashable artifacts: {resolved}")
    hashes = _sha256_files(artifacts, root=resolved)
    aggregate = hashlib.sha256()
    for name, digest in sorted(hashes.items()):
        aggregate.update(name.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
    return f"local-sha256-{aggregate.hexdigest()}", hashes


def _sha256_files(paths: tuple[Path, ...], *, root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path.relative_to(root))] = digest.hexdigest()
    return hashes
