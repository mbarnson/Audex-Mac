"""CLI for autonomous audio-capability evaluation preparation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .audio_evaluation import (
    AudioEvaluationCase,
    AudioEvaluationRun,
    EvaluationTrack,
    RunVerdict,
)
from .audio_evaluation_adapters import (
    AudexVllmTtaGenerationAdapter,
    AudexVllmUnderstandingAdapter,
)
from .audio_evaluation_ast import (
    AST_REPO_ID,
    AST_REVISION,
    build_ast_case_requests,
    write_ast_worker_request,
)
from .audio_evaluation_ast_labels import explicit_ast_label_maps
from .audio_evaluation_clap import (
    CLAP_REPO_ID,
    CLAP_REVISION,
    build_clap_case_requests,
    write_clap_worker_request,
)
from .audio_evaluation_datasets import MaterializedAudio
from .audio_evaluation_generation import TtaRecipe
from .audio_evaluation_hf import (
    DatasetPin,
    HfAudioMaterializer,
    HfDatasetClient,
    fetch_verified_rows,
)
from .audio_evaluation_openl3 import (
    default_full_openl3_requests,
    write_openl3_worker_request,
)
from .audio_evaluation_oracles import SignalSanityOracleSuite
from .audio_evaluation_runner import (
    AudioEvaluationRunner,
    OracleSuite,
    UnqualifiedOracleSuite,
)
from .audio_evaluation_suite import (
    AUDIOCAPS_AUDIO_PIN,
    AUDIOCAPS_CAPTION_PIN,
    ESC50_PIN,
    MMAU_PIN,
    SONG_DESCRIBER_PIN,
    build_full_cases_from_rows,
    build_smoke_cases_from_rows,
    build_standard_cases_from_rows,
)
from .audio_evaluation_xcodec import (
    XCodec1Config,
    XCodec1WavDecoder,
    resolve_xcodec1_config,
)
from .audio_runtime import preflight_audio_runtime
from .conversations import DEFAULT_DEMO_CONTEXT_TOKENS
from .models import (
    AUDEX_2B_REPO,
    AUDEX_30B_NVFP4_REPO,
    AUDEX_30B_REPO,
    SUPPORTED_MODELS,
)

DEFAULT_AUDIO_EVAL_ROOT = Path(".audex/runs/audio-capabilities")
DEFAULT_AUDIO_EVAL_CACHE = Path(".audex/cache/audio-eval")
DEFAULT_AUDIO_EVAL_SEED = 20260710


def main(
    argv: list[str] | None = None,
    *,
    fetch_rows: (
        Callable[
            ...,
            tuple[Mapping[str, Any], ...],
        ]
        | None
    ) = None,
    materialize_audio: Callable[[Mapping[str, Any]], MaterializedAudio] | None = None,
    runtime_factory: Callable[[Path | None, str], Any] | None = None,
    decoder_factory: (
        Callable[[XCodec1Config | None], Callable[[Any, Path, Any], None]] | None
    ) = None,
    oracle_suite_factory: Callable[[], OracleSuite] | None = None,
    model_path_resolver: Callable[[str, str], tuple[Path, str]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or run autonomous Audex audio-capability evaluation"
    )
    parser.add_argument("--tier", choices=("smoke", "standard", "full"), required=True)
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="build the pinned case manifest and selected audio cache without model inference",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_AUDIO_EVAL_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_AUDIO_EVAL_CACHE)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--cases-from-run",
        type=Path,
        default=None,
        help="reuse cases from a previous materialized audio-evaluation run",
    )
    parser.add_argument("--master-seed", type=int, default=DEFAULT_AUDIO_EVAL_SEED)
    parser.add_argument("--model", choices=("30b", "2b"), default="30b")
    parser.add_argument("--profile", choices=("bf16", "nvfp4"), default="bf16")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="override local Audex checkpoint folder for full evaluation execution",
    )
    parser.add_argument(
        "--xcodec1-path",
        type=Path,
        default=None,
        help="local hf-audio/xcodec-hubert-general-balanced snapshot for TTA decoding",
    )
    parser.add_argument(
        "--xcodec-device",
        default=None,
        help="XCodec torch device; defaults to auto, or set cpu/mps/cuda explicitly",
    )
    parser.add_argument(
        "--generation-oracles",
        choices=("signal", "unqualified"),
        default="signal",
        help=(
            "local generation oracle suite; signal is smoke-level waveform sanity "
            "only, not semantic caption alignment"
        ),
    )
    parser.add_argument(
        "--openl3-reference-stats-root",
        type=Path,
        default=None,
        help=(
            "directory containing pinned full-tier stable-audio-metrics OpenL3 "
            "reference .npz files; when set for standard/full materialization, "
            "writes generation/openl3-request.json"
        ),
    )
    parser.add_argument(
        "--capability-target",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help=(
            "numeric pass/fail target, repeatable; names must end in _min or _max "
            "and refer to summary metrics such as accuracy_min"
        ),
    )
    parser.add_argument(
        "--skip-esc50",
        action="store_true",
        help=(
            "explicitly omit ESC-50 cases when the pinned Hugging Face rows "
            "endpoint is unavailable; records the omission in the manifest"
        ),
    )
    parser.add_argument(
        "--skip-song-describer",
        action="store_true",
        help=(
            "explicitly omit optional SongDescriber cases when the pinned "
            "Hugging Face rows endpoint is unavailable; records the omission in "
            "the manifest"
        ),
    )
    args = parser.parse_args(argv)
    try:
        capability_targets = _parse_capability_targets(
            tuple(args.capability_target or ())
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.materialize_only and args.cases_from_run is not None:
        parser.error("--cases-from-run is only valid for execution runs")
    if args.materialize_only and capability_targets:
        parser.error("--capability-target is only valid for execution runs")
    if args.tier in {"standard", "full"} and not args.materialize_only:
        parser.error(
            f"{args.tier} execution is blocked until semantic generation oracles "
            "are implemented; use --materialize-only to prepare the manifest"
        )

    if args.model == "2b" and args.profile == "nvfp4":
        parser.error("--profile nvfp4 is only defined for --model 30b")

    model_path = args.model_path
    model_repo = _repo_for_eval_model(args.model, args.profile)
    if (
        not args.materialize_only
        and model_path is None
        and (runtime_factory is None or model_path_resolver is not None)
    ):
        active_model_path_resolver = model_path_resolver or _resolve_cached_model_path
        model_path, model_repo = active_model_path_resolver(args.model, args.profile)

    xcodec_config: XCodec1Config | None = None
    if not args.materialize_only and decoder_factory is None:
        xcodec_config = resolve_xcodec1_config(
            explicit_path=args.xcodec1_path,
            device=args.xcodec_device,
        )

    hf_token = os.environ.get("HF_TOKEN") or _dotenv_value("HF_TOKEN")
    client = HfDatasetClient(token=hf_token)

    def default_fetch(pin: DatasetPin) -> tuple[Mapping[str, Any], ...]:
        return fetch_verified_rows(pin, client=client)

    def active_fetch(pin: DatasetPin) -> tuple[Mapping[str, Any], ...]:
        print(
            "Audio evaluation: fetching " f"{pin.repo_id}/{pin.config}/{pin.split}...",
            flush=True,
        )
        if fetch_rows is None:
            rows = default_fetch(pin)
        else:
            rows = fetch_rows(pin, client=client)
        print(
            "Audio evaluation: fetched "
            f"{len(rows)} rows from {pin.repo_id}/{pin.split}.",
            flush=True,
        )
        return rows

    active_materializer = (
        materialize_audio
        or HfAudioMaterializer(
            client=client,
            cache_dir=args.cache_dir,
        ).materialize
    )

    if args.cases_from_run is not None:
        cases = _load_cases_from_run(args.cases_from_run)
    else:
        mmau_rows = active_fetch(MMAU_PIN)
        esc50_rows: tuple[Mapping[str, Any], ...] = ()
        if args.skip_esc50:
            print(
                "Audio evaluation: skipping ESC-50 by explicit --skip-esc50.",
                flush=True,
            )
        else:
            esc50_rows = active_fetch(ESC50_PIN)
        audiocaps_rows = active_fetch(AUDIOCAPS_CAPTION_PIN)
        song_rows: tuple[Mapping[str, Any], ...] = ()
        if args.skip_song_describer:
            print(
                "Audio evaluation: skipping SongDescriber by explicit "
                "--skip-song-describer.",
                flush=True,
            )
        else:
            song_rows = active_fetch(SONG_DESCRIBER_PIN)
        cases = _build_cases_from_rows(
            tier=args.tier,
            mmau_rows=tuple(mmau_rows),
            esc50_rows=tuple(esc50_rows),
            audiocaps_rows=tuple(audiocaps_rows),
            song_describer_rows=tuple(song_rows),
            master_seed=args.master_seed,
            materialize_audio=active_materializer,
        )
    run_id = args.run_id or time.strftime("audio-eval-%Y%m%d-%H%M%S")
    run = AudioEvaluationRun.create(
        root=args.run_root,
        run_id=run_id,
        tier=args.tier,
        master_seed=args.master_seed,
        cases=cases,
        manifest_metadata={
            "mode": "materialize_only" if args.materialize_only else "execute",
            "model": {
                "size": args.model,
                "profile": args.profile,
                "repo_id": model_repo,
                "path": str(model_path) if model_path is not None else None,
                "snapshot_revision": _hf_snapshot_revision(model_path),
                "file_hashes": _small_file_hashes(
                    model_path,
                    (
                        "config.json",
                        "generation_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "chat_template.jinja",
                        "audio_preprocessor/preprocessor_config.json",
                    ),
                ),
                "context": _model_context_payload(args.model, model_path),
            },
            "understanding_protocol": {
                "answer_space": "single constrained multiple-choice label",
                "scoring": "exact normalized label match; prose fails closed",
            },
            "generation_recipe": {
                "name": "audex_tta_cfg3_xcodec1",
                **asdict(TtaRecipe()),
            },
            "generation_oracles": args.generation_oracles,
            "openl3_reference_stats_root": (
                str(args.openl3_reference_stats_root)
                if args.openl3_reference_stats_root is not None
                else None
            ),
            "capability_targets": capability_targets,
            "oracle_registry": _oracle_registry_payload(),
            "omitted_datasets": _omitted_datasets(args),
            "source_cases_run": (
                str(args.cases_from_run) if args.cases_from_run is not None else None
            ),
            "datasets": [
                _pin_payload(pin)
                for pin in (
                    MMAU_PIN,
                    ESC50_PIN,
                    AUDIOCAPS_CAPTION_PIN,
                    AUDIOCAPS_AUDIO_PIN,
                    SONG_DESCRIBER_PIN,
                )
            ],
        },
        environment={
            "hf_token_present": bool(hf_token),
            **_environment_payload(
                model_path=model_path,
                model_repo=model_repo,
                xcodec_config=xcodec_config,
                generation_oracles=args.generation_oracles,
            ),
        },
    )
    _write_openl3_worker_request_if_configured(
        run,
        tier=args.tier,
        reference_stats_root=args.openl3_reference_stats_root,
    )
    if args.materialize_only:
        print(f"Audio evaluation materialized: {run.run_dir}")
        print(f"Cases: {len(cases)}")
        return 0

    active_runtime_factory = runtime_factory or _load_vllm_runtime
    active_oracle_suite_factory = oracle_suite_factory or _oracle_suite_factory(
        args.generation_oracles
    )
    runtime = active_runtime_factory(model_path, args.profile)
    if decoder_factory is None:
        assert xcodec_config is not None
        decoder = XCodec1WavDecoder(xcodec_config)
    else:
        decoder = decoder_factory(xcodec_config)
    summary = AudioEvaluationRunner(
        understanding=AudexVllmUnderstandingAdapter(runtime=runtime),
        generation=AudexVllmTtaGenerationAdapter(
            runtime=runtime,
            raw_dir=run.run_dir / "media" / "raw",
            enhanced_dir=run.run_dir / "media" / "enhanced",
            decode_to_wav=decoder,
        ),
        oracles=active_oracle_suite_factory(),
    ).run(
        run,
        master_seed=args.master_seed,
        capability_targets=capability_targets,
    )
    _write_completed_generation_worker_requests(run)
    print(f"Audio evaluation run: {run.run_dir}")
    print(f"Cases: {len(cases)}")
    print(f"Summary: {run.summary_path}")
    print(f"Verdict: {summary.verdict.value}")
    for failure in summary.protocol_failures:
        print(f"Protocol failure: {failure}")
    return 0 if summary.verdict in {RunVerdict.CHARACTERIZED, RunVerdict.PASS} else 2


def _pin_payload(pin: DatasetPin) -> dict[str, Any]:
    return {
        "repo_id": pin.repo_id,
        "revision": pin.revision,
        "config": pin.config,
        "split": pin.split,
        "license": pin.license,
        "expected_rows": pin.expected_rows,
    }


def _parse_capability_targets(raw_targets: tuple[str, ...]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for raw_target in raw_targets:
        name, separator, raw_value = str(raw_target).partition("=")
        name = name.strip()
        raw_value = raw_value.strip()
        if not separator or not name or not raw_value:
            raise ValueError(
                "capability targets must be formatted as NAME=VALUE, "
                f"got {raw_target!r}"
            )
        if not name.endswith(("_min", "_max")):
            raise ValueError(f"capability target {name!r} must end in _min or _max")
        if name in targets:
            raise ValueError(f"duplicate capability target: {name}")
        try:
            targets[name] = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"capability target {name!r} must have a numeric value"
            ) from exc
    return targets


def _oracle_registry_payload() -> dict[str, Any]:
    return {
        "signal": {
            "status": "implemented",
            "authority": "smoke_structural_signal_only",
            "semantic_caption_alignment": False,
        },
        "clap": {
            "status": "implemented_worker_boundary_unqualified",
            "repo_id": CLAP_REPO_ID,
            "revision": CLAP_REVISION,
            "purpose": "caption_alignment_and_retrieval_diagnostics",
            "qualification_gate": {
                "calibration": "fixed ESC-50 hard-negative calibration",
                "min_4way_hard_negative_top1": 0.70,
                "min_matched_over_foil": 0.85,
            },
        },
        "ast": {
            "status": "implemented_worker_boundary_unqualified",
            "repo_id": AST_REPO_ID,
            "revision": AST_REVISION,
            "purpose": "audioset_event_sanity_diagnostics",
            "qualification_gate": {
                "calibration": "known pinned calibration split",
                "logit_policy": "sigmoid over raw logits; AudioSet is multi-label",
                "device_policy": "explicit device; no silent CPU fallback",
            },
        },
        "openl3_fd": {
            "status": "implemented_external_worker_unqualified",
            "implementation": "pinned_stable_audio_metrics_openl3_fd",
            "worker_python": "3.11",
            "purpose": "full_tier_paper_style_fd_openl3",
            "qualification_gate": {
                "identical_sets": "near_zero",
                "permutation_invariance": True,
                "unrelated_corpora": "materially_worse",
                "fixed_vectors_reproduce_within_tolerance": True,
            },
        },
    }


def _build_cases_from_rows(
    *,
    tier: str,
    mmau_rows: tuple[Mapping[str, Any], ...],
    esc50_rows: tuple[Mapping[str, Any], ...],
    audiocaps_rows: tuple[Mapping[str, Any], ...],
    song_describer_rows: tuple[Mapping[str, Any], ...],
    master_seed: int,
    materialize_audio: Callable[[Mapping[str, Any]], MaterializedAudio],
) -> tuple[AudioEvaluationCase, ...]:
    if tier == "smoke":
        return build_smoke_cases_from_rows(
            mmau_rows=mmau_rows,
            esc50_rows=esc50_rows,
            audiocaps_rows=audiocaps_rows,
            song_describer_rows=song_describer_rows,
            master_seed=master_seed,
            materialize_audio=materialize_audio,
        )
    if tier == "standard":
        return build_standard_cases_from_rows(
            mmau_rows=mmau_rows,
            esc50_rows=esc50_rows,
            audiocaps_rows=audiocaps_rows,
            song_describer_rows=song_describer_rows,
            master_seed=master_seed,
            materialize_audio=materialize_audio,
        )
    if tier == "full":
        return build_full_cases_from_rows(
            mmau_rows=mmau_rows,
            esc50_rows=esc50_rows,
            audiocaps_rows=audiocaps_rows,
            song_describer_rows=song_describer_rows,
            materialize_audio=materialize_audio,
        )
    raise ValueError(f"unsupported audio evaluation tier: {tier}")


def _write_openl3_worker_request_if_configured(
    run: AudioEvaluationRun,
    *,
    tier: str,
    reference_stats_root: Path | None,
) -> None:
    if tier not in {"standard", "full"} or reference_stats_root is None:
        return
    write_openl3_worker_request(
        run.run_dir / "generation" / "openl3-request.json",
        run_id=run.run_dir.name,
        requests=default_full_openl3_requests(
            run.run_dir,
            reference_stats_root=reference_stats_root,
        ),
    )


def _write_completed_generation_worker_requests(run: AudioEvaluationRun) -> None:
    generated_wavs = _generated_wav_by_case_id(run.run_dir)
    if not generated_wavs:
        return
    cases = tuple(
        case
        for case in run.cases
        if case.track is EvaluationTrack.GENERATION and case.case_id in generated_wavs
    )
    if not cases:
        return
    clap_requests = build_clap_case_requests(
        cases,
        generated_wav_by_case_id=generated_wavs,
    )
    if clap_requests:
        write_clap_worker_request(
            run.run_dir / "generation" / "clap-request.json",
            run_id=run.run_dir.name,
            requests=clap_requests,
        )
    expected_labels, forbidden_labels = explicit_ast_label_maps(cases)
    labeled_cases = tuple(case for case in cases if case.case_id in expected_labels)
    if labeled_cases:
        write_ast_worker_request(
            run.run_dir / "generation" / "ast-request.json",
            run_id=run.run_dir.name,
            requests=build_ast_case_requests(
                labeled_cases,
                generated_wav_by_case_id=generated_wavs,
                expected_labels_by_case_id=expected_labels,
                forbidden_labels_by_case_id=forbidden_labels,
            ),
        )


def _generated_wav_by_case_id(run_dir: Path) -> dict[str, str]:
    path = run_dir / "generation" / "outputs.jsonl"
    generated_wavs: dict[str, str] = {}
    if not path.is_file():
        return generated_wavs
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        case_id = str(payload.get("case_id", "")).strip()
        enhanced_wav_path = str(payload.get("enhanced_wav_path") or "").strip()
        raw_wav_path = str(payload.get("raw_wav_path") or "").strip()
        wav_path = enhanced_wav_path or raw_wav_path
        if case_id and wav_path:
            generated_wavs[case_id] = wav_path
    return generated_wavs


def _environment_payload(
    *,
    model_path: Path | None,
    model_repo: str,
    xcodec_config: XCodec1Config | None,
    generation_oracles: str,
) -> dict[str, Any]:
    return {
        "audex_eval": {
            "model_repo": model_repo,
            "model_path_exists": (
                model_path.exists() if model_path is not None else None
            ),
            "xcodec1": (
                {
                    "repo_id": xcodec_config.repo_id,
                    "path": str(xcodec_config.path),
                    "path_exists": xcodec_config.path.exists(),
                    "snapshot_revision": _hf_snapshot_revision(xcodec_config.path),
                    "file_hashes": _small_file_hashes(
                        xcodec_config.path,
                        ("config.json",),
                    ),
                    "device": xcodec_config.device,
                }
                if xcodec_config is not None
                else None
            ),
            "generation_oracles": generation_oracles,
        },
        "git": _git_payload(),
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "dependencies": _dependency_versions(
            (
                "mlx",
                "mlx-lm",
                "numpy",
                "scipy",
                "soundfile",
                "torch",
                "transformers",
                "vllm",
            )
        ),
    }


def _model_context_payload(model_size: str, model_path: Path | None) -> dict[str, Any]:
    checkpoint_limit = _checkpoint_declared_context(model_path)
    effective_limit = (
        min(DEFAULT_DEMO_CONTEXT_TOKENS, checkpoint_limit)
        if checkpoint_limit is not None
        else None
    )
    return {
        "model_card_max_tokens": _model_card_max_context(model_size),
        "configured_demo_max_tokens": DEFAULT_DEMO_CONTEXT_TOKENS,
        "checkpoint_max_position_embeddings": checkpoint_limit,
        "effective_engine_max_model_len": effective_limit,
    }


def _model_card_max_context(model_size: str) -> int | None:
    if model_size == "30b":
        return 1_000_000
    if model_size == "2b":
        return 128_000
    return None


def _small_file_hashes(
    root: Path | None,
    relative_paths: tuple[str, ...],
    *,
    max_bytes: int = 50_000_000,
) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}
    hashes: dict[str, dict[str, Any]] = {}
    for relative_path in relative_paths:
        path = root / relative_path
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue
        if stat.st_size > max_bytes:
            hashes[relative_path] = {
                "bytes": stat.st_size,
                "sha256": None,
                "skipped": "file_too_large",
            }
            continue
        hashes[relative_path] = {
            "bytes": stat.st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return hashes


def _checkpoint_declared_context(model_path: Path | None) -> int | None:
    if model_path is None:
        return None
    config_path = model_path / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("max_position_embeddings")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _git_payload() -> dict[str, Any]:
    head = _git_text("rev-parse", "HEAD")
    branch = _git_text("rev-parse", "--abbrev-ref", "HEAD")
    status = _git_bytes("status", "--short", "--untracked-files=no")
    diff = _git_bytes("diff", "--binary", "HEAD") or b""
    dirty = bool(status.strip()) if status is not None else False
    return {
        "available": head is not None,
        "commit": head,
        "branch": branch,
        "dirty": dirty,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest() if dirty else None,
    }


def _git_text(*args: str) -> str | None:
    output = _git_bytes(*args)
    if output is None:
        return None
    return output.decode("utf-8", errors="replace").strip() or None


def _git_bytes(*args: str) -> bytes | None:
    try:
        result = subprocess.run(
            ("git", *args),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=Path.cwd(),
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _dependency_versions(packages: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _hf_snapshot_revision(path: Path | None) -> str | None:
    if path is None:
        return None
    parts = path.expanduser().parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots":
            return parts[index + 1]
    return None


def _load_cases_from_run(run_dir: Path) -> tuple[AudioEvaluationCase, ...]:
    cases: list[AudioEvaluationCase] = []
    for track in EvaluationTrack:
        path = run_dir / track.value / "cases.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"case manifest not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["track"] = EvaluationTrack(payload["track"])
            payload["choices"] = tuple(payload.get("choices", ()))
            cases.append(AudioEvaluationCase(**payload))
    if not cases:
        raise ValueError(f"no cases found in {run_dir}")
    return tuple(cases)


def _dotenv_value(name: str, path: Path = Path(".env")) -> str | None:
    if not path.is_file():
        return None
    prefix = f"{name}="
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        value = line.split("=", 1)[1].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value or None
    return None


def _omitted_datasets(args: argparse.Namespace) -> list[dict[str, str]]:
    omitted: list[dict[str, str]] = []
    if args.skip_esc50:
        omitted.append(
            {
                "repo_id": ESC50_PIN.repo_id,
                "reason": "explicit --skip-esc50",
            }
        )
    if args.skip_song_describer:
        omitted.append(
            {
                "repo_id": SONG_DESCRIBER_PIN.repo_id,
                "reason": "explicit --skip-song-describer",
            }
        )
    return omitted


def _load_vllm_runtime(model_path: Path | None, profile: str) -> Any:
    del profile
    if model_path is None:
        raise ValueError("model_path is required for the default vLLM runtime")
    from .vllm_runtime import AudexAsyncVllmRuntime

    return AudexAsyncVllmRuntime.from_model_path(model_path)


def _oracle_suite_factory(name: str) -> Callable[[], OracleSuite]:
    if name == "signal":
        return SignalSanityOracleSuite
    if name == "unqualified":
        return UnqualifiedOracleSuite
    raise ValueError(f"unknown generation oracle suite: {name}")


def _repo_for_eval_model(model: str, profile: str) -> str:
    if model == "2b":
        return AUDEX_2B_REPO
    if model == "30b" and profile == "nvfp4":
        return AUDEX_30B_NVFP4_REPO
    if model == "30b":
        return AUDEX_30B_REPO
    raise ValueError(
        f"unsupported eval model selection: model={model} profile={profile}"
    )


def _resolve_cached_model_path(model: str, profile: str) -> tuple[Path, str]:
    repo_id = _repo_for_eval_model(model, profile)
    selected = next(item for item in SUPPORTED_MODELS if item.repo_id == repo_id)
    preflight = preflight_audio_runtime(selected)
    if preflight.ready and preflight.model_path is not None:
        return preflight.model_path, repo_id
    missing = ", ".join(preflight.missing_items) or "unknown missing model files"
    raise RuntimeError(
        f"Audex evaluation requires a complete cached speech checkpoint for "
        f"{repo_id}; missing: {missing}. Pass --model-path to override."
    )
