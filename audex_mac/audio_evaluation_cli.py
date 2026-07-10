"""CLI for autonomous audio-capability evaluation preparation."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .audio_evaluation import AudioEvaluationRun, RunVerdict
from .audio_evaluation_adapters import (
    AudexVllmTtaGenerationAdapter,
    AudexVllmUnderstandingAdapter,
)
from .audio_evaluation_datasets import MaterializedAudio
from .audio_evaluation_hf import (
    DatasetPin,
    HfAudioMaterializer,
    HfDatasetClient,
    fetch_verified_rows,
)
from .audio_evaluation_oracles import SignalSanityOracleSuite
from .audio_evaluation_runner import (
    AudioEvaluationRunner,
    OracleSuite,
    UnqualifiedOracleSuite,
)
from .audio_evaluation_suite import (
    AUDIOCAPS_CAPTION_PIN,
    ESC50_PIN,
    MMAU_PIN,
    SONG_DESCRIBER_PIN,
    build_smoke_cases_from_rows,
)
from .audio_evaluation_xcodec import (
    XCodec1Config,
    XCodec1WavDecoder,
    resolve_xcodec1_config,
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
) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or run autonomous Audex audio-capability evaluation"
    )
    parser.add_argument("--tier", choices=("smoke",), required=True)
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="build the pinned case manifest and selected audio cache without model inference",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_AUDIO_EVAL_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_AUDIO_EVAL_CACHE)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_AUDIO_EVAL_SEED)
    parser.add_argument("--model", choices=("30b", "2b"), default="30b")
    parser.add_argument("--profile", choices=("bf16", "nvfp4"), default="bf16")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="local Audex checkpoint folder for full evaluation execution",
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
    args = parser.parse_args(argv)

    if (
        not args.materialize_only
        and args.model_path is None
        and runtime_factory is None
    ):
        parser.error(
            "full evaluation execution requires --model-path until eval-specific "
            "model selection is wired"
        )

    xcodec_config: XCodec1Config | None = None
    if not args.materialize_only and decoder_factory is None:
        xcodec_config = resolve_xcodec1_config(
            explicit_path=args.xcodec1_path,
            device=args.xcodec_device,
        )

    hf_token = os.environ.get("HF_TOKEN")
    client = HfDatasetClient(token=hf_token)

    def default_fetch(pin: DatasetPin) -> tuple[Mapping[str, Any], ...]:
        return fetch_verified_rows(pin, client=client)

    def active_fetch(pin: DatasetPin) -> tuple[Mapping[str, Any], ...]:
        if fetch_rows is None:
            return default_fetch(pin)
        return fetch_rows(pin, client=client)

    active_materializer = (
        materialize_audio
        or HfAudioMaterializer(
            client=client,
            cache_dir=args.cache_dir,
        ).materialize
    )

    mmau_rows = active_fetch(MMAU_PIN)
    esc50_rows = active_fetch(ESC50_PIN)
    audiocaps_rows = active_fetch(AUDIOCAPS_CAPTION_PIN)
    song_rows = active_fetch(SONG_DESCRIBER_PIN)
    cases = build_smoke_cases_from_rows(
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
                "path": str(args.model_path) if args.model_path is not None else None,
            },
            "generation_oracles": args.generation_oracles,
            "datasets": [
                _pin_payload(pin)
                for pin in (
                    MMAU_PIN,
                    ESC50_PIN,
                    AUDIOCAPS_CAPTION_PIN,
                    SONG_DESCRIBER_PIN,
                )
            ],
        },
        environment={
            "hf_token_present": bool(hf_token),
        },
    )
    if args.materialize_only:
        print(f"Audio evaluation materialized: {run.run_dir}")
        print(f"Cases: {len(cases)}")
        return 0

    active_runtime_factory = runtime_factory or _load_vllm_runtime
    active_oracle_suite_factory = oracle_suite_factory or _oracle_suite_factory(
        args.generation_oracles
    )
    runtime = active_runtime_factory(args.model_path, args.profile)
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
            decode_to_wav=decoder,
        ),
        oracles=active_oracle_suite_factory(),
    ).run(run, master_seed=args.master_seed)
    print(f"Audio evaluation run: {run.run_dir}")
    print(f"Cases: {len(cases)}")
    print(f"Summary: {run.summary_path}")
    print(f"Verdict: {summary.verdict.value}")
    for failure in summary.protocol_failures:
        print(f"Protocol failure: {failure}")
    return 0 if summary.verdict is RunVerdict.CHARACTERIZED else 2


def _pin_payload(pin: DatasetPin) -> dict[str, Any]:
    return {
        "repo_id": pin.repo_id,
        "revision": pin.revision,
        "config": pin.config,
        "split": pin.split,
        "license": pin.license,
        "expected_rows": pin.expected_rows,
    }


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
