"""CLI for autonomous audio-capability evaluation preparation."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .audio_evaluation import AudioEvaluationRun
from .audio_evaluation_datasets import MaterializedAudio
from .audio_evaluation_hf import (
    DatasetPin,
    HfAudioMaterializer,
    HfDatasetClient,
    fetch_verified_rows,
)
from .audio_evaluation_suite import (
    AUDIOCAPS_CAPTION_PIN,
    ESC50_PIN,
    MMAU_PIN,
    SONG_DESCRIBER_PIN,
    build_smoke_cases_from_rows,
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
    args = parser.parse_args(argv)

    if not args.materialize_only:
        parser.error(
            "full evaluation execution is not wired yet; use --materialize-only "
            "until the native XCodec decoder and local oracle qualification path exist"
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
            "mode": "materialize_only",
            "model": {
                "size": args.model,
                "profile": args.profile,
            },
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
    print(f"Audio evaluation materialized: {run.run_dir}")
    print(f"Cases: {len(cases)}")
    return 0


def _pin_payload(pin: DatasetPin) -> dict[str, Any]:
    return {
        "repo_id": pin.repo_id,
        "revision": pin.revision,
        "config": pin.config,
        "split": pin.split,
        "license": pin.license,
        "expected_rows": pin.expected_rows,
    }
