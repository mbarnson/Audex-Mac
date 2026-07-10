#!/usr/bin/env python3
"""Benchmark Audex vLLM Metal speech-token sampling hot paths."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from audex_mac.patches import vllm_metal_cfg  # noqa: E402

DEFAULT_HIDDEN_SIZE = 2688
DEFAULT_VOCAB_SIZE = 205312
DEFAULT_CODEC_MIN_ID = 131077
DEFAULT_CODEC_MAX_ID = 196612
DEFAULT_SPEECHGEN_END_ID = 131076


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Microbenchmark Audex TTS sampling on MLX/Metal without loading "
            "the full model."
        )
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--cfg-pairs",
        type=int,
        default=0,
        help=(
            "benchmark paired CFG rows instead of unpaired no-CFG rows. The "
            "effective batch size becomes cfg-pairs * 2."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--codec-min-id", type=int, default=DEFAULT_CODEC_MIN_ID)
    parser.add_argument("--codec-max-id", type=int, default=DEFAULT_CODEC_MAX_ID)
    parser.add_argument(
        "--speechgen-end-id", type=int, default=DEFAULT_SPEECHGEN_END_ID
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="MLX random seed used for synthetic hidden/head tensors.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="optional path to write benchmark results as JSON",
    )
    parser.add_argument(
        "--fail-if-pending-full-ms-over",
        type=float,
        default=None,
        help=(
            "exit nonzero when pending full-vocab projection plus sampling is "
            "slower than this average milliseconds threshold"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.cfg_pairs < 0:
        raise SystemExit("--cfg-pairs must be non-negative")
    if args.cfg_pairs and args.batch_size != 1:
        raise SystemExit("--batch-size and --cfg-pairs are mutually exclusive")

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)

    allowed_window = (
        int(args.codec_min_id),
        int(args.codec_max_id),
        int(args.speechgen_end_id),
    )
    codec_window_size = args.codec_max_id - args.codec_min_id + 2
    sampling_params = _sampling_params_for_args(args, allowed_window)
    plan = vllm_metal_cfg._build_cfg_pair_sampling_plan(sampling_params)
    effective_batch_size = len(sampling_params)

    hidden = mx.random.normal((effective_batch_size, args.hidden_size)).astype(
        mx.bfloat16
    )
    full_head = mx.random.normal((args.vocab_size, args.hidden_size)).astype(
        mx.bfloat16
    )
    end_head = full_head[args.speechgen_end_id : args.speechgen_end_id + 1]
    codec_head = full_head[args.codec_min_id : args.codec_max_id + 1]
    window_head = mx.concatenate((end_head, codec_head), axis=0)
    raw_window_logits = mx.random.normal(
        (effective_batch_size, codec_window_size)
    ).astype(mx.float32)
    mx.eval(hidden, full_head, window_head, raw_window_logits)

    results = {
        "device": str(mx.default_device()),
        "batch_size": effective_batch_size,
        "requested_batch_size": args.batch_size,
        "cfg_pairs": args.cfg_pairs,
        "cfg_scale": args.cfg_scale if args.cfg_pairs else None,
        "sample_rows": len(plan.sample_row_indices),
        "output_rows": len(sampling_params),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "codec_window_size": codec_window_size,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "benchmarks": {
            "raw_window_sample": _bench(
                mx,
                args,
                lambda: _sample_window_logits(
                    mx,
                    raw_window_logits,
                    sampling_params,
                    plan,
                ),
            ),
            "pending_full_projection_then_window_sample": _bench(
                mx,
                args,
                lambda: _pending_full_projection_then_sample(
                    mx,
                    hidden,
                    full_head,
                    sampling_params,
                    plan,
                    allowed_window,
                ),
            ),
            "pending_window_projection_then_sample": _bench(
                mx,
                args,
                lambda: _pending_window_projection_then_sample(
                    mx,
                    hidden,
                    window_head,
                    sampling_params,
                    plan,
                ),
            ),
        },
    }

    text = json.dumps(results, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")

    threshold = args.fail_if_pending_full_ms_over
    if threshold is not None:
        avg_ms = results["benchmarks"]["pending_full_projection_then_window_sample"][
            "avg_ms"
        ]
        if avg_ms > threshold:
            print(
                "FAIL: pending_full_projection_then_window_sample avg_ms "
                f"{avg_ms:.3f} > {threshold:.3f}",
                flush=True,
            )
            return 1
    return 0


def _bench(
    mx: Any, args: argparse.Namespace, func: Callable[[], Any]
) -> dict[str, Any]:
    for _ in range(args.warmup):
        mx.eval(func())
    durations_ms: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        mx.eval(func())
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "avg_ms": round(statistics.fmean(durations_ms), 3),
        "median_ms": round(statistics.median(durations_ms), 3),
        "min_ms": round(min(durations_ms), 3),
        "max_ms": round(max(durations_ms), 3),
        "samples_ms": [round(value, 3) for value in durations_ms],
    }


def _sample_window_logits(
    mx: Any,
    logits: Any,
    sampling_params: list[SimpleNamespace],
    plan: Any,
) -> Any:
    sample_logits = vllm_metal_cfg._build_native_sample_logits(
        logits,
        sampling_params,
        plan,
        mx,
        allowed_window=None,
    )
    return vllm_metal_cfg._sample_random_tokens_mlx(
        sample_logits,
        plan,
        mx,
        logits_are_allowed_window=True,
    )


def _pending_full_projection_then_sample(
    mx: Any,
    hidden: Any,
    full_head: Any,
    sampling_params: list[SimpleNamespace],
    plan: Any,
    allowed_window: tuple[int, int, int],
) -> Any:
    logits = hidden.astype(mx.float32) @ mx.transpose(full_head).astype(mx.float32)
    sample_logits = vllm_metal_cfg._build_native_sample_logits(
        logits,
        sampling_params,
        plan,
        mx,
        allowed_window=allowed_window,
    )
    return vllm_metal_cfg._sample_random_tokens_mlx(
        sample_logits,
        plan,
        mx,
        logits_are_allowed_window=True,
    )


def _pending_window_projection_then_sample(
    mx: Any,
    hidden: Any,
    window_head: Any,
    sampling_params: list[SimpleNamespace],
    plan: Any,
) -> Any:
    logits = hidden.astype(mx.float32) @ mx.transpose(window_head).astype(mx.float32)
    sample_logits = vllm_metal_cfg._build_native_sample_logits(
        logits,
        sampling_params,
        plan,
        mx,
        allowed_window=None,
    )
    return vllm_metal_cfg._sample_random_tokens_mlx(
        sample_logits,
        plan,
        mx,
        logits_are_allowed_window=True,
    )


def _sampling_params_for_args(
    args: argparse.Namespace,
    allowed_window: tuple[int, int, int],
) -> list[SimpleNamespace]:
    def extra_args() -> dict[str, int | float | str]:
        return {
            "audex_tts_codec_min_id": allowed_window[0],
            "audex_tts_codec_max_id": allowed_window[1],
            "audex_tts_speechgen_end_id": allowed_window[2],
        }

    if args.cfg_pairs:
        params: list[SimpleNamespace] = []
        for pair_index in range(args.cfg_pairs):
            pair_id = f"pair-{pair_index}"
            params.append(
                SimpleNamespace(
                    temperature=args.temperature,
                    top_k=args.top_k,
                    extra_args={
                        **extra_args(),
                        "cfg_role": "cond",
                        "cfg_pair_id": pair_id,
                        "cfg_scale": args.cfg_scale,
                    },
                )
            )
            params.append(
                SimpleNamespace(
                    temperature=args.temperature,
                    top_k=args.top_k,
                    extra_args={
                        **extra_args(),
                        "cfg_role": "uncond",
                        "cfg_pair_id": pair_id,
                    },
                )
            )
        return params

    return [
        SimpleNamespace(
            temperature=args.temperature,
            top_k=args.top_k,
            extra_args=extra_args(),
        )
        for _ in range(args.batch_size)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
