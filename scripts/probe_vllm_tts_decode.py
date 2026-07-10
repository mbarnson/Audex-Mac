#!/usr/bin/env python3
"""Generate one vLLM Metal Audex TTS sample, decode it, and write diagnostics."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from audex_mac.audio_contract import NVIDIA_TTS_CFG_SCALE
from audex_mac.audio_runtime import preflight_audio_runtime
from audex_mac.models import AUDEX_2B_REPO, AUDEX_30B_REPO, SUPPORTED_MODELS
from audex_mac.speech_decoder import (
    AudexSpeechDecoderSession,
    load_speech_decoder_config,
    load_speech_decoder_weights_mlx,
)
from audex_mac.speech_output import RUNS_DIR, write_pcm16_wav
from audex_mac.vllm_runtime import (
    AudexAsyncVllmRuntime,
    AudexVllmRuntime,
    extract_tts_codec_frames,
)
from audex_mac.vllm_sts_requests import build_tts_cfg_requests, build_tts_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Audex vLLM Metal TTS decode.")
    parser.add_argument("--model", default="audex-30b-a3b")
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="text to synthesize; repeat for multi-request no-CFG batch probes",
    )
    parser.add_argument(
        "--cfg-segment",
        action="append",
        default=None,
        help=(
            "explicit CFG TTS segment to synthesize as one conditional/"
            "unconditional pair; repeat to probe multi-pair CFG concurrency"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chunk-frames", type=int, default=5)
    parser.add_argument(
        "--codec-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "pass the speech codec/end-token window into the vLLM Metal sampler "
            "(default: enabled; use --no-codec-window for A/B diagnostics)"
        ),
    )
    parser.add_argument(
        "--no-cfg",
        action="store_true",
        help="probe a single TTS request without classifier-free guidance",
    )
    parser.add_argument(
        "--parallel-requests",
        type=int,
        default=1,
        help=(
            "submit this many identical TTS requests together; diagnostic for "
            "continuous-batching throughput"
        ),
    )
    parser.add_argument(
        "--sync-final",
        action="store_true",
        help=(
            "use synchronous LLM.generate() for no-CFG final-output TTS; "
            "diagnostic for async streaming overhead"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    aliases = {
        "audex-2b": AUDEX_2B_REPO,
        "2b": AUDEX_2B_REPO,
        "audex-30b-a3b": AUDEX_30B_REPO,
        "30b": AUDEX_30B_REPO,
    }
    model_id = aliases.get(args.model, args.model)
    model = next(
        candidate for candidate in SUPPORTED_MODELS if candidate.repo_id == model_id
    )
    preflight = preflight_audio_runtime(model)
    if (
        not preflight.ready
        or preflight.model_path is None
        or preflight.decoder_path is None
    ):
        raise RuntimeError(f"Audex runtime is not ready: {preflight.missing_items}")

    if not args.text and not args.cfg_segment:
        raise ValueError("provide --text or one or more --cfg-segment values")
    if args.cfg_segment and args.no_cfg:
        raise ValueError("--cfg-segment cannot be combined with --no-cfg")
    if args.cfg_segment and args.sync_final:
        raise ValueError("--cfg-segment cannot be combined with --sync-final")
    if args.sync_final and not args.no_cfg:
        raise ValueError("--sync-final is currently supported only with --no-cfg.")
    if args.sync_final and (len(args.text or ()) > 1 or args.parallel_requests != 1):
        raise ValueError("--sync-final supports one no-CFG TTS request at a time.")

    runtime = (
        AudexVllmRuntime.from_model_path(preflight.model_path)
        if args.sync_final
        else AudexAsyncVllmRuntime.from_model_path(preflight.model_path)
    )
    runtime_stats = runtime.stats
    if args.cfg_segment:
        _require_cfg_runtime_for_segment_probe(runtime)
    print("probe: runtime loaded", flush=True)
    started = time.time()
    all_token_ids: list[int] = []
    all_codec_frames: list[int] = []
    reached_end = False
    event_count = 0
    pair_id = f"probe-{time.strftime('%Y%m%d%H%M%S')}"
    parallel_requests = max(1, int(args.parallel_requests))
    request_texts = tuple(str(text) for text in (args.cfg_segment or args.text or ()))
    if len(request_texts) > 1 and not args.no_cfg and not args.cfg_segment:
        raise ValueError("Repeated --text is supported only with --no-cfg.")
    token_ids_by_request: dict[str, list[int]] = {}
    codec_frames_by_request: dict[str, list[int]] = {}
    reached_end_by_request: dict[str, bool] = {}
    if args.sync_final:
        request_text = request_texts[0]
        result = runtime.generate_tts(
            request_text,
            max_tokens=args.max_tokens,
        )
        codec = extract_tts_codec_frames(result.token_ids, runtime.token_map)
        token_ids_by_request["tts"] = list(result.token_ids)
        codec_frames_by_request["tts"] = list(codec.generated_codec_frames)
        reached_end_by_request["tts"] = codec.reached_end_token
        event_count = 1
        requests = ()
        print(
            "probe: sync final "
            f"tokens={len(result.token_ids)} "
            f"frames={len(codec.generated_codec_frames)} "
            f"end={codec.reached_end_token} "
            f"elapsed={result.elapsed_seconds}",
            flush=True,
        )
    elif args.cfg_segment:
        segments = request_texts
        requests = []
        cond_debug_to_index: dict[str, int] = {}
        for index, segment in enumerate(segments):
            cond, uncond = runtime.build_tts_cfg_pair(
                segment,
                pair_id=f"{pair_id}-seg-{index}",
                max_tokens=args.max_tokens,
            )
            cond_name = f"tts-segment-{index}-cond"
            uncond_name = f"tts-segment-{index}-uncond"
            requests.extend(
                (
                    replace(
                        cond,
                        debug_name=cond_name,
                        request_id_suffix=f"seg-{index}-cond",
                        sampling=replace(cond.sampling, output_kind="DELTA"),
                    ),
                    replace(
                        uncond,
                        debug_name=uncond_name,
                        request_id_suffix=f"seg-{index}-uncond",
                        sampling=replace(uncond.sampling, output_kind="DELTA"),
                    ),
                )
            )
            cond_debug_to_index[cond_name] = index
        finished_segments: set[int] = set()
        print(
            "probe: CFG segments built "
            + " ".join(
                f"segment_{index}_chars={len(segment)}"
                for index, segment in enumerate(segments)
            ),
            flush=True,
        )
        async for delta in runtime.stream_many(
            tuple(requests),
            include_cumulative_token_ids=False,
        ):
            segment_index = cond_debug_to_index.get(delta.request_debug_name)
            if segment_index is None:
                continue
            event_count += 1
            request_key = f"segment-{segment_index}"
            token_ids = token_ids_by_request.setdefault(request_key, [])
            frames = codec_frames_by_request.setdefault(request_key, [])
            reached_end = reached_end_by_request.get(request_key, False)
            for token_id in delta.new_token_ids:
                token_ids.append(token_id)
                if token_id == runtime.token_map.speechgen_end:
                    reached_end = True
                    continue
                codec_frame = runtime.token_map.speech_codec.get(token_id)
                if codec_frame is not None:
                    frames.append(codec_frame)
            reached_end_by_request[request_key] = reached_end
            segment_finished = delta.finished or reached_end
            if segment_finished:
                finished_segments.add(segment_index)
            if (
                event_count == 1
                or event_count % 25 == 0
                or segment_finished
                or len(finished_segments) == len(segments)
            ):
                print(
                    "probe: cfg segment delta "
                    f"events={event_count} segment={segment_index} "
                    f"frames={len(frames)} "
                    f"segment_finished={segment_finished} "
                    f"finished={len(finished_segments) == len(segments)} "
                    f"elapsed={delta.elapsed_seconds}",
                    flush=True,
                )
            if len(finished_segments) == len(segments):
                break
    elif args.no_cfg:
        no_cfg_texts = (
            request_texts * parallel_requests
            if len(request_texts) == 1
            else request_texts
        )
        requests = []
        for index, text in enumerate(no_cfg_texts):
            request = (
                runtime.build_tts_request(text, max_tokens=args.max_tokens)
                if args.codec_window
                else build_tts_request(
                    runtime.tokenizer,
                    text,
                    speechgen_end_id=runtime.token_map.speechgen_end,
                    eos_token_id=runtime.tokenizer.eos_token_id,
                    max_tokens=args.max_tokens,
                    skip_paged_logits_eval=True,
                )
            )
            requests.append(
                replace(
                    request,
                    debug_name=f"tts-{index}",
                    request_id_suffix=f"probe-{index}",
                    sampling=replace(request.sampling, output_kind="DELTA"),
                )
            )
        requests = tuple(requests)
    elif args.codec_window:
        requests = build_tts_cfg_requests(
            runtime.tokenizer,
            request_texts[0],
            speechgen_end_id=runtime.token_map.speechgen_end,
            eos_token_id=runtime.tokenizer.eos_token_id,
            pair_id=pair_id,
            codec_min_id=min(runtime.token_map.speech_codec),
            codec_max_id=max(runtime.token_map.speech_codec),
            max_tokens=args.max_tokens,
        )
    else:
        requests = build_tts_cfg_requests(
            runtime.tokenizer,
            request_texts[0],
            speechgen_end_id=runtime.token_map.speechgen_end,
            eos_token_id=runtime.tokenizer.eos_token_id,
            pair_id=pair_id,
            max_tokens=args.max_tokens,
        )
    if not args.sync_final and not args.cfg_segment:
        prompt_summary = _request_prompt_summary(requests)
        print(
            f"probe: requests built {prompt_summary}",
            flush=True,
        )
        finished_requests: set[str] = set()
        async for delta in runtime.stream_many(requests):
            if (
                delta.request_debug_name != "tts-cond"
                and not delta.request_debug_name.startswith("tts-")
            ):
                continue
            event_count += 1
            codec = extract_tts_codec_frames(delta.token_ids, runtime.token_map)
            request_key = delta.request_debug_name
            token_ids_by_request[request_key] = list(delta.token_ids)
            codec_frames_by_request[request_key] = list(codec.generated_codec_frames)
            reached_end_by_request[request_key] = codec.reached_end_token
            if delta.finished or codec.reached_end_token:
                finished_requests.add(request_key)
            if event_count == 1 or event_count % 25 == 0 or codec.reached_end_token:
                print(
                    "probe: cond delta "
                    f"events={event_count} tokens={len(delta.token_ids)} "
                    f"frames={len(codec.generated_codec_frames)} "
                    f"finished={delta.finished} end={codec.reached_end_token} "
                    f"elapsed={delta.elapsed_seconds}",
                    flush=True,
                )
            if len(finished_requests) >= len(requests):
                break
    first_request_key = next(iter(token_ids_by_request), "")
    all_token_ids = token_ids_by_request.get(first_request_key, [])
    all_codec_frames = codec_frames_by_request.get(first_request_key, [])
    reached_end = reached_end_by_request.get(first_request_key, False)

    generation_seconds = round(time.time() - started, 3)
    submitted_cfg_pair_count = (
        len(args.cfg_segment or ())
        if args.cfg_segment
        else (len(requests) // 2 if not args.no_cfg and not args.sync_final else 0)
    )
    aggregate_generated_token_count = sum(
        len(token_ids) for token_ids in token_ids_by_request.values()
    )
    aggregate_generated_codec_frame_count = sum(
        len(frames) for frames in codec_frames_by_request.values()
    )
    decoder_config = load_speech_decoder_config(preflight.decoder_path)
    decoder_weights = load_speech_decoder_weights_mlx(preflight.decoder_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    wav_path = args.output_dir / f"tts-probe-vllm-{timestamp}.wav"
    json_path = args.output_dir / f"tts-probe-vllm-{timestamp}.json"
    per_request_outputs = []
    all_samples: list[float] = []
    for index, (request_key, request_frames) in enumerate(
        codec_frames_by_request.items()
    ):
        decoder_session = AudexSpeechDecoderSession(
            weights=decoder_weights,
            config=decoder_config,
            chunk_frames=args.chunk_frames,
        )
        samples: list[float] = []
        frames = tuple((frame,) for frame in request_frames)
        for sample_rate, waveform in decoder_session.push(frames):
            if sample_rate != decoder_config.sample_rate:
                raise RuntimeError(f"unexpected decoder sample_rate={sample_rate}")
            samples.extend(float(sample) for sample in waveform.tolist())
        for sample_rate, waveform in decoder_session.flush():
            if sample_rate != decoder_config.sample_rate:
                raise RuntimeError(f"unexpected decoder sample_rate={sample_rate}")
            samples.extend(float(sample) for sample in waveform.tolist())
        request_wav_path = (
            wav_path
            if index == 0
            else args.output_dir / f"tts-probe-vllm-{timestamp}-{request_key}.wav"
        )
        write_pcm16_wav(
            request_wav_path,
            samples,
            sample_rate=decoder_config.sample_rate,
        )
        if index == 0:
            all_samples = samples
        per_request_outputs.append(
            {
                "request_key": request_key,
                "text": (
                    request_texts[index]
                    if index < len(request_texts)
                    else request_texts[0]
                ),
                "generated_token_count": len(token_ids_by_request.get(request_key, ())),
                "generated_codec_frame_count": len(request_frames),
                "reached_end_token": reached_end_by_request.get(request_key, False),
                "sample_count": len(samples),
                "wav_path": str(request_wav_path),
            }
        )
    write_pcm16_wav(wav_path, all_samples, sample_rate=decoder_config.sample_rate)
    payload = {
        "text": request_texts[0],
        "texts": list(request_texts),
        "model_path": str(preflight.model_path),
        "decoder_path": str(preflight.decoder_path),
        "runtime": {
            "engine_class": runtime_stats.engine_class,
            "cfg_enabled": runtime_stats.cfg_enabled,
            "cfg_scale": (
                runtime.cfg_config.cfg_scale
                if runtime.cfg_config is not None
                else None
            ),
            "max_num_seqs": (
                runtime.cfg_config.max_num_seqs
                if runtime.cfg_config is not None
                else None
            ),
            "max_num_batched_tokens": (
                runtime.cfg_config.max_num_batched_tokens
                if runtime.cfg_config is not None
                else None
            ),
            "max_model_len": (
                runtime.cfg_config.max_model_len
                if runtime.cfg_config is not None
                else None
            ),
            "scheduler_reserve_full_isl": (
                runtime.cfg_config.scheduler_reserve_full_isl
                if runtime.cfg_config is not None
                else None
            ),
            "nonpaged_kv_capacity_seqs": runtime_stats.nonpaged_kv_capacity_seqs,
        },
        "max_tokens": args.max_tokens,
        "elapsed_seconds": round(time.time() - started, 3),
        "generation_seconds": generation_seconds,
        "event_count": event_count,
        "request_count": len(requests) if not args.sync_final else 1,
        "cfg_segment_count": len(args.cfg_segment or ()),
        "submitted_cfg_pair_count": submitted_cfg_pair_count,
        "generated_token_count": len(all_token_ids),
        "generated_codec_frame_count": len(all_codec_frames),
        "parallel_requests": parallel_requests,
        "aggregate_generated_token_count": aggregate_generated_token_count,
        "aggregate_generated_codec_frame_count": aggregate_generated_codec_frame_count,
        "aggregate_tokens_per_second": round(
            aggregate_generated_token_count / max(generation_seconds, 1.0e-9),
            3,
        ),
        "aggregate_codec_frames_per_second": round(
            aggregate_generated_codec_frame_count / max(generation_seconds, 1.0e-9),
            3,
        ),
        "first_request_tokens_per_second": round(
            len(all_token_ids) / max(generation_seconds, 1.0e-9),
            3,
        ),
        "generated_token_ids": all_token_ids,
        "generated_codec_frames": all_codec_frames,
        "codec_window": bool(args.codec_window),
        "cfg_enabled": not bool(args.no_cfg),
        "sync_final": bool(args.sync_final),
        "reached_end_token": reached_end,
        "hit_max_tokens": len(all_token_ids) >= args.max_tokens and not reached_end,
        "sample_count": len(all_samples),
        "sample_rate": decoder_config.sample_rate,
        "wav_path": str(wav_path),
        "request_outputs": per_request_outputs,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    return asyncio.run(main_async())


def _require_cfg_runtime_for_segment_probe(runtime) -> None:
    cfg_config = getattr(runtime, "cfg_config", None)
    cfg_enabled = bool(getattr(cfg_config, "enabled", False))
    cfg_scale = getattr(cfg_config, "cfg_scale", None)
    if cfg_enabled and cfg_scale == NVIDIA_TTS_CFG_SCALE:
        return
    raise RuntimeError(
        "--cfg-segment requires real Audex CFG wiring. Set "
        "AUDEX_VLLM_TTS_CFG=1 and AUDEX_VLLM_ENABLE_CFG_WIRING=1 before "
        "running this standalone probe. "
        f"Observed cfg_enabled={cfg_enabled} cfg_scale={cfg_scale!r}; "
        f"expected cfg_scale={NVIDIA_TTS_CFG_SCALE}."
    )


def _request_prompt_summary(requests) -> str:
    parts: list[str] = []
    for index, request in enumerate(requests):
        prompt = request.prompt
        if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
            parts.append(f"request_{index}_tokens={len(prompt['prompt_token_ids'])}")
        else:
            parts.append(f"request_{index}_chars={len(str(prompt))}")
    return " ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
