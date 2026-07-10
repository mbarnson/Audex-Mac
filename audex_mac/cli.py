"""Audex-Mac command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .audio_encoder import run_audio_encoder_smoke
from .audio_features import extract_audex_input_features
from .audio_pcm import prepare_audex_pcm_clips, prepare_audex_wav_clips
from .audio_projector import run_audio_projector_smoke
from .audio_runtime import preflight_audio_runtime
from .audio_splice import run_audio_embedding_splice_smoke
from .bootstrap import model_download_notice
from .conversations import DEFAULT_DEMO_CONTEXT_TOKENS, ConversationStore
from .model_select import (
    HuggingFaceSnapshotProbe,
    ModelReadiness,
    download_model_snapshot,
    select_model,
)
from .models import (
    AUDEX_2B_REPO,
    AUDEX_30B_NVFP4_REPO,
    AUDEX_30B_REPO,
    SUPPORTED_MODELS,
    AudexModel,
)
from .personas import DEFAULT_PERSONA_NAME, load_persona
from .speech_decoder import run_speech_decoder_smoke
from .speech_generation import run_speech_token_generation_smoke
from .speech_output import run_speech_output_smoke
from .speech_policy import assistant_prefix
from .sts_cli import (
    DEFAULT_RESPONSE_MAX_TOKENS,
    AudexSpeechToSpeechSession,
    run_fixture_turn,
    run_interactive_ptt,
)
from .text_benchmark import load_text_benchmark
from .text_generation import run_text_benchmark
from .text_runtime import preflight_text_runtime
from .tts_quality import TTS_QUALITY_RECIPES, tts_quality_recipe
from .vendor_pins import fetch_vllm_metal_upstream_head, load_vllm_metal_pin
from .vllm_commands import (
    run_vllm_fixture_turn,
    run_vllm_interactive_ptt,
    run_vllm_tts_quality_probe,
    run_vllm_tts_text_probe,
)
from .vllm_diagnostics import run_vllm_metal_diagnostics

MODEL_CHOICES = {
    "audex-2b": AUDEX_2B_REPO,
    "audex-30b-a3b": AUDEX_30B_REPO,
    "audex-30b-a3b-nvfp4": AUDEX_30B_NVFP4_REPO,
}
DEFAULT_TEXT_BACKEND = "vllm"
DEFAULT_STS_BACKEND = "vllm"


def _demo_context_tokens(raw_value: str) -> int:
    value = int(raw_value)
    if not 1 <= value <= DEFAULT_DEMO_CONTEXT_TOKENS:
        raise ValueError(
            "--max-context-tokens must be between 1 and "
            f"{DEFAULT_DEMO_CONTEXT_TOKENS}"
        )
    return value


def _model_by_repo(repo_id: str) -> AudexModel:
    return next(model for model in SUPPORTED_MODELS if model.repo_id == repo_id)


def _model_size_notice(model: AudexModel) -> str:
    return "a very large download" if model.higher_reasoning else "about 10 GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audex-Mac local speech demo")
    parser.add_argument("--thinking", action="store_true", help="allow thinking mode")
    parser.add_argument(
        "--select-model-only",
        action="store_true",
        help="resolve the model and print startup policy without launching audio",
    )
    parser.add_argument(
        "--check-upstream",
        action="store_true",
        help="check whether upstream vLLM Metal has moved beyond the pin",
    )
    parser.add_argument(
        "--show-text-benchmark",
        action="store_true",
        help="print the text benchmark contract and exit",
    )
    parser.add_argument(
        "--preflight-text-runtime",
        action="store_true",
        help="check local artifacts needed for the text-only benchmark",
    )
    parser.add_argument(
        "--diagnose-vllm-metal",
        action="store_true",
        help="write a vLLM Metal device/path diagnostic report and exit",
    )
    parser.add_argument(
        "--diagnose-vllm-generation",
        action="store_true",
        help="with --diagnose-vllm-metal, load vLLM and run text generation timing probes",
    )
    parser.add_argument(
        "--diagnose-generation-max-tokens",
        type=int,
        default=None,
        help="max tokens for the long vLLM diagnostic generation probe",
    )
    parser.add_argument(
        "--diagnose-vllm-sts-smoke",
        action="store_true",
        help=(
            "with --diagnose-vllm-metal, run one default async vLLM "
            "speech-to-speech fixture turn and record timing evidence"
        ),
    )
    parser.add_argument(
        "--diagnose-vllm-sts-play",
        action="store_true",
        help=(
            "with --diagnose-vllm-sts-smoke, play the generated audio through "
            "the continuous PCM playback path and require playback timing evidence"
        ),
    )
    parser.add_argument(
        "--diagnose-vllm-sts-audio-fixture",
        type=Path,
        default=None,
        help=(
            "optional 16 kHz WAV fixture for --diagnose-vllm-sts-smoke; "
            "defaults to one second of generated silence"
        ),
    )
    parser.add_argument(
        "--diagnose-vllm-sts-speech-max-tokens",
        type=int,
        default=None,
        help=(
            "maximum speech-codec tokens for --diagnose-vllm-sts-smoke; "
            "diagnostic-only and does not change interactive STS defaults"
        ),
    )
    parser.add_argument(
        "--diagnose-vllm-native-sampling-debug",
        action="store_true",
        help=(
            "with vLLM Metal STS/TTS diagnostics, enable native sampler timing "
            "stderr; disabled by default so smoke probes measure the normal "
            "fast path"
        ),
    )
    parser.add_argument(
        "--diagnose-vllm-tts-batch-size",
        type=int,
        default=None,
        help=(
            "diagnostic-only number of parallel TTS CFG pairs to submit to one "
            "async vLLM Metal engine"
        ),
    )
    parser.add_argument(
        "--diagnose-vllm-tts-batch-max-tokens",
        type=int,
        default=128,
        help="maximum speech-codec tokens per request for --diagnose-vllm-tts-batch-size",
    )
    parser.add_argument(
        "--diagnose-vllm-tts-batch-text",
        default="Please explain Python context managers in two concise sentences.",
        help="text to synthesize for --diagnose-vllm-tts-batch-size",
    )
    parser.add_argument(
        "--diagnose-vllm-tts-batch-no-cfg",
        action="store_true",
        help=(
            "diagnostic-only: run --diagnose-vllm-tts-batch-size with plain TTS "
            "requests instead of NVIDIA CFG pairs"
        ),
    )
    parser.add_argument(
        "--diagnose-vllm-tts-text",
        default=None,
        help=(
            "diagnostic-only fixed text to synthesize through the product vLLM "
            "speech-output path; respects AUDEX_VLLM_TTS_CFG"
        ),
    )
    parser.add_argument(
        "--diagnose-vllm-tts-text-file",
        type=Path,
        default=None,
        help=(
            "diagnostic-only UTF-8 text file to synthesize through the product "
            "vLLM speech-output path"
        ),
    )
    parser.add_argument(
        "--diagnose-vllm-tts-quality-corpus",
        type=Path,
        default=None,
        help=(
            "diagnostic-only JSON corpus of long passages to synthesize through "
            "one warm product vLLM speech-output session"
        ),
    )
    parser.add_argument(
        "--diagnose-vllm-tts-quality-recipe",
        choices=tuple(TTS_QUALITY_RECIPES),
        default="plain-reference",
        help="enforced TTS recipe recorded in the private quality manifest",
    )
    parser.add_argument(
        "--diagnose-vllm-tts-quality-seed",
        type=int,
        default=20260709,
        help="fixed vLLM sampling seed for a TTS quality corpus run",
    )
    parser.add_argument(
        "--preflight-audio-runtime",
        action="store_true",
        help="check local artifacts needed for native Audex speech-to-speech",
    )
    parser.add_argument(
        "--preflight-audio-features",
        action="store_true",
        help="run Audex audio preprocessing and validate NV-Whisper feature shape",
    )
    parser.add_argument(
        "--preflight-audio-projector",
        action="store_true",
        help="load Audex audio projector weights in MLX and validate output shape",
    )
    parser.add_argument(
        "--preflight-audio-encoder",
        action="store_true",
        help="run Audex audio encoder and projector weights in MLX on zero features",
    )
    parser.add_argument(
        "--preflight-audio-splice",
        action="store_true",
        help="splice Audex audio embeddings into an MLX text-model prompt smoke",
    )
    parser.add_argument(
        "--preflight-speech-token-generation",
        action="store_true",
        help="load the full-vocab Audex MLX head and generate speech-codec tokens",
    )
    parser.add_argument(
        "--preflight-speech-decoder",
        action="store_true",
        help="load Audex causal speech decoder weights in MLX and decode waveform",
    )
    parser.add_argument(
        "--preflight-speech-output",
        action="store_true",
        help="generate Audex speech tokens, decode them, and write a WAV smoke",
    )
    parser.add_argument(
        "--audio-fixture",
        type=Path,
        default=None,
        help="optional 16 kHz mono/stereo PCM WAV fixture for audio feature preflight",
    )
    parser.add_argument(
        "--input-wav",
        type=Path,
        default=None,
        help="run one speech-to-speech turn from a 16 kHz PCM WAV instead of microphone capture",
    )
    parser.add_argument(
        "--persona",
        default=DEFAULT_PERSONA_NAME,
        help="persona name from personas/ or path to a markdown persona file",
    )
    parser.add_argument(
        "--new-conversation",
        action="store_true",
        help="start a new persistent conversation and make it the default resume target",
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="resume a specific persistent conversation id",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=_demo_context_tokens,
        default=DEFAULT_DEMO_CONTEXT_TOKENS,
        help="exact active conversation budget, capped at the 262144-token demo limit",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="disable binary conversation KV cache load/save for diagnostics",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="write output audio but do not play it through the local audio device",
    )
    parser.add_argument(
        "--response-max-tokens",
        type=int,
        default=DEFAULT_RESPONSE_MAX_TOKENS,
        help="maximum Audex text response tokens for the STS CLI",
    )
    parser.add_argument(
        "--speech-max-tokens",
        type=int,
        default=None,
        help=(
            "maximum Audex speech-codec tokens for the STS CLI; "
            "default scales from response length"
        ),
    )
    parser.add_argument(
        "--run-text-benchmark",
        action="store_true",
        help="run the ten-turn Audex text benchmark and write a transcript log",
    )
    parser.add_argument(
        "--text-backend",
        choices=("mlx", "vllm"),
        default=DEFAULT_TEXT_BACKEND,
        help=(
            "text benchmark backend; vllm is the default Metal path, "
            "mlx is an explicit diagnostic fallback"
        ),
    )
    parser.add_argument(
        "--sts-backend",
        choices=("vllm", "mlx"),
        default=DEFAULT_STS_BACKEND,
        help=(
            "speech-to-speech backend; vllm is the default Metal path, "
            "mlx is an explicit diagnostic fallback"
        ),
    )
    parser.add_argument(
        "--limit-text-turns",
        type=int,
        default=None,
        help="development-only limit for text benchmark turns",
    )
    parser.add_argument(
        "--model",
        choices=("auto", *MODEL_CHOICES.keys()),
        default="auto",
        help="model to target for development preflights",
    )
    parser.add_argument(
        "--yes-download",
        action="store_true",
        help="approve downloading the selected Audex model if it is missing",
    )
    args = parser.parse_args(argv)
    if (
        args.diagnose_vllm_sts_speech_max_tokens is not None
        and args.diagnose_vllm_sts_speech_max_tokens <= 0
    ):
        parser.error("--diagnose-vllm-sts-speech-max-tokens must be positive")
    if args.diagnose_vllm_tts_batch_size is not None:
        if args.diagnose_vllm_tts_batch_size <= 0:
            parser.error("--diagnose-vllm-tts-batch-size must be positive")
        if args.diagnose_vllm_tts_batch_max_tokens <= 0:
            parser.error("--diagnose-vllm-tts-batch-max-tokens must be positive")
    if (
        args.diagnose_vllm_tts_text is not None
        and args.diagnose_vllm_tts_text_file is not None
    ):
        parser.error(
            "--diagnose-vllm-tts-text and --diagnose-vllm-tts-text-file "
            "are mutually exclusive"
        )
    if args.diagnose_vllm_tts_quality_corpus is not None and (
        args.diagnose_vllm_tts_text is not None
        or args.diagnose_vllm_tts_text_file is not None
    ):
        parser.error(
            "--diagnose-vllm-tts-quality-corpus cannot be combined with "
            "--diagnose-vllm-tts-text or --diagnose-vllm-tts-text-file"
        )
    if args.diagnose_vllm_sts_play:
        args.diagnose_vllm_sts_smoke = True
    if (
        args.diagnose_vllm_generation
        or args.diagnose_vllm_sts_smoke
        or args.diagnose_vllm_tts_batch_size is not None
    ):
        args.diagnose_vllm_metal = True

    if args.show_text_benchmark:
        benchmark = load_text_benchmark()
        print(f"Benchmark: {benchmark.name}")
        print(f"Turns: {len(benchmark.turns)}")
        print(f"max_tokens: {benchmark.max_tokens}")
        print(f"sampler_reference: {benchmark.sampler_reference}")
        return 0

    if args.check_upstream:
        pin = load_vllm_metal_pin()
        upstream = fetch_vllm_metal_upstream_head()
        print(f"vLLM Metal pin: {pin.pinned_commit}")
        print(f"vLLM Metal upstream main: {upstream}")
        if upstream != pin.pinned_commit:
            print("WARNING: upstream vLLM Metal has moved; startup remains pinned.")
        return 0

    readiness: ModelReadiness = (
        "speech"
        if (
            args.diagnose_vllm_sts_smoke
            or args.diagnose_vllm_tts_batch_size is not None
            or args.diagnose_vllm_tts_text is not None
            or args.diagnose_vllm_tts_text_file is not None
        )
        else "text" if _text_command(args) else "speech"
    )
    probe = HuggingFaceSnapshotProbe()
    selection = select_model(probe, readiness=readiness)
    selected_model = (
        selection.selected
        if args.model == "auto"
        else _model_by_repo(MODEL_CHOICES[args.model])
    )
    selected_cached = (
        selection.cached
        if args.model == "auto" and selected_model == selection.selected
        else probe.is_cached(selected_model, readiness)
    )
    if args.model == "auto":
        for message in selection.user_messages:
            print(message)
    print(f"Selected model: {selected_model.repo_id}")
    print(f"Thinking enabled: {args.thinking}")
    if args.sts_backend == "vllm":
        cfg_value = os.environ.get("AUDEX_VLLM_TTS_CFG", "")
        cfg_enabled = cfg_value.strip().lower() in {"1", "true", "yes", "on"}
        print(
            "TTS recipe: "
            + (
                "CFG3 quality mode (temperature=1.0, top_k=80, cfg_scale=3.0)"
                if cfg_enabled
                else "plain fast mode (CFG disabled)"
            )
        )
    prefix = assistant_prefix(thinking_enabled=args.thinking)
    if prefix:
        print("Speech-to-speech default: non-thinking mode")

    if args.select_model_only:
        return 0

    if args.diagnose_vllm_metal:
        result = run_vllm_metal_diagnostics(
            selected_model,
            run_generation=args.diagnose_vllm_generation,
            generation_max_tokens=args.diagnose_generation_max_tokens,
            run_sts_smoke=args.diagnose_vllm_sts_smoke,
            sts_audio_fixture=args.diagnose_vllm_sts_audio_fixture,
            sts_play_audio=args.diagnose_vllm_sts_play,
            sts_speech_max_tokens=args.diagnose_vllm_sts_speech_max_tokens,
            native_sampling_debug=args.diagnose_vllm_native_sampling_debug,
            tts_batch_size=args.diagnose_vllm_tts_batch_size,
            tts_batch_max_tokens=args.diagnose_vllm_tts_batch_max_tokens,
            tts_batch_text=args.diagnose_vllm_tts_batch_text,
            tts_batch_cfg=not args.diagnose_vllm_tts_batch_no_cfg,
        )
        report = result.report
        platform = report["vllm_metal"].get("platform", {})
        platform_resolution = report["platform_resolution_probe"]
        raw_current_platform = platform_resolution.get("current_platform", {})
        repaired_current_platform = platform_resolution.get(
            "current_platform_after_audex_patches",
            {},
        )
        mlx = report["vllm_metal"].get("mlx", {})
        spawn_mlx = report["spawn_probe"].get("mlx", {})
        print(f"vLLM Metal diagnostic report: {result.run_log_path}")
        print(
            "vLLM Metal platform: "
            f"device_type_facade={platform.get('device_type_facade')} "
            f"device_name={platform.get('device_name')} "
            f"paged_attention={report['vllm_metal'].get('config', {}).get('use_paged_attention')}"
        )
        print(
            "Parent MLX: "
            f"metal_available={mlx.get('metal_available')} "
            f"default_device={mlx.get('default_device')} "
            f"probe_array_device={mlx.get('probe_array_device')}"
        )
        print(
            "Spawn probe MLX: "
            f"metal_available={spawn_mlx.get('metal_available')} "
            f"default_device={spawn_mlx.get('default_device')} "
            f"probe_array_device={spawn_mlx.get('probe_array_device')}"
        )
        generation_probe = report["generation_probe"]
        if generation_probe.get("enabled"):
            print(
                "vLLM generation probe: "
                f"ready={generation_probe.get('ready')} "
                f"model_load_seconds={generation_probe.get('model_load_seconds')}"
            )
            if generation_probe.get("long_probe"):
                long_probe = generation_probe["long_probe"]
                print(
                    "vLLM generation throughput: "
                    f"generated_tokens={long_probe.get('generated_tokens')} "
                    f"tokens_per_second={long_probe.get('tokens_per_second')}"
                )
        sts_probe = report["sts_probe"]
        if sts_probe.get("enabled"):
            streaming = sts_probe.get("speech_streaming", {})
            print(
                "vLLM STS smoke probe: "
                f"ready={sts_probe.get('ready')} "
                f"engine_class={sts_probe.get('engine_class')} "
                f"elapsed_seconds={sts_probe.get('elapsed_seconds')}"
            )
            print(
                "vLLM STS smoke streaming: "
                f"vllm_token_streaming={streaming.get('vllm_token_streaming')} "
                f"decoder_streaming={streaming.get('decoder_streaming')} "
                f"playback_transport={streaming.get('playback_transport')} "
                f"first_audio_ready_seconds={streaming.get('first_audio_ready_seconds')} "
                f"first_playback_started_seconds={streaming.get('first_playback_started_seconds')} "
                f"generated_codec_frame_count={streaming.get('generated_codec_frame_count')}"
            )
            timing = sts_probe.get("vllm_metal_timing", {})
            latest_paged = timing.get("latest_paged_sample", {})
            if latest_paged:
                print(
                    "vLLM Metal TTS timing: "
                    f"count={latest_paged.get('count')} "
                    f"avg_ms={latest_paged.get('avg_ms')} "
                    f"last_ms={latest_paged.get('last_ms')} "
                    f"native_sample_ms={latest_paged.get('native_sample_ms')} "
                    f"native_sampled_rows={latest_paged.get('native_sampled_rows')} "
                    f"native_output_rows={latest_paged.get('native_output_rows')} "
                    f"skipped_logits_eval={latest_paged.get('skipped_logits_eval')} "
                    f"mx_eval_ms={latest_paged.get('mx_eval_ms')}"
                )
            assessment = sts_probe.get("sts_timing_assessment", {})
            if assessment:
                print(
                    "vLLM STS timing assessment: "
                    f"codec_fps={assessment.get('codec_frames_per_second')} "
                    f"realtime_ratio={assessment.get('audio_realtime_ratio')} "
                    f"paged_avg_ms={assessment.get('paged_sample_avg_ms')} "
                    f"skipped_logits_eval={assessment.get('skipped_logits_eval')} "
                    f"native_ms_per_step={assessment.get('native_sample_ms_per_step')} "
                    f"native_ms_per_sample={assessment.get('native_sample_ms_per_sampled_row')} "
                    f"dominant_eval_per_step={assessment.get('dominant_mx_eval_per_step_category')} "
                    f"native_row_ratio={assessment.get('native_sampling_row_ratio')} "
                    f"likely_bottleneck={assessment.get('likely_bottleneck')}"
                )
        tts_batch_probe = report.get("tts_batch_probe", {})
        if tts_batch_probe.get("enabled"):
            print(
                "vLLM TTS batch probe: "
                f"ready={tts_batch_probe.get('ready')} "
                f"batch_size={tts_batch_probe.get('batch_size')} "
                f"cfg_enabled={tts_batch_probe.get('cfg_enabled')} "
                f"request_count={tts_batch_probe.get('request_count')} "
                f"elapsed_seconds={tts_batch_probe.get('elapsed_seconds')} "
                f"codec_frames={tts_batch_probe.get('total_codec_frame_count')} "
                f"codec_fps={tts_batch_probe.get('codec_frames_per_second')} "
                f"min_frames={tts_batch_probe.get('min_codec_frames_per_request')} "
                f"max_frames={tts_batch_probe.get('max_codec_frames_per_request')} "
                f"reached_end={tts_batch_probe.get('reached_end_count')} "
                f"hit_max={tts_batch_probe.get('hit_max_token_count')}"
            )
        if platform.get("device_type_facade") == "cpu":
            print(
                "Note: vLLM Metal intentionally exposes a PyTorch CPU facade; "
                "treat MLX device evidence as authoritative for Metal execution."
            )
        if raw_current_platform or repaired_current_platform:
            print(
                "vLLM current_platform: "
                f"raw={raw_current_platform.get('class')} "
                f"after_audex_patches={repaired_current_platform.get('class')}"
            )
        verdict = report["verdict"]
        if verdict["ready"]:
            print("vLLM Metal diagnostic: ready")
            return 0
        print("vLLM Metal diagnostic: not ready")
        for failure in verdict["failures"]:
            print(f"Diagnostic failure: {failure}")
        if not report["parent_process"]["metal_policy"]["ready"]:
            return 2
        if report["spawn_probe"].get("returncode") != 0:
            return 2
        return 2

    if not selected_cached:
        notice = model_download_notice(
            selected_model.repo_id, _model_size_notice(selected_model)
        )
        print(notice)
        approved = args.yes_download
        if not approved:
            answer = input("Download now? [y/N] ").strip().lower()
            approved = answer in {"y", "yes"}
        if not approved:
            print("Model download was not approved; startup cannot continue.")
            return 2
        download_model_snapshot(selected_model, readiness)

    if args.preflight_text_runtime:
        preflight = preflight_text_runtime(selected_model)
        print(f"Text benchmark: {preflight.benchmark.name}")
        print(f"Text model path: {preflight.model_path}")
        print(f"Text max_tokens: {preflight.benchmark.max_tokens}")
        print(
            "Text sampler: "
            f"temperature={preflight.benchmark.generation['temperature']} "
            f"top_p={preflight.benchmark.generation['top_p']} "
            f"seed={preflight.benchmark.generation['seed']}"
        )
        if preflight.patch_report is not None:
            print(
                "Audex patches: "
                f"mlx_lm_nemotron_dense={preflight.patch_report.mlx_lm_nemotron_dense} "
                f"mlx_lm_nemotron_h_audex={preflight.patch_report.mlx_lm_nemotron_h_audex} "
                f"vllm_metal_platform_repair={preflight.patch_report.vllm_metal_platform_repair} "
                f"vllm_nemotron_dense={preflight.patch_report.vllm_nemotron_dense} "
                f"vllm_metal_audex_adapter={preflight.patch_report.vllm_metal_audex_adapter}"
            )
        if preflight.metal_policy is not None:
            metal_env = " ".join(
                f"{name}={value}" for name, value in preflight.metal_policy.env.items()
            )
            print(f"Metal/MLX policy: {metal_env}")
            if preflight.metal_policy.mlx_metal_available is not None:
                print(
                    "Metal/MLX live check: "
                    f"mlx_metal_available={preflight.metal_policy.mlx_metal_available} "
                    f"mlx_default_device={preflight.metal_policy.mlx_default_device}"
                )
        if preflight.ready:
            print("Text runtime preflight: ready")
            return 0
        print("Text runtime preflight: not ready")
        for item in preflight.missing_items:
            print(f"Missing: {item}")
        return 2

    if (
        args.preflight_audio_runtime
        or args.preflight_audio_features
        or args.preflight_audio_projector
        or args.preflight_audio_encoder
        or args.preflight_audio_splice
        or args.preflight_speech_token_generation
        or args.preflight_speech_decoder
        or args.preflight_speech_output
    ):
        preflight = preflight_audio_runtime(selected_model)
        _print_audio_preflight(preflight)
        print("Audio input contract: 16000 Hz PCM, 30.0s clips, 750 tokens/clip")
        print("Audio semantic models: Audex only; no Whisper/Kokoro/Silero")
        if (
            args.preflight_audio_features
            or args.preflight_audio_projector
            or args.preflight_audio_encoder
            or args.preflight_audio_splice
            or args.preflight_speech_token_generation
            or args.preflight_speech_decoder
            or args.preflight_speech_output
        ) and (not preflight.ready or preflight.model_path is None):
            label = (
                "Speech output preflight"
                if args.preflight_speech_output
                else (
                    "Speech decoder preflight"
                    if args.preflight_speech_decoder
                    else (
                        "Speech-token generation preflight"
                        if args.preflight_speech_token_generation
                        else (
                            "Audio splice preflight"
                            if args.preflight_audio_splice
                            else (
                                "Audio encoder preflight"
                                if args.preflight_audio_encoder
                                else (
                                    "Audio projector preflight"
                                    if args.preflight_audio_projector
                                    else "Audio feature preflight"
                                )
                            )
                        )
                    )
                )
            )
            print(f"{label}: not ready")
            for item in preflight.missing_items:
                print(f"Missing: {item}")
            return 2
        if args.preflight_audio_features:
            clips = (
                prepare_audex_wav_clips(args.audio_fixture)
                if args.audio_fixture is not None
                else prepare_audex_pcm_clips([0.0], sample_rate=16_000)
            )
            features = extract_audex_input_features(
                clips,
                preprocessor_path=preflight.model_path / "audio_preprocessor",
            )
            source = args.audio_fixture if args.audio_fixture is not None else "silence"
            print(f"Audio feature source: {source}")
            print(
                "Audio features: "
                f"extractor={features.feature_extractor_type} "
                f"clips={features.num_clips} "
                f"shape={features.feature_shape} "
                f"dtype={features.feature_dtype}"
            )
            print("Audio feature preflight: ready")
            return 0
        if args.preflight_audio_projector:
            projector = run_audio_projector_smoke(preflight.model_path)
            print(
                "Audio projector: "
                f"backend={projector.backend} "
                f"device={projector.device} "
                f"input_shape={projector.input_shape} "
                f"output_shape={projector.output_shape} "
                f"weight_dtype={projector.weight_dtype} "
                f"output_dtype={projector.output_dtype} "
                f"activation={projector.activation}"
            )
            print("Audio projector preflight: ready")
            return 0
        if args.preflight_audio_encoder:
            encoder = run_audio_encoder_smoke(preflight.model_path)
            print(
                "Audio encoder: "
                f"backend={encoder.backend} "
                f"device={encoder.device} "
                f"input_shape={encoder.input_shape} "
                f"encoder_shape={encoder.encoder_shape} "
                f"projected_shape={encoder.projected_shape} "
                f"encoder_dtype={encoder.encoder_dtype} "
                f"projected_dtype={encoder.projected_dtype} "
                f"layers={encoder.encoder_layers}"
            )
            print("Audio encoder preflight: ready")
            return 0
        if args.preflight_audio_splice:
            if preflight.snapshot_check.snapshot_path is None:
                print("Audio splice preflight: not ready")
                print("Missing: model snapshot path")
                return 2
            splice = run_audio_embedding_splice_smoke(
                full_model_path=preflight.model_path,
                text_model_path=(
                    preflight.snapshot_check.snapshot_path
                    / selected_model.text_checkpoint_dirs[0]
                ),
            )
            print(
                "Audio splice: "
                f"backend={splice.backend} "
                f"device={splice.device} "
                f"prompt_tokens={splice.prompt_tokens} "
                f"sound_tokens={splice.sound_tokens} "
                f"audio_embedding_shape={splice.audio_embedding_shape} "
                f"input_embedding_shape={splice.input_embedding_shape} "
                f"spliced_embedding_shape={splice.spliced_embedding_shape} "
                f"generated_token_id={splice.generated_token_id} "
                f"logprobs_shape={splice.logprobs_shape}"
            )
            print("Audio splice preflight: ready")
            return 0
        if args.preflight_speech_token_generation:
            speech = run_speech_token_generation_smoke(
                full_model_path=preflight.model_path,
            )
            print(
                "Speech-token generation: "
                f"backend={speech.backend} "
                f"device={speech.device} "
                f"model_type={speech.model_type} "
                f"vocab_size={speech.vocab_size} "
                f"prompt_tokens={speech.prompt_tokens} "
                f"prompt_max_token_id={speech.prompt_max_token_id} "
                f"speechgen_start={speech.speechgen_start_id} "
                f"speechgen_end={speech.speechgen_end_id} "
                f"codec_tokens={speech.codec_token_count} "
                f"generated_token_ids={speech.generated_token_ids} "
                f"generated_token_text={speech.generated_token_text} "
                f"generated_codec_frames={speech.generated_codec_frames} "
                f"logprobs_shape={speech.logprobs_shape}"
            )
            print(
                "Speech-token sampler: "
                f"temperature={speech.temperature} "
                f"top_p={speech.top_p} "
                f"top_k={speech.top_k} "
                f"cfg_scale_reference={speech.cfg_scale_reference} "
                f"cfg_applied={speech.cfg_applied}"
            )
            if speech.ready:
                print("Speech-token generation preflight: ready")
                return 0
            print("Speech-token generation preflight: not ready")
            return 2
        if args.preflight_speech_decoder:
            if preflight.decoder_path is None:
                print("Speech decoder preflight: not ready")
                print("Missing: decoder path")
                return 2
            decoder = run_speech_decoder_smoke(decoder_path=preflight.decoder_path)
            print(
                "Speech decoder: "
                f"backend={decoder.backend} "
                f"device={decoder.device} "
                f"frame_count={decoder.frame_count} "
                f"input_shape={decoder.input_shape} "
                f"vq_embedding_shape={decoder.vq_embedding_shape} "
                f"waveform_shape={decoder.waveform_shape} "
                f"waveform_dtype={decoder.waveform_dtype} "
                f"sample_rate={decoder.sample_rate} "
                f"hop_length={decoder.hop_length} "
                f"lookahead_steps={decoder.lookahead_steps} "
                f"finite={decoder.finite} "
                f"peak_abs={decoder.peak_abs:.6f}"
            )
            if decoder.ready:
                print("Speech decoder preflight: ready")
                return 0
            print("Speech decoder preflight: not ready")
            return 2
        if args.preflight_speech_output:
            if preflight.decoder_path is None:
                print("Speech output preflight: not ready")
                print("Missing: decoder path")
                return 2
            output = run_speech_output_smoke(
                full_model_path=preflight.model_path,
                decoder_path=preflight.decoder_path,
            )
            print(
                "Speech output: "
                f"backend={output.backend} "
                f"device={output.device} "
                f"prompt_tokens={output.prompt_tokens} "
                f"generated_token_ids={output.generated_token_ids} "
                f"generated_codec_frames={output.generated_codec_frames} "
                f"waveform_shape={output.waveform_shape} "
                f"sample_rate={output.sample_rate} "
                f"hop_length={output.hop_length} "
                f"finite={output.finite} "
                f"peak_abs={output.peak_abs:.6f}"
            )
            print(f"Speech output WAV: {output.wav_path}")
            print(f"Speech output run log: {output.run_log_path}")
            if output.ready:
                print("Speech output preflight: ready")
                return 0
            print("Speech output preflight: not ready")
            return 2
        if preflight.ready:
            print("Audio runtime preflight: ready")
            return 0
        print("Audio runtime preflight: not ready")
        for item in preflight.missing_items:
            print(f"Missing: {item}")
        return 2

    if args.run_text_benchmark:
        print(f"Text backend: {args.text_backend}")
        run = run_text_benchmark(
            selected_model,
            thinking_enabled=args.thinking,
            limit_turns=args.limit_text_turns,
            backend=args.text_backend,
        )
        print(f"Text benchmark run log: {run.run_log_path}")
        if run.transcript:
            print("Last assistant turn:")
            print(run.transcript[-1]["assistant"])
        assessment = run.assessment
        if assessment.full_benchmark_evaluated:
            print(
                "Text runtime compatibility: "
                + ("passed" if assessment.compatible else "failed")
            )
        else:
            print("Text runtime compatibility: partial limited-turn diagnostic")
        for observation in assessment.quality_observations:
            if observation.satisfied:
                continue
            print(
                f"Model quality observation: {observation.name}: {observation.detail}"
            )
        if assessment.compatible:
            return 0
        for failure in assessment.compatibility_failures:
            print(f"Compatibility failure: {failure}")
        return 2

    preflight = preflight_audio_runtime(selected_model)
    if (
        not preflight.ready
        or preflight.model_path is None
        or preflight.decoder_path is None
    ):
        _print_audio_preflight(preflight)
        print("Speech-to-speech CLI: not ready")
        for item in preflight.missing_items:
            print(f"Missing: {item}")
        return 2
    if args.diagnose_vllm_tts_quality_corpus is not None:
        manifest_path = run_vllm_tts_quality_probe(
            full_model_path=preflight.model_path,
            decoder_path=preflight.decoder_path,
            corpus_path=args.diagnose_vllm_tts_quality_corpus,
            recipe=tts_quality_recipe(
                args.diagnose_vllm_tts_quality_recipe,
                seed=args.diagnose_vllm_tts_quality_seed,
            ),
            selected_model_repo=selected_model.repo_id,
            speech_max_tokens=args.speech_max_tokens,
        )
        print(f"vLLM TTS quality manifest: {manifest_path}")
        return 0
    if (
        args.diagnose_vllm_tts_text is not None
        or args.diagnose_vllm_tts_text_file is not None
    ):
        text = (
            args.diagnose_vllm_tts_text_file.read_text(encoding="utf-8")
            if args.diagnose_vllm_tts_text_file is not None
            else str(args.diagnose_vllm_tts_text)
        )
        speech = run_vllm_tts_text_probe(
            full_model_path=preflight.model_path,
            decoder_path=preflight.decoder_path,
            text=text,
            selected_model_repo=selected_model.repo_id,
            play=not args.no_play,
            speech_max_tokens=args.speech_max_tokens,
        )
        print(f"vLLM TTS text probe WAV: {speech.wav_path}")
        print(f"vLLM TTS text probe run log: {speech.run_log_path}")
        run_log = json.loads(speech.run_log_path.read_text(encoding="utf-8"))
        print(
            "vLLM TTS text probe: "
            f"codec_frames={run_log.get('generated_codec_frame_count', len(speech.generated_codec_frames))} "
            f"codec_fps={run_log.get('codec_frames_per_second')} "
            f"first_audio_ready_seconds={run_log.get('first_audio_ready_seconds', speech.first_audio_ready_seconds)} "
            f"hit_max_tokens={run_log.get('hit_max_tokens', speech.hit_max_tokens)}"
        )
        return 0
    persona = load_persona(args.persona)
    conversation_store = ConversationStore()
    if args.new_conversation:
        conversation = conversation_store.create(
            persona_id=persona.persona_id,
            persona_path=persona.path,
            system_prompt=persona.system_prompt,
            max_context_tokens=args.max_context_tokens,
        )
        resumed_conversation = False
    elif args.conversation_id is not None:
        conversation = conversation_store.load(args.conversation_id)
        if conversation.max_context_tokens != args.max_context_tokens:
            conversation.max_context_tokens = args.max_context_tokens
            conversation_store.save(conversation)
        conversation_store.set_current(conversation.conversation_id)
        resumed_conversation = True
    else:
        conversation, resumed_conversation = (
            conversation_store.resume_current_or_create(
                persona_id=persona.persona_id,
                persona_path=persona.path,
                system_prompt=persona.system_prompt,
                max_context_tokens=args.max_context_tokens,
            )
        )
    print(
        "Conversation: "
        f"{conversation.conversation_id} "
        f"({'resumed' if resumed_conversation else 'new'}, "
        f"persona={conversation.persona_id})"
    )
    print(f"Conversation transcript: {conversation.transcript_path}")

    if args.input_wav is not None:
        if args.sts_backend == "vllm":
            turn = run_vllm_fixture_turn(
                full_model_path=preflight.model_path,
                decoder_path=preflight.decoder_path,
                input_wav_path=args.input_wav,
                selected_model_repo=selected_model.repo_id,
                play=not args.no_play,
                response_max_tokens=args.response_max_tokens,
                speech_max_tokens=args.speech_max_tokens,
                thinking_enabled=args.thinking,
                conversation=conversation,
                conversation_store=conversation_store,
                persona=persona,
            )
        else:
            print(
                "Audex STS: using direct MLX diagnostic fallback.",
                flush=True,
            )
            print("Audex STS: loading persistent MLX session...", flush=True)
            session = AudexSpeechToSpeechSession(
                full_model_path=preflight.model_path,
                decoder_path=preflight.decoder_path,
                selected_model_repo=selected_model.repo_id,
                thinking_enabled=args.thinking,
                response_max_tokens=args.response_max_tokens,
                speech_max_tokens=args.speech_max_tokens,
                conversation=conversation,
                conversation_store=conversation_store,
                persona=persona,
                enable_kv_cache=not args.no_kv_cache,
            )
            stats = session.stats
            print(
                "Audex STS: session ready "
                f"(model={stats.model_load_seconds:.3f}s, "
                f"audio={stats.audio_component_load_seconds:.3f}s, "
                f"decoder={stats.decoder_load_seconds:.3f}s, "
                f"speech_warmup={stats.speech_warmup_seconds:.3f}s).",
                flush=True,
            )
            turn = run_fixture_turn(
                full_model_path=preflight.model_path,
                decoder_path=preflight.decoder_path,
                input_wav_path=args.input_wav,
                selected_model_repo=selected_model.repo_id,
                play=not args.no_play,
                response_max_tokens=args.response_max_tokens,
                speech_max_tokens=args.speech_max_tokens,
                thinking_enabled=args.thinking,
                session=session,
            )
    else:
        if args.sts_backend == "vllm":
            run_vllm_interactive_ptt(
                full_model_path=preflight.model_path,
                decoder_path=preflight.decoder_path,
                selected_model_repo=selected_model.repo_id,
                play=not args.no_play,
                response_max_tokens=args.response_max_tokens,
                speech_max_tokens=args.speech_max_tokens,
                thinking_enabled=args.thinking,
                conversation=conversation,
                conversation_store=conversation_store,
                persona=persona,
                conversation_resumed=resumed_conversation,
            )
        else:
            print(
                "Audex STS: using direct MLX diagnostic fallback.",
                flush=True,
            )
            run_interactive_ptt(
                full_model_path=preflight.model_path,
                decoder_path=preflight.decoder_path,
                selected_model_repo=selected_model.repo_id,
                play=not args.no_play,
                response_max_tokens=args.response_max_tokens,
                speech_max_tokens=args.speech_max_tokens,
                thinking_enabled=args.thinking,
                conversation=conversation,
                conversation_store=conversation_store,
                persona=persona,
                enable_kv_cache=not args.no_kv_cache,
                conversation_resumed=resumed_conversation,
            )
        return 0
    print(f"Transcript: {turn.transcript}")
    print(f"Response: {turn.response_text}")
    print(f"Speech output WAV: {turn.output_wav_path}")
    print(f"Speech-to-speech run log: {turn.run_log_path}")
    return 0


def _text_command(args: argparse.Namespace) -> bool:
    return bool(
        args.preflight_text_runtime
        or args.run_text_benchmark
        or args.diagnose_vllm_metal
        or args.diagnose_vllm_generation
    )


def _print_audio_preflight(preflight) -> None:
    print(f"Audio model path: {preflight.model_path}")
    if preflight.audio_components is not None:
        components = preflight.audio_components
        print(
            "Audio components: "
            f"architecture={','.join(components.architecture)} "
            f"model_type={components.model_type} "
            f"audio_model_type={components.audio_model_type}"
        )
        print(
            "Audio encoder/projector: "
            f"layers={components.audio_encoder_layers} "
            f"hidden={components.audio_encoder_hidden_size} "
            f"mel_bins={components.audio_mel_bins} "
            f"max_source_positions={components.audio_max_source_positions} "
            f"audio_weights={components.audio_encoder_weight_count} "
            f"shards={','.join(components.audio_weight_shards)}"
        )
        print(
            "Audio tokens: "
            f"sound_token_id={components.sound_token_id} "
            f"embeddings_per_clip={components.sound_embeddings_per_clip}"
        )
    if preflight.audio_preprocessor is not None:
        preprocessor = preflight.audio_preprocessor
        print(
            "Audio preprocessor: "
            f"type={preprocessor.feature_extractor_type} "
            f"feature_size={preprocessor.feature_size} "
            f"n_samples={preprocessor.n_samples} "
            f"nb_max_frames={preprocessor.nb_max_frames} "
            f"sampling_rate={preprocessor.sampling_rate}"
        )
    print(f"Audex decoder path: {preflight.decoder_path}")
    if preflight.decoder is not None:
        print(
            "Audex decoder config: "
            f"sample_rate={preflight.decoder.sample_rate} "
            f"lookahead_steps={preflight.decoder.lookahead_steps} "
            f"codebook_size={preflight.decoder.codebook_size}"
        )
    if preflight.speech_tokenizer is not None:
        print(
            "Speech tokenizer: "
            f"speechgen_start={preflight.speech_tokenizer.speechgen_start} "
            f"speechgen_end={preflight.speech_tokenizer.speechgen_end} "
            f"codec_tokens={preflight.speech_tokenizer.codec_token_count}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
