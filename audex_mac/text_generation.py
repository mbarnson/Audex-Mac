"""Text-generation runners for the Audex text benchmark."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .metal_policy import inspect_metal_runtime
from .models import AudexModel
from .patches import apply_audex_runtime_patches
from .text_benchmark import TextBenchmark
from .text_chat import (
    CHAT_STOP_MARKERS,
    clean_text_completion,
    complete_text_assistant_turn,
    render_text_chat_prompt,
)
from .text_gate import TextBenchmarkAssessment, evaluate_text_benchmark
from .text_runtime import TextBackend, TextRuntimePreflight, preflight_text_runtime

STOP_MARKERS = CHAT_STOP_MARKERS
RUNS_DIR = Path(__file__).resolve().parents[1] / ".audex" / "runs"


@dataclass(frozen=True, slots=True)
class TextBenchmarkRun:
    run_log_path: Path
    transcript: list[dict[str, Any]]
    assessment: TextBenchmarkAssessment


def run_text_benchmark(
    model: AudexModel,
    *,
    thinking_enabled: bool = False,
    limit_turns: int | None = None,
    backend: TextBackend = "vllm",
) -> TextBenchmarkRun:
    preflight = preflight_text_runtime(model, backend=backend)
    if not preflight.ready:
        raise RuntimeError(
            "Text runtime is not ready: " + ", ".join(preflight.missing_items)
        )
    if preflight.model_path is None:
        raise RuntimeError("Text runtime is missing a model path")

    return _run_text_benchmark_from_preflight(
        preflight,
        thinking_enabled=thinking_enabled,
        limit_turns=limit_turns,
        backend=backend,
    )


def _run_text_benchmark_from_preflight(
    preflight: TextRuntimePreflight,
    *,
    thinking_enabled: bool,
    limit_turns: int | None,
    backend: TextBackend,
) -> TextBenchmarkRun:
    metal_policy = inspect_metal_runtime()
    if not metal_policy.ready:
        raise RuntimeError(f"Metal/MLX runtime policy is not ready: {metal_policy}")
    apply_audex_runtime_patches()

    if backend == "mlx":
        return _run_text_benchmark_mlx(
            preflight,
            metal_policy=metal_policy,
            thinking_enabled=thinking_enabled,
            limit_turns=limit_turns,
        )
    if backend == "vllm":
        return _run_text_benchmark_vllm(
            preflight,
            metal_policy=metal_policy,
            thinking_enabled=thinking_enabled,
            limit_turns=limit_turns,
        )
    raise ValueError(f"Unsupported text backend: {backend}")


def _run_text_benchmark_mlx(
    preflight: TextRuntimePreflight,
    *,
    metal_policy: Any,
    thinking_enabled: bool,
    limit_turns: int | None,
) -> TextBenchmarkRun:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    benchmark = preflight.benchmark
    turns = benchmark.turns if limit_turns is None else benchmark.turns[:limit_turns]
    sampler_config = _sampler_config(benchmark)
    sampler = make_sampler(
        temp=sampler_config["temperature"],
        top_p=sampler_config["top_p"],
    )
    load_started_at = time.time()
    model, tokenizer = load(
        str(preflight.model_path),
        tokenizer_config={"trust_remote_code": True},
    )
    model_load_seconds = round(time.time() - load_started_at, 3)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _benchmark_system_prompt(benchmark)}
    ]
    transcript: list[dict[str, Any]] = []
    started_at = time.time()

    for index, turn in enumerate(turns, start=1):
        user_text = turn["content"]
        messages.append({"role": "user", "content": user_text})
        prompt = render_text_chat_prompt(
            tokenizer,
            messages,
            model_path=preflight.model_path,
            thinking_enabled=thinking_enabled,
        )
        mx.random.seed(sampler_config["seed"])
        turn_start = time.time()
        generated = ""
        last_response = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=sampler_config["max_tokens"],
            sampler=sampler,
        ):
            generated += response.text
            last_response = response
            if _contains_stop_marker(generated):
                break
        assistant_turn = complete_text_assistant_turn(
            generated,
            thinking_enabled=thinking_enabled,
        )
        assistant_text = assistant_turn.answer
        messages.append({"role": "assistant", "content": assistant_turn.raw_content})
        transcript.append(
            {
                "turn": index,
                "user": user_text,
                "assistant": assistant_text,
                "assistant_raw": assistant_turn.raw_content,
                "elapsed_seconds": round(time.time() - turn_start, 3),
                "prompt_tokens": getattr(last_response, "prompt_tokens", None),
                "prompt_tps": _round_optional(
                    getattr(last_response, "prompt_tps", None)
                ),
                "generation_tokens": getattr(last_response, "generation_tokens", None),
                "generation_tps": _round_optional(
                    getattr(last_response, "generation_tps", None)
                ),
                "finish_reason": getattr(last_response, "finish_reason", None),
                "peak_memory_gb": _round_optional(
                    getattr(last_response, "peak_memory", None)
                ),
            }
        )

    run_log = _build_run_log(
        preflight,
        backend="mlx",
        benchmark=benchmark,
        sampler=sampler_config,
        thinking_enabled=thinking_enabled,
        metal_policy=metal_policy,
        started_at=started_at,
        transcript=transcript,
        extra={"model_load_seconds": model_load_seconds},
    )
    assessment = _evaluate_assessment(
        benchmark,
        transcript,
        limit_turns=limit_turns,
    )
    _record_assessment(run_log, assessment)
    run_log_path = _write_run_log(run_log)
    return TextBenchmarkRun(
        run_log_path=run_log_path,
        transcript=transcript,
        assessment=assessment,
    )


def _run_text_benchmark_vllm(
    preflight: TextRuntimePreflight,
    *,
    metal_policy: Any,
    thinking_enabled: bool,
    limit_turns: int | None,
) -> TextBenchmarkRun:
    from vllm import LLM, SamplingParams

    benchmark = preflight.benchmark
    turns = benchmark.turns if limit_turns is None else benchmark.turns[:limit_turns]
    sampler_config = _sampler_config(benchmark)
    sampling_params = SamplingParams(
        temperature=sampler_config["temperature"],
        top_p=sampler_config["top_p"],
        max_tokens=sampler_config["max_tokens"],
        seed=sampler_config["seed"],
    )
    load_started_at = time.time()
    llm = LLM(
        str(preflight.model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()
    model_load_seconds = round(time.time() - load_started_at, 3)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _benchmark_system_prompt(benchmark)}
    ]
    transcript: list[dict[str, Any]] = []
    started_at = time.time()

    for index, turn in enumerate(turns, start=1):
        user_text = turn["content"]
        messages.append({"role": "user", "content": user_text})
        prompt = render_text_chat_prompt(
            tokenizer,
            messages,
            model_path=preflight.model_path,
            thinking_enabled=thinking_enabled,
        )
        turn_start = time.time()
        request_output = llm.generate([prompt], sampling_params)[0]
        elapsed_seconds = round(time.time() - turn_start, 3)
        output = request_output.outputs[0]
        assistant_turn = complete_text_assistant_turn(
            output.text,
            thinking_enabled=thinking_enabled,
        )
        assistant_text = assistant_turn.answer
        messages.append({"role": "assistant", "content": assistant_turn.raw_content})
        transcript.append(
            _vllm_turn_record(
                turn=index,
                user=user_text,
                assistant=assistant_text,
                assistant_raw=assistant_turn.raw_content,
                elapsed_seconds=elapsed_seconds,
                request_output=request_output,
                completion_output=output,
            )
        )

    run_log = _build_run_log(
        preflight,
        backend="vllm",
        benchmark=benchmark,
        sampler=sampler_config,
        thinking_enabled=thinking_enabled,
        metal_policy=metal_policy,
        started_at=started_at,
        transcript=transcript,
        extra={
            "engine": "vllm.LLM",
            "model_load_seconds": model_load_seconds,
        },
    )
    assessment = _evaluate_assessment(
        benchmark,
        transcript,
        limit_turns=limit_turns,
    )
    _record_assessment(run_log, assessment)
    run_log_path = _write_run_log(run_log)
    return TextBenchmarkRun(
        run_log_path=run_log_path,
        transcript=transcript,
        assessment=assessment,
    )


def clean_generation(text: str) -> str:
    return clean_text_completion(text)


def _contains_stop_marker(text: str) -> bool:
    return any(marker in text for marker in STOP_MARKERS)


def _round_optional(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def _vllm_turn_record(
    *,
    turn: int,
    user: str,
    assistant: str,
    assistant_raw: str | None = None,
    elapsed_seconds: float,
    request_output: Any,
    completion_output: Any,
) -> dict[str, Any]:
    token_ids = tuple(getattr(completion_output, "token_ids", ()) or ())
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    generated_tokens = len(token_ids)
    return {
        "turn": turn,
        "user": user,
        "assistant": assistant,
        "assistant_raw": assistant if assistant_raw is None else assistant_raw,
        "elapsed_seconds": elapsed_seconds,
        "prompt_tokens": (
            len(prompt_token_ids) if prompt_token_ids is not None else None
        ),
        "generation_tokens": generated_tokens,
        "generation_tps": (
            round(generated_tokens / elapsed_seconds, 3)
            if elapsed_seconds > 0
            else None
        ),
        "finish_reason": getattr(completion_output, "finish_reason", None),
        "stop_reason": getattr(completion_output, "stop_reason", None),
    }


def _sampler_config(benchmark: TextBenchmark) -> dict[str, Any]:
    return {
        "temperature": float(benchmark.generation["temperature"]),
        "top_p": float(benchmark.generation["top_p"]),
        "seed": int(benchmark.generation["seed"]),
        "max_tokens": int(benchmark.generation["max_tokens"]),
    }


def _build_run_log(
    preflight: TextRuntimePreflight,
    *,
    backend: TextBackend,
    benchmark: TextBenchmark,
    sampler: dict[str, Any],
    thinking_enabled: bool,
    metal_policy: Any,
    started_at: float,
    transcript: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_log = {
        "selected_model": preflight.model.repo_id,
        "model_path": str(preflight.model_path),
        "backend": backend,
        "benchmark": benchmark.name,
        "sampler": sampler,
        "thinking_enabled": thinking_enabled,
        "metal_runtime": {
            "env": metal_policy.env,
            "mlx_metal_available": metal_policy.mlx_metal_available,
            "mlx_default_device": metal_policy.mlx_default_device,
        },
        "turns": len(transcript),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "transcript": transcript,
    }
    if preflight.patch_report is not None:
        run_log["audex_patches"] = asdict(preflight.patch_report)
    if extra:
        run_log.update(extra)
    return run_log


def _write_run_log(run_log: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_log_path = RUNS_DIR / f"text-benchmark-{time.strftime('%Y%m%d-%H%M%S')}.json"
    run_log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
    return run_log_path


def _record_assessment(
    run_log: dict[str, Any],
    assessment: TextBenchmarkAssessment,
) -> None:
    run_log["text_compatibility"] = {
        "full_benchmark_evaluated": assessment.full_benchmark_evaluated,
        "compatible": assessment.compatible,
        "failures": list(assessment.compatibility_failures),
    }
    run_log["text_quality_observations"] = [
        asdict(observation) for observation in assessment.quality_observations
    ]
    run_log["text_evaluation_policy"] = {
        "exact_token_parity_required": assessment.exact_token_parity_required,
        "logit_parity_required": assessment.logit_parity_required,
        "quality_observations_are_blocking": False,
    }


def _evaluate_assessment(
    benchmark: TextBenchmark,
    transcript: list[dict[str, Any]],
    *,
    limit_turns: int | None,
) -> TextBenchmarkAssessment:
    if limit_turns is not None:
        failures = (
            ("every observed benchmark turn must produce a non-empty answer",)
            if any(not str(turn.get("assistant", "")).strip() for turn in transcript)
            else ()
        )
        return TextBenchmarkAssessment(
            compatibility_failures=failures,
            quality_observations=(),
            full_benchmark_evaluated=False,
        )
    return evaluate_text_benchmark(benchmark, transcript)


def _benchmark_system_prompt(benchmark: TextBenchmark) -> str:
    return benchmark.system
