"""Concrete model adapters for autonomous Audex audio evaluation."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from .audio_evaluation import AudioEvaluationCase
from .audio_evaluation_generation import (
    NVIDIA_TTA_RECIPE,
    TtaOutputInspection,
    TtaRecipe,
    build_tta_requests,
    inspect_tta_output,
)
from .audio_evaluation_runner import GenerationAttempt, UnderstandingAttempt
from .audio_pcm import load_wav_pcm
from .vllm_sts_requests import build_audio_messages_response_request

UNDERSTANDING_TEMPERATURE = 0.7
UNDERSTANDING_TOP_P = 0.9
UNDERSTANDING_MAX_TOKENS = 32
UNDERSTANDING_PROMPT_TEXT = (
    "Listen to the attached non-speech audio and answer the previous question. "
    "Return only the requested constrained answer."
)
_EVAL_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None


class AudexVllmUnderstandingAdapter:
    """Answer one non-speech audio-understanding case using an Audex vLLM runtime."""

    def __init__(
        self,
        *,
        runtime: Any,
        run_sync: Callable[[Any], Any] | None = None,
    ) -> None:
        self._runtime = runtime
        self._run_sync = run_sync or _run_async

    def answer(
        self,
        case: AudioEvaluationCase,
        *,
        seed: int,
    ) -> UnderstandingAttempt:
        if case.audio_path is None:
            raise ValueError(f"understanding case has no audio_path: {case.case_id}")
        loaded = load_wav_pcm(Path(case.audio_path))
        if loaded.sample_rate != 16_000:
            raise ValueError(
                f"understanding case audio must be 16000 Hz: {case.audio_path}"
            )
        request = build_audio_messages_response_request(
            self._runtime.tokenizer,
            [{"role": "user", "content": case.prompt}],
            loaded.samples,
            sample_rate=loaded.sample_rate,
            prompt_text=UNDERSTANDING_PROMPT_TEXT,
            enable_reasoning=False,
            max_tokens=UNDERSTANDING_MAX_TOKENS,
            trim_padded_audio_embeddings=True,
        )
        request = replace(
            request,
            sampling=replace(
                request.sampling,
                temperature=UNDERSTANDING_TEMPERATURE,
                top_p=UNDERSTANDING_TOP_P,
                seed=seed,
            ),
        )
        result = self._run_sync(self._runtime.generate_one_final(request))
        return UnderstandingAttempt(
            raw_answer=result.text.strip(),
            elapsed_seconds=result.elapsed_seconds,
            finish_reason=result.finish_reason,
        )


class AudexVllmTtaGenerationAdapter:
    """Generate one text-to-audio case and decode it through an injected decoder."""

    def __init__(
        self,
        *,
        runtime: Any,
        raw_dir: Path,
        enhanced_dir: Path | None = None,
        decode_to_wav: Callable[[TtaOutputInspection, Path, AudioEvaluationCase], None],
        enhance_wav: Callable[[Path, Path, AudioEvaluationCase], None] | None = None,
        allow_nvidia_reference_output: bool = False,
        recipe: TtaRecipe = NVIDIA_TTA_RECIPE,
        run_sync: Callable[[Any], Any] | None = None,
    ) -> None:
        self._runtime = runtime
        self._raw_dir = raw_dir
        self._enhanced_dir = enhanced_dir or (raw_dir.parent / "enhanced")
        self._decode_to_wav = decode_to_wav
        self._enhance_wav = enhance_wav
        self._allow_nvidia_reference_output = allow_nvidia_reference_output
        self._recipe = recipe
        self._run_sync = run_sync or _run_async

    def generate(
        self,
        case: AudioEvaluationCase,
        *,
        seed: int,
    ) -> GenerationAttempt:
        return self.generate_many(((case, seed),))[0]

    def generate_many(
        self,
        cases: tuple[tuple[AudioEvaluationCase, int], ...],
    ) -> tuple[GenerationAttempt, ...]:
        """Submit every CFG pair together, then decode attempts in case order."""

        if not cases:
            return ()
        requests: list[Any] = []
        for case, seed in cases:
            caption = case.caption or case.prompt
            requests.extend(
                build_tta_requests(
                    self._runtime.tokenizer,
                    caption=caption,
                    case_id=case.case_id,
                    seed=seed,
                    recipe=self._recipe,
                )
            )
        results = self._run_sync(self._runtime.generate_many_final(tuple(requests)))
        attempts: list[GenerationAttempt] = []
        for index, (case, _seed) in enumerate(cases):
            attempts.append(self._decode_attempt(case, results[index * 2]))
        return tuple(attempts)

    def _decode_attempt(
        self, case: AudioEvaluationCase, cond_result: Any
    ) -> GenerationAttempt:
        inspection = inspect_tta_output(
            self._runtime.tokenizer,
            cond_result.token_ids,
            recipe=self._recipe,
        )
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        raw_wav_path = self._raw_dir / f"{case.case_id}.wav"
        enhanced_wav_path: Path | None = None
        should_decode = inspection.valid or (
            self._allow_nvidia_reference_output
            and inspection.nvidia_reference_decodable
        )
        if should_decode:
            self._decode_to_wav(inspection, raw_wav_path, case)
            if self._enhance_wav is not None:
                enhanced_wav_path = self._enhanced_dir / f"{case.case_id}.wav"
                enhanced_wav_path.parent.mkdir(parents=True, exist_ok=True)
                self._enhance_wav(raw_wav_path, enhanced_wav_path, case)
        else:
            raw_wav_path.write_bytes(b"")
        return GenerationAttempt(
            raw_wav_path=raw_wav_path,
            enhanced_wav_path=enhanced_wav_path,
            structure=inspection,
            signal_metrics=_signal_metrics(raw_wav_path),
            elapsed_seconds=cond_result.elapsed_seconds,
            finish_reason=cond_result.finish_reason,
        )


def build_nvidia_tta_generation_adapter(
    *,
    runtime: Any,
    raw_dir: Path,
    decode_to_wav: Callable[[TtaOutputInspection, Path, AudioEvaluationCase], None],
    enhanced_dir: Path | None = None,
    enhance_wav: Callable[[Path, Path, AudioEvaluationCase], None] | None = None,
    run_sync: Callable[[Any], Any] | None = None,
) -> AudexVllmTtaGenerationAdapter:
    """Build the one supported production TTA adapter preset."""

    return AudexVllmTtaGenerationAdapter(
        runtime=runtime,
        raw_dir=raw_dir,
        enhanced_dir=enhanced_dir,
        decode_to_wav=decode_to_wav,
        enhance_wav=enhance_wav,
        run_sync=run_sync,
        recipe=NVIDIA_TTA_RECIPE,
        allow_nvidia_reference_output=True,
    )


def _signal_metrics(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {
            "finite": False,
            "nonempty": False,
            "file_bytes": 0,
        }
    metrics: dict[str, Any] = {
        "finite": True,
        "nonempty": True,
        "file_bytes": path.stat().st_size,
    }
    try:
        loaded = load_wav_pcm(path)
    except Exception:
        return metrics
    values = tuple(float(sample) for sample in loaded.samples)
    finite = all(math.isfinite(sample) for sample in values)
    peak = max((abs(sample) for sample in values), default=0.0)
    rms = (
        math.sqrt(sum(sample * sample for sample in values) / len(values))
        if values
        else 0.0
    )
    dc_offset = sum(values) / len(values) if values else 0.0
    sample_delta_peak = max(
        (
            abs(current - previous)
            for previous, current in zip(values, values[1:], strict=False)
        ),
        default=0.0,
    )
    zero_crossings = sum(
        1
        for previous, current in zip(values, values[1:], strict=False)
        if (previous < 0.0 < current) or (previous > 0.0 > current)
    )
    zero_crossing_rate = zero_crossings / (len(values) - 1) if len(values) > 1 else 0.0
    metrics.update(
        {
            "finite": finite,
            "sample_rate": loaded.sample_rate,
            "channels": loaded.channels,
            "duration_seconds": (
                len(values) / loaded.sample_rate if loaded.sample_rate else 0.0
            ),
            "peak": peak,
            "rms": rms,
            "dc_offset": dc_offset,
            "sample_delta_peak": sample_delta_peak,
            "zero_crossing_rate": zero_crossing_rate,
            "clipped": peak >= 0.999,
        }
    )
    return metrics


def _run_async(awaitable: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        global _EVAL_ASYNC_LOOP
        if _EVAL_ASYNC_LOOP is None or _EVAL_ASYNC_LOOP.is_closed():
            _EVAL_ASYNC_LOOP = asyncio.new_event_loop()
        return _EVAL_ASYNC_LOOP.run_until_complete(awaitable)
    raise RuntimeError(
        "audio evaluation adapters are synchronous; call them outside an active "
        f"event loop {loop!r}"
    )


def run_sync_model_call(awaitable: Any) -> Any:
    """Run one model awaitable on the shared synchronous adapter loop."""

    return _run_async(awaitable)
