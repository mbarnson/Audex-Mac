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
    TtaOutputInspection,
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


class AudexVllmUnderstandingAdapter:
    """Answer one non-speech audio-understanding case using an Audex vLLM runtime."""

    def __init__(self, *, runtime: Any) -> None:
        self._runtime = runtime

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
        result = _run_async(self._runtime.generate_one_final(request))
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
        decode_to_wav: Callable[[TtaOutputInspection, Path, AudioEvaluationCase], None],
    ) -> None:
        self._runtime = runtime
        self._raw_dir = raw_dir
        self._decode_to_wav = decode_to_wav

    def generate(
        self,
        case: AudioEvaluationCase,
        *,
        seed: int,
    ) -> GenerationAttempt:
        caption = case.caption or case.prompt
        cond, uncond = build_tta_requests(
            self._runtime.tokenizer,
            caption=caption,
            case_id=case.case_id,
            seed=seed,
        )
        cond_result, _uncond_result = _run_async(
            self._runtime.generate_many_final((cond, uncond))
        )
        inspection = inspect_tta_output(
            self._runtime.tokenizer,
            cond_result.token_ids,
        )
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        raw_wav_path = self._raw_dir / f"{case.case_id}.wav"
        if inspection.valid:
            self._decode_to_wav(inspection, raw_wav_path, case)
        else:
            raw_wav_path.write_bytes(b"")
        return GenerationAttempt(
            raw_wav_path=raw_wav_path,
            enhanced_wav_path=None,
            structure=inspection,
            signal_metrics=_signal_metrics(raw_wav_path),
            elapsed_seconds=cond_result.elapsed_seconds,
            finish_reason=cond_result.finish_reason,
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
    metrics.update(
        {
            "finite": finite,
            "sample_rate": loaded.sample_rate,
            "channels": loaded.channels,
            "duration_seconds": (
                len(values) / loaded.sample_rate if loaded.sample_rate else 0.0
            ),
            "peak": peak,
            "clipped": peak >= 0.999,
        }
    )
    return metrics


def _run_async(awaitable: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError(
        "audio evaluation adapters are synchronous; call them outside an active "
        f"event loop {loop!r}"
    )
