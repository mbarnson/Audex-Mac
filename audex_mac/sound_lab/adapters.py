"""Audex model adapters for Sound Lab planning, design, and generation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..audio_evaluation import AudioEvaluationCase, EvaluationTrack
from ..audio_evaluation_adapters import (
    AudexVllmTtaGenerationAdapter,
    run_sync_model_call,
)
from ..vllm_sts_requests import build_text_messages_response_request
from .session import (
    GeneratedSound,
    SoundGenerationAttempt,
    SoundGenerationOutcome,
    SoundGenerationRequest,
    VariantBrief,
    VariantDesignError,
    VariantDesignResult,
)
from .tools import RenderSoundsCall, parse_sound_lab_tool_call

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PLANNER_MAX_TOKENS = 768
_DESIGNER_MAX_TOKENS = 1536
NVIDIA_CFG_PAIRS_PER_BATCH = 2
_RETRY_SEED_XOR = 0x5A17_D3C9

_RENDER_SOUNDS_TOOL = {
    "type": "function",
    "function": {
        "name": "render_sounds",
        "description": (
            "Queue one to five non-speech sounds for blind audition. Use this "
            "whenever the user asks to create, generate, make, or audition sounds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "brief": {
                    "type": "string",
                    "description": "The complete creative sound request.",
                },
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Number of blind candidates requested.",
                },
                "constraints": {
                    "type": "object",
                    "description": "Explicit acoustic or content constraints.",
                },
                "parent_asset_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Prior opaque assets to use as design lineage.",
                },
            },
            "required": ["brief", "count"],
        },
    },
}

_PLANNER_SYSTEM_PROMPT = """You are the Audex Sound Lab orchestrator.
For a request to create, generate, make, vary, or audition non-speech audio, call
render_sounds. Preserve the user's creative intent in the brief. Infer a count
from the request, defaulting to 3 and never exceeding 5. Put only explicit user
constraints in constraints. Use parent_asset_ids only when opaque asset IDs were
provided. You may answer ordinary questions normally. Never claim that a render
has completed because the tool runs after your response."""

_DESIGNER_SYSTEM_PROMPT = """You are Audex Sound Lab's sound designer.
Expand one creative request into the exact requested number of genuinely
different AudioCaps-style captions. Each caption is one literal present-tense
sentence of 3 to 24 words describing only what is audible: source, action,
environment, distance, and simple timing when relevant. Vary meaningful
acoustic dimensions while preserving explicit constraints. Never write an
instruction, production request, quality claim, negative prompt, rationale, or
phrases such as create, generate, sound effect, cinematic, or high-quality.
Return only strict JSON with this shape:
{"variants":[{"caption":"...","difference":"..."}]}
Do not use markdown."""

_DESIGNER_REPAIR_PROMPT = """Your previous response could not be validated.
Return a corrected response containing exactly the requested number of variants.
Return only one JSON object with a variants array. Each variant requires a
nonempty caption and difference. Do not use Markdown or add commentary."""


class AudexSoundLabPlanner:
    """Turn one user utterance into a validated Sound Lab tool call or reply."""

    def __init__(self, *, runtime: Any) -> None:
        self._runtime = runtime

    def plan(self, user_text: str) -> RenderSoundsCall | str:
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_text.strip()},
        ]
        base_request = build_text_messages_response_request(
            self._runtime.tokenizer,
            messages,
            prompt_text="",
            enable_reasoning=False,
            max_tokens=_PLANNER_MAX_TOKENS,
        )
        prompt = self._runtime.tokenizer.apply_chat_template(
            messages,
            tools=[_RENDER_SOUNDS_TOOL],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        request = replace(base_request, prompt=prompt, debug_name="sound-lab-tool")
        result = run_sync_model_call(self._runtime.generate_one_final(request))
        raw = result.text.strip()
        if "<tool_call>" in raw:
            return parse_sound_lab_tool_call(raw)
        reply = _THINK_RE.sub("", raw).strip()
        if not reply:
            raise ValueError("Audex Sound Lab planner returned no response")
        return reply


class AudexVariantDesigner:
    """Expand a render tool call into strict, distinct generation captions."""

    def __init__(self, *, runtime: Any) -> None:
        self._runtime = runtime

    def design(
        self,
        call: RenderSoundsCall,
        *,
        job_id: str,
    ) -> VariantDesignResult:
        payload = {
            "job_id": job_id,
            "brief": call.brief,
            "count": call.count,
            "constraints": call.constraints,
            "parent_asset_ids": list(call.parent_asset_ids),
        }
        messages = [
            {"role": "system", "content": _DESIGNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=True, sort_keys=True),
            },
        ]
        first = self._generate(messages, debug_name="sound-lab-design")
        try:
            variants = _parse_variants(
                first,
                expected_count=call.count,
                job_id=job_id,
            )
        except ValueError as first_error:
            repair_messages = [
                *messages,
                {"role": "assistant", "content": first},
                {
                    "role": "user",
                    "content": f"{_DESIGNER_REPAIR_PROMPT}\nValidation error: {first_error}",
                },
            ]
            try:
                second = self._generate(
                    repair_messages,
                    debug_name="sound-lab-design-repair",
                )
            except Exception as repair_error:
                raise VariantDesignError(
                    (
                        str(first_error),
                        "Sound Lab designer repair request failed: "
                        f"{type(repair_error).__name__}: {repair_error}",
                    ),
                    (first,),
                ) from repair_error
            try:
                variants = _parse_variants(
                    second,
                    expected_count=call.count,
                    job_id=job_id,
                )
            except ValueError as second_error:
                raise VariantDesignError(
                    (str(first_error), str(second_error)),
                    (first, second),
                ) from second_error
            return VariantDesignResult(
                variants=variants,
                raw_attempts=(first, second),
                repair_used=True,
            )
        return VariantDesignResult(variants=variants, raw_attempts=(first,))

    def _generate(self, messages: list[dict[str, str]], *, debug_name: str) -> str:
        request = build_text_messages_response_request(
            self._runtime.tokenizer,
            messages,
            prompt_text="",
            enable_reasoning=False,
            max_tokens=_DESIGNER_MAX_TOKENS,
        )
        request = replace(request, debug_name=debug_name)
        result = run_sync_model_call(self._runtime.generate_one_final(request))
        return result.text


class AudexTtaSoundGenerator:
    """Generate one CFG3 candidate through the complete XCodec WAV path."""

    def __init__(
        self,
        *,
        runtime: Any,
        decode_to_wav: Any,
        enhance_wav: Any | None = None,
        adapter_factory: Callable[..., AudexVllmTtaGenerationAdapter] = (
            AudexVllmTtaGenerationAdapter
        ),
    ) -> None:
        self._runtime = runtime
        self._decode_to_wav = decode_to_wav
        self._enhance_wav = enhance_wav
        self._adapter_factory = adapter_factory

    def generate_many(
        self,
        requests: tuple[SoundGenerationRequest, ...],
        *,
        output_dir: Path,
    ) -> Iterator[SoundGenerationOutcome]:
        for start in range(0, len(requests), NVIDIA_CFG_PAIRS_PER_BATCH):
            yield from self._generate_wave(
                requests[start : start + NVIDIA_CFG_PAIRS_PER_BATCH],
                output_dir=output_dir,
            )

    def _generate_wave(
        self,
        requests: tuple[SoundGenerationRequest, ...],
        *,
        output_dir: Path,
    ) -> Iterator[SoundGenerationOutcome]:
        cases = tuple(
            (
                _generation_case(request.asset_id, request.variant),
                request.variant.seed,
            )
            for request in requests
        )
        adapter = self._adapter_factory(
            runtime=self._runtime,
            raw_dir=output_dir / "raw",
            enhanced_dir=output_dir / "enhanced",
            decode_to_wav=self._decode_to_wav,
            enhance_wav=self._enhance_wav,
            allow_nvidia_reference_output=True,
        )
        attempts = adapter.generate_many(cases)
        retry_indexes: list[int] = []
        for index, (request, attempt) in enumerate(
            zip(requests, attempts, strict=True)
        ):
            if not _sound_lab_attempt_usable(attempt):
                retry_indexes.append(index)
                continue
            yield _sound_lab_success(
                request,
                attempt,
                seed_used=cases[index][1],
                elapsed_seconds=attempt.elapsed_seconds,
                attempts=(_attempt_record(attempt, seed=cases[index][1]),),
            )

        if not retry_indexes:
            return
        retry_cases = tuple(
            (
                cases[index][0],
                cases[index][1] ^ _RETRY_SEED_XOR,
            )
            for index in retry_indexes
        )
        try:
            retry_attempts = adapter.generate_many(retry_cases)
        except Exception as exc:
            for index in retry_indexes:
                request = requests[index]
                initial_seed = cases[index][1]
                retry_seed = initial_seed ^ _RETRY_SEED_XOR
                technical = _technical_attempt_record(seed=retry_seed, error=exc)
                yield SoundGenerationOutcome(
                    asset_id=request.asset_id,
                    error=(
                        "Audex retry batch failed; "
                        f"initial=(seed={initial_seed}; "
                        f"{_structure_summary(attempts[index].structure)}); "
                        f"retry=(seed={retry_seed}; {technical.failures[0]})"
                    ),
                    attempts=(
                        _attempt_record(attempts[index], seed=initial_seed),
                        technical,
                    ),
                )
            return
        for index, retry_attempt in zip(retry_indexes, retry_attempts, strict=True):
            request = requests[index]
            first_attempt = attempts[index]
            retry_seed = cases[index][1] ^ _RETRY_SEED_XOR
            elapsed_seconds = (
                first_attempt.elapsed_seconds + retry_attempt.elapsed_seconds
            )
            if _sound_lab_attempt_usable(retry_attempt):
                yield _sound_lab_success(
                    request,
                    retry_attempt,
                    seed_used=retry_seed,
                    elapsed_seconds=elapsed_seconds,
                    attempts=(
                        _attempt_record(first_attempt, seed=cases[index][1]),
                        _attempt_record(retry_attempt, seed=retry_seed),
                    ),
                )
                continue
            yield SoundGenerationOutcome(
                asset_id=request.asset_id,
                error=_sound_lab_failure(
                    retry_attempt,
                    first_attempt=first_attempt,
                    initial_seed=cases[index][1],
                    final_seed=retry_seed,
                ),
                attempts=(
                    _attempt_record(first_attempt, seed=cases[index][1]),
                    _attempt_record(retry_attempt, seed=retry_seed),
                ),
            )


def _sound_lab_success(
    request: SoundGenerationRequest,
    attempt: Any,
    *,
    seed_used: int,
    elapsed_seconds: float,
    attempts: tuple[SoundGenerationAttempt, ...],
) -> SoundGenerationOutcome:
    wav_path = attempt.enhanced_wav_path or attempt.raw_wav_path
    return SoundGenerationOutcome(
        asset_id=request.asset_id,
        generated=GeneratedSound(
            wav_path=wav_path,
            duration_seconds=10.0,
            elapsed_seconds=elapsed_seconds,
            seed_used=seed_used,
        ),
        attempts=attempts,
    )


def _attempt_record(attempt: Any, *, seed: int) -> SoundGenerationAttempt:
    structure = attempt.structure
    return SoundGenerationAttempt(
        seed=seed,
        elapsed_seconds=attempt.elapsed_seconds,
        frame_count=structure.frame_count,
        duration_seconds=structure.duration_seconds,
        reached_end_token=structure.reached_end_token,
        failures=structure.failures,
    )


def _technical_attempt_record(*, seed: int, error: Exception) -> SoundGenerationAttempt:
    return SoundGenerationAttempt(
        seed=seed,
        elapsed_seconds=0.0,
        frame_count=0,
        duration_seconds=0.0,
        reached_end_token=False,
        failures=(f"technical_failure: {type(error).__name__}: {error}",),
    )


def _sound_lab_attempt_usable(attempt: Any) -> bool:
    return bool(attempt.structure.nvidia_reference_decodable)


def _sound_lab_failure(
    attempt: Any,
    *,
    first_attempt: Any | None,
    initial_seed: int,
    final_seed: int,
) -> str:
    if first_attempt is None:
        return (
            "Audex generated an unusable audio token stream: "
            f"seed={final_seed}; {_structure_summary(attempt.structure)}"
        )
    return (
        "Audex generated an unusable audio token stream after one retry; "
        f"initial=(seed={initial_seed}; {_structure_summary(first_attempt.structure)}); "
        f"retry=(seed={final_seed}; {_structure_summary(attempt.structure)})"
    )


def _structure_summary(structure: Any) -> str:
    failures = "; ".join(structure.failures) or "unknown_structure_failure"
    return (
        f"{failures}; "
        f"frames={structure.frame_count}; duration={structure.duration_seconds:.3f}s; "
        f"reached_end={structure.reached_end_token}"
    )


def _generation_case(asset_id: str, variant: VariantBrief) -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id=asset_id,
        track=EvaluationTrack.GENERATION,
        dataset_id="audex-sound-lab",
        dataset_revision="local-v1",
        dataset_config="interactive",
        dataset_split="session",
        source_row_id=asset_id,
        source_row_hash=asset_id,
        license="user-directed-local-artifact",
        category="interactive",
        prompt=variant.caption,
        caption=variant.caption,
    )


def _parse_variants(
    raw: str,
    *,
    expected_count: int,
    job_id: str,
) -> tuple[VariantBrief, ...]:
    payload = _extract_design_payload(raw)
    variant_keys = [
        name for name in ("variants", "sounds", "candidates") if name in payload
    ]
    if len(variant_keys) != 1:
        raise ValueError("Sound Lab designer JSON must contain one variant list")
    items = payload[variant_keys[0]]
    if not isinstance(items, list) or len(items) != expected_count:
        actual = len(items) if isinstance(items, list) else "non-list"
        raise ValueError(
            f"Sound Lab designer returned {actual} variants; expected {expected_count}"
        )
    variants: list[VariantBrief] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Sound Lab variant {index} has an invalid schema")
        caption = _one_string(item, ("caption", "prompt", "description"))
        difference = _one_string(
            item,
            ("difference", "rationale", "variation", "reason"),
            required=False,
        )
        if caption is None:
            raise ValueError(f"Sound Lab variant {index} caption is invalid")
        if not _is_literal_audio_caption(caption):
            raise ValueError(
                f"Sound Lab variant {index} is not a literal AudioCaps-style caption"
            )
        if difference is None:
            raise ValueError(f"Sound Lab variant {index} difference is invalid")
        variants.append(
            VariantBrief(
                caption=" ".join(caption.split()),
                difference=" ".join(difference.split()),
                seed=_variant_seed(job_id, index, caption),
            )
        )
    normalized_captions = {variant.caption.casefold() for variant in variants}
    if len(normalized_captions) != len(variants):
        raise ValueError("Sound Lab designer returned duplicate captions")
    return tuple(variants)


def _is_literal_audio_caption(caption: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", caption)
    if not 3 <= len(words) <= 24:
        return False
    lowered = " ".join(caption.casefold().split())
    forbidden = (
        "create ",
        "generate ",
        "make ",
        "sound effect",
        "high-quality",
        "high quality",
        "cinematic",
        "audio of ",
        "please ",
        "do not ",
        "without ",
    )
    return not any(phrase in lowered for phrase in forbidden)


def _extract_design_payload(raw: str) -> dict[str, Any]:
    cleaned = _THINK_RE.sub("", raw).strip()
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(cleaned, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and any(
            key in value for key in ("variants", "sounds", "candidates")
        ):
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError(
            "Audex Sound Lab designer did not return one unambiguous JSON object"
        )
    return candidates[0]


def _one_string(
    item: dict[str, Any],
    names: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    values = {
        str(item[name]).strip()
        for name in names
        if isinstance(item.get(name), str) and str(item[name]).strip()
    }
    if len(values) > 1:
        raise ValueError(f"Sound Lab variant has conflicting fields: {names}")
    if not values:
        if required:
            return None
        return None
    return values.pop()


def _variant_seed(job_id: str, index: int, caption: str) -> int:
    digest = hashlib.sha256(f"{job_id}\0{index}\0{caption}".encode()).digest()
    return int.from_bytes(digest[:4], "big")
