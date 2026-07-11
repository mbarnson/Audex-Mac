"""Audex model adapters for Sound Lab planning, design, and generation."""

from __future__ import annotations

import hashlib
import json
import re
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
    VariantBrief,
    VariantDesignError,
    VariantDesignResult,
)
from .tools import RenderSoundsCall, parse_sound_lab_tool_call

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PLANNER_MAX_TOKENS = 768
_DESIGNER_MAX_TOKENS = 1536

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
different text-to-audio captions. Vary meaningful acoustic dimensions such as
source, distance, environment, material, temporal envelope, intensity, or
recording character. Preserve every explicit constraint. Each caption must
describe only audible content and must stand alone as a text-to-audio prompt.
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

    def __init__(self, *, runtime: Any, decode_to_wav: Any) -> None:
        self._runtime = runtime
        self._decode_to_wav = decode_to_wav

    def generate(
        self,
        variant: VariantBrief,
        *,
        asset_id: str,
        output_dir: Path,
    ) -> GeneratedSound:
        case = AudioEvaluationCase(
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
        attempt = AudexVllmTtaGenerationAdapter(
            runtime=self._runtime,
            raw_dir=output_dir / "raw",
            enhanced_dir=output_dir / "enhanced",
            decode_to_wav=self._decode_to_wav,
        ).generate(case, seed=variant.seed)
        if not attempt.structure.valid:
            raise RuntimeError(
                "Audex generated an invalid audio token stream: "
                + "; ".join(attempt.structure.failures)
            )
        wav_path = attempt.enhanced_wav_path or attempt.raw_wav_path
        return GeneratedSound(
            wav_path=wav_path,
            duration_seconds=attempt.structure.duration_seconds,
            elapsed_seconds=attempt.elapsed_seconds,
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
