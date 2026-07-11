"""Audex model adapters for Sound Lab planning, design, and generation."""

from __future__ import annotations

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
from .session import GeneratedSound, VariantBrief
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
{"variants":[{"caption":"...","difference":"...","seed":123}]}
Seeds must be distinct integers from 0 through 4294967295. Do not use markdown."""


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
    ) -> tuple[VariantBrief, ...]:
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
        request = build_text_messages_response_request(
            self._runtime.tokenizer,
            messages,
            prompt_text="",
            enable_reasoning=False,
            max_tokens=_DESIGNER_MAX_TOKENS,
        )
        request = replace(request, debug_name="sound-lab-design")
        result = run_sync_model_call(self._runtime.generate_one_final(request))
        return _parse_variants(result.text, expected_count=call.count)


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


def _parse_variants(raw: str, *, expected_count: int) -> tuple[VariantBrief, ...]:
    cleaned = _THINK_RE.sub("", raw).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Audex Sound Lab designer did not return strict JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"variants"}:
        raise ValueError("Sound Lab designer JSON must contain only variants")
    items = payload["variants"]
    if not isinstance(items, list) or len(items) != expected_count:
        actual = len(items) if isinstance(items, list) else "non-list"
        raise ValueError(
            f"Sound Lab designer returned {actual} variants; expected {expected_count}"
        )
    variants: list[VariantBrief] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != {
            "caption",
            "difference",
            "seed",
        }:
            raise ValueError(f"Sound Lab variant {index} has an invalid schema")
        caption = item["caption"]
        difference = item["difference"]
        seed = item["seed"]
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Sound Lab variant {index} caption is invalid")
        if not isinstance(difference, str) or not difference.strip():
            raise ValueError(f"Sound Lab variant {index} difference is invalid")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError(f"Sound Lab variant {index} seed is invalid")
        variants.append(
            VariantBrief(
                caption=" ".join(caption.split()),
                difference=" ".join(difference.split()),
                seed=seed,
            )
        )
    if len({variant.seed for variant in variants}) != len(variants):
        raise ValueError("Sound Lab designer returned duplicate seeds")
    return tuple(variants)
