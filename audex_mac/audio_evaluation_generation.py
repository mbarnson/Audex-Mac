"""Audex text-to-audio request and codec-stream contracts.

This module intentionally has no decoder or model-loading dependencies.  It is
the seam shared by the evaluator, vLLM adapters, and fast structural tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .vllm_sts_requests import (
    VllmGenerationRequest,
    VllmSamplingPlan,
)

AUDIOGEN_START_TOKEN = "<audiogen_start>"
AUDIOGEN_END_TOKEN = "<audiogen_end>"
XCODEC1_CODEBOOK_SIZE = 1024
XCODEC1_GENERATED_CODEBOOKS = 4
XCODEC1_FRAMES_PER_SECOND = 50
DEFAULT_TTA_TARGET_SECONDS = 10.0
DEFAULT_TTA_CODEC_TOKENS = 2000
_AUDIO_CODEC_RE = re.compile(r"^<audiocodec_(\d+)>$")


@dataclass(frozen=True, slots=True)
class TtaRecipe:
    """Pinned NVIDIA text-to-audio generation recipe."""

    cfg_scale: float = 3.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 80
    max_tokens: int = 2048
    target_seconds: float = DEFAULT_TTA_TARGET_SECONDS
    codec_token_cap: int = DEFAULT_TTA_CODEC_TOKENS
    target_frame_tolerance: int = 1

    def __post_init__(self) -> None:
        if self.cfg_scale <= 1.0:
            raise ValueError("autonomous TTA evaluation requires CFG scale > 1")
        if self.codec_token_cap <= 0:
            raise ValueError("codec_token_cap must be positive")
        if self.codec_token_cap % XCODEC1_GENERATED_CODEBOOKS:
            raise ValueError("codec_token_cap must contain complete RVQ frames")
        expected = round(
            self.target_seconds
            * XCODEC1_FRAMES_PER_SECOND
            * XCODEC1_GENERATED_CODEBOOKS
        )
        if self.codec_token_cap != expected:
            raise ValueError(
                "codec_token_cap does not match target_seconds at the XCodec1 rate"
            )
        if self.max_tokens <= self.codec_token_cap:
            raise ValueError("max_tokens must leave room for <audiogen_end>")
        if self.target_frame_tolerance < 0:
            raise ValueError("target_frame_tolerance must be non-negative")


@dataclass(frozen=True, slots=True)
class TtaOutputInspection:
    """Structural facts derived from one generated token stream."""

    codec_ids: tuple[int, ...]
    codec_token_count: int
    frame_count: int
    duration_seconds: float
    reached_end_token: bool
    first_phase_mismatch: dict[str, int] | None
    unexpected_token_ids: tuple[int, ...]
    failures: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.failures


def build_tta_requests(
    tokenizer: Any,
    *,
    caption: str,
    case_id: str,
    seed: int,
    recipe: TtaRecipe | None = None,
) -> tuple[VllmGenerationRequest, VllmGenerationRequest]:
    """Build a length-matched conditional/unconditional CFG request pair."""

    recipe = recipe or TtaRecipe()
    caption = caption.strip()
    case_id = case_id.strip()
    if not caption:
        raise ValueError("TTA caption must not be empty")
    if not case_id:
        raise ValueError("TTA case_id must not be empty")

    token_maps = _build_codec_token_maps(tokenizer)
    phase_token_ids = _build_phase_token_ids(token_maps)
    cond_prompt = _build_tta_prompt(caption)
    uncond_prompt = _build_tta_null_prompt(cond_prompt, tokenizer)
    cond_ids, uncond_ids = _tokenize_equal_length_pair(
        cond_prompt,
        uncond_prompt,
        tokenizer,
    )
    pair_id = f"tta-{case_id}"
    tta_rvq = {
        "phase_token_ids": phase_token_ids,
        "start_tid": token_maps["start_tid"],
        "end_tid": token_maps["end_tid"],
        "codec_cap": recipe.codec_token_cap,
        "start_in_prompt": True,
    }

    def request(role: str, prompt_token_ids: list[int]) -> VllmGenerationRequest:
        return VllmGenerationRequest(
            prompt={"prompt_token_ids": prompt_token_ids},
            sampling=VllmSamplingPlan(
                max_tokens=recipe.max_tokens,
                temperature=recipe.temperature,
                top_p=recipe.top_p,
                top_k=recipe.top_k,
                seed=seed,
                stop=(AUDIOGEN_END_TOKEN,),
                stop_token_ids=(token_maps["end_tid"],),
                extra_args={
                    "cfg_scale": recipe.cfg_scale,
                    "cfg_role": role,
                    "cfg_pair_id": pair_id,
                    "tta_rvq": tta_rvq,
                },
            ),
            debug_name=f"{pair_id}-{role}",
            request_id_suffix=f"{pair_id}-{role}",
        )

    return request("cond", cond_ids), request("uncond", uncond_ids)


def inspect_tta_output(
    tokenizer: Any,
    token_ids: list[int] | tuple[int, ...],
    *,
    recipe: TtaRecipe | None = None,
) -> TtaOutputInspection:
    """Validate and extract one generated XCodec1 token stream."""

    recipe = recipe or TtaRecipe()
    token_maps = _build_codec_token_maps(tokenizer)
    audio_codec: dict[int, int] = token_maps["audio_codec"]
    start_tid = token_maps["start_tid"]
    end_tid = token_maps["end_tid"]

    codec_ids: list[int] = []
    unexpected: list[int] = []
    reached_end = False
    for raw_token_id in token_ids:
        token_id = int(raw_token_id)
        if token_id == start_tid and not codec_ids:
            continue
        if token_id == end_tid:
            reached_end = True
            break
        codec_id = audio_codec.get(token_id)
        if codec_id is None:
            unexpected.append(token_id)
        else:
            codec_ids.append(codec_id)

    mismatch: dict[str, int] | None = None
    for index, codec_id in enumerate(codec_ids):
        actual_phase = codec_id // XCODEC1_CODEBOOK_SIZE
        expected_phase = index % XCODEC1_GENERATED_CODEBOOKS
        if actual_phase != expected_phase:
            mismatch = {
                "index": index,
                "codec_id": codec_id,
                "actual_phase": actual_phase,
                "expected_phase": expected_phase,
            }
            break

    failures: list[str] = []
    if not reached_end:
        failures.append("missing_end_token")
    if len(codec_ids) % XCODEC1_GENERATED_CODEBOOKS:
        failures.append("incomplete_rvq_frame")
    expected_frames = recipe.codec_token_cap // XCODEC1_GENERATED_CODEBOOKS
    frame_count = len(codec_ids) // XCODEC1_GENERATED_CODEBOOKS
    if abs(frame_count - expected_frames) > recipe.target_frame_tolerance:
        failures.append("incomplete_target")
    if mismatch is not None:
        failures.append("rvq_phase_mismatch")
    if unexpected:
        failures.append("unexpected_token")

    return TtaOutputInspection(
        codec_ids=tuple(codec_ids),
        codec_token_count=len(codec_ids),
        frame_count=frame_count,
        duration_seconds=frame_count / XCODEC1_FRAMES_PER_SECOND,
        reached_end_token=reached_end,
        first_phase_mismatch=mismatch,
        unexpected_token_ids=tuple(unexpected),
        failures=tuple(failures),
    )


def _build_tta_prompt(caption: str) -> str:
    system_prompt = (
        "You are a helpful and harmless assistant.\n\n"
        "You are not allowed to use any tools."
    )
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|text to audio|> Generate audio for this caption. {caption}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n<think></think><audiogen_start>"
    )


def _build_tta_null_prompt(cond_prompt: str, tokenizer: Any) -> str:
    target_length = len(tokenizer.encode(cond_prompt))

    def template(null_text: str) -> str:
        return _build_tta_prompt(null_text)

    empty_length = len(tokenizer.encode(template("")))
    unknown_count = max(1, target_length - empty_length)
    best_prompt: str | None = None
    best_delta: int | None = None
    for _ in range(64):
        prompt = template("<unk>" * unknown_count)
        length = len(tokenizer.encode(prompt))
        delta = length - target_length
        if delta == 0:
            return prompt
        if best_delta is None or abs(delta) < abs(best_delta):
            best_prompt = prompt
            best_delta = delta
        unknown_count = max(1, unknown_count - 1 if delta > 0 else unknown_count + 1)
    assert best_prompt is not None
    return best_prompt


def _tokenize_equal_length_pair(
    cond_prompt: str,
    uncond_prompt: str,
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    cond_ids = [int(token_id) for token_id in tokenizer.encode(cond_prompt)]
    uncond_ids = [int(token_id) for token_id in tokenizer.encode(uncond_prompt)]
    if len(cond_ids) != len(uncond_ids):
        raise ValueError(
            "TTA CFG prompts could not be token-length matched: "
            f"cond={len(cond_ids)}, uncond={len(uncond_ids)}"
        )
    return cond_ids, uncond_ids


def _build_codec_token_maps(tokenizer: Any) -> dict[str, Any]:
    vocab = tokenizer.get_vocab()
    try:
        start_tid = int(vocab[AUDIOGEN_START_TOKEN])
        end_tid = int(vocab[AUDIOGEN_END_TOKEN])
    except KeyError as exc:
        raise ValueError(f"tokenizer is missing required marker {exc.args[0]}") from exc

    audio_codec: dict[int, int] = {}
    for token, raw_token_id in vocab.items():
        match = _AUDIO_CODEC_RE.fullmatch(str(token))
        if match is not None:
            audio_codec[int(raw_token_id)] = int(match.group(1))
    if not audio_codec:
        raise ValueError("tokenizer has no <audiocodec_*> tokens")
    return {
        "audio_codec": audio_codec,
        "start_tid": start_tid,
        "end_tid": end_tid,
    }


def _build_phase_token_ids(token_maps: dict[str, Any]) -> list[list[int]]:
    codec_to_token = {
        codec_id: token_id for token_id, codec_id in token_maps["audio_codec"].items()
    }
    phases: list[list[int]] = []
    for phase in range(XCODEC1_GENERATED_CODEBOOKS):
        first = phase * XCODEC1_CODEBOOK_SIZE
        token_ids: list[int] = []
        for codec_id in range(first, first + XCODEC1_CODEBOOK_SIZE):
            token_id = codec_to_token.get(codec_id)
            if token_id is None:
                raise ValueError(
                    f"incomplete codec vocab for RVQ phase {phase}: "
                    f"missing codec id {codec_id}"
                )
            token_ids.append(token_id)
        phases.append(token_ids)
    return phases
