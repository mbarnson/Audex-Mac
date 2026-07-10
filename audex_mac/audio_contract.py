"""Audex native audio and speech-token contract helpers."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_values import optional_int

SAMPLE_RATE = 16000
SOUND_PLACEHOLDER = "<sound>"
SOUND_TOKEN = "<so_embedding>"
SOUND_START_TOKEN = "<so_start>"
SOUND_END_TOKEN = "<so_end>"
SPEECHGEN_START_TOKEN = "<speechgen_start>"
SPEECHGEN_END_TOKEN = "<speechgen_end>"
IM_END_TOKEN = "<|im_end|>"

DEFAULT_AUDIO_PROMPT = "Transcribe the input speech."
DEFAULT_TTS_PREFIX = "<|text to speech|> Generate speech for this transcription. "
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and harmless assistant.\n\n"
    "You are not allowed to use any tools."
)
NVIDIA_TTS_TEMPERATURE = 0.8
NVIDIA_TTS_TOP_P = 1.0
NVIDIA_TTS_TOP_K = 0
NVIDIA_TTS_CFG_TEMPERATURE = 1.0
NVIDIA_TTS_CFG_TOP_P = 1.0
NVIDIA_TTS_CFG_TOP_K = 80
NVIDIA_TTS_CFG_SCALE = 3.0
NVIDIA_TTS_CFG_PAIRS_PER_BATCH = 2
NVIDIA_ASR_TEMPERATURE = 0.0
NVIDIA_ASR_TOP_P = 1.0
NVIDIA_TEXT_TEMPERATURE = 1.0
NVIDIA_TEXT_TOP_P = 0.95
DEFAULT_SOUND_CLIP_DURATION = 30.0
DEFAULT_SOUND_EMBEDDING_SIZE = 750
MAX_AUDIO_SECONDS = 900
MAX_AUDIO_CLIPS = 30
DEFAULT_DECODER_CHUNK_FRAMES = 75


@dataclass(frozen=True, slots=True)
class AudioPromptPlan:
    prompt: str
    sample_count: int
    sample_rate: int
    num_clips: int
    embeddings_per_clip: int
    num_embeddings: int


@dataclass(frozen=True, slots=True)
class SpeechCodecTokenMap:
    speech_codec: dict[int, int]
    speechgen_start: int
    speechgen_end: int


@dataclass(frozen=True, slots=True)
class DecoderPreflight:
    decoder_path: Path
    ready: bool
    missing_files: tuple[str, ...]
    sample_rate: int | None
    lookahead_steps: int | None
    codebook_size: int | None


@dataclass(frozen=True, slots=True)
class SpeechTokenizerPreflight:
    tokenizer_path: Path
    ready: bool
    speechgen_start: int | None
    speechgen_end: int | None
    codec_token_count: int
    error: str | None = None


def audio_clip_count(
    sample_count: int,
    *,
    sample_rate: int = SAMPLE_RATE,
    clip_duration: float = DEFAULT_SOUND_CLIP_DURATION,
) -> int:
    if sample_count < 0:
        raise ValueError(f"sample_count must be non-negative, got {sample_count}")
    clip_samples = int(round(sample_rate * clip_duration))
    if clip_samples <= 0:
        raise ValueError(f"Invalid clip_samples: {clip_samples}")
    return max(1, math.ceil(max(1, sample_count) / clip_samples))


def audio_embedding_count(
    num_clips: int,
    *,
    embeddings_per_clip: int = DEFAULT_SOUND_EMBEDDING_SIZE,
) -> int:
    if num_clips <= 0:
        raise ValueError(f"num_clips must be positive, got {num_clips}")
    if embeddings_per_clip <= 0:
        raise ValueError(
            f"embeddings_per_clip must be positive, got {embeddings_per_clip}"
        )
    return num_clips * embeddings_per_clip


def build_audio_prompt_plan(
    prompt: str = DEFAULT_AUDIO_PROMPT,
    *,
    sample_count: int,
    sample_rate: int = SAMPLE_RATE,
    embeddings_per_clip: int = DEFAULT_SOUND_EMBEDDING_SIZE,
    clip_duration: float = DEFAULT_SOUND_CLIP_DURATION,
) -> AudioPromptPlan:
    num_clips = audio_clip_count(
        sample_count,
        sample_rate=sample_rate,
        clip_duration=clip_duration,
    )
    if num_clips > MAX_AUDIO_CLIPS:
        raise ValueError(
            f"Audio needs {num_clips} clips > MAX_AUDIO_CLIPS={MAX_AUDIO_CLIPS}"
        )
    return AudioPromptPlan(
        prompt=prompt,
        sample_count=sample_count,
        sample_rate=sample_rate,
        num_clips=num_clips,
        embeddings_per_clip=embeddings_per_clip,
        num_embeddings=audio_embedding_count(
            num_clips,
            embeddings_per_clip=embeddings_per_clip,
        ),
    )


def expand_sound_placeholder(prompt: str, num_embeddings: int) -> str:
    count = prompt.count(SOUND_PLACEHOLDER)
    if count != 1:
        raise ValueError(f"Expected exactly one {SOUND_PLACEHOLDER}, found {count}")
    if num_embeddings <= 0:
        raise ValueError(f"num_embeddings must be positive, got {num_embeddings}")
    replacement = SOUND_START_TOKEN + (SOUND_TOKEN * num_embeddings) + SOUND_END_TOKEN
    return prompt.replace(SOUND_PLACEHOLDER, replacement)


def build_audio_chat_prompt(plan: AudioPromptPlan, *, thinking_enabled: bool) -> str:
    user_prompt = f"<|im_start|>user\n{SOUND_PLACEHOLDER}\n{plan.prompt}<|im_end|>\n"
    assistant_prefix = "<think>\n" if thinking_enabled else "<think></think>"
    prompt = user_prompt + f"<|im_start|>assistant\n{assistant_prefix}"
    return expand_sound_placeholder(prompt, plan.num_embeddings)


def build_tts_prompt(text: str, tokenizer: Any) -> str:
    """Build NVIDIA's Audex TTS prompt, ending at `<speechgen_start>`."""

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{DEFAULT_TTS_PREFIX}{text}"},
    ]
    return (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + SPEECHGEN_START_TOKEN
    )


def build_tts_null_prompt(
    cond_prompt: str,
    tokenizer: Any,
    *,
    max_iters: int = 64,
) -> str:
    """Build NVIDIA's same-length null prompt for TTS CFG."""

    target_len = len(tokenizer.encode(cond_prompt))

    def template(null_text: str) -> str:
        return build_tts_prompt(null_text, tokenizer)

    base_len = len(tokenizer.encode(template("")))
    n_unk = max(1, target_len - base_len)
    for _ in range(max_iters):
        prompt = template("<unk>" * n_unk)
        cur_len = len(tokenizer.encode(prompt))
        if cur_len == target_len:
            return prompt
        if cur_len < target_len:
            n_unk += 1
        else:
            n_unk = max(1, n_unk - 1)
            if n_unk == 1:
                break
    return template("<unk>" * n_unk)


def tokenize_tts_cfg_pair(
    cond_prompt: str,
    uncond_prompt: str,
    tokenizer: Any,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Tokenize and pad NVIDIA's conditional/unconditional TTS CFG pair."""

    cond_ids = [int(token_id) for token_id in tokenizer.encode(cond_prompt)]
    uncond_ids = [int(token_id) for token_id in tokenizer.encode(uncond_prompt)]
    if len(cond_ids) == len(uncond_ids):
        return tuple(cond_ids), tuple(uncond_ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer needs pad_token_id or eos_token_id for TTS CFG.")

    target_len = max(len(cond_ids), len(uncond_ids))
    cond_ids += [int(pad_id)] * (target_len - len(cond_ids))
    uncond_ids += [int(pad_id)] * (target_len - len(uncond_ids))
    return tuple(cond_ids), tuple(uncond_ids)


def build_codec_token_map(vocab: dict[str, int]) -> SpeechCodecTokenMap:
    speech_codec: dict[int, int] = {}
    start_id = vocab.get(SPEECHGEN_START_TOKEN)
    end_id = vocab.get(SPEECHGEN_END_TOKEN)
    if start_id is None:
        raise ValueError(f"Tokenizer is missing {SPEECHGEN_START_TOKEN}.")
    if end_id is None:
        raise ValueError(f"Tokenizer is missing {SPEECHGEN_END_TOKEN}.")

    for token, token_id in vocab.items():
        if match := re.fullmatch(r"<speechcodec_(\d+)>", token):
            speech_codec[int(token_id)] = int(match.group(1))
    if not speech_codec:
        raise ValueError("Tokenizer has no <speechcodec_*> tokens.")
    return SpeechCodecTokenMap(
        speech_codec=speech_codec,
        speechgen_start=int(start_id),
        speechgen_end=int(end_id),
    )


def load_tokenizer_vocab(tokenizer_path: Path) -> dict[str, int]:
    raw = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"Tokenizer JSON has no model object: {tokenizer_path}")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"Tokenizer JSON has no model.vocab object: {tokenizer_path}")
    return {str(token): int(token_id) for token, token_id in vocab.items()}


def preflight_speech_tokenizer(tokenizer_path: Path) -> SpeechTokenizerPreflight:
    try:
        token_map = build_codec_token_map(load_tokenizer_vocab(tokenizer_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return SpeechTokenizerPreflight(
            tokenizer_path=tokenizer_path,
            ready=False,
            speechgen_start=None,
            speechgen_end=None,
            codec_token_count=0,
            error=str(exc),
        )
    return SpeechTokenizerPreflight(
        tokenizer_path=tokenizer_path,
        ready=True,
        speechgen_start=token_map.speechgen_start,
        speechgen_end=token_map.speechgen_end,
        codec_token_count=len(token_map.speech_codec),
    )


def iter_new_speech_frames(
    token_ids: Sequence[int],
    token_map: SpeechCodecTokenMap,
    state: dict[str, Any],
) -> Iterator[list[int]]:
    offset = int(state.setdefault("offset", 0))
    active = bool(state.setdefault("active", False))
    done = bool(state.setdefault("done", False))

    if done:
        return

    for token_id in token_ids[offset:]:
        if token_id == token_map.speechgen_start:
            active = True
            continue
        if token_id == token_map.speechgen_end:
            done = True
            break
        if active and token_id in token_map.speech_codec:
            yield [token_map.speech_codec[token_id]]

    state["offset"] = len(token_ids)
    state["active"] = active
    state["done"] = done


def preflight_decoder(decoder_path: Path) -> DecoderPreflight:
    required = (
        "config.json",
        "model.safetensors",
        "modeling_audex_causal_speech_decoder.py",
        "configuration_audex_causal_speech_decoder.py",
        "streaming_utils.py",
    )
    missing = tuple(name for name in required if not (decoder_path / name).is_file())
    sample_rate = None
    lookahead_steps = None
    codebook_size = None
    config_path = decoder_path / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        sample_rate = optional_int(config.get("sample_rate"))
        lookahead_steps = optional_int(config.get("lookahead_steps"))
        codebook_size = optional_int(config.get("codebook_size"))
    return DecoderPreflight(
        decoder_path=decoder_path,
        ready=not missing and sample_rate == SAMPLE_RATE,
        missing_files=missing,
        sample_rate=sample_rate,
        lookahead_steps=lookahead_steps,
        codebook_size=codebook_size,
    )
