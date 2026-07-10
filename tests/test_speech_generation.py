from __future__ import annotations

import pytest

from audex_mac.audio_contract import (
    NVIDIA_TTS_CFG_SCALE,
    NVIDIA_TTS_TEMPERATURE,
    NVIDIA_TTS_TOP_K,
    NVIDIA_TTS_TOP_P,
)
from audex_mac.speech_generation import SpeechTokenGenerationSmokeResult

pytestmark = pytest.mark.fast


def test_speech_token_generation_result_requires_full_vocab_and_codec_output() -> None:
    result = SpeechTokenGenerationSmokeResult(
        backend="mlx_lm",
        device="Device(gpu, 0)",
        model_type="nemotron_dense_audex",
        vocab_size=205312,
        prompt_tokens=57,
        prompt_max_token_id=131075,
        speechgen_start_id=131075,
        speechgen_end_id=131076,
        codec_token_count=65536,
        generated_token_ids=(131077, 131078),
        generated_token_text=("<speechcodec_0>", "<speechcodec_1>"),
        generated_codec_frames=(0, 1),
        logprobs_shape=(205312,),
        reached_end_token=False,
        hit_max_tokens=True,
        temperature=NVIDIA_TTS_TEMPERATURE,
        top_p=NVIDIA_TTS_TOP_P,
        top_k=NVIDIA_TTS_TOP_K,
        cfg_scale_reference=NVIDIA_TTS_CFG_SCALE,
        cfg_applied=True,
    )

    assert result.ready is True
    assert result.cfg_applied is True


def test_speech_token_generation_result_rejects_text_only_vocab() -> None:
    result = SpeechTokenGenerationSmokeResult(
        backend="mlx_lm",
        device="Device(gpu, 0)",
        model_type="nemotron_dense",
        vocab_size=131072,
        prompt_tokens=57,
        prompt_max_token_id=131075,
        speechgen_start_id=131075,
        speechgen_end_id=131076,
        codec_token_count=65536,
        generated_token_ids=(131077,),
        generated_token_text=("<speechcodec_0>",),
        generated_codec_frames=(0,),
        logprobs_shape=(131072,),
        reached_end_token=False,
        hit_max_tokens=True,
        temperature=NVIDIA_TTS_TEMPERATURE,
        top_p=NVIDIA_TTS_TOP_P,
        top_k=NVIDIA_TTS_TOP_K,
        cfg_scale_reference=NVIDIA_TTS_CFG_SCALE,
        cfg_applied=True,
    )

    assert result.ready is False
