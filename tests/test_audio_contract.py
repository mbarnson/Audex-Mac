from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.audio_contract import (
    DEFAULT_SOUND_EMBEDDING_SIZE,
    DEFAULT_TTS_PREFIX,
    NVIDIA_TTS_CFG_PAIRS_PER_BATCH,
    NVIDIA_TTS_CFG_SCALE,
    NVIDIA_TTS_CFG_TEMPERATURE,
    NVIDIA_TTS_CFG_TOP_K,
    NVIDIA_TTS_CFG_TOP_P,
    NVIDIA_TTS_TEMPERATURE,
    NVIDIA_TTS_TOP_K,
    NVIDIA_TTS_TOP_P,
    SOUND_END_TOKEN,
    SOUND_START_TOKEN,
    SOUND_TOKEN,
    SPEECHGEN_START_TOKEN,
    build_audio_chat_prompt,
    build_audio_prompt_plan,
    build_codec_token_map,
    build_tts_null_prompt,
    build_tts_prompt,
    expand_sound_placeholder,
    iter_new_speech_frames,
    preflight_decoder,
    preflight_speech_tokenizer,
    tokenize_tts_cfg_pair,
)
from audex_mac.audio_runtime import preflight_audio_runtime
from audex_mac.models import DEFAULT_MODEL, SUPPORTED_MODELS

pytestmark = pytest.mark.fast


def test_audio_prompt_expands_one_short_clip_to_750_sound_embeddings() -> None:
    plan = build_audio_prompt_plan(sample_count=16_000)

    prompt = build_audio_chat_prompt(plan, thinking_enabled=False)

    assert plan.num_clips == 1
    assert plan.num_embeddings == DEFAULT_SOUND_EMBEDDING_SIZE
    assert prompt.count(SOUND_START_TOKEN) == 1
    assert prompt.count(SOUND_END_TOKEN) == 1
    assert prompt.count(SOUND_TOKEN) == DEFAULT_SOUND_EMBEDDING_SIZE
    assert "<think></think>" in prompt


def test_audio_prompt_counts_multiple_30_second_clips() -> None:
    plan = build_audio_prompt_plan(sample_count=480_001)

    assert plan.num_clips == 2
    assert plan.num_embeddings == 1500


def test_audio_prompt_rejects_inputs_longer_than_nvidia_limit() -> None:
    with pytest.raises(ValueError, match="MAX_AUDIO_CLIPS"):
        build_audio_prompt_plan(sample_count=(30 * 480_000) + 1)


def test_expand_sound_placeholder_requires_exactly_one_placeholder() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        expand_sound_placeholder("no audio here", 750)

    with pytest.raises(ValueError, match="exactly one"):
        expand_sound_placeholder("<sound> and <sound>", 750)


def test_audio_chat_prompt_can_enable_thinking_explicitly() -> None:
    plan = build_audio_prompt_plan(sample_count=1)

    prompt = build_audio_chat_prompt(plan, thinking_enabled=True)

    assert "<think>\n" in prompt
    assert "<think></think>" not in prompt


def test_tts_prompt_uses_nvidia_non_thinking_template() -> None:
    tokenizer = FakeChatTokenizer()

    prompt = build_tts_prompt("Hello.", tokenizer)

    assert prompt.endswith(SPEECHGEN_START_TOKEN)
    assert prompt.endswith("<|im_start|>assistant\n<speechgen_start>")
    assert DEFAULT_TTS_PREFIX in prompt
    assert tokenizer.enable_thinking is False


def test_nvidia_tts_no_cfg_sampler_defaults_are_recorded() -> None:
    assert NVIDIA_TTS_TEMPERATURE == 0.8
    assert NVIDIA_TTS_TOP_P == 1.0
    assert NVIDIA_TTS_TOP_K == 0


def test_nvidia_tts_cfg_sampler_defaults_are_recorded() -> None:
    assert NVIDIA_TTS_CFG_TEMPERATURE == 1.0
    assert NVIDIA_TTS_CFG_TOP_P == 1.0
    assert NVIDIA_TTS_CFG_TOP_K == 80
    assert NVIDIA_TTS_CFG_SCALE == 3.0
    assert NVIDIA_TTS_CFG_PAIRS_PER_BATCH == 2


def test_tts_null_prompt_matches_conditional_prompt_length_for_cfg() -> None:
    tokenizer = FakeChatTokenizer()
    cond_prompt = build_tts_prompt("Hello world.", tokenizer)

    null_prompt = build_tts_null_prompt(cond_prompt, tokenizer)

    assert "<unk><unk>" in null_prompt
    assert len(tokenizer.encode(null_prompt)) == len(tokenizer.encode(cond_prompt))


def test_tts_cfg_pair_is_padded_to_same_length() -> None:
    tokenizer = FakeChatTokenizer()
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = 99

    cond_ids, uncond_ids = tokenize_tts_cfg_pair("one two three", "one", tokenizer)

    assert cond_ids == (0, 1, 2)
    assert uncond_ids == (0, 99, 99)


def test_speech_codec_iteration_tracks_incremental_generation_state() -> None:
    token_map = build_codec_token_map(
        {
            "<speechgen_start>": 10,
            "<speechgen_end>": 11,
            "<speechcodec_0>": 20,
            "<speechcodec_42>": 21,
        }
    )
    state: dict[str, object] = {}

    first = list(iter_new_speech_frames([1, 10, 20], token_map, state))
    second = list(iter_new_speech_frames([1, 10, 20, 21, 11, 20], token_map, state))
    third = list(iter_new_speech_frames([1, 10, 20, 21, 11, 20, 21], token_map, state))

    assert first == [[0]]
    assert second == [[42]]
    assert third == []
    assert state["done"] is True


def test_preflight_decoder_reports_ready_decoder_artifacts(tmp_path: Path) -> None:
    write_decoder(tmp_path)

    result = preflight_decoder(tmp_path)

    assert result.ready is True
    assert result.sample_rate == 16000
    assert result.lookahead_steps == 4
    assert result.codebook_size == 65536


def test_preflight_decoder_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    write_decoder(tmp_path, sample_rate=24_000)

    result = preflight_decoder(tmp_path)

    assert result.ready is False


def test_preflight_speech_tokenizer_detects_codec_tokens(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(
        json.dumps(
            {
                "model": {
                    "vocab": {
                        "<speechgen_start>": 100,
                        "<speechgen_end>": 101,
                        "<speechcodec_0>": 102,
                        "<speechcodec_1023>": 1125,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = preflight_speech_tokenizer(tokenizer)

    assert result.ready is True
    assert result.speechgen_start == 100
    assert result.speechgen_end == 101
    assert result.codec_token_count == 2


def test_audio_runtime_preflight_checks_speech_snapshot_decoder_and_tokenizer(
    tmp_path: Path,
) -> None:
    make_speech_snapshot(tmp_path)

    result = preflight_audio_runtime(DEFAULT_MODEL, cache_root=tmp_path)

    assert result.ready is True
    assert result.model_path is not None
    assert result.model_path.name == "checkpoint_folder_full"
    assert result.audio_components is not None
    assert result.audio_components.ready is True
    assert result.audio_components.audio_encoder_weight_count == 1
    assert result.audio_preprocessor is not None
    assert result.audio_preprocessor.ready is True
    assert result.audio_preprocessor.feature_size == 128
    assert result.decoder is not None
    assert result.decoder.ready is True
    assert result.speech_tokenizer is not None
    assert result.speech_tokenizer.codec_token_count == 2


def test_speech_runtime_does_not_require_nvidia_demo_scripts() -> None:
    for model in SUPPORTED_MODELS:
        assert not any(
            path.startswith("inference_scripts_vllm/")
            for path in model.speech_required_files
        )


def write_decoder(path: Path, *, sample_rate: int = 16000) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "sample_rate": sample_rate,
                "lookahead_steps": 4,
                "codebook_size": 65536,
            }
        ),
        encoding="utf-8",
    )
    for filename in (
        "model.safetensors",
        "modeling_audex_causal_speech_decoder.py",
        "configuration_audex_causal_speech_decoder.py",
        "streaming_utils.py",
    ):
        (path / filename).write_text("", encoding="utf-8")


class FakeChatTokenizer:
    def __init__(self) -> None:
        self.enable_thinking: bool | None = None
        self.pad_token_id: int | None = 0
        self.eos_token_id: int | None = 2

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        self.enable_thinking = enable_thinking
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        return (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def encode(self, prompt: str) -> list[int]:
        if DEFAULT_TTS_PREFIX in prompt:
            count = 10 + prompt.count("<unk>")
            count += prompt.count("Hello") + prompt.count("world")
            return list(range(count))
        return list(range(len(prompt.split())))


def make_speech_snapshot(cache_root: Path) -> Path:
    repo_dir = cache_root / "models--nvidia--Nemotron-Labs-Audex-2B"
    snapshot = repo_dir / "snapshots" / "rev"
    checkpoint = snapshot / "checkpoint_folder_full"
    checkpoint.mkdir(parents=True)
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("rev", encoding="utf-8")

    for rel in DEFAULT_MODEL.speech_required_files:
        path = snapshot / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    (checkpoint / "config.json").write_text(
        json.dumps(audex_audio_config()),
        encoding="utf-8",
    )
    (checkpoint / "audio_preprocessor" / "preprocessor_config.json").write_text(
        json.dumps(audex_preprocessor_config()),
        encoding="utf-8",
    )
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "lm_head.weight": "model.safetensors",
                    "audio_encoder.conv1.weight": "model.safetensors",
                    "audio_projector.norm.weight": "model.safetensors",
                    "audio_projector.fc1.weight": "model.safetensors",
                    "audio_projector.fc2.weight": "model.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "model.safetensors").write_text("", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "vocab": {
                        "<speechgen_start>": 100,
                        "<speechgen_end>": 101,
                        "<speechcodec_0>": 102,
                        "<speechcodec_1023>": 1125,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    write_decoder(snapshot / "audex_causal_speech_decoder")
    return snapshot


def audex_audio_config() -> dict:
    return {
        "architectures": ["NemotronDenseAudexForConditionalGeneration"],
        "model_type": "nemotron_dense_audex",
        "audio_model_type": "NV-Whisper",
        "audio_encoder_hidden_size": 1280,
        "audio_config": {
            "encoder_layers": 32,
            "num_mel_bins": 128,
            "max_source_positions": 1500,
        },
        "sound_token": "<so_embedding>",
        "sound_token_id": 29,
        "sound_start_token": "<so_start>",
        "sound_end_token": "<so_end>",
        "sound_target_rate": 16000,
        "sound_clip_duration": 30.0,
        "sound_embedding_size": 750,
    }


def audex_preprocessor_config() -> dict:
    return {
        "chunk_length": 30,
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 128,
        "hop_length": 160,
        "n_fft": 400,
        "n_samples": 480000,
        "nb_max_frames": 3000,
        "padding_side": "right",
        "padding_value": 0.0,
        "processor_class": "Qwen2AudioProcessor",
        "return_attention_mask": True,
        "sampling_rate": 16000,
    }
