from __future__ import annotations

import sys
import types

import pytest

from audex_mac.audio_contract import (
    DEFAULT_TTS_PREFIX,
    NVIDIA_TTS_CFG_SCALE,
    NVIDIA_TTS_CFG_TEMPERATURE,
    NVIDIA_TTS_CFG_TOP_K,
    NVIDIA_TTS_CFG_TOP_P,
    NVIDIA_TTS_TEMPERATURE,
    NVIDIA_TTS_TOP_K,
    NVIDIA_TTS_TOP_P,
    SAMPLE_RATE,
    SPEECHGEN_START_TOKEN,
)
from audex_mac.vllm_sts_requests import (
    AUDEX_TEXT_STATE_APPEND_MODE,
    AUDEX_TEXT_STATE_BOUNDARY_ARG,
    AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY,
    AUDEX_TEXT_STATE_KEY_ARG,
    AUDEX_TEXT_STATE_MODE_ARG,
    AUDEX_TEXT_STATE_PREFIX_TOKEN_COUNT_ARG,
    AUDEX_TEXT_STATE_PREFIX_TOKEN_HASH_ARG,
    DEFAULT_ASR_MAX_TOKENS,
    DEFAULT_ASR_PROMPT,
    DEFAULT_TEXT_MAX_TOKENS,
    DEFAULT_TTS_MAX_TOKENS,
    DEFAULT_VLLM_AUDIO_RESPONSE_PROMPT,
    DEFAULT_VLLM_TEXT_PROMPT,
    build_asr_projected_embeddings_request,
    build_asr_request,
    build_audio_history_prime_request,
    build_audio_messages_response_request,
    build_audio_response_prefix_token_ids,
    build_text_messages_generation_prompt,
    build_text_messages_history_prompt,
    build_text_messages_response_request,
    build_text_response_request,
    build_tts_cfg_requests,
    build_tts_request,
    compose_system_prompt,
    compose_text_input,
    make_projected_embeddings_vllm_serializable,
    projected_audio_embedding_count,
)

pytestmark = pytest.mark.fast


class FakeVllmTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.enable_thinking: bool | None = None
        self.messages: list[dict[str, str]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.enable_thinking = enable_thinking
        self.messages = messages
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
        return list(range(max(1, len(prompt.split()))))

    def get_vocab(self) -> dict[str, int]:
        return {
            "<speechgen_start>": 10,
            "<speechgen_end>": 11,
            "<so_embedding>": 12,
            "<so_start>": 13,
            "<so_end>": 14,
            "<speechcodec_0>": 100,
            "<speechcodec_1>": 101,
            "<audiocodec_0>": 200,
        }


class PrefixStableTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert enable_thinking is False
        rendered = "".join(
            f"<|im_start|>{message['role']}\n" f"{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    def encode(self, prompt: str) -> list[int]:
        return [ord(char) for char in prompt]

    def get_vocab(self) -> dict[str, int]:
        return {}


def expected_text_modality_guard() -> dict[str, object]:
    return {
        "audex_disallow_token_ranges": [[100, 101], [200, 200]],
        "audex_disallow_token_ids": [10, 11, 12, 13, 14],
    }


def test_compose_text_input_matches_nvidia_web_shape() -> None:
    assert compose_text_input("Prompt", "Transcript") == "Prompt\n\nTranscript"
    assert compose_text_input("", "Transcript") == "Transcript"


def test_compose_system_prompt_appends_response_policy() -> None:
    assert compose_system_prompt("System.", "Policy.") == "System.\n\nPolicy."
    assert compose_system_prompt("System.", "") == "System."
    assert compose_system_prompt("", "Policy.") == "Policy."


def test_build_audio_messages_response_request_preserves_history_and_adds_audio() -> (
    None
):
    class AudioResponseTokenizer(PrefixStableTokenizer):
        def get_vocab(self) -> dict[str, int]:
            return FakeVllmTokenizer().get_vocab()

    tokenizer = AudioResponseTokenizer()
    samples = (0.0, 0.25)
    messages = [
        {"role": "system", "content": "Be Matt's conversational assistant."},
        {"role": "user", "content": "Earlier question."},
        {"role": "assistant", "content": "Earlier answer."},
    ]

    request = build_audio_messages_response_request(
        tokenizer,
        messages,
        samples,
        sample_rate=16_000,
    )

    assert isinstance(request.prompt, dict)
    rendered = request.prompt["prompt"]
    assert "Earlier question." in rendered
    assert "Earlier answer." in rendered
    assert DEFAULT_VLLM_AUDIO_RESPONSE_PROMPT in rendered
    assert "<so_embedding>" in rendered
    assert request.prompt["multi_modal_data"] == {"audio": [(samples, 16_000)]}
    assert request.debug_name == "audio-response"
    assert request.sampling.extra_args == expected_text_modality_guard()


def test_build_audio_response_can_trim_only_its_padded_audio_tail() -> None:
    tokenizer = PrefixStableTokenizer()
    samples = (0.0,) * 46_695

    request = build_audio_messages_response_request(
        tokenizer,
        [],
        samples,
        trim_padded_audio_embeddings=True,
    )

    assert isinstance(request.prompt, dict)
    audio = request.prompt["multi_modal_data"]["audio"][0]
    assert audio["audex_raw_audio_samples"] is samples
    assert audio["sample_rate"] == 16_000
    assert audio["audex_raw_audio_num_embeddings"] == 73


def test_audio_history_prime_is_exact_prefix_with_one_generated_token() -> None:
    class AudioPrefixTokenizer(PrefixStableTokenizer):
        def encode(self, prompt: str) -> list[int]:
            before, marker, after = prompt.partition("<so_embedding>")
            assert marker
            return [
                *(ord(char) for char in before),
                999,
                *(ord(char) for char in after),
            ]

        def get_vocab(self) -> dict[str, int]:
            return {"<so_embedding>": 999}

    tokenizer = AudioPrefixTokenizer()
    messages = [{"role": "system", "content": "Stable persona."}]
    prefix_ids = build_audio_response_prefix_token_ids(tokenizer, messages)

    request = build_audio_history_prime_request(
        tokenizer,
        messages,
        conversation_state_key="conv:audio",
        conversation_state_prefix_token_count=len(prefix_ids),
        conversation_state_prefix_token_hash="hash-1",
    )

    assert isinstance(request.prompt, dict)
    assert (
        request.prompt["multi_modal_data"]["audio"][0]["audex_raw_audio_num_embeddings"]
        == 1
    )
    assert request.sampling.max_tokens == 1
    assert request.sampling.extra_args[AUDEX_TEXT_STATE_KEY_ARG] == "conv:audio"
    assert (
        request.sampling.extra_args[AUDEX_TEXT_STATE_BOUNDARY_ARG]
        == AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY
    )


def test_asr_request_uses_nvidia_audio_prompt_and_multimodal_data() -> None:
    tokenizer = FakeVllmTokenizer()
    audio = [0.0, 0.1]

    request = build_asr_request(tokenizer, audio, audio_placeholder="<so_embedding>")

    assert tokenizer.enable_thinking is False
    assert tokenizer.messages is not None
    assert tokenizer.messages[1]["content"] == f"{DEFAULT_ASR_PROMPT}\n<so_embedding>"
    assert request.debug_name == "asr"
    assert request.prompt["multi_modal_data"] == {"audio": [(audio, SAMPLE_RATE)]}
    assert request.sampling.as_kwargs() == {
        "max_tokens": DEFAULT_ASR_MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "extra_args": expected_text_modality_guard(),
    }


def test_asr_projected_embeddings_request_uses_audex_embedding_payload() -> None:
    tokenizer = FakeVllmTokenizer()
    projected = type("Projected", (), {"shape": (750, 2048)})()

    request = build_asr_projected_embeddings_request(tokenizer, projected)

    assert tokenizer.enable_thinking is False
    assert tokenizer.messages is not None
    assert tokenizer.messages[1]["content"].count("<so_embedding>") == 750
    assert request.debug_name == "asr-projected"
    assert request.prompt["multi_modal_data"] == {
        "audio": [{"audex_projected_embeddings": projected}]
    }
    assert request.sampling.as_kwargs() == {
        "max_tokens": DEFAULT_ASR_MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "extra_args": expected_text_modality_guard(),
    }


def test_asr_projected_embeddings_request_converts_mlx_array_for_vllm_ipc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeVllmTokenizer()
    fake_mlx_array_type = type("array", (), {"shape": (750, 2048)})
    fake_mlx_array_type.__module__ = "mlx.core"
    projected = fake_mlx_array_type()
    converted = object()

    fake_bridge = types.ModuleType("vllm_metal.pytorch_backend.tensor_bridge")

    def fake_mlx_to_torch(value, *, device):
        assert value is projected
        assert device == "cpu"
        return converted

    fake_bridge.mlx_to_torch = fake_mlx_to_torch
    monkeypatch.setitem(sys.modules, fake_bridge.__name__, fake_bridge)

    request = build_asr_projected_embeddings_request(tokenizer, projected)

    assert request.prompt["multi_modal_data"] == {
        "audio": [{"audex_projected_embeddings": converted}]
    }
    assert make_projected_embeddings_vllm_serializable(projected) is converted


def test_projected_audio_embedding_count_accepts_2d_or_3d_embeddings() -> None:
    two_d = type("Projected2D", (), {"shape": (750, 2048)})()
    three_d = type("Projected3D", (), {"shape": (2, 750, 2048)})()

    assert projected_audio_embedding_count(two_d) == 750
    assert projected_audio_embedding_count(three_d) == 1500


def test_text_response_request_defaults_to_non_thinking_and_4096_tokens() -> None:
    tokenizer = FakeVllmTokenizer()

    request = build_text_response_request(tokenizer, "hello")

    assert tokenizer.enable_thinking is False
    assert tokenizer.messages is not None
    assert DEFAULT_VLLM_TEXT_PROMPT in tokenizer.messages[0]["content"]
    assert tokenizer.messages[1]["content"] == "hello"
    assert request.debug_name == "text"
    assert request.sampling.as_kwargs() == {
        "max_tokens": DEFAULT_TEXT_MAX_TOKENS,
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_args": expected_text_modality_guard(),
    }


def test_text_response_request_can_enable_reasoning_explicitly() -> None:
    tokenizer = FakeVllmTokenizer()

    build_text_response_request(tokenizer, "hello", enable_reasoning=True)

    assert tokenizer.enable_thinking is True


def test_text_messages_response_request_preserves_conversation_history() -> None:
    tokenizer = FakeVllmTokenizer()
    messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Earlier question."},
        {"role": "assistant", "content": "Earlier answer."},
        {"role": "user", "content": "Current question."},
    ]

    request = build_text_messages_response_request(tokenizer, messages)

    assert tokenizer.enable_thinking is False
    assert messages[-1]["content"] == "Current question."
    assert tokenizer.messages is not None
    assert tokenizer.messages[0]["role"] == "system"
    assert tokenizer.messages[0]["content"].startswith("System.")
    assert tokenizer.messages[1:-1] == messages[1:-1]
    assert tokenizer.messages[-1]["role"] == "user"
    assert DEFAULT_VLLM_TEXT_PROMPT in tokenizer.messages[0]["content"]
    assert "NVIDIA" not in tokenizer.messages[0]["content"]
    assert "Nemotron-Cascade" not in tokenizer.messages[0]["content"]
    assert "[CRITICAL]" not in tokenizer.messages[0]["content"]
    assert tokenizer.messages[-1]["content"] == "Current question."
    assert request.debug_name == "text"
    assert request.sampling.as_kwargs() == {
        "max_tokens": DEFAULT_TEXT_MAX_TOKENS,
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_args": expected_text_modality_guard(),
    }


def test_text_messages_response_request_can_carry_conversation_state_hint() -> None:
    tokenizer = FakeVllmTokenizer()
    messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Current question."},
    ]

    request = build_text_messages_response_request(
        tokenizer,
        messages,
        conversation_state_key="conv-1",
        conversation_state_boundary=AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY,
        conversation_state_prefix_token_count=42,
        conversation_state_prefix_token_hash="abc123",
    )

    kwargs = request.sampling.as_kwargs()
    assert kwargs["extra_args"] == {
        **expected_text_modality_guard(),
        AUDEX_TEXT_STATE_KEY_ARG: "conv-1",
        AUDEX_TEXT_STATE_MODE_ARG: AUDEX_TEXT_STATE_APPEND_MODE,
        AUDEX_TEXT_STATE_BOUNDARY_ARG: AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY,
        AUDEX_TEXT_STATE_PREFIX_TOKEN_COUNT_ARG: 42,
        AUDEX_TEXT_STATE_PREFIX_TOKEN_HASH_ARG: "abc123",
    }


def test_text_messages_history_prompt_is_prefix_of_next_generation_prompt() -> None:
    tokenizer = PrefixStableTokenizer()
    history_after_turn = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Earlier question."},
        {"role": "assistant", "content": "Earlier answer."},
    ]
    next_turn_messages = [
        *history_after_turn,
        {"role": "user", "content": "Current question."},
    ]

    history_prompt = build_text_messages_history_prompt(
        tokenizer,
        history_after_turn,
    )
    next_prompt = build_text_messages_generation_prompt(
        tokenizer,
        next_turn_messages,
    )
    history_tokens = tokenizer.encode(history_prompt)
    next_tokens = tokenizer.encode(next_prompt)

    assert next_tokens[: len(history_tokens)] == history_tokens
    assert next_prompt.startswith(history_prompt)


def test_tts_request_ends_at_speechgen_start_and_uses_stop_token_ids() -> None:
    tokenizer = FakeVllmTokenizer()

    request = build_tts_request(
        tokenizer,
        "Hello world.",
        speechgen_end_id=101,
        eos_token_id=tokenizer.eos_token_id,
    )

    assert request.prompt.endswith(SPEECHGEN_START_TOKEN)
    assert request.debug_name == "tts"
    assert request.sampling.as_kwargs() == {
        "max_tokens": DEFAULT_TTS_MAX_TOKENS,
        "temperature": NVIDIA_TTS_TEMPERATURE,
        "top_p": NVIDIA_TTS_TOP_P,
        "detokenize": False,
        "stop_token_ids": [101, 2],
    }
    assert request.sampling.top_k == (NVIDIA_TTS_TOP_K or None)
    assert "allowed_token_ids" not in request.sampling.as_kwargs()


def test_tts_request_can_carry_metal_codec_window_metadata() -> None:
    tokenizer = FakeVllmTokenizer()

    request = build_tts_request(
        tokenizer,
        "Hello world.",
        speechgen_end_id=101,
        eos_token_id=tokenizer.eos_token_id,
        codec_min_id=102,
        codec_max_id=103,
    )

    assert request.sampling.extra_args == {
        "audex_tts_codec_min_id": 102,
        "audex_tts_codec_max_id": 103,
        "audex_tts_speechgen_end_id": 101,
    }
    assert "allowed_token_ids" not in request.sampling.as_kwargs()


def test_tts_request_accepts_diagnostic_sampler_and_seed() -> None:
    tokenizer = FakeVllmTokenizer()

    request = build_tts_request(
        tokenizer,
        "Hello world.",
        speechgen_end_id=101,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.8,
        top_p=1.0,
        top_k=0,
        seed=20260709,
        require_compact_window_decode=True,
    )

    assert request.sampling.temperature == 0.8
    assert request.sampling.top_p == 1.0
    assert request.sampling.top_k is None
    assert request.sampling.seed == 20260709
    assert request.sampling.as_kwargs()["seed"] == 20260709
    assert request.sampling.extra_args == {
        "audex_tts_require_compact_window_decode": True
    }


def test_tts_request_can_carry_metal_lazy_logits_hint() -> None:
    tokenizer = FakeVllmTokenizer()

    request = build_tts_request(
        tokenizer,
        "Hello world.",
        speechgen_end_id=101,
        eos_token_id=tokenizer.eos_token_id,
        skip_paged_logits_eval=True,
    )

    assert request.sampling.extra_args == {
        "audex_tts_skip_paged_logits_eval": True,
    }
    assert "allowed_token_ids" not in request.sampling.as_kwargs()


def test_tts_cfg_requests_create_paired_conditional_and_unconditional_requests() -> (
    None
):
    tokenizer = FakeVllmTokenizer()

    cond, uncond = build_tts_cfg_requests(
        tokenizer,
        "Hello world.",
        speechgen_end_id=101,
        eos_token_id=tokenizer.eos_token_id,
        pair_id="pair-1",
    )

    assert cond.debug_name == "tts-cond"
    assert uncond.debug_name == "tts-uncond"
    assert cond.request_id_suffix == "cond"
    assert uncond.request_id_suffix == "uncond"
    assert cond.prompt.keys() == {"prompt_token_ids"}
    assert uncond.prompt.keys() == {"prompt_token_ids"}
    assert len(cond.prompt["prompt_token_ids"]) == len(
        uncond.prompt["prompt_token_ids"]
    )
    assert cond.sampling.extra_args == {
        "cfg_scale": NVIDIA_TTS_CFG_SCALE,
        "cfg_role": "cond",
        "cfg_pair_id": "pair-1",
    }
    assert cond.sampling.temperature == NVIDIA_TTS_CFG_TEMPERATURE
    assert cond.sampling.top_p == NVIDIA_TTS_CFG_TOP_P
    assert cond.sampling.top_k == NVIDIA_TTS_CFG_TOP_K
    assert cond.sampling.detokenize is False
    assert uncond.sampling.extra_args == {
        "cfg_scale": NVIDIA_TTS_CFG_SCALE,
        "cfg_role": "uncond",
        "cfg_pair_id": "pair-1",
    }
    assert uncond.sampling.temperature == NVIDIA_TTS_CFG_TEMPERATURE
    assert uncond.sampling.top_p == NVIDIA_TTS_CFG_TOP_P
    assert uncond.sampling.top_k == NVIDIA_TTS_CFG_TOP_K
    assert uncond.sampling.detokenize is False
    assert "allowed_token_ids" not in cond.sampling.as_kwargs()
    assert "allowed_token_ids" not in uncond.sampling.as_kwargs()


def test_tts_cfg_requests_accept_matched_sampler_scale_and_seed() -> None:
    tokenizer = FakeVllmTokenizer()

    cond, uncond = build_tts_cfg_requests(
        tokenizer,
        "Hello world.",
        speechgen_end_id=101,
        eos_token_id=tokenizer.eos_token_id,
        pair_id="pair-matched",
        temperature=0.8,
        top_p=1.0,
        top_k=0,
        cfg_scale=2.0,
        seed=20260709,
        require_compact_window_decode=True,
    )

    for request in (cond, uncond):
        assert request.sampling.temperature == 0.8
        assert request.sampling.top_p == 1.0
        assert request.sampling.top_k is None
        assert request.sampling.seed == 20260709
        assert request.sampling.extra_args["cfg_scale"] == 2.0
        assert (
            request.sampling.extra_args["audex_tts_require_compact_window_decode"]
            is True
        )
