from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from audex_mac import vllm_runtime as vllm_runtime_module
from audex_mac.audio_contract import SOUND_TOKEN
from audex_mac.vllm_cfg import AudexVllmCfgConfig
from audex_mac.vllm_runtime import (
    AudexAsyncVllmRuntime,
    AudexVllmRuntime,
    clean_transcription,
    extract_spoken_answer,
    extract_tts_codec_frames,
    split_tts_text_segments,
)
from audex_mac.vllm_sts_requests import (
    AUDEX_TEXT_STATE_APPEND_MODE,
    AUDEX_TEXT_STATE_BOUNDARY_ARG,
    AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY,
    AUDEX_TEXT_STATE_KEY_ARG,
    AUDEX_TEXT_STATE_MODE_ARG,
    AUDEX_TEXT_STATE_PREFIX_TOKEN_COUNT_ARG,
    AUDEX_TEXT_STATE_PREFIX_TOKEN_HASH_ARG,
    VllmSamplingPlan,
    VllmTtsSamplingConfig,
    build_tts_request,
)

pytestmark = pytest.mark.fast


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = None

    def __init__(self) -> None:
        self.enable_thinking_values: list[bool] = []

    def get_vocab(self) -> dict[str, int]:
        return {
            "<speechgen_start>": 100,
            "<speechgen_end>": 101,
            "<speechcodec_0>": 102,
            "<speechcodec_1>": 103,
        }

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
        self.enable_thinking_values.append(enable_thinking)
        turns = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        return turns + "<|im_start|>assistant\n"

    def encode(self, prompt: str) -> list[int]:
        return list(range(10 + prompt.count("<unk>") + prompt.count("hello")))


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[list[object], list[FakeSamplingParams]]] = []

    def generate(self, prompts, sampling_params):
        self.calls.append((list(prompts), list(sampling_params)))
        outputs = []
        for index, _prompt in enumerate(prompts):
            if isinstance(_prompt, dict) and set(_prompt) == {"prompt_token_ids"}:
                text = "<speechcodec_0><speechgen_end>"
                token_ids = (102 + index, 101)
            elif isinstance(_prompt, str) and "<speechgen_start>" in _prompt:
                text = "<speechcodec_0><speechgen_end>"
                token_ids = (102, 101)
            elif index == 0:
                text = "Language: English\n'Transcribed text'<|im_end|>"
                token_ids = (100 + index, 101 + index)
            else:
                text = "<speechgen_start><speechcodec_0><speechgen_end>"
                token_ids = (100 + index, 101 + index)
            outputs.append(
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text=text,
                            token_ids=token_ids,
                            finish_reason="stop",
                        )
                    ]
                )
            )
        return outputs


def expected_text_modality_guard() -> dict[str, object]:
    return {
        "audex_disallow_token_ranges": [[102, 103]],
        "audex_disallow_token_ids": [100, 101],
    }


class FakeAsyncEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[object, FakeSamplingParams, str]] = []
        self.reset_prefix_cache_calls = 0

    async def reset_prefix_cache(self) -> bool:
        self.reset_prefix_cache_calls += 1
        return True

    async def generate(self, prompt, sampling_params, request_id):
        self.calls.append((prompt, sampling_params, request_id))
        if isinstance(prompt, dict) and "multi_modal_data" in prompt:
            yield _fake_request_output(
                text="Language: English\n'Async transcript'<|im_end|>",
                token_ids=(1,),
                finished=True,
            )
            return
        if isinstance(prompt, str) and "<speechgen_start>" not in prompt:
            output_kind = sampling_params.kwargs.get("output_kind")
            delta = (
                output_kind == "DELTA" or getattr(output_kind, "name", None) == "DELTA"
            )
            if delta:
                yield _fake_request_output(
                    text="<think>notes</think>Async ",
                    token_ids=(2,),
                    finished=False,
                )
                yield _fake_request_output(
                    text="response.<|im_end|>",
                    token_ids=(3,),
                    finished=True,
                )
                return
            yield _fake_request_output(
                text="<think>notes</think>Async response.<|im_end|>",
                token_ids=(2,),
                finished=True,
            )
            return
        role = None
        extra_args = sampling_params.kwargs.get("extra_args")
        if extra_args:
            role = extra_args.get("cfg_role")
        output_kind = sampling_params.kwargs.get("output_kind")
        delta = output_kind == "DELTA" or getattr(output_kind, "name", None) == "DELTA"
        if role == "uncond":
            if delta:
                yield _fake_request_output(text="", token_ids=(102,), finished=False)
                yield _fake_request_output(text="", token_ids=(101,), finished=True)
                return
            yield _fake_request_output(text="", token_ids=(102,), finished=False)
            yield _fake_request_output(text="", token_ids=(102, 101), finished=True)
            return
        if delta:
            yield _fake_request_output(text="", token_ids=(102,), finished=False)
            yield _fake_request_output(text="", token_ids=(103,), finished=False)
            yield _fake_request_output(text="", token_ids=(101,), finished=True)
            return
        yield _fake_request_output(text="", token_ids=(102,), finished=False)
        yield _fake_request_output(text="", token_ids=(102, 103), finished=False)
        yield _fake_request_output(text="", token_ids=(102, 103, 101), finished=True)


def make_runtime(
    *,
    tts_sampling_config: VllmTtsSamplingConfig | None = None,
) -> AudexVllmRuntime:
    return AudexVllmRuntime(
        model_path=Path("/tmp/audex/checkpoint_folder_full"),
        tokenizer=FakeTokenizer(),
        engine=FakeEngine(),
        sampling_params_cls=FakeSamplingParams,
        model_load_seconds=1.25,
        tts_sampling_config=tts_sampling_config,
    )


def make_async_runtime() -> AudexAsyncVllmRuntime:
    return AudexAsyncVllmRuntime(
        model_path=Path("/tmp/audex/checkpoint_folder_full"),
        tokenizer=FakeTokenizer(),
        engine=FakeAsyncEngine(),
        sampling_params_cls=FakeSamplingParams,
        model_load_seconds=1.25,
    )


def _fake_request_output(
    *,
    text: str,
    token_ids: tuple[int, ...],
    finished: bool,
):
    return SimpleNamespace(
        finished=finished,
        outputs=[
            SimpleNamespace(
                text=text,
                token_ids=token_ids,
                finish_reason="stop" if finished else None,
            )
        ],
    )


def test_from_model_path_fails_loudly_when_required_cfg_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return FakeTokenizer()

    class FakeLLM:
        def __init__(self, **kwargs):
            raise AssertionError("LLM should not be constructed after CFG failure")

    def fake_configure(engine_kwargs, model_path, *, cfg_scale):
        assert cfg_scale == 3.0
        return AudexVllmCfgConfig(
            enabled=True,
            cfg_scale=3.0,
            script_dir=None,
            logits_processors=(),
            max_model_len=None,
            max_num_batched_tokens=None,
            max_num_seqs=None,
            scheduler_reserve_full_isl=None,
            error="missing inference_scripts_vllm/audiogen_scripts",
        )

    monkeypatch.setattr(
        vllm_runtime_module,
        "apply_audex_runtime_patches",
        lambda: None,
    )
    monkeypatch.setattr(
        vllm_runtime_module,
        "configure_audex_vllm_cfg",
        fake_configure,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    monkeypatch.setenv("AUDEX_VLLM_ENABLE_CFG_WIRING", "1")

    with pytest.raises(RuntimeError, match="Audex vLLM CFG is not ready"):
        AudexVllmRuntime.from_model_path(tmp_path / "checkpoint_folder_full")


def test_vllm_runtime_disables_cfg_wiring_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_cfg_scales = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return FakeTokenizer()

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_configure(engine_kwargs, model_path, *, cfg_scale):
        captured_cfg_scales.append(cfg_scale)
        return AudexVllmCfgConfig(
            enabled=False,
            cfg_scale=cfg_scale,
            script_dir=None,
            logits_processors=(),
            max_model_len=None,
            max_num_batched_tokens=None,
            max_num_seqs=None,
            scheduler_reserve_full_isl=None,
        )

    monkeypatch.delenv("AUDEX_VLLM_ENABLE_CFG_WIRING", raising=False)
    monkeypatch.setattr(
        vllm_runtime_module,
        "apply_audex_runtime_patches",
        lambda: None,
    )
    monkeypatch.setattr(
        vllm_runtime_module,
        "configure_audex_vllm_cfg",
        fake_configure,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )

    runtime = AudexVllmRuntime.from_model_path(tmp_path / "checkpoint_folder_full")

    assert captured_cfg_scales == [0.0]
    assert runtime.stats.cfg_enabled is False


def test_base_engine_kwargs_use_mac_friendly_memory_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("AUDEX_VLLM_GPU_MEMORY_UTILIZATION", raising=False)

    kwargs = vllm_runtime_module._base_engine_kwargs(
        tmp_path / "checkpoint_folder_full",
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=vllm_runtime_module.DEFAULT_GPU_MEMORY_UTILIZATION,
    )

    assert kwargs["gpu_memory_utilization"] == 0.60
    assert kwargs["max_model_len"] == 262_144


def test_base_engine_kwargs_clamp_context_to_checkpoint_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("AUDEX_VLLM_GPU_MEMORY_UTILIZATION", raising=False)
    model_path = tmp_path / "checkpoint_folder_full"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"max_position_embeddings": 131072}',
        encoding="utf-8",
    )

    kwargs = vllm_runtime_module._base_engine_kwargs(
        model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=vllm_runtime_module.DEFAULT_GPU_MEMORY_UTILIZATION,
    )

    assert kwargs["max_model_len"] == 131_072


def test_base_engine_kwargs_allow_memory_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_GPU_MEMORY_UTILIZATION", "0.66")

    kwargs = vllm_runtime_module._base_engine_kwargs(
        tmp_path / "checkpoint_folder_full",
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=vllm_runtime_module.DEFAULT_GPU_MEMORY_UTILIZATION,
    )

    assert kwargs["gpu_memory_utilization"] == 0.66


def test_base_engine_kwargs_reject_invalid_memory_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_GPU_MEMORY_UTILIZATION", "1.5")

    with pytest.raises(ValueError, match="AUDEX_VLLM_GPU_MEMORY_UTILIZATION"):
        vllm_runtime_module._base_engine_kwargs(
            tmp_path / "checkpoint_folder_full",
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=vllm_runtime_module.DEFAULT_GPU_MEMORY_UTILIZATION,
        )


def test_base_engine_kwargs_can_omit_memory_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_GPU_MEMORY_UTILIZATION", "0.66")

    kwargs = vllm_runtime_module._base_engine_kwargs(
        tmp_path / "checkpoint_folder_full",
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=None,
    )

    assert "gpu_memory_utilization" not in kwargs


def test_async_vllm_runtime_stream_many_yields_token_deltas() -> None:
    async def run_test():
        runtime = make_async_runtime()
        request = build_tts_request(
            runtime.tokenizer,
            "hello",
            speechgen_end_id=runtime.token_map.speechgen_end,
            eos_token_id=runtime.tokenizer.eos_token_id,
        )
        deltas = [delta async for delta in runtime.stream_many((request,))]

        assert [delta.new_token_ids for delta in deltas] == [
            (102,),
            (103,),
            (101,),
        ]
        assert deltas[-1].finished is True
        assert runtime.engine.calls[0][2].startswith("audex-tts-")

    asyncio.run(run_test())


def test_async_vllm_runtime_tts_cfg_requests_match_nvidia_extra_args() -> None:
    async def run_test():
        runtime = make_async_runtime()
        async for _event in runtime.stream_tts_cfg_codec_frames(
            "hello",
            pair_id="pair-1",
            max_tokens=8,
        ):
            pass

        cond_sampling = runtime.engine.calls[0][1].kwargs
        uncond_sampling = runtime.engine.calls[1][1].kwargs
        assert cond_sampling["extra_args"] == {
            "audex_tts_codec_min_id": 102,
            "audex_tts_codec_max_id": 103,
            "audex_tts_speechgen_end_id": 101,
            "cfg_scale": 3.0,
            "cfg_role": "cond",
            "cfg_pair_id": "pair-1",
        }
        assert uncond_sampling["extra_args"] == {
            "audex_tts_codec_min_id": 102,
            "audex_tts_codec_max_id": 103,
            "audex_tts_speechgen_end_id": 101,
            "cfg_scale": 3.0,
            "cfg_role": "uncond",
            "cfg_pair_id": "pair-1",
        }
        for sampling in (cond_sampling, uncond_sampling):
            assert {
                key: sampling[key]
                for key in ("temperature", "top_p", "top_k", "output_kind")
            } == {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 80,
                "output_kind": "DELTA",
            }

    asyncio.run(run_test())


def test_async_vllm_runtime_streams_tts_cfg_codec_frames_from_conditional() -> None:
    async def run_test():
        runtime = make_async_runtime()
        events = [
            event
            async for event in runtime.stream_tts_cfg_codec_frames(
                "hello",
                pair_id="pair-1",
                max_tokens=8,
            )
        ]

        assert [event.new_codec_frames for event in events] == [
            (0,),
            (1,),
            (),
        ]
        assert [event.generated_token_ids for event in events] == [
            (),
            (),
            (102, 103, 101),
        ]
        assert events[-1].reached_end_token is True
        assert events[-1].finished is True
        request_ids = [call[2] for call in runtime.engine.calls]
        assert any(
            request_id.startswith("audex-tts-cond-") for request_id in request_ids
        )
        assert any(
            request_id.startswith("audex-tts-uncond-") for request_id in request_ids
        )

    asyncio.run(run_test())


def test_split_tts_text_segments_targets_four_ordered_chunks() -> None:
    segments = split_tts_text_segments(
        "Context managers handle setup and cleanup around resources. "
        "They keep files, locks, and transactions tidy even when errors happen.",
        target_segments=4,
    )

    assert len(segments) == 4
    joined = " ".join(segment.rstrip(",") for segment in segments)
    for expected in ("Context", "resources.", "They", "happen."):
        assert expected in joined


def test_async_vllm_runtime_streams_segmented_tts_in_order() -> None:
    async def run_test():
        runtime = make_async_runtime()
        events = [
            event
            async for event in runtime.stream_tts_cfg_segmented_codec_frames(
                "Context managers handle setup and cleanup around resources. "
                "They keep files, locks, and transactions tidy even when errors happen.",
                pair_id="pair-1",
                max_tokens=64,
                target_segments=4,
            )
        ]

        emitted_frames = [frame for event in events for frame in event.new_codec_frames]
        assert emitted_frames == [0, 1, 0, 1, 0, 1, 0, 1]
        assert [event.generated_token_ids for event in events] == [
            (),
            (),
            (102, 103, 101),
            (),
            (),
            (102, 103, 101),
            (),
            (),
            (102, 103, 101),
            (),
            (),
            (102, 103, 101),
        ]
        assert [event.reached_end_token for event in events] == [
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
        ]
        assert events[-1].reached_end_token is True
        assert events[-1].finished is True
        request_ids = [call[2] for call in runtime.engine.calls]
        assert len(request_ids) == 8
        assert any("tts-segment-0-cond" in request_id for request_id in request_ids)
        assert any("tts-segment-3-uncond" in request_id for request_id in request_ids)
        assert all(
            call[1].kwargs["output_kind"] == "DELTA" for call in runtime.engine.calls
        )

    asyncio.run(run_test())


def test_async_vllm_runtime_streams_explicit_cfg_segments_in_order() -> None:
    async def run_test():
        runtime = make_async_runtime()
        events = [
            event
            async for event in runtime.stream_tts_cfg_segments_codec_frames(
                ("First chunk.", "Second chunk."),
                pair_id="pair-2",
                max_tokens_per_segment=(32, 48),
            )
        ]

        emitted_frames = [frame for event in events for frame in event.new_codec_frames]
        assert emitted_frames == [0, 1, 0, 1]
        assert [event.generated_token_ids for event in events] == [
            (),
            (),
            (102, 103, 101),
            (),
            (),
            (102, 103, 101),
        ]
        assert [event.reached_end_token for event in events] == [
            False,
            False,
            True,
            False,
            False,
            True,
        ]
        request_ids = [call[2] for call in runtime.engine.calls]
        assert len(request_ids) == 4
        assert any("tts-segment-0-cond" in request_id for request_id in request_ids)
        assert any("tts-segment-1-uncond" in request_id for request_id in request_ids)
        max_tokens = [call[1].kwargs["max_tokens"] for call in runtime.engine.calls]
        assert max_tokens == [32, 32, 48, 48]
        assert all(
            call[1].kwargs["output_kind"] == "DELTA" for call in runtime.engine.calls
        )

    asyncio.run(run_test())


def test_async_vllm_runtime_streams_no_cfg_segmented_tts_in_order() -> None:
    async def run_test():
        runtime = make_async_runtime()
        events = [
            event
            async for event in runtime.stream_tts_segmented_codec_frames(
                ("First chunk.", "Second chunk."),
                max_tokens_per_segment=(32, 32),
            )
        ]

        emitted_frames = [frame for event in events for frame in event.new_codec_frames]
        assert emitted_frames == [0, 1, 0, 1]
        assert [event.generated_token_ids for event in events] == [
            (),
            (),
            (102, 103, 101),
            (),
            (),
            (102, 103, 101),
        ]
        assert events[-1].reached_end_token is True
        assert events[-1].finished is True
        request_ids = [call[2] for call in runtime.engine.calls]
        assert len(request_ids) == 2
        assert any("tts-segment-0" in request_id for request_id in request_ids)
        assert any("tts-segment-1" in request_id for request_id in request_ids)
        samplings = [call[1].kwargs for call in runtime.engine.calls]
        assert all(sampling["output_kind"] == "DELTA" for sampling in samplings)

    asyncio.run(run_test())


def test_async_vllm_runtime_transcribes_projected_audio_final_output() -> None:
    async def run_test():
        runtime = make_async_runtime()

        result = await runtime.transcribe_projected_audio(
            SimpleNamespace(shape=(3, 2048)),
            num_embeddings=3,
        )

        assert result.text == "Async transcript"
        assert result.request_debug_name == "asr-projected"
        prompt = runtime.engine.calls[0][0]
        assert isinstance(prompt, dict)
        assert "multi_modal_data" in prompt
        assert prompt["prompt"].count(SOUND_TOKEN) == 3

    asyncio.run(run_test())


def test_async_vllm_runtime_resets_prefix_cache() -> None:
    async def run_test():
        runtime = make_async_runtime()

        assert await runtime.reset_prefix_cache() is True
        assert runtime.engine.reset_prefix_cache_calls == 1

    asyncio.run(run_test())


def test_async_vllm_runtime_generates_text_response_from_messages_final_output() -> (
    None
):
    async def run_test():
        runtime = make_async_runtime()
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Earlier question."},
            {"role": "assistant", "content": "Earlier answer."},
            {"role": "user", "content": "Current question."},
        ]

        result = await runtime.generate_text_response_from_messages(messages)

        assert result.text == "Async response."
        assert result.request_debug_name == "text"
        prompt = runtime.engine.calls[0][0]
        assert "Earlier question." in prompt
        assert "Earlier answer." in prompt
        assert "Current question." in prompt
        assert runtime.tokenizer.enable_thinking_values[-1] is False

    asyncio.run(run_test())


def test_async_vllm_runtime_passes_text_conversation_state_hint() -> None:
    async def run_test():
        runtime = make_async_runtime()
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Current question."},
        ]

        await runtime.generate_text_response_from_messages(
            messages,
            conversation_state_key="conv-1",
            conversation_state_boundary=AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY,
            conversation_state_prefix_token_count=12,
            conversation_state_prefix_token_hash="hash-1",
        )

        extra_args = runtime.engine.calls[0][1].kwargs["extra_args"]
        assert extra_args[AUDEX_TEXT_STATE_KEY_ARG] == "conv-1"
        assert extra_args[AUDEX_TEXT_STATE_MODE_ARG] == AUDEX_TEXT_STATE_APPEND_MODE
        assert extra_args[AUDEX_TEXT_STATE_BOUNDARY_ARG] == (
            AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY
        )
        assert extra_args[AUDEX_TEXT_STATE_PREFIX_TOKEN_COUNT_ARG] == 12
        assert extra_args[AUDEX_TEXT_STATE_PREFIX_TOKEN_HASH_ARG] == "hash-1"

    asyncio.run(run_test())


def test_async_vllm_runtime_streams_text_response_from_messages() -> None:
    async def run_test():
        runtime = make_async_runtime()
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Current question."},
        ]

        deltas = [
            delta
            async for delta in runtime.stream_text_response_from_messages(messages)
        ]

        assert deltas[-1].text == "Async response."
        assert [delta.text for delta in deltas] == ["Async", "Async response."]
        assert deltas[-1].token_ids == (2, 3)
        assert deltas[-1].new_token_ids == (3,)
        assert deltas[-1].finished is True
        assert runtime.engine.calls[0][1].kwargs["output_kind"] == "DELTA"

    asyncio.run(run_test())


def test_async_vllm_runtime_generate_tts_cfg_pair_returns_final_outputs() -> None:
    async def run_test():
        runtime = make_async_runtime()

        cond, uncond = await runtime.generate_tts_cfg_pair(
            "hello",
            pair_id="pair-1",
            max_tokens=8,
        )

        assert cond.request_debug_name == "tts-cond"
        assert cond.token_ids == (102, 103, 101)
        assert uncond.request_debug_name == "tts-uncond"
        assert uncond.token_ids == (102, 101)
        extra_args = [call[1].kwargs["extra_args"] for call in runtime.engine.calls]
        assert extra_args[0]["cfg_role"] == "cond"
        assert extra_args[0]["cfg_pair_id"] == "pair-1"
        assert extra_args[1]["cfg_role"] == "uncond"
        assert extra_args[1]["cfg_pair_id"] == "pair-1"

    asyncio.run(run_test())


def test_async_vllm_runtime_generate_tts_uses_single_non_cfg_request() -> None:
    async def run_test():
        runtime = make_async_runtime()

        result = await runtime.generate_tts("hello", max_tokens=8)

        assert result.request_debug_name == "tts"
        assert result.token_ids == (102, 103, 101)
        assert len(runtime.engine.calls) == 1
        sampling = runtime.engine.calls[0][1].kwargs
        assert sampling["extra_args"] == {
            "audex_tts_codec_min_id": 102,
            "audex_tts_codec_max_id": 103,
            "audex_tts_speechgen_end_id": 101,
            "audex_tts_skip_paged_logits_eval": True,
        }
        assert sampling["detokenize"] is False
        assert "output_kind" not in sampling

    asyncio.run(run_test())


def test_async_vllm_runtime_streams_tts_codec_frames_without_cfg() -> None:
    async def run_test():
        runtime = make_async_runtime()
        events = [
            event
            async for event in runtime.stream_tts_codec_frames("hello", max_tokens=8)
        ]

        assert [event.new_codec_frames for event in events] == [
            (0,),
            (1,),
            (),
        ]
        assert [event.generated_token_ids for event in events] == [
            (),
            (),
            (102, 103, 101),
        ]
        assert events[-1].reached_end_token is True
        assert events[-1].finished is True
        assert len(runtime.engine.calls) == 1
        sampling = runtime.engine.calls[0][1].kwargs
        assert sampling["extra_args"] == {
            "audex_tts_codec_min_id": 102,
            "audex_tts_codec_max_id": 103,
            "audex_tts_speechgen_end_id": 101,
            "audex_tts_skip_paged_logits_eval": True,
        }
        assert sampling["detokenize"] is False
        assert sampling["output_kind"] == "DELTA"

    asyncio.run(run_test())


def test_async_vllm_runtime_stream_many_accumulates_delta_outputs() -> None:
    async def run_test():
        runtime = make_async_runtime()
        request = runtime.build_tts_request("hello", max_tokens=8)
        request = replace(
            request,
            sampling=replace(request.sampling, output_kind="DELTA"),
        )

        events = [event async for event in runtime.stream_many((request,))]

        assert [event.new_token_ids for event in events] == [
            (102,),
            (103,),
            (101,),
        ]
        assert [event.token_ids for event in events] == [
            (102,),
            (102, 103),
            (102, 103, 101),
        ]
        sampling = runtime.engine.calls[0][1].kwargs
        assert sampling["output_kind"] == "DELTA"

    asyncio.run(run_test())


def test_async_vllm_runtime_stream_many_can_skip_cumulative_delta_outputs() -> None:
    async def run_test():
        runtime = make_async_runtime()
        request = runtime.build_tts_request("hello", max_tokens=8)
        request = replace(
            request,
            sampling=replace(request.sampling, output_kind="DELTA"),
        )

        events = [
            event
            async for event in runtime.stream_many(
                (request,),
                include_cumulative_token_ids=False,
            )
        ]

        assert [event.new_token_ids for event in events] == [
            (102,),
            (103,),
            (101,),
        ]
        assert [event.token_ids for event in events] == [(), (), ()]
        sampling = runtime.engine.calls[0][1].kwargs
        assert sampling["output_kind"] == "DELTA"

    asyncio.run(run_test())


def test_async_vllm_runtime_stream_many_detects_delta_enum_like_output_kind() -> None:
    async def run_test():
        runtime = make_async_runtime()
        request = runtime.build_tts_request("hello", max_tokens=8)
        request = replace(
            request,
            sampling=replace(
                request.sampling,
                output_kind=SimpleNamespace(name="DELTA"),
            ),
        )

        events = [event async for event in runtime.stream_many((request,))]

        assert [event.new_token_ids for event in events] == [
            (102,),
            (103,),
            (101,),
        ]
        assert [event.token_ids for event in events] == [
            (102,),
            (102, 103),
            (102, 103, 101),
        ]

    asyncio.run(run_test())


def test_sampling_kwargs_converts_delta_output_kind_to_vllm_enum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_sampling_params = ModuleType("vllm.sampling_params")
    marker = object()
    fake_sampling_params.RequestOutputKind = {"DELTA": marker}
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sampling_params)

    kwargs = vllm_runtime_module._sampling_kwargs(
        VllmSamplingPlan(
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            output_kind="DELTA",
        )
    )

    assert kwargs["output_kind"] is marker


def test_async_vllm_runtime_stats_match_sync_runtime_contract() -> None:
    runtime = make_async_runtime()

    stats = runtime.stats

    assert stats.model_load_seconds == 1.25
    assert stats.engine_class.endswith(".FakeAsyncEngine")
    assert stats.model_path == Path("/tmp/audex/checkpoint_folder_full")
    assert stats.cfg_enabled is False
    assert stats.cfg_script_dir is None
    assert stats.cfg_scheduler_reserve_full_isl is None
    assert stats.nonpaged_kv_capacity_seqs is None
    assert stats.max_model_len == 262_144


def test_async_vllm_runtime_stats_include_nonpaged_capacity_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS", "4")
    runtime = make_async_runtime()

    assert runtime.stats.nonpaged_kv_capacity_seqs == 4


def test_clean_transcription_extracts_quoted_asr_text() -> None:
    assert clean_transcription("Language: English\n'hello there'<|im_end|>") == (
        "hello there"
    )


def test_extract_spoken_answer_removes_thinking_block() -> None:
    assert extract_spoken_answer("<think>notes</think>Final answer.") == (
        "Final answer."
    )


def test_extract_spoken_answer_removes_audex_prompt_leakage() -> None:
    assert (
        extract_spoken_answer(
            "Audex was built by NVIDIA based on the Nemotron-Cascade-2 architecture.\n"
            "It is ready to help you with conversations about code and ideas.\n\n"
            "[CRITICAL] Place each sentence on its own separate line.\n\n"
            "Go routines are lightweight threads managed by the Go runtime."
        )
        == "Go routines are lightweight threads managed by the Go runtime."
    )


def test_extract_spoken_answer_removes_observed_audex_identity_leakage() -> None:
    assert (
        extract_spoken_answer(
            "Audex is created by NVIDIA based on the Nemotron-Cascade-2 architecture.  \n"
            "The language design and tooling for safety make it attractive to developers.  \n"
            "Rust's borrow checker enforces memory safety at compile time without a garbage collector."
        )
        == "The language design and tooling for safety make it attractive to developers.\n"
        "Rust's borrow checker enforces memory safety at compile time without a garbage collector."
    )
    assert (
        extract_spoken_answer(
            "Audex is a conversational partner created by NVIDIA based on the Nemotron-Cascade-2 architecture.  \n"
            "It can help you think through technical decisions and compare languages.  \n"
            "You have written a privacy server in Go that uses protobufs."
        )
        == "You have written a privacy server in Go that uses protobufs."
    )
    assert (
        extract_spoken_answer(
            "Audex is created by NVIDIA based upon the Nemotron-Cascade-2 architecture.\n"
            "Rust prioritizes compile-time safety, Go prioritizes runtime simplicity.\n"
            "Your turn Matt"
        )
        == "Rust prioritizes compile-time safety, Go prioritizes runtime simplicity."
    )


def test_vllm_runtime_transcribes_audio_through_persistent_engine() -> None:
    runtime = make_runtime()

    result = runtime.transcribe_audio([0.0, 0.1])

    assert result.text == "Transcribed text"
    assert result.request_debug_name == "asr"
    assert runtime.engine.calls[0][0][0]["multi_modal_data"]["audio"][0][1] == 16000
    assert runtime.engine.calls[0][1][0].kwargs == {
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_p": 1.0,
        "extra_args": expected_text_modality_guard(),
    }


def test_vllm_runtime_transcribes_projected_audio_embeddings() -> None:
    runtime = make_runtime()
    projected = SimpleNamespace(shape=(750, 2048))

    result = runtime.transcribe_projected_audio(projected, num_embeddings=750)

    assert result.text == "Transcribed text"
    assert result.request_debug_name == "asr-projected"
    prompt = runtime.engine.calls[0][0][0]
    assert prompt["multi_modal_data"] == {
        "audio": [{"audex_projected_embeddings": projected}]
    }


def test_vllm_runtime_generates_text_response_non_thinking_by_default() -> None:
    runtime = make_runtime()

    result = runtime.generate_text_response("hello")

    assert result.request_debug_name == "text"
    assert runtime.tokenizer.enable_thinking_values[-1] is False
    assert runtime.engine.calls[0][1][0].kwargs["max_tokens"] == 4096


def test_vllm_runtime_generates_text_response_from_conversation_messages() -> None:
    runtime = make_runtime()
    messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Earlier question."},
        {"role": "assistant", "content": "Earlier answer."},
        {"role": "user", "content": "Current question."},
    ]

    result = runtime.generate_text_response_from_messages(messages)

    assert result.request_debug_name == "text"
    prompt = runtime.engine.calls[0][0][0]
    assert "Earlier question." in prompt
    assert "Earlier answer." in prompt
    assert "Current question." in prompt
    assert runtime.tokenizer.enable_thinking_values[-1] is False


def test_vllm_runtime_submits_tts_cfg_pair_to_one_engine_call() -> None:
    runtime = make_runtime()

    cond, uncond = runtime.generate_tts_cfg_pair("hello", pair_id="pair-1")
    codec = runtime.extract_tts_codec_frames(cond)

    prompts, sampling = runtime.engine.calls[0]
    assert cond.request_debug_name == "tts-cond"
    assert uncond.request_debug_name == "tts-uncond"
    assert codec.generated_codec_frames == (0,)
    assert codec.reached_end_token is True
    assert len(prompts) == 2
    assert set(prompts[0]) == {"prompt_token_ids"}
    assert set(prompts[1]) == {"prompt_token_ids"}
    assert sampling[0].kwargs["extra_args"] == {
        "audex_tts_codec_min_id": 102,
        "audex_tts_codec_max_id": 103,
        "audex_tts_speechgen_end_id": 101,
        "cfg_scale": 3.0,
        "cfg_role": "cond",
        "cfg_pair_id": "pair-1",
    }
    assert sampling[1].kwargs["extra_args"] == {
        "audex_tts_codec_min_id": 102,
        "audex_tts_codec_max_id": 103,
        "audex_tts_speechgen_end_id": 101,
        "cfg_scale": 3.0,
        "cfg_role": "uncond",
        "cfg_pair_id": "pair-1",
    }
    assert sampling[0].kwargs["temperature"] == 1.0
    assert sampling[0].kwargs["top_p"] == 1.0
    assert sampling[0].kwargs["top_k"] == 80
    assert sampling[1].kwargs["temperature"] == 1.0
    assert sampling[1].kwargs["top_p"] == 1.0
    assert sampling[1].kwargs["top_k"] == 80


def test_vllm_runtime_applies_diagnostic_tts_recipe_to_real_requests() -> None:
    runtime = make_runtime(
        tts_sampling_config=VllmTtsSamplingConfig(
            temperature=0.8,
            top_p=1.0,
            top_k=0,
            cfg_scale=2.0,
            seed=20260709,
            require_compact_window_decode=True,
        )
    )

    cond, uncond = runtime.generate_tts_cfg_pair("hello", pair_id="pair-matched")

    _prompts, sampling = runtime.engine.calls[0]
    assert cond.request_debug_name == "tts-cond"
    assert uncond.request_debug_name == "tts-uncond"
    for params in sampling:
        assert params.kwargs["temperature"] == 0.8
        assert params.kwargs["top_p"] == 1.0
        assert "top_k" not in params.kwargs
        assert params.kwargs["seed"] == 20260709
        assert params.kwargs["extra_args"]["cfg_scale"] == 2.0
        assert (
            params.kwargs["extra_args"]["audex_tts_require_compact_window_decode"]
            is True
        )


def test_runtime_ignores_ambient_tts_sampler_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_TTS_TEMPERATURE", "0.0")
    monkeypatch.setenv("AUDEX_VLLM_TTS_TOP_K", "99")
    runtime = make_runtime()

    runtime.generate_tts("hello")

    _prompts, sampling = runtime.engine.calls[0]
    assert sampling[0].kwargs["temperature"] == 0.8
    assert "top_k" not in sampling[0].kwargs


def test_vllm_runtime_generates_tts_with_single_non_cfg_request() -> None:
    runtime = make_runtime()

    result = runtime.generate_tts("hello")
    codec = runtime.extract_tts_codec_frames(result)

    prompts, sampling = runtime.engine.calls[0]
    assert result.request_debug_name == "tts"
    assert codec.generated_codec_frames == (0,)
    assert codec.reached_end_token is True
    assert len(prompts) == 1
    assert isinstance(prompts[0], str)
    assert "<speechgen_start>" in prompts[0]
    assert sampling[0].kwargs["extra_args"] == {
        "audex_tts_codec_min_id": 102,
        "audex_tts_codec_max_id": 103,
        "audex_tts_speechgen_end_id": 101,
        "audex_tts_skip_paged_logits_eval": True,
    }
    assert sampling[0].kwargs["detokenize"] is False


def test_vllm_runtime_stats_identifies_engine_and_model_path() -> None:
    runtime = make_runtime()

    stats = runtime.stats

    assert stats.model_load_seconds == 1.25
    assert stats.engine_class.endswith(".FakeEngine")
    assert stats.model_path == Path("/tmp/audex/checkpoint_folder_full")
    assert stats.cfg_enabled is False
    assert stats.cfg_script_dir is None


def test_extract_tts_codec_frames_maps_vllm_generated_ids() -> None:
    runtime = make_runtime()
    codec_token_ids = {
        codec_frame: token_id
        for token_id, codec_frame in runtime.token_map.speech_codec.items()
    }

    codec = extract_tts_codec_frames(
        (codec_token_ids[0], 999, runtime.token_map.speechgen_end),
        runtime.token_map,
    )

    assert codec.generated_codec_frames == (0,)
    assert codec.reached_end_token is True


def test_extract_tts_codec_frames_ignores_optional_speechgen_start() -> None:
    runtime = make_runtime()
    codec_token_ids = {
        codec_frame: token_id
        for token_id, codec_frame in runtime.token_map.speech_codec.items()
    }

    codec = extract_tts_codec_frames(
        (
            runtime.token_map.speechgen_start,
            codec_token_ids[1],
            runtime.token_map.speechgen_end,
        ),
        runtime.token_map,
    )

    assert codec.generated_codec_frames == (1,)
    assert codec.reached_end_token is True
