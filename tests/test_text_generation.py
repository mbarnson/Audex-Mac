from __future__ import annotations

from types import SimpleNamespace

import pytest

from audex_mac.patches.runtime import AudexPatchReport
from audex_mac.text_benchmark import TextBenchmark
from audex_mac.text_generation import (
    _build_run_log,
    _contains_stop_marker,
    _sampler_config,
    _vllm_turn_record,
    clean_generation,
)

pytestmark = pytest.mark.fast


def test_clean_generation_trims_stop_markers() -> None:
    assert clean_generation("hello<|im_end|>ignored") == "hello"
    assert clean_generation("hello<|end_of_text|>ignored") == "hello"
    assert clean_generation("hello<|eot_id|>ignored") == "hello"


def test_contains_stop_marker_detects_chat_end() -> None:
    assert _contains_stop_marker("hello<|im_end|>") is True
    assert _contains_stop_marker("hello") is False


def test_sampler_config_preserves_benchmark_values() -> None:
    benchmark = TextBenchmark(
        name="test",
        system="system",
        generation={
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 100,
            "max_tokens": 4096,
            "thinking_enabled": False,
        },
        turns=[{"role": "user", "content": "q"} for _ in range(10)],
        pass_criteria=["coherent"],
        sampler_reference="test",
    )

    assert _sampler_config(benchmark) == {
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 100,
        "max_tokens": 4096,
    }


def test_vllm_turn_record_captures_generation_throughput() -> None:
    request_output = SimpleNamespace(prompt_token_ids=[1, 2, 3])
    completion_output = SimpleNamespace(
        token_ids=(10, 11, 12, 13),
        finish_reason="length",
        stop_reason=None,
    )

    record = _vllm_turn_record(
        turn=1,
        user="question",
        assistant="answer",
        elapsed_seconds=2.0,
        request_output=request_output,
        completion_output=completion_output,
    )

    assert record == {
        "turn": 1,
        "user": "question",
        "assistant": "answer",
        "elapsed_seconds": 2.0,
        "prompt_tokens": 3,
        "generation_tokens": 4,
        "generation_tps": 2.0,
        "finish_reason": "length",
        "stop_reason": None,
    }


def test_run_log_records_audex_patch_report() -> None:
    benchmark = TextBenchmark(
        name="test",
        system="system",
        generation={
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 100,
            "max_tokens": 4096,
            "thinking_enabled": False,
        },
        turns=[{"role": "user", "content": "q"} for _ in range(10)],
        pass_criteria=["coherent"],
        sampler_reference="test",
    )
    patch_report = AudexPatchReport(
        transformers_local_dynamic_modules=True,
        mlx_lm_nemotron_dense=True,
        mlx_lm_nemotron_h_audex=True,
        vllm_metal_platform_repair=True,
        vllm_metal_device_info_api=True,
        vllm_metal_nonpaged_capacity=True,
        vllm_nemotron_dense=True,
        vllm_metal_audex_adapter=True,
    )
    preflight = SimpleNamespace(
        model=SimpleNamespace(repo_id="model"),
        model_path="/tmp/model",
        patch_report=patch_report,
    )
    metal_policy = SimpleNamespace(
        env={"VLLM_MLX_DEVICE": "gpu"},
        mlx_metal_available=True,
        mlx_default_device="Device(gpu, 0)",
    )

    run_log = _build_run_log(
        preflight,
        backend="vllm",
        benchmark=benchmark,
        sampler=_sampler_config(benchmark),
        thinking_enabled=False,
        metal_policy=metal_policy,
        started_at=0,
        transcript=[],
    )

    assert run_log["audex_patches"] == {
        "transformers_local_dynamic_modules": True,
        "mlx_lm_nemotron_dense": True,
        "mlx_lm_nemotron_h_audex": True,
        "vllm_metal_platform_repair": True,
        "vllm_metal_device_info_api": True,
        "vllm_metal_nonpaged_capacity": True,
        "vllm_nemotron_dense": True,
        "vllm_metal_audex_adapter": True,
    }
