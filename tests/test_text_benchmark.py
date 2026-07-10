from __future__ import annotations

import pytest

from audex_mac.text_benchmark import (
    MIN_MAX_TOKENS,
    MIN_TURNS,
    TextBenchmark,
    load_text_benchmark,
    validate_text_benchmark,
)

pytestmark = pytest.mark.fast


def test_text_benchmark_contract_loads() -> None:
    benchmark = load_text_benchmark()

    assert len(benchmark.turns) >= MIN_TURNS
    assert benchmark.max_tokens >= MIN_MAX_TOKENS
    assert benchmark.generation["temperature"] == 1.0
    assert benchmark.generation["top_p"] == 0.95
    assert benchmark.generation["thinking_enabled"] is False
    assert "run_text_vllm_example.py" in benchmark.sampler_reference


def test_text_benchmark_rejects_too_few_turns() -> None:
    benchmark = TextBenchmark(
        name="bad",
        system="system",
        generation={
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 100,
            "max_tokens": 4096,
            "thinking_enabled": False,
        },
        turns=[{"role": "user", "content": "x"}],
        pass_criteria=[],
        sampler_reference="ref",
    )

    with pytest.raises(ValueError, match="at least"):
        validate_text_benchmark(benchmark)


def test_text_benchmark_rejects_short_max_tokens() -> None:
    benchmark = TextBenchmark(
        name="bad",
        system="system",
        generation={
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 100,
            "max_tokens": 1024,
            "thinking_enabled": False,
        },
        turns=[{"role": "user", "content": str(i)} for i in range(10)],
        pass_criteria=[],
        sampler_reference="ref",
    )

    with pytest.raises(ValueError, match="max_tokens"):
        validate_text_benchmark(benchmark)


def test_text_benchmark_rejects_default_thinking() -> None:
    benchmark = TextBenchmark(
        name="bad",
        system="system",
        generation={
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 100,
            "max_tokens": 4096,
            "thinking_enabled": True,
        },
        turns=[{"role": "user", "content": str(i)} for i in range(10)],
        pass_criteria=[],
        sampler_reference="ref",
    )

    with pytest.raises(ValueError, match="non-thinking"):
        validate_text_benchmark(benchmark)
