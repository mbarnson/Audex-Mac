from __future__ import annotations

import pytest

from audex_mac.text_benchmark import load_text_benchmark
from audex_mac.text_gate import evaluate_text_benchmark

pytestmark = pytest.mark.fast


def passing_transcript() -> list[dict[str, object]]:
    answers = [f"answer {index}" for index in range(1, 11)]
    answers[0] += "\n```python\ndef is_palindrome(s):\n    return True\n```"
    answers[2] += " unicodedata.normalize('NFKD', text)"
    answers[4] += "\n```python\ndef chunked(values, size):\n    yield values\n```"
    answers[5] += " raise ValueError when size <= 0"
    answers[6] += """
```python
import itertools
import unicodedata

def chunked(values, size):
    if size <= 0:
        raise ValueError("size")
    iterator = iter(values)
    while chunk := list(itertools.islice(iterator, size)):
        yield chunk

def is_palindrome(text):
    cleaned = "".join(
        char.lower()
        for char in unicodedata.normalize("NFKD", text)
        if char.isalnum()
    )
    return cleaned == cleaned[::-1]
```
"""
    answers[8] += " [[3, 1, 4], [1, 5, 9], [2]]"
    answers[9] += " chunked is_palindrome O(N) ValueError"
    return [
        {"turn": index, "assistant": answer}
        for index, answer in enumerate(answers, start=1)
    ]


def test_text_assessment_separates_compatible_runtime_and_quality_observations() -> (
    None
):
    result = evaluate_text_benchmark(load_text_benchmark(), passing_transcript())

    assert result.compatible is True
    assert result.compatibility_failures == ()
    assert result.full_benchmark_evaluated is True
    assert all(observation.satisfied for observation in result.quality_observations)
    assert result.exact_token_parity_required is False
    assert result.logit_parity_required is False


def test_text_assessment_records_reasoning_error_without_failing_runtime() -> None:
    transcript = passing_transcript()
    transcript[8]["assistant"] = "[[3, 1, 4], [1, 5, 9], [5, 2]]"

    result = evaluate_text_benchmark(load_text_benchmark(), transcript)

    assert result.compatible is True
    chunking = next(
        observation
        for observation in result.quality_observations
        if observation.name == "contextual_chunking_answer"
    )
    assert chunking.satisfied is False
    assert chunking.detail == ("turn 9 did not produce [[3, 1, 4], [1, 5, 9], [2]]")


def test_text_assessment_rejects_empty_runtime_generation() -> None:
    transcript = passing_transcript()
    transcript[4]["assistant"] = ""

    result = evaluate_text_benchmark(load_text_benchmark(), transcript)

    assert result.compatible is False
    assert result.compatibility_failures == (
        "every benchmark turn must produce a non-empty answer",
    )


def test_text_assessment_rejects_catastrophic_cross_prompt_collapse() -> None:
    transcript = passing_transcript()
    for turn in transcript:
        turn["assistant"] = "the same cached answer"

    result = evaluate_text_benchmark(load_text_benchmark(), transcript)

    assert result.compatible is False
    assert result.compatibility_failures == (
        "benchmark answers collapsed across unrelated prompts",
    )
