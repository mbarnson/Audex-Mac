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


def test_text_gate_accepts_observable_benchmark_outcomes() -> None:
    result = evaluate_text_benchmark(load_text_benchmark(), passing_transcript())

    assert result.passed is True
    assert result.failures == ()
    assert result.exact_token_parity_required is False
    assert result.logit_parity_required is False


def test_text_gate_rejects_incorrect_contextual_chunking_answer() -> None:
    transcript = passing_transcript()
    transcript[8]["assistant"] = "[[3, 1, 4], [1, 5, 9], [5, 2]]"

    result = evaluate_text_benchmark(load_text_benchmark(), transcript)

    assert result.passed is False
    assert result.failures == ("turn 9 does not produce [[3, 1, 4], [1, 5, 9], [2]]",)
