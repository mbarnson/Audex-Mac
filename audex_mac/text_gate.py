"""Deterministic acceptance gate for the checked-in text benchmark."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from .text_benchmark import TextBenchmark


@dataclass(frozen=True, slots=True)
class TextGateResult:
    failures: tuple[str, ...]
    evaluated: bool = True
    exact_token_parity_required: bool = False
    logit_parity_required: bool = False

    @property
    def passed(self) -> bool:
        return self.evaluated and not self.failures


def evaluate_text_benchmark(
    benchmark: TextBenchmark,
    transcript: list[dict[str, Any]],
) -> TextGateResult:
    """Evaluate observable outcomes promised by the benchmark conversation."""

    failures: list[str] = []
    if len(transcript) != len(benchmark.turns):
        failures.append(
            f"expected {len(benchmark.turns)} turns, observed {len(transcript)}"
        )
        return TextGateResult(tuple(failures))

    answers = [str(turn.get("assistant", "")).strip() for turn in transcript]
    if any(not answer for answer in answers):
        failures.append("every benchmark turn must produce a non-empty answer")
    if len(set(answers)) != len(answers):
        failures.append("benchmark contains a repeated assistant answer")

    combined = "\n".join(answers).lower()
    if "unicodedata" not in combined or "normalize" not in combined:
        failures.append("accent handling does not retain a unicodedata normalization")
    if "valueerror" not in combined or not re.search(r"size\s*<=\s*0", combined):
        failures.append("chunked does not deliberately reject size <= 0")

    expected_chunks = "[[3,1,4],[1,5,9],[2]]"
    turn_nine = re.sub(r"\s+", "", answers[8])
    if expected_chunks not in turn_nine:
        failures.append("turn 9 does not produce [[3, 1, 4], [1, 5, 9], [2]]")

    final = answers[-1].lower()
    for required in ("chunked", "is_palindrome", "o(n)", "valueerror"):
        if required not in final:
            failures.append(f"final summary does not retain {required!r}")

    for turn_number in (1, 3, 5, 7, 8):
        for block in _python_fences(answers[turn_number - 1]):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                failures.append(
                    f"turn {turn_number} contains invalid Python: {exc.msg}"
                )

    module_source = next(
        (
            block
            for turn_number in (8, 7)
            for block in reversed(_python_fences(answers[turn_number - 1]))
            if "def chunked" in block and "def is_palindrome" in block
        ),
        None,
    )
    if module_source is None:
        failures.append("reviewed mini-module does not define both benchmark APIs")
    else:
        compact_module = re.sub(r"\s+", "", module_source.lower())
        for required in (
            "unicodedata.normalize",
            "[::-1]",
            "itertools.islice",
            "size<=0",
            "raisevalueerror",
            "yieldchunk",
        ):
            if required not in compact_module:
                failures.append(
                    f"reviewed mini-module lacks required behavior {required!r}"
                )

    return TextGateResult(tuple(failures))


def _python_fences(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL))
