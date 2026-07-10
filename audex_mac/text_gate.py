"""Runtime compatibility and quality assessment for the text benchmark."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from .text_benchmark import TextBenchmark


@dataclass(frozen=True, slots=True)
class TextQualityObservation:
    name: str
    satisfied: bool
    detail: str


@dataclass(frozen=True, slots=True)
class TextBenchmarkAssessment:
    compatibility_failures: tuple[str, ...]
    quality_observations: tuple[TextQualityObservation, ...]
    full_benchmark_evaluated: bool = True
    exact_token_parity_required: bool = False
    logit_parity_required: bool = False

    @property
    def compatible(self) -> bool:
        return not self.compatibility_failures


def evaluate_text_benchmark(
    benchmark: TextBenchmark,
    transcript: list[dict[str, Any]],
) -> TextBenchmarkAssessment:
    """Separate runtime compatibility failures from model-quality observations."""

    compatibility_failures: list[str] = []
    if len(transcript) != len(benchmark.turns):
        compatibility_failures.append(
            f"expected {len(benchmark.turns)} turns, observed {len(transcript)}"
        )
        return TextBenchmarkAssessment(tuple(compatibility_failures), ())

    answers = [str(turn.get("assistant", "")).strip() for turn in transcript]
    if any(not answer for answer in answers):
        compatibility_failures.append(
            "every benchmark turn must produce a non-empty answer"
        )
    distinct_answers = len(set(answer for answer in answers if answer))
    catastrophic_distinct_floor = max(1, len(answers) // 5)
    if answers and distinct_answers <= catastrophic_distinct_floor:
        compatibility_failures.append(
            "benchmark answers collapsed across unrelated prompts"
        )

    observations: list[TextQualityObservation] = []
    combined = "\n".join(answers).lower()
    normalization_present = "unicodedata" in combined and "normalize" in combined
    observations.append(
        TextQualityObservation(
            name="accent_normalization",
            satisfied=normalization_present,
            detail=(
                "a unicodedata normalization approach was present"
                if normalization_present
                else "no unicodedata normalization approach was found"
            ),
        )
    )
    size_guard_present = "valueerror" in combined and bool(
        re.search(r"size\s*<=\s*0", combined)
    )
    observations.append(
        TextQualityObservation(
            name="chunk_size_guard",
            satisfied=size_guard_present,
            detail=(
                "chunked deliberately rejected size <= 0"
                if size_guard_present
                else "chunked did not clearly reject size <= 0"
            ),
        )
    )

    expected_chunks = "[[3,1,4],[1,5,9],[2]]"
    turn_nine = re.sub(r"\s+", "", answers[8])
    chunking_satisfied = expected_chunks in turn_nine
    observations.append(
        TextQualityObservation(
            name="contextual_chunking_answer",
            satisfied=chunking_satisfied,
            detail=(
                "turn 9 produced [[3, 1, 4], [1, 5, 9], [2]]"
                if chunking_satisfied
                else "turn 9 did not produce [[3, 1, 4], [1, 5, 9], [2]]"
            ),
        )
    )

    final = answers[-1].lower()
    missing_summary_terms = tuple(
        required
        for required in ("chunked", "is_palindrome", "o(n)", "valueerror")
        if required not in final
    )
    observations.append(
        TextQualityObservation(
            name="final_summary_context",
            satisfied=not missing_summary_terms,
            detail=(
                "final summary retained both APIs, complexity, and failure modes"
                if not missing_summary_terms
                else f"final summary omitted {', '.join(missing_summary_terms)}"
            ),
        )
    )

    syntax_failures: list[str] = []
    for turn_number in (1, 3, 5, 7, 8):
        for block in _python_fences(answers[turn_number - 1]):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                syntax_failures.append(
                    f"turn {turn_number} contains invalid Python: {exc.msg}"
                )
    observations.append(
        TextQualityObservation(
            name="python_syntax",
            satisfied=not syntax_failures,
            detail=(
                "all fenced Python parsed successfully"
                if not syntax_failures
                else "; ".join(syntax_failures)
            ),
        )
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
    module_contract_failures: list[str] = []
    if module_source is None:
        module_contract_failures.append(
            "reviewed mini-module did not define both benchmark APIs"
        )
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
                module_contract_failures.append(
                    f"reviewed mini-module lacks required behavior {required!r}"
                )
    observations.append(
        TextQualityObservation(
            name="reviewed_module_contract",
            satisfied=not module_contract_failures,
            detail=(
                "reviewed mini-module retained both requested API contracts"
                if not module_contract_failures
                else "; ".join(module_contract_failures)
            ),
        )
    )

    observations.append(
        TextQualityObservation(
            name="distinct_answer_ratio",
            satisfied=distinct_answers == len(answers),
            detail=f"{distinct_answers}/{len(answers)} assistant answers were distinct",
        )
    )

    return TextBenchmarkAssessment(
        compatibility_failures=tuple(compatibility_failures),
        quality_observations=tuple(observations),
    )


def _python_fences(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL))
