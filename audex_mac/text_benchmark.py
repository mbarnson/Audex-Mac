"""Text benchmark contract for the Audex text-only gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = ROOT_DIR / "benchmarks" / "text_conversation.json"
MIN_TURNS = 10
MIN_MAX_TOKENS = 4096


@dataclass(frozen=True, slots=True)
class TextBenchmark:
    name: str
    system: str
    generation: dict[str, Any]
    turns: list[dict[str, str]]
    pass_criteria: list[str]
    sampler_reference: str

    @property
    def max_tokens(self) -> int:
        return int(self.generation["max_tokens"])


def load_text_benchmark(path: Path = DEFAULT_BENCHMARK_PATH) -> TextBenchmark:
    data = json.loads(path.read_text(encoding="utf-8"))
    benchmark = TextBenchmark(
        name=data["name"],
        system=data["system"],
        generation=dict(data["generation"]),
        turns=list(data["turns"]),
        pass_criteria=list(data["pass_criteria"]),
        sampler_reference=data["sampler_reference"],
    )
    validate_text_benchmark(benchmark)
    return benchmark


def validate_text_benchmark(benchmark: TextBenchmark) -> None:
    if len(benchmark.turns) < MIN_TURNS:
        raise ValueError(
            f"text benchmark requires at least {MIN_TURNS} turns; "
            f"got {len(benchmark.turns)}"
        )
    if benchmark.max_tokens < MIN_MAX_TOKENS:
        raise ValueError(
            f"text benchmark max_tokens must be at least {MIN_MAX_TOKENS}; "
            f"got {benchmark.max_tokens}"
        )
    if benchmark.generation.get("thinking_enabled") is not False:
        raise ValueError("speech demo text benchmark must default to non-thinking mode")
    required_sampler_keys = {"temperature", "top_p", "seed", "max_tokens"}
    missing = required_sampler_keys - benchmark.generation.keys()
    if missing:
        raise ValueError(
            f"text benchmark generation is missing keys: {sorted(missing)}"
        )
