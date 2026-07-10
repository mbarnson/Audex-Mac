#!/usr/bin/env python3
"""Batch-transcribe and score one generated TTS quality manifest."""

from __future__ import annotations

import argparse
import json
import wave
from collections import defaultdict
from pathlib import Path

from audex_mac.tts_oracle import MlxAudioTranscriber

from evaluate_tts_wav import (
    consecutive_repetition_summary,
    required_term_summary,
    sequence_similarity,
    word_error_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--model",
        default="mlx-community/parakeet-tdt-0.6b-v3",
        help="MLX-Audio STT model name",
    )
    parser.add_argument("--max-wer", type=float, default=0.45)
    return parser.parse_args()


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def quality_sample_passed(
    *,
    word_errors: dict[str, object],
    repetition: dict[str, object],
    required_terms: dict[str, object],
    terminated_cleanly: bool,
    max_word_error_rate: float,
) -> bool:
    return (
        float(word_errors["word_error_rate"]) <= max_word_error_rate
        and not bool(repetition["excessive"])
        and not bool(required_terms["missing"])
        and terminated_cleanly
    )


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    transcriber = MlxAudioTranscriber(args.model)
    load_seconds = transcriber.load()
    evaluations: list[dict[str, object]] = []
    for sample in manifest["samples"]:
        wav_path = Path(sample["wav_path"])
        result = transcriber.transcribe_file(wav_path)
        expected = str(sample["text"])
        actual = str(result.get("text", ""))
        ratio = sequence_similarity(expected, actual)
        repetition = consecutive_repetition_summary(
            actual,
            ngram_size=4,
            max_allowed_repetitions=3,
        )
        word_errors = word_error_summary(expected, actual)
        required_terms = required_term_summary(
            actual,
            tuple(sample.get("required_terms", [])),
        )
        run_log_path = Path(sample["run_log_path"])
        run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
        terminated_cleanly = bool(run_log.get("reached_end_token")) and not bool(
            run_log.get("hit_max_tokens")
        )
        passed = quality_sample_passed(
            word_errors=word_errors,
            repetition=repetition,
            required_terms=required_terms,
            terminated_cleanly=terminated_cleanly,
            max_word_error_rate=args.max_wer,
        )
        evaluations.append(
            {
                "case_id": sample["case_id"],
                "category": sample["category"],
                "passed": passed,
                "ratio": ratio,
                "word_errors": word_errors,
                "required_terms": required_terms,
                "repetition": repetition,
                "terminated_cleanly": terminated_cleanly,
                "duration_seconds": round(wav_duration_seconds(wav_path), 3),
                "expected": expected,
                "actual": actual,
                "wav_path": str(wav_path),
                "run_log_path": str(run_log_path),
                "transcribe_seconds": round(float(result.get("elapsed", 0.0)), 3),
            }
        )

    categories: dict[str, list[dict[str, object]]] = defaultdict(list)
    for evaluation in evaluations:
        categories[str(evaluation["category"])].append(evaluation)

    def aggregate(items: list[dict[str, object]]) -> dict[str, object]:
        count = len(items)
        return {
            "sample_count": count,
            "pass_count": sum(bool(item["passed"]) for item in items),
            "mean_ratio": sum(float(item["ratio"]) for item in items) / count,
            "mean_word_error_rate": sum(
                float(item["word_errors"]["word_error_rate"]) for item in items
            )
            / count,
            "mean_required_term_recall": sum(
                float(item["required_terms"]["recall"]) for item in items
            )
            / count,
        }

    payload = {
        "schema_version": 1,
        "recipe_id": manifest["recipe_id"],
        "source_manifest": str(args.manifest),
        "model": args.model,
        "max_word_error_rate": args.max_wer,
        "load_seconds": round(load_seconds, 3),
        "passed": all(bool(item["passed"]) for item in evaluations),
        "aggregate": aggregate(evaluations),
        "categories": {
            category: aggregate(items) for category, items in sorted(categories.items())
        },
        "evaluations": evaluations,
    }
    output_path = args.json_out or args.manifest.with_suffix(".eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
