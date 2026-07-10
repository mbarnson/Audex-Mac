#!/usr/bin/env python3
"""Evaluate an Audex TTS WAV by transcribing it with a local STT oracle."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

from audex_mac.tts_oracle import MlxAudioTranscriber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a WAV with MLX-Audio and compare it to expected text."
    )
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("--expected")
    parser.add_argument(
        "--expected-from-speech-log",
        type=Path,
        help="speech-output JSON whose tts_segment_texts should be used as expected text",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="optional path where the evaluation payload should be written",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/parakeet-tdt-0.6b-v3",
        help="MLX-Audio STT model name",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.55,
        help="minimum SequenceMatcher ratio for a passing evaluation",
    )
    parser.add_argument(
        "--repetition-ngram-size",
        type=int,
        default=4,
        help="word n-gram size used for consecutive repetition detection",
    )
    parser.add_argument(
        "--max-consecutive-repetitions",
        type=int,
        default=3,
        help="fail if any n-gram repeats consecutively more than this count",
    )
    parser.add_argument(
        "--required-term",
        action="append",
        default=[],
        help="word or phrase that must appear in the oracle transcription; repeatable",
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def sequence_similarity(expected: str, actual: str) -> float:
    return difflib.SequenceMatcher(
        None,
        normalize(expected),
        normalize(actual),
    ).ratio()


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", text.lower(), flags=re.UNICODE)


def word_error_summary(expected: str, actual: str) -> dict[str, int | float]:
    reference = normalized_words(expected)
    hypothesis = normalized_words(actual)
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    table: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)
    ]
    for row in range(1, rows):
        table[row][0] = (row, 0, row, 0)
    for column in range(1, columns):
        table[0][column] = (column, 0, 0, column)
    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                table[row][column] = table[row - 1][column - 1]
                continue
            sub = table[row - 1][column - 1]
            delete = table[row - 1][column]
            insert = table[row][column - 1]
            table[row][column] = min(
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            )
    errors, substitutions, deletions, insertions = table[-1][-1]
    denominator = max(1, len(reference))
    return {
        "word_error_rate": errors / denominator,
        "reference_word_count": len(reference),
        "hypothesis_word_count": len(hypothesis),
        "errors": errors,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def required_term_summary(
    actual: str,
    required_terms: tuple[str, ...],
) -> dict[str, object]:
    actual_words = normalized_words(actual)
    matched: list[str] = []
    missing: list[str] = []
    for term in required_terms:
        term_words = normalized_words(term)
        found = bool(term_words) and any(
            actual_words[index : index + len(term_words)] == term_words
            for index in range(len(actual_words) - len(term_words) + 1)
        )
        (matched if found else missing).append(term)
    total = len(required_terms)
    return {
        "recall": len(matched) / total if total else 1.0,
        "matched": matched,
        "missing": missing,
        "total": total,
    }


def consecutive_repetition_summary(
    text: str,
    *,
    ngram_size: int,
    max_allowed_repetitions: int,
) -> dict[str, object]:
    words = normalized_words(text)
    ngram_size = max(1, int(ngram_size))
    max_allowed_repetitions = max(1, int(max_allowed_repetitions))
    best_count = 0
    best_ngram: tuple[str, ...] = ()
    index = 0
    while index + ngram_size <= len(words):
        ngram = tuple(words[index : index + ngram_size])
        count = 1
        next_index = index + ngram_size
        while (
            next_index + ngram_size <= len(words)
            and tuple(words[next_index : next_index + ngram_size]) == ngram
        ):
            count += 1
            next_index += ngram_size
        if count > best_count:
            best_count = count
            best_ngram = ngram
        index += max(1, ngram_size * count) if count > 1 else 1
    return {
        "excessive": best_count > max_allowed_repetitions,
        "max_consecutive_repetitions": best_count,
        "max_allowed_repetitions": max_allowed_repetitions,
        "ngram_size": ngram_size,
        "ngram": " ".join(best_ngram),
    }


def expected_text_from_speech_log(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    segment_texts = payload.get("tts_segment_texts")
    if not isinstance(segment_texts, dict) or not segment_texts:
        raise ValueError(f"speech log does not contain tts_segment_texts: {path}")

    def sort_key(item: tuple[object, object]) -> tuple[int, str]:
        key = str(item[0])
        try:
            return int(key), key
        except ValueError:
            return 0, key

    return " ".join(
        str(text).strip()
        for _index, text in sorted(segment_texts.items(), key=sort_key)
        if str(text).strip()
    )


def resolve_expected_text(
    *,
    expected: str | None,
    expected_from_speech_log: Path | None,
) -> str:
    if expected:
        return expected
    if expected_from_speech_log is not None:
        return expected_text_from_speech_log(expected_from_speech_log)
    raise ValueError("provide --expected or --expected-from-speech-log")


def main() -> int:
    args = parse_args()
    if not args.wav_path.is_file():
        print(f"WAV does not exist: {args.wav_path}", file=sys.stderr)
        return 2
    try:
        expected = resolve_expected_text(
            expected=args.expected,
            expected_from_speech_log=args.expected_from_speech_log,
        )
    except Exception as exc:
        print(f"Could not resolve expected text: {exc}", file=sys.stderr)
        return 2

    try:
        transcriber = MlxAudioTranscriber(args.model)
        load_seconds = transcriber.load()
    except Exception as exc:
        print(
            "Could not load the MLX-Audio STT oracle; "
            f"install the oracle extra with `pip install -e '.[oracle]'`: {exc}",
            file=sys.stderr,
        )
        return 2

    result = transcriber.transcribe_file(args.wav_path)
    actual = result.get("text", "")
    ratio = sequence_similarity(expected, actual)
    repetition = consecutive_repetition_summary(
        actual,
        ngram_size=args.repetition_ngram_size,
        max_allowed_repetitions=args.max_consecutive_repetitions,
    )
    word_errors = word_error_summary(expected, actual)
    required_terms = required_term_summary(actual, tuple(args.required_term))
    passed = (
        ratio >= args.min_ratio
        and not repetition["excessive"]
        and not required_terms["missing"]
    )
    payload = {
        "passed": passed,
        "ratio": ratio,
        "min_ratio": args.min_ratio,
        "repetition": repetition,
        "word_errors": word_errors,
        "required_terms": required_terms,
        "expected": expected,
        "actual": actual,
        "wav_path": str(args.wav_path),
        "model": args.model,
        "load_seconds": round(load_seconds, 3),
        "transcribe_seconds": round(float(result.get("elapsed", 0.0)), 3),
        "segments": result.get("segments", []),
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
