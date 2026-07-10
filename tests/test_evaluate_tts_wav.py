from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast


def load_evaluator_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "evaluate_tts_wav.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_tts_wav", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_expected_text_from_speech_log_joins_segments_in_numeric_order(
    tmp_path: Path,
) -> None:
    evaluator = load_evaluator_module()
    speech_log = tmp_path / "speech-output.json"
    speech_log.write_text(
        json.dumps(
            {
                "tts_segment_texts": {
                    "10": "third",
                    "0": "first",
                    "2": "second",
                }
            }
        ),
        encoding="utf-8",
    )

    assert evaluator.expected_text_from_speech_log(speech_log) == ("first second third")


def test_resolve_expected_text_prefers_explicit_expected(tmp_path: Path) -> None:
    evaluator = load_evaluator_module()
    missing_log = tmp_path / "missing.json"

    assert (
        evaluator.resolve_expected_text(
            expected="explicit text",
            expected_from_speech_log=missing_log,
        )
        == "explicit text"
    )


def test_resolve_expected_text_requires_source() -> None:
    evaluator = load_evaluator_module()

    with pytest.raises(ValueError, match="provide --expected"):
        evaluator.resolve_expected_text(
            expected=None,
            expected_from_speech_log=None,
        )


def test_consecutive_repetition_summary_accepts_normal_reused_words() -> None:
    evaluator = load_evaluator_module()

    summary = evaluator.consecutive_repetition_summary(
        "It runs enter on entry and exit on exit, then it exits cleanly.",
        ngram_size=4,
        max_allowed_repetitions=3,
    )

    assert summary == {
        "excessive": False,
        "max_consecutive_repetitions": 1,
        "max_allowed_repetitions": 3,
        "ngram_size": 4,
        "ngram": "it runs enter on",
    }


def test_consecutive_repetition_summary_flags_pathological_phrase_loop() -> None:
    evaluator = load_evaluator_module()

    summary = evaluator.consecutive_repetition_summary(
        "with fewer lines of concurrent code faster "
        "with fewer lines of concurrent code faster "
        "with fewer lines of concurrent code faster "
        "with fewer lines of concurrent code faster",
        ngram_size=7,
        max_allowed_repetitions=3,
    )

    assert summary == {
        "excessive": True,
        "max_consecutive_repetitions": 4,
        "max_allowed_repetitions": 3,
        "ngram_size": 7,
        "ngram": "with fewer lines of concurrent code faster",
    }


def test_consecutive_repetition_summary_allows_threshold_count() -> None:
    evaluator = load_evaluator_module()

    summary = evaluator.consecutive_repetition_summary(
        "one two three one two three one two three",
        ngram_size=3,
        max_allowed_repetitions=3,
    )

    assert summary["excessive"] is False
    assert summary["max_consecutive_repetitions"] == 3


def test_word_error_summary_reports_insertions_deletions_and_substitutions() -> None:
    evaluator = load_evaluator_module()

    summary = evaluator.word_error_summary(
        "one two three four",
        "one too four extra",
    )

    assert summary == {
        "word_error_rate": 0.75,
        "reference_word_count": 4,
        "hypothesis_word_count": 4,
        "errors": 3,
        "substitutions": 1,
        "deletions": 1,
        "insertions": 1,
    }


def test_required_term_summary_checks_normalized_word_sequences() -> None:
    evaluator = load_evaluator_module()

    summary = evaluator.required_term_summary(
        "Saoirse Ronan met the team near sixteen kilohertz.",
        ("Saoirse Ronan", "16 kilohertz", "missing phrase"),
    )

    assert summary == {
        "recall": pytest.approx(1 / 3),
        "matched": ["Saoirse Ronan"],
        "missing": ["16 kilohertz", "missing phrase"],
        "total": 3,
    }
