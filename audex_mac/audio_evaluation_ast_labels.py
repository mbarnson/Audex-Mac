"""Explicit AST label fixtures for local audio-evaluation controls."""

from __future__ import annotations

from collections.abc import Iterable

from .audio_evaluation import AudioEvaluationCase, EvaluationTrack

# Keep this local subset small and explicit so structured-control fixtures
# cannot silently introduce arbitrary caption-derived labels.
PINNED_AST_LABEL_FIXTURE_VOCABULARY = frozenset(
    {
        "Acoustic guitar",
        "Air conditioning",
        "Alarm clock",
        "Animal",
        "Applause",
        "Bark",
        "Beep, bleep",
        "Bell",
        "Camera",
        "Cheering",
        "Children shouting",
        "Clicking",
        "Clapping",
        "Crowd",
        "Dog",
        "Door",
        "Guitar",
        "Glass",
        "Hum",
        "Jazz",
        "Liquid",
        "Motorcycle",
        "Music",
        "Narration, monologue",
        "Piano",
        "Radio",
        "Rain",
        "Silence",
        "Speech",
        "Telephone bell ringing",
        "Thunderstorm",
        "Train horn",
        "Vibration",
        "Walk, footsteps",
        "White noise",
        "Wind noise (microphone)",
    }
)

STRUCTURED_CONTROL_AST_LABELS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "quantity-01": (("Dog", "Bark"), ("Speech", "Music")),
    "quantity-02": (("Bell",), ("Speech",)),
    "quantity-03": (("Camera", "Clicking"), ("Speech",)),
    "quantity-04": (("Clapping",), ("Speech",)),
    "distance-01": (("Thunderstorm", "Rain"), ("Speech",)),
    "distance-02": (("Motorcycle",), ("Speech",)),
    "distance-03": (("Children shouting",), ("Music",)),
    "distance-04": (("Train horn",), ("Speech",)),
    "temporal-01": (("Glass", "Bark"), ("Speech",)),
    "temporal-02": (("Door", "Walk, footsteps"), ("Speech",)),
    "temporal-03": (("Telephone bell ringing", "Vibration"), ("Speech",)),
    "temporal-04": (("Rain", "Thunderstorm"), ("Speech",)),
    "quality-01": (("Radio", "Jazz"), ("Speech",)),
    "quality-02": (("Acoustic guitar", "Guitar"), ("Speech",)),
    "quality-03": (("Liquid",), ("Speech",)),
    "quality-04": (("Cheering", "Crowd"), ("Speech",)),
    "negative-01": (("Silence",), ("Speech", "Music")),
    "negative-02": (("White noise",), ("Speech", "Music")),
    "negative-03": (("Hum", "Air conditioning"), ("Speech", "Music")),
    "negative-04": (("Wind noise (microphone)",), ("Speech", "Music", "Animal")),
    "trap-01": (("Piano",), ("Speech", "Narration, monologue")),
    "trap-02": (("Rain", "Thunderstorm"), ("Speech",)),
    "trap-03": (("Alarm clock", "Beep, bleep"), ("Speech",)),
    "trap-04": (("Applause", "Crowd"), ("Speech",)),
}


def explicit_ast_label_maps(
    cases: Iterable[AudioEvaluationCase],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Return explicit AST labels for cases with local, hand-authored fixtures."""

    expected: dict[str, tuple[str, ...]] = {}
    forbidden: dict[str, tuple[str, ...]] = {}
    for case in cases:
        if case.track is not EvaluationTrack.GENERATION:
            continue
        labels = STRUCTURED_CONTROL_AST_LABELS.get(case.source_row_id)
        if labels is None:
            continue
        expected_labels, forbidden_labels = labels
        expected[case.case_id] = expected_labels
        forbidden[case.case_id] = forbidden_labels
    return expected, forbidden
