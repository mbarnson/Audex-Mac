"""Explicit AST label fixtures for local audio-evaluation controls."""

from __future__ import annotations

from collections.abc import Iterable

from .audio_evaluation import AudioEvaluationCase, EvaluationTrack
from .audio_evaluation_esc50 import ESC50_HARD_NEGATIVES

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

# Hand-audited against AST_REVISION's id2label. A few ESC-50 classes without an
# exact AudioSet class intentionally use the nearest explicit event labels.
ESC50_AST_EXPECTED_LABELS: dict[str, tuple[str, ...]] = {
    "dog": ("Dog",),
    "rooster": ("Chicken, rooster",),
    "pig": ("Pig",),
    "cow": ("Cattle, bovinae",),
    "frog": ("Frog",),
    "cat": ("Cat",),
    "hen": ("Chicken, rooster",),
    "insects": ("Insect",),
    "sheep": ("Sheep",),
    "crow": ("Crow",),
    "rain": ("Rain",),
    "sea_waves": ("Waves, surf",),
    "crackling_fire": ("Crackle",),
    "crickets": ("Cricket",),
    "chirping_birds": ("Bird vocalization, bird call, bird song",),
    "water_drops": ("Drip",),
    "wind": ("Wind",),
    "pouring_water": ("Pour",),
    "toilet_flush": ("Toilet flush",),
    "thunderstorm": ("Thunderstorm",),
    "crying_baby": ("Baby cry, infant cry",),
    "sneezing": ("Sneeze",),
    "clapping": ("Clapping",),
    "breathing": ("Breathing",),
    "coughing": ("Cough",),
    "footsteps": ("Walk, footsteps",),
    "laughing": ("Laughter",),
    "brushing_teeth": ("Toothbrush",),
    "snoring": ("Snoring",),
    "drinking_sipping": ("Liquid",),
    "door_wood_knock": ("Knock",),
    "mouse_click": ("Mouse", "Clicking"),
    "keyboard_typing": ("Typing",),
    "door_wood_creaks": ("Creak",),
    "can_opening": ("Creak", "Clicking"),
    "washing_machine": ("Mechanical fan", "Hum"),
    "vacuum_cleaner": ("Vacuum cleaner",),
    "clock_alarm": ("Alarm clock",),
    "clock_tick": ("Tick-tock",),
    "glass_breaking": ("Glass",),
    "helicopter": ("Helicopter",),
    "chainsaw": ("Chainsaw",),
    "siren": ("Siren",),
    "car_horn": ("Vehicle horn, car horn, honking",),
    "engine": ("Engine",),
    "train": ("Train",),
    "church_bells": ("Church bell",),
    "airplane": ("Aircraft",),
    "fireworks": ("Fireworks",),
    "hand_saw": ("Sawing",),
}

if ESC50_AST_EXPECTED_LABELS.keys() != ESC50_HARD_NEGATIVES.keys():
    raise RuntimeError("ESC-50 AST label map must cover the fixed 50 categories")

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
