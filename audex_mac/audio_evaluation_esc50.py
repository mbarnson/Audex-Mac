"""Fixed within-domain ESC-50 hard negatives for evaluation and calibration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

ESC50_CATEGORY_GROUPS = (
    (
        "dog",
        "rooster",
        "pig",
        "cow",
        "frog",
        "cat",
        "hen",
        "insects",
        "sheep",
        "crow",
    ),
    (
        "rain",
        "sea_waves",
        "crackling_fire",
        "crickets",
        "chirping_birds",
        "water_drops",
        "wind",
        "pouring_water",
        "toilet_flush",
        "thunderstorm",
    ),
    (
        "crying_baby",
        "sneezing",
        "clapping",
        "breathing",
        "coughing",
        "footsteps",
        "laughing",
        "brushing_teeth",
        "snoring",
        "drinking_sipping",
    ),
    (
        "door_wood_knock",
        "mouse_click",
        "keyboard_typing",
        "door_wood_creaks",
        "can_opening",
        "washing_machine",
        "vacuum_cleaner",
        "clock_alarm",
        "clock_tick",
        "glass_breaking",
    ),
    (
        "helicopter",
        "chainsaw",
        "siren",
        "car_horn",
        "engine",
        "train",
        "church_bells",
        "airplane",
        "fireworks",
        "hand_saw",
    ),
)

ESC50_HARD_NEGATIVES = {
    category: tuple(group[(index + offset) % len(group)] for offset in (1, 2, 3))
    for group in ESC50_CATEGORY_GROUPS
    for index, category in enumerate(group)
}


def esc50_hard_foil_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Return one deterministic within-domain foil for every represented class."""

    categories = {
        str(row.get("category", "")).strip().lower()
        for row in rows
        if str(row.get("category", "")).strip()
    }
    unknown = sorted(categories - ESC50_HARD_NEGATIVES.keys())
    if unknown:
        raise ValueError(f"unknown ESC-50 categories: {unknown}")
    return {
        category: ESC50_HARD_NEGATIVES[category][0] for category in sorted(categories)
    }
