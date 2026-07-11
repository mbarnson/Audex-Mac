"""Named baseline and published paper target profiles for audio evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BaselineTargetProfile:
    name: str
    summary_path: str
    summary_sha256: str
    targets: dict[str, float]


def baseline_target_profile(
    *,
    name: str,
    summary_path: Path,
) -> BaselineTargetProfile:
    """Derive the documented Standard regression gates from a blessed summary."""

    clean_name = name.strip()
    if not clean_name:
        raise ValueError("baseline name must not be empty")
    raw = summary_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("baseline summary must be a JSON object")
    if (
        float(payload.get("case_completeness", 0.0)) != 1.0
        or payload.get("verdict") == "PROTOCOL_FAIL"
        or payload.get("protocol_failures")
    ):
        raise ValueError("baseline must be a complete protocol-valid run")
    accuracy = _required_number(payload, "accuracy")
    generation = _required_mapping(payload, "generation")
    completed = _required_number(generation, "completed_cases")
    structurally_valid = _required_number(generation, "structurally_valid")
    semantic = _required_mapping(generation, "semantic_metrics")
    clap = _required_mapping(semantic, "clap")
    hard_foil_win_rate = _required_number(clap, "hard_foil_win_rate")
    return BaselineTargetProfile(
        name=clean_name,
        summary_path=str(summary_path),
        summary_sha256=hashlib.sha256(raw).hexdigest(),
        targets={
            "accuracy_min": max(0.0, accuracy - 0.02),
            "hard_foil_win_rate_min": max(0.0, hard_foil_win_rate - 0.05),
            "generation_structural_failures_max": completed - structurally_valid,
        },
    )


def paper_reproduction_targets(
    *,
    tier: str,
    model: str,
    profile: str,
) -> dict[str, float]:
    """Return opt-in-by-profile BF16 gates derived from published Audex scores."""

    if tier != "full" or profile != "bf16":
        return {}
    published = {
        "30b": {
            "sound": 0.815,
            "music": 0.775,
            "audiocaps": 66.9,
            "song_describer": 62.7,
        },
        "2b": {
            "sound": 0.751,
            "music": 0.723,
            "audiocaps": 79.3,
            "song_describer": 78.4,
        },
    }
    values = published.get(model)
    if values is None:
        raise ValueError(f"unsupported paper target model: {model}")
    return {
        "understanding_sound_accuracy_min": values["sound"] - 0.03,
        "understanding_music_accuracy_min": values["music"] - 0.03,
        "fd_openl3_audiocaps_max": values["audiocaps"] * 1.10,
        "fd_openl3_song_describer_max": values["song_describer"] * 1.10,
    }


def merge_capability_targets(*target_sets: dict[str, float]) -> dict[str, float]:
    """Merge gates without allowing one source to weaken another."""

    merged: dict[str, float] = {}
    for target_set in target_sets:
        for name, raw_value in target_set.items():
            value = float(raw_value)
            if name not in merged:
                merged[name] = value
            elif name.endswith("_min"):
                merged[name] = max(merged[name], value)
            elif name.endswith("_max"):
                merged[name] = min(merged[name], value)
            else:
                raise ValueError(f"capability target has unsupported suffix: {name}")
    return dict(sorted(merged.items()))


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"baseline summary is missing mapping metric: {key}")
    return value


def _required_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"baseline summary is missing numeric metric: {key}")
    return float(value)
