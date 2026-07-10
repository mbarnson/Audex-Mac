"""Pinned external dependency metadata."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PINS_PATH = ROOT_DIR / "vendor_pins.json"
VLLM_METAL_HEAD_API = (
    "https://api.github.com/repos/vllm-project/vllm-metal/commits/main"
)


@dataclass(frozen=True, slots=True)
class VllmMetalPin:
    repo: str
    pinned_commit: str
    install_url: str
    pin_note: str


def load_vllm_metal_pin(path: Path = DEFAULT_PINS_PATH) -> VllmMetalPin:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data["vllm_metal"]
    return VllmMetalPin(
        repo=values["repo"],
        pinned_commit=values["pinned_commit"],
        install_url=values["install_url"],
        pin_note=values["pin_note"],
    )


def fetch_vllm_metal_upstream_head(
    fetch_json: Callable[[str], dict] | None = None,
) -> str:
    """Return the current upstream vLLM Metal main commit."""

    fetch_json = fetch_json or _fetch_json
    data = fetch_json(VLLM_METAL_HEAD_API)
    sha = data.get("sha")
    if not isinstance(sha, str) or len(sha) < 12:
        raise RuntimeError("GitHub response did not contain a valid commit sha.")
    return sha


def _fetch_json(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))
