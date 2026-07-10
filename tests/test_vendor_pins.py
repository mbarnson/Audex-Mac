from __future__ import annotations

import pytest

from audex_mac.vendor_pins import fetch_vllm_metal_upstream_head, load_vllm_metal_pin

pytestmark = pytest.mark.fast


def test_loads_vllm_metal_pin() -> None:
    pin = load_vllm_metal_pin()

    assert pin.repo == "https://github.com/vllm-project/vllm-metal"
    assert pin.pinned_commit == "cd72e7d6d5c3eec452afe2693c3a45a0564d7650"
    assert pin.pinned_commit in pin.install_url


def test_fetch_upstream_head_from_fetcher() -> None:
    head = fetch_vllm_metal_upstream_head(
        fetch_json=lambda url: {
            "sha": "1111111111111111111111111111111111111111",
            "url": url,
        }
    )

    assert head == "1111111111111111111111111111111111111111"
