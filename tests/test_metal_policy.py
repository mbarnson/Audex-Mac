from __future__ import annotations

import pytest

from audex_mac.metal_policy import REQUIRED_METAL_ENV, enforce_metal_env

pytestmark = pytest.mark.fast


def test_enforce_metal_env_sets_required_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in REQUIRED_METAL_ENV:
        monkeypatch.delenv(name, raising=False)

    policy = enforce_metal_env()

    assert policy.env == REQUIRED_METAL_ENV


def test_enforce_metal_env_rejects_cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_METAL_USE_MLX", "1")
    monkeypatch.setenv("VLLM_MLX_DEVICE", "cpu")
    monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "0")
    monkeypatch.setenv("VLLM_METAL_MEMORY_FRACTION", "auto")

    with pytest.raises(RuntimeError, match="VLLM_MLX_DEVICE"):
        enforce_metal_env()


def test_enforce_metal_env_requires_auto_memory_for_non_paged_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_METAL_USE_MLX", "1")
    monkeypatch.setenv("VLLM_MLX_DEVICE", "gpu")
    monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "0")
    monkeypatch.setenv("VLLM_METAL_MEMORY_FRACTION", "0.85")

    with pytest.raises(RuntimeError, match="VLLM_METAL_MEMORY_FRACTION"):
        enforce_metal_env()
