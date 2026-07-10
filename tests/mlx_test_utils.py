from __future__ import annotations

import importlib.util
from typing import Any

import pytest


def require_mlx_core() -> Any:
    """Return mlx.core or skip when MLX is absent or Metal is unavailable."""

    if importlib.util.find_spec("mlx") is None:
        pytest.skip("MLX is installed in the vLLM Metal runtime, not this test venv")
    import mlx.core as mx

    try:
        mx.eval(mx.zeros((1,), dtype=mx.float32))
    except RuntimeError as exc:
        pytest.skip(f"MLX runtime is unavailable in this test environment: {exc}")
    return mx
