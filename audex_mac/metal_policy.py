"""Audex-Mac Metal/MLX runtime policy."""

from __future__ import annotations

import os
from dataclasses import dataclass

REQUIRED_METAL_ENV = {
    "VLLM_METAL_USE_MLX": "1",
    "VLLM_MLX_DEVICE": "gpu",
    "VLLM_METAL_USE_PAGED_ATTENTION": "0",
    "VLLM_METAL_MEMORY_FRACTION": "auto",
}


@dataclass(frozen=True, slots=True)
class MetalRuntimePolicy:
    env: dict[str, str]
    mlx_metal_available: bool | None = None
    mlx_default_device: str | None = None

    @property
    def ready(self) -> bool:
        env_ready = all(
            self.env.get(name) == expected
            for name, expected in REQUIRED_METAL_ENV.items()
        )
        return env_ready and self.mlx_metal_available is not False


def enforce_metal_env() -> MetalRuntimePolicy:
    """Require Audex inference to use vLLM Metal's MLX GPU path."""

    resolved: dict[str, str] = {}
    for name, expected in REQUIRED_METAL_ENV.items():
        current = os.environ.get(name)
        if current is None:
            os.environ[name] = expected
            current = expected
        if current != expected:
            raise RuntimeError(
                f"Audex-Mac requires {name}={expected!r} for Metal/MLX inference; "
                f"current value is {current!r}."
            )
        resolved[name] = current
    return MetalRuntimePolicy(env=resolved)


def inspect_metal_runtime() -> MetalRuntimePolicy:
    """Return enforced env plus live MLX Metal availability when importable."""

    policy = enforce_metal_env()
    try:
        import mlx.core as mx

        mx.set_default_device(mx.Device(mx.DeviceType.gpu))
        return MetalRuntimePolicy(
            env=policy.env,
            mlx_metal_available=bool(mx.metal.is_available()),
            mlx_default_device=str(mx.default_device()),
        )
    except Exception:
        return policy
