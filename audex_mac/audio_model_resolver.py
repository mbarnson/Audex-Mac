"""Shared Audex audio-model profile and cached-runtime resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .audio_runtime import preflight_audio_runtime
from .models import (
    AUDEX_2B_REPO,
    AUDEX_30B_NVFP4_REPO,
    AUDEX_30B_REPO,
    SUPPORTED_MODELS,
)


def audio_model_repo(model: str, profile: str) -> str:
    if model == "2b" and profile == "bf16":
        return AUDEX_2B_REPO
    if model == "30b" and profile == "nvfp4":
        return AUDEX_30B_NVFP4_REPO
    if model == "30b" and profile == "bf16":
        return AUDEX_30B_REPO
    raise ValueError(
        f"unsupported Audex audio model selection: model={model} profile={profile}"
    )


def resolve_cached_audio_model(model: str, profile: str) -> tuple[Path, str]:
    repo_id = audio_model_repo(model, profile)
    selected = next(item for item in SUPPORTED_MODELS if item.repo_id == repo_id)
    preflight = preflight_audio_runtime(selected)
    if preflight.ready and preflight.model_path is not None:
        return preflight.model_path, repo_id
    missing = ", ".join(preflight.missing_items) or "unknown missing model files"
    raise RuntimeError(
        f"Audex requires a complete cached speech checkpoint for {repo_id}; "
        f"missing: {missing}. Pass --model-path to override."
    )


def load_audio_vllm_runtime(model_path: Path | None, profile: str) -> Any:
    del profile
    if model_path is None:
        raise ValueError("model_path is required for the default vLLM runtime")
    from .vllm_runtime import AudexAsyncVllmRuntime

    return AudexAsyncVllmRuntime.from_model_path(model_path)
