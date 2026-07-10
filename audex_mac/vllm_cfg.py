"""Audex vLLM CFG engine configuration."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_contract import NVIDIA_TTS_CFG_SCALE
from .conversations import DEFAULT_DEMO_CONTEXT_TOKENS
from .patches.vllm_metal_cfg import AudexMetalCFGTokenSyncInstaller
from .vllm_sts_requests import (
    DEFAULT_ASR_MAX_TOKENS,
    DEFAULT_TEXT_MAX_TOKENS,
    DEFAULT_TTS_MAX_TOKENS,
)

DEFAULT_CFG_NUM_SEQS = 2
MIN_CFG_BATCHED_TOKENS = 8192
MIN_CFG_NUM_SEQS = 16
DIAGNOSTIC_CFG_MAX_NUM_SEQS_ENV = "AUDEX_VLLM_CFG_MAX_NUM_SEQS"
DIAGNOSTIC_CFG_MAX_BATCHED_TOKENS_ENV = "AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS"
DIAGNOSTIC_CFG_MAX_MODEL_LEN_ENV = "AUDEX_VLLM_CFG_MAX_MODEL_LEN"
DIAGNOSTIC_CFG_SCHEDULER_RESERVE_FULL_ISL_ENV = (
    "AUDEX_VLLM_CFG_SCHEDULER_RESERVE_FULL_ISL"
)
AUDEX_VLLM_AUDIOGEN_SCRIPTS = "inference_scripts_vllm/audiogen_scripts"


@dataclass(frozen=True, slots=True)
class AudexVllmCfgConfig:
    enabled: bool
    cfg_scale: float
    script_dir: Path | None
    logits_processors: tuple[str, ...]
    max_model_len: int | None
    max_num_batched_tokens: int | None
    max_num_seqs: int | None
    scheduler_reserve_full_isl: bool | None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.enabled and self.error is None


def require_audex_vllm_cfg_ready(config: AudexVllmCfgConfig) -> None:
    """Fail loudly if required Audex CFG wiring is unavailable."""

    if config.enabled and config.error is not None:
        raise RuntimeError(f"Audex vLLM CFG is not ready: {config.error}")


def configure_audex_vllm_cfg(
    engine_kwargs: dict[str, Any],
    model_path: Path,
    *,
    cfg_scale: float = NVIDIA_TTS_CFG_SCALE,
    asr_max_tokens: int = DEFAULT_ASR_MAX_TOKENS,
    text_max_tokens: int = DEFAULT_TEXT_MAX_TOKENS,
    tts_max_tokens: int = DEFAULT_TTS_MAX_TOKENS,
) -> AudexVllmCfgConfig:
    """Apply NVIDIA's vLLM CFG engine wiring for Audex TTS."""

    if cfg_scale <= 1.0:
        _append_logits_processor_once(engine_kwargs, AudexMetalCFGTokenSyncInstaller)
        return AudexVllmCfgConfig(
            enabled=False,
            cfg_scale=cfg_scale,
            script_dir=None,
            logits_processors=(),
            max_model_len=None,
            max_num_batched_tokens=None,
            max_num_seqs=None,
            scheduler_reserve_full_isl=None,
        )

    script_dir = find_audex_audiogen_scripts(model_path)
    if script_dir is None:
        return AudexVllmCfgConfig(
            enabled=True,
            cfg_scale=cfg_scale,
            script_dir=None,
            logits_processors=(),
            max_model_len=None,
            max_num_batched_tokens=None,
            max_num_seqs=None,
            scheduler_reserve_full_isl=None,
            error=f"missing {AUDEX_VLLM_AUDIOGEN_SCRIPTS} near {model_path}",
        )

    _prepend_import_path(script_dir)

    from vllm_cfg_patch import apply_cfg_patches

    apply_cfg_patches()
    from cfg_logits_processor import CFGLogitsProcessor

    configured_max_model_len = engine_kwargs.get(
        "max_model_len", DEFAULT_DEMO_CONTEXT_TOKENS
    )
    if not isinstance(configured_max_model_len, int):
        configured_max_model_len = DEFAULT_DEMO_CONTEXT_TOKENS
    max_model_len = max(
        configured_max_model_len,
        4096,
        asr_max_tokens + 1024,
        text_max_tokens + 1024,
        tts_max_tokens + 1024,
    )
    diagnostic_max_model_len = _diagnostic_positive_int(
        DIAGNOSTIC_CFG_MAX_MODEL_LEN_ENV
    )
    if diagnostic_max_model_len is not None:
        max_model_len = diagnostic_max_model_len
    max_num_batched_tokens = max(MIN_CFG_BATCHED_TOKENS, max_model_len)
    diagnostic_max_batched_tokens = _diagnostic_positive_int(
        DIAGNOSTIC_CFG_MAX_BATCHED_TOKENS_ENV
    )
    if diagnostic_max_batched_tokens is not None:
        max_num_batched_tokens = max(max_model_len, diagnostic_max_batched_tokens)
    max_num_seqs = max(MIN_CFG_NUM_SEQS, DEFAULT_CFG_NUM_SEQS)
    diagnostic_max_num_seqs = _diagnostic_positive_int(DIAGNOSTIC_CFG_MAX_NUM_SEQS_ENV)
    if diagnostic_max_num_seqs is not None:
        max_num_seqs = diagnostic_max_num_seqs
    scheduler_reserve_full_isl = _diagnostic_bool(
        DIAGNOSTIC_CFG_SCHEDULER_RESERVE_FULL_ISL_ENV
    )
    _append_logits_processor_once(engine_kwargs, CFGLogitsProcessor)
    _append_logits_processor_once(engine_kwargs, AudexMetalCFGTokenSyncInstaller)

    engine_kwargs.update(
        enable_prefix_caching=False,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    if scheduler_reserve_full_isl is not None:
        engine_kwargs["scheduler_reserve_full_isl"] = scheduler_reserve_full_isl
    return AudexVllmCfgConfig(
        enabled=True,
        cfg_scale=cfg_scale,
        script_dir=script_dir,
        logits_processors=tuple(
            _logits_processor_name(cls)
            for cls in tuple(engine_kwargs.get("logits_processors") or ())
        ),
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        scheduler_reserve_full_isl=scheduler_reserve_full_isl,
    )


def find_audex_audiogen_scripts(model_path: Path) -> Path | None:
    """Find NVIDIA's bundled Audex vLLM audiogen scripts for a checkpoint path."""

    model_path = model_path.resolve()
    candidates = [model_path, *model_path.parents]
    for candidate in candidates:
        script_dir = candidate / AUDEX_VLLM_AUDIOGEN_SCRIPTS
        if (script_dir / "cfg_logits_processor.py").is_file() and (
            script_dir / "vllm_cfg_patch.py"
        ).is_file():
            return script_dir
    return None


def _prepend_import_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    existing = os.environ.get("PYTHONPATH")
    parts = [] if not existing else existing.split(os.pathsep)
    if path_str not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([path_str, *parts])


def _append_logits_processor_once(
    engine_kwargs: dict[str, Any],
    processor: Any,
) -> None:
    processors = list(engine_kwargs.get("logits_processors") or ())
    processor_name = _logits_processor_name(processor)
    if all(
        _logits_processor_name(existing) != processor_name for existing in processors
    ):
        processors.append(processor)
    engine_kwargs["logits_processors"] = processors


def _diagnostic_positive_int(env_name: str) -> int | None:
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _diagnostic_bool(env_name: str) -> bool | None:
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{env_name} must be a boolean like 1/0, true/false, or yes/no; "
        f"got {value!r}"
    )


def _logits_processor_name(processor: Any) -> str:
    return f"{processor.__module__}.{processor.__name__}"
