from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from audex_mac.audio_contract import NVIDIA_TTS_CFG_SCALE

pytestmark = pytest.mark.fast


def _load_probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "probe_vllm_tts_decode.py"
    spec = importlib.util.spec_from_file_location("probe_vllm_tts_decode", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cfg_segment_probe_requires_real_cfg_runtime() -> None:
    probe = _load_probe_module()
    runtime = SimpleNamespace(cfg_config=SimpleNamespace(enabled=False, cfg_scale=0.0))

    with pytest.raises(RuntimeError, match="requires real Audex CFG wiring"):
        probe._require_cfg_runtime_for_segment_probe(runtime)


def test_cfg_segment_probe_accepts_nvidia_cfg_scale() -> None:
    probe = _load_probe_module()
    runtime = SimpleNamespace(
        cfg_config=SimpleNamespace(enabled=True, cfg_scale=NVIDIA_TTS_CFG_SCALE)
    )

    probe._require_cfg_runtime_for_segment_probe(runtime)
