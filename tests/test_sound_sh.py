from __future__ import annotations

from pathlib import Path


def test_sound_sh_is_an_isolated_cfg3_sound_lab_launcher() -> None:
    script = Path("sound.sh").read_text(encoding="utf-8")

    assert 'export AUDEX_VLLM_TTS_CFG="1"' in script
    assert 'export AUDEX_VLLM_ENABLE_CFG_WIRING="1"' in script
    assert 'export AUDEX_VLLM_CFG_MAX_MODEL_LEN="8192"' in script
    assert 'export AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS="4"' in script
    assert 'exec "${ROOT_DIR}/start.sh" sound-lab --profile bf16 "$@"' in script
    assert "source start.sh" not in script

    start_script = Path("start.sh").read_text(encoding="utf-8")
    assert "sound-lab|web)" in start_script
    assert "NEEDS_AUDIO_EVAL_DEPS=1" in start_script
