from __future__ import annotations

from pathlib import Path


def test_sound_sh_delegates_recipe_configuration_to_sound_lab_cli() -> None:
    script = Path("sound.sh").read_text(encoding="utf-8")

    assert "AUDEX_VLLM_TTS_CFG" not in script
    assert "AUDEX_VLLM_ENABLE_CFG_WIRING" not in script
    assert "AUDEX_VLLM_CFG_MAX_MODEL_LEN" not in script
    assert "AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS" not in script
    assert 'exec "${ROOT_DIR}/start.sh" sound-lab --profile bf16 "$@"' in script
    assert "source start.sh" not in script

    start_script = Path("start.sh").read_text(encoding="utf-8")
    assert "sound-lab|web)" in start_script
    assert "NEEDS_AUDIO_EVAL_DEPS=1" in start_script
