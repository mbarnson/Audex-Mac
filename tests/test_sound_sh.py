from __future__ import annotations

from pathlib import Path


def test_sound_sh_is_an_isolated_cfg3_sound_lab_launcher() -> None:
    script = Path("sound.sh").read_text(encoding="utf-8")

    assert 'export AUDEX_VLLM_TTS_CFG="1"' in script
    assert 'exec "${ROOT_DIR}/start.sh" sound-lab "$@"' in script
    assert "source start.sh" not in script

    start_script = Path("start.sh").read_text(encoding="utf-8")
    assert "sound-lab)" in start_script
    assert "NEEDS_AUDIO_EVAL_DEPS=1" in start_script
