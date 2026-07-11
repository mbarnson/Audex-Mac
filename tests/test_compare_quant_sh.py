from pathlib import Path


def test_compare_quant_runs_bf16_and_nvfp4_before_blind_packaging() -> None:
    script = Path("compare-quant.sh").read_text(encoding="utf-8")

    assert "tta_quant_quality_corpus.json" in script
    assert "--profile bf16" in script
    assert "--profile nvfp4" in script
    assert script.index("--profile bf16") < script.index("--profile nvfp4")
    assert "tta-quant-quality package" in script
    assert 'RUN_LABEL="${2:-tta-quant}"' in script
    assert "private-key.json" in script


def test_compare_voices_selects_the_voice_corpus_and_separate_run_label() -> None:
    script = Path("compare-voices.sh").read_text(encoding="utf-8")

    assert "compare-quant.sh" in script
    assert "tta_quant_voice_corpus.json" in script
    assert "tta-voice-quant" in script
