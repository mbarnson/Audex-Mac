from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from audex_mac.audio_evaluation_generation import TtaOutputInspection
from audex_mac.audio_evaluation_runner import GenerationAttempt
from audex_mac.tta_quality import (
    configure_nvidia_tta_environment,
    create_blind_quant_listening_set,
    load_tta_quality_corpus,
    render_tta_quality_manifest,
)
from audex_mac.tta_quality_cli import _model_provenance


@pytest.mark.fast
def test_tta_quant_corpus_uses_literal_unique_caption_cases() -> None:
    corpus = load_tta_quality_corpus(Path("scripts/tta_quant_quality_corpus.json"))

    assert corpus.master_seed == 20260710
    assert len(corpus.cases) == 8
    assert len({case.caption.casefold() for case in corpus.cases}) == 8
    assert {case.case_id for case in corpus.cases} >= {
        "door-slam",
        "dog-barks",
        "passing-motorcycle",
        "solo-cello",
    }


@pytest.mark.fast
def test_tta_voice_corpus_covers_distinct_vocal_behaviors() -> None:
    corpus = load_tta_quality_corpus(Path("scripts/tta_quant_voice_corpus.json"))

    assert len(corpus.cases) == 10
    captions = " ".join(case.caption.casefold() for case in corpus.cases)
    for behavior in (
        "sighs",
        "whispers",
        "growls",
        "shouts",
        "screams",
        "sings",
        "moans",
        "swallows",
        "hisses through her teeth",
        "grunts",
    ):
        assert behavior in captions


@pytest.mark.fast
def test_nvidia_tta_environment_pins_reference_batch_and_context() -> None:
    env: dict[str, str] = {}

    configure_nvidia_tta_environment(env)

    assert env == {
        "AUDEX_VLLM_TTS_CFG": "1",
        "AUDEX_VLLM_ENABLE_CFG_WIRING": "1",
        "AUDEX_VLLM_CFG_MAX_MODEL_LEN": "8192",
        "AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS": "4",
    }


@pytest.mark.fast
def test_model_provenance_reads_snapshot_and_hashes_config(tmp_path: Path) -> None:
    revision = "a" * 40
    model_path = tmp_path / "snapshots" / revision / "checkpoint_folder_full"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    actual_revision, hashes = _model_provenance(model_path)

    assert actual_revision == revision
    assert set(hashes) == {"config.json"}


@pytest.mark.fast
def test_blind_quant_set_hides_profile_but_preserves_caption_groups(
    tmp_path: Path,
) -> None:
    manifests = tuple(
        _manifest(tmp_path, profile=profile) for profile in ("bf16", "nvfp4")
    )
    listening = create_blind_quant_listening_set(
        manifest_paths=manifests,
        output_dir=tmp_path / "blind",
        key_path=tmp_path / "private-key.json",
    )

    assert len(listening.sample_paths) == 4
    sheet = listening.listening_path.read_text(encoding="utf-8")
    assert "A door slams in a hallway." in sheet
    assert "bf16" not in sheet.casefold()
    assert "nvfp4" not in sheet.casefold()
    key = json.loads(listening.key_path.read_text(encoding="utf-8"))
    assert {item["profile"] for item in key["samples"]} == {"bf16", "nvfp4"}
    assert len(key["packaging_nonce"]) == 64
    assert {item["model_revision"] for item in key["recipes"]} == {"fixture-rev"}


@pytest.mark.fast
def test_quant_renderer_uses_identical_seeded_two_pair_waves(tmp_path: Path) -> None:
    corpus = load_tta_quality_corpus(Path("scripts/tta_quant_quality_corpus.json"))
    adapter = FakeGenerationAdapter(tmp_path)

    manifest_path = render_tta_quality_manifest(
        corpus=corpus,
        generation=adapter,
        profile="bf16",
        model_repo="nvidia/audex",
        model_revision="fixture-rev",
        model_file_hashes={"config.json": "fixture-sha256"},
        output_dir=tmp_path / "run",
        xcodec_identity="xcodec-fixture",
        enhancement_identity="vae-fixture",
    )

    assert [len(batch) for batch in adapter.batches] == [2, 2, 2, 2]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["samples"]) == 8
    assert all(sample["duration_seconds"] == 10.0 for sample in manifest["samples"])
    assert all(Path(sample["wav_path"]).is_file() for sample in manifest["samples"])


class FakeGenerationAdapter:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.batches: list[tuple[tuple[object, int], ...]] = []

    def generate_many(
        self, cases: tuple[tuple[object, int], ...]
    ) -> tuple[GenerationAttempt, ...]:
        self.batches.append(cases)
        attempts = []
        for case, _seed in cases:
            case_id = case.case_id
            raw = self.tmp_path / f"{case_id}-raw.wav"
            enhanced = self.tmp_path / f"{case_id}-enhanced.wav"
            raw.write_bytes(b"raw")
            enhanced.write_bytes(b"enhanced")
            attempts.append(
                GenerationAttempt(
                    raw_wav_path=raw,
                    enhanced_wav_path=enhanced,
                    structure=TtaOutputInspection(
                        codec_ids=(),
                        codec_token_count=2000,
                        frame_count=500,
                        duration_seconds=10.0,
                        reached_end_token=True,
                        first_phase_mismatch=None,
                        unexpected_token_ids=(),
                        failures=(),
                    ),
                    signal_metrics={"finite": True, "nonempty": True},
                    elapsed_seconds=1.0,
                    finish_reason="stop",
                )
            )
        return tuple(attempts)


def _manifest(tmp_path: Path, *, profile: str) -> Path:
    samples = []
    for case_id, caption in (
        ("door", "A door slams in a hallway."),
        ("rain", "Rain falls on a metal roof."),
    ):
        wav = tmp_path / f"{profile}-{case_id}.wav"
        with wave.open(str(wav), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(48_000)
            output.writeframes(b"\0\0" * 480)
        samples.append(
            {
                "case_id": case_id,
                "caption": caption,
                "seed": 123,
                "wav_path": str(wav),
            }
        )
    path = tmp_path / f"{profile}.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "profile": profile,
                "model_size": "30b",
                "model_repo": f"fixture/{profile}",
                "model_revision": "fixture-rev",
                "model_file_hashes": {"config.json": "fixture-sha256"},
                "engine_max_model_len": 8192,
                "recipe": "nvidia-tta-reference-v1",
                "master_seed": 20260710,
                "xcodec_identity": "xcodec-fixture",
                "enhancement_identity": "vae-fixture",
                "enhancement_seed": 0,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return path
