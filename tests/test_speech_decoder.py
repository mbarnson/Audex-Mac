from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.speech_decoder import (
    AudexSpeechDecoderConfig,
    AudexSpeechDecoderSession,
    SpeechDecoderSmokeResult,
    configured_speech_decoder_device,
    expected_speech_decoder_weight_keys,
    load_speech_decoder_config,
)

pytestmark = pytest.mark.fast


def test_speech_decoder_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUDEX_SPEECH_DECODER_DEVICE", raising=False)
    mx = type("FakeMx", (), {"cpu": "cpu", "gpu": "gpu"})()

    assert configured_speech_decoder_device(mx) == "cpu"


def test_speech_decoder_device_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDEX_SPEECH_DECODER_DEVICE", "gpu")
    mx = type("FakeMx", (), {"cpu": "cpu", "gpu": "gpu"})()

    assert configured_speech_decoder_device(mx) == "gpu"


def test_load_speech_decoder_config_reads_nvidia_shape(tmp_path: Path) -> None:
    write_decoder_config(tmp_path)

    config = load_speech_decoder_config(tmp_path)

    assert config.hidden_dim == 2048
    assert config.depth == 12
    assert config.heads == 32
    assert config.head_dim == 64
    assert config.hop_length == 320
    assert config.sample_rate == 16_000
    assert config.codebook_size == 65_536


def test_expected_speech_decoder_weight_keys_include_embedder_bias() -> None:
    keys = expected_speech_decoder_weight_keys(decoder_config(depth=2))

    assert "audex_speech_token_embedder.project_out.bias" in keys
    assert "audex_speech_token_embedder.project_out.weight" in keys
    assert "module.lookahead_conv.weight" in keys
    assert "module.lookahead_proj.weight" in keys
    assert "module.backbone.transformers.1.mlp.fc2.weight" in keys
    assert len(keys) == 8 + (2 * 6)


def test_expected_speech_decoder_weight_keys_omit_lookahead_when_disabled() -> None:
    keys = expected_speech_decoder_weight_keys(
        decoder_config(depth=1, lookahead_steps=0),
    )

    assert "module.lookahead_conv.weight" not in keys
    assert "module.lookahead_proj.weight" not in keys


def test_speech_decoder_smoke_result_requires_finite_16khz_audio() -> None:
    result = SpeechDecoderSmokeResult(
        backend="mlx",
        device="Device(gpu, 0)",
        frame_count=8,
        input_shape=(8, 1),
        vq_embedding_shape=(1, 8, 2048),
        waveform_shape=(2560,),
        waveform_dtype="mlx.core.float32",
        sample_rate=16_000,
        hop_length=320,
        lookahead_steps=4,
        finite=True,
        peak_abs=0.5,
    )

    assert result.ready is True


def test_speech_decoder_smoke_result_rejects_wrong_sample_count() -> None:
    result = SpeechDecoderSmokeResult(
        backend="mlx",
        device="Device(gpu, 0)",
        frame_count=8,
        input_shape=(8, 1),
        vq_embedding_shape=(1, 8, 2048),
        waveform_shape=(2559,),
        waveform_dtype="mlx.core.float32",
        sample_rate=16_000,
        hop_length=320,
        lookahead_steps=4,
        finite=True,
        peak_abs=0.5,
    )

    assert result.ready is False


def test_speech_decoder_session_buffers_future_lookahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeEmbedding:
        dtype = "fake"

        def __init__(self, frames) -> None:
            self.frames = tuple(tuple(frame) for frame in frames)
            self.shape = (1, len(self.frames), 4)

    class FakeWaveform:
        pass

    def fake_embed(token_frames, weights, config):
        return FakeEmbedding(token_frames)

    def fake_decode(vq_emb, weights, config, cache, *, lookahead_vq_emb=None):
        calls.append(
            {
                "frames": vq_emb.frames,
                "lookahead": (
                    None if lookahead_vq_emb is None else lookahead_vq_emb.frames
                ),
                "cache_position": cache.position,
            }
        )
        cache.advance(len(vq_emb.frames))
        return FakeWaveform()

    monkeypatch.setattr(
        "audex_mac.speech_decoder.embed_speech_token_frames_mlx", fake_embed
    )
    monkeypatch.setattr(
        "audex_mac.speech_decoder.decode_speech_embeddings_cached_mlx",
        fake_decode,
    )

    session = AudexSpeechDecoderSession(
        weights={},
        config=decoder_config(lookahead_steps=2),
        chunk_frames=1,
    )

    assert session.push([(1,)]) == []
    assert session.push([(2,)]) == []
    chunks = session.push([(3,)])

    assert len(chunks) == 1
    assert calls == [
        {
            "frames": ((1,),),
            "lookahead": ((2,), (3,)),
            "cache_position": 0,
        }
    ]


def write_decoder_config(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "hidden_dim": 2048,
                "depth": 12,
                "heads": 32,
                "pos_meb_dim": 64,
                "hop_length": 320,
                "vq_dim": 2048,
                "lookahead_steps": 4,
                "sample_rate": 16000,
                "codebook_levels": [4, 4, 4, 4, 4, 4, 4, 4],
                "codebook_size": 65536,
                "token_embed_dim": 8,
            }
        ),
        encoding="utf-8",
    )


def decoder_config(
    *,
    depth: int = 12,
    lookahead_steps: int = 4,
) -> AudexSpeechDecoderConfig:
    return AudexSpeechDecoderConfig(
        hidden_dim=2048,
        depth=depth,
        heads=32,
        pos_meb_dim=64,
        hop_length=320,
        vq_dim=2048,
        lookahead_steps=lookahead_steps,
        sample_rate=16_000,
        codebook_levels=(4, 4, 4, 4, 4, 4, 4, 4),
        codebook_size=65_536,
        token_embed_dim=8,
    )
