from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from audex_mac.audio_evaluation_enhancement import (
    NvidiaEnhancementVae,
    enhancement_vae_artifact_identity,
    resolve_enhancement_vae_config,
)


class FakeTorch:
    class device:
        def __init__(self, name: str) -> None:
            self.name = name

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)


@pytest.mark.fast
def test_nvidia_enhancement_vae_loads_once_and_reseeds_each_output(
    tmp_path: Path,
) -> None:
    root = _enhancement_root(tmp_path)
    loads: list[dict[str, Any]] = []
    enhancements: list[dict[str, Any]] = []

    def load_model(**kwargs: Any) -> object:
        loads.append(kwargs)
        return object()

    def enhance_file(**kwargs: Any) -> None:
        enhancements.append(kwargs)
        kwargs["output_path"].write_bytes(b"enhanced")

    module = SimpleNamespace(load_model=load_model, enhance_file=enhance_file)
    torch = FakeTorch()
    enhancer = NvidiaEnhancementVae(
        resolve_enhancement_vae_config(root, device="mps", seed=0),
        module_loader=lambda _path: module,
        torch_module=torch,
    )
    source = tmp_path / "raw.wav"
    source.write_bytes(b"raw")

    enhancer(source, tmp_path / "one.wav", object())
    enhancer(source, tmp_path / "two.wav", object())

    assert len(loads) == 1
    assert loads[0]["checkpoint_path"].name.endswith(".safetensors")
    assert loads[0]["device"].name == "mps"
    assert torch.seeds == [0, 0]
    assert [call["deterministic"] for call in enhancements] == [False, False]


@pytest.mark.fast
def test_enhancement_config_requires_nvidia_checkpoint_and_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "enhancement_VAE"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="enhancement VAE"):
        resolve_enhancement_vae_config(root)


@pytest.mark.fast
def test_local_enhancement_identity_hashes_all_effective_artifacts(
    tmp_path: Path,
) -> None:
    root = _enhancement_root(tmp_path)

    first = enhancement_vae_artifact_identity(root)
    (root / "config.json").write_text('{"changed": true}', encoding="utf-8")
    second = enhancement_vae_artifact_identity(root)

    assert first.startswith("local-sha256-")
    assert first != second


def _enhancement_root(tmp_path: Path) -> Path:
    root = tmp_path / "enhancement_VAE"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "XCodec_RVQ4_mono_causal_fp32.safetensors").write_bytes(b"weights")
    (root / "enhancement_vae.py").write_text("# fixture", encoding="utf-8")
    return root
