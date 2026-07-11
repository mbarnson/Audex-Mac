"""NVIDIA's released Audex enhancement-VAE postprocessor."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .audio_evaluation_xcodec import choose_torch_device

ENHANCEMENT_CHECKPOINT = "XCodec_RVQ4_mono_causal_fp32.safetensors"
ENHANCEMENT_CONFIG = "config.json"
ENHANCEMENT_SOURCE = "enhancement_vae.py"
_HF_REVISION_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class EnhancementVaeConfig:
    root: Path
    device: str = "auto"
    seed: int = 0
    deterministic: bool = False


class NvidiaEnhancementVae:
    """Lazily load and apply NVIDIA's exact XCodec1 enhancement model."""

    def __init__(
        self,
        config: EnhancementVaeConfig,
        *,
        module_loader: Callable[[Path], Any] | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.config = config
        self._module_loader = module_loader or _load_enhancement_module
        self._torch = torch_module
        self._module: Any | None = None
        self._model: Any | None = None

    def __call__(self, source: Path, destination: Path, case: Any) -> None:
        del case
        module, model, torch = self._load()
        torch.manual_seed(self.config.seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        module.enhance_file(
            model=model,
            input_path=Path(source),
            output_path=Path(destination),
            deterministic=self.config.deterministic,
        )

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is None:
            torch = self._torch or importlib.import_module("torch")
            device_name = choose_torch_device(torch, requested=self.config.device)
            module = self._module_loader(self.config.root / ENHANCEMENT_SOURCE)
            self._model = module.load_model(
                checkpoint_path=self.config.root / ENHANCEMENT_CHECKPOINT,
                config_path=self.config.root / ENHANCEMENT_CONFIG,
                device=torch.device(device_name),
            )
            self._module = module
            self._torch = torch
        assert self._module is not None
        assert self._torch is not None
        return self._module, self._model, self._torch


def resolve_enhancement_vae_config(
    root: str | Path,
    *,
    device: str = "auto",
    seed: int = 0,
    deterministic: bool = False,
) -> EnhancementVaeConfig:
    path = Path(root).expanduser().resolve()
    missing = [
        name
        for name in (ENHANCEMENT_CHECKPOINT, ENHANCEMENT_CONFIG, ENHANCEMENT_SOURCE)
        if not (path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"NVIDIA enhancement VAE is incomplete at {path}: missing {missing}"
        )
    return EnhancementVaeConfig(
        root=path,
        device=device,
        seed=seed,
        deterministic=deterministic,
    )


def enhancement_vae_artifact_identity(root: str | Path) -> str:
    path = Path(root).expanduser().resolve()
    if _HF_REVISION_RE.fullmatch(path.parent.name):
        return f"hf-{path.parent.name}"
    artifacts = tuple(
        path / name
        for name in (ENHANCEMENT_CHECKPOINT, ENHANCEMENT_CONFIG, ENHANCEMENT_SOURCE)
    )
    missing = [artifact for artifact in artifacts if not artifact.is_file()]
    if missing:
        raise FileNotFoundError(f"enhancement VAE artifacts are missing: {missing}")
    digest = hashlib.sha256()
    for artifact in artifacts:
        digest.update(artifact.name.encode("utf-8"))
        with artifact.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"local-sha256-{digest.hexdigest()}"


def _load_enhancement_module(source: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"audex_nvidia_enhancement_{abs(hash(source))}", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load NVIDIA enhancement VAE source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
