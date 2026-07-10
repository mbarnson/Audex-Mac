"""Text-only Audex runtime preflight for the first real model milestone."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from .checkpoints import HF_CACHE_ROOT, SnapshotCheck, verify_snapshot
from .metal_policy import MetalRuntimePolicy, inspect_metal_runtime
from .models import AudexModel
from .patches.runtime import AudexPatchReport, apply_audex_runtime_patches
from .text_benchmark import TextBenchmark, load_text_benchmark


@dataclass(frozen=True, slots=True)
class RuntimeDependencyCheck:
    module_name: str
    present: bool


@dataclass(frozen=True, slots=True)
class TextRuntimePreflight:
    model: AudexModel
    benchmark: TextBenchmark
    snapshot_check: SnapshotCheck
    dependency_checks: tuple[RuntimeDependencyCheck, ...]
    patch_report: AudexPatchReport | None = None
    metal_policy: MetalRuntimePolicy | None = None

    @property
    def ready(self) -> bool:
        ready = (
            self.snapshot_check.complete
            and all(check.present for check in self.dependency_checks)
            and (self.patch_report is None or self.patch_report.ready)
        )
        return ready and (self.metal_policy is None or self.metal_policy.ready)

    @property
    def model_path(self) -> Path | None:
        if self.snapshot_check.snapshot_path is None:
            return None
        return self.snapshot_check.snapshot_path / self.model.text_checkpoint_dirs[0]

    @property
    def missing_items(self) -> tuple[str, ...]:
        missing = list(self.snapshot_check.missing_summary)
        for check in self.dependency_checks:
            if check.present:
                continue
            if check.module_name.startswith("python>="):
                missing.append(check.module_name)
            else:
                missing.append(f"python module {check.module_name}")
        if (
            self.patch_report is not None
            and not self.patch_report.mlx_lm_nemotron_dense
        ):
            missing.append("Audex MLX-LM nemotron_dense patch")
        if (
            self.patch_report is not None
            and not self.patch_report.mlx_lm_nemotron_h_audex
        ):
            missing.append("Audex MLX-LM nemotron_h_audex patch")
        if (
            self.patch_report is not None
            and not self.patch_report.vllm_metal_platform_repair
        ):
            missing.append("Audex vLLM Metal current_platform repair patch")
        if self.patch_report is not None and not self.patch_report.vllm_nemotron_dense:
            missing.append("Audex vLLM NemotronDenseForCausalLM patch")
        if (
            self.patch_report is not None
            and not self.patch_report.vllm_metal_audex_adapter
        ):
            missing.append("Audex vLLM Metal multimodal adapter patch")
        if self.metal_policy is not None and not self.metal_policy.ready:
            missing.append("Metal/MLX runtime policy")
        return tuple(missing)


def preflight_text_runtime(
    model: AudexModel,
    benchmark: TextBenchmark | None = None,
    cache_root: Path = HF_CACHE_ROOT,
    apply_patches: bool = True,
) -> TextRuntimePreflight:
    """Check whether the selected Audex text-only benchmark path can run."""

    benchmark = benchmark or load_text_benchmark()
    snapshot_check = verify_snapshot(
        model,
        required_files=model.text_required_files,
        checkpoint_dirs=model.text_checkpoint_dirs,
        cache_root=cache_root,
    )
    dependency_checks = tuple(
        [RuntimeDependencyCheck("python>=3.12,<3.14", _python_version_supported())]
        + [
            RuntimeDependencyCheck(name, importlib.util.find_spec(name) is not None)
            for name in ("vllm", "vllm_metal")
        ]
    )
    metal_policy = inspect_metal_runtime() if apply_patches else None
    return TextRuntimePreflight(
        model=model,
        benchmark=benchmark,
        snapshot_check=snapshot_check,
        dependency_checks=dependency_checks,
        patch_report=apply_audex_runtime_patches() if apply_patches else None,
        metal_policy=metal_policy,
    )


def _python_version_supported() -> bool:
    return (3, 12) <= sys.version_info[:2] < (3, 14)
