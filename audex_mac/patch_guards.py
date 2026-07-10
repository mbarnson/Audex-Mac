"""Patch guard policy for pinned vLLM Metal monkey patches."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from importlib import import_module
from inspect import signature
from pathlib import Path
from types import ModuleType
from typing import Protocol, TextIO

PATCH_LEDGER_PATH = "docs/engineering/patches.md"


class ModuleProvider(Protocol):
    def import_module(self, module_name: str) -> ModuleType: ...


class ImportlibModuleProvider:
    def import_module(self, module_name: str) -> ModuleType:
        return import_module(module_name)


@dataclass(frozen=True, slots=True)
class PatchTarget:
    module: str
    symbol_path: tuple[str, ...]
    required_parameters: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        return ".".join((self.module, *self.symbol_path))


@dataclass(frozen=True, slots=True)
class VllmMetalState:
    installed_commit: str
    pinned_commit: str
    upstream_head: str | None


@dataclass(frozen=True, slots=True)
class PatchGuardResult:
    startup_allowed: bool
    warnings: tuple[str, ...]
    missing_symbol: str | None = None
    update_prompt: str | None = None


class PatchGuardError(RuntimeError):
    """Startup refusal raised when the installed patch target is unsafe."""

    def __init__(self, result: PatchGuardResult) -> None:
        self.result = result
        detail = result.missing_symbol or "unknown incompatibility"
        super().__init__(
            f"vLLM Metal patch guard failed: {detail}. "
            f"See {PATCH_LEDGER_PATH} for reapplication guidance."
        )


REQUIRED_TARGETS: tuple[PatchTarget, ...] = (
    PatchTarget("vllm_metal.v1.model_adapter", ("DefaultModelAdapter",)),
    PatchTarget(
        "vllm_metal.v1.model_adapter",
        ("DefaultModelAdapter", "build_multimodal_adapter"),
        required_parameters=("self", "model", "hf_config"),
    ),
    PatchTarget(
        "vllm_metal.v1.model_adapter",
        ("DefaultModelAdapter", "should_force_text_backbone"),
        required_parameters=("self", "hf_config"),
    ),
    PatchTarget(
        "vllm_metal.v1.model_adapter",
        ("DefaultModelAdapter", "normalize_model_config"),
        required_parameters=("self", "model_config"),
    ),
    PatchTarget(
        "vllm_metal.v1.model_lifecycle", ("ModelLifecycle", "_load_generation_model")
    ),
    PatchTarget("vllm_metal.v1.model_runner", ("MetalModelRunner", "load_model")),
)


def run_patch_guards(
    state: VllmMetalState,
    provider: ModuleProvider | None = None,
    targets: tuple[PatchTarget, ...] = REQUIRED_TARGETS,
) -> PatchGuardResult:
    """Validate that the pinned vLLM Metal install is safe to patch."""

    if state.installed_commit != state.pinned_commit:
        return PatchGuardResult(
            startup_allowed=False,
            warnings=(),
            missing_symbol="vLLM Metal installed commit does not match the pin",
        )

    provider = provider or ImportlibModuleProvider()
    for target in targets:
        error = _validate_target(target, provider)
        if error is not None:
            return PatchGuardResult(
                startup_allowed=False,
                warnings=(),
                missing_symbol=error,
            )

    warnings: list[str] = []
    update_prompt = None
    if state.upstream_head and state.upstream_head != state.pinned_commit:
        warnings.append(
            "Pinned vLLM Metal is behind upstream; startup will continue using "
            "the pinned commit."
        )
        update_prompt = build_update_prompt(state.pinned_commit, state.upstream_head)

    return PatchGuardResult(
        startup_allowed=True,
        warnings=tuple(warnings),
        update_prompt=update_prompt,
    )


def enforce_patch_guards(
    state: VllmMetalState,
    *,
    provider: ModuleProvider | None = None,
    targets: tuple[PatchTarget, ...] = REQUIRED_TARGETS,
    update_prompt_path: Path | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> PatchGuardResult:
    """Enforce patch compatibility and persist any upstream-update prompt."""

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    result = run_patch_guards(state, provider=provider, targets=targets)
    if not result.startup_allowed:
        error = PatchGuardError(result)
        print(str(error), file=stderr)
        raise error
    for warning in result.warnings:
        print(f"Audex-Mac patch guard warning: {warning}", file=stdout)
    if result.update_prompt is not None and update_prompt_path is not None:
        update_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        update_prompt_path.write_text(result.update_prompt, encoding="utf-8")
        print(
            f"Audex-Mac patch update prompt: {update_prompt_path}",
            file=stdout,
        )
    return result


def _validate_target(target: PatchTarget, provider: ModuleProvider) -> str | None:
    try:
        current = provider.import_module(target.module)
    except Exception:
        return target.module

    for part in target.symbol_path:
        if not hasattr(current, part):
            return target.display_name
        current = getattr(current, part)

    if target.required_parameters:
        try:
            params = signature(current).parameters
        except (TypeError, ValueError):
            return target.display_name
        for name in target.required_parameters:
            if name not in params:
                return f"{target.display_name}({name})"

    return None


def build_update_prompt(old_commit: str, new_commit: str) -> str:
    """Prompt shown when upstream vLLM Metal moved beyond the pin."""

    return f"""Role: You are a senior Python/MLX/vLLM Metal coding agent working in the Audex-Mac repository.

# Goal
Update Audex-Mac's vLLM Metal monkey patches from pinned commit {old_commit} to upstream commit {new_commit}, while preserving the project goal: native local Mac Audex speech-to-speech inference from ./start.sh.

# Success criteria
- Audex-Mac still pins vLLM Metal to an explicit commit.
- docs/engineering/patches.md is updated with every changed patch, upstream symbol/file touched, reason, and reapply notes.
- Patch guards fail loudly if the upstream API shape changes again.
- Existing tests pass, and any new patch behavior has targeted tests.
- ./start.sh can still detect cached Audex models and reach the CLI startup path.
- Do not silently remove the monkey-patch mechanism unless Audex-Mac no longer needs it and the replacement is proven.

# Constraints
- Keep Audex-Mac as the owner of demo integration code; do not fork vLLM Metal into this repo.
- Do not change NVIDIA sampler settings creatively.
- Do not add separate STT/TTS/VAD models.
- Preserve MIT licensing for Audex-Mac code and NVIDIA license notices for model/code artifacts.
- Prefer small, explicit edits over broad refactors.

# Context
- Current pinned vLLM Metal commit: {old_commit}
- Upstream vLLM Metal HEAD: {new_commit}
- Patch ledger: docs/engineering/patches.md
- Startup path: ./start.sh
- Primary validated model: nvidia/Nemotron-Labs-Audex-30B-A3B
- Minimally tested smaller model: nvidia/Nemotron-Labs-Audex-2B

# Validation
After changes, run the most relevant fast tests first, then a startup smoke test. Report any model-download or full-inference steps that were skipped because they are slow or require local cached weights.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enforce Audex-Mac's pinned vLLM Metal patch contract."
    )
    parser.add_argument("--installed-commit", required=True)
    parser.add_argument("--pinned-commit", required=True)
    parser.add_argument("--upstream-head")
    parser.add_argument("--update-prompt-path", type=Path)
    args = parser.parse_args(argv)
    try:
        enforce_patch_guards(
            VllmMetalState(
                installed_commit=args.installed_commit,
                pinned_commit=args.pinned_commit,
                upstream_head=args.upstream_head,
            ),
            update_prompt_path=args.update_prompt_path,
        )
    except PatchGuardError:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
