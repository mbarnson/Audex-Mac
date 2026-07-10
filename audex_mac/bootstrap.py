"""Bootstrap policy for the one-command CLI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BootstrapState:
    venv_exists: bool
    dependency_state_matches: bool
    model_cached: bool


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    create_venv: bool
    install_huggingface_hub: bool
    install_pinned_dependencies: bool
    prompt_for_model_download: bool
    proceed_to_model_selection: bool


def plan_bootstrap(state: BootstrapState) -> BootstrapPlan:
    """Return the startup actions required for the current local state."""

    needs_deps = (not state.venv_exists) or (not state.dependency_state_matches)
    return BootstrapPlan(
        create_venv=not state.venv_exists,
        install_huggingface_hub=needs_deps,
        install_pinned_dependencies=needs_deps,
        prompt_for_model_download=not state.model_cached,
        proceed_to_model_selection=not needs_deps,
    )


def model_download_notice(model_id: str, approx_size: str) -> str:
    """User-facing model download notice."""

    return (
        f"Audex-Mac needs to download {model_id} ({approx_size}). "
        "The model is governed by NVIDIA's model license; Audex-Mac's MIT "
        "license applies only to this repository's source code."
    )
