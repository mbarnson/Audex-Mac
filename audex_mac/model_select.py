"""Model cache detection and model selection policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from .checkpoints import verify_snapshot
from .models import DEFAULT_MODEL, HIGHER_REASONING_MODEL, SUPPORTED_MODELS, AudexModel
from .text_chat import AUDEX_CHAT_TEMPLATE_RELATIVE_PATH

ModelReadiness = Literal["speech", "text"]
MACOS_CASE_COLLISION_IGNORE_PATTERNS = ("license/*",)


class ModelCacheProbe(Protocol):
    """Checks whether a Hugging Face snapshot is fully present locally."""

    def is_cached(
        self, model: AudexModel, readiness: ModelReadiness = "speech"
    ) -> bool:
        """Return True when all required model files are already cached."""


class HuggingFaceSnapshotProbe:
    """Cache probe backed by ``huggingface_hub.snapshot_download``."""

    def is_cached(
        self, model: AudexModel, readiness: ModelReadiness = "speech"
    ) -> bool:
        required_files, checkpoint_dirs, allow_patterns = _readiness_requirements(
            model,
            readiness,
        )
        check = verify_snapshot(
            model,
            required_files=required_files,
            checkpoint_dirs=checkpoint_dirs,
        )
        if check.complete:
            return True
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - bootstrap installs this.
            raise RuntimeError(
                "huggingface_hub is required before model cache detection runs."
            ) from exc
        try:
            snapshot_download(
                repo_id=model.repo_id,
                allow_patterns=list(allow_patterns),
                ignore_patterns=list(MACOS_CASE_COLLISION_IGNORE_PATTERNS),
                local_files_only=True,
            )
        except Exception:
            return False
        check = verify_snapshot(
            model,
            required_files=required_files,
            checkpoint_dirs=checkpoint_dirs,
        )
        return check.complete


@dataclass(frozen=True, slots=True)
class ModelSelection:
    selected: AudexModel
    cached: bool
    defaulted: bool
    user_messages: tuple[str, ...]
    log_messages: tuple[str, ...]


def select_model(
    probe: ModelCacheProbe,
    readiness: ModelReadiness = "speech",
) -> ModelSelection:
    """Select the model according to the demo's model policy."""

    cached_models = [
        model for model in SUPPORTED_MODELS if probe.is_cached(model, readiness)
    ]
    if cached_models:
        selected = cached_models[0]
        reason = (
            f"{selected.label} was selected because it was already cached."
            if selected.higher_reasoning
            else f"{selected.label} was selected because it was already cached."
        )
        return ModelSelection(
            selected=selected,
            cached=True,
            defaulted=False,
            user_messages=(),
            log_messages=(reason,),
        )

    user_message = (
        f"Defaulting to {DEFAULT_MODEL.label} for first-run speed and lower "
        "memory pressure."
    )
    higher_message = (
        f"For improved reasoning, pre-download {HIGHER_REASONING_MODEL.repo_id}; "
        "Audex-Mac will use it automatically when fully cached."
    )
    return ModelSelection(
        selected=DEFAULT_MODEL,
        cached=False,
        defaulted=True,
        user_messages=(user_message, higher_message),
        log_messages=(user_message, higher_message),
    )


def _readiness_requirements(
    model: AudexModel,
    readiness: ModelReadiness,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if readiness == "speech":
        return (
            model.speech_required_files,
            model.speech_checkpoint_dirs,
            model.required_patterns,
        )
    if readiness == "text":
        if model.text_checkpoint_dirs == model.speech_checkpoint_dirs:
            return (
                model.text_required_files,
                model.text_checkpoint_dirs,
                model.required_patterns,
            )
        return (
            model.text_required_files + (AUDEX_CHAT_TEMPLATE_RELATIVE_PATH,),
            model.text_checkpoint_dirs,
            (
                "checkpoint_folder_textonly/*",
                AUDEX_CHAT_TEMPLATE_RELATIVE_PATH,
                "inference_scripts_vllm/textonly_scripts/*",
            ),
        )
    raise ValueError(f"Unsupported model readiness target: {readiness}")


def download_model_snapshot(
    model: AudexModel,
    readiness: ModelReadiness = "speech",
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - bootstrap installs this.
        raise RuntimeError(
            "huggingface_hub is required to download Audex models."
        ) from exc

    _, _, allow_patterns = _readiness_requirements(model, readiness)
    snapshot_download(
        repo_id=model.repo_id,
        allow_patterns=list(allow_patterns),
        ignore_patterns=list(MACOS_CASE_COLLISION_IGNORE_PATTERNS),
        local_files_only=False,
    )
