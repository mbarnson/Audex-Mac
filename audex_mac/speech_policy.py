"""Speech-to-speech policy invariants."""

from __future__ import annotations

FORBIDDEN_SEMANTIC_MODELS = frozenset({"whisper", "kokoro", "silero"})
ALLOWED_AUDIO_PLUMBING = frozenset({"resample", "normalize", "codec_convert"})
NON_THINKING_PREFIX = "<think></think>"


def validate_no_forbidden_models(loaded_model_names: list[str]) -> None:
    """Fail if a separate semantic model is loaded in the STS path."""

    lowered = [name.lower() for name in loaded_model_names]
    for forbidden in FORBIDDEN_SEMANTIC_MODELS:
        if any(forbidden in name for name in lowered):
            raise RuntimeError(f"Forbidden semantic model loaded: {forbidden}")


def assistant_prefix(*, thinking_enabled: bool) -> str:
    """Return the assistant prefix for speech-to-speech generation."""

    return "" if thinking_enabled else NON_THINKING_PREFIX
