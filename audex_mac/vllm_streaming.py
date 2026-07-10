"""vLLM streaming API inspection for the Audex runtime plan."""

from __future__ import annotations

import inspect
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VllmStreamingSupport:
    sync_llm_generate_streams: bool
    async_engine_available: bool
    async_generate_is_asyncgen: bool
    request_output_kind_available: bool
    cumulative_output_kind_available: bool
    final_only_output_kind_available: bool
    error: str | None = None

    @property
    def ready_for_async_token_streaming(self) -> bool:
        return (
            self.async_engine_available
            and self.async_generate_is_asyncgen
            and self.request_output_kind_available
            and self.cumulative_output_kind_available
            and self.final_only_output_kind_available
            and self.error is None
        )


def inspect_vllm_streaming_support() -> VllmStreamingSupport:
    """Inspect the pinned vLLM API shape needed for token streaming."""

    try:
        from vllm import LLM, AsyncLLMEngine
        from vllm.sampling_params import RequestOutputKind
    except Exception as exc:
        return VllmStreamingSupport(
            sync_llm_generate_streams=False,
            async_engine_available=False,
            async_generate_is_asyncgen=False,
            request_output_kind_available=False,
            cumulative_output_kind_available=False,
            final_only_output_kind_available=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    async_generate = getattr(AsyncLLMEngine, "generate", None)
    sync_generate = getattr(LLM, "generate", None)
    return VllmStreamingSupport(
        sync_llm_generate_streams=inspect.isasyncgenfunction(sync_generate),
        async_engine_available=True,
        async_generate_is_asyncgen=inspect.isasyncgenfunction(async_generate),
        request_output_kind_available=True,
        cumulative_output_kind_available=hasattr(RequestOutputKind, "CUMULATIVE"),
        final_only_output_kind_available=hasattr(RequestOutputKind, "FINAL_ONLY"),
    )
