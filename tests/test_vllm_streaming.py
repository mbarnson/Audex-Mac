from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from audex_mac.vllm_streaming import inspect_vllm_streaming_support

pytestmark = pytest.mark.fast


class FakeLLM:
    def generate(self):
        return []


class FakeAsyncLLMEngine:
    async def generate(self):
        yield object()


def test_inspect_vllm_streaming_support_detects_async_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_vllm = SimpleNamespace(AsyncLLMEngine=FakeAsyncLLMEngine, LLM=FakeLLM)
    fake_sampling_params = SimpleNamespace(
        RequestOutputKind=SimpleNamespace(
            CUMULATIVE=object(),
            FINAL_ONLY=object(),
        )
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sampling_params)

    support = inspect_vllm_streaming_support()

    assert support.ready_for_async_token_streaming is True
    assert support.sync_llm_generate_streams is False
    assert support.async_generate_is_asyncgen is True


def test_inspect_vllm_streaming_support_reports_missing_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    monkeypatch.delitem(sys.modules, "vllm.sampling_params", raising=False)

    support = inspect_vllm_streaming_support()

    assert support.ready_for_async_token_streaming is False
    assert support.error is not None
