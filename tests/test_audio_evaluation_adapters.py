from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import Any

import pytest

from audex_mac.audio_evaluation import AudioEvaluationCase, EvaluationTrack
from audex_mac.audio_evaluation_adapters import (
    AudexVllmTtaGenerationAdapter,
    AudexVllmUnderstandingAdapter,
)
from audex_mac.audio_evaluation_generation import TtaOutputInspection
from audex_mac.vllm_runtime import VllmRequestResult


class FakeAsyncRuntime:
    def __init__(self) -> None:
        self.tokenizer = FakeTtaTokenizer()
        self.requests: list[Any] = []
        self.request_batches: list[tuple[Any, ...]] = []
        self.loop_ids: list[int] = []

    async def generate_one_final(self, request: Any) -> VllmRequestResult:
        import asyncio

        self.loop_ids.append(id(asyncio.get_running_loop()))
        self.requests.append(request)
        return VllmRequestResult(
            text="Answer: B",
            token_ids=(11, 12),
            elapsed_seconds=0.25,
            finish_reason="stop",
            request_debug_name=request.debug_name,
        )

    async def generate_many_final(
        self, requests: tuple[Any, ...]
    ) -> tuple[VllmRequestResult, ...]:
        import asyncio

        self.loop_ids.append(id(asyncio.get_running_loop()))
        self.request_batches.append(requests)
        self.requests.extend(requests)
        end = self.tokenizer.get_vocab()["<audiogen_end>"]
        valid_tokens = tuple(
            self.tokenizer.codec_token_id(index)
            for index in _phase_valid_codec_ids(2000)
        )
        return tuple(
            VllmRequestResult(
                text="",
                token_ids=(valid_tokens + (end,)) if index % 2 == 0 else (end,),
                elapsed_seconds=1.5,
                finish_reason="stop",
                request_debug_name=request.debug_name,
            )
            for index, request in enumerate(requests)
        )


class FakeEarlyEndRuntime(FakeAsyncRuntime):
    async def generate_many_final(
        self, requests: tuple[Any, ...]
    ) -> tuple[VllmRequestResult, ...]:
        self.request_batches.append(requests)
        end = self.tokenizer.get_vocab()["<audiogen_end>"]
        early_tokens = tuple(
            self.tokenizer.codec_token_id(index)
            for index in _phase_valid_codec_ids(400)
        )
        return tuple(
            VllmRequestResult(
                text="",
                token_ids=(early_tokens + (end,)) if index % 2 == 0 else (end,),
                elapsed_seconds=0.5,
                finish_reason="stop",
                request_debug_name=request.debug_name,
            )
            for index, request in enumerate(requests)
        )


class FakeTtaTokenizer:
    eos_token_id = 2

    def __init__(self) -> None:
        self._vocab = {
            "<audiogen_start>": 100,
            "<audiogen_end>": 101,
            "<sound>": 102,
            "<|audio_bos|>": 103,
            "<|audio_eos|>": 104,
        }
        self._vocab.update(
            {f"<audiocodec_{index}>": 1000 + index for index in range(4096)}
        )

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def codec_token_id(self, codec_id: int) -> int:
        return self._vocab[f"<audiocodec_{codec_id}>"]

    def encode(self, text: str) -> list[int]:
        tokens = re.findall(r"<unk>|[^\s]+", text)
        return [abs(hash(part)) % 500 + 200 for part in tokens]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )


def _phase_valid_codec_ids(count: int) -> tuple[int, ...]:
    return tuple((index % 4) * 1024 + (index // 4) % 1024 for index in range(count))


def _understanding_case(audio_path: Path) -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id="mmau-1",
        track=EvaluationTrack.UNDERSTANDING,
        dataset_id="fixture",
        dataset_revision="rev",
        dataset_config="default",
        dataset_split="test",
        source_row_id="1",
        source_row_hash="hash",
        license="CC0",
        category="sound",
        prompt="What is heard?\nA. Rain\nB. Bell\nReturn only the single choice letter.",
        expected_answer="B",
        audio_path=str(audio_path),
        choices=("A", "B"),
    )


def _generation_case(case_id: str = "audiocaps-1") -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id=case_id,
        track=EvaluationTrack.GENERATION,
        dataset_id="fixture",
        dataset_revision="rev",
        dataset_config="default",
        dataset_split="test",
        source_row_id="1",
        source_row_hash="hash",
        license="CC0",
        category="audiocaps",
        prompt="A bell rings in a quiet room.",
        caption="A bell rings in a quiet room.",
    )


@pytest.mark.fast
def test_understanding_adapter_builds_isolated_audio_question_request(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "clip.wav"
    _write_silent_wav(audio_path)
    runtime = FakeAsyncRuntime()

    attempt = AudexVllmUnderstandingAdapter(runtime=runtime).answer(
        _understanding_case(audio_path),
        seed=123,
    )

    assert attempt.raw_answer == "Answer: B"
    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert "What is heard?" in request.prompt["prompt"]
    assert "<so_embedding>" in request.prompt["prompt"]
    assert request.sampling.temperature == pytest.approx(0.7)
    assert request.sampling.top_p == pytest.approx(0.9)
    assert request.sampling.seed == 123


@pytest.mark.fast
def test_generation_adapter_uses_tta_cfg_pair_and_injected_decoder(
    tmp_path: Path,
) -> None:
    runtime = FakeAsyncRuntime()
    decoded: list[TtaOutputInspection] = []

    def decoder(
        inspection: TtaOutputInspection,
        destination: Path,
        case: AudioEvaluationCase,
    ) -> None:
        del case
        decoded.append(inspection)
        _write_tone_wav(destination)

    enhanced_calls: list[tuple[Path, Path]] = []

    def enhancer(source: Path, destination: Path, case: AudioEvaluationCase) -> None:
        del case
        enhanced_calls.append((source, destination))
        _write_tone_wav(destination, sample_rate=48_000)

    attempt = AudexVllmTtaGenerationAdapter(
        runtime=runtime,
        raw_dir=tmp_path / "raw",
        enhanced_dir=tmp_path / "enhanced",
        decode_to_wav=decoder,
        enhance_wav=enhancer,
    ).generate(_generation_case(), seed=456)

    assert len(runtime.requests) == 2
    assert runtime.requests[0].sampling.extra_args["cfg_scale"] == 3.0
    assert runtime.requests[0].sampling.extra_args["cfg_role"] == "cond"
    assert runtime.requests[1].sampling.extra_args["cfg_role"] == "uncond"
    assert decoded[0].valid is True
    assert attempt.raw_wav_path.is_file()
    assert attempt.enhanced_wav_path == tmp_path / "enhanced" / "audiocaps-1.wav"
    assert attempt.enhanced_wav_path.is_file()
    assert enhanced_calls == [(attempt.raw_wav_path, attempt.enhanced_wav_path)]
    with wave.open(str(attempt.enhanced_wav_path), "rb") as enhanced:
        assert enhanced.getframerate() == 48_000
        assert enhanced.getnchannels() == 1
        assert enhanced.getnframes() == 4800
    assert attempt.signal_metrics["nonempty"] is True
    assert attempt.signal_metrics["rms"] > 0.0
    assert abs(attempt.signal_metrics["dc_offset"]) < 0.01
    assert attempt.signal_metrics["sample_delta_peak"] > 0.0
    assert attempt.signal_metrics["zero_crossing_rate"] > 0.0
    assert attempt.finish_reason == "stop"


@pytest.mark.fast
def test_generation_adapter_submits_multiple_cfg_pairs_as_one_vllm_batch(
    tmp_path: Path,
) -> None:
    runtime = FakeAsyncRuntime()
    decoded: list[str] = []

    def decoder(
        inspection: TtaOutputInspection,
        destination: Path,
        case: AudioEvaluationCase,
    ) -> None:
        assert inspection.valid
        decoded.append(case.case_id)
        _write_tone_wav(destination)

    attempts = AudexVllmTtaGenerationAdapter(
        runtime=runtime,
        raw_dir=tmp_path / "raw",
        enhanced_dir=tmp_path / "enhanced",
        decode_to_wav=decoder,
    ).generate_many(
        (
            (_generation_case("sound-1"), 111),
            (_generation_case("sound-2"), 222),
        )
    )

    assert len(runtime.request_batches) == 1
    assert len(runtime.request_batches[0]) == 4
    assert [
        request.sampling.extra_args["cfg_role"] for request in runtime.requests
    ] == [
        "cond",
        "uncond",
        "cond",
        "uncond",
    ]
    assert len(attempts) == 2
    assert decoded == ["sound-1", "sound-2"]


@pytest.mark.fast
def test_generation_adapter_decodes_clean_early_end_only_when_enabled(
    tmp_path: Path,
) -> None:
    decoded: list[float] = []

    def decoder(
        inspection: TtaOutputInspection,
        destination: Path,
        case: AudioEvaluationCase,
    ) -> None:
        del case
        decoded.append(inspection.duration_seconds)
        _write_tone_wav(destination)

    attempt = AudexVllmTtaGenerationAdapter(
        runtime=FakeEarlyEndRuntime(),
        raw_dir=tmp_path / "raw",
        decode_to_wav=decoder,
        allow_early_preview_seconds=1.0,
    ).generate(_generation_case(), seed=456)

    assert attempt.structure.valid is False
    assert attempt.structure.failures == ("incomplete_target",)
    assert decoded == [2.0]
    assert attempt.signal_metrics["nonempty"] is True


@pytest.mark.fast
def test_evaluation_adapters_reuse_one_event_loop_for_persistent_vllm_runtime(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "clip.wav"
    _write_silent_wav(audio_path)
    runtime = FakeAsyncRuntime()
    adapter = AudexVllmUnderstandingAdapter(runtime=runtime)

    adapter.answer(_understanding_case(audio_path), seed=1)
    adapter.answer(_understanding_case(audio_path), seed=2)

    assert len(runtime.loop_ids) == 2
    assert len(set(runtime.loop_ids)) == 1


def _write_silent_wav(path: Path) -> None:
    import wave

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 1600)


def _write_tone_wav(path: Path, *, sample_rate: int = 16_000) -> None:
    import math
    import wave

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for index in range(sample_rate // 10):
            sample = int(8000 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate))
            frames.extend(sample.to_bytes(2, "little", signed=True))
        wav.writeframes(bytes(frames))
