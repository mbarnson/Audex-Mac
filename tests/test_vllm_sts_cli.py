from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from audex_mac import vllm_commands, vllm_sts_cli
from audex_mac.audio_contract import SpeechCodecTokenMap
from audex_mac.interactive_input import InputKind, TurnInput
from audex_mac.preemptive_turn import PreemptiveTurnCoordinator
from audex_mac.speech_output import float_samples_to_pcm16_bytes, write_pcm16_wav
from audex_mac.tts_quality import tts_quality_recipe
from audex_mac.tts_text import (
    prepare_text_for_tts,
    split_cfg_spoken_tts_chunks,
    split_spoken_tts_chunks,
    streamed_tts_chunks_from_text,
)
from audex_mac.vllm_commands import (
    run_vllm_tts_quality_probe,
    run_vllm_tts_text_probe,
)
from audex_mac.vllm_runtime import (
    VllmRequestResult,
    VllmSpeechCodecResult,
    VllmStreamDelta,
    VllmTtsCodecStreamEvent,
)
from audex_mac.vllm_speech import (
    DEFAULT_VLLM_INTERLEAVED_PLAYBACK_LATENCY,
    DEFAULT_VLLM_INTERLEAVED_PLAYBACK_PREBUFFER_SECONDS,
    DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES,
    DEFAULT_VLLM_STREAM_DECODER_STEADY_CHUNK_FRAMES,
    _effective_cfg_tts_target_segments,
    _interleaved_initial_ready_batching_enabled,
    _interleaved_tail_batching_enabled,
    _playback_prebuffer_seconds,
    _retain_tts_artifacts_enabled,
    _vllm_tts_cfg_prime_first_segment_enabled,
)
from audex_mac.vllm_sts_cli import (
    VllmSpeechToSpeechSession,
    _enable_cfg_wiring_if_tts_cfg_requested,
    _reset_prefix_cache_before_tts_enabled,
    _sanitize_prompt_history,
    _text_to_tts_interleaving_enabled,
    _token_hash,
    _vllm_tts_cfg_enabled,
)

pytestmark = pytest.mark.fast


def test_vllm_default_decoder_chunk_frames_tracks_playback_tuning() -> None:
    assert DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES == 1
    assert DEFAULT_VLLM_STREAM_DECODER_STEADY_CHUNK_FRAMES == 8


def test_audio_conversation_state_reuses_primed_prefix_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_session(tmp_path, runtime=None, async_runtime=FakeAsyncRuntime())
    tokenizations = 0

    def audio_history_tokens() -> tuple[int, ...]:
        nonlocal tokenizations
        tokenizations += 1
        return (1, 2, 3)

    monkeypatch.setattr(session, "_audio_history_prompt_tokens", audio_history_tokens)

    first = session._audio_conversation_state_kwargs()
    second = session._audio_conversation_state_kwargs()
    session._audio_conversation_state_cache = None
    third = session._audio_conversation_state_kwargs()
    session.shutdown()

    assert first == second == third
    assert tokenizations == 2


def test_vllm_session_switches_conversation_keys_without_reloading_runtime(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    session = make_session(tmp_path, runtime)
    store = SimpleNamespace(save=lambda _conversation: None)
    first = SimpleNamespace(
        conversation_id="chat-one",
        messages=[
            {"role": "system", "content": "System."},
            {"role": "user", "content": "First topic"},
        ],
        max_context_tokens=262_144,
        token_count=10,
    )
    second = SimpleNamespace(
        conversation_id="chat-two",
        messages=[
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Second topic"},
            {"role": "assistant", "content": "Second answer"},
        ],
        max_context_tokens=262_144,
        token_count=20,
    )

    session.activate_conversation(first, store)
    session.activate_conversation(second, store)
    session.activate_conversation(first, store)

    assert session.runtime is runtime
    assert session.conversation is first
    assert session.messages[-1]["content"] == "First topic"
    assert session.turns == 1
    assert session._text_conversation_state_kwargs()["conversation_state_key"] == (
        "chat-one"
    )
    session.shutdown()


def test_no_cfg_model_audio_is_eligible_for_semantic_latency_gate() -> None:
    speech = SimpleNamespace(
        first_audio_ready_at=100.5,
        first_playback_started_at=100.8,
        first_playback_estimated_audible_at=100.95,
        segments=("A direct model answer.",),
    )

    diagnostics = vllm_sts_cli._semantic_audio_diagnostics(
        speech=speech,
        turn_submitted_at=100.0,
        play=True,
        cfg_enabled=False,
        text_to_tts_interleaved=True,
    )

    assert diagnostics["gate_eligible"] is True
    assert diagnostics["gate_passed"] is True


class FakeScalar:
    def __init__(self, value) -> None:
        self.value = value

    def item(self):
        return self.value


class FakeWaveform:
    shape = (640,)

    def __init__(self, samples: tuple[float, ...]) -> None:
        self.samples = samples

    def tolist(self) -> list[float]:
        return list(self.samples)


class FakeMx:
    gpu = "gpu"
    clear_cache_count = 0

    def set_default_device(self, device) -> None:
        self.device = device

    def default_device(self):
        return "Device(gpu, 0)"

    def isfinite(self, waveform: FakeWaveform) -> FakeWaveform:
        return waveform

    def all(self, _value) -> FakeScalar:
        return FakeScalar(True)

    def abs(self, waveform: FakeWaveform) -> FakeWaveform:
        return waveform

    def max(self, waveform: FakeWaveform) -> FakeScalar:
        return FakeScalar(max(abs(sample) for sample in waveform.samples))

    def clear_cache(self) -> None:
        self.clear_cache_count += 1


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        words = text.split()
        return [
            999 if "<so_embedding>" in word else index
            for index, word in enumerate(words)
        ]

    def get_vocab(self) -> dict[str, int]:
        return {"<so_embedding>": 999}

    def apply_chat_template(self, messages, **_kwargs) -> str:
        return "\n".join(
            f"{message['role']}:{message['content']}" for message in messages
        )


class FakeRuntime:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.stats = SimpleNamespace(
            model_load_seconds=1.0,
            engine_class="fake.vllm.Engine",
        )
        self.calls: list[tuple[str, object]] = []

    def transcribe_audio(
        self,
        samples,
        *,
        sample_rate: int = 16_000,
    ) -> VllmRequestResult:
        self.calls.append(("asr", tuple(samples), sample_rate))
        return VllmRequestResult(
            text="hello",
            token_ids=(1,),
            elapsed_seconds=0.1,
            finish_reason="stop",
            request_debug_name="asr",
        )

    def transcribe_projected_audio(
        self,
        projected_embeddings,
        *,
        num_embeddings: int | None = None,
    ) -> VllmRequestResult:
        self.calls.append(("asr-projected", projected_embeddings, num_embeddings))
        return VllmRequestResult(
            text="hello",
            token_ids=(1,),
            elapsed_seconds=0.1,
            finish_reason="stop",
            request_debug_name="asr-projected",
        )

    def generate_text_response_from_messages(
        self,
        messages: list[dict[str, str]],
        *,
        enable_reasoning: bool,
        max_tokens: int | None = None,
        **_state_kwargs,
    ) -> VllmRequestResult:
        self.calls.append(("text", (messages, enable_reasoning, max_tokens)))
        return VllmRequestResult(
            text="Hi back.",
            token_ids=(2,),
            elapsed_seconds=0.2,
            finish_reason="stop",
            request_debug_name="text",
        )

    def generate_tts(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
    ) -> VllmRequestResult:
        self.calls.append(("tts", (text, max_tokens)))
        return VllmRequestResult(
            text="",
            token_ids=(102, 101),
            elapsed_seconds=0.3,
            finish_reason="stop",
            request_debug_name="tts",
        )

    def extract_tts_codec_frames(
        self,
        result: VllmRequestResult,
    ) -> VllmSpeechCodecResult:
        return VllmSpeechCodecResult(
            generated_token_ids=result.token_ids,
            generated_codec_frames=(0, 1),
            reached_end_token=True,
        )


class FakeAsyncRuntime:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.token_map = SpeechCodecTokenMap(
            speech_codec={102: 0, 103: 1},
            speechgen_start=100,
            speechgen_end=101,
        )
        self.stats = SimpleNamespace(
            model_load_seconds=1.5,
            engine_class="fake.vllm.AsyncEngine",
        )
        self.calls: list[tuple[str, object]] = []

    async def reset_prefix_cache(self) -> bool:
        self.calls.append(("reset-prefix-cache", None))
        return True

    async def prime_audio_response_history(self, messages, **state_kwargs):
        self.calls.append(("audio-history-prime", (messages, state_kwargs)))
        return VllmRequestResult(
            text="",
            token_ids=(2,),
            elapsed_seconds=0.01,
            finish_reason="length",
            request_debug_name="audio-history-prime",
        )

    async def transcribe_audio(
        self,
        samples,
        *,
        sample_rate: int = 16_000,
    ) -> VllmRequestResult:
        self.calls.append(("asr", tuple(samples), sample_rate))
        return VllmRequestResult(
            text="hello",
            token_ids=(1,),
            elapsed_seconds=0.1,
            finish_reason="stop",
            request_debug_name="asr",
        )

    async def transcribe_projected_audio(
        self,
        projected_embeddings,
        *,
        num_embeddings: int | None = None,
    ) -> VllmRequestResult:
        self.calls.append(("asr-projected", projected_embeddings, num_embeddings))
        return VllmRequestResult(
            text="hello",
            token_ids=(1,),
            elapsed_seconds=0.1,
            finish_reason="stop",
            request_debug_name="asr-projected",
        )

    async def generate_text_response_from_messages(
        self,
        messages: list[dict[str, str]],
        *,
        enable_reasoning: bool,
        max_tokens: int | None = None,
        **_state_kwargs,
    ) -> VllmRequestResult:
        self.calls.append(("text", (messages, enable_reasoning, max_tokens)))
        return VllmRequestResult(
            text="Hi back.",
            token_ids=(2,),
            elapsed_seconds=0.2,
            finish_reason="stop",
            request_debug_name="text",
        )

    async def stream_text_response_from_messages(
        self,
        messages: list[dict[str, str]],
        *,
        enable_reasoning: bool,
        max_tokens: int | None = None,
        **_state_kwargs,
    ):
        self.calls.append(("text-stream", (messages, enable_reasoning, max_tokens)))
        yield VllmStreamDelta(
            text="Hi back.",
            token_ids=(2,),
            new_token_ids=(2,),
            elapsed_seconds=0.2,
            finished=True,
            finish_reason="stop",
            request_debug_name="text",
            request_id="fake-text",
        )

    async def stream_audio_response_from_messages(
        self,
        messages: list[dict[str, str]],
        samples,
        *,
        sample_rate: int,
        enable_reasoning: bool,
        max_tokens: int | None = None,
        **_state_kwargs,
    ):
        self.calls.append(
            (
                "audio-response-stream",
                (messages, tuple(samples), sample_rate, enable_reasoning, max_tokens),
            )
        )
        yield VllmStreamDelta(
            text="A list is an ordered collection.",
            token_ids=(2,),
            new_token_ids=(2,),
            elapsed_seconds=0.15,
            finished=True,
            finish_reason="stop",
            request_debug_name="audio-response",
            request_id="fake-audio-response",
        )

    async def generate_tts(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
    ) -> VllmRequestResult:
        self.calls.append(("tts", (text, max_tokens)))
        assert max_tokens is not None and max_tokens >= 2400
        return VllmRequestResult(
            text="",
            token_ids=(102, 103, 101),
            elapsed_seconds=0.3,
            finish_reason="stop",
            request_debug_name="tts",
        )

    async def stream_tts_codec_frames(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
    ):
        self.calls.append(("tts-stream", (text, max_tokens)))
        assert max_tokens is not None and max_tokens > 0
        yield VllmTtsCodecStreamEvent(
            generated_token_ids=(102,),
            new_codec_frames=(0,),
            reached_end_token=False,
            finished=False,
            elapsed_seconds=0.1,
        )
        yield VllmTtsCodecStreamEvent(
            generated_token_ids=(102, 103),
            new_codec_frames=(1,),
            reached_end_token=False,
            finished=False,
            elapsed_seconds=0.2,
        )
        yield VllmTtsCodecStreamEvent(
            generated_token_ids=(102, 103, 101),
            new_codec_frames=(),
            reached_end_token=True,
            finished=True,
            elapsed_seconds=0.3,
        )

    async def stream_tts_cfg_codec_frames(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
    ):
        self.calls.append(("tts-cfg-stream", (text, max_tokens)))
        assert max_tokens is not None and max_tokens > 0
        yield VllmTtsCodecStreamEvent(
            generated_token_ids=(103,),
            new_codec_frames=(1,),
            reached_end_token=False,
            finished=False,
            elapsed_seconds=0.1,
        )
        yield VllmTtsCodecStreamEvent(
            generated_token_ids=(103, 101),
            new_codec_frames=(),
            reached_end_token=True,
            finished=True,
            elapsed_seconds=0.2,
        )

    async def stream_tts_cfg_segments_codec_frames(
        self,
        segments: tuple[str, ...],
        *,
        max_tokens_per_segment: tuple[int | None, ...],
        prime_first_segment: bool = False,
    ):
        self.calls.append(
            (
                "tts-cfg-segmented-stream",
                (segments, max_tokens_per_segment, prime_first_segment),
            )
        )
        for index, _segment in enumerate(segments):
            yield VllmTtsCodecStreamEvent(
                generated_token_ids=(),
                new_codec_frames=(index + 10,),
                reached_end_token=False,
                finished=False,
                elapsed_seconds=0.1 + index,
                segment_index=index,
                segment_finished=False,
            )
            yield VllmTtsCodecStreamEvent(
                generated_token_ids=(103, 101),
                new_codec_frames=(),
                reached_end_token=True,
                finished=index == len(segments) - 1,
                elapsed_seconds=0.2 + index,
                segment_index=index,
                segment_finished=True,
            )

    async def stream_tts_segmented_codec_frames(
        self,
        segments: tuple[str, ...],
        *,
        max_tokens_per_segment: tuple[int | None, ...],
    ):
        self.calls.append(("tts-segmented-stream", (segments, max_tokens_per_segment)))
        for index, _segment in enumerate(segments):
            yield VllmTtsCodecStreamEvent(
                generated_token_ids=(),
                new_codec_frames=(index,),
                reached_end_token=False,
                finished=False,
                elapsed_seconds=0.1 + index,
                segment_index=index,
                segment_finished=False,
            )
            yield VllmTtsCodecStreamEvent(
                generated_token_ids=(102, 101),
                new_codec_frames=(),
                reached_end_token=True,
                finished=index == len(segments) - 1,
                elapsed_seconds=0.2 + index,
                segment_index=index,
                segment_finished=True,
            )


def make_session(
    tmp_path: Path,
    runtime: FakeRuntime | None,
    *,
    async_runtime: FakeAsyncRuntime | None = None,
) -> VllmSpeechToSpeechSession:
    session = object.__new__(VllmSpeechToSpeechSession)
    session.full_model_path = tmp_path / "model"
    session.decoder_path = tmp_path / "decoder"
    session.selected_model_repo = "nvidia/Nemotron-Labs-Audex-2B"
    session.output_dir = tmp_path
    session.thinking_enabled = False
    session.response_max_tokens = 4096
    session.speech_max_tokens = None
    session.persona = SimpleNamespace(
        persona_id="assistant",
        path=tmp_path / "assistant.md",
        system_prompt="System.",
    )
    session.conversation_store = None
    session.conversation = None
    session.messages = [{"role": "system", "content": "System."}]
    session.turns = 0
    session._last_interleaved_text_stream_stats = {}
    session._audio_conversation_state_cache = None
    session.mx = FakeMx()
    session.runtime = runtime
    session.async_runtime = async_runtime
    session._async_loop = (
        asyncio.new_event_loop()
        if async_runtime is not None and runtime is None
        else None
    )
    session.decoder_config = SimpleNamespace(sample_rate=16_000, hop_length=320)
    session.decoder_weights = {}
    session.decoder_load_seconds = 0.5
    session.audio_component_load_seconds = 0.25
    return session


def test_vllm_resumed_startup_greeting_uses_history_without_persisting_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeAsyncRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=runtime)
    session.conversation = SimpleNamespace(
        conversation_id="conversation-1",
        user_name="Matt",
        max_context_tokens=262_144,
    )
    session.messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Tell me about the history of Gilroy."},
        {"role": "assistant", "content": "Gilroy grew around agriculture."},
    ]
    original_messages = list(session.messages)

    session.speak_startup_greeting(conversation_resumed=True, play=False)

    assert session.messages == original_messages
    text_calls = [call for call in runtime.calls if call[0] == "text"]
    assert len(text_calls) == 1
    greeting_messages, enable_reasoning, max_tokens = text_calls[0][1]
    assert greeting_messages[:-1] == original_messages
    assert greeting_messages[-1]["role"] == "user"
    assert "most recent substantive exchange" in greeting_messages[-1]["content"]
    assert enable_reasoning is False
    assert max_tokens == 96
    assert "Audex STS: startup greeting: Hi back." in capsys.readouterr().out
    session.shutdown()


def test_vllm_resumed_startup_greeting_is_limited_to_two_sentences() -> None:
    assert (
        vllm_sts_cli._limit_spoken_sentences(
            "Welcome back. We discussed Gilroy. What is next? This must be dropped.",
            max_sentences=2,
        )
        == "Welcome back. We discussed Gilroy."
    )


def test_vllm_typed_turn_skips_asr_and_speaks_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeAsyncRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=runtime)
    session.tts_cfg_enabled = True
    speech_calls: list[dict[str, object]] = []

    async def fake_speech(**kwargs):
        speech_calls.append(kwargs)
        return SimpleNamespace(
            wav_path=tmp_path / "answer.wav",
            run_log_path=tmp_path / "speech.json",
            first_audio_ready_seconds=0.3,
            first_playback_started_seconds=0.4,
            first_audio_ready_at=100.6,
            first_playback_started_at=100.875,
            first_playback_estimated_audible_at=100.925,
            segments=("Hi back.",),
            reached_end_token=True,
            hit_max_tokens=False,
        )

    monkeypatch.setattr(
        session,
        "generate_speech_output_streaming_from_async_runtime",
        fake_speech,
    )
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli._text_to_tts_interleaving_enabled",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli._reset_prefix_cache_before_tts_enabled",
        lambda: False,
    )

    result = session.run_turn_from_text(
        user_text="First line.\n\nSecond line.",
        play=True,
        turn_submitted_at=100.0,
    )

    assert result.transcript == "First line.\n\nSecond line."
    assert result.input_wav_path is None
    assert result.response_text == "Hi back."
    assert [call[0] for call in runtime.calls] == ["text"]
    assert speech_calls[0]["text"] == "Hi back."
    log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert log["input_mode"] == "text"
    assert log["typed_text"] == "First line.\n\nSecond line."
    assert log["asr_skipped"] is True
    assert log["semantic_audio"]["source"] == "model_response_tts"
    assert log["semantic_audio"]["first_spoken_chunk_text"] == "Hi back."
    assert log["semantic_audio"]["cfg_enabled"] is True
    assert log["semantic_audio"]["gate_seconds"] == 1.0
    assert log["semantic_audio"]["gate_passed"] is True
    assert log["semantic_audio"]["measurement_endpoint"] == (
        "estimated_first_dac_sample"
    )
    assert log["timings"]["turn_submit_to_first_audio_ready_seconds"] == 0.6
    assert log["timings"]["turn_submit_to_first_device_write_seconds"] == 0.875
    assert log["timings"]["turn_submit_to_first_estimated_audible_seconds"] == 0.925
    session.shutdown()


def test_vllm_interactive_cli_routes_typed_text_and_shuts_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    result = vllm_sts_cli.SpeechToSpeechTurnResult(
        transcript="Typed question",
        response_text="Spoken answer",
        input_wav_path=None,
        output_wav_path=tmp_path / "answer.wav",
        run_log_path=tmp_path / "turn.json",
        played=True,
    )

    class FakeSession:
        def __init__(self, **kwargs) -> None:
            pass

        @property
        def stats(self):
            return vllm_sts_cli.VllmSpeechToSpeechSessionStats(1.0, 0.0, 0.5, 0)

        def speak_startup_greeting(self, **kwargs) -> None:
            pass

        def run_turn_from_text(
            self,
            *,
            user_text: str,
            play: bool,
            turn_submitted_at: float | None = None,
        ):
            assert turn_submitted_at is not None
            events.append(("text", user_text, play))
            return result

        def shutdown(self) -> None:
            events.append("shutdown")

    inputs = iter(
        [TurnInput(InputKind.TEXT, "Typed question"), TurnInput(InputKind.QUIT)]
    )
    monkeypatch.setattr(vllm_commands, "VllmSpeechToSpeechSession", FakeSession)
    monkeypatch.setattr(vllm_commands, "read_turn_input", lambda: next(inputs))
    monkeypatch.setattr(
        vllm_commands,
        "_start_recording",
        lambda: pytest.fail("typed input must not start recording"),
    )

    actual = vllm_commands.run_vllm_interactive_ptt(
        full_model_path=tmp_path / "model",
        decoder_path=tmp_path / "decoder",
        output_dir=tmp_path,
        play=True,
    )

    assert actual is result
    assert events == [("text", "Typed question", True), "shutdown"]


def test_vllm_interactive_cli_alternates_typed_and_spoken_turns_in_one_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    def turn_result(label: str, input_wav_path: Path | None = None):
        return vllm_sts_cli.SpeechToSpeechTurnResult(
            transcript=label,
            response_text=f"Answer to {label}",
            input_wav_path=input_wav_path,
            output_wav_path=tmp_path / f"{label}.wav",
            run_log_path=tmp_path / f"{label}.json",
            played=True,
        )

    class FakeRecording:
        def stop(self) -> list[float]:
            return [0.0, 0.1, -0.1]

    class FakeSession:
        def __init__(self, **kwargs) -> None:
            events.append("session")

        @property
        def stats(self):
            return vllm_sts_cli.VllmSpeechToSpeechSessionStats(1.0, 0.0, 0.5, 0)

        def speak_startup_greeting(self, **kwargs) -> None:
            pass

        def run_turn_from_text(
            self,
            *,
            user_text: str,
            play: bool,
            turn_submitted_at: float | None = None,
        ):
            assert turn_submitted_at is not None
            events.append(("text", user_text))
            return turn_result(user_text)

        def run_turn_from_wav(
            self,
            *,
            input_wav_path: Path,
            play: bool,
            turn_submitted_at: float | None = None,
        ):
            assert turn_submitted_at is not None
            events.append(("speech", input_wav_path.is_file()))
            return turn_result("speech", input_wav_path)

        def run_preemptive_recorded_turn(
            self,
            *,
            recording,
            input_wav_path: Path,
            play: bool,
        ):
            write_pcm16_wav(
                input_wav_path,
                recording.stop(),
                sample_rate=16_000,
            )
            events.append(("speech", input_wav_path.is_file()))
            return turn_result("speech", input_wav_path)

        def shutdown(self) -> None:
            events.append("shutdown")

    inputs = iter(
        [
            TurnInput(InputKind.TEXT, "typed one"),
            TurnInput(InputKind.RECORD),
            TurnInput(InputKind.TEXT, "typed two"),
            TurnInput(InputKind.QUIT),
        ]
    )
    monkeypatch.setattr(vllm_commands, "VllmSpeechToSpeechSession", FakeSession)
    monkeypatch.setattr(vllm_commands, "read_turn_input", lambda: next(inputs))
    monkeypatch.setattr(vllm_commands, "_start_recording", FakeRecording)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")

    actual = vllm_commands.run_vllm_interactive_ptt(
        full_model_path=tmp_path / "model",
        decoder_path=tmp_path / "decoder",
        output_dir=tmp_path,
        play=True,
    )

    assert actual is not None and actual.transcript == "typed two"
    assert events == [
        "session",
        ("text", "typed one"),
        ("speech", True),
        ("text", "typed two"),
        "shutdown",
    ]


def test_vllm_session_lazy_loads_cli_projection_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlx_package = ModuleType("mlx")
    fake_mx = ModuleType("mlx.core")
    fake_mx.gpu = "gpu"
    fake_mx.set_default_device = lambda _device: None
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    audio_loads: list[str] = []

    def fake_audio_loader(name: str):
        def load(_path: Path):
            audio_loads.append(name)
            return (
                {"conv1.weight": SimpleNamespace(dtype="bf16")}
                if name == "encoder_weights"
                else name
            )

        return load

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.load_audio_encoder_config",
        fake_audio_loader("encoder_config"),
    )
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.load_audio_encoder_weights_mlx",
        fake_audio_loader("encoder_weights"),
    )
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.load_audio_projector_config",
        fake_audio_loader("projector_config"),
    )
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.load_audio_projector_weights_mlx",
        fake_audio_loader("projector_weights"),
    )
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.load_speech_decoder_config",
        lambda _path: SimpleNamespace(sample_rate=16_000, hop_length=320),
    )
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.load_speech_decoder_weights_mlx",
        lambda _path: {},
    )

    session = VllmSpeechToSpeechSession(
        full_model_path=tmp_path / "model",
        decoder_path=tmp_path / "decoder",
        runtime=FakeRuntime(),
    )

    assert audio_loads == []
    assert session.audio_component_load_seconds == 0.0

    session._ensure_audio_projection_components_loaded()
    session._ensure_audio_projection_components_loaded()

    assert audio_loads == [
        "encoder_config",
        "encoder_weights",
        "projector_config",
        "projector_weights",
    ]
    assert session.audio_component_load_seconds >= 0.0


def test_vllm_sts_session_runs_asr_text_tts_decoder_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    runtime = FakeRuntime()
    session = make_session(tmp_path, runtime)

    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.weights = weights
            self.config = config
            self.chunk_frames = chunk_frames
            self.pushed: list[tuple[tuple[int, ...], ...]] = []
            self.flush_count = 0
            decoder_sessions.append(self)

        def push(self, frames):
            self.pushed.append(tuple(tuple(frame) for frame in frames))
            return [(self.config.sample_rate, FakeWaveform((0.0, 0.2, -0.2)))]

        def flush(self):
            self.flush_count += 1
            return [(self.config.sample_rate, FakeWaveform((0.0,) * 637))]

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.run_turn_from_wav(input_wav_path=input_wav, play=False)

    assert result.transcript == "hello"
    assert result.response_text == "Hi back."
    assert result.output_wav_path.is_file()
    assert result.run_log_path.is_file()
    assert runtime.calls[0][0] == "asr"
    assert runtime.calls[0][2] == 16_000
    assert len(runtime.calls[0][1]) == 3
    assert runtime.calls[1] == (
        "text",
        (
            [
                {"role": "system", "content": "System."},
                {"role": "user", "content": "hello"},
            ],
            False,
            4096,
        ),
    )
    assert runtime.calls[2][0] == "tts"
    assert runtime.calls[2][1][0] == "Hi back."
    assert runtime.calls[2][1][1] >= 2400
    assert len(decoder_sessions) == 1
    decoder_session = decoder_sessions[0]
    assert decoder_session.chunk_frames == DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES
    assert decoder_session.pushed == [((0,), (1,))]
    assert decoder_session.flush_count == 1
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["backend"] == "vllm"
    assert run_log["vllm"]["engine_class"] == "fake.vllm.Engine"
    assert run_log["vllm"]["tts_reached_end_token"] is True
    speech_log = json.loads(
        Path(run_log["speech_output_run_log_path"]).read_text(encoding="utf-8")
    )
    assert speech_log["decoder_streaming"] is True
    assert speech_log["vllm_token_streaming"] is False
    assert len(speech_log["chunk_wav_paths"]) == 2
    assert session.messages == [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi back."},
    ]


def test_vllm_text_only_turns_share_history_without_running_tts(tmp_path: Path) -> None:
    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    runtime = FakeRuntime()
    session = make_session(tmp_path, runtime)

    typed = session.run_text_only_turn_from_text(user_text="Typed first.")
    spoken = session.run_text_only_turn_from_wav(input_wav_path=input_wav)

    assert typed.transcript == "Typed first."
    assert typed.response_text == "Hi back."
    assert typed.input_wav_path is None
    assert spoken.transcript == "hello"
    assert spoken.response_text == "Hi back."
    assert spoken.input_wav_path == input_wav
    assert [call[0] for call in runtime.calls] == ["text", "asr", "text"]
    assert [message["role"] for message in session.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    second_prompt = runtime.calls[2][1][0]
    assert second_prompt[1:4] == [
        {"role": "user", "content": "Typed first."},
        {"role": "assistant", "content": "Hi back."},
        {"role": "user", "content": "hello"},
    ]
    assert json.loads(typed.run_log_path.read_text())["output_mode"] == "text"
    assert json.loads(spoken.run_log_path.read_text())["input_mode"] == "speech"
    session.shutdown()


def test_vllm_audio_understanding_answers_prompt_without_mutating_chat_history(
    tmp_path: Path,
) -> None:
    class UnderstandingRuntime(FakeAsyncRuntime):
        async def generate_one_final(self, request):
            self.calls.append(("understanding", request))
            return VllmRequestResult(
                text="A distant train horn echoes through rain.",
                token_ids=(7, 8),
                elapsed_seconds=0.4,
                finish_reason="stop",
                request_debug_name="audio-understanding",
            )

    input_wav = tmp_path / "ambience.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    runtime = UnderstandingRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=runtime)
    original_messages = list(session.messages)

    result = session.understand_audio(
        input_wav_path=input_wav,
        prompt="What is happening in this recording?",
    )

    assert result.transcript == "What is happening in this recording?"
    assert result.response_text == "A distant train horn echoes through rain."
    assert result.input_wav_path == input_wav
    assert session.messages == original_messages
    assert runtime.calls[0][0] == "understanding"
    request = runtime.calls[0][1]
    assert request.debug_name == "audio-understanding"
    assert request.prompt["multi_modal_data"]["audio"]
    session.shutdown()


def test_vllm_sts_session_scrubs_prompt_leakage_before_static_tts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LeakyRuntime(FakeRuntime):
        def generate_text_response_from_messages(
            self,
            messages: list[dict[str, str]],
            *,
            enable_reasoning: bool,
            max_tokens: int | None = None,
            **_state_kwargs,
        ) -> VllmRequestResult:
            self.calls.append(("text", (messages, enable_reasoning, max_tokens)))
            return VllmRequestResult(
                text=(
                    "Audex was built by NVIDIA based on the "
                    "Nemotron-Cascade-2 architecture.\n"
                    "It is ready to help you with conversations about code.\n"
                    "[CRITICAL] Place each sentence on its own separate line.\n"
                    "Go routines are lightweight concurrent functions."
                ),
                token_ids=(2,),
                elapsed_seconds=0.2,
                finish_reason="stop",
                request_debug_name="text",
            )

    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    runtime = LeakyRuntime()
    session = make_session(tmp_path, runtime)

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return [(self.config.sample_rate, FakeWaveform((0.0, 0.0)))]

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.run_turn_from_wav(input_wav_path=input_wav, play=False)

    assert result.response_text == "Go routines are lightweight concurrent functions."
    assert runtime.calls[2][1][0] == "Go routines are lightweight concurrent functions."
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "Go routines are lightweight concurrent functions.",
    }


def test_vllm_sts_session_can_run_entire_turn_on_async_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    async_runtime = FakeAsyncRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=async_runtime)

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return [(self.config.sample_rate, FakeWaveform((0.0, 0.0)))]

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.run_turn_from_wav(input_wav_path=input_wav, play=False)

    assert result.transcript == "hello"
    assert result.response_text == "Hi back."
    assert async_runtime.calls[0][0] == "asr"
    assert async_runtime.calls[0][2] == 16_000
    assert len(async_runtime.calls[0][1]) == 3
    assert async_runtime.calls[1] == (
        "text-stream",
        (
            [
                {"role": "system", "content": "System."},
                {"role": "user", "content": "hello"},
            ],
            False,
            4096,
        ),
    )
    assert async_runtime.calls[2][0] == "tts-stream"
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["vllm"]["engine_class"] == "fake.vllm.AsyncEngine"
    assert run_log["text_to_tts_interleaved"] is True
    assert run_log["text_to_tts_streaming"]["text_stream_event_count"] == 1
    assert run_log["text_to_tts_streaming"]["tts_text_chunk_count"] == 1
    assert run_log["text_to_tts_streaming"]["text_to_tts_chunk_chars"] == [
        len("Hi back.")
    ]
    assert (
        run_log["text_to_tts_streaming"]["first_text_to_tts_chunk_seconds"] is not None
    )
    speech_log = json.loads(
        Path(run_log["speech_output_run_log_path"]).read_text(encoding="utf-8")
    )
    assert speech_log["streaming"] is True
    assert speech_log["vllm_token_streaming"] is True
    assert speech_log["text_to_tts_interleaved"] is True
    assert speech_log["first_tts_chunk_ready_seconds"] is not None
    assert speech_log["tts_segment_ready_seconds"]


def test_direct_audio_response_defers_audex_asr_until_tts_generation_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    runtime = FakeAsyncRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=runtime)

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return [(self.config.sample_rate, FakeWaveform((0.0, 0.0)))]

    monkeypatch.setenv("AUDEX_VLLM_DIRECT_AUDIO_RESPONSE", "1")
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.run_turn_from_wav(input_wav_path=input_wav, play=False)

    assert result.transcript == "hello"
    assert result.response_text == "A list is an ordered collection."
    assert [call[0] for call in runtime.calls] == [
        "audio-response-stream",
        "tts-stream",
        "asr",
        "audio-history-prime",
    ]
    assert session.messages == [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "A list is an ordered collection."},
    ]
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["response_source"] == "audio"
    assert run_log["text_to_tts_streaming"]["text_stream_event_count"] == 1
    assert run_log["timings"]["asr_wall_seconds"] >= 0


def test_direct_audio_response_can_be_disabled_per_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    runtime = FakeAsyncRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=runtime)

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return [(self.config.sample_rate, FakeWaveform((0.0, 0.0)))]

    monkeypatch.setenv("AUDEX_VLLM_DIRECT_AUDIO_RESPONSE", "1")
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.run_turn_from_wav(
        input_wav_path=input_wav,
        play=False,
        direct_audio_response=False,
    )

    assert result.transcript == "hello"
    assert result.response_text == "Hi back."
    assert [call[0] for call in runtime.calls] == [
        "asr",
        "text-stream",
        "tts-stream",
        "audio-history-prime",
    ]
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["response_source"] == "transcript"


def test_preemptive_spoken_turn_stages_audex_asr_and_cfg_audio_before_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    runtime = FakeAsyncRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=runtime)
    session.tts_cfg_enabled = True
    recording = vllm_sts_cli._Recording(stream=FakeStream())
    recording.append_chunk([[0.02], [-0.02]] * 16)
    recording.append_chunk([[0.0]] * 96)
    input_wav = tmp_path / "preemptive.wav"
    writes: list[bytes] = []

    class FakeRawOutputStream:
        latency = 0.02

        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

        def write(self, pcm: bytes) -> bool:
            writes.append(pcm)
            return False

    class FakeSoundDevice:
        RawOutputStream = FakeRawOutputStream

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return []

        def reset(self) -> None:
            pass

    def delayed_submit() -> float:
        import time

        time.sleep(0.03)
        return time.time()

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice())
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.PreemptiveTurnCoordinator",
        lambda: PreemptiveTurnCoordinator(
            min_recording_seconds=0.001,
            silence_seconds=0.005,
            poll_seconds=0.001,
        ),
    )
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli._wait_for_turn_submission",
        delayed_submit,
    )

    try:
        result = session.run_preemptive_recorded_turn(
            recording=recording,
            input_wav_path=input_wav,
            play=True,
        )
    finally:
        session.shutdown()

    assert input_wav.is_file()
    assert result.transcript == "hello"
    assert [call[0] for call in runtime.calls] == [
        "asr",
        "text-stream",
        "tts-cfg-stream",
    ]
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["preemptive_generation"]["enabled"] is True
    assert run_log["preemptive_generation"]["staged_voice_revision"] == 1
    assert run_log["preemptive_generation"]["final_voice_revision"] == 1
    assert run_log["semantic_audio"]["source"] == "model_response_tts"
    assert run_log["semantic_audio"]["cfg_enabled"] is True
    assert run_log["semantic_audio"]["measurement_endpoint"] == (
        "estimated_first_dac_sample"
    )
    assert run_log["semantic_audio"]["gate_passed"] is True
    assert writes
    speech_log = json.loads(
        Path(run_log["speech_output_run_log_path"]).read_text(encoding="utf-8")
    )
    assert speech_log["playback_start_gated"] is True
    assert speech_log["playback_gate_released"] is True


def test_vllm_sts_session_can_reset_prefix_cache_before_static_tts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    async_runtime = FakeAsyncRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=async_runtime)
    monkeypatch.setenv("AUDEX_VLLM_TTS_CFG", "1")
    monkeypatch.setenv("AUDEX_VLLM_STREAM_TEXT_TO_TTS", "0")

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return [(self.config.sample_rate, FakeWaveform((0.0, 0.0)))]

        def reset(self):
            return None

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.run_turn_from_wav(input_wav_path=input_wav, play=False)

    assert [call[0] for call in async_runtime.calls] == [
        "asr",
        "text",
        "reset-prefix-cache",
        "tts-cfg-stream",
    ]
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["tts_prefix_cache_reset"] is True


def test_vllm_prefix_cache_reset_defaults_only_for_cfg_tts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUDEX_VLLM_TTS_CFG", raising=False)
    monkeypatch.delenv("AUDEX_VLLM_RESET_PREFIX_CACHE_BEFORE_TTS", raising=False)
    assert _reset_prefix_cache_before_tts_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_TTS_CFG", "1")
    assert _reset_prefix_cache_before_tts_enabled() is True

    monkeypatch.setenv("AUDEX_VLLM_RESET_PREFIX_CACHE_BEFORE_TTS", "0")
    assert _reset_prefix_cache_before_tts_enabled() is False


def test_vllm_sts_session_fails_if_cfg_prefix_cache_reset_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    async_runtime = FakeAsyncRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=async_runtime)
    monkeypatch.setenv("AUDEX_VLLM_TTS_CFG", "1")
    monkeypatch.setenv("AUDEX_VLLM_STREAM_TEXT_TO_TTS", "0")

    async def refuse_reset() -> bool:
        return False

    monkeypatch.setattr(async_runtime, "reset_prefix_cache", refuse_reset)

    with pytest.raises(RuntimeError, match="could not reset its prefix cache"):
        session.run_turn_from_wav(input_wav_path=input_wav, play=False)


def test_interleaved_tts_scrubs_prompt_leakage_before_speaking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LeakyAsyncRuntime(FakeAsyncRuntime):
        async def stream_text_response_from_messages(
            self,
            messages: list[dict[str, str]],
            *,
            enable_reasoning: bool,
            max_tokens: int | None = None,
            **_state_kwargs,
        ):
            self.calls.append(("text-stream", (messages, enable_reasoning, max_tokens)))
            yield VllmStreamDelta(
                text=(
                    "Audex is created by NVIDIA based on the "
                    "Nemotron-Cascade-2 architecture.\n"
                    "[CRITICAL] Place each sentence on its own separate line.\n"
                    "Rust uses ownership to keep memory safe."
                ),
                token_ids=(2,),
                new_token_ids=(2,),
                elapsed_seconds=0.2,
                finished=True,
                finish_reason="stop",
                request_debug_name="text",
                request_id="fake-text",
            )

    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    async_runtime = LeakyAsyncRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=async_runtime)

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return [(self.config.sample_rate, FakeWaveform((0.0, 0.0)))]

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.run_turn_from_wav(input_wav_path=input_wav, play=False)

    assert result.response_text == "Rust uses ownership to keep memory safe."
    assert async_runtime.calls[2] == (
        "tts-stream",
        ("Rust uses ownership to keep memory safe.", 512),
    )
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "Rust uses ownership to keep memory safe.",
    }


def test_interleaved_tts_scrubs_incremental_prompt_leakage_before_speaking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncrementalLeakyAsyncRuntime(FakeAsyncRuntime):
        async def stream_text_response_from_messages(
            self,
            messages: list[dict[str, str]],
            *,
            enable_reasoning: bool,
            max_tokens: int | None = None,
            **_state_kwargs,
        ):
            self.calls.append(("text-stream", (messages, enable_reasoning, max_tokens)))
            yield VllmStreamDelta(
                text="Audex was built by NVIDIA based on the Nemotron-Cascade-2 architecture.\n",
                token_ids=(2,),
                new_token_ids=(2,),
                elapsed_seconds=0.1,
                finished=False,
                finish_reason=None,
                request_debug_name="text",
                request_id="fake-text",
            )
            yield VllmStreamDelta(
                text=(
                    "Audex was built by NVIDIA based on the Nemotron-Cascade-2 architecture.\n"
                    "It is ready to help you with conversations about code and ideas.\n"
                    "[CRITICAL] Place each sentence on its own separate line.\n"
                    "Go routines are lightweight functions that run concurrently."
                ),
                token_ids=(2, 3),
                new_token_ids=(3,),
                elapsed_seconds=0.2,
                finished=True,
                finish_reason="stop",
                request_debug_name="text",
                request_id="fake-text",
            )

    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    async_runtime = IncrementalLeakyAsyncRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=async_runtime)

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return [(self.config.sample_rate, FakeWaveform((0.0, 0.0)))]

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.run_turn_from_wav(input_wav_path=input_wav, play=False)

    assert (
        result.response_text
        == "Go routines are lightweight functions that run concurrently."
    )
    assert async_runtime.calls[2] == (
        "tts-stream",
        ("Go routines are lightweight functions that run concurrently.", 512),
    )
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "Go routines are lightweight functions that run concurrently.",
    }


def test_text_to_tts_interleaving_defaults_on_for_non_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUDEX_VLLM_STREAM_TEXT_TO_TTS", raising=False)
    monkeypatch.delenv("AUDEX_VLLM_TTS_CFG", raising=False)

    assert _text_to_tts_interleaving_enabled(thinking_enabled=False) is True
    assert _text_to_tts_interleaving_enabled(thinking_enabled=True) is False

    monkeypatch.setenv("AUDEX_VLLM_STREAM_TEXT_TO_TTS", "0")
    assert _text_to_tts_interleaving_enabled(thinking_enabled=False) is False

    monkeypatch.setenv("AUDEX_VLLM_STREAM_TEXT_TO_TTS", "off")
    assert _text_to_tts_interleaving_enabled(thinking_enabled=False) is False

    monkeypatch.setenv("AUDEX_VLLM_STREAM_TEXT_TO_TTS", "1")
    assert _text_to_tts_interleaving_enabled(thinking_enabled=False) is True


def test_cfg_tts_env_keeps_text_to_tts_interleaving_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_STREAM_TEXT_TO_TTS", "1")
    monkeypatch.setenv("AUDEX_VLLM_TTS_CFG", "1")

    assert _vllm_tts_cfg_enabled() is True
    assert _text_to_tts_interleaving_enabled(thinking_enabled=False) is True


def test_dynamic_interleaved_chunk_uses_honest_cfg_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeAsyncRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=runtime)
    session.tts_cfg_enabled = True

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return []

        def reset(self) -> None:
            pass

    async def chunks():
        yield "This is the first coherent semantic sentence."

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    try:
        result = session._run_async(
            session.generate_speech_output_streaming_from_async_runtime(
                text="fallback",
                max_tokens=2400,
                play=False,
                tts_chunk_source=chunks(),
                text_to_tts_interleaved=True,
            )
        )
    finally:
        session.shutdown()

    tts_calls = [call for call in runtime.calls if call[0].startswith("tts")]
    assert tts_calls == [
        (
            "tts-cfg-stream",
            ("This is the first coherent semantic sentence.", 512),
        )
    ]
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["tts_cfg_enabled"] is True
    assert run_log["text_to_tts_interleaved"] is True


def test_cfg_tts_env_enables_vllm_cfg_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUDEX_VLLM_ENABLE_CFG_WIRING", raising=False)
    monkeypatch.delenv("AUDEX_VLLM_TTS_CFG", raising=False)

    _enable_cfg_wiring_if_tts_cfg_requested()
    assert "AUDEX_VLLM_ENABLE_CFG_WIRING" not in os.environ

    monkeypatch.setenv("AUDEX_VLLM_TTS_CFG", "1")
    _enable_cfg_wiring_if_tts_cfg_requested()
    assert os.environ["AUDEX_VLLM_ENABLE_CFG_WIRING"] == "1"


def test_cfg_tts_prime_first_segment_defaults_off_and_can_be_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUDEX_VLLM_TTS_CFG_PRIME_FIRST_SEGMENT", raising=False)
    assert _vllm_tts_cfg_prime_first_segment_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_TTS_CFG_PRIME_FIRST_SEGMENT", "0")
    assert _vllm_tts_cfg_prime_first_segment_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_TTS_CFG_PRIME_FIRST_SEGMENT", "false")
    assert _vllm_tts_cfg_prime_first_segment_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_TTS_CFG_PRIME_FIRST_SEGMENT", "1")
    assert _vllm_tts_cfg_prime_first_segment_enabled() is True


def test_cfg_tts_target_segments_env_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUDEX_VLLM_CFG_TTS_TARGET_SEGMENTS", raising=False)
    monkeypatch.delenv("AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS", raising=False)
    assert _effective_cfg_tts_target_segments(8) == 8

    monkeypatch.setenv("AUDEX_VLLM_CFG_TTS_TARGET_SEGMENTS", "6")
    assert _effective_cfg_tts_target_segments(8) == 6

    monkeypatch.setenv("AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS", "4")
    assert _effective_cfg_tts_target_segments(8) == 4

    monkeypatch.setenv("AUDEX_VLLM_CFG_TTS_TARGET_SEGMENTS", "0")
    assert _effective_cfg_tts_target_segments(8) == 4

    monkeypatch.setenv("AUDEX_VLLM_CFG_TTS_TARGET_SEGMENTS", "bad")
    assert _effective_cfg_tts_target_segments(8) == 4


def test_interleaved_tail_batching_defaults_off_and_can_be_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL", raising=False)
    assert _interleaved_tail_batching_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL", "0")
    assert _interleaved_tail_batching_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL", "off")
    assert _interleaved_tail_batching_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL", "1")
    assert _interleaved_tail_batching_enabled() is True


def test_vllm_playback_prebuffer_keeps_no_cfg_short_answers_low_latency() -> None:
    assert _playback_prebuffer_seconds(("Short answer.",), tts_cfg_enabled=False) == 0.8


def test_vllm_playback_prebuffer_uses_adaptive_cfg_short_answer_buffer() -> None:
    assert _playback_prebuffer_seconds(("Short answer.",), tts_cfg_enabled=True) == 2.0


def test_vllm_playback_prebuffer_keeps_short_two_segment_cfg_answer_low_latency() -> (
    None
):
    chunks = (
        "A context manager implements enter and exit.",
        "It guarantees cleanup even if an exception occurs.",
    )
    assert _playback_prebuffer_seconds(chunks, tts_cfg_enabled=True) == 2.0


def test_vllm_playback_prebuffer_preserves_cfg_long_answer_guard() -> None:
    chunks = (
        "This is a longer answer segment that needs enough buffered speech.",
        "This is another longer answer segment that keeps CFG playback fed.",
        "This is a third segment used to exercise the multi-chunk guard.",
    )
    assert _playback_prebuffer_seconds(chunks, tts_cfg_enabled=True) == 4.0


def test_interleaved_initial_ready_batching_defaults_off_and_can_be_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "AUDEX_VLLM_INTERLEAVED_TTS_BATCH_INITIAL_READY",
        raising=False,
    )
    assert _interleaved_initial_ready_batching_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_INITIAL_READY", "0")
    assert _interleaved_initial_ready_batching_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_INITIAL_READY", "off")
    assert _interleaved_initial_ready_batching_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_INITIAL_READY", "1")
    assert _interleaved_initial_ready_batching_enabled() is True


def test_retain_tts_artifacts_defaults_off_and_can_be_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUDEX_VLLM_RETAIN_TTS_ARTIFACTS", raising=False)
    assert _retain_tts_artifacts_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_RETAIN_TTS_ARTIFACTS", "0")
    assert _retain_tts_artifacts_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_RETAIN_TTS_ARTIFACTS", "off")
    assert _retain_tts_artifacts_enabled() is False

    monkeypatch.setenv("AUDEX_VLLM_RETAIN_TTS_ARTIFACTS", "1")
    assert _retain_tts_artifacts_enabled() is True


def test_vllm_sts_session_streams_tts_when_async_runtime_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime()
    session = make_session(tmp_path, runtime, async_runtime=FakeAsyncRuntime())
    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config
            self.chunk_frames = chunk_frames
            self.pushed: list[tuple[tuple[int, ...], ...]] = []
            self.flush_count = 0
            decoder_sessions.append(self)

        def push(self, frames):
            self.pushed.append(tuple(tuple(frame) for frame in frames))
            if len(self.pushed) == 1:
                return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]
            return []

        def flush(self):
            self.flush_count += 1
            return [(self.config.sample_rate, FakeWaveform((0.0, 0.0)))]

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.generate_speech_output(
        text="Hi back.",
        max_tokens=2400,
        play=False,
        decoder_chunk_frames=2,
    )

    assert result.streaming is True
    assert result.generated_codec_frames == ()
    assert result.reached_end_token is True
    assert result.chunk_wav_paths == ()
    assert runtime.calls == []
    assert len(decoder_sessions) == 1
    decoder_session = decoder_sessions[0]
    assert decoder_session.chunk_frames == 2
    assert decoder_session.pushed == [((0,), (1,))]
    assert decoder_session.flush_count == 1
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["streaming"] is True
    assert run_log["vllm_token_streaming"] is True
    assert run_log["decoder_streaming"] is True
    assert run_log["decoder_after_token_stream"] is False
    assert run_log["tts_cfg_enabled"] is False
    assert run_log["decoded_chunk_count"] == 2
    assert run_log["chunk_wav_paths"] == []
    assert run_log["heavy_tts_artifacts_retained"] is False
    assert run_log["generated_token_id_count"] == 3
    assert run_log["generated_codec_frame_count"] == 2
    assert run_log["generated_codec_frames"] == []
    assert run_log["audio_duration_seconds"] == 0.0
    assert run_log["audio_realtime_ratio"] >= 0
    assert run_log["codec_frames_per_second"] > 0
    assert run_log["stream_event_count"] == 3
    assert run_log["first_token_event_seconds"] == 0.1
    assert run_log["last_token_event_seconds"] == 0.3
    assert run_log["first_codec_frame_seconds"] == 0.1
    assert run_log["last_codec_frame_seconds"] == 0.2
    assert run_log["first_codec_frame_wall_seconds"] is not None
    assert run_log["last_codec_frame_wall_seconds"] is not None
    assert run_log["first_decoder_push_started_seconds"] is not None
    assert run_log["first_decoder_push_finished_seconds"] is not None
    assert run_log["first_decoder_push_frame_count"] == 2
    assert run_log["first_decoder_wait_after_codec_seconds"] >= 0
    assert run_log["stream_finished_seconds"] is not None
    assert run_log["playback_close_seconds"] is None
    assert run_log["pcm_pack_seconds"] >= 0
    assert run_log["player_enqueue_seconds"] == 0
    assert run_log["wav_write_seconds"] >= 0
    assert run_log["decoder_push_seconds"] >= 0
    assert run_log["decoder_flush_seconds"] >= 0
    assert run_log["decoder_reset_seconds"] >= 0
    assert run_log["mlx_clear_cache_count"] == 2
    assert run_log["mlx_clear_cache_seconds"] >= 0
    assert run_log["tts_segment_hit_max_tokens"] == {"0": False}


def test_vllm_sts_session_can_skip_speech_decoder_for_timing_bisect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_session(tmp_path, FakeRuntime(), async_runtime=FakeAsyncRuntime())
    monkeypatch.setenv("AUDEX_VLLM_SKIP_SPEECH_DECODER", "1")
    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.pushed: list[tuple[tuple[int, ...], ...]] = []
            self.flush_count = 0
            self.reset_count = 0
            decoder_sessions.append(self)

        def push(self, frames):
            self.pushed.append(tuple(tuple(frame) for frame in frames))
            raise AssertionError("decoder push should be skipped")

        def flush(self):
            self.flush_count += 1
            raise AssertionError("decoder flush should be skipped")

        def reset(self):
            self.reset_count += 1
            raise AssertionError("decoder reset should be skipped")

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.generate_speech_output(
        text="Hi back.",
        max_tokens=2400,
        play=True,
        decoder_chunk_frames=2,
    )

    assert result.generated_codec_frames == ()
    assert result.waveform_shape == (0,)
    assert result.first_audio_ready_seconds is None
    assert len(decoder_sessions) == 1
    decoder_session = decoder_sessions[0]
    assert decoder_session.pushed == []
    assert decoder_session.flush_count == 0
    assert decoder_session.reset_count == 0
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["speech_decoder_skipped"] is True
    assert run_log["playback_transport"] is None
    assert run_log["generated_codec_frame_count"] == 2
    assert run_log["decoded_chunk_count"] == 0
    assert run_log["waveform_shape"] == [0]
    assert run_log["codec_frames_per_second"] > 0
    assert run_log["decoder_push_seconds"] == 0
    assert run_log["decoder_flush_seconds"] == 0
    assert run_log["decoder_reset_seconds"] == 0


def test_run_vllm_tts_text_probe_uses_product_session_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    result = SimpleNamespace(run_log_path=tmp_path / "tts.json")

    class FakeSession:
        def __init__(self, **kwargs) -> None:
            calls.append(("init", kwargs))

        def _speech_max_tokens_for_text(self, text: str) -> int:
            calls.append(("budget-text", text))
            return 777

        def generate_speech_output(self, **kwargs):
            calls.append(("generate", kwargs))
            return result

        def shutdown(self) -> None:
            calls.append(("shutdown", None))

    monkeypatch.setattr(
        "audex_mac.vllm_commands.VllmSpeechToSpeechSession", FakeSession
    )

    actual = run_vllm_tts_text_probe(
        full_model_path=tmp_path / "model",
        decoder_path=tmp_path / "decoder",
        text="fixed text",
        selected_model_repo="repo",
        output_dir=tmp_path / "runs",
        play=False,
        speech_max_tokens=123,
    )

    assert actual is result
    assert calls == [
        (
            "init",
            {
                "full_model_path": tmp_path / "model",
                "decoder_path": tmp_path / "decoder",
                "selected_model_repo": "repo",
                "output_dir": tmp_path / "runs",
                "speech_max_tokens": 123,
            },
        ),
        ("budget-text", "fixed text"),
        (
            "generate",
            {
                "text": "fixed text",
                "max_tokens": 777,
                "play": False,
                "artifact_prefix": "tts-text-probe-vllm",
            },
        ),
        ("shutdown", None),
    ]


def test_run_vllm_tts_quality_probe_reuses_one_session_and_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "first_passage",
                        "category": "prosody",
                        "text": " ".join(f"first{index}" for index in range(45)),
                        "required_terms": ["first17"],
                    },
                    {
                        "id": "second_passage",
                        "category": "precision",
                        "text": " ".join(f"second{index}" for index in range(48)),
                        "required_terms": ["second23"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, object]] = []

    class FakeSession:
        def __init__(self, **kwargs) -> None:
            calls.append(("init", kwargs))

        def _speech_max_tokens_for_text(self, text: str) -> int:
            return len(text)

        def generate_speech_output(self, **kwargs):
            calls.append(("generate", kwargs))
            prefix = kwargs["artifact_prefix"]
            wav_path = tmp_path / f"{prefix}.wav"
            run_log_path = tmp_path / f"{prefix}.json"
            wav_path.write_bytes(b"RIFF")
            run_log_path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(wav_path=wav_path, run_log_path=run_log_path)

        def shutdown(self) -> None:
            calls.append(("shutdown", None))

    monkeypatch.setattr(
        "audex_mac.vllm_commands.VllmSpeechToSpeechSession", FakeSession
    )
    manifest_path = run_vllm_tts_quality_probe(
        full_model_path=tmp_path / "model",
        decoder_path=tmp_path / "decoder",
        corpus_path=corpus_path,
        recipe=tts_quality_recipe("nvidia-tts-cfg", seed=20260709),
        selected_model_repo="repo",
        output_dir=tmp_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["recipe_id"] == "nvidia-tts-cfg"
    assert manifest["seed"] == 20260709
    assert manifest["controlled_segments_per_case"] == 1
    assert manifest["compact_window_decode"] is True
    assert [item["case_id"] for item in manifest["samples"]] == [
        "first_passage",
        "second_passage",
    ]
    assert [call[0] for call in calls] == ["init", "generate", "generate", "shutdown"]
    assert calls[1][1]["artifact_prefix"] == (
        "tts-quality-nvidia-tts-cfg-first_passage"
    )
    assert calls[1][1]["play"] is False
    assert calls[1][1]["tts_segments"] == (calls[1][1]["text"],)


@pytest.mark.parametrize(
    ("cfg_enabled", "expected_call"),
    [(False, "tts-stream"), (True, "tts-cfg-stream")],
)
def test_quality_segments_bypass_product_chunkers_for_both_tts_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg_enabled: bool,
    expected_call: str,
) -> None:
    runtime = FakeAsyncRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=runtime)
    session.tts_cfg_enabled = cfg_enabled

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return [(self.config.sample_rate, FakeWaveform((0.0, 0.0)))]

        def reset(self):
            return None

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )
    controlled_passage = "One exact long passage. It must remain a single segment."

    try:
        result = session.generate_speech_output(
            text="Text that the product chunker would otherwise inspect.",
            max_tokens=2400,
            play=False,
            tts_segments=(controlled_passage,),
        )
    finally:
        session.shutdown()

    tts_calls = [call for call in runtime.calls if call[0].startswith("tts")]
    assert tts_calls == [(expected_call, (controlled_passage, 2400))]
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["tts_segment_texts"] == {"0": controlled_passage}


def test_interleaved_tts_batches_all_ready_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL", "1")
    monkeypatch.setenv("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_INITIAL_READY", "1")

    class MultiChunkTextRuntime(FakeAsyncRuntime):
        async def stream_text_response_from_messages(
            self,
            messages: list[dict[str, str]],
            *,
            enable_reasoning: bool,
            max_tokens: int | None = None,
            **_state_kwargs,
        ):
            self.calls.append(("text-stream", (messages, enable_reasoning, max_tokens)))
            yield VllmStreamDelta(
                text="One. Two. Three. Four. Five. Six.",
                token_ids=(2,),
                new_token_ids=(2,),
                elapsed_seconds=0.2,
                finished=True,
                finish_reason="stop",
                request_debug_name="text",
                request_id="fake-text",
            )

    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    runtime = MultiChunkTextRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=runtime)
    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config
            self.pushed: list[tuple[tuple[int, ...], ...]] = []
            self.flush_count = 0
            self.reset_count = 0
            decoder_sessions.append(self)

        def push(self, frames):
            self.pushed.append(tuple(tuple(frame) for frame in frames))
            return []

        def flush(self):
            self.flush_count += 1
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def reset(self):
            self.reset_count += 1

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    try:
        result = session.run_turn_from_wav(input_wav_path=input_wav, play=False)
    finally:
        session.shutdown()

    assert result.response_text == "One. Two. Three. Four. Five. Six."
    assert [call[0] for call in runtime.calls] == [
        "asr",
        "text-stream",
        "tts-segmented-stream",
    ]
    assert runtime.calls[2][1][0] == ("One. Two. Three.", "Four. Five. Six.")
    decoder_session = decoder_sessions[0]
    assert decoder_session.flush_count == 2
    assert decoder_session.reset_count == 2
    run_log = json.loads(Path(result.run_log_path).read_text(encoding="utf-8"))
    speech_log = json.loads(
        Path(run_log["speech_output_run_log_path"]).read_text(encoding="utf-8")
    )
    assert speech_log["tts_interleaved_all_ready_batched"] is True
    assert speech_log["tts_interleaved_tail_batched"] is False
    assert speech_log["tts_segment_texts"] == {
        "0": "One. Two. Three.",
        "1": "Four. Five. Six.",
    }
    assert speech_log["tts_observed_segments"] == 2


@pytest.mark.parametrize(
    ("cfg_enabled", "first_tts_call", "tail_tts_call"),
    [
        (False, "tts-stream", "tts-segmented-stream"),
        (True, "tts-cfg-stream", "tts-cfg-segmented-stream"),
    ],
)
def test_interleaved_tts_batches_tail_chunks_after_first_spoken_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg_enabled: bool,
    first_tts_call: str,
    tail_tts_call: str,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL", "1")

    class DelayedTailTextRuntime(FakeAsyncRuntime):
        async def stream_text_response_from_messages(
            self,
            messages: list[dict[str, str]],
            *,
            enable_reasoning: bool,
            max_tokens: int | None = None,
            **_state_kwargs,
        ):
            self.calls.append(("text-stream", (messages, enable_reasoning, max_tokens)))
            yield VllmStreamDelta(
                text="One. Two. Three. Four incomplete",
                token_ids=(2,),
                new_token_ids=(2,),
                elapsed_seconds=0.2,
                finished=False,
                finish_reason=None,
                request_debug_name="text",
                request_id="fake-text",
            )
            await asyncio.sleep(0.1)
            yield VllmStreamDelta(
                text="One. Two. Three. Four. Five. Six.",
                token_ids=(2, 3),
                new_token_ids=(3,),
                elapsed_seconds=0.4,
                finished=True,
                finish_reason="stop",
                request_debug_name="text",
                request_id="fake-text",
            )

    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    runtime = DelayedTailTextRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=runtime)
    session.tts_cfg_enabled = cfg_enabled
    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config
            self.flush_count = 0
            self.reset_count = 0
            decoder_sessions.append(self)

        def push(self, frames):
            return []

        def flush(self):
            self.flush_count += 1
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def reset(self):
            self.reset_count += 1

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    try:
        result = session.run_turn_from_wav(input_wav_path=input_wav, play=False)
    finally:
        session.shutdown()

    assert result.response_text == "One. Two. Three. Four. Five. Six."
    assert [call[0] for call in runtime.calls] == [
        "asr",
        "text-stream",
        first_tts_call,
        tail_tts_call,
    ]
    assert runtime.calls[2][1][0] == "One. Two. Three."
    assert runtime.calls[3][1][0] == ("Four. Five. Six.",)
    decoder_session = decoder_sessions[0]
    assert decoder_session.flush_count == 2
    assert decoder_session.reset_count == 2
    run_log = json.loads(Path(result.run_log_path).read_text(encoding="utf-8"))
    speech_log = json.loads(
        Path(run_log["speech_output_run_log_path"]).read_text(encoding="utf-8")
    )
    assert speech_log["tts_interleaved_all_ready_batched"] is False
    assert speech_log["tts_interleaved_tail_batched"] is True
    assert speech_log["tts_segment_texts"] == {
        "0": "One. Two. Three.",
        "1": "Four. Five. Six.",
    }
    assert speech_log["tts_observed_segments"] == 2


def test_interleaved_tts_batches_initial_ready_chunks_before_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL", "1")
    monkeypatch.setenv("AUDEX_VLLM_INTERLEAVED_TTS_BATCH_INITIAL_READY", "1")
    first_sentence = (
        "Migrating to Rust could give you strong safety guarantees around memory "
        "and concurrency, which may reduce bugs in a protobuf-heavy, highly "
        "parallel service."
    )
    final_text = first_sentence + " Tail sentence."

    class InitialMultiChunkTextRuntime(FakeAsyncRuntime):
        async def stream_text_response_from_messages(
            self,
            messages: list[dict[str, str]],
            *,
            enable_reasoning: bool,
            max_tokens: int | None = None,
            **_state_kwargs,
        ):
            self.calls.append(("text-stream", (messages, enable_reasoning, max_tokens)))
            yield VllmStreamDelta(
                text=first_sentence,
                token_ids=(2,),
                new_token_ids=(2,),
                elapsed_seconds=0.2,
                finished=False,
                finish_reason=None,
                request_debug_name="text",
                request_id="fake-text",
            )
            await asyncio.sleep(0.1)
            yield VllmStreamDelta(
                text=final_text,
                token_ids=(2, 3),
                new_token_ids=(3,),
                elapsed_seconds=0.4,
                finished=True,
                finish_reason="stop",
                request_debug_name="text",
                request_id="fake-text",
            )

    input_wav = tmp_path / "input.wav"
    write_pcm16_wav(input_wav, [0.0, 0.25, -0.25], sample_rate=16_000)
    runtime = InitialMultiChunkTextRuntime()
    session = make_session(tmp_path, runtime=None, async_runtime=runtime)

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            return []

        def flush(self):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def reset(self):
            pass

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    try:
        result = session.run_turn_from_wav(input_wav_path=input_wav, play=False)
    finally:
        session.shutdown()

    assert result.response_text == final_text
    assert [call[0] for call in runtime.calls] == [
        "asr",
        "text-stream",
        "tts-segmented-stream",
        "tts-segmented-stream",
    ]
    assert runtime.calls[2][1][0] == (
        "Migrating to Rust could give you strong safety guarantees around memory "
        "and concurrency,",
        "which may reduce bugs in a protobuf-heavy, highly parallel service.",
    )
    assert runtime.calls[3][1][0] == ("Tail sentence.",)
    speech_log = json.loads(Path(result.run_log_path).read_text(encoding="utf-8"))
    speech_log = json.loads(
        Path(speech_log["speech_output_run_log_path"]).read_text(encoding="utf-8")
    )
    assert speech_log["tts_interleaved_all_ready_batched"] is False
    assert speech_log["tts_interleaved_initial_ready_batched"] is True
    assert speech_log["tts_interleaved_tail_batched"] is True
    assert speech_log["tts_observed_segments"] == 3


def test_vllm_sts_session_primes_decoder_with_lookahead_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ThreeFrameAsyncRuntime(FakeAsyncRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.token_map.speech_codec[104] = 2

        async def stream_tts_codec_frames(
            self,
            text: str,
            *,
            max_tokens: int | None = None,
        ):
            self.calls.append(("tts-stream", (text, max_tokens)))
            for index, _token_id in enumerate((102, 103, 104), start=1):
                yield VllmTtsCodecStreamEvent(
                    generated_token_ids=(102, 103, 104)[:index],
                    new_codec_frames=(index - 1,),
                    reached_end_token=False,
                    finished=False,
                    elapsed_seconds=0.1 * index,
                )
            yield VllmTtsCodecStreamEvent(
                generated_token_ids=(102, 103, 104, 101),
                new_codec_frames=(),
                reached_end_token=True,
                finished=True,
                elapsed_seconds=0.4,
            )

    session = make_session(
        tmp_path,
        runtime=None,
        async_runtime=ThreeFrameAsyncRuntime(),
    )
    session.decoder_config.lookahead_steps = 1
    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config
            self.chunk_frames = chunk_frames
            self.pushed: list[tuple[tuple[int, ...], ...]] = []
            self.flush_count = 0
            decoder_sessions.append(self)

        def push(self, frames):
            self.pushed.append(tuple(tuple(frame) for frame in frames))
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            self.flush_count += 1
            return []

        def reset(self):
            pass

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.generate_speech_output(
        text="Hi back.",
        max_tokens=2400,
        play=False,
        decoder_chunk_frames=2,
    )

    assert result.generated_codec_frames == ()
    decoder_session = decoder_sessions[0]
    assert decoder_session.pushed == [((0,), (1,), (2,))]
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["first_decoder_push_frame_count"] == 3


def test_vllm_sts_session_uses_larger_steady_decoder_chunks_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ManyFrameAsyncRuntime(FakeAsyncRuntime):
        async def stream_tts_codec_frames(
            self,
            text: str,
            *,
            max_tokens: int | None = None,
        ):
            self.calls.append(("tts-stream", (text, max_tokens)))
            for index in range(67):
                yield VllmTtsCodecStreamEvent(
                    generated_token_ids=(),
                    new_codec_frames=(index,),
                    reached_end_token=False,
                    finished=False,
                    elapsed_seconds=0.01 * (index + 1),
                )
            yield VllmTtsCodecStreamEvent(
                generated_token_ids=(102, 101),
                new_codec_frames=(),
                reached_end_token=True,
                finished=True,
                elapsed_seconds=0.5,
            )

    session = make_session(
        tmp_path,
        runtime=None,
        async_runtime=ManyFrameAsyncRuntime(),
    )
    session.decoder_config.lookahead_steps = 1
    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config
            self.chunk_frames = chunk_frames
            self.pushed: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
            decoder_sessions.append(self)

        def push(self, frames):
            self.pushed.append(
                (self.chunk_frames, tuple(tuple(frame) for frame in frames))
            )
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return []

        def reset(self):
            pass

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.generate_speech_output(
        text="Hi back.",
        max_tokens=2400,
        play=False,
    )

    assert result.generated_codec_frames == ()
    decoder_session = decoder_sessions[0]
    pushed_chunk_frames = [
        chunk_frames for chunk_frames, _frames in decoder_session.pushed
    ]
    pushed_frame_counts = [
        len(frames) for _chunk_frames, frames in decoder_session.pushed
    ]
    assert pushed_chunk_frames[0] == DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES
    assert all(
        chunk_frames == DEFAULT_VLLM_STREAM_DECODER_STEADY_CHUNK_FRAMES
        for chunk_frames in pushed_chunk_frames[1:]
    )
    assert pushed_frame_counts[0] == DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES + 1
    assert all(
        frame_count == DEFAULT_VLLM_STREAM_DECODER_STEADY_CHUNK_FRAMES
        for frame_count in pushed_frame_counts[1:-1]
    )
    assert pushed_frame_counts[-1] == 1
    assert sum(pushed_frame_counts) == 67
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["decoder_chunk_frames"] == DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES
    assert (
        run_log["decoder_steady_chunk_frames"]
        == DEFAULT_VLLM_STREAM_DECODER_STEADY_CHUNK_FRAMES
    )


def test_vllm_sts_session_streams_playback_chunks_when_play_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_session(tmp_path, runtime=None, async_runtime=FakeAsyncRuntime())
    enqueued: list[bytes] = []
    latencies: list[str] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            if frames == ((0,),):
                return []
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return [(self.config.sample_rate, FakeWaveform((0.0, 0.0)))]

    class FakeContinuousPlayer:
        first_playback_started_seconds = 0.25

        def __init__(
            self,
            *,
            started_at: float,
            sample_rate: int,
            output_sample_rate: int,
            output_blocksize: int,
            prebuffer_seconds: float,
            latency: str,
        ) -> None:
            self.started_at = started_at
            self.sample_rate = sample_rate
            self.prebuffer_seconds = prebuffer_seconds
            latencies.append(latency)

        def start(self) -> None:
            pass

        def enqueue_pcm(self, pcm: bytes) -> None:
            enqueued.append(pcm)

        def close(self) -> None:
            pass

        def diagnostics(self) -> dict[str, object]:
            return {
                "device_underflow_count": 0,
                "queue_underrun_count": 0,
                "queue_overrun_count": 0,
                "chunks_written": len(enqueued),
            }

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli._ContinuousPcmPlayer",
        FakeContinuousPlayer,
    )

    result = session.generate_speech_output(
        text="Hi back.",
        max_tokens=2400,
        play=True,
        decoder_chunk_frames=2,
    )

    assert result.first_playback_started_seconds == 0.25
    assert latencies == ["low"]
    assert enqueued == [
        float_samples_to_pcm16_bytes((0.1, -0.1)),
        float_samples_to_pcm16_bytes((0.0, 0.0)),
    ]
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["playback_transport"] == "sounddevice_raw_output_stream"
    assert run_log["playback_prebuffer_seconds"] == 0.8
    assert run_log["first_playback_started_seconds"] == 0.25
    assert run_log["first_playback_after_audio_seconds"] is not None
    assert run_log["playback_diagnostics"]["chunks_written"] == 2
    assert run_log["player_enqueue_seconds"] >= 0


def test_vllm_sts_session_streams_pcm_to_external_sink_without_device_playback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_session(tmp_path, runtime=None, async_runtime=FakeAsyncRuntime())
    emitted: list[tuple[int, bytes]] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            if frames == ((0,),):
                return []
            return [(self.config.sample_rate, FakeWaveform((0.25, -0.25)))]

        def flush(self):
            return [(self.config.sample_rate, FakeWaveform((0.0, 0.0)))]

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    result = session.generate_speech_output(
        text="Stream this response.",
        max_tokens=2400,
        play=False,
        decoder_chunk_frames=2,
        pcm_chunk_sink=lambda sample_rate, pcm: emitted.append((sample_rate, pcm)),
    )

    assert emitted == [
        (
            session.decoder_config.sample_rate,
            float_samples_to_pcm16_bytes((0.25, -0.25)),
        ),
        (
            session.decoder_config.sample_rate,
            float_samples_to_pcm16_bytes((0.0, 0.0)),
        ),
    ]
    assert result.wav_path.is_file()


def test_vllm_sts_interleaved_playback_requests_buffered_device_latency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_session(tmp_path, runtime=None, async_runtime=FakeAsyncRuntime())
    prebuffers: list[float] = []
    latencies: list[str | float] = []
    gates: list[object] = []
    first_playback_callbacks: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config

        def push(self, frames):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def flush(self):
            return []

        def reset(self) -> None:
            pass

    class FakeContinuousPlayer:
        first_playback_started_seconds = 0.25

        def __init__(
            self,
            *,
            started_at: float,
            sample_rate: int,
            output_sample_rate: int,
            output_blocksize: int,
            prebuffer_seconds: float,
            latency: str | float,
            start_gate=None,
            first_playback_callback=None,
        ) -> None:
            prebuffers.append(prebuffer_seconds)
            latencies.append(latency)
            gates.append(start_gate)
            first_playback_callbacks.append(first_playback_callback)

        def start(self) -> None:
            pass

        def enqueue_pcm(self, pcm: bytes) -> None:
            pass

        def close(self) -> None:
            pass

        def diagnostics(self) -> dict[str, object]:
            return {
                "device_underflow_count": 0,
                "queue_underrun_count": 0,
                "queue_overrun_count": 0,
                "chunks_written": 1,
            }

    async def chunks():
        yield "A streamed answer chunk."

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli._ContinuousPcmPlayer",
        FakeContinuousPlayer,
    )
    playback_gate = vllm_sts_cli._PlaybackStartGate()
    playback_gate.release()

    result = session._run_async(
        session.generate_speech_output_streaming_from_async_runtime(
            text="fallback",
            max_tokens=2400,
            play=True,
            tts_chunk_source=chunks(),
            text_to_tts_interleaved=True,
            playback_start_gate=playback_gate,
        )
    )

    assert result.first_playback_started_seconds == 0.25
    assert prebuffers == [DEFAULT_VLLM_INTERLEAVED_PLAYBACK_PREBUFFER_SECONDS]
    assert latencies == [DEFAULT_VLLM_INTERLEAVED_PLAYBACK_LATENCY]
    assert gates == [playback_gate]
    assert callable(first_playback_callbacks[0])
    first_playback_callbacks[0](playback_gate.released_at + 0.2, None)
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert (
        run_log["playback_prebuffer_seconds"]
        == DEFAULT_VLLM_INTERLEAVED_PLAYBACK_PREBUFFER_SECONDS
    )
    assert run_log["playback_start_gated"] is True
    assert run_log["playback_gate_released"] is True
    milestone_path = Path(run_log["live_milestone_path"])
    milestone = json.loads(milestone_path.read_text(encoding="utf-8"))
    assert milestone["source"] == "model_response_tts"
    assert milestone["cfg_enabled"] is False
    assert milestone["text_to_tts_interleaved"] is True
    assert milestone["submit_to_first_estimated_audible_seconds"] == 0.2


def test_vllm_sts_session_reuses_one_event_loop_for_async_runtime(
    tmp_path: Path,
) -> None:
    session = make_session(tmp_path, runtime=None, async_runtime=FakeAsyncRuntime())

    async def running_loop_id() -> int:
        return id(asyncio.get_running_loop())

    try:
        first_loop_id = session._run_async(running_loop_id())
        second_loop_id = session._run_async(running_loop_id())
    finally:
        session.shutdown()

    assert first_loop_id == second_loop_id
    assert session._async_loop is not None
    assert session._async_loop.is_closed()


def test_vllm_sts_session_shutdown_cancels_pending_async_tasks(
    tmp_path: Path,
) -> None:
    session = make_session(tmp_path, runtime=None, async_runtime=FakeAsyncRuntime())
    cancelled = {"value": False}

    async def pending_task() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled["value"] = True
            raise

    task = session._async_loop.create_task(pending_task())
    session._async_loop.run_until_complete(asyncio.sleep(0))

    session.shutdown()

    assert task.cancelled()
    assert cancelled["value"] is True
    assert session._async_loop is not None
    assert session._async_loop.is_closed()


def test_vllm_sts_default_tts_ignores_segmented_cfg_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_session(
        tmp_path,
        runtime=None,
        async_runtime=FakeAsyncRuntime(),
    )
    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config
            self.pushed: list[tuple[tuple[int, ...], ...]] = []
            self.flush_count = 0
            self.reset_count = 0
            decoder_sessions.append(self)

        def push(self, frames):
            self.pushed.append(tuple(tuple(frame) for frame in frames))
            return []

        def flush(self):
            self.flush_count += 1
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def reset(self):
            self.reset_count += 1

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    try:
        result = session.generate_speech_output(
            text="First sentence. Second sentence.",
            max_tokens=2400,
            play=False,
            tts_target_segments=2,
        )
    finally:
        session.shutdown()

    assert result.generated_codec_frames == ()
    assert len(decoder_sessions) == 1
    decoder_session = decoder_sessions[0]
    assert decoder_session.pushed == [((0,),), ((1,),)]
    assert decoder_session.flush_count == 1
    assert decoder_session.reset_count == 1
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["tts_target_segments"] == 2
    assert run_log["tts_observed_segments"] == 1
    assert run_log["tts_cfg_enabled"] is False
    assert run_log["decoder_streaming"] is True
    assert run_log["decoder_after_token_stream"] is False
    assert run_log["waveform_shape"][0] == 2


def test_vllm_sts_can_enable_cfg_tts_for_static_speech_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_runtime = FakeAsyncRuntime()
    session = make_session(
        tmp_path,
        runtime=None,
        async_runtime=async_runtime,
    )
    monkeypatch.setenv("AUDEX_VLLM_TTS_CFG", "1")
    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config
            self.pushed: list[tuple[tuple[int, ...], ...]] = []
            self.flush_count = 0
            self.reset_count = 0
            decoder_sessions.append(self)

        def push(self, frames):
            self.pushed.append(tuple(tuple(frame) for frame in frames))
            return []

        def flush(self):
            self.flush_count += 1
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def reset(self):
            self.reset_count += 1

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    try:
        result = session.generate_speech_output(
            text="CFG speech.",
            max_tokens=2400,
            play=False,
            tts_target_segments=1,
        )
    finally:
        session.shutdown()

    assert result.generated_codec_frames == ()
    assert async_runtime.calls == [("tts-cfg-stream", ("CFG speech.", 2400))]
    assert len(decoder_sessions) == 1
    decoder_session = decoder_sessions[0]
    assert decoder_session.pushed == [((1,),)]
    assert decoder_session.flush_count == 1
    assert decoder_session.reset_count == 1
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["tts_cfg_enabled"] is True
    assert run_log["text_to_tts_interleaved"] is False
    assert run_log["tts_segment_codec_frame_counts"] == {"0": 1}


def test_vllm_sts_cfg_tts_batches_static_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_runtime = FakeAsyncRuntime()
    session = make_session(
        tmp_path,
        runtime=None,
        async_runtime=async_runtime,
    )
    monkeypatch.setenv("AUDEX_VLLM_TTS_CFG", "1")
    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config
            self.pushed: list[tuple[tuple[int, ...], ...]] = []
            self.flush_count = 0
            self.reset_count = 0
            decoder_sessions.append(self)

        def push(self, frames):
            self.pushed.append(tuple(tuple(frame) for frame in frames))
            return []

        def flush(self):
            self.flush_count += 1
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def reset(self):
            self.reset_count += 1

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    try:
        result = session.generate_speech_output(
            text="One. Two. Three. Four.",
            max_tokens=2400,
            play=False,
            decoder_chunk_frames=1,
        )
    finally:
        session.shutdown()

    assert [call[0] for call in async_runtime.calls] == ["tts-cfg-segmented-stream"]
    assert async_runtime.calls[0][1] == (
        ("One.", "Two.", "Three.", "Four."),
        (512, 512, 512, 512),
        False,
    )
    assert result.generated_codec_frames == ()
    decoder_session = decoder_sessions[0]
    assert decoder_session.pushed == [((10,),), ((11,),), ((12,),), ((13,),)]
    assert decoder_session.flush_count == 4
    assert decoder_session.reset_count == 4
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["tts_cfg_enabled"] is True
    assert run_log["tts_cfg_prime_first_segment"] is False
    assert run_log["tts_concurrent_segments"] is True
    assert run_log["tts_cfg_min_tail_chars"] == 40
    assert run_log["text_to_tts_interleaved"] is False
    assert run_log["tts_observed_segments"] == 4
    assert run_log["tts_segment_codec_frame_counts"] == {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 1,
    }
    assert run_log["tts_segment_finished"] == {
        "0": True,
        "1": True,
        "2": True,
        "3": True,
    }


def test_vllm_sts_cfg_tts_target_segments_env_changes_static_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_runtime = FakeAsyncRuntime()
    session = make_session(
        tmp_path,
        runtime=None,
        async_runtime=async_runtime,
    )
    monkeypatch.setenv("AUDEX_VLLM_TTS_CFG", "1")
    monkeypatch.setenv("AUDEX_VLLM_CFG_TTS_TARGET_SEGMENTS", "3")
    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config
            decoder_sessions.append(self)

        def push(self, frames):
            return []

        def flush(self):
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def reset(self):
            return None

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    try:
        result = session.generate_speech_output(
            text="One. Two. Three. Four.",
            max_tokens=2400,
            play=False,
            decoder_chunk_frames=1,
        )
    finally:
        session.shutdown()

    assert [call[0] for call in async_runtime.calls] == ["tts-cfg-segmented-stream"]
    assert async_runtime.calls[0][1] == (
        ("One. Two.", "Three.", "Four."),
        (512, 512, 512),
        False,
    )
    assert len(decoder_sessions) == 1
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["tts_requested_target_segments"] == 8
    assert run_log["tts_target_segments"] == 3
    assert run_log["tts_observed_segments"] == 3


def test_prepare_text_for_tts_makes_inline_python_names_speakable() -> None:
    prepared = prepare_text_for_tts(
        "Use `__enter__`, `__exit__`, and `with`; avoid explicit try/final."
    )

    assert prepared == "Use enter, exit, and with; avoid explicit try finally."


def test_prepare_text_for_tts_preserves_newline_chunk_boundaries() -> None:
    prepared = prepare_text_for_tts("First sentence.\n\nSecond sentence.")

    assert prepared == "First sentence.\n\nSecond sentence."
    assert split_spoken_tts_chunks(prepared) == (
        "First sentence.",
        "Second sentence.",
    )


def test_split_spoken_tts_chunks_prefers_newlines_then_three_sentences() -> None:
    chunks = split_spoken_tts_chunks(
        "One. Two. Three. Four.\n\nFive. Six. Seven. Eight."
    )

    assert chunks == (
        "One. Two. Three.",
        "Four.",
        "Five. Six. Seven.",
        "Eight.",
    )


def test_split_cfg_spoken_tts_chunks_splits_long_sentence_atoms() -> None:
    chunks = split_cfg_spoken_tts_chunks(
        "A context manager is an object that implements enter and exit. "
        "It runs enter on entry, exit on exit, guaranteeing cleanup even if an "
        "exception occurs."
    )

    assert chunks == (
        "A context manager is an object that implements enter and exit.",
        "It runs enter on entry, exit on exit,",
        "guaranteeing cleanup even if an exception occurs.",
    )


def test_split_cfg_spoken_tts_chunks_caps_to_target_segments() -> None:
    text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine."
    chunks = split_cfg_spoken_tts_chunks(
        text,
        target_segments=8,
    )

    assert len(chunks) == 8
    assert " ".join(chunks) == text
    assert any(chunk.count(".") == 2 for chunk in chunks)


def test_split_cfg_spoken_tts_chunks_balances_by_text_length() -> None:
    chunks = split_cfg_spoken_tts_chunks(
        "Alpha one. Bravo two. Charlie three. Delta four. Echo five. Foxtrot six.",
        target_segments=3,
    )

    assert chunks == (
        "Alpha one. Bravo two.",
        "Charlie three. Delta four.",
        "Echo five. Foxtrot six.",
    )


def test_split_cfg_spoken_tts_chunks_merges_underfilled_tail_at_capacity() -> None:
    chunks = split_cfg_spoken_tts_chunks(
        "A context manager opens a resource before the block starts. "
        "It closes the resource when the block exits. "
        "Cleanup runs even if an exception is raised. "
        "The exit hook receives exception details. "
        "Files and locks are common examples. "
        "The with statement keeps cleanup local. "
        "This avoids a manual try finally. "
        "Useful.",
        target_segments=8,
    )

    assert len(chunks) == 7
    assert chunks == (
        "A context manager opens a resource before the block starts.",
        "It closes the resource when the block exits.",
        "Cleanup runs even if an exception is raised.",
        "The exit hook receives exception details.",
        "Files and locks are common examples.",
        "The with statement keeps cleanup local.",
        "This avoids a manual try finally. Useful.",
    )


def test_split_cfg_spoken_tts_chunks_balances_nine_atoms_into_eight_groups() -> None:
    chunks = split_cfg_spoken_tts_chunks(
        "One. Two. Three. Four. Five. Six. Seven. Eight. Nine.",
        target_segments=8,
    )

    assert len(chunks) == 8
    assert chunks == (
        "One. Two.",
        "Three.",
        "Four.",
        "Five.",
        "Six.",
        "Seven.",
        "Eight.",
        "Nine.",
    )


def test_split_cfg_spoken_tts_chunks_partitions_split_atoms_to_target() -> None:
    chunks = split_cfg_spoken_tts_chunks(
        "A context manager is an object that implements enter and exit. "
        "It runs enter on entry, exit on exit, guaranteeing cleanup even if an "
        "exception occurs. "
        "The exit method receives exception details so it can clean up properly. "
        "Common use cases include files, database connections, and locks. "
        "The exit method can return False to suppress exceptions or True to "
        "re-raise them. "
        "The with statement makes sure cleanup runs automatically. "
        "You don't need a separate try-finally block for cleanup. "
        "This pattern is useful",
        target_segments=8,
    )

    assert chunks == (
        "A context manager is an object that implements enter and exit.",
        "It runs enter on entry, exit on exit,",
        "guaranteeing cleanup even if an exception occurs.",
        "The exit method receives exception details so it can clean up properly.",
        "Common use cases include files, database connections, and locks.",
        "The exit method can return False to suppress exceptions or True to "
        "re-raise them.",
        "The with statement makes sure cleanup runs automatically.",
        "You don't need a separate try-finally block for cleanup. "
        "This pattern is useful",
    )


def test_split_cfg_spoken_tts_chunks_keeps_all_short_chunks() -> None:
    chunks = split_cfg_spoken_tts_chunks(
        "One. Two. Three. Four. Five. Six. Seven. Eight.",
        target_segments=8,
    )

    assert chunks == (
        "One.",
        "Two.",
        "Three.",
        "Four.",
        "Five.",
        "Six.",
        "Seven.",
        "Eight.",
    )


def test_split_spoken_tts_chunks_caps_long_two_sentence_chunks() -> None:
    chunks = split_spoken_tts_chunks(
        "First sentence has a few ordinary words. "
        "Second sentence is deliberately much longer because it describes a "
        "migration from Go to Rust with protobufs, NATS Jetstream, Aurora, "
        "parallelism, AWS library support, team pressure, and a careful "
        "tradeoff between stronger safety guarantees and delivery speed. "
        "Third sentence closes the thought.",
        max_chars_per_chunk=120,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert " ".join(chunks).startswith("First sentence")


def test_split_spoken_tts_chunks_splits_long_unpunctuated_text() -> None:
    text = (
        "hey matt thanks for joining in lets dive into go routines maybe start "
        "with how you launch one with go func in the background any specific "
        "angle you want to explore"
    )

    chunks = split_spoken_tts_chunks(text)

    assert len(chunks) == 2
    assert " ".join(chunks) == text
    assert all(55 <= len(chunk) <= 160 for chunk in chunks)
    assert chunks[0].endswith("go routines")
    assert chunks[1].startswith("maybe start")


def test_split_spoken_tts_chunks_splits_one_overlong_sentence() -> None:
    text = (
        "hey matt thanks for joining in let's dive into go routines maybe start "
        "with how you launch one with go func in the background what's been "
        "interesting for you so far?"
    )

    chunks = split_spoken_tts_chunks(text)

    assert len(chunks) == 2
    assert " ".join(chunks) == text
    assert all(45 <= len(chunk) <= 120 for chunk in chunks)
    assert chunks[0].endswith("go func in")
    assert chunks[1].startswith("the background what's been")


def test_streamed_tts_chunks_waits_for_stable_sentence_boundaries() -> None:
    chunks, emitted = streamed_tts_chunks_from_text(
        "One complete sentence. Two incomplete",
        0,
        final=False,
    )
    assert chunks == ("One complete sentence.",)
    assert emitted == len("One complete sentence. ")

    chunks, emitted = streamed_tts_chunks_from_text(
        "One complete sentence. Two complete sentences. Three complete sentences. "
        "Four incomplete",
        emitted,
        final=False,
    )
    assert chunks == ()

    chunks, emitted = streamed_tts_chunks_from_text(
        "One complete sentence. Two complete sentences. Three complete sentences. "
        "Four incomplete but now finished.",
        emitted,
        final=True,
    )
    assert chunks == (
        "Two complete sentences. Three complete sentences. "
        "Four incomplete but now finished.",
    )
    assert emitted == len(
        "One complete sentence. Two complete sentences. Three complete sentences. "
        "Four incomplete but now finished."
    )


def test_streamed_tts_chunks_emits_first_coherent_sentence_without_waiting_for_120_chars() -> (
    None
):
    first_sentence = "That approach should give us a coherent semantic opening."
    text = f"{first_sentence} The next thought is not finished"

    chunks, emitted = streamed_tts_chunks_from_text(text, 0, final=False)

    assert chunks == (first_sentence,)
    assert emitted == len(first_sentence) + 1


def test_streamed_tts_chunks_emits_stable_opening_clause_before_sentence_end() -> None:
    text = (
        "A python list is a simple, ordered collection of items you can add, "
        "remove, or change"
    )

    chunks, emitted = streamed_tts_chunks_from_text(text, 0, final=False)

    assert chunks == ("A python list is a simple, ordered collection of items.",)
    assert emitted == len("A python list is a simple, ordered collection of items")

    final_chunks, final_emitted = streamed_tts_chunks_from_text(
        f"{text}.",
        emitted,
        final=True,
    )

    assert final_chunks == ("You can add, remove, or change.",)
    assert final_emitted == len(text) + 1


def test_streamed_tts_chunks_splits_before_relative_clause() -> None:
    text = (
        "A Python list is a mutable, ordered collection that holds items indexed "
        "by position"
    )

    chunks, emitted = streamed_tts_chunks_from_text(text, 0, final=False)

    assert chunks == ("A Python list is a mutable, ordered collection.",)
    assert emitted == len("A Python list is a mutable, ordered collection")


def test_streamed_tts_chunks_emits_one_substantial_sentence() -> None:
    text = (
        "Migrating to Rust could give you strong safety guarantees around memory "
        "and concurrency, which may reduce bugs in a protobuf-heavy, highly "
        "parallel service. "
        "The trade-off is the larger learning curve, longer development time, "
        "and less mature AWS SDK support compared with Go. "
        "In practice, you would need to weigh the risk of production bugs."
    )

    chunks, emitted = streamed_tts_chunks_from_text(text, 0, final=False)

    assert chunks == (
        "Migrating to Rust could give you strong safety guarantees around memory "
        "and concurrency,",
        "which may reduce bugs in a protobuf-heavy, highly parallel service.",
    )
    assert emitted == len(
        "Migrating to Rust could give you strong safety guarantees around memory "
        "and concurrency, which may reduce bugs in a protobuf-heavy, highly "
        "parallel service. "
    )


def test_streamed_tts_chunks_splits_initial_ready_sentence_more_aggressively() -> None:
    text = (
        "GogoGOI routes by grouping requests into a single call and by pushing "
        "results straight to the database.\n"
    )

    assert split_spoken_tts_chunks(text) == (
        "GogoGOI routes by grouping requests into a single call and by pushing "
        "results straight to the database.",
    )

    chunks, emitted = streamed_tts_chunks_from_text(text, 0, final=False)

    assert chunks == (
        "GogoGOI routes by grouping requests into a single call",
        "and by pushing results straight to the database.",
    )
    assert emitted == len(text)


def test_streamed_tts_chunks_emits_on_newline_before_two_sentences() -> None:
    chunks, emitted = streamed_tts_chunks_from_text(
        "Short line.\nNext sentence is still streaming",
        0,
        final=False,
    )

    assert chunks == ("Short line.",)
    assert emitted == len("Short line.\n")


def test_streamed_tts_chunks_does_not_repeat_when_scrubbed_text_shrinks() -> None:
    original = (
        "Rust makes concurrency explicit and safe by default. "
        "Go embraces lightweight goroutines as a natural tool for concurrency. "
        "Rust's borrow checker enforces data sharing rules."
    )
    chunks, emitted = streamed_tts_chunks_from_text(original, 0, final=False)

    assert chunks == ("Rust makes concurrency explicit and safe by default.",)
    assert emitted > 0

    shrunk_after_scrub = "Rust makes concurrency explicit and safe by default."
    chunks, emitted = streamed_tts_chunks_from_text(
        shrunk_after_scrub,
        emitted,
        final=False,
    )

    assert chunks == ()
    assert emitted == len(shrunk_after_scrub)


def test_streamed_tts_chunks_emits_long_unpunctuated_text_at_word_boundary() -> None:
    text = (
        "hey matt thanks for joining in lets dive into go routines maybe start "
        "with how you launch one with go func in the background any specific "
        "angle you want to explore while the response is still streaming and "
        "the model has not added ordinary punctuation yet but we still need a "
        "stable spoken chunk"
    )

    chunks, emitted = streamed_tts_chunks_from_text(text, 0, final=False)

    assert chunks
    assert text.startswith(chunks[0])
    assert chunks == (
        "hey matt thanks for joining in lets dive into go routines maybe start "
        "with how you launch one with go func in the background any specific",
    )
    assert emitted == len(chunks[0])


def test_vllm_sts_session_splits_long_no_cfg_tts_into_spoken_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeAsyncRuntime()
    session = make_session(
        tmp_path,
        runtime=None,
        async_runtime=runtime,
    )
    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config
            self.pushed: list[tuple[tuple[int, ...], ...]] = []
            self.flush_count = 0
            self.reset_count = 0
            decoder_sessions.append(self)

        def push(self, frames):
            self.pushed.append(tuple(tuple(frame) for frame in frames))
            return []

        def flush(self):
            self.flush_count += 1
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def reset(self):
            self.reset_count += 1

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    try:
        result = session.generate_speech_output(
            text="One. Two. Three. Four.",
            max_tokens=2400,
            play=False,
            decoder_chunk_frames=2,
        )
    finally:
        session.shutdown()

    tts_calls = [call for call in runtime.calls if call[0] == "tts-stream"]
    assert [call[1][0] for call in tts_calls] == ["One. Two. Three.", "Four."]
    assert [call[1][1] for call in tts_calls] == [512, 512]
    assert result.generated_codec_frames == ()
    decoder_session = decoder_sessions[0]
    assert decoder_session.flush_count == 2
    assert decoder_session.reset_count == 2
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["tts_observed_segments"] == 2
    assert run_log["tts_segment_texts"] == {
        "0": "One. Two. Three.",
        "1": "Four.",
    }
    assert run_log["tts_segment_codec_frame_counts"] == {"0": 2, "1": 2}
    assert run_log["tts_segment_max_tokens"] == {"0": 512, "1": 512}
    assert run_log["tts_segment_wall_seconds"]["0"] >= 0
    assert run_log["tts_segment_wall_seconds"]["1"] >= 0
    assert run_log["mlx_clear_cache_count"] == 3
    assert run_log["tts_segment_hit_max_tokens"] == {"0": False, "1": False}


def test_vllm_sts_session_can_batch_no_cfg_tts_chunks_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_CONCURRENT_TTS_CHUNKS", "1")
    runtime = FakeAsyncRuntime()
    session = make_session(
        tmp_path,
        runtime=None,
        async_runtime=runtime,
    )
    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.config = config
            self.pushed: list[tuple[tuple[int, ...], ...]] = []
            self.flush_count = 0
            self.reset_count = 0
            decoder_sessions.append(self)

        def push(self, frames):
            self.pushed.append(tuple(tuple(frame) for frame in frames))
            return []

        def flush(self):
            self.flush_count += 1
            return [(self.config.sample_rate, FakeWaveform((0.1, -0.1)))]

        def reset(self):
            self.reset_count += 1

    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.AudexSpeechDecoderSession",
        FakeDecoderSession,
    )

    try:
        result = session.generate_speech_output(
            text="One. Two. Three. Four.",
            max_tokens=2400,
            play=False,
            decoder_chunk_frames=1,
        )
    finally:
        session.shutdown()

    assert [call[0] for call in runtime.calls] == ["tts-segmented-stream"]
    assert runtime.calls[0][1] == (
        ("One. Two. Three.", "Four."),
        (512, 512),
    )
    assert result.generated_codec_frames == ()
    decoder_session = decoder_sessions[0]
    assert decoder_session.pushed == [((0,),), ((1,),)]
    assert decoder_session.flush_count == 2
    assert decoder_session.reset_count == 2
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["tts_concurrent_segments"] is True
    assert run_log["tts_observed_segments"] == 2
    assert run_log["tts_segment_codec_frame_counts"] == {"0": 1, "1": 1}
    assert run_log["tts_segment_finished"] == {"0": True, "1": True}


def test_vllm_sts_chunk_budget_scales_with_chunk_text_tokens(
    tmp_path: Path,
) -> None:
    session = make_session(tmp_path, runtime=None, async_runtime=FakeAsyncRuntime())

    try:
        budget = session._speech_max_tokens_for_tts_chunk(
            " ".join(f"word{i}" for i in range(20)),
            utterance_max_tokens=2400,
            chunk_count=2,
        )
    finally:
        session.shutdown()

    assert budget == 20 * 64


def test_vllm_sts_chunk_budget_scales_above_default_for_long_chunks(
    tmp_path: Path,
) -> None:
    session = make_session(tmp_path, runtime=None, async_runtime=FakeAsyncRuntime())

    try:
        budget = session._speech_max_tokens_for_tts_chunk(
            " ".join(f"word{i}" for i in range(50)),
            utterance_max_tokens=2400,
            chunk_count=2,
        )
    finally:
        session.shutdown()

    assert budget == 50 * 64


def test_vllm_sts_chunk_budget_preserves_single_chunk_utterance_budget(
    tmp_path: Path,
) -> None:
    session = make_session(tmp_path, runtime=None, async_runtime=FakeAsyncRuntime())

    try:
        budget = session._speech_max_tokens_for_tts_chunk(
            "short",
            utterance_max_tokens=2400,
            chunk_count=1,
        )
    finally:
        session.shutdown()

    assert budget == 2400


def test_sanitize_prompt_history_removes_prior_assistant_prompt_leakage() -> None:
    messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Talk about Rust."},
        {
            "role": "assistant",
            "content": (
                "Audex is created by NVIDIA based upon the "
                "Nemotron-Cascade-2 architecture.\n"
                "[CRITICAL] Place each sentence on its own separate line.\n"
                "Rust uses ownership to keep memory safe.\n"
                "Your turn Matt"
            ),
        },
    ]

    assert _sanitize_prompt_history(messages) == [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Talk about Rust."},
        {"role": "assistant", "content": "Rust uses ownership to keep memory safe."},
    ]


def test_sanitize_prompt_history_refreshes_stale_resumed_system_prompt() -> None:
    messages = [
        {
            "role": "system",
            "content": "You are a spoken conversation partner for Audex-Mac.",
        },
        {"role": "user", "content": "Talk about Rust."},
        {"role": "assistant", "content": "Rust uses ownership."},
    ]

    assert _sanitize_prompt_history(messages, system_prompt="Current persona.") == [
        {"role": "system", "content": "Current persona."},
        {"role": "user", "content": "Talk about Rust."},
        {"role": "assistant", "content": "Rust uses ownership."},
    ]


def test_validate_text_prompt_messages_rejects_over_context_without_pruning() -> None:
    session = VllmSpeechToSpeechSession.__new__(VllmSpeechToSpeechSession)
    session.async_runtime = SimpleNamespace(
        cfg_config=SimpleNamespace(max_model_len=20)
    )
    session.runtime = None
    session.conversation = None
    session.persona = SimpleNamespace(system_prompt="System.")
    session.thinking_enabled = False
    session._last_text_context_stats = {}

    def fake_count(messages, *, max_tokens):
        return sum(len(message["content"].split()) for message in messages)

    session._count_text_generation_prompt_tokens = fake_count

    messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "old user words one two three four five"},
        {"role": "assistant", "content": "old assistant words one two three four"},
        {"role": "user", "content": "recent user words"},
        {"role": "assistant", "content": "recent assistant words"},
        {"role": "user", "content": "current question"},
    ]

    with pytest.raises(RuntimeError, match="will not silently drop"):
        session._validate_text_prompt_messages(messages, max_tokens=5)

    assert session._last_text_context_stats == {
        "fits": False,
        "messages_before": 6,
        "prompt_tokens": 24,
        "prompt_token_budget": 15,
        "context_token_limit": 20,
        "response_max_tokens": 5,
    }


def test_validate_text_prompt_messages_uses_runtime_256k_context_limit() -> None:
    session = VllmSpeechToSpeechSession.__new__(VllmSpeechToSpeechSession)
    session.async_runtime = SimpleNamespace(
        cfg_config=SimpleNamespace(max_model_len=None),
        stats=SimpleNamespace(max_model_len=262_144),
    )
    session.runtime = None
    session.conversation = SimpleNamespace(max_context_tokens=1_000_000)
    session.persona = SimpleNamespace(system_prompt="System.")
    session.thinking_enabled = False
    session._last_text_context_stats = {}
    session._count_text_generation_prompt_tokens = lambda messages, max_tokens: 5_025
    messages = [{"role": "user", "content": "resumed history"}]

    assert (
        session._validate_text_prompt_messages(
            messages,
            max_tokens=4_096,
        )
        == messages
    )
    assert session._last_text_context_stats["context_token_limit"] == 262_144


def test_align_conversation_context_limit_clamps_to_loaded_model(
    tmp_path: Path,
) -> None:
    session = VllmSpeechToSpeechSession.__new__(VllmSpeechToSpeechSession)
    session.runtime = SimpleNamespace(stats=SimpleNamespace(max_model_len=131_072))
    session.async_runtime = None
    session.conversation = SimpleNamespace(max_context_tokens=262_144)

    class FakeStore:
        def __init__(self) -> None:
            self.saved = []

        def save(self, conversation) -> None:
            self.saved.append(conversation)

    session.conversation_store = FakeStore()

    session._align_conversation_context_limit()

    assert session.conversation.max_context_tokens == 131_072
    assert session.conversation_store.saved == [session.conversation]


def test_text_conversation_state_kwargs_use_committed_history_prefix(
    tmp_path: Path,
) -> None:
    session = make_session(tmp_path, runtime=FakeRuntime())
    session.conversation = SimpleNamespace(conversation_id="conv-1")
    session.messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Earlier question."},
        {"role": "assistant", "content": "Earlier answer."},
    ]
    tokens = session._text_history_prompt_tokens()

    assert session._text_conversation_state_kwargs() == {
        "conversation_state_key": "conv-1",
        "conversation_state_prefix_token_count": len(tokens),
        "conversation_state_prefix_token_hash": _token_hash(tokens),
    }


def test_persist_sanitized_history_invalidates_stale_kv_cache(tmp_path: Path) -> None:
    session = make_session(tmp_path, runtime=None, async_runtime=FakeAsyncRuntime())
    saved = []

    class FakeStore:
        def save(self, conversation) -> None:
            saved.append(list(conversation.messages))

    conversation = SimpleNamespace(
        root=tmp_path,
        conversation_id="conv",
        messages=[
            {"role": "system", "content": "System."},
            {"role": "assistant", "content": "Leaked line."},
        ],
        token_count=None,
        max_context_tokens=1_000_000,
    )
    kv_cache_path = tmp_path / "conv.kv.safetensors"
    kv_cache_path.write_bytes(b"stale")
    session.conversation = conversation
    session.conversation_store = FakeStore()
    session.messages = [
        {"role": "system", "content": "System."},
        {"role": "assistant", "content": "Clean line."},
    ]

    session._persist_conversation(announce=False, invalidate_kv_cache=True)

    assert not kv_cache_path.exists()
    assert conversation.messages == session.messages
    assert saved == [session.messages]


def test_resumed_history_sanitization_invalidates_stale_state_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlx_package = ModuleType("mlx")
    fake_mx = ModuleType("mlx.core")
    fake_mx.gpu = "gpu"
    fake_mx.set_default_device = lambda _device: None
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.load_speech_decoder_config",
        lambda _path: SimpleNamespace(sample_rate=16_000, hop_length=320),
    )
    monkeypatch.setattr(
        "audex_mac.vllm_sts_cli.load_speech_decoder_weights_mlx",
        lambda _path: {},
    )

    saved = []

    class FakeStore:
        def save(self, conversation) -> None:
            saved.append(list(conversation.messages))

    conversation = SimpleNamespace(
        root=tmp_path,
        conversation_id="conv",
        messages=[
            {"role": "system", "content": "Old persona."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi."},
        ],
        token_count=None,
        max_context_tokens=1_000_000,
    )
    kv_cache_path = tmp_path / "conv.kv.safetensors"
    kv_cache_path.write_bytes(b"stale")
    persona = SimpleNamespace(
        persona_id="assistant",
        path=tmp_path / "assistant.md",
        system_prompt="Current persona.",
    )

    session = VllmSpeechToSpeechSession(
        full_model_path=tmp_path / "model",
        decoder_path=tmp_path / "decoder",
        selected_model_repo="nvidia/Nemotron-Labs-Audex-2B",
        output_dir=tmp_path,
        persona=persona,
        conversation_store=FakeStore(),
        conversation=conversation,
        runtime=FakeRuntime(),
    )

    assert session.messages[0] == {"role": "system", "content": "Current persona."}
    assert conversation.messages == session.messages
    assert saved == [session.messages]
    assert not kv_cache_path.exists()
