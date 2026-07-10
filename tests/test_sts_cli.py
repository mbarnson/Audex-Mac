from __future__ import annotations

import inspect
import json
import sys
import time
import wave
from array import array
from pathlib import Path

import pytest

from audex_mac import sts_cli
from audex_mac.conversations import ConversationStore
from audex_mac.interactive_input import InputKind, TurnInput
from audex_mac.personas import Persona
from audex_mac.speech_output import (
    SpeechOutputSmokeResult,
    float_samples_to_pcm16_bytes,
    write_pcm16_wav,
)

pytestmark = pytest.mark.fast


class _WhitespaceTokenizer:
    def apply_chat_template(self, messages, **kwargs) -> str:
        return "\n".join(
            f"{message['role']}:{message['content']}" for message in messages
        )

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _attach_minimal_conversation_state(
    session: sts_cli.AudexSpeechToSpeechSession,
    tmp_path: Path,
) -> None:
    session.conversation = None
    session.conversation_store = None
    session.enable_kv_cache = True
    session.persona = Persona(
        persona_id="assistant",
        path=tmp_path / "assistant.md",
        metadata={},
        prompt="Speak briefly.",
    )
    session.messages = [{"role": "system", "content": session.persona.system_prompt}]


def _speech_output_result(*, wav_path: Path, run_log_path: Path, **overrides):
    values = {
        "backend": "mlx",
        "device": "Device(gpu, 0)",
        "prompt_tokens": 57,
        "generated_token_ids": (166944,),
        "generated_codec_frames": (35867,),
        "reached_end_token": False,
        "hit_max_tokens": False,
        "waveform_shape": (320,),
        "sample_rate": 16_000,
        "hop_length": 320,
        "finite": True,
        "peak_abs": 0.1,
        "wav_path": wav_path,
        "run_log_path": run_log_path,
    }
    values.update(overrides)
    return SpeechOutputSmokeResult(**values)


def test_shared_write_pcm16_wav_writes_capture_fixture(tmp_path: Path) -> None:
    wav_path = tmp_path / "capture.wav"

    write_pcm16_wav(wav_path, [-1.5, 0.0, 1.5], sample_rate=16_000)

    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == 3


def test_continuous_pcm_player_uses_one_raw_output_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[bytes] = []
    stream_kwargs: list[dict[str, object]] = []

    class FakeRawOutputStream:
        def __init__(self, **kwargs) -> None:
            stream_kwargs.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        @property
        def write_available(self) -> int:
            return 512 - len(writes)

        @property
        def latency(self) -> float:
            return 0.05

        def write(self, pcm: bytes) -> bool:
            writes.append(pcm)
            return len(writes) == 2

    class FakeSoundDevice:
        RawOutputStream = FakeRawOutputStream

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice())

    player = sts_cli._ContinuousPcmPlayer(started_at=0.0, sample_rate=16_000)
    player.start()
    player.enqueue_samples([0.0, 1.0])
    player.enqueue_samples([-1.0])
    player.close()

    assert stream_kwargs == [
        {
            "samplerate": 16_000,
            "channels": 1,
            "dtype": "int16",
            "latency": "low",
        }
    ]
    assert writes == [
        float_samples_to_pcm16_bytes([0.0, 1.0]),
        float_samples_to_pcm16_bytes([-1.0]),
    ]
    assert player.first_playback_started_seconds is not None
    assert player.first_playback_estimated_audible_at is not None
    assert (
        player.first_playback_estimated_audible_at
        >= player.first_playback_started_at + 0.05
    )
    diagnostics = player.diagnostics()
    assert diagnostics["device_underflow_count"] == 1
    assert diagnostics["queue_underrun_count"] == 0
    assert diagnostics["queue_overrun_count"] == 0
    assert diagnostics["chunks_written"] == 2
    assert diagnostics["bytes_written"] == sum(len(pcm) for pcm in writes)
    assert diagnostics["min_write_available_frames"] == 511
    assert diagnostics["latency"] == "low"
    assert diagnostics["output_latency_seconds"] == 0.05
    assert diagnostics["playback_drain_seconds"] >= 0.0


def test_pcm16_linear_upsampler_preserves_stream_continuity() -> None:
    first = array("h", [0, 300]).tobytes()
    second = array("h", [600]).tobytes()

    first_output, previous = sts_cli._upsample_pcm16_linear(first, factor=3)
    second_output, previous = sts_cli._upsample_pcm16_linear(
        second,
        factor=3,
        previous_sample=previous,
    )

    assert list(array("h", first_output)) == [0, 0, 0, 100, 200, 300]
    assert list(array("h", second_output)) == [400, 500, 600]
    assert previous == 600


def test_continuous_pcm_player_accepts_explicit_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_kwargs: list[dict[str, object]] = []

    class FakeRawOutputStream:
        def __init__(self, **kwargs) -> None:
            stream_kwargs.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def write(self, _pcm: bytes) -> bool:
            return False

    class FakeSoundDevice:
        RawOutputStream = FakeRawOutputStream

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice())

    player = sts_cli._ContinuousPcmPlayer(
        started_at=0.0,
        sample_rate=16_000,
        latency="high",
    )
    player.start()
    player.enqueue_samples([0.0])
    player.close()

    assert stream_kwargs[0]["latency"] == "high"
    assert player.diagnostics()["latency"] == "high"


def test_continuous_pcm_player_holds_generated_audio_until_gate_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[bytes] = []

    class FakeRawOutputStream:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def write(self, pcm: bytes) -> bool:
            writes.append(pcm)
            return False

    class FakeSoundDevice:
        RawOutputStream = FakeRawOutputStream

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice())
    gate = sts_cli._PlaybackStartGate()
    player = sts_cli._ContinuousPcmPlayer(
        started_at=0.0,
        sample_rate=16_000,
        prebuffer_seconds=0.0,
        start_gate=gate,
    )
    player.start()
    player.enqueue_samples([0.25, -0.25])

    assert writes == []
    submitted_at = time.time() - 0.1
    gate.release(released_at=submitted_at)
    player.close()

    assert writes == [float_samples_to_pcm16_bytes([0.25, -0.25])]
    assert gate.released is True
    assert gate.cancelled is False
    assert gate.released_at == submitted_at


def test_continuous_pcm_player_reports_first_device_and_audible_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    milestones: list[tuple[float, float | None]] = []

    class FakeRawOutputStream:
        latency = 0.05

        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def write(self, _pcm: bytes) -> bool:
            return False

    class FakeSoundDevice:
        RawOutputStream = FakeRawOutputStream

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice())
    player = sts_cli._ContinuousPcmPlayer(
        started_at=time.time(),
        sample_rate=16_000,
        prebuffer_seconds=0.0,
        first_playback_callback=lambda device_at, audible_at: milestones.append(
            (device_at, audible_at)
        ),
    )
    player.start()
    player.enqueue_samples([0.25, -0.25])
    player.close()

    assert len(milestones) == 1
    device_at, audible_at = milestones[0]
    assert audible_at is not None
    assert audible_at >= device_at + 0.05


def test_continuous_pcm_player_discards_stale_generated_audio_when_gate_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[bytes] = []

    class FakeRawOutputStream:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def write(self, pcm: bytes) -> bool:
            writes.append(pcm)
            return False

    class FakeSoundDevice:
        RawOutputStream = FakeRawOutputStream

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice())
    gate = sts_cli._PlaybackStartGate()
    player = sts_cli._ContinuousPcmPlayer(
        started_at=0.0,
        sample_rate=16_000,
        prebuffer_seconds=0.0,
        start_gate=gate,
    )
    player.start()
    player.enqueue_samples([0.25, -0.25])
    gate.cancel()
    player.close()

    assert writes == []
    assert gate.cancelled is True


def test_recording_exposes_thread_safe_snapshot_and_voice_revision() -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.stopped = False
            self.closed = False

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    stream = FakeStream()
    recording = sts_cli._Recording(stream=stream)
    recording.append_chunk([[0.0], [0.001]])
    quiet = recording.snapshot()
    recording.append_chunk([[0.02], [-0.03]])
    voiced = recording.snapshot()

    assert quiet.sample_count == 2
    assert quiet.voice_revision == 0
    assert quiet.quiet_sample_count == 2
    assert voiced.samples == (0.0, 0.001, 0.02, -0.03)
    assert voiced.sample_count == 4
    assert voiced.voice_revision == 1
    assert voiced.last_voice_at is not None
    assert voiced.quiet_sample_count == 0
    assert recording.stop() == [0.0, 0.001, 0.02, -0.03]
    assert stream.stopped is True
    assert stream.closed is True


def test_continuous_pcm_player_adaptive_prebuffer_expands_for_slow_arrival() -> None:
    player = sts_cli._ContinuousPcmPlayer(
        started_at=0.0,
        sample_rate=16_000,
        prebuffer_seconds=4.0,
    )
    player._arrival_rate_bytes_per_second = 16_000.0

    assert player._seconds_for_bytes(player._target_prebuffer_bytes()) == 8.0


def test_continuous_pcm_player_adaptive_prebuffer_keeps_configured_realtime_target() -> (
    None
):
    player = sts_cli._ContinuousPcmPlayer(
        started_at=0.0,
        sample_rate=16_000,
        prebuffer_seconds=4.0,
    )
    player._arrival_rate_bytes_per_second = 32_000.0

    assert player._seconds_for_bytes(player._target_prebuffer_bytes()) == 4.0


def test_continuous_pcm_player_adaptive_prebuffer_caps_very_slow_arrival() -> None:
    player = sts_cli._ContinuousPcmPlayer(
        started_at=0.0,
        sample_rate=16_000,
        prebuffer_seconds=4.0,
    )
    player._arrival_rate_bytes_per_second = 1.0

    assert player._seconds_for_bytes(player._target_prebuffer_bytes()) == 8.0


def test_continuous_pcm_player_adaptive_prebuffer_keeps_minimum_floor() -> None:
    player = sts_cli._ContinuousPcmPlayer(
        started_at=0.0,
        sample_rate=16_000,
        prebuffer_seconds=0.01,
    )
    player._arrival_rate_bytes_per_second = 32_000.0

    assert player._seconds_for_bytes(player._target_prebuffer_bytes()) == 0.08


def test_continuous_pcm_player_can_disable_adaptive_prebuffer() -> None:
    player = sts_cli._ContinuousPcmPlayer(
        started_at=0.0,
        sample_rate=16_000,
        prebuffer_seconds=4.0,
        adaptive_prebuffer=False,
    )
    player._arrival_rate_bytes_per_second = 1.0

    assert player._seconds_for_bytes(player._target_prebuffer_bytes()) == 4.0


def test_continuous_pcm_player_does_not_count_initial_prebuffer_as_overrun() -> None:
    player = sts_cli._ContinuousPcmPlayer(started_at=0.0, sample_rate=16_000)

    player.enqueue_pcm(b"\0" * (16_000 * 2 * 4))

    diagnostics = player.diagnostics()
    assert diagnostics["queue_overrun_count"] == 0
    assert diagnostics["queue_high_water_seconds"] == 4.0


def test_continuous_pcm_player_counts_overrun_after_playback_starts() -> None:
    player = sts_cli._ContinuousPcmPlayer(started_at=0.0, sample_rate=16_000)
    player._playback_started = True

    player.enqueue_pcm(b"\0" * (16_000 * 2 * 4))

    assert player.diagnostics()["queue_overrun_count"] == 1


def test_continuous_pcm_player_allows_intentional_prebuffer_after_playback_starts() -> (
    None
):
    player = sts_cli._ContinuousPcmPlayer(
        started_at=0.0,
        sample_rate=16_000,
        prebuffer_seconds=8.0,
    )
    player._playback_started = True

    player.enqueue_pcm(b"\0" * (16_000 * 2 * 8))

    diagnostics = player.diagnostics()
    assert diagnostics["queue_overrun_count"] == 0
    assert diagnostics["queue_high_water_seconds"] == 8.0


def test_continuous_pcm_player_drains_buffer_before_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    player = sts_cli._ContinuousPcmPlayer(started_at=0.0, sample_rate=16_000)
    player._playback_buffered_until = 12.5

    monkeypatch.setattr(sts_cli.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(sts_cli.time, "sleep", slept.append)

    player._drain_playback_buffer()

    assert slept == [2.5]
    assert player.diagnostics()["playback_drain_seconds"] == 2.5


def test_continuous_pcm_player_accepts_prepacked_pcm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[bytes] = []

    class FakeRawOutputStream:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def write(self, pcm: bytes) -> bool:
            writes.append(pcm)
            return False

    class FakeSoundDevice:
        RawOutputStream = FakeRawOutputStream

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice())

    pcm = float_samples_to_pcm16_bytes([0.0, 1.0])
    player = sts_cli._ContinuousPcmPlayer(started_at=0.0, sample_rate=16_000)
    player.start()
    player.enqueue_pcm(pcm)
    player.close()

    assert writes == [pcm]


def test_clean_transcription_extracts_quoted_nvidia_style_output() -> None:
    assert (
        sts_cli._clean_transcription("The transcription is 'hello there'<|im_end|>")
        == "hello there"
    )


def test_clean_streaming_transcription_prefers_partial_quoted_content() -> None:
    assert (
        sts_cli._clean_streaming_transcription(
            "Language: English. The spoken content of the audio is 'hello"
        )
        == "hello"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Language: English", ""),
        ("The content of the input audio", ""),
        ("The spoken content", ""),
        ("The spoken content of the audio", ""),
        (
            "The content of the input audio " "hi there can you talk to me?",
            "hi there can you talk to me?",
        ),
        (
            "Language: English\nThe spoken content of the audio "
            "Hi there. Can you talk to me?",
            "Hi there. Can you talk to me?",
        ),
        (
            "The spoken content of the audio is " "Hi there. Can you talk to me?",
            "Hi there. Can you talk to me?",
        ),
    ],
)
def test_clean_streaming_transcription_hides_asr_wrapper_partials(
    raw: str,
    expected: str,
) -> None:
    assert sts_cli._clean_streaming_transcription(raw) == expected


def test_clean_response_text_extracts_spoken_answer_after_thinking() -> None:
    assert (
        sts_cli._clean_response_text(
            "<think>\nprivate reasoning\n</think>\nThis is the spoken answer."
        )
        == "This is the spoken answer."
    )


def test_session_speech_budget_scales_from_response_tokens() -> None:
    session = object.__new__(sts_cli.AudexSpeechToSpeechSession)
    session.speech_max_tokens = None

    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))

    session.tokenizer = FakeTokenizer()

    assert session._speech_max_tokens_for_text("short answer") == 2400
    assert session._speech_max_tokens_for_text("word " * 100) == 6400


def test_session_speech_budget_honors_explicit_override() -> None:
    session = object.__new__(sts_cli.AudexSpeechToSpeechSession)
    session.speech_max_tokens = 123

    assert session._speech_max_tokens_for_text("word " * 100) == 123


def test_session_persists_conversation_and_binary_kv_cache(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    persona = Persona(
        persona_id="assistant",
        path=tmp_path / "assistant.md",
        metadata={},
        prompt="Speak briefly.",
    )
    conversation = store.create(
        persona_id=persona.persona_id,
        persona_path=persona.path,
        system_prompt=persona.system_prompt,
    )
    session = object.__new__(sts_cli.AudexSpeechToSpeechSession)
    session.conversation = conversation
    session.conversation_store = store
    session.messages = list(conversation.messages) + [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi."},
    ]
    session.enable_kv_cache = True
    session.selected_model_repo = "nvidia/Nemotron-Labs-Audex-30B-A3B"
    session.thinking_enabled = False
    session.persona = persona
    saved: dict[str, object] = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs) -> str:
            return "\n".join(f"{m['role']}:{m['content']}" for m in messages)

        def encode(self, text: str) -> list[int]:
            return [ord(char) for char in text]

    class FakeArray(list):
        def __getitem__(self, item):
            return self

    class FakeMx:
        int32 = "int32"

        def array(self, values, dtype=None):
            return FakeArray(values)

        def eval(self, value) -> None:
            return None

    class FakeCache:
        state = "state"

    class FakeCacheModule:
        def make_prompt_cache(self, model):
            return [FakeCache()]

        def save_prompt_cache(self, path, cache, metadata):
            saved["path"] = path
            saved["cache"] = cache
            saved["metadata"] = metadata
            Path(path).write_bytes(b"mlx safetensors")

    class FakeModel:
        def __call__(self, tokens, cache):
            saved["model_called"] = True

    session.tokenizer = FakeTokenizer()
    session.mx = FakeMx()
    session.cache_module = FakeCacheModule()
    session.model = FakeModel()

    session._persist_conversation()

    reloaded = store.load(conversation.conversation_id)
    assert reloaded.token_count is not None
    assert reloaded.messages[-1] == {"role": "assistant", "content": "Hi."}
    assert (
        conversation.transcript_path.read_text(encoding="utf-8").count("## User") == 1
    )
    assert saved["model_called"] is True
    assert Path(saved["path"]).name.endswith(".kv.safetensors")
    metadata = saved["metadata"]
    assert metadata["format"] == "audex-mac-conversation-kv-v1"
    assert metadata["conversation_id"] == conversation.conversation_id
    assert metadata["prompt_token_count"] == str(reloaded.token_count)


def test_session_loads_matching_binary_kv_cache(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    conversation = store.create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System.",
    )
    session = object.__new__(sts_cli.AudexSpeechToSpeechSession)
    session.conversation = conversation
    session.enable_kv_cache = True
    session.messages = list(conversation.messages)
    session.selected_model_repo = "nvidia/Nemotron-Labs-Audex-30B-A3B"
    session.thinking_enabled = False
    cache_path = tmp_path / f"{conversation.conversation_id}.kv.safetensors"
    cache_path.write_bytes(b"mlx safetensors")

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs) -> str:
            return "\n".join(f"{m['role']}:{m['content']}" for m in messages)

        def encode(self, text: str) -> list[int]:
            return [ord(char) for char in text]

    tokens = tuple(FakeTokenizer().encode("system:System."))

    class FakeCacheModule:
        def load_prompt_cache(self, path, return_metadata=False):
            assert Path(path) == cache_path
            assert return_metadata is True
            return ["cache"], {
                "conversation_id": conversation.conversation_id,
                "prompt_token_count": str(len(tokens)),
                "prompt_token_hash": sts_cli._token_hash(tokens),
                "selected_model": "nvidia/Nemotron-Labs-Audex-30B-A3B",
            }

    session.tokenizer = FakeTokenizer()
    session.cache_module = FakeCacheModule()

    session._load_conversation_prompt_cache()

    assert session.text_prompt_cache == ["cache"]
    assert session.text_prompt_cache_token_count == len(tokens)


def test_startup_greeting_uses_first_run_text_for_new_conversation(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    conversation = store.create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System.",
    )

    assert (
        sts_cli.startup_greeting_text(
            conversation=conversation,
            conversation_resumed=False,
        )
        == "Hi! I'm Audex. What should I call you, and what should we talk about today?"
    )


def test_startup_greeting_uses_name_when_resuming(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    conversation = store.create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System.",
    )
    conversation.user_name = "Pat"

    assert (
        sts_cli.startup_greeting_text(
            conversation=conversation,
            conversation_resumed=True,
        )
        == "Hi, Pat! Nice to hear from you again. What do you want to talk about today?"
    )


def test_startup_greeting_says_hi_when_resuming_without_name(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    conversation = store.create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System.",
    )

    assert (
        sts_cli.startup_greeting_text(
            conversation=conversation,
            conversation_resumed=True,
        )
        == "Hi! Nice to hear from you again. What do you want to talk about today?"
    )


def test_startup_greeting_does_not_reintroduce_a_resumed_named_user(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    conversation = store.create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System.",
    )
    conversation.user_name = "Matt"

    assert (
        sts_cli.startup_greeting_text(
            conversation=conversation,
            conversation_resumed=True,
        )
        == "Hi, Matt! Nice to hear from you again. What do you want to talk about today?"
    )


def test_interactive_ptt_speaks_startup_greeting_before_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeSession:
        def __init__(self, **kwargs) -> None:
            events.append("session")

        @property
        def stats(self):
            return sts_cli.AudexSpeechToSpeechSessionStats(
                model_load_seconds=1.0,
                audio_component_load_seconds=2.0,
                decoder_load_seconds=3.0,
                speech_warmup_seconds=4.0,
                turns=0,
            )

        def speak_startup_greeting(self, *, conversation_resumed: bool, play: bool):
            events.append(f"greeting:{conversation_resumed}:{play}")

    def fake_input() -> TurnInput:
        events.append("input")
        return TurnInput(InputKind.QUIT)

    monkeypatch.setattr(sts_cli, "AudexSpeechToSpeechSession", FakeSession)
    monkeypatch.setattr(sts_cli, "read_turn_input", fake_input)

    result = sts_cli.run_interactive_ptt(
        full_model_path=tmp_path / "model",
        decoder_path=tmp_path / "decoder",
        output_dir=tmp_path,
        play=True,
        conversation_resumed=True,
    )

    assert result is None
    assert events == ["session", "greeting:True:True", "input"]


def test_interactive_cli_routes_typed_text_without_starting_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    result = sts_cli.SpeechToSpeechTurnResult(
        transcript="First line\nSecond line",
        response_text="Spoken answer.",
        input_wav_path=None,
        output_wav_path=tmp_path / "answer.wav",
        run_log_path=tmp_path / "typed.json",
        played=True,
    )

    class FakeSession:
        def __init__(self, **kwargs) -> None:
            pass

        @property
        def stats(self):
            return sts_cli.AudexSpeechToSpeechSessionStats(1.0, 0.0, 0.5, 0.2, 0)

        def speak_startup_greeting(self, **kwargs) -> None:
            pass

        def run_turn_from_text(self, *, user_text: str, play: bool):
            events.append(("text", user_text, play))
            return result

    inputs = iter(
        [
            TurnInput(InputKind.TEXT, "First line\nSecond line"),
            TurnInput(InputKind.QUIT),
        ]
    )
    monkeypatch.setattr(sts_cli, "AudexSpeechToSpeechSession", FakeSession)
    monkeypatch.setattr(sts_cli, "read_turn_input", lambda: next(inputs))
    monkeypatch.setattr(
        sts_cli,
        "_start_recording",
        lambda: pytest.fail("typed input must not start recording"),
    )

    actual = sts_cli.run_interactive_ptt(
        full_model_path=tmp_path / "model",
        decoder_path=tmp_path / "decoder",
        output_dir=tmp_path,
        play=True,
    )

    assert actual is result
    assert events == [("text", "First line\nSecond line", True)]


def test_direct_mlx_typed_turn_skips_asr_and_writes_honest_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object.__new__(sts_cli.AudexSpeechToSpeechSession)
    session.output_dir = tmp_path
    session.selected_model_repo = "model"
    session.response_max_tokens = 128
    session.speech_max_tokens = 2400
    session.thinking_enabled = False
    session.turns = 0
    session.conversation = None
    session.persona = Persona(
        persona_id="assistant",
        path=tmp_path / "assistant.md",
        metadata={},
        prompt="Speak briefly.",
    )
    session.messages = [{"role": "system", "content": "Speak briefly."}]
    session.model_load_seconds = 1.0
    session.audio_component_load_seconds = 0.0
    session.decoder_load_seconds = 0.5
    session.speech_warmup_seconds = 0.2
    session._count_messages_tokens = lambda _messages: 3
    speech = _speech_output_result(
        wav_path=tmp_path / "answer.wav",
        run_log_path=tmp_path / "speech.json",
        first_audio_ready_seconds=0.3,
        first_playback_started_seconds=None,
    )
    monkeypatch.setattr(
        session,
        "transcribe_wav",
        lambda _path: pytest.fail("typed turns must skip ASR"),
    )
    monkeypatch.setattr(session, "generate_text_response", lambda _text: "Answer.")
    monkeypatch.setattr(
        session,
        "generate_speech_output_streaming",
        lambda **_kwargs: speech,
    )

    result = session.run_turn_from_text(user_text="Typed question.", play=False)

    assert result.input_wav_path is None
    assert result.transcript == "Typed question."
    log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert log["input_mode"] == "text"
    assert log["typed_text"] == "Typed question."
    assert log["asr_skipped"] is True


def test_audex_source_uses_mlx_cache_not_pickle_or_numpy() -> None:
    source_root = Path(sts_cli.__file__).resolve().parent
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in source_root.glob("*.py")
        if path.name != "__init__.py"
    )

    assert "pickle" not in source
    assert "import numpy" not in source
    assert ".cpu().numpy()" not in source
    assert "save_prompt_cache" in source
    assert "load_prompt_cache" in source


def test_split_tts_segments_uses_sentence_boundaries() -> None:
    assert sts_cli._split_tts_segments("First sentence. Second sentence!") == (
        "First sentence.",
        "Second sentence!",
    )


def test_split_tts_segments_breaks_long_sentence() -> None:
    text = " ".join(["word"] * 120)

    segments = sts_cli._split_tts_segments(text, max_chars=80)

    assert len(segments) > 1
    assert all(len(segment) <= 80 for segment in segments)


def test_split_tts_segments_filters_markdown_only_fragments() -> None:
    assert sts_cli._split_tts_segments("A useful answer.\n\n**") == (
        "A useful answer.",
    )
    assert sts_cli._split_tts_segments("**Important:** say this.") == (
        "Important: say this.",
    )


def test_split_tts_segments_drops_short_truncated_tail() -> None:
    assert sts_cli._split_tts_segments("One complete sentence. In") == (
        "One complete sentence.",
    )


def test_streaming_speech_output_writes_chunks_and_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object.__new__(sts_cli.AudexSpeechToSpeechSession)
    session.output_dir = tmp_path
    session.full_model_path = tmp_path / "model"
    session.decoder_path = tmp_path / "decoder"

    class FakeDecoderConfig:
        sample_rate = 16_000
        hop_length = 320
        lookahead_steps = 4

    class FakeScalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    class FakeMx:
        def abs(self, waveform):
            return [abs(sample) for sample in waveform.tolist()]

        def max(self, values):
            return FakeScalar(max(values))

        def default_device(self) -> str:
            return "Device(gpu, 0)"

    class FakeWaveform:
        @property
        def shape(self) -> tuple[int, ...]:
            return (len(self.tolist()),)

        def tolist(self) -> list[float]:
            return [0.1, -0.1] * 320

    session.decoder_config = FakeDecoderConfig()
    session.decoder_weights = {}
    session.mx = FakeMx()
    session.token_map = sts_cli.build_codec_token_map(
        {
            "<speechgen_start>": 131075,
            "<speechgen_end>": 131076,
            "<speechcodec_7>": 131077,
            "<speechcodec_8>": 131078,
            "<speechcodec_9>": 131079,
            "<speechcodec_10>": 131080,
        }
    )
    players: list[object] = []

    class FakeContinuousPcmPlayer:
        first_playback_started_seconds = 0.25

        def __init__(self, *, started_at: float, sample_rate: int) -> None:
            self.started_at = started_at
            self.sample_rate = sample_rate
            self.started = False
            self.closed = False
            self.enqueued: list[tuple[float, ...]] = []
            players.append(self)

        def start(self) -> None:
            self.started = True

        def enqueue_samples(self, samples) -> None:
            self.enqueued.append(tuple(float(sample) for sample in samples))

        def close(self) -> None:
            self.closed = True

        def diagnostics(self) -> dict[str, object]:
            return {
                "device_underflow_count": 2,
                "queue_underrun_count": 1,
                "queue_overrun_count": 0,
                "queue_high_water_seconds": 0.42,
                "chunks_written": 3,
                "bytes_written": 3840,
                "min_write_available_frames": 128,
                "latency": "low",
                "playback_drain_seconds": 0.4,
            }

    decoder_sessions: list[object] = []

    class FakeDecoderSession:
        def __init__(self, *, weights, config, chunk_frames: int) -> None:
            self.weights = weights
            self.config = config
            self.chunk_frames = chunk_frames
            self.pushed: list[tuple[tuple[int, ...], ...]] = []
            self.flush_count = 0
            self.reset_count = 0
            decoder_sessions.append(self)

        def push(self, frames):
            self.pushed.append(tuple(tuple(frame) for frame in frames))
            return [(self.config.sample_rate, FakeWaveform())]

        def flush(self):
            self.flush_count += 1
            return [(self.config.sample_rate, FakeWaveform())]

        def reset(self) -> None:
            self.reset_count += 1

    def fake_generate_speech_tokens(
        *,
        text: str,
        max_tokens: int,
        progress_callback=None,
        token_callback=None,
    ):
        assert max_tokens == 12
        for token_id in (131077, 131078, 131076):
            if token_callback is not None:
                token_callback(token_id)
        return sts_cli.SpeechTokenGenerationSmokeResult(
            backend="mlx_lm",
            device="Device(gpu, 0)",
            model_type="nemotron_h_audex",
            vocab_size=205312,
            prompt_tokens=10,
            prompt_max_token_id=131075,
            speechgen_start_id=131075,
            speechgen_end_id=131076,
            codec_token_count=4,
            generated_token_ids=(131077, 131078, 131076),
            generated_token_text=(
                "<speechcodec_7>",
                "<speechcodec_8>",
                "<speechgen_end>",
            ),
            generated_codec_frames=(7, 8),
            logprobs_shape=(205312,),
            reached_end_token=True,
            hit_max_tokens=False,
            temperature=0.8,
            top_p=1.0,
            top_k=0,
            cfg_scale_reference=3.0,
            cfg_applied=True,
        )

    monkeypatch.setattr(sts_cli, "AudexSpeechDecoderSession", FakeDecoderSession)
    monkeypatch.setattr(sts_cli, "_ContinuousPcmPlayer", FakeContinuousPcmPlayer)
    session.generate_speech_tokens = fake_generate_speech_tokens

    result = session.generate_speech_output_streaming(
        text="First. Second.",
        max_tokens_per_segment=12,
        play=True,
        decoder_chunk_frames=2,
    )

    assert result.streaming is True
    assert result.ready is True
    assert result.segments == ("First.", "Second.")
    assert len(result.chunk_wav_paths) >= 2
    assert result.first_audio_ready_seconds is not None
    assert result.first_playback_started_seconds == 0.25
    assert result.wav_path.is_file()
    assert result.run_log_path.is_file()
    assert len(decoder_sessions) == 1
    decoder_session = decoder_sessions[0]
    assert decoder_session.chunk_frames == 2
    assert decoder_session.pushed == [((7,), (8,)), ((7,), (8,))]
    assert decoder_session.flush_count == 2
    assert decoder_session.reset_count == 2
    assert len(players) == 1
    player = players[0]
    assert player.sample_rate == 16_000
    assert player.started is True
    assert player.closed is True
    assert len(player.enqueued) >= 2
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["playback_transport"] == "sounddevice_raw_output_stream"
    assert run_log["playback_prebuffer_seconds"] == 0.8
    assert run_log["playback_diagnostics"] == {
        "device_underflow_count": 2,
        "queue_underrun_count": 1,
        "queue_overrun_count": 0,
        "queue_high_water_seconds": 0.42,
        "chunks_written": 3,
        "bytes_written": 3840,
        "min_write_available_frames": 128,
        "latency": "low",
        "playback_drain_seconds": 0.4,
    }


def test_run_turn_uses_streaming_speech_with_default_segment_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"RIFF")
    output_wav = tmp_path / "output.wav"
    output_wav.write_bytes(b"RIFF")
    output_log = tmp_path / "speech-output.json"
    output_log.write_text("{}\n", encoding="utf-8")
    session = object.__new__(sts_cli.AudexSpeechToSpeechSession)
    session.output_dir = tmp_path
    session.selected_model_repo = "nvidia/Nemotron-Labs-Audex-30B-A3B"
    session.response_max_tokens = 4096
    session.speech_max_tokens = None
    session.thinking_enabled = False
    session.turns = 0
    session.model_load_seconds = 1.0
    session.audio_component_load_seconds = 2.0
    session.decoder_load_seconds = 3.0
    session.speech_warmup_seconds = 4.0
    session.tokenizer = _WhitespaceTokenizer()
    _attach_minimal_conversation_state(session, tmp_path)

    calls: list[tuple[str, int]] = []

    def fake_transcribe(wav_path: Path) -> str:
        return "talk to me"

    def fake_response(transcript: str) -> str:
        return "One sentence. Another sentence."

    def fake_speech_output(**kwargs):
        calls.append((kwargs["text"], kwargs["max_tokens_per_segment"]))
        return _speech_output_result(
            wav_path=output_wav,
            run_log_path=output_log,
            reached_end_token=True,
            streaming=True,
            first_audio_ready_seconds=0.5,
        )

    session.transcribe_wav = fake_transcribe
    session.generate_text_response = fake_response
    session.generate_speech_output_streaming = fake_speech_output

    result = session.run_turn_from_wav(input_wav_path=input_wav, play=False)

    assert result.output_wav_path == output_wav
    assert calls == [("One sentence. Another sentence.", 2400)]
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["speech_max_tokens"] == 2400
    assert run_log["timings"]["first_audio_ready_seconds"] >= 0.5
    assert run_log["timings"]["tts_first_audio_ready_seconds"] == 0.5
    assert run_log["timings"]["session_speech_warmup_seconds"] == 4.0


def test_run_turn_does_not_speak_acknowledgement_before_asr_when_playing(
    tmp_path: Path,
) -> None:
    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"RIFF")
    answer_wav = tmp_path / "answer.wav"
    answer_wav.write_bytes(b"RIFF")
    answer_log = tmp_path / "answer.json"
    answer_log.write_text("{}\n", encoding="utf-8")
    session = object.__new__(sts_cli.AudexSpeechToSpeechSession)
    session.output_dir = tmp_path
    session.selected_model_repo = "nvidia/Nemotron-Labs-Audex-30B-A3B"
    session.response_max_tokens = 4096
    session.speech_max_tokens = None
    session.thinking_enabled = False
    session.turns = 0
    session.model_load_seconds = 1.0
    session.audio_component_load_seconds = 2.0
    session.decoder_load_seconds = 3.0
    session.speech_warmup_seconds = 4.0
    session.tokenizer = _WhitespaceTokenizer()
    _attach_minimal_conversation_state(session, tmp_path)
    events: list[str] = []

    def fake_transcribe(wav_path: Path) -> str:
        events.append("asr")
        return "talk to me"

    def fake_response(transcript: str) -> str:
        events.append("text")
        return "One sentence."

    def fake_speech_output_from_segments(**kwargs):
        text = " ".join(kwargs["text_segments"])
        events.append(f"tts:{text}")
        return _speech_output_result(
            wav_path=answer_wav,
            run_log_path=answer_log,
            reached_end_token=True,
            streaming=True,
            first_audio_ready_seconds=4.5,
            first_playback_started_seconds=4.6,
        )

    session.transcribe_wav = fake_transcribe
    session.generate_text_response = fake_response
    session.generate_speech_output_streaming_from_text_segments = (
        fake_speech_output_from_segments
    )

    result = session.run_turn_from_wav(input_wav_path=input_wav, play=True)

    assert result.output_wav_path == answer_wav
    assert events == ["asr", "text", "tts:One sentence."]
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["acknowledgement_output_wav_path"] is None
    assert run_log["timings"]["first_audio_ready_seconds"] == 4.5
    assert run_log["timings"]["first_playback_started_seconds"] == 4.6
    assert run_log["timings"]["ack_first_audio_ready_seconds"] is None
    assert run_log["timings"]["tts_first_audio_ready_seconds"] == 4.5


def test_run_turn_streams_answer_tts_after_three_generated_sentences(
    tmp_path: Path,
) -> None:
    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"RIFF")
    answer_wav = tmp_path / "answer.wav"
    answer_wav.write_bytes(b"RIFF")
    answer_log = tmp_path / "answer.json"
    answer_log.write_text("{}\n", encoding="utf-8")
    session = object.__new__(sts_cli.AudexSpeechToSpeechSession)
    session.output_dir = tmp_path
    session.selected_model_repo = "nvidia/Nemotron-Labs-Audex-30B-A3B"
    session.response_max_tokens = 4096
    session.speech_max_tokens = None
    session.thinking_enabled = False
    session.turns = 0
    session.model_load_seconds = 1.0
    session.audio_component_load_seconds = 2.0
    session.decoder_load_seconds = 3.0
    session.speech_warmup_seconds = 4.0
    session.tokenizer = _WhitespaceTokenizer()
    _attach_minimal_conversation_state(session, tmp_path)
    spoken_batches: list[str] = []

    def fake_transcribe(wav_path: Path) -> str:
        return "talk to me"

    def fake_response(transcript: str, *, response_text_callback=None) -> str:
        assert response_text_callback is not None
        response_text_callback("First. Second. Third.")
        return "First. Second. Third. Fourth."

    def fake_speech_output_from_segments(**kwargs):
        spoken_batches.extend(kwargs["text_segments"])
        return _speech_output_result(
            wav_path=answer_wav,
            run_log_path=answer_log,
            reached_end_token=True,
            streaming=True,
            first_audio_ready_seconds=0.8,
            first_playback_started_seconds=1.1,
            segments=tuple(spoken_batches),
        )

    session.transcribe_wav = fake_transcribe
    session.generate_text_response = fake_response
    session.generate_speech_output_streaming_from_text_segments = (
        fake_speech_output_from_segments
    )

    result = session.run_turn_from_wav(input_wav_path=input_wav, play=True)

    assert result.response_text == "First. Second. Third. Fourth."
    assert spoken_batches == ["First. Second. Third.", "Fourth."]
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))
    assert run_log["speech_output_run_log_path"] == str(answer_log)


def test_persistent_session_warms_speech_output_path_at_startup() -> None:
    source = inspect.getsource(sts_cli.AudexSpeechToSpeechSession.__init__)

    assert "self._warm_speech_output_path()" in source
    assert "self.speech_warmup_seconds" in source


def test_persistent_session_reuses_mlx_lm_tokenizer() -> None:
    source = inspect.getsource(sts_cli.AudexSpeechToSpeechSession.__init__)

    assert "self.model, self.tokenizer = load(" in source
    assert "AutoTokenizer.from_pretrained" not in source


def test_run_fixture_turn_orchestrates_audex_only_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"RIFF")
    output_wav = tmp_path / "output.wav"
    output_wav.write_bytes(b"RIFF")
    output_log = tmp_path / "speech-output.json"
    output_log.write_text("{}\n", encoding="utf-8")
    calls: list[tuple[str, str]] = []

    def fake_transcribe(
        *, full_model_path: Path, wav_path: Path, max_tokens: int = 256
    ) -> str:
        calls.append(("transcribe", wav_path.name))
        return "hello"

    def fake_response(**kwargs) -> str:
        transcript = kwargs["transcript"]
        calls.append(("respond", transcript))
        return "Hi back."

    def fake_speech_output(**kwargs):
        calls.append(("tts", kwargs["text"]))
        return _speech_output_result(
            wav_path=output_wav,
            run_log_path=output_log,
            hit_max_tokens=True,
        )

    monkeypatch.setattr(sts_cli, "transcribe_wav_with_audex", fake_transcribe)
    monkeypatch.setattr(sts_cli, "generate_audex_text_response", fake_response)
    monkeypatch.setattr(sts_cli, "run_speech_output_smoke", fake_speech_output)

    result = sts_cli.run_fixture_turn(
        full_model_path=tmp_path / "model",
        decoder_path=tmp_path / "decoder",
        input_wav_path=input_wav,
        selected_model_repo="nvidia/Nemotron-Labs-Audex-2B",
        output_dir=tmp_path,
        play=False,
    )

    assert result.transcript == "hello"
    assert result.response_text == "Hi back."
    assert result.output_wav_path == output_wav
    assert result.run_log_path.is_file()
    assert result.played is False
    assert calls == [
        ("transcribe", "input.wav"),
        ("respond", "hello"),
        ("tts", "Hi back."),
    ]
    run_log = result.run_log_path.read_text(encoding="utf-8")
    assert '"selected_model": "nvidia/Nemotron-Labs-Audex-2B"' in run_log
    assert '"response_max_tokens": 4096' in run_log
    assert '"speech_max_tokens": 2400' in run_log
    assert '"timings"' in run_log
