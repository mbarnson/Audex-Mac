from __future__ import annotations

from pathlib import Path

import pytest

from audex_mac.speech_output import write_pcm16_wav
from audex_mac.sts_cli import SpeechToSpeechTurnResult
from audex_mac.vllm_commands import run_vllm_preemptive_replay

pytestmark = pytest.mark.fast


def test_preemptive_replay_feeds_recording_in_realtime_chunks(tmp_path: Path) -> None:
    input_wav = tmp_path / "input.wav"
    samples = tuple(0.02 if index % 2 else -0.02 for index in range(160))
    write_pcm16_wav(input_wav, samples, sample_rate=16_000)
    observed_sample_counts: list[int] = []
    expected = SpeechToSpeechTurnResult(
        transcript="fixture",
        response_text="response",
        input_wav_path=input_wav,
        output_wav_path=tmp_path / "output.wav",
        run_log_path=tmp_path / "output.json",
        played=False,
    )

    class FakeSession:
        def run_preemptive_recorded_turn(
            self,
            *,
            recording,
            input_wav_path: Path,
            play: bool,
            wait_for_submission,
        ):
            assert recording.activity().sample_count == 0
            wait_for_submission()
            observed_sample_counts.append(recording.activity().sample_count)
            assert play is False
            assert input_wav_path.parent == tmp_path
            return expected

    actual = run_vllm_preemptive_replay(
        full_model_path=tmp_path / "model",
        decoder_path=tmp_path / "decoder",
        input_wav_path=input_wav,
        submission_delay_seconds=0.0,
        replay_chunk_seconds=0.001,
        output_dir=tmp_path,
        play=False,
        session=FakeSession(),
    )

    assert actual is expected
    assert observed_sample_counts == [len(samples)]
