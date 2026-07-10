from __future__ import annotations

import asyncio
import time

import pytest

from audex_mac.preemptive_turn import PreemptiveTurnCoordinator
from audex_mac.sts_cli import _Recording

pytestmark = pytest.mark.fast


class _FakeStream:
    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_preemptive_coordinator_cancels_stale_voice_and_keeps_latest_candidate() -> (
    None
):
    recording = _Recording(stream=_FakeStream())
    recording.append_chunk([[0.02], [-0.02]] * 16)
    recording.append_chunk([[0.0]] * 32)
    coordinator: PreemptiveTurnCoordinator[str] = PreemptiveTurnCoordinator(
        min_recording_seconds=0.001,
        silence_seconds=0.002,
        poll_seconds=0.001,
    )
    prepared_revisions: list[int] = []
    cancelled_revisions: list[int] = []
    gates: list[object] = []

    async def prepare(snapshot, gate) -> str:
        prepared_revisions.append(snapshot.voice_revision)
        gates.append(gate)
        if snapshot.voice_revision == 1:
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                cancelled_revisions.append(snapshot.voice_revision)
                raise
        return f"revision-{snapshot.voice_revision}"

    def submit_after_revision() -> float:
        time.sleep(0.012)
        recording.append_chunk([[0.03], [-0.03]] * 16)
        recording.append_chunk([[0.0]] * 32)
        time.sleep(0.012)
        return time.time()

    submission = asyncio.run(
        coordinator.capture(
            recording,
            play=True,
            wait_for_submission=submit_after_revision,
            prepare=prepare,
        )
    )

    assert submission.prepared == "revision-2"
    assert submission.preparation_error is None
    assert prepared_revisions == [1, 2]
    assert cancelled_revisions == [1]
    assert gates[0].cancelled is True
    assert gates[1].released is True
    assert submission.final_activity.voice_revision == 2


def test_preemptive_coordinator_uses_captured_quiet_audio_not_callback_delay() -> None:
    recording = _Recording(stream=_FakeStream())
    recording.append_chunk([[0.02], [-0.02]] * 16)
    coordinator: PreemptiveTurnCoordinator[str] = PreemptiveTurnCoordinator(
        min_recording_seconds=0.001,
        silence_seconds=0.002,
        poll_seconds=0.001,
    )
    prepared: list[int] = []

    async def prepare(snapshot, _gate) -> str:
        prepared.append(snapshot.voice_revision)
        return "prepared"

    def submit_after_callback_gap() -> float:
        time.sleep(0.01)
        return time.time()

    submission = asyncio.run(
        coordinator.capture(
            recording,
            play=False,
            wait_for_submission=submit_after_callback_gap,
            prepare=prepare,
        )
    )

    assert prepared == []
    assert submission.prepared is None
