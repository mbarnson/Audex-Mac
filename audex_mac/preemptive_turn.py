"""Pre-submit orchestration for recorded Audex turns."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass

from .audio_pcm import SAMPLE_RATE
from .sts_cli import _PlaybackStartGate, _Recording, _RecordingSnapshot


@dataclass(frozen=True, slots=True)
class PreemptiveSubmission[PreparedTurn]:
    submitted_at: float
    samples: tuple[float, ...]
    final_activity: _RecordingSnapshot
    prepared: PreparedTurn | None
    preparation_error: Exception | None


class PreemptiveTurnCoordinator[PreparedTurn]:
    """Stage one replaceable semantic candidate and validate it at submission."""

    def __init__(
        self,
        *,
        min_recording_seconds: float = 1.0,
        silence_seconds: float = 0.18,
        poll_seconds: float = 0.01,
    ) -> None:
        self.min_recording_seconds = max(0.0, float(min_recording_seconds))
        self.silence_seconds = max(0.0, float(silence_seconds))
        self.poll_seconds = max(0.001, float(poll_seconds))

    async def capture(
        self,
        recording: _Recording,
        *,
        play: bool,
        wait_for_submission: Callable[[], float],
        prepare: Callable[
            [_RecordingSnapshot, _PlaybackStartGate | None],
            Awaitable[PreparedTurn],
        ],
    ) -> PreemptiveSubmission[PreparedTurn]:
        submit_task = asyncio.create_task(asyncio.to_thread(wait_for_submission))
        candidate_task: asyncio.Task[PreparedTurn] | None = None
        candidate_gate: _PlaybackStartGate | None = None
        candidate_voice_revision: int | None = None
        last_staged_voice_revision = -1

        async def discard_candidate() -> None:
            nonlocal candidate_task, candidate_gate, candidate_voice_revision
            if candidate_gate is not None:
                candidate_gate.cancel()
            if candidate_task is not None and not candidate_task.done():
                candidate_task.cancel()
            if candidate_task is not None:
                with suppress(BaseException):
                    await candidate_task
            candidate_task = None
            candidate_gate = None
            candidate_voice_revision = None

        while not submit_task.done():
            activity = recording.activity()
            if (
                candidate_task is not None
                and candidate_voice_revision != activity.voice_revision
            ):
                await discard_candidate()
            quiet_seconds = activity.quiet_sample_count / SAMPLE_RATE
            if (
                candidate_task is None
                and activity.voice_revision > last_staged_voice_revision
                and activity.voice_revision > 0
                and activity.sample_count
                >= int(SAMPLE_RATE * self.min_recording_seconds)
                and quiet_seconds >= self.silence_seconds
            ):
                snapshot = recording.snapshot()
                candidate_voice_revision = snapshot.voice_revision
                last_staged_voice_revision = snapshot.voice_revision
                candidate_gate = _PlaybackStartGate() if play else None
                candidate_task = asyncio.create_task(prepare(snapshot, candidate_gate))
            await asyncio.wait({submit_task}, timeout=self.poll_seconds)

        submitted_at = await submit_task
        samples = tuple(recording.stop())
        final_activity = recording.activity()
        stable_candidate = bool(
            candidate_task is not None
            and candidate_voice_revision == final_activity.voice_revision
        )
        if stable_candidate and candidate_task is not None:
            if candidate_gate is not None:
                candidate_gate.release(released_at=submitted_at)
            try:
                prepared = await candidate_task
            except Exception as exc:
                return PreemptiveSubmission(
                    submitted_at=submitted_at,
                    samples=samples,
                    final_activity=final_activity,
                    prepared=None,
                    preparation_error=exc,
                )
            return PreemptiveSubmission(
                submitted_at=submitted_at,
                samples=samples,
                final_activity=final_activity,
                prepared=prepared,
                preparation_error=None,
            )

        await discard_candidate()
        return PreemptiveSubmission(
            submitted_at=submitted_at,
            samples=samples,
            final_activity=final_activity,
            prepared=None,
            preparation_error=None,
        )
