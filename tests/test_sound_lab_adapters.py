from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from audex_mac.audio_evaluation_generation import TtaOutputInspection
from audex_mac.audio_evaluation_runner import GenerationAttempt
from audex_mac.sound_lab.adapters import (
    AudexSoundLabPlanner,
    AudexTtaSoundGenerator,
    AudexVariantDesigner,
)
from audex_mac.sound_lab.session import (
    SoundGenerationRequest,
    VariantBrief,
    VariantDesignError,
)
from audex_mac.sound_lab.tools import RenderSoundsCall
from audex_mac.vllm_runtime import VllmRequestResult


class FakeTokenizer:
    def __init__(self) -> None:
        self.tool_sets: list[list[dict[str, Any]] | None] = []

    def get_vocab(self) -> dict[str, int]:
        return {}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        self.tool_sets.append(tools)
        return "\n".join(message["content"] for message in messages)


class FakeRuntime:
    def __init__(self, responses: tuple[str | Exception, ...]) -> None:
        self.tokenizer = FakeTokenizer()
        self.responses = iter(responses)
        self.requests: list[Any] = []

    async def generate_one_final(self, request: Any) -> VllmRequestResult:
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return VllmRequestResult(
            text=response,
            token_ids=(),
            elapsed_seconds=0.1,
            finish_reason="stop",
            request_debug_name=request.debug_name,
        )


@pytest.mark.fast
def test_audex_planner_uses_tool_template_and_designer_requires_distinct_json() -> None:
    runtime = FakeRuntime(
        (
            """<tool_call><function=render_sounds>
<parameter=brief>Two thunderclaps.</parameter>
<parameter=count>2</parameter>
<parameter=constraints>{}</parameter>
<parameter=parent_asset_ids>[]</parameter>
</function></tool_call>""",
            """{"variants":[
{"caption":"A dry nearby thunder crack.","difference":"near and sharp","seed":11},
{"caption":"A distant rolling thunderclap.","difference":"far and long","seed":22}
]}""",
        )
    )
    planner = AudexSoundLabPlanner(runtime=runtime)

    call = planner.plan("Make two thunderclaps.")
    design = AudexVariantDesigner(runtime=runtime).design(call, job_id="job-1")

    assert isinstance(call, RenderSoundsCall)
    assert call.count == 2
    assert len({variant.seed for variant in design.variants}) == 2
    assert runtime.tokenizer.tool_sets[-2] is not None
    assert runtime.tokenizer.tool_sets[-2][0]["function"]["name"] == "render_sounds"
    assert [request.debug_name for request in runtime.requests] == [
        "sound-lab-tool",
        "sound-lab-design",
    ]


@pytest.mark.fast
def test_audex_designer_rejects_the_wrong_variant_count() -> None:
    wrong_count = '{"variants":[{"caption":"One sound","difference":"only one"}]}'
    runtime = FakeRuntime((wrong_count, wrong_count))
    call = RenderSoundsCall("Two sounds", 2, {}, ())

    with pytest.raises(VariantDesignError, match="expected 2"):
        AudexVariantDesigner(runtime=runtime).design(call, job_id="job-1")


@pytest.mark.fast
def test_audex_designer_accepts_unambiguous_fenced_json_and_common_aliases() -> None:
    raw = """Here are the requested variations:
```json
{"sounds":[
{"prompt":"A beagle barking in a tiled kitchen.","rationale":"indoor reflections"},
{"prompt":"A husky barking across a snowy field.","rationale":"open winter space"}
]}
```"""
    runtime = FakeRuntime((raw,))
    call = RenderSoundsCall("Two dogs barking", 2, {}, ())

    result = AudexVariantDesigner(runtime=runtime).design(call, job_id="job-dogs")

    assert [variant.caption for variant in result.variants] == [
        "A beagle barking in a tiled kitchen.",
        "A husky barking across a snowy field.",
    ]
    assert len({variant.seed for variant in result.variants}) == 2
    assert result.raw_attempts == (raw,)
    assert result.repair_used is False


@pytest.mark.fast
def test_audex_designer_repairs_one_malformed_response_then_stops() -> None:
    malformed = '{"variants":[{"caption":"A terrier barking"}'
    repaired = """{"variants":[
{"caption":"A terrier barking in a hallway.","difference":"small indoor dog"},
{"caption":"A mastiff barking in a farmyard.","difference":"large outdoor dog"}
]}"""
    runtime = FakeRuntime((malformed, repaired))
    call = RenderSoundsCall("Two dogs barking", 2, {}, ())

    result = AudexVariantDesigner(runtime=runtime).design(call, job_id="job-repair")

    assert len(result.variants) == 2
    assert result.raw_attempts == (malformed, repaired)
    assert result.repair_used is True
    assert [request.debug_name for request in runtime.requests] == [
        "sound-lab-design",
        "sound-lab-design-repair",
    ]


@pytest.mark.fast
def test_audex_designer_fails_closed_after_one_repair_and_retains_attempts() -> None:
    attempts = ("not json", '{"variants": [}')
    runtime = FakeRuntime(attempts)
    call = RenderSoundsCall("Two dogs barking", 2, {}, ())

    with pytest.raises(VariantDesignError) as error_info:
        AudexVariantDesigner(runtime=runtime).design(call, job_id="job-failed")

    assert error_info.value.raw_attempts == attempts
    assert len(error_info.value.errors) == 2
    assert len(runtime.requests) == 2


@pytest.mark.fast
def test_audex_designer_repairs_missing_difference_instead_of_inventing_it() -> None:
    missing = '{"variants":[{"caption":"A terrier barking"}]}'
    repaired = (
        '{"variants":[{"caption":"A terrier barking",'
        '"difference":"small dog in a reflective hallway"}]}'
    )
    runtime = FakeRuntime((missing, repaired))
    call = RenderSoundsCall("One dog barking", 1, {}, ())

    result = AudexVariantDesigner(runtime=runtime).design(call, job_id="job-reason")

    assert result.repair_used is True
    assert result.variants[0].difference == "small dog in a reflective hallway"


@pytest.mark.fast
def test_audex_designer_repairs_duplicate_captions() -> None:
    duplicate = """{"variants":[
{"caption":"A dog barking.","difference":"indoors"},
{"caption":"A dog barking.","difference":"outdoors"}
]}"""
    repaired = """{"variants":[
{"caption":"A beagle barking indoors.","difference":"small reflective room"},
{"caption":"A shepherd barking outdoors.","difference":"large open field"}
]}"""
    runtime = FakeRuntime((duplicate, repaired))

    result = AudexVariantDesigner(runtime=runtime).design(
        RenderSoundsCall("Two dogs", 2, {}, ()),
        job_id="job-duplicate",
    )

    assert result.repair_used is True
    assert len({variant.caption for variant in result.variants}) == 2


@pytest.mark.fast
def test_audex_designer_retains_first_attempt_when_repair_inference_fails() -> None:
    first = "not json"
    runtime = FakeRuntime((first, RuntimeError("engine stopped")))

    with pytest.raises(VariantDesignError) as error_info:
        AudexVariantDesigner(runtime=runtime).design(
            RenderSoundsCall("Two dogs", 2, {}, ()),
            job_id="job-engine-error",
        )

    assert error_info.value.raw_attempts == (first,)
    assert error_info.value.repair_used is True
    assert "engine stopped" in str(error_info.value)


@pytest.mark.fast
def test_sound_generator_salvages_clean_early_audio_and_retries_only_failures(
    tmp_path: Path,
) -> None:
    early = _generation_attempt(tmp_path, failures=("incomplete_target",), frames=100)
    tiny = _generation_attempt(tmp_path, failures=("incomplete_target",), frames=20)
    malformed = _generation_attempt(
        tmp_path,
        failures=("missing_end_token", "incomplete_target"),
        frames=50,
        reached_end=False,
    )
    recovered = _generation_attempt(tmp_path, failures=(), frames=500)
    still_bad = _generation_attempt(
        tmp_path,
        failures=("missing_end_token", "incomplete_target"),
        frames=40,
        reached_end=False,
    )
    adapter = FakeGenerationAdapter(((early, tiny, malformed), (recovered, still_bad)))

    outcomes = tuple(
        AudexTtaSoundGenerator(
            runtime=object(),
            decode_to_wav=object(),
            adapter_factory=lambda **_kwargs: adapter,
        ).generate_many(
            (
                _sound_request("asset-1", 11),
                _sound_request("asset-2", 22),
                _sound_request("asset-3", 33),
            ),
            output_dir=tmp_path,
        )
    )

    assert [len(call) for call in adapter.calls] == [3, 2]
    assert adapter.calls[1][0][1] != 22
    assert adapter.calls[1][1][1] != 33
    assert outcomes[0].generated is not None
    assert outcomes[0].generated.duration_seconds == 2.0
    assert outcomes[1].generated is not None
    assert outcomes[1].generated.elapsed_seconds == 3.0
    assert outcomes[1].generated.seed_used == adapter.calls[1][0][1]
    assert [attempt.seed for attempt in outcomes[1].attempts] == [
        22,
        adapter.calls[1][0][1],
    ]
    assert outcomes[1].attempts[0].failures == ("incomplete_target",)
    assert outcomes[2].error is not None
    assert "after one retry" in outcomes[2].error
    assert "initial=(seed=33; missing_end_token; incomplete_target" in outcomes[2].error
    assert "retry=(seed=" in outcomes[2].error
    assert "missing_end_token; incomplete_target; frames=40" in outcomes[2].error
    assert "frames=40" in outcomes[2].error
    assert "reached_end=False" in outcomes[2].error


@pytest.mark.fast
def test_sound_generator_preserves_first_pass_success_and_retry_failure_evidence(
    tmp_path: Path,
) -> None:
    ready = _generation_attempt(tmp_path, failures=(), frames=500)
    malformed = _generation_attempt(
        tmp_path,
        failures=("missing_end_token", "incomplete_target"),
        frames=10,
        reached_end=False,
    )
    adapter = ExplodingRetryAdapter((ready, malformed))
    outcomes = AudexTtaSoundGenerator(
        runtime=object(),
        decode_to_wav=object(),
        adapter_factory=lambda **_kwargs: adapter,
    ).generate_many(
        (_sound_request("asset-ready", 11), _sound_request("asset-retry", 22)),
        output_dir=tmp_path,
    )

    assert next(outcomes).asset_id == "asset-ready"
    failed = next(outcomes)
    assert failed.asset_id == "asset-retry"
    assert "retry batch failed" in str(failed.error)
    assert [attempt.seed for attempt in failed.attempts] == [
        22,
        22 ^ 0x5A17_D3C9,
    ]
    assert failed.attempts[1].failures == (
        "technical_failure: RuntimeError: retry engine failed",
    )


class FakeGenerationAdapter:
    def __init__(self, batches: tuple[tuple[GenerationAttempt, ...], ...]) -> None:
        self._batches = iter(batches)
        self.calls: list[tuple[tuple[Any, int], ...]] = []

    def generate_many(
        self, cases: tuple[tuple[Any, int], ...]
    ) -> tuple[GenerationAttempt, ...]:
        self.calls.append(cases)
        return next(self._batches)


class ExplodingRetryAdapter:
    def __init__(self, first_batch: tuple[GenerationAttempt, ...]) -> None:
        self.first_batch = first_batch
        self.call_count = 0

    def generate_many(
        self, cases: tuple[tuple[Any, int], ...]
    ) -> tuple[GenerationAttempt, ...]:
        del cases
        self.call_count += 1
        if self.call_count == 1:
            return self.first_batch
        raise RuntimeError("retry engine failed")


def _sound_request(asset_id: str, seed: int) -> SoundGenerationRequest:
    return SoundGenerationRequest(
        asset_id=asset_id,
        variant=VariantBrief("A bark.", "variation", seed),
    )


def _generation_attempt(
    tmp_path: Path,
    *,
    failures: tuple[str, ...],
    frames: int,
    reached_end: bool = True,
) -> GenerationAttempt:
    path = tmp_path / f"{frames}-{'-'.join(failures) or 'valid'}.wav"
    path.write_bytes(b"RIFF")
    return GenerationAttempt(
        raw_wav_path=path,
        enhanced_wav_path=None,
        structure=TtaOutputInspection(
            codec_ids=(),
            codec_token_count=frames * 4,
            frame_count=frames,
            duration_seconds=frames / 50,
            reached_end_token=reached_end,
            first_phase_mismatch=None,
            unexpected_token_ids=(),
            failures=failures,
        ),
        signal_metrics={},
        elapsed_seconds=1.5,
        finish_reason="stop",
    )
