from __future__ import annotations

from typing import Any

import pytest

from audex_mac.sound_lab.adapters import AudexSoundLabPlanner, AudexVariantDesigner
from audex_mac.sound_lab.session import VariantDesignError
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
