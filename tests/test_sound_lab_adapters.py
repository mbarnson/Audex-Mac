from __future__ import annotations

from typing import Any

import pytest

from audex_mac.sound_lab.adapters import AudexSoundLabPlanner, AudexVariantDesigner
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
    def __init__(self, responses: tuple[str, ...]) -> None:
        self.tokenizer = FakeTokenizer()
        self.responses = iter(responses)
        self.requests: list[Any] = []

    async def generate_one_final(self, request: Any) -> VllmRequestResult:
        self.requests.append(request)
        return VllmRequestResult(
            text=next(self.responses),
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
    variants = AudexVariantDesigner(runtime=runtime).design(call, job_id="job-1")

    assert isinstance(call, RenderSoundsCall)
    assert call.count == 2
    assert [variant.seed for variant in variants] == [11, 22]
    assert runtime.tokenizer.tool_sets[-2] is not None
    assert runtime.tokenizer.tool_sets[-2][0]["function"]["name"] == "render_sounds"
    assert [request.debug_name for request in runtime.requests] == [
        "sound-lab-tool",
        "sound-lab-design",
    ]


@pytest.mark.fast
def test_audex_designer_rejects_the_wrong_variant_count() -> None:
    runtime = FakeRuntime(
        ('{"variants":[{"caption":"One sound","difference":"only one","seed":1}]}',)
    )
    call = RenderSoundsCall("Two sounds", 2, {}, ())

    with pytest.raises(ValueError, match="expected 2"):
        AudexVariantDesigner(runtime=runtime).design(call, job_id="job-1")
