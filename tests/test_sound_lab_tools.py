from __future__ import annotations

import pytest

from audex_mac.sound_lab.tools import RenderSoundsCall, parse_sound_lab_tool_call


@pytest.mark.fast
def test_parse_render_sounds_call_from_nemotron_xml() -> None:
    raw = """<think></think>I can make those.
<tool_call>
<function=render_sounds>
<parameter=brief>
Five close, ugly explosions with distinct acoustic spaces.
</parameter>
<parameter=count>
5
</parameter>
<parameter=constraints>
{"duration_seconds": 10, "avoid": ["music", "speech"]}
</parameter>
<parameter=parent_asset_ids>
["asset-a"]
</parameter>
</function>
</tool_call>"""

    parsed = parse_sound_lab_tool_call(raw)

    assert parsed == RenderSoundsCall(
        brief="Five close, ugly explosions with distinct acoustic spaces.",
        count=5,
        constraints={"duration_seconds": 10, "avoid": ["music", "speech"]},
        parent_asset_ids=("asset-a",),
        preamble="I can make those.",
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "raw",
    [
        "<tool_call><function=delete_files></function></tool_call>",
        (
            "<tool_call><function=render_sounds>"
            "<parameter=brief>x</parameter>"
            "<parameter=count>9</parameter>"
            "<parameter=constraints>{}</parameter>"
            "<parameter=parent_asset_ids>[]</parameter>"
            "</function></tool_call>"
        ),
        (
            "<tool_call><function=render_sounds>"
            "<parameter=brief>x</parameter>"
            "<parameter=count>2</parameter>"
            "<parameter=constraints>{not-json}</parameter>"
            "<parameter=parent_asset_ids>[]</parameter>"
            "</function></tool_call>trailing text"
        ),
    ],
)
def test_parse_sound_lab_tool_call_fails_closed(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_sound_lab_tool_call(raw)


@pytest.mark.fast
def test_parse_render_sounds_treats_null_constraints_as_empty() -> None:
    parsed = parse_sound_lab_tool_call(
        "<tool_call><function=render_sounds>"
        "<parameter=brief>Two impacts</parameter>"
        "<parameter=count>2</parameter>"
        "<parameter=constraints>null</parameter>"
        "</function></tool_call>"
    )

    assert parsed.constraints == {}
