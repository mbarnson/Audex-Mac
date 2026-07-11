from __future__ import annotations

import re

import pytest

from audex_mac.audio_evaluation_generation import (
    TtaRecipe,
    build_tta_requests,
    inspect_tta_output,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self._vocab = {
            "<audiogen_start>": 10,
            "<audiogen_end>": 11,
            **{f"<audiocodec_{codec_id}>": 100 + codec_id for codec_id in range(4096)},
        }

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str) -> list[int]:
        pieces = re.findall(r"<unk>|[A-Za-z0-9'-]+|[^\s]", text)
        return list(range(len(pieces)))


@pytest.mark.fast
def test_build_tta_requests_uses_nvidia_cfg3_recipe_and_phase_mask() -> None:
    tokenizer = FakeTokenizer()

    cond, uncond = build_tta_requests(
        tokenizer,
        caption="A dog barks twice beside a passing train.",
        case_id="audiocaps-17",
        seed=73,
    )

    assert cond.debug_name == "tta-audiocaps-17-cond"
    assert uncond.debug_name == "tta-audiocaps-17-uncond"
    assert len(cond.prompt["prompt_token_ids"]) == len(
        uncond.prompt["prompt_token_ids"]
    )
    assert cond.sampling.max_tokens == 2048
    assert cond.sampling.temperature == 1.0
    assert cond.sampling.top_p == 1.0
    assert cond.sampling.top_k == 80
    assert cond.sampling.seed == 73
    assert cond.sampling.stop == ("<audiogen_end>",)
    assert cond.sampling.stop_token_ids == (11,)
    assert cond.sampling.extra_args == {
        "cfg_scale": 3.0,
        "cfg_role": "cond",
        "cfg_pair_id": "tta-audiocaps-17",
        "tta_rvq": {
            "phase_token_ids": [
                list(range(100, 1124)),
                list(range(1124, 2148)),
                list(range(2148, 3172)),
                list(range(3172, 4196)),
            ],
            "start_tid": 10,
            "end_tid": 11,
            "codec_cap": 2000,
            "start_in_prompt": True,
        },
    }
    assert uncond.sampling.extra_args == {
        **cond.sampling.extra_args,
        "cfg_role": "uncond",
    }


@pytest.mark.fast
def test_build_tta_requests_rejects_incomplete_audio_codec_vocab() -> None:
    tokenizer = FakeTokenizer()
    tokenizer._vocab.pop("<audiocodec_2048>")

    with pytest.raises(ValueError, match="RVQ phase 2.*2048"):
        build_tta_requests(
            tokenizer,
            caption="Rain on a metal roof.",
            case_id="fixture-rain",
            seed=1,
        )


@pytest.mark.fast
def test_inspect_tta_output_requires_complete_phase_valid_ten_second_stream() -> None:
    tokenizer = FakeTokenizer()
    token_ids = [
        100 + phase * 1024 + frame % 1024 for frame in range(500) for phase in range(4)
    ]
    token_ids.append(11)

    result = inspect_tta_output(tokenizer, token_ids, recipe=TtaRecipe())

    assert result.valid
    assert result.codec_token_count == 2000
    assert result.frame_count == 500
    assert result.duration_seconds == 10.0
    assert result.reached_end_token
    assert result.first_phase_mismatch is None


@pytest.mark.fast
@pytest.mark.parametrize("frames", [499, 501])
def test_inspect_tta_output_accepts_one_complete_frame_target_tolerance(
    frames: int,
) -> None:
    tokenizer = FakeTokenizer()
    token_ids = [
        100 + phase * 1024 + frame % 1024
        for frame in range(frames)
        for phase in range(4)
    ]
    token_ids.append(11)

    result = inspect_tta_output(tokenizer, token_ids, recipe=TtaRecipe())

    assert result.valid
    assert result.frame_count == frames
    assert result.first_phase_mismatch is None


@pytest.mark.fast
def test_inspect_tta_output_reports_phase_and_truncation_failures() -> None:
    tokenizer = FakeTokenizer()
    token_ids = [100, 1124, 3172, 2148]

    result = inspect_tta_output(tokenizer, token_ids, recipe=TtaRecipe())

    assert not result.valid
    assert result.first_phase_mismatch == {
        "index": 2,
        "codec_id": 3072,
        "actual_phase": 3,
        "expected_phase": 2,
    }
    assert "missing_end_token" in result.failures
    assert "incomplete_target" in result.failures


@pytest.mark.fast
def test_clean_early_end_is_usable_only_as_an_explicit_preview_policy() -> None:
    tokenizer = FakeTokenizer()
    token_ids = [
        100 + phase * 1024 + frame % 1024 for frame in range(100) for phase in range(4)
    ]
    token_ids.append(11)

    result = inspect_tta_output(tokenizer, token_ids, recipe=TtaRecipe())

    assert result.valid is False
    assert result.failures == ("incomplete_target",)
    assert result.duration_seconds == 2.0
    assert result.usable_early_preview(minimum_duration_seconds=1.0) is True
    assert result.usable_early_preview(minimum_duration_seconds=3.0) is False
