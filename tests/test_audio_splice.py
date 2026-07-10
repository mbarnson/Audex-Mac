from __future__ import annotations

import pytest

from audex_mac.audio_splice import (
    find_sound_token_positions,
    splice_audio_embeddings_mlx,
    validate_audio_splice_plan,
)
from tests.mlx_test_utils import require_mlx_core

pytestmark = pytest.mark.fast


def test_find_sound_token_positions_returns_all_placeholders() -> None:
    assert find_sound_token_positions([10, 29, 29, 31], sound_token_id=29) == (1, 2)


def test_validate_audio_splice_plan_rejects_count_mismatch() -> None:
    with pytest.raises(ValueError, match="Mismatch"):
        validate_audio_splice_plan(
            [10, 29, 29, 31],
            (1, 2048),
            sound_token_id=29,
        )


def test_validate_audio_splice_plan_accepts_matching_count() -> None:
    plan = validate_audio_splice_plan(
        [10, 29, 29, 31],
        (2, 2048),
        sound_token_id=29,
    )

    assert plan.sound_positions == (1, 2)
    assert plan.audio_embedding_shape == (2, 2048)


def test_splice_audio_embeddings_mlx_replaces_only_sound_tokens() -> None:
    mx = require_mlx_core()

    token_ids = mx.array([10, 29, 29, 31], dtype=mx.int32)
    input_embeddings = mx.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=mx.float32,
    )
    audio_embeddings = mx.array(
        [
            [20.0, 21.0],
            [30.0, 31.0],
        ],
        dtype=mx.float32,
    )

    spliced = splice_audio_embeddings_mlx(
        token_ids,
        input_embeddings,
        audio_embeddings,
        sound_token_id=29,
    )

    assert spliced.tolist() == [
        [1.0, 1.0],
        [20.0, 21.0],
        [30.0, 31.0],
        [4.0, 4.0],
    ]
