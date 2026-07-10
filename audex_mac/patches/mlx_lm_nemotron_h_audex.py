"""MLX-LM shim for NVIDIA Audex 30B-A3B Nemotron-H checkpoints.

MLX-LM's upstream Nemotron-H implementation is text-only. Audex-Mac runs the
Audex audio encoder/projector separately in MLX, splices the projected audio
embeddings into the prompt, then uses the language backbone for generation.
This shim keeps the upstream backbone but makes it loadable from the full Audex
checkpoint and compatible with MLX-LM's ``input_embeddings`` generation path.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.nemotron_h import *  # noqa: F401,F403
from mlx_lm.models.nemotron_h import (
    ArraysCache,
    create_attention_mask,
    create_ssm_mask,
)
from mlx_lm.models.nemotron_h import (
    Model as _UpstreamModel,
)
from mlx_lm.models.nemotron_h import (
    NemotronHModel as _UpstreamNemotronHModel,
)


class NemotronHModel(_UpstreamNemotronHModel):
    @property
    def embed_tokens(self):
        return self.embeddings

    def __call__(
        self,
        inputs,
        cache: Any | None = None,
        input_embeddings: mx.array | None = None,
    ):
        hidden_states = (
            input_embeddings
            if input_embeddings is not None
            else self.embeddings(inputs)
        )

        if cache is None:
            cache = [None] * len(self.layers)
        attn_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_cache = cache[self.ssm_idx]
        ssm_mask = (
            create_ssm_mask(hidden_states, ssm_cache)
            if isinstance(ssm_cache, ArraysCache)
            else None
        )

        cache_counter = 0
        for layer in self.layers:
            if layer.block_type == "M" or layer.block_type == "*":
                c = cache[cache_counter]
                cache_counter += 1
            else:
                c = None
            if layer.block_type == "M" and not isinstance(c, ArraysCache):
                c = None

            mask = attn_mask if layer.block_type == "*" else ssm_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm_f(hidden_states)


class Model(_UpstreamModel):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = NemotronHModel(args)

    @property
    def model(self):
        return self.backbone

    def __call__(
        self,
        inputs: mx.array,
        cache: Any | None = None,
        input_embeddings: mx.array | None = None,
    ):
        out = self.backbone(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
        )
        return self.lm_head(out)

    def sanitize(self, weights):
        sanitized = super().sanitize(weights)
        return {
            key: value
            for key, value in sanitized.items()
            if not key.startswith(("audio_encoder.", "audio_projector."))
        }
