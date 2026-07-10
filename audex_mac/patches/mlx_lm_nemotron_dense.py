"""MLX-LM model implementation for NVIDIA Audex text-only Nemotron Dense.

This mirrors the Hugging Face `modeling_nemotron_dense.py` shipped with
NVIDIA's Audex checkpoints closely enough for MLX-LM/vLLM Metal to construct
and load the text-only weights without mutating the Hugging Face cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    norm_eps: float
    vocab_size: int
    head_dim: int | None = None
    hidden_act: str = "relu2"
    max_position_embeddings: int = 131072
    num_key_value_heads: int | None = None
    rope_parameters: dict[str, Any] | None = None
    rope_theta: float = 100000000.0
    partial_rotary_factor: float = 1.0
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        rope_parameters = self.rope_parameters or {}
        self.rope_theta = float(rope_parameters.get("rope_theta", self.rope_theta))
        self.partial_rotary_factor = float(
            rope_parameters.get("partial_rotary_factor", self.partial_rotary_factor)
        )


@partial(mx.compile, shapeless=True)
def relu_squared(x: mx.array) -> mx.array:
    return nn.relu(x).square()


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = int(args.num_key_value_heads)
        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            int(args.partial_rotary_factor * self.head_dim),
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(batch, seq_len, self.n_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(batch, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch, seq_len, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(relu_squared(self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask, cache=cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class NemotronDenseModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            h = layer(h, mask, cache=layer_cache)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = NemotronDenseModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        remapped: dict[str, mx.array] = {}
        for key, value in weights.items():
            if "self_attn.rotary_emb.inv_freq" in key:
                continue
            if key.startswith(("audio_encoder.", "audio_projector.")):
                continue
            remapped[key.replace("model.embeddings.", "model.embed_tokens.")] = value
        if self.args.tie_word_embeddings:
            remapped.pop("lm_head.weight", None)
        return remapped

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]
