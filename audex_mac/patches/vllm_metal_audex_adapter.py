"""Audex multimodal adapter patch for the pinned vLLM Metal runtime."""

from __future__ import annotations

import gc
import importlib.abc
import importlib.machinery
import inspect
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from audex_mac.audio_contract import (
    DEFAULT_SOUND_EMBEDDING_SIZE,
    SOUND_END_TOKEN,
    SOUND_START_TOKEN,
    SOUND_TOKEN,
    audio_clip_count,
    audio_embedding_count,
)
from audex_mac.audio_pcm import SAMPLE_RATE, prepare_audex_pcm_clips

AUDEX_MODEL_TYPES = frozenset({"nemotron_dense_audex", "nemotron_h_audex"})
AUDEX_ARCHITECTURES = frozenset(
    {
        "NemotronDenseAudexForConditionalGeneration",
        "NemotronHAudexForConditionalGeneration",
        "NemotronDenseForCausalLM",
        "NemotronHForCausalLM",
    }
)
PATCH_SENTINEL = "_audex_mac_audex_adapter_patch"
TEXT_BACKBONE_PATCH_SENTINEL = "_audex_mac_audex_text_backbone_patch"
LIFECYCLE_PATCH_SENTINEL = "_audex_mac_audex_lifecycle_patch"
PATCHING_PATCH_SENTINEL = "_audex_mac_audex_patching_patch"
SDPA_WRAPPER_PATCH_SENTINEL = "_audex_mac_audex_sdpa_wrapper_patch"
SDPA_NO_ROPE_PATCH_SENTINEL = "_audex_mac_audex_sdpa_no_rope_patch"
PROCESSOR_PATCH_SENTINEL = "_audex_mac_audex_processor_patch"
PROCESSOR_IMPORT_HOOK_SENTINEL = "_audex_mac_audex_processor_import_hook"
RENDERER_PATCH_SENTINEL = "_audex_mac_renderer_mm_state_patch"
NON_PAGED_MM_PREFILL_PATCH_SENTINEL = "_audex_mac_non_paged_mm_prefill_patch"
NEMOTRON_MODULE = "vllm.model_executor.models.nemotron"
NEMOTRON_PROCESSOR_TARGETS = {
    "vllm.model_executor.models.nemotron": "NemotronForCausalLM",
    "vllm.model_executor.models.nemotron_h": "NemotronHForCausalLM",
}
LAST_ERROR: str | None = None
AUDIO_EMBEDDING_TRIM_ENV = "AUDEX_VLLM_TRIM_PADDED_AUDIO_EMBEDDINGS"
EAGER_AUDIO_COMPONENTS_ENV = "AUDEX_VLLM_EAGER_AUDIO_COMPONENTS"
TEXT_STATE_KEY_ARG = "audex_text_state_key"
TEXT_STATE_PREFIX_TOKEN_COUNT_ARG = "audex_text_state_prefix_token_count"
TEXT_STATE_PREFIX_TOKEN_HASH_ARG = "audex_text_state_prefix_token_hash"
TEXT_STATE_BOUNDARY_ARG = "audex_text_state_boundary"
TEXT_STATE_COMMITTED_HISTORY_BOUNDARY = "committed_history_prefill"


def _debug(message: str) -> None:
    if os.environ.get("AUDEX_VLLM_MM_DEBUG") == "1":
        print(f"Audex vLLM MM debug: {message}", flush=True)


@dataclass(frozen=True, slots=True)
class AudexAudioEncodeResult:
    """Audex audio tower output for one vLLM multimodal feature."""

    hidden_states: Any
    deepstack_visual_embeds: None = None


def projected_audio_embedding_count(projected_embeddings: Any) -> int:
    shape = getattr(projected_embeddings, "shape", None)
    if shape is None:
        raise ValueError("Audex projected audio embeddings must expose shape.")
    if len(shape) == 2:
        return int(shape[0])
    if len(shape) == 3:
        return int(shape[0]) * int(shape[1])
    raise ValueError(
        "Audex projected audio embeddings must be 2D or 3D, got "
        f"shape={tuple(shape)}."
    )


def find_projected_audio_placeholder_range(
    prompt_token_ids: list[int] | tuple[int, ...],
    *,
    sound_token_id: int,
    num_embeddings: int,
) -> tuple[int, int]:
    if num_embeddings <= 0:
        raise ValueError(f"num_embeddings must be positive, got {num_embeddings}")
    needle = [int(sound_token_id)] * int(num_embeddings)
    haystack = [int(token_id) for token_id in prompt_token_ids]
    for offset in range(0, len(haystack) - len(needle) + 1):
        if haystack[offset : offset + len(needle)] == needle:
            return offset, len(needle)
    raise ValueError(
        "Audex prompt does not contain a contiguous projected-audio placeholder "
        f"run of {num_embeddings} tokens for sound_token_id={sound_token_id}."
    )


def build_projected_audio_feature_spec(
    *,
    projected_embeddings: Any,
    prompt_token_ids: list[int] | tuple[int, ...],
    sound_token_id: int,
    identifier: str = "audex-audio-0",
    feature_spec_cls: Any | None = None,
    placeholder_range_cls: Any | None = None,
) -> Any:
    """Build the vLLM feature spec Audex needs for projected audio embeddings."""

    if feature_spec_cls is None or placeholder_range_cls is None:
        from vllm_metal.multimodal.feature_spec import (
            MultiModalFeatureSpec,
            PlaceholderRange,
        )

        feature_spec_cls = feature_spec_cls or MultiModalFeatureSpec
        placeholder_range_cls = placeholder_range_cls or PlaceholderRange

    num_embeddings = projected_audio_embedding_count(projected_embeddings)
    offset, length = find_projected_audio_placeholder_range(
        prompt_token_ids,
        sound_token_id=sound_token_id,
        num_embeddings=num_embeddings,
    )
    return feature_spec_cls(
        data={"audex_projected_embeddings": projected_embeddings},
        modality="audio",
        identifier=identifier,
        mm_position=placeholder_range_cls(offset=offset, length=length),
    )


def build_raw_audio_feature_spec(
    *,
    raw_audio_samples: Any,
    sample_rate: int,
    prompt_token_ids: list[int] | tuple[int, ...],
    sound_token_id: int,
    num_embeddings: int,
    identifier: str = "audex-raw-audio-0",
    feature_spec_cls: Any | None = None,
    placeholder_range_cls: Any | None = None,
) -> Any:
    """Build the vLLM feature spec Audex needs for raw PCM audio."""

    if feature_spec_cls is None or placeholder_range_cls is None:
        from vllm_metal.multimodal.feature_spec import (
            MultiModalFeatureSpec,
            PlaceholderRange,
        )

        feature_spec_cls = feature_spec_cls or MultiModalFeatureSpec
        placeholder_range_cls = placeholder_range_cls or PlaceholderRange

    offset, length = find_projected_audio_placeholder_range(
        prompt_token_ids,
        sound_token_id=sound_token_id,
        num_embeddings=num_embeddings,
    )
    return feature_spec_cls(
        data={
            "audex_raw_audio_samples": raw_audio_samples,
            "sample_rate": int(sample_rate),
            "audex_raw_audio_num_embeddings": int(num_embeddings),
        },
        modality="audio",
        identifier=identifier,
        mm_position=placeholder_range_cls(offset=offset, length=length),
    )


def _patch_nemotron_h_mixer_attention_detection() -> bool:
    """Teach vLLM Metal graph walking about Nemotron-H full-attention mixers."""

    try:
        patching_module = importlib.import_module("vllm_metal.attention.patching")
    except Exception as exc:
        global LAST_ERROR
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return False
    if getattr(patching_module, PATCHING_PATCH_SENTINEL, False):
        return True

    original_find_attn_attr = patching_module.find_attn_attr

    def find_attn_attr_with_audex_mixer(layer: Any) -> str | None:
        attn_attr = original_find_attn_attr(layer)
        if attn_attr is not None:
            return attn_attr
        if getattr(layer, "block_type", None) == "*" and hasattr(layer, "mixer"):
            return "mixer"
        return None

    patching_module.find_attn_attr = find_attn_attr_with_audex_mixer
    setattr(patching_module, PATCHING_PATCH_SENTINEL, True)
    return True


def _identity_rope(x: Any, *, offset: int = 0) -> Any:
    return x


def _normalize_nemotron_h_attention_contract(attention: Any) -> None:
    if type(attention).__name__ != "NemotronHAttention":
        return
    if not hasattr(attention, "n_heads") and hasattr(attention, "num_heads"):
        object.__setattr__(attention, "n_heads", int(attention.num_heads))
    if not hasattr(attention, "n_kv_heads") and hasattr(
        attention,
        "num_key_value_heads",
    ):
        object.__setattr__(
            attention,
            "n_kv_heads",
            int(attention.num_key_value_heads),
        )
    if not hasattr(attention, "rope"):
        object.__setattr__(attention, "rope", _identity_rope)


def _patch_nemotron_h_sdpa_wrapper_contract() -> bool:
    """Normalize Nemotron-H attention before vLLM Metal wraps it as SDPA."""

    try:
        wrapper_module = importlib.import_module(
            "vllm_metal.attention.impls.sdpa_wrapper"
        )
        sdpa_module = importlib.import_module("vllm_metal.attention.impls.sdpa")
    except Exception as exc:
        global LAST_ERROR
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return False
    wrapper_cls = getattr(wrapper_module, "SDPAPagedAttentionWrapper", None)
    if wrapper_cls is None:
        LAST_ERROR = "SDPAPagedAttentionWrapper unavailable"
        return False
    if getattr(wrapper_cls, SDPA_WRAPPER_PATCH_SENTINEL, False):
        return _patch_nemotron_h_sdpa_no_rope(sdpa_module)

    original_init = wrapper_cls.__init__

    def init_with_nemotron_h_contract(
        self: Any,
        inner: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        _normalize_nemotron_h_attention_contract(inner)
        original_init(self, inner, *args, **kwargs)

    wrapper_cls.__init__ = init_with_nemotron_h_contract
    wrapper_cls._audex_mac_original_init = original_init  # noqa: SLF001
    setattr(wrapper_cls, SDPA_WRAPPER_PATCH_SENTINEL, True)
    _patch_nemotron_h_sdpa_no_rope(sdpa_module)
    return True


def _patch_nemotron_h_sdpa_no_rope(sdpa_module: Any) -> bool:
    if getattr(sdpa_module, SDPA_NO_ROPE_PATCH_SENTINEL, False):
        return True

    original_apply_attention_rope = sdpa_module.apply_attention_rope

    def apply_attention_rope_with_nemotron_h(
        attn_module: Any,
        queries: Any,
        keys: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        if type(attn_module).__name__ == "NemotronHAttention":
            return queries, keys
        return original_apply_attention_rope(
            attn_module,
            queries,
            keys,
            *args,
            **kwargs,
        )

    sdpa_module.apply_attention_rope = apply_attention_rope_with_nemotron_h
    sdpa_module._audex_mac_original_apply_attention_rope = (  # noqa: SLF001
        original_apply_attention_rope
    )
    setattr(sdpa_module, SDPA_NO_ROPE_PATCH_SENTINEL, True)
    return True


class AudexProjectedAudioItems:
    """Parsed Audex audio payloads for vLLM request processing."""

    modality = "audio"

    def __init__(self, data: Any) -> None:
        self._items = self._normalize(data)

    @staticmethod
    def _normalize(data: Any) -> list[Mapping[str, Any]]:
        if _is_raw_audio_pair(data):
            data_items = [_raw_audio_mapping(data)]
        elif isinstance(data, Mapping):
            data_items = [data]
        else:
            data_items = [
                _raw_audio_mapping(item) if _is_raw_audio_pair(item) else item
                for item in list(data)
            ]
        if not data_items:
            return []
        for item in data_items:
            if not isinstance(item, Mapping):
                raise ValueError(
                    "Audex audio input must be a projected-audio mapping, "
                    "a raw-audio tuple, or a list of those; got "
                    f"{type(item).__name__}."
                )
            if not _is_raw_audio_mapping(item):
                AudexMultimodalAdapter._feature_value(
                    item,
                    "audex_projected_embeddings",
                    "projected_embeddings",
                    "audio_embeddings",
                    "audio_embeds",
                )
        return data_items

    def __len__(self) -> int:
        return self.get_count()

    def __iter__(self):
        return iter(self._items)

    def get_count(self) -> int:
        return len(self._items)

    def get(self, index: int) -> Mapping[str, Any]:
        return self._items[index]

    def get_all(self) -> list[Mapping[str, Any]]:
        return list(self._items)

    def get_item_for_hash(self, index: int) -> object:
        if _is_raw_audio_mapping(self._items[index]):
            samples = self._items[index]["audex_raw_audio_samples"]
            return {
                "audex_raw_audio_sample_rate": int(
                    self._items[index].get("sample_rate", SAMPLE_RATE)
                ),
                "audex_raw_audio_sample_count": len(samples),
                "object_id": id(samples),
            }
        embeddings = AudexMultimodalAdapter._feature_value(
            self._items[index],
            "audex_projected_embeddings",
            "projected_embeddings",
            "audio_embeddings",
            "audio_embeds",
        )
        shape = tuple(getattr(embeddings, "shape", ()))
        return {"audex_projected_audio_shape": shape, "object_id": id(embeddings)}

    def get_all_items_for_hash(self) -> list[object]:
        return [self.get_item_for_hash(index) for index in range(len(self))]

    def get_processor_data(self) -> Mapping[str, object]:
        return {}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {}


class AudexProjectedAudioDataParser:
    """vLLM multimodal parser for Audex preprojected audio embeddings."""

    def parse_mm_data(self, mm_data: Mapping[str, Any]) -> Any:
        from vllm.multimodal.parse import MultiModalDataItems

        mm_items = MultiModalDataItems()
        for modality, value in mm_data.items():
            if modality != "audio":
                raise ValueError(f"Unsupported Audex modality: {modality}")
            parsed = self._parse_audio_data(value)
            if parsed.get_count() > 0:
                mm_items[modality] = parsed
        return mm_items

    def _parse_audio_data(self, data: Any) -> AudexProjectedAudioItems:
        return AudexProjectedAudioItems(data)


class AudexProcessingInfo:
    """Minimal vLLM processing info for Audex projected audio."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    @property
    def model_id(self) -> str:
        return self.ctx.model_config.model

    @property
    def supported_mm_limits(self) -> Mapping[str, int | None]:
        return self.get_supported_mm_limits()

    @property
    def allowed_mm_limits(self) -> Mapping[str, int | None]:
        return self.get_supported_mm_limits()

    @property
    def skip_prompt_length_check(self) -> bool:
        return False

    @property
    def default_tok_params(self) -> Any:
        from vllm.renderers import TokenizeParams

        model_config = self.ctx.model_config
        encoder_config = model_config.encoder_config or {}
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=True,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        *,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"audio": DEFAULT_SOUND_EMBEDDING_SIZE}

    def get_tokenizer(self) -> Any:
        return self.ctx.get_tokenizer()

    def get_data_parser(self) -> AudexProjectedAudioDataParser:
        return AudexProjectedAudioDataParser()

    def parse_mm_data(
        self,
        mm_data: Mapping[str, Any],
        *,
        validate: bool = True,
    ) -> Any:
        mm_items = self.get_data_parser().parse_mm_data(mm_data)
        if validate and mm_items.get_count("audio", strict=False) > 1:
            raise ValueError("Audex-Mac supports one projected audio item per prompt.")
        return mm_items


class AudexDummyInputsBuilder:
    """Dummy inputs for vLLM profiling without invoking a HF audio processor."""

    def __init__(self, info: AudexProcessingInfo) -> None:
        self.info = info

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return SOUND_TOKEN * int(mm_counts.get("audio", 0))

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any],
    ) -> Mapping[str, list[Mapping[str, Any]]]:
        count = int(mm_counts.get("audio", 0))
        if count <= 0:
            return {"audio": []}
        return {
            "audio": [
                {
                    "audex_projected_embeddings": _DummyProjectedAudioEmbeddings(
                        max(1, min(seq_len, 1))
                    )
                }
                for _ in range(count)
            ]
        }

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any],
    ) -> Any:
        from vllm.multimodal.processing.inputs import ProcessorInputs

        dummy_text = self.get_dummy_text(mm_counts)
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        dummy_mm_items = self.info.parse_mm_data(dummy_mm_data, validate=False)
        return ProcessorInputs(
            prompt=dummy_text,
            mm_data_items=dummy_mm_items,
            tokenization_kwargs={"truncation": False},
        )


@dataclass(frozen=True, slots=True)
class _DummyProjectedAudioEmbeddings:
    tokens: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.tokens, 1)


@dataclass(slots=True)
class _AudioProjectionComponents:
    mx: Any
    encoder_config: Any
    encoder_weights: dict[str, Any]
    projector_config: Any
    projector_weights: dict[str, Any]
    preprocessor_path: Path

    @classmethod
    def load(cls, model_path: Path) -> _AudioProjectionComponents:
        import mlx.core as mx

        from audex_mac.audio_encoder import (
            load_audio_encoder_config,
            load_audio_encoder_weights_mlx,
        )
        from audex_mac.audio_projector import (
            load_audio_projector_config,
            load_audio_projector_weights_mlx,
        )

        components = cls(
            mx=mx,
            encoder_config=load_audio_encoder_config(model_path),
            encoder_weights=load_audio_encoder_weights_mlx(model_path),
            projector_config=load_audio_projector_config(model_path),
            projector_weights=load_audio_projector_weights_mlx(model_path),
            preprocessor_path=model_path / "audio_preprocessor",
        )
        return components

    def extract_features(self, clips: Any) -> Any:
        from audex_mac.audio_features import extract_audex_input_features

        return extract_audex_input_features(
            clips,
            preprocessor_path=self.preprocessor_path,
        )

    def encode_features(self, feature_array: Any) -> Any:
        from audex_mac.audio_encoder import encode_audio_features_mlx

        return encode_audio_features_mlx(
            feature_array,
            self.encoder_weights,
            self.encoder_config,
        )

    def project_hidden_states(self, hidden_states: Any) -> Any:
        from audex_mac.audio_projector import project_audio_hidden_states_mlx

        return project_audio_hidden_states_mlx(
            hidden_states,
            self.projector_weights,
            self.projector_config,
        )


def _is_raw_audio_pair(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return False
    if len(value) != 2:
        return False
    _samples, sample_rate = value
    return isinstance(sample_rate, int)


def _raw_audio_mapping(value: Sequence[Any]) -> Mapping[str, Any]:
    samples, sample_rate = value
    return {
        "audex_raw_audio_samples": samples,
        "sample_rate": int(sample_rate),
    }


def _is_raw_audio_mapping(item: Mapping[str, Any]) -> bool:
    return "audex_raw_audio_samples" in item


def _audio_samples_to_list(samples: Any) -> list[Any]:
    tolist = getattr(samples, "tolist", None)
    if callable(tolist):
        return list(tolist())
    return list(samples)


def _audio_sample_count(samples: Any) -> int:
    shape = getattr(samples, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    try:
        return len(samples)
    except TypeError:
        return len(_audio_samples_to_list(samples))


def raw_audio_num_embeddings(
    samples: Any,
    *,
    sample_rate: int,
    trim_padded: bool | None = None,
) -> int:
    if trim_padded is None:
        trim_padded = os.environ.get(AUDIO_EMBEDDING_TRIM_ENV) == "1"
    if trim_padded:
        sample_count = max(1, _audio_sample_count(samples))
        clip_samples = sample_rate * 30
        full_clips, tail_samples = divmod(sample_count, clip_samples)
        full_embeddings = full_clips * DEFAULT_SOUND_EMBEDDING_SIZE
        if tail_samples == 0:
            return max(DEFAULT_SOUND_EMBEDDING_SIZE, full_embeddings)
        tail_embeddings = math.ceil(
            tail_samples * DEFAULT_SOUND_EMBEDDING_SIZE / clip_samples
        )
        return full_embeddings + max(
            1,
            min(DEFAULT_SOUND_EMBEDDING_SIZE, tail_embeddings),
        )
    return audio_embedding_count(
        audio_clip_count(_audio_sample_count(samples), sample_rate=sample_rate),
        embeddings_per_clip=DEFAULT_SOUND_EMBEDDING_SIZE,
    )


def _torch_audio_samples(samples: Any) -> Any:
    try:
        import torch
    except ImportError:
        return _audio_samples_to_list(samples)

    if isinstance(samples, torch.Tensor):
        return samples.detach().cpu().to(dtype=torch.float32).contiguous()
    return torch.tensor(_audio_samples_to_list(samples), dtype=torch.float32)


def _torch_sample_rate(sample_rate: int) -> Any:
    try:
        import torch
    except ImportError:
        return int(sample_rate)

    return torch.tensor([int(sample_rate)], dtype=torch.int32)


def _shared_audio_field(field_config_cls: Any) -> Any:
    try:
        return field_config_cls.shared(
            "audio",
            batch_size=1,
            keep_on_cpu=True,
        )
    except TypeError:
        return field_config_cls.shared("audio", batch_size=1)


def _ensure_projected_audio_placeholder_run(
    prompt_ids: list[int],
    *,
    sound_token_id: int,
    sound_start_token_id: int,
    sound_end_token_id: int,
    num_embeddings: int,
) -> list[int]:
    try:
        offset, length = find_projected_audio_placeholder_range(
            prompt_ids,
            sound_token_id=sound_token_id,
            num_embeddings=num_embeddings,
        )
        before = prompt_ids[offset - 1] if offset > 0 else None
        after_index = offset + length
        after = prompt_ids[after_index] if after_index < len(prompt_ids) else None
        if int(before or -1) == int(sound_start_token_id) and int(after or -1) == int(
            sound_end_token_id
        ):
            return prompt_ids
        return [
            *prompt_ids[:offset],
            int(sound_start_token_id),
            *([int(sound_token_id)] * int(num_embeddings)),
            int(sound_end_token_id),
            *prompt_ids[offset + length :],
        ]
    except ValueError:
        pass

    for index, token_id in enumerate(prompt_ids):
        if int(token_id) != int(sound_token_id):
            continue
        return [
            *prompt_ids[:index],
            int(sound_start_token_id),
            *([int(sound_token_id)] * int(num_embeddings)),
            int(sound_end_token_id),
            *prompt_ids[index + 1 :],
        ]
    raise ValueError(
        "Audex prompt does not contain a sound placeholder token "
        f"for sound_token_id={sound_token_id}."
    )


class AudexProjectedAudioProcessor:
    """vLLM processor that forwards raw Audex audio or projected embeddings."""

    def __init__(
        self,
        info: AudexProcessingInfo,
        dummy_inputs: AudexDummyInputsBuilder,
        *,
        cache: Any | None = None,
    ) -> None:
        self.info = info
        self.dummy_inputs = dummy_inputs
        self.cache = cache
        self._audio_components: _AudioProjectionComponents | None = None

    def _get_mm_fields_config(
        self,
        hf_inputs: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, Any]:
        return {}

    def _get_prompt_updates(
        self,
        mm_items: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: Any,
    ) -> tuple[()]:
        return ()

    def apply(self, inputs: Any, timing_ctx: Any) -> Any:
        from vllm.inputs import mm_input
        from vllm.multimodal.inputs import (
            MultiModalFieldConfig,
            MultiModalKwargsItem,
            MultiModalKwargsItems,
        )

        tokenizer = self.info.get_tokenizer()
        prompt_ids = _tokenize_prompt(tokenizer, inputs.prompt)
        sound_token_id = _token_id(tokenizer, SOUND_TOKEN)
        sound_start_token_id = _token_id(tokenizer, SOUND_START_TOKEN)
        sound_end_token_id = _token_id(tokenizer, SOUND_END_TOKEN)
        audio_items = inputs.mm_data_items.get_items("audio", AudexProjectedAudioItems)

        mm_kwargs_items: list[Any] = []
        mm_hashes: dict[str, list[str]] = {"audio": []}
        mm_placeholders: dict[str, list[Any]] = {"audio": []}
        for item_index, item in enumerate(audio_items.get_all()):
            if _is_raw_audio_mapping(item):
                samples = item["audex_raw_audio_samples"]
                sample_rate = int(item.get("sample_rate", SAMPLE_RATE))
                if sample_rate != SAMPLE_RATE:
                    raise ValueError(
                        f"Audex raw audio must be {SAMPLE_RATE} Hz, got "
                        f"{sample_rate}."
                    )
                num_embeddings = int(
                    item.get("audex_raw_audio_num_embeddings")
                    or raw_audio_num_embeddings(samples, sample_rate=sample_rate)
                )
                prompt_ids = _ensure_projected_audio_placeholder_run(
                    prompt_ids,
                    sound_token_id=sound_token_id,
                    sound_start_token_id=sound_start_token_id,
                    sound_end_token_id=sound_end_token_id,
                    num_embeddings=num_embeddings,
                )
                feature = build_raw_audio_feature_spec(
                    raw_audio_samples=samples,
                    sample_rate=sample_rate,
                    prompt_token_ids=prompt_ids,
                    sound_token_id=sound_token_id,
                    num_embeddings=num_embeddings,
                    identifier=_raw_audio_identifier(
                        inputs,
                        item_index,
                        samples,
                        sample_rate,
                        num_embeddings,
                    ),
                )
                _debug(
                    "processor raw audio feature "
                    f"index={item_index} samples={_audio_sample_count(samples)} "
                    f"sample_rate={sample_rate} "
                    f"prompt_tokens={len(prompt_ids)} "
                    f"offset={feature.mm_position.offset} "
                    f"length={feature.mm_position.length}"
                )
                shared_field = _shared_audio_field(MultiModalFieldConfig)
                mm_kwargs_items.append(
                    MultiModalKwargsItem(
                        {
                            "audex_raw_audio_samples": shared_field.build_elems(
                                "audex_raw_audio_samples",
                                _torch_audio_samples(samples),
                            )[0],
                            "sample_rate": shared_field.build_elems(
                                "sample_rate",
                                _torch_sample_rate(sample_rate),
                            )[0],
                            "audex_raw_audio_num_embeddings": (
                                shared_field.build_elems(
                                    "audex_raw_audio_num_embeddings",
                                    _torch_sample_rate(num_embeddings),
                                )[0]
                            ),
                        }
                    )
                )
                mm_hashes["audio"].append(feature.identifier)
                mm_placeholders["audio"].append(feature.mm_position)
                continue

            embeddings = self._project_or_get_embeddings(item)
            prompt_ids = _ensure_projected_audio_placeholder_run(
                prompt_ids,
                sound_token_id=sound_token_id,
                sound_start_token_id=sound_start_token_id,
                sound_end_token_id=sound_end_token_id,
                num_embeddings=projected_audio_embedding_count(embeddings),
            )
            feature = build_projected_audio_feature_spec(
                projected_embeddings=embeddings,
                prompt_token_ids=prompt_ids,
                sound_token_id=sound_token_id,
                identifier=_projected_audio_identifier(
                    inputs,
                    item_index,
                    embeddings,
                ),
            )
            _debug(
                "processor feature "
                f"index={item_index} embeddings_shape="
                f"{tuple(getattr(embeddings, 'shape', ()))!r} "
                f"prompt_tokens={len(prompt_ids)} "
                f"offset={feature.mm_position.offset} "
                f"length={feature.mm_position.length}"
            )
            field_elem = MultiModalFieldConfig.shared(
                "audio",
                batch_size=1,
            ).build_elems("audex_projected_embeddings", embeddings)[0]
            mm_kwargs_items.append(
                MultiModalKwargsItem({"audex_projected_embeddings": field_elem})
            )
            mm_hashes["audio"].append(feature.identifier)
            mm_placeholders["audio"].append(feature.mm_position)

        return mm_input(
            prompt_token_ids=prompt_ids,
            mm_kwargs=MultiModalKwargsItems({"audio": mm_kwargs_items}),
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )

    def _project_or_get_embeddings(self, item: Mapping[str, Any]) -> Any:
        if _is_raw_audio_mapping(item):
            return self._project_raw_audio(item)
        return AudexMultimodalAdapter._feature_value(
            item,
            "audex_projected_embeddings",
            "projected_embeddings",
            "audio_embeddings",
            "audio_embeds",
        )

    def _project_raw_audio(self, item: Mapping[str, Any]) -> Any:
        samples = _audio_samples_to_list(item["audex_raw_audio_samples"])
        sample_rate = int(item.get("sample_rate", SAMPLE_RATE))
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Audex raw audio must be {SAMPLE_RATE} Hz, got {sample_rate}."
            )
        components = self._load_audio_components()
        clips = prepare_audex_pcm_clips(samples, sample_rate=sample_rate)
        features = components.extract_features(clips)
        feature_array = components.mx.array(features.input_features)
        feature_array = feature_array.astype(
            components.encoder_weights["conv1.weight"].dtype
        )
        encoder_hidden = components.encode_features(feature_array)
        projected = components.project_hidden_states(encoder_hidden)
        if projected.ndim == 3:
            clips_count, tokens_per_clip, hidden = projected.shape
            projected = components.mx.reshape(
                projected,
                (int(clips_count) * int(tokens_per_clip), int(hidden)),
            )
        components.mx.eval(projected)
        return projected

    def _load_audio_components(self) -> _AudioProjectionComponents:
        if self._audio_components is None:
            self._audio_components = _AudioProjectionComponents.load(
                Path(self.info.model_id)
            )
        return self._audio_components

    def __call__(
        self,
        prompt: str,
        mm_items: Any,
        mm_uuid_items: Any | None = None,
        hf_processor_mm_kwargs: Mapping[str, object] | None = None,
    ) -> Any:
        from vllm.multimodal.processing.inputs import ProcessorInputs
        from vllm.multimodal.processing.processor import TimingContext

        return self.apply(
            ProcessorInputs(
                prompt=prompt,
                mm_data_items=mm_items,
                mm_uuid_items=mm_uuid_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs or {},
            ),
            TimingContext(enabled=False),
        )


def _tokenize_prompt(tokenizer: Any, prompt: str | list[int]) -> list[int]:
    if isinstance(prompt, list):
        return list(prompt)
    try:
        return list(tokenizer.encode(prompt, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(prompt))


def _token_id(tokenizer: Any, token: str) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        token_id = convert(token)
        if token_id is not None:
            return int(token_id)
    vocab = tokenizer.get_vocab()
    if token not in vocab:
        raise ValueError(f"Audex tokenizer does not contain required token {token!r}.")
    return int(vocab[token])


def _projected_audio_identifier(inputs: Any, item_index: int, embeddings: Any) -> str:
    mm_uuid_items = getattr(inputs, "mm_uuid_items", None) or {}
    audio_uuids = mm_uuid_items.get("audio") if hasattr(mm_uuid_items, "get") else None
    if audio_uuids and audio_uuids[item_index]:
        return str(audio_uuids[item_index])
    shape = "x".join(str(part) for part in getattr(embeddings, "shape", ()))
    return f"audex-projected-audio-{item_index}-{shape}-{id(embeddings):x}"


def _raw_audio_identifier(
    inputs: Any,
    item_index: int,
    samples: Any,
    sample_rate: int,
    num_embeddings: int,
) -> str:
    mm_uuid_items = getattr(inputs, "mm_uuid_items", None) or {}
    audio_uuids = mm_uuid_items.get("audio") if hasattr(mm_uuid_items, "get") else None
    if audio_uuids and audio_uuids[item_index]:
        return str(audio_uuids[item_index])
    return (
        f"audex-raw-audio-{item_index}-"
        f"{_audio_sample_count(samples)}x{int(sample_rate)}-"
        f"e{int(num_embeddings)}-{id(samples):x}"
    )


class AudexMultimodalAdapter:
    """vLLM Metal adapter for Audex native audio embeddings.

    This patch installs the Audex selection seam and accepts Audex raw audio
    request payloads by projecting them through Audex's MLX audio tower before
    forwarding projected embeddings into vLLM Metal's multimodal path.
    """

    forward_ready: bool = True
    requires_explicit_positions: bool = True
    _SUPPORTED_EMBEDS_KWARGS: tuple[str, ...] = (
        "input_embeddings",
        "inputs_embeds",
    )

    def __init__(
        self,
        *,
        model: Any,
        text_model: Any,
        embeds_kwarg: str,
        embed_tokens_path: str,
        call_parameters: set[str],
        model_path: Path | None = None,
    ) -> None:
        self._model = model
        self._text_model = text_model
        self._embeds_kwarg = embeds_kwarg
        self._embed_tokens_path = embed_tokens_path
        self._call_parameters = call_parameters
        self._model_path = model_path
        self._audio_components: _AudioProjectionComponents | None = None
        self._hybrid_cache: list[Any] | None = None
        self._hybrid_cache_used = False

    @classmethod
    def from_loaded_model(
        cls,
        model: Any,
        *,
        model_path: Path | str | None = None,
    ) -> AudexMultimodalAdapter:
        text_model = cls._resolve_text_model(model)
        call_parameters = cls._detect_call_parameters(text_model)
        embeds_kwarg = cls._detect_embeds_kwarg(call_parameters)
        embed_tokens, embed_tokens_path = cls._resolve_embed_tokens(text_model)
        adapter = cls(
            model=model,
            text_model=text_model,
            embeds_kwarg=embeds_kwarg,
            embed_tokens_path=embed_tokens_path,
            call_parameters=call_parameters,
            model_path=Path(model_path) if model_path is not None else None,
        )
        adapter._embed_tokens = embed_tokens
        if os.environ.get(EAGER_AUDIO_COMPONENTS_ENV) == "1":
            adapter._load_audio_components()
        return adapter

    @staticmethod
    def _resolve_text_model(model: Any) -> Any:
        if hasattr(model, "language_model"):
            return model.language_model
        return model

    @staticmethod
    def _resolve_embed_tokens(text_model: Any) -> tuple[Any, str]:
        candidates = (
            (
                "model.embed_tokens",
                getattr(getattr(text_model, "model", None), "embed_tokens", None),
            ),
            (
                "model.model.embed_tokens",
                getattr(
                    getattr(getattr(text_model, "model", None), "model", None),
                    "embed_tokens",
                    None,
                ),
            ),
            ("embed_tokens", getattr(text_model, "embed_tokens", None)),
        )
        for label, embed_tokens in candidates:
            if callable(embed_tokens):
                return embed_tokens, label
        raise RuntimeError(
            "Audex vLLM adapter could not find a callable embed_tokens path on "
            f"{type(text_model).__module__}.{type(text_model).__name__}."
        )

    @staticmethod
    def _detect_call_parameters(text_model: Any) -> set[str]:
        try:
            signature = inspect.signature(text_model.__call__)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot inspect Audex text model __call__ signature: {exc}"
            ) from exc
        return set(signature.parameters)

    @classmethod
    def _detect_embeds_kwarg(cls, parameters: set[str]) -> str:
        for candidate in cls._SUPPORTED_EMBEDS_KWARGS:
            if candidate in parameters:
                return candidate
        raise RuntimeError(
            "Audex text model __call__ accepts none of "
            f"{cls._SUPPORTED_EMBEDS_KWARGS}; got parameters "
            f"{sorted(parameters)}."
        )

    def text_model(self) -> Any:
        return self._text_model

    def embed_tokens(self, input_ids: Any) -> Any:
        return self._embed_tokens(input_ids)

    def encode_multimodal(self, features: list[Any]) -> list[AudexAudioEncodeResult]:
        _debug(f"encode_multimodal features={len(features)}")
        return [self._encode_audio_feature(feature) for feature in features]

    def _encode_audio_feature(self, feature: Any) -> AudexAudioEncodeResult:
        modality = getattr(feature, "modality", None)
        if modality not in {"audio", "sound"}:
            raise ValueError(
                "Audex vLLM adapter only supports audio/sound features; "
                f"got modality={modality!r}."
            )
        data = getattr(feature, "data", None)
        if data is None:
            raise ValueError("Audex audio feature data is required.")
        if self._has_feature_key(data, "audex_raw_audio_samples"):
            embeddings = self._project_raw_audio_feature(data)
        else:
            embeddings = self._as_mlx(
                self._feature_value(
                    data,
                    "audex_projected_embeddings",
                    "projected_embeddings",
                    "audio_embeddings",
                    "audio_embeds",
                )
            )
        return self._encode_projected_embeddings(feature, embeddings)

    def _encode_projected_embeddings(
        self,
        feature: Any,
        embeddings: Any,
    ) -> AudexAudioEncodeResult:
        embeddings = self._as_mlx(embeddings)
        try:
            import mlx.core as mx

            mx.eval(embeddings)
        except Exception:
            pass
        if embeddings.ndim == 3:
            clips, tokens_per_clip, hidden = embeddings.shape
            embeddings = embeddings.reshape(
                int(clips) * int(tokens_per_clip),
                int(hidden),
            )
        try:
            import mlx.core as mx

            mx.eval(embeddings)
        except Exception:
            pass
        if embeddings.ndim != 2:
            raise ValueError(
                "Audex projected audio embeddings must be 2D or 3D, got "
                f"shape={embeddings.shape}."
            )
        expected_embeds = self._feature_embed_count(feature)
        _debug(
            "encode feature "
            f"id={getattr(feature, 'identifier', None)!r} "
            f"shape={tuple(getattr(embeddings, 'shape', ()))!r} "
            f"offset={getattr(getattr(feature, 'mm_position', None), 'offset', None)} "
            f"length={getattr(getattr(feature, 'mm_position', None), 'length', None)}"
        )
        if expected_embeds is not None and int(embeddings.shape[0]) != expected_embeds:
            raise ValueError(
                "Audex projected audio embedding count mismatch: "
                f"feature expects {expected_embeds}, embeddings have "
                f"{int(embeddings.shape[0])}."
            )
        return AudexAudioEncodeResult(hidden_states=embeddings)

    def _project_raw_audio_feature(self, data: Any) -> Any:
        projection_started = time.perf_counter()
        samples = self._feature_value(data, "audex_raw_audio_samples")
        sample_rate = self._scalar_int(
            self._feature_value(data, "sample_rate", "audex_sample_rate")
        )
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Audex raw audio must be {SAMPLE_RATE} Hz, got {sample_rate}."
            )
        components = self._load_audio_components()
        components_ready_at = time.perf_counter()
        clips = prepare_audex_pcm_clips(
            _audio_samples_to_list(samples),
            sample_rate=sample_rate,
        )
        clips_ready_at = time.perf_counter()
        features = components.extract_features(clips)
        features_ready_at = time.perf_counter()
        feature_array = components.mx.array(features.input_features)
        feature_array = feature_array.astype(
            components.encoder_weights["conv1.weight"].dtype
        )
        expected_embeddings = self._scalar_int(
            self._feature_value(data, "audex_raw_audio_num_embeddings")
        )
        full_embeddings = int(components.encoder_config.max_source_positions) // 2
        encoder_embeddings = min(expected_embeddings, full_embeddings)
        if encoder_embeddings < full_embeddings:
            feature_array = feature_array[:, :, : encoder_embeddings * 4]
        array_ready_at = time.perf_counter()
        encoder_hidden = components.encode_features(feature_array)
        projected = components.project_hidden_states(encoder_hidden)
        if projected.ndim == 3:
            clips_count, tokens_per_clip, hidden = projected.shape
            projected = components.mx.reshape(
                projected,
                (int(clips_count) * int(tokens_per_clip), int(hidden)),
            )
        if int(projected.shape[0]) > expected_embeddings:
            projected = projected[:expected_embeddings]
        graph_ready_at = time.perf_counter()
        components.mx.eval(projected)
        evaluated_at = time.perf_counter()
        _debug(
            "projected raw audio in EngineCore "
            f"samples={_audio_sample_count(samples)} sample_rate={sample_rate} "
            f"shape={tuple(getattr(projected, 'shape', ()))!r} "
            f"encoder_embeddings={encoder_embeddings} "
            "timings_ms="
            f"load:{(components_ready_at - projection_started) * 1000:.1f},"
            f"clips:{(clips_ready_at - components_ready_at) * 1000:.1f},"
            f"features:{(features_ready_at - clips_ready_at) * 1000:.1f},"
            f"array:{(array_ready_at - features_ready_at) * 1000:.1f},"
            f"graph:{(graph_ready_at - array_ready_at) * 1000:.1f},"
            f"eval:{(evaluated_at - graph_ready_at) * 1000:.1f},"
            f"total:{(evaluated_at - projection_started) * 1000:.1f}"
        )
        return projected

    def _load_audio_components(self) -> _AudioProjectionComponents:
        if self._model_path is None:
            raise RuntimeError(
                "Audex raw-audio vLLM projection requires the full model path."
            )
        if self._audio_components is None:
            self._audio_components = _AudioProjectionComponents.load(self._model_path)
        return self._audio_components

    @staticmethod
    def _has_feature_key(data: Any, key: str) -> bool:
        try:
            return key in data
        except TypeError:
            return False

    @staticmethod
    def _scalar_int(value: Any) -> int:
        item = getattr(value, "item", None)
        if callable(item):
            return int(item())
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            listed = tolist()
            if isinstance(listed, list):
                return int(listed[0])
            return int(listed)
        return int(value)

    @staticmethod
    def _feature_embed_count(feature: Any) -> int | None:
        mm_position = getattr(feature, "mm_position", None)
        if mm_position is None:
            return None
        get_num_embeds = getattr(mm_position, "get_num_embeds", None)
        if callable(get_num_embeds):
            return int(get_num_embeds())
        length = getattr(mm_position, "length", None)
        return int(length) if length is not None else None

    @staticmethod
    def _feature_value(data: Any, *keys: str) -> Any:
        for key in keys:
            try:
                value = data[key]
            except (KeyError, TypeError):
                continue
            return getattr(value, "data", value)
        available = sorted(data.keys()) if hasattr(data, "keys") else type(data)
        raise ValueError(
            "Audex audio feature data must include one of "
            f"{keys}; available={available}."
        )

    @staticmethod
    def _as_mlx(value: Any) -> Any:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx

                return torch_to_mlx(value)
        except Exception:
            pass
        if hasattr(value, "shape") and hasattr(value, "ndim"):
            return value
        try:
            import mlx.core as mx

            return mx.array(value)
        except Exception as exc:
            raise RuntimeError(
                "Audex vLLM adapter requires MLX-compatible projected audio "
                "embeddings."
            ) from exc

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[Any],
    ) -> tuple[Any, int]:
        import mlx.core as mx

        positions = mx.arange(len(input_tokens), dtype=mx.int32)[None, None, :]
        return mx.broadcast_to(positions, (3, 1, len(input_tokens))), 0

    @staticmethod
    def _clear_paged_context_segment_positions() -> None:
        try:
            from vllm_metal.attention.context import get_context
        except Exception:
            return
        ctx = get_context()
        if ctx is None:
            return
        segment_positions = getattr(ctx, "segment_positions", None)
        if segment_positions is None:
            return
        ctx.segment_positions = [None for _ in segment_positions]

    def call_lm(
        self,
        input_ids: Any,
        inputs_embeds: Any,
        cache: list[Any],
        position_ids: Any,
        *,
        visual_pos_masks: Any | None = None,
        deepstack_visual_embeds: Any | None = None,
    ) -> Any:
        if deepstack_visual_embeds is not None:
            raise RuntimeError("Audex does not use deepstack visual embeddings.")
        self._clear_paged_context_segment_positions()
        cache = self._prepare_hybrid_cache(
            cache,
            input_ids=input_ids,
            visual_pos_masks=visual_pos_masks,
        )
        try:
            import mlx.core as mx

            mx.eval(inputs_embeds)
        except Exception:
            pass
        _debug(
            "call_lm "
            f"input_ids_shape={tuple(getattr(input_ids, 'shape', ()))!r} "
            f"inputs_embeds_shape={tuple(getattr(inputs_embeds, 'shape', ()))!r} "
            f"embeds_kwarg={self._embeds_kwarg!r} "
            f"embed_tokens_path={self._embed_tokens_path!r} "
            f"call_parameters={sorted(self._call_parameters)!r}"
        )
        kwargs = {self._embeds_kwarg: inputs_embeds}
        if "cache" in self._call_parameters:
            kwargs["cache"] = cache
        if "position_ids" in self._call_parameters:
            kwargs["position_ids"] = position_ids
        return self._text_model(input_ids, **kwargs)

    def _prepare_hybrid_cache(
        self,
        cache: list[Any],
        *,
        input_ids: Any,
        visual_pos_masks: Any | None,
    ) -> list[Any]:
        if not self._model_cache_has_arrays():
            return cache
        seq_len = self._input_sequence_length(input_ids)
        if self._hybrid_cache is None:
            self._hybrid_cache = self._make_model_cache()
            self._hybrid_cache_used = False
            _debug(
                "initialized Audex hybrid cache "
                f"seq_len={seq_len} visual={visual_pos_masks is not None}"
            )
        elif seq_len > 1 and self._hybrid_cache_used:
            self._replace_hybrid_cache(
                seq_len=seq_len,
                visual_pos_masks=visual_pos_masks,
            )
        if self._hybrid_cache is None:
            return cache

        hybrid: list[Any] = []
        for index, provided_cache in enumerate(cache):
            model_cache = (
                self._hybrid_cache[index] if index < len(self._hybrid_cache) else None
            )
            if type(model_cache).__name__ == "ArraysCache":
                hybrid.append(model_cache)
            else:
                hybrid.append(provided_cache)
        self._hybrid_cache_used = True
        return hybrid

    def _replace_hybrid_cache(
        self,
        *,
        seq_len: int,
        visual_pos_masks: Any | None,
    ) -> None:
        old_cache = self._hybrid_cache
        self._hybrid_cache = None
        self._hybrid_cache_used = False
        del old_cache
        gc.collect()
        self._clear_mlx_cache()
        self._hybrid_cache = self._make_model_cache()
        _debug(
            "replaced Audex hybrid cache "
            f"seq_len={seq_len} visual={visual_pos_masks is not None}"
        )

    @staticmethod
    def _clear_mlx_cache() -> None:
        try:
            import mlx.core as mx
        except Exception:
            return
        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            with suppress(Exception):
                clear_cache()

    def _make_model_cache(self) -> list[Any] | None:
        make_cache = getattr(self._text_model, "make_cache", None)
        if not callable(make_cache):
            make_cache = getattr(self._model, "make_cache", None)
        if not callable(make_cache):
            return None
        cache = list(make_cache())
        return cache

    def _model_cache_has_arrays(self) -> bool:
        probe = self._hybrid_cache
        if probe is None:
            probe = self._make_model_cache()
            self._hybrid_cache = probe
        if not probe:
            return False
        return any(type(item).__name__ == "ArraysCache" for item in probe)

    @staticmethod
    def _input_sequence_length(input_ids: Any) -> int:
        shape = getattr(input_ids, "shape", None)
        if shape is not None:
            if len(shape) >= 2:
                return int(shape[1])
            if len(shape) == 1:
                return int(shape[0])
        try:
            return len(input_ids[0])
        except Exception:
            return len(input_ids)


def is_audex_hf_config(hf_config: Any) -> bool:
    model_type = str(getattr(hf_config, "model_type", ""))
    if model_type in AUDEX_MODEL_TYPES:
        return True
    architectures = getattr(hf_config, "architectures", ()) or ()
    return any(architecture in AUDEX_ARCHITECTURES for architecture in architectures)


def patch_default_model_adapter() -> bool:
    global LAST_ERROR
    LAST_ERROR = None
    try:
        from vllm_metal.v1.model_adapter import DefaultModelAdapter
    except Exception as exc:
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return False

    try:
        _patch_audex_text_backbone_selection(DefaultModelAdapter)
        _patch_audex_generation_model_install()
        _patch_nemotron_h_mixer_attention_detection()
        _patch_nemotron_h_sdpa_wrapper_contract()
    except Exception as exc:
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return False

    if getattr(DefaultModelAdapter, PATCH_SENTINEL, False):
        return (
            patch_vllm_audex_processor()
            and _patch_vllm_renderer_mm_state()
            and _patch_vllm_metal_non_paged_multimodal_prefill()
            and _patch_nemotron_h_mixer_attention_detection()
            and _patch_nemotron_h_sdpa_wrapper_contract()
        )

    original_build_multimodal_adapter = DefaultModelAdapter.build_multimodal_adapter

    def build_multimodal_adapter(self: Any, model: Any, hf_config: Any) -> Any:
        if is_audex_hf_config(hf_config):
            return AudexMultimodalAdapter.from_loaded_model(model)
        return original_build_multimodal_adapter(self, model, hf_config)

    DefaultModelAdapter.build_multimodal_adapter = build_multimodal_adapter
    setattr(DefaultModelAdapter, PATCH_SENTINEL, True)
    DefaultModelAdapter._audex_mac_original_build_multimodal_adapter = (  # noqa: SLF001
        original_build_multimodal_adapter
    )
    return (
        patch_vllm_audex_processor()
        and _patch_vllm_renderer_mm_state()
        and _patch_vllm_metal_non_paged_multimodal_prefill()
        and _patch_nemotron_h_mixer_attention_detection()
        and _patch_nemotron_h_sdpa_wrapper_contract()
    )


def _patch_vllm_metal_non_paged_multimodal_prefill() -> bool:
    """Allow Audex projected-audio prefill on vLLM Metal's fast non-paged path."""

    global LAST_ERROR
    try:
        model_runner_module = importlib.import_module("vllm_metal.v1.model_runner")
    except ModuleNotFoundError:
        return True
    except Exception as exc:
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return False

    runner_cls = getattr(model_runner_module, "MetalModelRunner", None)
    if runner_cls is None:
        LAST_ERROR = "vllm_metal.v1.model_runner.MetalModelRunner unavailable"
        return False
    if getattr(runner_cls, NON_PAGED_MM_PREFILL_PATCH_SENTINEL, False):
        return True

    original_reject_scheduled_encoder_inputs = (
        runner_cls._reject_scheduled_encoder_inputs
    )
    original_handle_new_requests = runner_cls._handle_new_requests
    original_prefill_single = runner_cls._prefill_single

    def reject_scheduled_encoder_inputs_with_non_paged_audex(
        self: Any,
        scheduled_encoder_inputs: dict[str, list[int]],
    ) -> None:
        if (
            scheduled_encoder_inputs
            and getattr(self, "_paged_attention_runtime", None) is None
            and _runner_has_audex_adapter(self)
        ):
            self._run_vision_encoders(scheduled_encoder_inputs)
            return
        return original_reject_scheduled_encoder_inputs(
            self,
            scheduled_encoder_inputs,
        )

    def handle_new_requests_with_non_paged_audex_mm(
        self: Any,
        batch: Any,
        new_reqs: list[Any],
        scheduler_output: Any,
    ) -> None:
        if (
            getattr(self, "_paged_attention_runtime", None) is not None
            or not _runner_has_audex_adapter(self)
            or not _new_requests_include_multimodal(self, new_reqs)
        ):
            return original_handle_new_requests(
                self,
                batch,
                new_reqs,
                scheduler_output,
            )

        batch.new_reqs_by_id = {req.req_id: req for req in new_reqs}
        for new_req in new_reqs:
            req_id = new_req.req_id
            pooling_params = new_req.pooling_params
            model_runner_module.validate_pooling_request(
                new_req,
                self.model_config,
                paged_attention_enabled=False,
            )

            token_ids = new_req.prompt_token_ids or []
            sampling_params = (
                new_req.sampling_params or model_runner_module.SamplingParams()
            )
            lora_id = model_runner_module._lora_id_from_request_data(new_req)
            if new_req.lora_request is not None:
                self._lora.add_adapter(new_req.lora_request)

            if not token_ids:
                batch.add_output(req_id, [0])
                continue

            generator = model_runner_module._create_request_generator(
                self.device,
                sampling_params,
            )
            if self._is_mm_request(req_id):
                prefix_length, prefix_cache = _verified_audio_prefix_snapshot(
                    self,
                    model_runner_module,
                    sampling_params,
                    token_ids,
                )
                next_token, cache, logprobs, mm_delta = (
                    _prefill_single_non_paged_audex_mm(
                        self,
                        model_runner_module,
                        req_id,
                        token_ids,
                        sampling_params,
                        generator,
                        prefix_length=prefix_length,
                        prefix_cache=prefix_cache,
                    )
                )
            else:
                next_token, cache, logprobs = original_prefill_single(
                    self,
                    token_ids,
                    sampling_params,
                    generator=generator,
                )
                mm_delta = None

            batch.add_output(req_id, [next_token], logprobs)
            self._request_states[req_id] = model_runner_module.RequestState(
                token_ids=list(token_ids) + [next_token],
                prompt_len=len(token_ids),
                cache=cache,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                generated_tokens=1,
                block_ids=[],
                lora_id=lora_id,
                mrope_position_delta=mm_delta,
            )

    runner_cls._reject_scheduled_encoder_inputs = (
        reject_scheduled_encoder_inputs_with_non_paged_audex
    )
    runner_cls._handle_new_requests = handle_new_requests_with_non_paged_audex_mm
    runner_cls._audex_mac_original_reject_scheduled_encoder_inputs = (  # noqa: SLF001
        original_reject_scheduled_encoder_inputs
    )
    runner_cls._audex_mac_original_handle_new_requests = (  # noqa: SLF001
        original_handle_new_requests
    )
    setattr(runner_cls, NON_PAGED_MM_PREFILL_PATCH_SENTINEL, True)
    return True


def _runner_has_audex_adapter(runner: Any) -> bool:
    adapter = getattr(runner, "_multimodal_adapter", None)
    return isinstance(adapter, AudexMultimodalAdapter) and bool(
        getattr(adapter, "forward_ready", False)
    )


def _new_requests_include_multimodal(runner: Any, new_reqs: list[Any]) -> bool:
    return any(runner._is_mm_request(req.req_id) for req in new_reqs)


def _prefill_single_non_paged_audex_mm(
    runner: Any,
    model_runner_module: Any,
    req_id: str,
    token_ids: list[int],
    sampling_params: Any,
    generator: Any,
    *,
    prefix_length: int = 0,
    prefix_cache: list[Any] | None = None,
) -> tuple[int, list[Any], Any, int]:
    adapter = runner._multimodal_adapter
    encoder_cache = runner.encoder_cache
    if adapter is None or encoder_cache is None:
        raise RuntimeError("Audex non-paged multimodal prefill needs encoder cache.")

    mm_features = sorted(
        encoder_cache.mm_features.get(req_id, []),
        key=lambda feature: feature.mm_position.offset,
    )
    if not mm_features:
        raise RuntimeError(
            f"Audex non-paged multimodal request {req_id!r} has no features."
        )

    if prefix_cache is None:
        captured_prefix_length, captured_working_cache = (
            _capture_requested_audio_prefix(
                runner,
                model_runner_module,
                adapter,
                sampling_params,
                token_ids,
            )
        )
        if captured_working_cache is not None:
            prefix_length = captured_prefix_length
            prefix_cache = captured_working_cache

    mx = importlib.import_module("mlx.core")
    if prefix_length < 0 or prefix_length >= len(token_ids):
        prefix_length = 0
        prefix_cache = None
    forward_token_ids = token_ids[prefix_length:]
    cache = (
        prefix_cache
        if prefix_cache is not None
        else model_runner_module.make_prompt_cache(runner._forward_model)
    )
    input_ids = mx.array([forward_token_ids], dtype=mx.int32)
    inputs_embeds_text = adapter.embed_tokens(input_ids)
    position_ids, mm_delta = adapter.get_mrope_input_positions(token_ids, mm_features)
    if prefix_length:
        position_ids = position_ids[..., prefix_length:]

    visual_mask = [False] * len(forward_token_ids)
    mm_embeds_parts: list[Any] = []
    for feature in mm_features:
        result = encoder_cache.encoder_outputs.get(feature.identifier)
        if result is None:
            raise RuntimeError(
                f"Encoder output for Audex feature {feature.identifier!r} "
                f"of request {req_id!r} is missing."
            )
        start = int(feature.mm_position.offset) - prefix_length
        length = _feature_length(feature)
        end = start + length
        if start < 0 or end > len(forward_token_ids):
            raise ValueError(
                f"Audex feature {feature.identifier!r} range {start}:{end} "
                f"is outside forwarded prompt suffix length "
                f"{len(forward_token_ids)} (prefix={prefix_length})."
            )
        hidden_states = result.hidden_states
        if int(getattr(hidden_states, "shape", (0,))[0]) != length:
            raise ValueError(
                f"Audex feature {feature.identifier!r} length mismatch: "
                f"placeholder length={length}, hidden states shape="
                f"{tuple(getattr(hidden_states, 'shape', ()))!r}."
            )
        visual_mask[start:end] = [True] * length
        mm_embeds_parts.append(hidden_states)

    visual_pos_masks = mx.array(visual_mask)[None, :]
    inputs_embeds = model_runner_module.merge_multimodal_embeddings(
        inputs_embeds_text,
        mm_embeds_parts,
        visual_pos_masks,
    )
    adapter._clear_paged_context_segment_positions()  # noqa: SLF001
    if prefix_cache is None:
        cache = adapter._prepare_hybrid_cache(  # noqa: SLF001
            cache,
            input_ids=input_ids,
            visual_pos_masks=visual_pos_masks,
        )
    mx.eval(inputs_embeds)
    forward_kwargs = {adapter._embeds_kwarg: inputs_embeds}  # noqa: SLF001
    if "cache" in adapter._call_parameters:  # noqa: SLF001
        forward_kwargs["cache"] = cache
    if "position_ids" in adapter._call_parameters:  # noqa: SLF001
        forward_kwargs["position_ids"] = position_ids
    model_output = adapter._text_model(input_ids, **forward_kwargs)  # noqa: SLF001

    logits = runner._extract_logits(model_output)
    last_logits = logits[:, -1, :]
    generators = {} if generator is None else {0: generator}
    sampling_batch = model_runner_module.SamplingBatch(
        [sampling_params],
        [token_ids],
        [[]],
        vocab_size=runner._vocab_size,
        device=runner.device,
        logitsprocs=runner._logitsprocs,
        generators=generators,
    )
    result = model_runner_module.sample_from_logits(
        last_logits,
        sampling_batch,
        runner._sampler,
        runner.device,
    )
    [next_token] = result.token_ids
    mx.eval(*[cache_item.state for cache_item in cache])
    return next_token, cache, result.logprobs, int(mm_delta)


def _verified_audio_prefix_snapshot(
    runner: Any,
    model_runner_module: Any,
    sampling_params: Any,
    token_ids: list[int],
) -> tuple[int, list[Any] | None]:
    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, dict):
        return 0, None
    state_key = extra_args.get(TEXT_STATE_KEY_ARG)
    prefix_count = extra_args.get(TEXT_STATE_PREFIX_TOKEN_COUNT_ARG)
    if not isinstance(state_key, str) or not state_key:
        return 0, None
    try:
        prefix_count = int(prefix_count)
    except (TypeError, ValueError):
        return 0, None
    snapshots = getattr(runner, "_audex_text_state_snapshots", None)
    snapshot = snapshots.get(state_key) if isinstance(snapshots, dict) else None
    if not isinstance(snapshot, dict) or not snapshot.get("reuse_eligible"):
        _debug(
            "audio history prefix unavailable "
            f"state_key={state_key!r} "
            f"known_keys={tuple(snapshots) if isinstance(snapshots, dict) else ()!r}"
        )
        return 0, None
    boundary_tokens = tuple(snapshot.get("boundary_tokens") or ())
    if (
        prefix_count <= 0
        or prefix_count >= len(token_ids)
        or snapshot.get("prefix_token_count") != prefix_count
        or len(boundary_tokens) != prefix_count
        or tuple(token_ids[:prefix_count]) != boundary_tokens
    ):
        requested_prefix = tuple(token_ids[: max(0, prefix_count)])
        mismatch_at = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(boundary_tokens, requested_prefix, strict=False)
                )
                if left != right
            ),
            None,
        )
        _debug(
            "audio history prefix rejected "
            f"state_key={state_key!r} requested_tokens={prefix_count} "
            f"snapshot_tokens={snapshot.get('prefix_token_count')} "
            f"mismatch_at={mismatch_at} "
            f"snapshot_head={boundary_tokens[:4]!r} "
            f"request_head={requested_prefix[:4]!r}"
        )
        return 0, None
    cached = snapshot.get("cache")
    if not isinstance(cached, list) or not cached:
        return 0, None
    merge = getattr(model_runner_module, "_merge_kv_caches", None)
    extract = getattr(model_runner_module, "_extract_kv_cache", None)
    if not callable(merge) or not callable(extract):
        return 0, None
    cloned = extract(merge([cached]), 0)
    _debug(
        "reusing verified audio history prefix "
        f"state_key={state_key!r} tokens={prefix_count}"
    )
    return prefix_count, cloned


def _capture_requested_audio_prefix(
    runner: Any,
    model_runner_module: Any,
    adapter: AudexMultimodalAdapter,
    sampling_params: Any,
    token_ids: list[int],
) -> tuple[int, list[Any] | None]:
    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, dict):
        return 0, None
    if extra_args.get(TEXT_STATE_BOUNDARY_ARG) != TEXT_STATE_COMMITTED_HISTORY_BOUNDARY:
        return 0, None
    state_key = extra_args.get(TEXT_STATE_KEY_ARG)
    try:
        prefix_count = int(extra_args.get(TEXT_STATE_PREFIX_TOKEN_COUNT_ARG))
    except (TypeError, ValueError):
        return 0, None
    if (
        not isinstance(state_key, str)
        or not state_key
        or prefix_count <= 0
        or prefix_count >= len(token_ids)
    ):
        return 0, None
    prefix_tokens = tuple(token_ids[:prefix_count])
    prefix_hash = _audex_token_hash(prefix_tokens)

    mx = importlib.import_module("mlx.core")
    cache = model_runner_module.make_prompt_cache(runner._forward_model)
    input_ids = mx.array([prefix_tokens], dtype=mx.int32)
    inputs_embeds = adapter.embed_tokens(input_ids)
    position_ids, _mm_delta = adapter.get_mrope_input_positions(
        list(prefix_tokens),
        [],
    )
    cache = adapter._prepare_hybrid_cache(  # noqa: SLF001
        cache,
        input_ids=input_ids,
        visual_pos_masks=None,
    )
    adapter._clear_paged_context_segment_positions()  # noqa: SLF001
    forward_kwargs = {adapter._embeds_kwarg: inputs_embeds}  # noqa: SLF001
    if "cache" in adapter._call_parameters:  # noqa: SLF001
        forward_kwargs["cache"] = cache
    if "position_ids" in adapter._call_parameters:  # noqa: SLF001
        forward_kwargs["position_ids"] = position_ids
    adapter._text_model(input_ids, **forward_kwargs)  # noqa: SLF001
    mx.eval(*[cache_item.state for cache_item in cache])

    merge = getattr(model_runner_module, "_merge_kv_caches", None)
    extract = getattr(model_runner_module, "_extract_kv_cache", None)
    if not callable(merge) or not callable(extract):
        return 0, None
    retained_cache = extract(merge([cache]), 0)
    snapshots = getattr(runner, "_audex_text_state_snapshots", None)
    if not isinstance(snapshots, dict):
        snapshots = {}
        runner._audex_text_state_snapshots = snapshots
    snapshots[state_key] = {
        "state_key": state_key,
        "boundary": TEXT_STATE_COMMITTED_HISTORY_BOUNDARY,
        "reuse_eligible": True,
        "prefix_token_count": prefix_count,
        "prefix_token_hash": prefix_hash,
        "boundary_tokens": prefix_tokens,
        "cache": retained_cache,
    }
    _debug(
        "captured exact multimodal audio prefix "
        f"state_key={state_key!r} tokens={prefix_count}"
    )
    return prefix_count, cache


def _audex_token_hash(tokens: Sequence[int]) -> str:
    digest = sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _feature_length(feature: Any) -> int:
    mm_position = feature.mm_position
    get_num_embeds = getattr(mm_position, "get_num_embeds", None)
    if callable(get_num_embeds):
        return int(get_num_embeds())
    return int(mm_position.length)


def _patch_audex_text_backbone_selection(DefaultModelAdapter: Any) -> None:
    """Route Audex loading through mlx_lm while preserving audio MM plumbing."""

    if getattr(DefaultModelAdapter, TEXT_BACKBONE_PATCH_SENTINEL, False):
        return

    original_should_force_text_backbone = DefaultModelAdapter.should_force_text_backbone
    original_normalize_model_config = DefaultModelAdapter.normalize_model_config

    def should_force_text_backbone(self: Any, hf_config: Any) -> bool:
        if is_audex_hf_config(hf_config):
            return True
        return original_should_force_text_backbone(self, hf_config)

    def normalize_model_config(self: Any, model_config: Any) -> None:
        hf_config = getattr(model_config, "hf_config", None)
        if is_audex_hf_config(hf_config):
            return None
        return original_normalize_model_config(self, model_config)

    DefaultModelAdapter.should_force_text_backbone = should_force_text_backbone
    DefaultModelAdapter.normalize_model_config = normalize_model_config
    DefaultModelAdapter._audex_mac_original_should_force_text_backbone = (  # noqa: SLF001
        original_should_force_text_backbone
    )
    DefaultModelAdapter._audex_mac_original_normalize_model_config = (  # noqa: SLF001
        original_normalize_model_config
    )
    setattr(DefaultModelAdapter, TEXT_BACKBONE_PATCH_SENTINEL, True)


def _patch_audex_generation_model_install() -> None:
    """Attach Audex multimodal state after loading via the text backbone."""

    try:
        from vllm_metal.v1.model_lifecycle import ModelLifecycle
    except ModuleNotFoundError:
        return

    if getattr(ModelLifecycle, LIFECYCLE_PATCH_SENTINEL, False):
        return

    original_install_generation_model = ModelLifecycle._install_generation_model

    def install_generation_model_with_audex_adapter(
        self: Any,
        loaded_model: Any,
        request: Any,
    ) -> None:
        original_install_generation_model(self, loaded_model, request)
        if not is_audex_hf_config(getattr(request, "hf_config", None)):
            return

        runner = self._runner
        if getattr(runner, "_multimodal_adapter", None) is not None:
            return

        model_path = getattr(request, "model_name", None)
        if model_path is None:
            multimodal_adapter = self._model_adapter.build_multimodal_adapter(
                loaded_model.model,
                request.hf_config,
            )
        else:
            multimodal_adapter = AudexMultimodalAdapter.from_loaded_model(
                loaded_model.model,
                model_path=model_path,
            )
        if multimodal_adapter is None:
            return

        from vllm_metal.v1.mm import EncoderCache

        runner._multimodal_adapter = multimodal_adapter
        runner.encoder_cache = EncoderCache()

    ModelLifecycle._install_generation_model = (
        install_generation_model_with_audex_adapter
    )
    ModelLifecycle._audex_mac_original_install_generation_model = (  # noqa: SLF001
        original_install_generation_model
    )
    setattr(ModelLifecycle, LIFECYCLE_PATCH_SENTINEL, True)


def patch_vllm_audex_processor() -> bool:
    global LAST_ERROR
    LAST_ERROR = None
    for module_name, model_class_name in NEMOTRON_PROCESSOR_TARGETS.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        try:
            _register_processor_on_nemotron_module(module, model_class_name)
        except Exception as exc:
            LAST_ERROR = f"{type(exc).__name__}: {exc}"
            return False
    _install_nemotron_processor_import_hook()
    return True


def _register_processor_on_nemotron_module(
    module: Any,
    model_class_name: str,
) -> bool:
    from vllm.multimodal import MULTIMODAL_REGISTRY

    model_cls = getattr(module, model_class_name)
    if getattr(model_cls, PROCESSOR_PATCH_SENTINEL, False):
        return True

    model_cls.supports_multimodal = True
    model_cls.supports_multimodal_raw_input_only = False
    MULTIMODAL_REGISTRY.register_processor(
        AudexProjectedAudioProcessor,
        info=AudexProcessingInfo,
        dummy_inputs=AudexDummyInputsBuilder,
    )(model_cls)
    setattr(model_cls, PROCESSOR_PATCH_SENTINEL, True)
    return True


def _install_nemotron_processor_import_hook() -> None:
    if any(
        getattr(finder, PROCESSOR_IMPORT_HOOK_SENTINEL, False)
        for finder in sys.meta_path
    ):
        return
    sys.meta_path.insert(0, _AudexNemotronProcessorImportHook())


class _AudexNemotronProcessorImportHook(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        setattr(self, PROCESSOR_IMPORT_HOOK_SENTINEL, True)

    def find_spec(
        self,
        fullname: str,
        path: Any,
        target: Any = None,
    ) -> Any:
        if fullname not in NEMOTRON_PROCESSOR_TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if isinstance(spec.loader, _AudexNemotronProcessorLoader):
            return spec
        spec.loader = _AudexNemotronProcessorLoader(spec.loader)
        return spec


class _AudexNemotronProcessorLoader(importlib.abc.Loader):
    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: Any) -> Any:
        create_module = getattr(self._wrapped, "create_module", None)
        if callable(create_module):
            return create_module(spec)
        return None

    def exec_module(self, module: Any) -> None:
        self._wrapped.exec_module(module)
        _register_processor_on_nemotron_module(
            module,
            NEMOTRON_PROCESSOR_TARGETS[module.__name__],
        )


def _patch_vllm_renderer_mm_state() -> bool:
    global LAST_ERROR
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.multimodal.registry import MultiModalTimingRegistry
        from vllm.renderers.base import BaseRenderer
        from vllm.utils.counter import AtomicCounter
        from vllm.utils.torch_utils import set_default_torch_num_threads
    except ModuleNotFoundError:
        return True
    except Exception as exc:
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return False

    if getattr(BaseRenderer, RENDERER_PATCH_SENTINEL, False):
        return True

    original_process_multimodal = BaseRenderer._process_multimodal

    def process_multimodal_with_audex_state(
        self: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if not hasattr(self, "_mm_req_counter"):
            self._mm_req_counter = AtomicCounter()
        if not hasattr(self, "_mm_timing_registry"):
            self._mm_timing_registry = MultiModalTimingRegistry(
                self.config.observability_config
            )
        if (
            self.mm_processor is None
            and MULTIMODAL_REGISTRY.supports_multimodal_inputs(self.config.model_config)
        ):
            mm_processor_cache = MULTIMODAL_REGISTRY.processor_cache_from_config(
                self.config
            )
            with set_default_torch_num_threads():
                self.mm_processor = MULTIMODAL_REGISTRY.create_processor(
                    self.config.model_config,
                    tokenizer=self.tokenizer,
                    cache=mm_processor_cache,
                )
        return original_process_multimodal(self, *args, **kwargs)

    BaseRenderer._process_multimodal = process_multimodal_with_audex_state
    BaseRenderer._audex_mac_original_process_multimodal = (  # noqa: SLF001
        original_process_multimodal
    )
    setattr(BaseRenderer, RENDERER_PATCH_SENTINEL, True)
    return True
