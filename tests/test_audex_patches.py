from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from audex_mac.audio_contract import (
    DEFAULT_SOUND_EMBEDDING_SIZE,
    SOUND_END_TOKEN,
    SOUND_START_TOKEN,
    SOUND_TOKEN,
)
from audex_mac.patches import (
    runtime,
    transformers_dynamic_module,
    vllm_metal_audex_adapter,
    vllm_metal_cfg,
)
from audex_mac.vllm_sts_requests import (
    AUDEX_TEXT_STATE_APPEND_MODE,
    AUDEX_TEXT_STATE_BOUNDARY_ARG,
    AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY,
    AUDEX_TEXT_STATE_KEY_ARG,
    AUDEX_TEXT_STATE_MODE_ARG,
    AUDEX_TEXT_STATE_PREFIX_TOKEN_COUNT_ARG,
    AUDEX_TEXT_STATE_PREFIX_TOKEN_HASH_ARG,
)

pytestmark = pytest.mark.fast


class FakeRegistry:
    def __init__(self) -> None:
        self.models: dict[str, str] = {}

    def get_supported_archs(self) -> set[str]:
        return set(self.models)

    def register_model(self, model_arch: str, model_cls: str) -> None:
        self.models[model_arch] = model_cls


class FakeMultiModalRegistry:
    def __init__(self) -> None:
        self.processor = None
        self.info = None
        self.dummy_inputs = None

    def register_processor(self, processor, *, info, dummy_inputs):
        self.processor = processor
        self.info = info
        self.dummy_inputs = dummy_inputs

        def wrapper(model_cls):
            model_cls._processor_factory = types.SimpleNamespace(
                processor=processor,
                info=info,
                dummy_inputs=dummy_inputs,
            )
            return model_cls

        return wrapper


def test_raw_audio_embedding_trim_is_opt_in_and_proportional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = (0.0,) * 46_695
    monkeypatch.delenv(
        vllm_metal_audex_adapter.AUDIO_EMBEDDING_TRIM_ENV,
        raising=False,
    )
    assert (
        vllm_metal_audex_adapter.raw_audio_num_embeddings(
            samples,
            sample_rate=16_000,
        )
        == DEFAULT_SOUND_EMBEDDING_SIZE
    )

    monkeypatch.setenv(vllm_metal_audex_adapter.AUDIO_EMBEDDING_TRIM_ENV, "1")
    assert (
        vllm_metal_audex_adapter.raw_audio_num_embeddings(
            samples,
            sample_rate=16_000,
        )
        == 73
    )


def install_fake_vllm_processor_modules(monkeypatch: pytest.MonkeyPatch):
    class FakeNemotronForCausalLM:
        supports_multimodal = False

    fake_registry = FakeMultiModalRegistry()

    module_names = (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.models",
    )
    for module_name in module_names:
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    fake_nemotron_module = types.ModuleType("vllm.model_executor.models.nemotron")
    fake_nemotron_module.NemotronForCausalLM = FakeNemotronForCausalLM
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.nemotron",
        fake_nemotron_module,
    )

    fake_multimodal_module = types.ModuleType("vllm.multimodal")
    fake_multimodal_module.MULTIMODAL_REGISTRY = fake_registry
    monkeypatch.setitem(sys.modules, "vllm.multimodal", fake_multimodal_module)
    return FakeNemotronForCausalLM, fake_registry


def install_fake_vllm_metal_patching(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_patching = types.ModuleType("vllm_metal.attention.patching")
    fake_patching.find_attn_attr = lambda _layer: None
    monkeypatch.setitem(sys.modules, "vllm_metal.attention.patching", fake_patching)


def install_fake_vllm_metal_sdpa_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSDPAPagedAttentionWrapper:
        def __init__(self, inner, *_args, **_kwargs) -> None:
            self.inner = inner

    fake_wrapper = types.ModuleType("vllm_metal.attention.impls.sdpa_wrapper")
    fake_wrapper.SDPAPagedAttentionWrapper = FakeSDPAPagedAttentionWrapper
    fake_sdpa = types.ModuleType("vllm_metal.attention.impls.sdpa")
    fake_sdpa.apply_attention_rope = lambda *_args, **_kwargs: (
        "rotated-q",
        "rotated-k",
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_metal.attention.impls.sdpa_wrapper",
        fake_wrapper,
    )
    monkeypatch.setitem(sys.modules, "vllm_metal.attention.impls.sdpa", fake_sdpa)


def install_fake_mlx_core_for_capacity(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metal_limit: int,
    active_memory: int,
    cache_memory: int = 0,
) -> None:
    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    fake_core.device_info = lambda: {
        "max_recommended_working_set_size": metal_limit,
    }
    fake_core.get_active_memory = lambda: active_memory
    fake_core.get_cache_memory = lambda: cache_memory
    fake_core.array = lambda value: value
    fake_core.eval = lambda *_args: None
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)


def test_vllm_nemotron_dense_patch_registers_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_registry = FakeRegistry()
    registry_module = types.SimpleNamespace(ModelRegistry=fake_registry)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.registry",
        registry_module,
    )

    assert runtime._register_vllm_nemotron_dense() is True
    assert runtime.VLLM_AUDEX_ARCHITECTURE in fake_registry.get_supported_archs()
    assert fake_registry.models == runtime.VLLM_AUDEX_ARCHITECTURE_ALIASES


def test_vllm_audex_model_info_patch_forces_multimodal_aliases() -> None:
    @dataclass(frozen=True)
    class FakeModelInfo:
        supports_multimodal: bool
        supports_multimodal_raw_input_only: bool
        requires_raw_input_tokens: bool

    class FakeRegistryWithModelInfo:
        def _try_inspect_model_cls(self, _model_arch: str):
            return FakeModelInfo(
                supports_multimodal=False,
                supports_multimodal_raw_input_only=True,
                requires_raw_input_tokens=True,
            )

    fake_registry = FakeRegistryWithModelInfo()

    assert runtime._patch_vllm_audex_model_info(fake_registry) is True

    model_info = fake_registry._try_inspect_model_cls(
        "NemotronHAudexForConditionalGeneration"
    )
    assert model_info == FakeModelInfo(
        supports_multimodal=True,
        supports_multimodal_raw_input_only=False,
        requires_raw_input_tokens=False,
    )


def test_runtime_patches_repair_vllm_platform_before_mlx_lm_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_transformers() -> bool:
        calls.append("transformers")
        return True

    def fake_repair() -> bool:
        calls.append("repair")
        return True

    def fake_install(module_name: str) -> bool:
        calls.append(f"install:{module_name}")
        return True

    def fake_registry() -> bool:
        calls.append("registry")
        return True

    def fake_device_info() -> bool:
        calls.append("device-info")
        return True

    def fake_capacity() -> bool:
        calls.append("capacity")
        return True

    def fake_adapter() -> bool:
        calls.append("adapter")
        return True

    monkeypatch.setattr(
        runtime,
        "_patch_transformers_local_dynamic_modules",
        fake_transformers,
    )
    monkeypatch.setattr(runtime, "_repair_vllm_metal_current_platform", fake_repair)
    monkeypatch.setattr(runtime, "_install_mlx_lm_module", fake_install)
    monkeypatch.setattr(runtime, "_patch_vllm_metal_device_info_api", fake_device_info)
    monkeypatch.setattr(
        runtime,
        "_patch_vllm_metal_nonpaged_capacity_override",
        fake_capacity,
    )
    monkeypatch.setattr(runtime, "_register_vllm_nemotron_dense", fake_registry)
    monkeypatch.setattr(runtime, "_patch_vllm_metal_audex_adapter", fake_adapter)

    report = runtime.apply_audex_runtime_patches()

    assert report.ready is True
    assert calls == [
        "transformers",
        "repair",
        f"install:{runtime.AUDEX_MLX_MODULE}",
        f"install:{runtime.AUDEX_MLX_H_MODULE}",
        "device-info",
        "capacity",
        "registry",
        "adapter",
    ]


def test_vllm_metal_device_info_patch_uses_current_mlx_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    fake_utils = types.ModuleType("vllm_metal.utils")
    fake_utils.set_wired_limit = lambda: calls.append(("original", None))
    fake_utils.logger = types.SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    fake_mx = types.SimpleNamespace(
        device_info=lambda: {"max_recommended_working_set_size": 123},
        metal=types.SimpleNamespace(
            device_info=lambda: (_ for _ in ()).throw(
                AssertionError("deprecated mx.metal.device_info should not be used")
            )
        ),
        set_wired_limit=lambda value: calls.append(("wired", value)),
    )
    fake_mlx = types.ModuleType("mlx")
    monkeypatch.setitem(sys.modules, "vllm_metal.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    assert runtime._patch_vllm_metal_device_info_api() is True

    fake_utils.set_wired_limit()

    assert getattr(
        fake_utils.set_wired_limit,
        runtime.VLLM_METAL_DEVICE_INFO_SENTINEL,
        False,
    )
    assert calls == [("wired", 123)]


def test_vllm_metal_nonpaged_capacity_patch_overrides_single_sequence_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeModelRunner:
        def scheduler_memory_reporting_mode(self, *, paged_attention_enabled):
            calls.append(("mode", paged_attention_enabled))
            return "single_sequence_estimate"

    class FakeWorker:
        model_runner = FakeModelRunner()
        metal_config = types.SimpleNamespace(use_paged_attention=False)
        vllm_config = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(gpu_memory_utilization=0.60)
        )

        def _one_sequence_kv_bytes(self):
            return 1024

    class FakePlanner:
        def __init__(self) -> None:
            self._worker = FakeWorker()

        def determine_available_memory(self):
            return 17

    fake_cache_policy = types.ModuleType("vllm_metal.v1.cache_policy")
    fake_cache_policy.WorkerCachePlanner = FakePlanner
    fake_cache_policy.logger = types.SimpleNamespace(
        warning=lambda *args: calls.append(("warning", args))
    )
    monkeypatch.setitem(sys.modules, "vllm_metal.v1.cache_policy", fake_cache_policy)
    monkeypatch.setenv(runtime.NONPAGED_KV_CAPACITY_SEQS_ENV, "4")
    install_fake_mlx_core_for_capacity(
        monkeypatch,
        metal_limit=32_768,
        active_memory=8_192,
    )

    assert runtime._patch_vllm_metal_nonpaged_capacity_override() is True

    assert FakePlanner().determine_available_memory() == 4096
    assert calls[0] == ("mode", False)
    assert calls[1][0] == "warning"


def test_vllm_metal_nonpaged_capacity_patch_rejects_over_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModelRunner:
        def scheduler_memory_reporting_mode(self, *, paged_attention_enabled):
            return "single_sequence_estimate"

    class FakeWorker:
        model_runner = FakeModelRunner()
        metal_config = types.SimpleNamespace(use_paged_attention=False)
        vllm_config = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(gpu_memory_utilization=0.50)
        )

        def _one_sequence_kv_bytes(self):
            return 4_096

    class FakePlanner:
        def __init__(self) -> None:
            self._worker = FakeWorker()

        def determine_available_memory(self):
            return 17

    fake_cache_policy = types.ModuleType("vllm_metal.v1.cache_policy")
    fake_cache_policy.WorkerCachePlanner = FakePlanner
    fake_cache_policy.logger = types.SimpleNamespace(warning=lambda *args: None)
    monkeypatch.setitem(sys.modules, "vllm_metal.v1.cache_policy", fake_cache_policy)
    monkeypatch.setenv(runtime.NONPAGED_KV_CAPACITY_SEQS_ENV, "4")
    install_fake_mlx_core_for_capacity(
        monkeypatch,
        metal_limit=16_384,
        active_memory=4_096,
        cache_memory=1_024,
    )

    assert runtime._patch_vllm_metal_nonpaged_capacity_override() is True

    with pytest.raises(RuntimeError, match="NONPAGED_KV_CAPACITY_SEQS=4"):
        FakePlanner().determine_available_memory()


def test_vllm_metal_nonpaged_capacity_patch_preserves_paged_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModelRunner:
        def scheduler_memory_reporting_mode(self, *, paged_attention_enabled):
            return "paged_attention_capacity"

    class FakeWorker:
        model_runner = FakeModelRunner()
        metal_config = types.SimpleNamespace(use_paged_attention=True)

    class FakePlanner:
        def __init__(self) -> None:
            self._worker = FakeWorker()

        def determine_available_memory(self):
            return 17

    fake_cache_policy = types.ModuleType("vllm_metal.v1.cache_policy")
    fake_cache_policy.WorkerCachePlanner = FakePlanner
    fake_cache_policy.logger = types.SimpleNamespace(warning=lambda *args: None)
    monkeypatch.setitem(sys.modules, "vllm_metal.v1.cache_policy", fake_cache_policy)
    monkeypatch.setenv(runtime.NONPAGED_KV_CAPACITY_SEQS_ENV, "4")

    assert runtime._patch_vllm_metal_nonpaged_capacity_override() is True

    assert FakePlanner().determine_available_memory() == 17


def test_nonpaged_kv_capacity_seqs_override_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(runtime.NONPAGED_KV_CAPACITY_SEQS_ENV, "0")

    with pytest.raises(ValueError, match=runtime.NONPAGED_KV_CAPACITY_SEQS_ENV):
        runtime._nonpaged_kv_capacity_seqs_override()


def test_transformers_dynamic_module_patch_preserves_snapshot_symlink_paths(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "models--nvidia--Audex" / "snapshots" / "rev"
    checkpoint = snapshot / "checkpoint_folder_full"
    blobs = tmp_path / "models--nvidia--Audex" / "blobs"
    checkpoint.mkdir(parents=True)
    blobs.mkdir(parents=True)
    audio_blob = blobs / "audio-config"
    base_blob = blobs / "base-config"
    audio_blob.write_text(
        "from .configuration_nemotron_h import NemotronHConfig\n",
        encoding="utf-8",
    )
    base_blob.write_text("class NemotronHConfig: pass\n", encoding="utf-8")
    (checkpoint / "configuration_nemotron_h_audio.py").symlink_to(audio_blob)
    (checkpoint / "configuration_nemotron_h.py").symlink_to(base_blob)

    fake_transformers = types.ModuleType("transformers")
    fake_dynamic = types.ModuleType("transformers.dynamic_module_utils")

    def original_cached_file(_model_path, filename, *_args, **_kwargs):
        return str((checkpoint / filename).resolve())

    def original_local_hash(*_args, **_kwargs):
        raise AssertionError("original hash should not see blob-resolved paths")

    def get_relative_import_files(module_file):
        module_path = Path(module_file)
        text = module_path.read_text(encoding="utf-8")
        if ".configuration_nemotron_h" in text:
            return [str(module_path.parent / "configuration_nemotron_h.py")]
        return []

    fake_dynamic.cached_file = original_cached_file
    fake_dynamic._compute_local_source_files_hash = original_local_hash
    fake_dynamic.get_relative_import_files = get_relative_import_files
    fake_transformers.dynamic_module_utils = fake_dynamic
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules,
        "transformers.dynamic_module_utils",
        fake_dynamic,
    )

    assert transformers_dynamic_module.patch_transformers_local_dynamic_modules()

    resolved = fake_dynamic.cached_file(
        str(checkpoint),
        "configuration_nemotron_h_audio.py",
    )
    assert resolved == str(checkpoint / "configuration_nemotron_h_audio.py")

    source_hash = fake_dynamic._compute_local_source_files_hash(
        str(checkpoint),
        resolved,
    )
    assert len(source_hash) == 16


def test_vllm_metal_platform_repair_replaces_cached_cpu_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCpuPlatform:
        pass

    class FakeMetalPlatform:
        pass

    fake_platforms = types.ModuleType("vllm.platforms")
    fake_platforms._current_platform = FakeCpuPlatform()
    fake_platforms.current_platform = fake_platforms._current_platform
    fake_platforms.resolve_current_platform_cls_qualname = (
        lambda: "vllm_metal.platform.MetalPlatform"
    )

    def resolve_obj_by_qualname(qualname: str):
        assert qualname == "vllm_metal.platform.MetalPlatform"
        return FakeMetalPlatform

    fake_import_utils = types.ModuleType("vllm.utils.import_utils")
    fake_import_utils.resolve_obj_by_qualname = resolve_obj_by_qualname
    fake_platforms.resolve_obj_by_qualname = resolve_obj_by_qualname
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    monkeypatch.setitem(sys.modules, "vllm.utils", types.ModuleType("vllm.utils"))
    monkeypatch.setitem(sys.modules, "vllm.utils.import_utils", fake_import_utils)

    assert runtime._repair_vllm_metal_current_platform() is True
    assert isinstance(fake_platforms._current_platform, FakeMetalPlatform)


def test_mlx_lm_patch_injects_nemotron_dense_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_dense_module = types.ModuleType(runtime.AUDEX_MLX_MODULE)
    fake_h_module = types.ModuleType(runtime.AUDEX_MLX_H_MODULE)
    monkeypatch.setitem(sys.modules, runtime.AUDEX_MLX_MODULE, fake_dense_module)
    monkeypatch.setitem(sys.modules, runtime.AUDEX_MLX_H_MODULE, fake_h_module)
    for module_name in runtime.MLX_LM_AUDEX_MODULES:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    assert runtime._install_mlx_lm_module(runtime.AUDEX_MLX_MODULE) is True
    assert sys.modules["mlx_lm.models.nemotron_dense"] is fake_dense_module
    assert sys.modules["mlx_lm.models.nemotron_dense_audex"] is fake_dense_module
    assert "mlx_lm.models.nemotron_h_audex" not in sys.modules

    assert runtime._install_mlx_lm_module(runtime.AUDEX_MLX_H_MODULE) is True
    assert sys.modules["mlx_lm.models.nemotron_h_audex"] is fake_h_module


def test_mlx_lm_patch_installs_lazy_module_without_importing_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module_name in (
        runtime.AUDEX_MLX_MODULE,
        "mlx_lm.models.nemotron_dense",
        "mlx_lm.models.nemotron_dense_audex",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    assert runtime._install_mlx_lm_module(runtime.AUDEX_MLX_MODULE) is True

    installed = sys.modules["mlx_lm.models.nemotron_dense"]
    assert isinstance(installed, runtime._LazyAudexModule)
    assert runtime.AUDEX_MLX_MODULE not in sys.modules


def test_vllm_metal_audex_adapter_patch_selects_audex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDefaultModelAdapter:
        def should_force_text_backbone(self, hf_config):
            return False

        def normalize_model_config(self, model_config) -> None:
            model_config.multimodal_config = None

        def build_multimodal_adapter(self, model, hf_config):
            return None

    fake_module = types.ModuleType("vllm_metal.v1.model_adapter")
    fake_module.DefaultModelAdapter = FakeDefaultModelAdapter
    monkeypatch.setitem(sys.modules, "vllm_metal.v1.model_adapter", fake_module)
    install_fake_vllm_metal_patching(monkeypatch)
    install_fake_vllm_metal_sdpa_wrapper(monkeypatch)
    fake_model_cls, fake_registry = install_fake_vllm_processor_modules(monkeypatch)

    assert vllm_metal_audex_adapter.patch_default_model_adapter() is True
    assert fake_model_cls.supports_multimodal is True
    assert fake_registry.processor is (
        vllm_metal_audex_adapter.AudexProjectedAudioProcessor
    )
    assert fake_registry.info is vllm_metal_audex_adapter.AudexProcessingInfo

    class FakeEmbedTokens:
        def __call__(self, input_ids):
            return input_ids

    class FakeLanguageModel:
        model = types.SimpleNamespace(embed_tokens=FakeEmbedTokens())

        def __call__(
            self,
            input_ids,
            *,
            cache=None,
            position_ids=None,
            input_embeddings=None,
        ):
            return input_embeddings

    model = types.SimpleNamespace(language_model=FakeLanguageModel())
    hf_config = types.SimpleNamespace(
        model_type="nemotron_dense_audex",
        architectures=["NemotronDenseAudexForConditionalGeneration"],
    )
    adapter = FakeDefaultModelAdapter().build_multimodal_adapter(model, hf_config)

    assert isinstance(adapter, vllm_metal_audex_adapter.AudexMultimodalAdapter)
    assert adapter.forward_ready is True


def test_vllm_metal_audex_adapter_forces_loader_without_clearing_multimodal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDefaultModelAdapter:
        def should_force_text_backbone(self, hf_config):
            return False

        def normalize_model_config(self, model_config) -> None:
            model_config.multimodal_config = None

        def build_multimodal_adapter(self, model, hf_config):
            return None

    fake_module = types.ModuleType("vllm_metal.v1.model_adapter")
    fake_module.DefaultModelAdapter = FakeDefaultModelAdapter
    monkeypatch.setitem(sys.modules, "vllm_metal.v1.model_adapter", fake_module)
    install_fake_vllm_metal_patching(monkeypatch)
    install_fake_vllm_metal_sdpa_wrapper(monkeypatch)
    install_fake_vllm_processor_modules(monkeypatch)

    hf_config = types.SimpleNamespace(
        model_type="nemotron_h_audex",
        architectures=["NemotronHAudexForConditionalGeneration"],
    )
    model_config = types.SimpleNamespace(
        hf_config=hf_config,
        multimodal_config={"audio": 1},
    )

    assert vllm_metal_audex_adapter.patch_default_model_adapter() is True

    adapter = FakeDefaultModelAdapter()
    assert adapter.should_force_text_backbone(hf_config) is True

    adapter.normalize_model_config(model_config)
    assert model_config.multimodal_config == {"audio": 1}


def test_vllm_metal_non_paged_audex_runner_patch_allows_encoder_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMetalModelRunner:
        def _reject_scheduled_encoder_inputs(self, scheduled_encoder_inputs):
            raise NotImplementedError("paged-only")

        def _handle_new_requests(self, batch, new_reqs, scheduler_output):
            self.handled = (batch, new_reqs, scheduler_output)

        def _prefill_single(self, token_ids, sampling_params, generator=None):
            return 1, [], None

        def _run_vision_encoders(self, scheduled_encoder_inputs):
            self.encoded = scheduled_encoder_inputs

    fake_runner_module = types.ModuleType("vllm_metal.v1.model_runner")
    fake_runner_module.MetalModelRunner = FakeMetalModelRunner
    monkeypatch.setitem(sys.modules, "vllm_metal", types.ModuleType("vllm_metal"))
    monkeypatch.setitem(sys.modules, "vllm_metal.v1", types.ModuleType("vllm_metal.v1"))
    monkeypatch.setitem(
        sys.modules,
        "vllm_metal.v1.model_runner",
        fake_runner_module,
    )

    adapter = vllm_metal_audex_adapter.AudexMultimodalAdapter(
        model=object(),
        text_model=object(),
        embeds_kwarg="inputs_embeds",
        embed_tokens_path="model.embed_tokens",
        call_parameters=set(),
    )
    runner = FakeMetalModelRunner()
    runner._paged_attention_runtime = None
    runner._multimodal_adapter = adapter

    assert vllm_metal_audex_adapter._patch_vllm_metal_non_paged_multimodal_prefill()
    runner._reject_scheduled_encoder_inputs({"req": [0]})

    assert runner.encoded == {"req": [0]}


def test_vllm_metal_audex_lifecycle_attaches_adapter_after_text_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModelLifecycle:
        def __init__(self) -> None:
            self._runner = types.SimpleNamespace(
                _multimodal_adapter=None,
                encoder_cache=None,
            )
            self._model_adapter = types.SimpleNamespace(
                build_multimodal_adapter=lambda model, hf_config: "audex-adapter"
            )

        def _install_generation_model(self, loaded_model, request) -> None:
            self._runner.model = loaded_model.model
            self._runner._is_vlm = request.is_vlm

    class FakeEncoderCache:
        pass

    fake_lifecycle_module = types.ModuleType("vllm_metal.v1.model_lifecycle")
    fake_lifecycle_module.ModelLifecycle = FakeModelLifecycle
    fake_mm_module = types.ModuleType("vllm_metal.v1.mm")
    fake_mm_module.EncoderCache = FakeEncoderCache
    monkeypatch.setitem(
        sys.modules,
        "vllm_metal.v1.model_lifecycle",
        fake_lifecycle_module,
    )
    monkeypatch.setitem(sys.modules, "vllm_metal.v1.mm", fake_mm_module)

    vllm_metal_audex_adapter._patch_audex_generation_model_install()

    lifecycle = FakeModelLifecycle()
    lifecycle._install_generation_model(
        types.SimpleNamespace(model="loaded-model"),
        types.SimpleNamespace(
            is_vlm=False,
            hf_config=types.SimpleNamespace(model_type="nemotron_h_audex"),
        ),
    )

    assert lifecycle._runner.model == "loaded-model"
    assert lifecycle._runner._is_vlm is False
    assert lifecycle._runner._multimodal_adapter == "audex-adapter"
    assert isinstance(lifecycle._runner.encoder_cache, FakeEncoderCache)


def test_vllm_metal_audex_patching_detects_only_attention_mixers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_patching = types.ModuleType("vllm_metal.attention.patching")

    def original_find_attn_attr(layer):
        if hasattr(layer, "self_attn"):
            return "self_attn"
        return None

    fake_patching.find_attn_attr = original_find_attn_attr
    monkeypatch.setitem(sys.modules, "vllm_metal.attention.patching", fake_patching)

    assert vllm_metal_audex_adapter._patch_nemotron_h_mixer_attention_detection()

    assert (
        fake_patching.find_attn_attr(
            types.SimpleNamespace(self_attn="already-supported")
        )
        == "self_attn"
    )
    assert (
        fake_patching.find_attn_attr(
            types.SimpleNamespace(block_type="*", mixer=object())
        )
        == "mixer"
    )
    for block_type in ("M", "-", "E"):
        assert (
            fake_patching.find_attn_attr(
                types.SimpleNamespace(block_type=block_type, mixer=object())
            )
            is None
        )


def test_vllm_metal_audex_normalizes_nemotron_h_attention_contract() -> None:
    class NemotronHAttention:
        num_heads = 24
        num_key_value_heads = 8

    attention = NemotronHAttention()

    vllm_metal_audex_adapter._normalize_nemotron_h_attention_contract(attention)

    assert attention.n_heads == 24
    assert attention.n_kv_heads == 8
    assert attention.rope("q", offset=123) == "q"


def test_vllm_metal_audex_bypasses_rope_for_nemotron_h_attention() -> None:
    class NemotronHAttention:
        pass

    sdpa_module = types.SimpleNamespace(
        apply_attention_rope=lambda *_args, **_kwargs: ("rotated-q", "rotated-k")
    )

    assert vllm_metal_audex_adapter._patch_nemotron_h_sdpa_no_rope(sdpa_module)

    assert sdpa_module.apply_attention_rope(NemotronHAttention(), "q", "k", [0, 1]) == (
        "q",
        "k",
    )
    assert sdpa_module.apply_attention_rope(object(), "q", "k", [0, 1]) == (
        "rotated-q",
        "rotated-k",
    )


def test_vllm_audex_processor_patch_registers_proxy_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model_cls, fake_registry = install_fake_vllm_processor_modules(monkeypatch)

    assert vllm_metal_audex_adapter.patch_vllm_audex_processor() is True

    assert fake_model_cls.supports_multimodal is True
    assert fake_model_cls.supports_multimodal_raw_input_only is False
    assert fake_registry.processor is (
        vllm_metal_audex_adapter.AudexProjectedAudioProcessor
    )
    assert fake_registry.dummy_inputs is (
        vllm_metal_audex_adapter.AudexDummyInputsBuilder
    )


def test_vllm_audex_processor_patch_registers_nemotron_h_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNemotronHForCausalLM:
        supports_multimodal = False

    fake_registry = FakeMultiModalRegistry()
    fake_h_module = types.ModuleType("vllm.model_executor.models.nemotron_h")
    fake_h_module.NemotronHForCausalLM = FakeNemotronHForCausalLM
    fake_multimodal_module = types.ModuleType("vllm.multimodal")
    fake_multimodal_module.MULTIMODAL_REGISTRY = fake_registry
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.nemotron_h",
        fake_h_module,
    )
    monkeypatch.setitem(sys.modules, "vllm.multimodal", fake_multimodal_module)

    assert vllm_metal_audex_adapter.patch_vllm_audex_processor() is True

    assert FakeNemotronHForCausalLM.supports_multimodal is True
    assert fake_registry.processor is (
        vllm_metal_audex_adapter.AudexProjectedAudioProcessor
    )
    assert fake_registry.info is vllm_metal_audex_adapter.AudexProcessingInfo


def test_audex_processing_info_exposes_default_tokenization_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FakeTokenizeParams:
        max_total_tokens: int
        do_lower_case: bool
        add_special_tokens: bool

    fake_renderers = types.ModuleType("vllm.renderers")
    fake_renderers.TokenizeParams = FakeTokenizeParams
    monkeypatch.setitem(sys.modules, "vllm.renderers", fake_renderers)

    ctx = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            model="audex",
            max_model_len=5120,
            encoder_config={"do_lower_case": True},
        )
    )
    info = vllm_metal_audex_adapter.AudexProcessingInfo(ctx)

    assert info.default_tok_params == FakeTokenizeParams(
        max_total_tokens=5120,
        do_lower_case=True,
        add_special_tokens=True,
    )


def test_vllm_renderer_patch_initializes_missing_multimodal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCounter:
        def __init__(self) -> None:
            self.value = 0

        def inc(self, amount: int) -> int:
            self.value += amount
            return self.value

    class FakeTimingRegistry:
        def __init__(self, observability_config) -> None:
            self.observability_config = observability_config

    class FakeRegistry:
        def supports_multimodal_inputs(self, _model_config) -> bool:
            return True

        def processor_cache_from_config(self, _config):
            return "cache"

        def create_processor(self, _model_config, *, tokenizer, cache):
            return {"tokenizer": tokenizer, "cache": cache}

    class FakeThreads:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return None

    class FakeBaseRenderer:
        def _process_multimodal(self, *_args, **_kwargs):
            return {
                "counter": self._mm_req_counter.inc(1),
                "timing": self._mm_timing_registry.observability_config,
                "processor": self.mm_processor,
            }

    fake_renderers_base = types.ModuleType("vllm.renderers.base")
    fake_renderers_base.BaseRenderer = FakeBaseRenderer
    fake_multimodal = types.ModuleType("vllm.multimodal")
    fake_multimodal.MULTIMODAL_REGISTRY = FakeRegistry()
    fake_registry_module = types.ModuleType("vllm.multimodal.registry")
    fake_registry_module.MultiModalTimingRegistry = FakeTimingRegistry
    fake_counter_module = types.ModuleType("vllm.utils.counter")
    fake_counter_module.AtomicCounter = FakeCounter
    fake_torch_utils = types.ModuleType("vllm.utils.torch_utils")
    fake_torch_utils.set_default_torch_num_threads = FakeThreads
    monkeypatch.setitem(sys.modules, "vllm.renderers.base", fake_renderers_base)
    monkeypatch.setitem(sys.modules, "vllm.multimodal", fake_multimodal)
    monkeypatch.setitem(sys.modules, "vllm.multimodal.registry", fake_registry_module)
    monkeypatch.setitem(sys.modules, "vllm.utils.counter", fake_counter_module)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", fake_torch_utils)

    assert vllm_metal_audex_adapter._patch_vllm_renderer_mm_state() is True

    renderer = FakeBaseRenderer()
    renderer.config = types.SimpleNamespace(
        model_config=object(),
        observability_config="observability",
    )
    renderer.tokenizer = "tokenizer"
    renderer.mm_processor = None

    assert renderer._process_multimodal() == {
        "counter": 1,
        "timing": "observability",
        "processor": {"tokenizer": "tokenizer", "cache": "cache"},
    }


def test_audex_projected_audio_processor_emits_vllm_multimodal_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class FakePlaceholderRange:
        offset: int
        length: int

        def get_num_embeds(self) -> int:
            return self.length

    @dataclass
    class FakeFeatureSpec:
        data: dict
        modality: str
        identifier: str
        mm_position: FakePlaceholderRange

    fake_vllm_inputs = types.ModuleType("vllm.inputs")

    def mm_input(**kwargs):
        return {"type": "multimodal", **kwargs}

    fake_vllm_inputs.mm_input = mm_input
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.inputs", fake_vllm_inputs)
    monkeypatch.setitem(sys.modules, "vllm_metal", types.ModuleType("vllm_metal"))
    monkeypatch.setitem(
        sys.modules,
        "vllm_metal.multimodal",
        types.ModuleType("vllm_metal.multimodal"),
    )
    fake_feature_module = types.ModuleType("vllm_metal.multimodal.feature_spec")
    fake_feature_module.MultiModalFeatureSpec = FakeFeatureSpec
    fake_feature_module.PlaceholderRange = FakePlaceholderRange
    monkeypatch.setitem(
        sys.modules,
        "vllm_metal.multimodal.feature_spec",
        fake_feature_module,
    )

    class FakeFieldConfig:
        @staticmethod
        def shared(modality: str, *, batch_size: int):
            assert modality == "audio"
            assert batch_size == 1
            return FakeFieldConfig()

        def build_elems(self, key: str, data):
            return [types.SimpleNamespace(key=key, data=data)]

    class FakeKwargsItem(dict):
        pass

    class FakeKwargsItems(dict):
        pass

    fake_mm_inputs = types.ModuleType("vllm.multimodal.inputs")
    fake_mm_inputs.MultiModalFieldConfig = FakeFieldConfig
    fake_mm_inputs.MultiModalKwargsItem = FakeKwargsItem
    fake_mm_inputs.MultiModalKwargsItems = FakeKwargsItems
    monkeypatch.setitem(sys.modules, "vllm.multimodal.inputs", fake_mm_inputs)

    class FakeTokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {SOUND_TOKEN: 29, SOUND_START_TOKEN: 30, SOUND_END_TOKEN: 31}

    class FakeEmbeddings:
        shape = (3, 2048)

    mm_data_items = types.SimpleNamespace(
        get_items=lambda modality, typ: typ(
            [{"audex_projected_embeddings": FakeEmbeddings()}]
        )
    )
    info = vllm_metal_audex_adapter.AudexProcessingInfo(
        types.SimpleNamespace(
            model_config=types.SimpleNamespace(model="audex-test"),
            get_tokenizer=lambda: FakeTokenizer(),
        )
    )
    assert info.supported_mm_limits == {"audio": 1}
    assert info.allowed_mm_limits == {"audio": 1}
    assert info.skip_prompt_length_check is False
    assert info.get_mm_max_tokens_per_item(
        seq_len=4096,
        mm_counts={"audio": 1},
    ) == {"audio": DEFAULT_SOUND_EMBEDDING_SIZE}
    processor = vllm_metal_audex_adapter.AudexProjectedAudioProcessor(
        info,
        vllm_metal_audex_adapter.AudexDummyInputsBuilder(info),
    )

    output = processor.apply(
        types.SimpleNamespace(
            prompt=[1, 29, 29, 29, 2],
            mm_data_items=mm_data_items,
            mm_uuid_items={"audio": ["audio-uuid"]},
        ),
        timing_ctx=types.SimpleNamespace(),
    )

    assert output["type"] == "multimodal"
    assert output["prompt_token_ids"] == [1, 30, 29, 29, 29, 31, 2]
    assert output["mm_hashes"] == {"audio": ["audio-uuid"]}
    field_elem = output["mm_kwargs"]["audio"][0]["audex_projected_embeddings"]
    assert field_elem.key == "audex_projected_embeddings"
    assert field_elem.data.shape == (3, 2048)
    assert output["mm_placeholders"]["audio"][0].offset == 2
    assert output["mm_placeholders"]["audio"][0].length == 3


def test_audex_audio_processor_forwards_raw_audio_tuple_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class FakePlaceholderRange:
        offset: int
        length: int

        def get_num_embeds(self) -> int:
            return self.length

    @dataclass
    class FakeFeatureSpec:
        data: dict
        modality: str
        identifier: str
        mm_position: FakePlaceholderRange

    fake_vllm_inputs = types.ModuleType("vllm.inputs")
    fake_vllm_inputs.mm_input = lambda **kwargs: {"type": "multimodal", **kwargs}
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.inputs", fake_vllm_inputs)
    monkeypatch.setitem(sys.modules, "vllm_metal", types.ModuleType("vllm_metal"))
    monkeypatch.setitem(
        sys.modules,
        "vllm_metal.multimodal",
        types.ModuleType("vllm_metal.multimodal"),
    )
    fake_feature_module = types.ModuleType("vllm_metal.multimodal.feature_spec")
    fake_feature_module.MultiModalFeatureSpec = FakeFeatureSpec
    fake_feature_module.PlaceholderRange = FakePlaceholderRange
    monkeypatch.setitem(
        sys.modules,
        "vllm_metal.multimodal.feature_spec",
        fake_feature_module,
    )

    class FakeFieldConfig:
        @staticmethod
        def shared(modality: str, *, batch_size: int):
            assert modality == "audio"
            assert batch_size == 1
            return FakeFieldConfig()

        def build_elems(self, key: str, data):
            return [types.SimpleNamespace(key=key, data=data)]

    fake_mm_inputs = types.ModuleType("vllm.multimodal.inputs")
    fake_mm_inputs.MultiModalFieldConfig = FakeFieldConfig
    fake_mm_inputs.MultiModalKwargsItem = dict
    fake_mm_inputs.MultiModalKwargsItems = dict
    monkeypatch.setitem(sys.modules, "vllm.multimodal.inputs", fake_mm_inputs)

    class FakeTokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {SOUND_TOKEN: 29, SOUND_START_TOKEN: 30, SOUND_END_TOKEN: 31}

    raw_audio = ([0.0, 0.1, -0.1], 16_000)
    mm_data_items = types.SimpleNamespace(
        get_items=lambda modality, typ: typ([raw_audio])
    )
    info = vllm_metal_audex_adapter.AudexProcessingInfo(
        types.SimpleNamespace(
            model_config=types.SimpleNamespace(model="audex-test"),
            get_tokenizer=lambda: FakeTokenizer(),
        )
    )
    processor = vllm_metal_audex_adapter.AudexProjectedAudioProcessor(
        info,
        vllm_metal_audex_adapter.AudexDummyInputsBuilder(info),
    )
    output = processor.apply(
        types.SimpleNamespace(
            prompt=[1, 29, 2],
            mm_data_items=mm_data_items,
            mm_uuid_items={"audio": ["raw-audio-uuid"]},
        ),
        timing_ctx=types.SimpleNamespace(),
    )

    assert output["prompt_token_ids"][:3] == [1, 30, 29]
    assert output["prompt_token_ids"][-2:] == [31, 2]
    assert output["prompt_token_ids"].count(29) == 750
    assert output["mm_hashes"] == {"audio": ["raw-audio-uuid"]}
    assert output["mm_placeholders"]["audio"][0].offset == 2
    assert output["mm_placeholders"]["audio"][0].length == 750
    samples_elem = output["mm_kwargs"]["audio"][0]["audex_raw_audio_samples"]
    sample_rate_elem = output["mm_kwargs"]["audio"][0]["sample_rate"]
    assert samples_elem.key == "audex_raw_audio_samples"
    assert samples_elem.data == [0.0, 0.1, -0.1]
    assert sample_rate_elem.key == "sample_rate"
    assert sample_rate_elem.data == 16_000


def test_audex_adapter_accepts_projected_audio_embeddings() -> None:
    class FakeEmbeddings:
        ndim = 2
        shape = (750, 2048)

    class FakeFeaturePosition:
        def get_num_embeds(self) -> int:
            return 750

    feature = types.SimpleNamespace(
        modality="audio",
        data={"audex_projected_embeddings": FakeEmbeddings()},
        mm_position=FakeFeaturePosition(),
    )
    adapter = vllm_metal_audex_adapter.AudexMultimodalAdapter(
        model=types.SimpleNamespace(),
        text_model=types.SimpleNamespace(),
        embeds_kwarg="input_embeddings",
        embed_tokens_path="model.embed_tokens",
        call_parameters={"input_embeddings"},
    )

    outputs = adapter.encode_multimodal([feature])

    assert outputs[0].hidden_states is feature.data["audex_projected_embeddings"]


def test_audex_adapter_converts_torch_transport_tensor_to_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTorchTensor:
        ndim = 2
        shape = (750, 2048)

    converted = object()
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = FakeTorchTensor
    fake_bridge = types.ModuleType("vllm_metal.pytorch_backend.tensor_bridge")
    fake_bridge.torch_to_mlx = lambda value: converted
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, fake_bridge.__name__, fake_bridge)

    assert (
        vllm_metal_audex_adapter.AudexMultimodalAdapter._as_mlx(FakeTorchTensor())
        is converted
    )


def test_audex_adapter_call_lm_omits_unsupported_position_ids() -> None:
    class FakeTextModel:
        def __init__(self) -> None:
            self.kwargs = None

        def __call__(self, input_ids, *, input_embeddings):
            self.kwargs = {"input_embeddings": input_embeddings}
            return "ok"

    text_model = FakeTextModel()
    adapter = vllm_metal_audex_adapter.AudexMultimodalAdapter(
        model=types.SimpleNamespace(),
        text_model=text_model,
        embeds_kwarg="input_embeddings",
        embed_tokens_path="model.embed_tokens",
        call_parameters={"input_embeddings"},
    )

    assert (
        adapter.call_lm(
            input_ids="ids",
            inputs_embeds="embeds",
            cache="cache",
            position_ids="positions",
        )
        == "ok"
    )
    assert text_model.kwargs == {"input_embeddings": "embeds"}


def test_audex_adapter_call_lm_clears_mrope_segment_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_context = types.SimpleNamespace(segment_positions=["audio-positions", None])
    fake_context_module = types.ModuleType("vllm_metal.attention.context")
    fake_context_module.get_context = lambda: fake_context
    monkeypatch.setitem(sys.modules, fake_context_module.__name__, fake_context_module)

    class FakeTextModel:
        def __call__(self, input_ids, *, input_embeddings):
            return {"input_ids": input_ids, "input_embeddings": input_embeddings}

    adapter = vllm_metal_audex_adapter.AudexMultimodalAdapter(
        model=types.SimpleNamespace(),
        text_model=FakeTextModel(),
        embeds_kwarg="input_embeddings",
        embed_tokens_path="model.embed_tokens",
        call_parameters={"input_embeddings"},
    )

    assert adapter.call_lm(
        input_ids="ids",
        inputs_embeds="embeds",
        cache="cache",
        position_ids="positions",
    ) == {"input_ids": "ids", "input_embeddings": "embeds"}
    assert fake_context.segment_positions == [None, None]


def test_audex_adapter_replaces_used_hybrid_cache_on_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    clear_cache_calls = []
    fake_mx.eval = lambda *_values: None
    fake_mx.clear_cache = lambda: clear_cache_calls.append("clear")
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    class ArraysCache:
        pass

    class FakeTextModel:
        def __init__(self) -> None:
            self.make_cache_calls = 0
            self.seen_caches = []

        def make_cache(self):
            self.make_cache_calls += 1
            return [ArraysCache()]

        def __call__(self, input_ids, *, input_embeddings, cache):
            self.seen_caches.append(cache)
            return {"input_ids": input_ids, "cache": cache}

    text_model = FakeTextModel()
    adapter = vllm_metal_audex_adapter.AudexMultimodalAdapter(
        model=types.SimpleNamespace(),
        text_model=text_model,
        embeds_kwarg="input_embeddings",
        embed_tokens_path="model.embed_tokens",
        call_parameters={"input_embeddings", "cache"},
    )
    input_ids = types.SimpleNamespace(shape=(1, 4))

    first = adapter.call_lm(
        input_ids=input_ids,
        inputs_embeds="embeds",
        cache=["provided"],
        position_ids="positions",
    )
    second = adapter.call_lm(
        input_ids=input_ids,
        inputs_embeds="embeds",
        cache=["provided"],
        position_ids="positions",
    )

    assert text_model.make_cache_calls == 2
    assert clear_cache_calls == ["clear"]
    assert first["cache"][0] is not second["cache"][0]
    assert type(first["cache"][0]).__name__ == "ArraysCache"
    assert type(second["cache"][0]).__name__ == "ArraysCache"


def test_audex_feature_spec_uses_projected_embedding_placeholder_run() -> None:
    @dataclass
    class FakePlaceholderRange:
        offset: int
        length: int

        def get_num_embeds(self) -> int:
            return self.length

    @dataclass
    class FakeFeatureSpec:
        data: dict
        modality: str
        identifier: str
        mm_position: FakePlaceholderRange

    class FakeEmbeddings:
        shape = (3, 2048)

    feature = vllm_metal_audex_adapter.build_projected_audio_feature_spec(
        projected_embeddings=FakeEmbeddings(),
        prompt_token_ids=[1, 29, 29, 29, 2],
        sound_token_id=29,
        identifier="audex-audio-test",
        feature_spec_cls=FakeFeatureSpec,
        placeholder_range_cls=FakePlaceholderRange,
    )

    assert feature.modality == "audio"
    assert feature.identifier == "audex-audio-test"
    assert feature.mm_position.offset == 1
    assert feature.mm_position.length == 3
    assert set(feature.data) == {"audex_projected_embeddings"}


def test_audex_feature_spec_rejects_missing_placeholder_run() -> None:
    class FakeEmbeddings:
        shape = (3, 2048)

    with pytest.raises(ValueError, match="placeholder run of 3 tokens"):
        vllm_metal_audex_adapter.build_projected_audio_feature_spec(
            projected_embeddings=FakeEmbeddings(),
            prompt_token_ids=[1, 29, 29, 2],
            sound_token_id=29,
            feature_spec_cls=object,
            placeholder_range_cls=object,
        )


def test_audex_adapter_rejects_unknown_audio_payload() -> None:
    feature = types.SimpleNamespace(
        modality="audio",
        data={"raw_audio": object()},
        mm_position=None,
    )
    adapter = vllm_metal_audex_adapter.AudexMultimodalAdapter(
        model=types.SimpleNamespace(),
        text_model=types.SimpleNamespace(),
        embeds_kwarg="input_embeddings",
        embed_tokens_path="model.embed_tokens",
        call_parameters={"input_embeddings"},
    )

    with pytest.raises(ValueError, match="audex_projected_embeddings"):
        adapter.encode_multimodal([feature])


def test_vllm_cfg_patch_records_text_state_snapshot_before_cleanup() -> None:
    cleaned: list[set[str]] = []

    class FakeMetalModelRunner:
        def _cleanup_finished_requests(self, evicted_req_ids, *args, **kwargs):
            cleaned.append(set(evicted_req_ids))
            for req_id in evicted_req_ids:
                self._request_states.pop(req_id, None)

    fake_model_runner = types.SimpleNamespace(MetalModelRunner=FakeMetalModelRunner)
    vllm_metal_cfg._patch_model_runner_persistent_batch_cache_cleanup(fake_model_runner)

    sampling_params = types.SimpleNamespace(
        extra_args={
            AUDEX_TEXT_STATE_KEY_ARG: "conv-1",
            AUDEX_TEXT_STATE_MODE_ARG: AUDEX_TEXT_STATE_APPEND_MODE,
            AUDEX_TEXT_STATE_PREFIX_TOKEN_COUNT_ARG: 12,
            AUDEX_TEXT_STATE_PREFIX_TOKEN_HASH_ARG: "hash-1",
        }
    )
    state = types.SimpleNamespace(
        token_ids=[1, 2, 3, 4],
        prompt_len=3,
        generated_tokens=1,
        cache=[object()],
        sampling_params=sampling_params,
    )
    runner = FakeMetalModelRunner()
    runner._request_states = {"req-1": state}

    runner._cleanup_finished_requests({"req-1"})

    assert cleaned == [{"req-1"}]
    assert runner._request_states == {}
    assert runner._audex_text_state_snapshots == {
        "conv-1": {
            "state_key": "conv-1",
            "request_id": "req-1",
            "mode": AUDEX_TEXT_STATE_APPEND_MODE,
            "boundary": "raw_generation_state",
            "committed_boundary_verified": False,
            "reuse_eligible": False,
            "reuse_blocked_reason": (
                "raw_generation_state_may_differ_from_committed_history"
            ),
            "prefix_token_count": 12,
            "prefix_token_hash": "hash-1",
            "prompt_len": 3,
            "token_count": 4,
            "token_hash": vllm_metal_cfg._token_hash([1, 2, 3, 4]),
            "boundary_token_count": 4,
            "boundary_token_hash": vllm_metal_cfg._token_hash([1, 2, 3, 4]),
            "generated_tokens": 1,
            "has_cache": True,
            "cache": None,
            "boundary_tokens": (),
        }
    }


def test_vllm_cfg_text_state_snapshot_marks_committed_prefill_reusable() -> None:
    prompt_tokens = [1, 2, 3]
    state = types.SimpleNamespace(
        token_ids=[*prompt_tokens, 99],
        prompt_len=len(prompt_tokens),
        generated_tokens=1,
        cache=[object()],
        sampling_params=types.SimpleNamespace(
            extra_args={
                AUDEX_TEXT_STATE_KEY_ARG: "conv-1",
                AUDEX_TEXT_STATE_MODE_ARG: AUDEX_TEXT_STATE_APPEND_MODE,
                AUDEX_TEXT_STATE_BOUNDARY_ARG: AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY,
                AUDEX_TEXT_STATE_PREFIX_TOKEN_COUNT_ARG: len(prompt_tokens),
                AUDEX_TEXT_STATE_PREFIX_TOKEN_HASH_ARG: vllm_metal_cfg._token_hash(
                    prompt_tokens
                ),
            }
        ),
    )

    metadata = vllm_metal_cfg._text_state_snapshot_metadata("req-1", state)

    assert metadata is not None
    assert metadata["boundary"] == AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY
    assert metadata["committed_boundary_verified"] is True
    assert metadata["reuse_eligible"] is True
    assert metadata["reuse_blocked_reason"] == ""
    assert metadata["boundary_token_count"] == len(prompt_tokens)
    assert metadata["boundary_token_hash"] == vllm_metal_cfg._token_hash(prompt_tokens)
    assert metadata["cache"] is state.cache
    assert metadata["boundary_tokens"] == tuple(prompt_tokens)


def test_verified_audio_prefix_snapshot_clones_only_exact_prefix() -> None:
    cached = [object()]
    runner = types.SimpleNamespace(
        _audex_text_state_snapshots={
            "conv:audio": {
                "reuse_eligible": True,
                "prefix_token_count": 3,
                "prefix_token_hash": "hash-1",
                "boundary_tokens": (1, 2, 3),
                "cache": cached,
            }
        }
    )
    module = types.SimpleNamespace(
        _merge_kv_caches=lambda caches: ("merged", caches),
        _extract_kv_cache=lambda merged, index: [merged, index],
    )
    sampling = types.SimpleNamespace(
        extra_args={
            "audex_text_state_key": "conv:audio",
            "audex_text_state_prefix_token_count": 3,
            "audex_text_state_prefix_token_hash": "hash-1",
        }
    )

    prefix_len, cloned = vllm_metal_audex_adapter._verified_audio_prefix_snapshot(
        runner,
        module,
        sampling,
        [1, 2, 3, 4, 5],
    )

    assert prefix_len == 3
    assert cloned == [("merged", [cached]), 0]
    assert vllm_metal_audex_adapter._verified_audio_prefix_snapshot(
        runner,
        module,
        sampling,
        [1, 2, 9, 4, 5],
    ) == (0, None)


def test_vllm_cfg_text_state_snapshot_ignores_missing_or_wrong_mode() -> None:
    wrong_mode = types.SimpleNamespace(
        sampling_params=types.SimpleNamespace(
            extra_args={
                AUDEX_TEXT_STATE_KEY_ARG: "conv-1",
                AUDEX_TEXT_STATE_MODE_ARG: "branch",
                AUDEX_TEXT_STATE_PREFIX_TOKEN_COUNT_ARG: 12,
                AUDEX_TEXT_STATE_PREFIX_TOKEN_HASH_ARG: "hash-1",
            }
        )
    )
    no_hint = types.SimpleNamespace(sampling_params=types.SimpleNamespace())

    assert vllm_metal_cfg._text_state_snapshot_metadata("req-1", wrong_mode) is None
    assert vllm_metal_cfg._text_state_snapshot_metadata("req-2", no_hint) is None
