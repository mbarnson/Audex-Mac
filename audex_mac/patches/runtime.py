"""Runtime monkey patches for Audex on the pinned Mac inference stack."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, replace
from types import MethodType, ModuleType
from typing import Any

MLX_LM_AUDEX_MODULES = {
    "mlx_lm.models.nemotron_dense": "audex_mac.patches.mlx_lm_nemotron_dense",
    "mlx_lm.models.nemotron_dense_audex": "audex_mac.patches.mlx_lm_nemotron_dense",
    "mlx_lm.models.nemotron_h_audex": "audex_mac.patches.mlx_lm_nemotron_h_audex",
}
AUDEX_MLX_MODULE = "audex_mac.patches.mlx_lm_nemotron_dense"
AUDEX_MLX_H_MODULE = "audex_mac.patches.mlx_lm_nemotron_h_audex"
VLLM_AUDEX_ARCHITECTURE = "NemotronDenseForCausalLM"
VLLM_PROXY_MODEL = "vllm.model_executor.models.nemotron:NemotronForCausalLM"
VLLM_AUDEX_ARCHITECTURE_ALIASES = {
    "NemotronDenseForCausalLM": "vllm.model_executor.models.nemotron:NemotronForCausalLM",
    "NemotronDenseAudexForConditionalGeneration": (
        "vllm.model_executor.models.nemotron:NemotronForCausalLM"
    ),
    "NemotronHAudexForConditionalGeneration": (
        "vllm.model_executor.models.nemotron_h:NemotronHForCausalLM"
    ),
}
VLLM_AUDEX_MODEL_INFO_SENTINEL = "_audex_mac_audex_model_info_patch"
VLLM_METAL_DEVICE_INFO_SENTINEL = "_audex_mac_device_info_patch"
VLLM_METAL_NONPAGED_CAPACITY_SENTINEL = "_audex_mac_nonpaged_capacity_patch"
NONPAGED_KV_CAPACITY_SEQS_ENV = "AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS"
LAST_PATCH_ERRORS: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class AudexPatchReport:
    transformers_local_dynamic_modules: bool
    mlx_lm_nemotron_dense: bool
    mlx_lm_nemotron_h_audex: bool
    vllm_metal_platform_repair: bool
    vllm_metal_device_info_api: bool
    vllm_metal_nonpaged_capacity: bool
    vllm_nemotron_dense: bool
    vllm_metal_audex_adapter: bool

    @property
    def ready(self) -> bool:
        return (
            self.transformers_local_dynamic_modules
            and self.mlx_lm_nemotron_dense
            and self.mlx_lm_nemotron_h_audex
            and self.vllm_metal_platform_repair
            and self.vllm_metal_device_info_api
            and self.vllm_metal_nonpaged_capacity
            and self.vllm_nemotron_dense
            and self.vllm_metal_audex_adapter
        )


def apply_audex_runtime_patches() -> AudexPatchReport:
    """Apply all Audex-owned runtime patches that are safe to repeat."""

    transformers_patch = _patch_transformers_local_dynamic_modules()
    platform_repair = _repair_vllm_metal_current_platform()
    return AudexPatchReport(
        transformers_local_dynamic_modules=transformers_patch,
        mlx_lm_nemotron_dense=_install_mlx_lm_module(AUDEX_MLX_MODULE),
        mlx_lm_nemotron_h_audex=_install_mlx_lm_module(AUDEX_MLX_H_MODULE),
        vllm_metal_platform_repair=platform_repair,
        vllm_metal_device_info_api=_patch_vllm_metal_device_info_api(),
        vllm_metal_nonpaged_capacity=_patch_vllm_metal_nonpaged_capacity_override(),
        vllm_nemotron_dense=_register_vllm_nemotron_dense(),
        vllm_metal_audex_adapter=_patch_vllm_metal_audex_adapter(),
    )


def _install_mlx_lm_module(audex_module_name: str) -> bool:
    for module_name, source_module_name in MLX_LM_AUDEX_MODULES.items():
        if source_module_name != audex_module_name:
            continue
        source_module = sys.modules.get(audex_module_name)
        if source_module is not None:
            sys.modules[module_name] = source_module
        else:
            sys.modules[module_name] = _LazyAudexModule(module_name, audex_module_name)
    return True


def _patch_transformers_local_dynamic_modules() -> bool:
    try:
        from . import transformers_dynamic_module
    except Exception as exc:
        LAST_PATCH_ERRORS["transformers_local_dynamic_modules"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return False
    try:
        patched = transformers_dynamic_module.patch_transformers_local_dynamic_modules()
    except Exception as exc:
        LAST_PATCH_ERRORS["transformers_local_dynamic_modules"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return False
    if not patched:
        LAST_PATCH_ERRORS["transformers_local_dynamic_modules"] = (
            transformers_dynamic_module.LAST_ERROR or "patch returned false"
        )
    return patched


class _LazyAudexModule(ModuleType):
    def __init__(self, alias_name: str, source_module_name: str) -> None:
        super().__init__(alias_name)
        self.__dict__["_audex_source_module_name"] = source_module_name

    def _load(self) -> ModuleType:
        source_module_name = self.__dict__["_audex_source_module_name"]
        source_module = importlib.import_module(source_module_name)
        self.__dict__.update(source_module.__dict__)
        return source_module

    def __getattr__(self, name: str) -> object:
        return getattr(self._load(), name)


def _register_vllm_nemotron_dense() -> bool:
    try:
        registry_module = importlib.import_module("vllm.model_executor.models.registry")
    except Exception as exc:
        LAST_PATCH_ERRORS["vllm_nemotron_dense"] = f"{type(exc).__name__}: {exc}"
        return False

    registry = getattr(registry_module, "ModelRegistry", None)
    if registry is None or not hasattr(registry, "register_model"):
        LAST_PATCH_ERRORS["vllm_nemotron_dense"] = "ModelRegistry unavailable"
        return False

    supported = registry.get_supported_archs()
    for architecture, proxy_model in VLLM_AUDEX_ARCHITECTURE_ALIASES.items():
        if architecture not in supported:
            registry.register_model(architecture, proxy_model)
    supported = registry.get_supported_archs()
    missing = [
        architecture
        for architecture in VLLM_AUDEX_ARCHITECTURE_ALIASES
        if architecture not in supported
    ]
    if missing:
        LAST_PATCH_ERRORS["vllm_nemotron_dense"] = (
            "missing architecture aliases: " + ", ".join(missing)
        )
        return False
    return _patch_vllm_audex_model_info(registry)


def _patch_vllm_audex_model_info(registry: Any) -> bool:
    if getattr(registry, VLLM_AUDEX_MODEL_INFO_SENTINEL, False):
        return True
    original_try_inspect = getattr(registry, "_try_inspect_model_cls", None)
    if original_try_inspect is None:
        return True

    def try_inspect_model_cls_with_audex_multimodal(
        self: Any,
        model_arch: str,
    ) -> Any:
        model_info = original_try_inspect(model_arch)
        if model_arch in VLLM_AUDEX_ARCHITECTURE_ALIASES and model_info is not None:
            return replace(
                model_info,
                supports_multimodal=True,
                supports_multimodal_raw_input_only=False,
                requires_raw_input_tokens=False,
            )
        return model_info

    registry._try_inspect_model_cls = MethodType(  # noqa: SLF001
        try_inspect_model_cls_with_audex_multimodal,
        registry,
    )
    setattr(registry, VLLM_AUDEX_MODEL_INFO_SENTINEL, True)
    return True


def _repair_vllm_metal_current_platform() -> bool:
    expected = "vllm_metal.platform.MetalPlatform"
    try:
        import vllm.platforms as platforms
    except Exception as exc:
        LAST_PATCH_ERRORS["vllm_metal_platform_repair"] = f"{type(exc).__name__}: {exc}"
        return False

    try:
        resolved = platforms.resolve_current_platform_cls_qualname()
    except Exception as exc:
        LAST_PATCH_ERRORS["vllm_metal_platform_repair"] = f"{type(exc).__name__}: {exc}"
        return False
    if resolved != expected:
        LAST_PATCH_ERRORS["vllm_metal_platform_repair"] = (
            f"resolved platform is {resolved!r}, not {expected!r}"
        )
        return False

    current = getattr(platforms, "_current_platform", None)
    current_qualname = (
        f"{type(current).__module__}.{type(current).__name__}"
        if current is not None
        else None
    )
    if current_qualname == expected:
        return True

    try:
        platforms._current_platform = platforms.resolve_obj_by_qualname(
            expected
        )()  # noqa: SLF001
    except Exception as exc:
        LAST_PATCH_ERRORS["vllm_metal_platform_repair"] = f"{type(exc).__name__}: {exc}"
        return False
    return True


def _patch_vllm_metal_device_info_api() -> bool:
    try:
        utils = importlib.import_module("vllm_metal.utils")
    except Exception as exc:
        LAST_PATCH_ERRORS["vllm_metal_device_info_api"] = f"{type(exc).__name__}: {exc}"
        return False

    current = getattr(utils, "set_wired_limit", None)
    if current is None:
        LAST_PATCH_ERRORS["vllm_metal_device_info_api"] = (
            "vllm_metal.utils.set_wired_limit unavailable"
        )
        return False
    if getattr(current, VLLM_METAL_DEVICE_INFO_SENTINEL, False):
        return True

    logger = getattr(utils, "logger", None)

    def set_wired_limit_with_current_mlx_api() -> None:
        try:
            import mlx.core as mx

            get_device_info = getattr(mx, "device_info", None)
            if callable(get_device_info):
                device_info = get_device_info()
            else:
                device_info = mx.metal.device_info()
            max_wired = int(device_info.get("max_recommended_working_set_size", 0))
            if max_wired > 0:
                mx.set_wired_limit(max_wired)
                if logger is not None:
                    logger.info(
                        "Set Metal wired_limit to %.1f GB",
                        max_wired / (1024**3),
                    )
        except Exception as exc:
            if logger is not None:
                logger.warning("Failed to set wired_limit: %s", exc)

    setattr(
        set_wired_limit_with_current_mlx_api,
        VLLM_METAL_DEVICE_INFO_SENTINEL,
        True,
    )
    set_wired_limit_with_current_mlx_api.__wrapped__ = current  # type: ignore[attr-defined]
    utils.set_wired_limit = set_wired_limit_with_current_mlx_api
    return True


def _patch_vllm_metal_nonpaged_capacity_override() -> bool:
    try:
        cache_policy = importlib.import_module("vllm_metal.v1.cache_policy")
    except Exception as exc:
        LAST_PATCH_ERRORS["vllm_metal_nonpaged_capacity"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return False

    planner_cls = getattr(cache_policy, "WorkerCachePlanner", None)
    if planner_cls is None:
        LAST_PATCH_ERRORS["vllm_metal_nonpaged_capacity"] = (
            "WorkerCachePlanner unavailable"
        )
        return False
    current = getattr(planner_cls, "determine_available_memory", None)
    if not callable(current):
        LAST_PATCH_ERRORS["vllm_metal_nonpaged_capacity"] = (
            "determine_available_memory unavailable"
        )
        return False
    if getattr(current, VLLM_METAL_NONPAGED_CAPACITY_SENTINEL, False):
        return True

    def determine_available_memory_with_audex_capacity(self: Any) -> int:
        capacity_seqs = _nonpaged_kv_capacity_seqs_override()
        if capacity_seqs is None:
            return current(self)

        mode = self._worker.model_runner.scheduler_memory_reporting_mode(
            paged_attention_enabled=self._worker.metal_config.use_paged_attention
        )
        if mode != "single_sequence_estimate":
            return current(self)

        one_sequence_bytes = self._worker._one_sequence_kv_bytes()
        available = one_sequence_bytes * capacity_seqs
        budget = _nonpaged_capacity_budget(self._worker)
        logger = getattr(cache_policy, "logger", None)
        log = getattr(logger, "warning", None)
        if budget is not None:
            metal_headroom_bytes = budget["metal_headroom_bytes"]
            headroom_message = (
                "Audex-Mac: non-paged capacity headroom "
                f"metal_limit={budget['metal_limit_bytes'] / 1e9:.2f} GB "
                f"active={budget['active_bytes'] / 1e9:.2f} GB "
                f"cache={budget['cache_bytes'] / 1e9:.2f} GB "
                f"metal_headroom={metal_headroom_bytes / 1e9:.2f} GB "
                f"gpu_memory_utilization={budget['gpu_memory_utilization']:.2f} "
                "gpu_utilization_headroom="
                f"{budget['gpu_utilization_headroom_bytes'] / 1e9:.2f} GB "
                f"requested_worst_case={available / 1e9:.2f} GB"
            )
            print(headroom_message, file=sys.stderr, flush=True)
            if callable(log):
                log(
                    headroom_message,
                )
            if available > metal_headroom_bytes:
                raise RuntimeError(
                    "Audex-Mac: refusing "
                    f"{NONPAGED_KV_CAPACITY_SEQS_ENV}={capacity_seqs} because "
                    "the max-length non-paged KV worst case exceeds the "
                    "current MLX/Metal headroom "
                    f"({available / 1e9:.2f} GB requested > "
                    f"{metal_headroom_bytes / 1e9:.2f} GB headroom). "
                    "Lower the capacity override or reduce concurrent CFG "
                    "segments before continuing."
                )
        if callable(log):
            log(
                "Audex-Mac: overriding non-paged scheduler capacity to %d "
                "max-length sequences via %s (%d × %.2f GB = %.2f GB)",
                capacity_seqs,
                NONPAGED_KV_CAPACITY_SEQS_ENV,
                capacity_seqs,
                one_sequence_bytes / 1e9,
                available / 1e9,
            )
        return available

    setattr(
        determine_available_memory_with_audex_capacity,
        VLLM_METAL_NONPAGED_CAPACITY_SENTINEL,
        True,
    )
    determine_available_memory_with_audex_capacity.__wrapped__ = current  # type: ignore[attr-defined]
    planner_cls.determine_available_memory = (
        determine_available_memory_with_audex_capacity
    )
    return True


def _nonpaged_capacity_budget(worker: Any) -> dict[str, float] | None:
    try:
        mx = importlib.import_module("mlx.core")
    except Exception:
        return None

    device_info = getattr(mx, "device_info", None)
    if not callable(device_info):
        return None
    metal_limit = int(device_info().get("max_recommended_working_set_size", 0))
    if metal_limit <= 0:
        return None

    _flush_mlx(mx)
    active_bytes = _mlx_memory_bytes(mx, "get_active_memory")
    cache_bytes = _mlx_memory_bytes(mx, "get_cache_memory")
    gpu_memory_utilization = _worker_gpu_memory_utilization(worker)
    gpu_utilization_bytes = int(metal_limit * gpu_memory_utilization)
    allocated_bytes = active_bytes + cache_bytes
    metal_headroom_bytes = metal_limit - allocated_bytes
    gpu_utilization_headroom_bytes = gpu_utilization_bytes - allocated_bytes
    return {
        "metal_limit_bytes": float(metal_limit),
        "gpu_memory_utilization": gpu_memory_utilization,
        "gpu_utilization_bytes": float(gpu_utilization_bytes),
        "active_bytes": float(active_bytes),
        "cache_bytes": float(cache_bytes),
        "metal_headroom_bytes": float(metal_headroom_bytes),
        "gpu_utilization_headroom_bytes": float(gpu_utilization_headroom_bytes),
    }


def _flush_mlx(mx: Any) -> None:
    array = getattr(mx, "array", None)
    evaluate = getattr(mx, "eval", None)
    if not callable(array) or not callable(evaluate):
        return
    try:
        evaluate(array([0]))
    except Exception:
        return


def _mlx_memory_bytes(mx: Any, method_name: str) -> int:
    method = getattr(mx, method_name, None)
    if not callable(method):
        return 0
    try:
        return max(0, int(method()))
    except Exception:
        return 0


def _worker_gpu_memory_utilization(worker: Any) -> float:
    cache_config = getattr(getattr(worker, "vllm_config", None), "cache_config", None)
    value = getattr(cache_config, "gpu_memory_utilization", None)
    if isinstance(value, int | float) and 0.0 < float(value) <= 1.0:
        return float(value)
    return 1.0


def _nonpaged_kv_capacity_seqs_override() -> int | None:
    value = os.environ.get(NONPAGED_KV_CAPACITY_SEQS_ENV)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{NONPAGED_KV_CAPACITY_SEQS_ENV} must be a positive integer, "
            f"got {value!r}"
        ) from exc
    if parsed <= 0:
        raise ValueError(
            f"{NONPAGED_KV_CAPACITY_SEQS_ENV} must be a positive integer, "
            f"got {value!r}"
        )
    return parsed


def _patch_vllm_metal_audex_adapter() -> bool:
    try:
        from . import vllm_metal_audex_adapter
    except Exception as exc:
        LAST_PATCH_ERRORS["vllm_metal_audex_adapter"] = f"{type(exc).__name__}: {exc}"
        return False
    try:
        patched = vllm_metal_audex_adapter.patch_default_model_adapter()
    except Exception as exc:
        LAST_PATCH_ERRORS["vllm_metal_audex_adapter"] = f"{type(exc).__name__}: {exc}"
        return False
    if not patched:
        LAST_PATCH_ERRORS["vllm_metal_audex_adapter"] = (
            vllm_metal_audex_adapter.LAST_ERROR or "patch returned false"
        )
    return patched
