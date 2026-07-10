"""Diagnostics for Audex-Mac's vLLM Metal runtime path."""

from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .audio_runtime import preflight_audio_runtime
from .metal_policy import REQUIRED_METAL_ENV, enforce_metal_env
from .models import AudexModel
from .text_runtime import preflight_text_runtime
from .vllm_streaming import inspect_vllm_streaming_support

RUNS_DIR = Path(__file__).resolve().parents[1] / ".audex" / "runs"
DEFAULT_STS_SMOKE_SPEECH_MAX_TOKENS = 256
DEFAULT_STS_SMOKE_MIN_RESPONSE_WORDS = 8
DIAGNOSTIC_ENV_KEYS = (
    "VLLM_METAL_USE_MLX",
    "VLLM_MLX_DEVICE",
    "VLLM_METAL_USE_PAGED_ATTENTION",
    "VLLM_METAL_MULTIMODAL_MODE",
    "VLLM_METAL_MEMORY_FRACTION",
    "VLLM_LOGGING_LEVEL",
    "VLLM_PLUGINS",
    "AUDEX_MAC_AUTO_PATCHES",
    "AUDEX_STS_SMOKE_TIMEOUT_SECONDS",
    "AUDEX_STS_SMOKE_SPEECH_MAX_TOKENS",
    "AUDEX_VLLM_NATIVE_SAMPLING_DEBUG",
    "AUDEX_VLLM_ENABLE_CFG_WIRING",
    "AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL",
    "AUDEX_VLLM_CFG_MAX_NUM_SEQS",
    "AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS",
    "AUDEX_VLLM_MATERIALIZE_DECODE_LOGITS",
    "TRANSFORMERS_VERBOSITY",
    "PYTHONPATH",
)
SOURCE_SCAN_PATTERNS = (
    "mx.cpu",
    "DeviceType.cpu",
    'device_type: str = "cpu"',
    "device_config",
    "use_paged_attention",
    "mlx_device",
)
APPLE_SYSCTL_KEYS = (
    "hw.optional.arm64",
    "hw.memsize",
    "hw.pagesize",
    "hw.cachelinesize",
    "hw.physicalcpu_max",
    "hw.logicalcpu_max",
    "hw.nperflevels",
    "sysctl.proc_translated",
)
APPLE_PERFLEVEL_SYSCTL_KEYS = (
    "cpusperl2",
    "cpusperl3",
    "l1dcachesize",
    "l1icachesize",
    "l2cachesize",
    "l3cachesize",
    "logicalcpu",
    "logicalcpu_max",
    "physicalcpu",
    "physicalcpu_max",
)
PAGED_TIMING_RE = re.compile(
    r"Audex vLLM Metal: paged sample timing "
    r"count=(?P<count>\d+) "
    r"avg_ms=(?P<avg_ms>[0-9.]+) "
    r"last_ms=(?P<last_ms>[0-9.]+) "
    r"decode_reqs=(?P<decode_reqs>\d+) "
    r"prefill_reqs=(?P<prefill_reqs>\d+) "
    r"decode_tokens=(?P<decode_tokens>\d+) "
    r"native_sample_ms=(?P<native_sample_ms>[0-9.]+)"
    r"(?: native_sampled_rows=(?P<native_sampled_rows>\d+))?"
    r"(?: native_output_rows=(?P<native_output_rows>\d+))?"
    r"(?: skipped_logits_eval=(?P<skipped_logits_eval>\d+))?"
    r"(?: native_detail_ms=(?P<native_detail_ms>\S+))?"
    r"(?: mx_eval_ms=(?P<mx_eval_ms>\S+))?"
    r"(?: mx_eval_shapes=(?P<mx_eval_shapes>\S+))?"
)
NON_PAGED_TIMING_RE = re.compile(
    r"Audex vLLM Metal: nonpaged decode timing "
    r"count=(?P<count>\d+) "
    r"avg_ms=(?P<avg_ms>[0-9.]+) "
    r"last_ms=(?P<last_ms>[0-9.]+) "
    r"decode_reqs=(?P<decode_reqs>\d+) "
    r"cached_reqs=(?P<cached_reqs>\d+) "
    r"batched=(?P<batched>[01]) "
    r"native_sample_ms=(?P<native_sample_ms>[0-9.]+)"
    r"(?: cfg_cond_reqs=(?P<cfg_cond_reqs>\d+))?"
    r"(?: cfg_uncond_reqs=(?P<cfg_uncond_reqs>\d+))?"
    r"(?: cfg_complete_pairs=(?P<cfg_complete_pairs>\d+))?"
    r"(?: native_sampled_rows=(?P<native_sampled_rows>\d+))?"
    r"(?: native_output_rows=(?P<native_output_rows>\d+))?"
    r"(?: tts_window_decode_count=(?P<tts_window_decode_count>\d+))?"
    r"(?: tts_window_weight_cache_hits=(?P<tts_window_weight_cache_hits>\d+))?"
    r"(?: tts_window_weight_cache_misses=(?P<tts_window_weight_cache_misses>\d+))?"
    r"(?: nonpaged_persistent_cache_hits=(?P<nonpaged_persistent_cache_hits>\d+))?"
    r"(?: nonpaged_persistent_cache_misses=(?P<nonpaged_persistent_cache_misses>\d+))?"
    r"(?: nonpaged_persistent_cache_flushes=(?P<nonpaged_persistent_cache_flushes>\d+))?"
    r"(?: native_detail_ms=(?P<native_detail_ms>\S+))?"
)
NATIVE_SAMPLING_USED_RE = re.compile(
    r"Audex vLLM Metal: native MLX sampling fast path used " r"(?P<count>\d+) time\(s\)"
)
NATIVE_SAMPLING_SKIPPED_RE = re.compile(
    r"Audex vLLM Metal: native MLX sampling fast path skipped: (?P<reason>.+)"
)


@dataclass(frozen=True, slots=True)
class SourceMatch:
    file: str
    line: int
    pattern: str
    text: str


@dataclass(frozen=True, slots=True)
class VllmMetalDiagnosticResult:
    run_log_path: Path
    report: dict[str, Any]


def run_vllm_metal_diagnostics(
    model: AudexModel,
    *,
    run_generation: bool = False,
    generation_max_tokens: int | None = None,
    run_sts_smoke: bool = False,
    sts_audio_fixture: Path | None = None,
    sts_play_audio: bool = False,
    sts_speech_max_tokens: int | None = None,
    native_sampling_debug: bool = False,
    tts_batch_size: int | None = None,
    tts_batch_max_tokens: int = 128,
    tts_batch_text: str = "Please explain Python context managers in two concise sentences.",
    tts_batch_cfg: bool = True,
    source_root: Path | None = None,
    output_dir: Path = RUNS_DIR,
) -> VllmMetalDiagnosticResult:
    """Collect fast evidence about vLLM Metal device/runtime selection."""

    started_at = time.time()
    metal_policy = enforce_metal_env()
    text_preflight = preflight_text_runtime(model, apply_patches=False)
    run_tts_batch = tts_batch_size is not None and tts_batch_size > 0
    speech_preflight = (
        preflight_audio_runtime(model) if run_sts_smoke or run_tts_batch else None
    )
    diagnostic_model_path = (
        speech_preflight.model_path
        if speech_preflight is not None and speech_preflight.model_path is not None
        else text_preflight.model_path
    )
    source_root = source_root or _default_vllm_metal_source_root()

    report: dict[str, Any] = {
        "schema": "audex-mac.vllm-metal-diagnostic.v1",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selected_model": model.repo_id,
        "model_path": str(diagnostic_model_path) if diagnostic_model_path else None,
        "parent_process": {
            "python": sys.executable,
            "argv": sys.argv,
            "env": _selected_env(os.environ),
            "apple_silicon_topology": _probe_apple_silicon_topology(),
            "metal_policy": {
                "env": metal_policy.env,
                "mlx_metal_available": None,
                "mlx_default_device": None,
                "ready": metal_policy.ready,
            },
        },
        "spawn_probe": _probe_spawned_worker(),
        "platform_resolution_probe": _probe_platform_resolution_subprocess(),
        "vllm_metal": _probe_vllm_metal_modules(),
        "vllm_streaming_api": asdict(inspect_vllm_streaming_support()),
        "audex_patches": _probe_audex_patches_subprocess(),
        "audex_processor": _probe_audex_processor_subprocess(),
        "audex_cfg": _probe_audex_cfg_subprocess(diagnostic_model_path),
        "text_runtime": {
            "model_path": (
                str(text_preflight.model_path) if text_preflight.model_path else None
            ),
            "ready": text_preflight.ready,
            "missing_items": list(text_preflight.missing_items),
            "dependency_checks": [
                asdict(check) for check in text_preflight.dependency_checks
            ],
        },
        "speech_runtime": (
            _speech_runtime_report(speech_preflight)
            if speech_preflight is not None
            else {"enabled": False}
        ),
        "model_adapter": _probe_model_adapter(model, diagnostic_model_path),
        "generation_probe": (
            _probe_vllm_generation(text_preflight, max_tokens=generation_max_tokens)
            if run_generation
            else {"enabled": False}
        ),
        "sts_probe": (
            _probe_vllm_sts_default_runtime(
                model,
                audio_fixture=sts_audio_fixture,
                play_audio=sts_play_audio,
                speech_max_tokens=(
                    sts_speech_max_tokens
                    if sts_speech_max_tokens is not None
                    else _sts_smoke_speech_max_tokens(sts_play_audio)
                ),
                native_sampling_debug=native_sampling_debug,
            )
            if run_sts_smoke
            else {"enabled": False}
        ),
        "tts_batch_probe": (
            _probe_vllm_tts_batch_runtime(
                model,
                batch_size=int(tts_batch_size or 0),
                max_tokens=tts_batch_max_tokens,
                text=tts_batch_text,
                use_cfg=tts_batch_cfg,
                native_sampling_debug=native_sampling_debug,
            )
            if run_tts_batch
            else {"enabled": False}
        ),
        "source_scan": {
            "root": str(source_root) if source_root is not None else None,
            "matches": [
                asdict(match) for match in _scan_vllm_metal_sources(source_root)
            ],
        },
        "interpretation": _interpret_expected_cpu_facade(),
    }
    report["verdict"] = _diagnostic_verdict(
        report,
        require_generation=run_generation,
        require_sts=run_sts_smoke,
        require_tts_batch=run_tts_batch,
    )
    report["elapsed_seconds"] = round(time.time() - started_at, 3)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = (
        output_dir / f"vllm-metal-diagnostic-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    run_log_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return VllmMetalDiagnosticResult(run_log_path=run_log_path, report=report)


def _selected_env(env: os._Environ[str] | dict[str, str]) -> dict[str, str | None]:
    return {name: env.get(name) for name in DIAGNOSTIC_ENV_KEYS}


def _probe_apple_silicon_topology() -> dict[str, Any]:
    """Capture Apple-recommended sysctl topology evidence for CPU contention."""

    if sys.platform != "darwin":
        return {
            "enabled": False,
            "reason": f"sys.platform={sys.platform}",
        }

    values: dict[str, int | str] = {}
    errors: dict[str, str] = {}
    for key in APPLE_SYSCTL_KEYS:
        value, error = _sysctl_value(key)
        if error is not None:
            errors[key] = error
        elif value is not None:
            values[key] = value

    perflevels: list[dict[str, Any]] = []
    nperflevels = values.get("hw.nperflevels")
    if isinstance(nperflevels, int):
        for index in range(nperflevels):
            level: dict[str, Any] = {"index": index}
            for suffix in APPLE_PERFLEVEL_SYSCTL_KEYS:
                key = f"hw.perflevel{index}.{suffix}"
                value, error = _sysctl_value(key)
                if error is not None:
                    level.setdefault("errors", {})[suffix] = error
                elif value is not None:
                    level[suffix] = value
            perflevels.append(level)

    return {
        "enabled": True,
        "source": "Apple-Silicon-CPU-Optimization-Guide.pdf Appendix B",
        "sysctl": values,
        "sysctl_errors": errors,
        "perflevels": perflevels,
        "interpretation": (
            "Use these values to size and interpret CPU-side scheduler, queue, "
            "and synchronization work; Activity Monitor CPU/GPU percentages "
            "alone do not distinguish expected vLLM Metal host work from MLX "
            "CPU fallback."
        ),
    }


def _sysctl_value(name: str) -> tuple[int | str | None, str | None]:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        return None, error or f"sysctl exited {completed.returncode}"
    raw = completed.stdout.strip()
    if not raw:
        return "", None
    try:
        return int(raw), None
    except ValueError:
        return raw, None


def _speech_runtime_report(preflight: Any) -> dict[str, Any]:
    return {
        "enabled": True,
        "ready": preflight.ready,
        "model_path": str(preflight.model_path) if preflight.model_path else None,
        "decoder_path": str(preflight.decoder_path) if preflight.decoder_path else None,
        "missing_items": list(preflight.missing_items),
    }


def _probe_spawned_worker() -> dict[str, Any]:
    code = """
import json
import os

keys = (
    "VLLM_METAL_USE_MLX",
    "VLLM_MLX_DEVICE",
    "VLLM_METAL_USE_PAGED_ATTENTION",
    "VLLM_METAL_MULTIMODAL_MODE",
    "VLLM_METAL_MEMORY_FRACTION",
    "AUDEX_MAC_AUTO_PATCHES",
    "PYTHONPATH",
)
result = {"env": {name: os.environ.get(name) for name in keys}}
try:
    import mlx.core as mx
    mx.set_default_device(mx.Device(mx.DeviceType.gpu))
    probe = mx.array([1.0])
    mx.eval(probe)
    result["mlx"] = {
        "metal_available": bool(mx.metal.is_available()),
        "default_device": str(mx.default_device()),
        "probe_array_device": str(getattr(probe, "device", mx.default_device())),
    }
except Exception as exc:
    result["mlx_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=15,
    )
    result: dict[str, Any] = {
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }
    try:
        result.update(json.loads(completed.stdout))
    except json.JSONDecodeError:
        result["stdout"] = completed.stdout.strip()
    return result


def _probe_platform_resolution_subprocess() -> dict[str, Any]:
    code = """
import json
import os

result = {
    "env": {
        "VLLM_PLUGINS": os.environ.get("VLLM_PLUGINS"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
    }
}
try:
    import importlib.metadata as metadata
    result["entry_points"] = [
        [ep.name, ep.value]
        for ep in metadata.entry_points(group="vllm.platform_plugins")
    ]
except Exception as exc:
    result["entry_points_error"] = f"{type(exc).__name__}: {exc}"
try:
    import vllm.platforms as platforms
    result["resolved_platform_qualname"] = platforms.resolve_current_platform_cls_qualname()
    current_platform = platforms.current_platform
    result["current_platform"] = {
        "class": f"{type(current_platform).__module__}.{type(current_platform).__name__}",
        "device_type": getattr(current_platform, "device_type", None),
        "device_name": getattr(current_platform, "device_name", None),
        "ray_device_key": getattr(current_platform, "ray_device_key", None),
    }
    result["init_trace_contains_vllm_metal"] = "vllm_metal" in getattr(platforms, "_init_trace", "")
except Exception as exc:
    result["platform_error"] = f"{type(exc).__name__}: {exc}"
try:
    from audex_mac.patches.runtime import apply_audex_runtime_patches
    apply_audex_runtime_patches()
    import vllm.platforms as repaired_platforms
    repaired_current_platform = repaired_platforms.current_platform
    result["current_platform_after_audex_patches"] = {
        "class": f"{type(repaired_current_platform).__module__}.{type(repaired_current_platform).__name__}",
        "device_type": getattr(repaired_current_platform, "device_type", None),
        "device_name": getattr(repaired_current_platform, "device_name", None),
        "ray_device_key": getattr(repaired_current_platform, "ray_device_key", None),
    }
except Exception as exc:
    result["current_platform_after_audex_patches_error"] = f"{type(exc).__name__}: {exc}"
try:
    import vllm_metal
    result["direct_vllm_metal_register"] = vllm_metal._register()
except Exception as exc:
    result["direct_vllm_metal_register_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=25,
    )
    result: dict[str, Any] = {
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }
    try:
        result.update(json.loads(completed.stdout))
    except json.JSONDecodeError:
        result["stdout"] = completed.stdout.strip()
    return result


def _probe_audex_patches_subprocess() -> dict[str, Any]:
    code = """
import json
import os
import sys
from dataclasses import asdict

try:
    from audex_mac.patches import runtime
    report = asdict(runtime.apply_audex_runtime_patches())
    if runtime.LAST_PATCH_ERRORS:
        report["errors"] = runtime.LAST_PATCH_ERRORS
except Exception as exc:
    report = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(report))
sys.stdout.flush()
os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=25,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {"stdout": completed.stdout.strip()}
    result["returncode"] = completed.returncode
    stderr = completed.stderr.strip()
    if stderr:
        result["stderr"] = stderr
    return result


def _probe_audex_processor_subprocess() -> dict[str, Any]:
    code = """
import json
import os
import sys
from types import SimpleNamespace

try:
    from audex_mac.audio_contract import SOUND_END_TOKEN, SOUND_START_TOKEN, SOUND_TOKEN
    from audex_mac.patches.runtime import apply_audex_runtime_patches
    from audex_mac.patches.vllm_metal_audex_adapter import (
        AudexDummyInputsBuilder,
        AudexProcessingInfo,
        AudexProjectedAudioDataParser,
        AudexProjectedAudioProcessor,
        NEMOTRON_MODULE,
        PROCESSOR_IMPORT_HOOK_SENTINEL,
    )

    patch_report = apply_audex_runtime_patches()

    class FakeProjectedEmbeddings:
        shape = (3, 2048)

    class FakeTokenizer:
        def get_vocab(self):
            return {SOUND_TOKEN: 29, SOUND_START_TOKEN: 30, SOUND_END_TOKEN: 31}

    ctx = SimpleNamespace(
        model_config=SimpleNamespace(model="audex-processor-probe"),
        get_tokenizer=lambda: FakeTokenizer(),
    )
    info = AudexProcessingInfo(ctx)
    processor = AudexProjectedAudioProcessor(
        info,
        AudexDummyInputsBuilder(info),
    )
    mm_items = AudexProjectedAudioDataParser().parse_mm_data(
        {"audio": [{"audex_projected_embeddings": FakeProjectedEmbeddings()}]}
    )
    output = processor.apply(
        SimpleNamespace(
            prompt=[1, 29, 29, 29, 2],
            mm_data_items=mm_items,
            mm_uuid_items={"audio": ["audex-processor-probe-audio"]},
        ),
        timing_ctx=SimpleNamespace(),
    )
    placeholder = output["mm_placeholders"]["audio"][0]
    result = {
        "ready": True,
        "patch_ready": bool(patch_report.ready),
        "processor_class": (
            f"{AudexProjectedAudioProcessor.__module__}."
            f"{AudexProjectedAudioProcessor.__name__}"
        ),
        "nemotron_module_loaded": NEMOTRON_MODULE in sys.modules,
        "nemotron_import_hook_installed": any(
            getattr(finder, PROCESSOR_IMPORT_HOOK_SENTINEL, False)
            for finder in sys.meta_path
        ),
        "output": {
            "type": output["type"],
            "prompt_token_ids": output["prompt_token_ids"],
            "mm_modalities": sorted(output["mm_kwargs"].keys()),
            "audio_mm_keys": sorted(output["mm_kwargs"]["audio"][0].keys()),
            "audio_hashes": output["mm_hashes"]["audio"],
            "placeholder_offset": int(placeholder.offset),
            "placeholder_length": int(placeholder.length),
            "placeholder_embeds": int(placeholder.get_num_embeds()),
        },
    }
except Exception as exc:
    result = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}

print(json.dumps(result))
sys.stdout.flush()
os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=25,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {
            "ready": False,
            "stdout": completed.stdout.strip(),
        }
    result["returncode"] = completed.returncode
    stderr = completed.stderr.strip()
    if stderr:
        result["stderr"] = stderr
    if completed.returncode != 0 and "error" not in result:
        result["ready"] = False
        result["error"] = f"processor subprocess exited {completed.returncode}"
    return result


def _probe_audex_cfg_subprocess(model_path: Path | None) -> dict[str, Any]:
    if model_path is None:
        return {
            "enabled": False,
            "ready": False,
            "reason": "model path unavailable",
        }

    code = """
import json
import os
import sys
from pathlib import Path

result = {"enabled": True}
try:
    from audex_mac.vllm_cfg import configure_audex_vllm_cfg
    from audex_mac.patches.vllm_metal_cfg import apply_vllm_metal_cfg_patches

    engine_kwargs = {"enable_prefix_caching": True}
    config = configure_audex_vllm_cfg(engine_kwargs, Path(sys.argv[1]))
    metal_patch_report = apply_vllm_metal_cfg_patches()
    result.update(
        {
            "ready": config.ready,
            "script_dir": str(config.script_dir) if config.script_dir else None,
            "logits_processors": list(config.logits_processors),
            "vllm_metal_patch": {
                "ready": metal_patch_report.ready,
                "sample_from_logits": metal_patch_report.sample_from_logits,
                "sample_prefill_tokens": metal_patch_report.sample_prefill_tokens,
                "model_runner_symbols": metal_patch_report.model_runner_symbols,
                "error": metal_patch_report.error,
            },
            "max_model_len": config.max_model_len,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "max_num_seqs": config.max_num_seqs,
            "enable_prefix_caching": engine_kwargs.get("enable_prefix_caching"),
            "error": config.error,
        }
    )
except Exception as exc:
    result.update({"ready": False, "error": f"{type(exc).__name__}: {exc}"})

print(json.dumps(result))
sys.stdout.flush()
os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(model_path)],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=25,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {
            "enabled": True,
            "ready": False,
            "stdout": completed.stdout.strip(),
        }
    result["returncode"] = completed.returncode
    stderr = completed.stderr.strip()
    if stderr:
        result["stderr"] = stderr
    if completed.returncode != 0 and "error" not in result:
        result["ready"] = False
        result["error"] = f"CFG subprocess exited {completed.returncode}"
    return result


def _probe_vllm_metal_modules() -> dict[str, Any]:
    code = r"""
import json

details = {
    "imports": {},
    "platform": {},
    "config": {},
    "mlx": {},
}
try:
    import mlx.core as mx

    probe = mx.array([1.0])
    mx.eval(probe)
    details["mlx"] = {
        "metal_available": bool(mx.metal.is_available()),
        "default_device": str(mx.default_device()),
        "probe_array_device": str(getattr(probe, "device", mx.default_device())),
    }
    details["imports"]["mlx.core"] = True
except Exception as exc:
    details["imports"]["mlx.core"] = False
    details["mlx"]["error"] = f"{type(exc).__name__}: {exc}"

try:
    import vllm_metal
    from vllm_metal.config import get_config
    from vllm_metal.platform import MetalPlatform

    config = get_config()
    details["imports"]["vllm_metal"] = True
    details["vllm_metal_file"] = getattr(vllm_metal, "__file__", None)
    details["config"] = {
        "use_mlx": config.use_mlx,
        "mlx_device": config.mlx_device,
        "use_paged_attention": config.use_paged_attention,
        "multimodal_mode": config.multimodal_mode,
        "memory_fraction": config.memory_fraction,
    }
    details["platform"] = {
        "class": f"{MetalPlatform.__module__}.{MetalPlatform.__name__}",
        "is_available": MetalPlatform.is_available(),
        "device_type_facade": MetalPlatform.device_type,
        "device_name_facade": MetalPlatform.device_name,
        "ray_device_key": MetalPlatform.ray_device_key,
        "device_name": MetalPlatform.get_device_name(),
    }
except Exception as exc:
    details["imports"]["vllm_metal"] = False
    details["platform"]["error"] = f"{type(exc).__name__}: {exc}"

try:
    from vllm.platforms import current_platform

    details["imports"]["vllm.platforms"] = True
    details["current_platform"] = {
        "class": f"{type(current_platform).__module__}.{type(current_platform).__name__}",
        "device_type": getattr(current_platform, "device_type", None),
        "device_name": getattr(current_platform, "device_name", None),
        "ray_device_key": getattr(current_platform, "ray_device_key", None),
    }
except Exception as exc:
    details["imports"]["vllm.platforms"] = False
    details["current_platform_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(details))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=25,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {"stdout": completed.stdout.strip()}
    result["returncode"] = completed.returncode
    stderr = completed.stderr.strip()
    if stderr:
        result["stderr"] = stderr
    return result


def _probe_vllm_metal_modules_inline() -> dict[str, Any]:
    details: dict[str, Any] = {
        "imports": {},
        "platform": {},
        "config": {},
        "mlx": {},
    }
    try:
        import mlx.core as mx

        probe = mx.array([1.0])
        mx.eval(probe)
        details["mlx"] = {
            "metal_available": bool(mx.metal.is_available()),
            "default_device": str(mx.default_device()),
            "probe_array_device": str(getattr(probe, "device", mx.default_device())),
        }
        details["imports"]["mlx.core"] = True
    except Exception as exc:
        details["imports"]["mlx.core"] = False
        details["mlx"]["error"] = f"{type(exc).__name__}: {exc}"

    try:
        import vllm_metal
        from vllm_metal.config import get_config
        from vllm_metal.platform import MetalPlatform

        config = get_config()
        details["imports"]["vllm_metal"] = True
        details["vllm_metal_file"] = getattr(vllm_metal, "__file__", None)
        details["config"] = {
            "use_mlx": config.use_mlx,
            "mlx_device": config.mlx_device,
            "use_paged_attention": config.use_paged_attention,
            "multimodal_mode": config.multimodal_mode,
            "memory_fraction": config.memory_fraction,
        }
        details["platform"] = {
            "class": f"{MetalPlatform.__module__}.{MetalPlatform.__name__}",
            "is_available": MetalPlatform.is_available(),
            "device_type_facade": MetalPlatform.device_type,
            "device_name_facade": MetalPlatform.device_name,
            "ray_device_key": MetalPlatform.ray_device_key,
            "device_name": MetalPlatform.get_device_name(),
        }
    except Exception as exc:
        details["imports"]["vllm_metal"] = False
        details["platform"]["error"] = f"{type(exc).__name__}: {exc}"

    try:
        from vllm.platforms import current_platform

        details["imports"]["vllm.platforms"] = True
        details["current_platform"] = {
            "class": f"{type(current_platform).__module__}.{type(current_platform).__name__}",
            "device_type": getattr(current_platform, "device_type", None),
            "device_name": getattr(current_platform, "device_name", None),
            "ray_device_key": getattr(current_platform, "ray_device_key", None),
        }
    except Exception as exc:
        details["imports"]["vllm.platforms"] = False
        details["current_platform_error"] = f"{type(exc).__name__}: {exc}"
    return details


def _probe_model_adapter(model: AudexModel, model_path: Path | None) -> dict[str, Any]:
    code = """
import json
import os
import sys
from pathlib import Path

from audex_mac.models import SUPPORTED_MODELS
from audex_mac.patches.runtime import apply_audex_runtime_patches
from audex_mac.vllm_diagnostics import _probe_model_adapter_inline

repo_id = sys.argv[1]
model_path_arg = sys.argv[2]
model = next(model for model in SUPPORTED_MODELS if model.repo_id == repo_id)
model_path = None if model_path_arg == "" else Path(model_path_arg)
apply_audex_runtime_patches()
print(json.dumps(_probe_model_adapter_inline(model, model_path)))
sys.stdout.flush()
os._exit(0)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            model.repo_id,
            str(model_path) if model_path is not None else "",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=25,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {"stdout": completed.stdout.strip()}
    result["returncode"] = completed.returncode
    stderr = completed.stderr.strip()
    if stderr:
        result["stderr"] = stderr
    return result


def _probe_model_adapter_inline(
    model: AudexModel, model_path: Path | None
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "selected_model": model.repo_id,
        "model_path": str(model_path) if model_path is not None else None,
    }
    config_path = model_path / "config.json" if model_path is not None else None
    config: dict[str, Any] = {}
    if config_path is not None and config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            details["config_error"] = f"JSONDecodeError: {exc}"
    details["hf_config"] = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "vocab_size": config.get("vocab_size"),
    }
    try:
        from vllm_metal.v1.model_adapter import DefaultModelAdapter

        from audex_mac.patches.vllm_metal_audex_adapter import (
            PATCH_SENTINEL,
            AudexMultimodalAdapter,
        )

        adapter = DefaultModelAdapter()
        details["class"] = f"{type(adapter).__module__}.{type(adapter).__name__}"
        details["source_file"] = inspect.getsourcefile(DefaultModelAdapter)
        details["has_build_multimodal_adapter"] = hasattr(
            adapter, "build_multimodal_adapter"
        )
        details["audex_native_multimodal_supported"] = _adapter_mentions_audex(
            DefaultModelAdapter
        )
        details["audex_patch_installed"] = bool(
            getattr(DefaultModelAdapter, PATCH_SENTINEL, False)
        )
        selected_adapter = adapter.build_multimodal_adapter(
            _DummyAudexModel(),
            SimpleNamespace(
                model_type=config.get("model_type") or "nemotron_dense_audex",
                architectures=config.get("architectures")
                or ["NemotronDenseAudexForConditionalGeneration"],
            ),
        )
        details["audex_adapter_selected"] = isinstance(
            selected_adapter,
            AudexMultimodalAdapter,
        )
        details["audex_adapter_forward_ready"] = bool(
            getattr(selected_adapter, "forward_ready", False)
        )
    except Exception as exc:
        details["error"] = f"{type(exc).__name__}: {exc}"
    return details


class _DummyEmbedTokens:
    def __call__(self, input_ids: Any) -> Any:
        return input_ids


class _DummyAudexTextModel:
    model = SimpleNamespace(embed_tokens=_DummyEmbedTokens())

    def __call__(
        self,
        input_ids: Any,
        *,
        cache: list[Any] | None = None,
        position_ids: Any | None = None,
        input_embeddings: Any | None = None,
    ) -> Any:
        return input_embeddings if input_embeddings is not None else input_ids


class _DummyAudexModel:
    language_model = _DummyAudexTextModel()


def _probe_vllm_generation(
    preflight: Any,
    *,
    max_tokens: int | None,
) -> dict[str, Any]:
    if not preflight.ready or preflight.model_path is None:
        return {
            "enabled": True,
            "ready": False,
            "error": "text runtime preflight is not ready",
            "missing_items": list(preflight.missing_items),
        }

    benchmark = preflight.benchmark
    long_max_tokens = max_tokens or int(benchmark.generation["max_tokens"])
    code = """
import json
import os
import sys
import time

model_path = sys.argv[1]
temperature = float(sys.argv[2])
top_p = float(sys.argv[3])
seed = int(sys.argv[4])
long_max_tokens = int(sys.argv[5])
system_prompt = sys.argv[6]

result = {
    "enabled": True,
    "subprocess": True,
    "max_tokens": long_max_tokens,
}
try:
    from audex_mac.patches.runtime import apply_audex_runtime_patches
    apply_audex_runtime_patches()

    from vllm import LLM, SamplingParams

    started = time.time()
    llm = LLM(
        model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enforce_eager=False,
    )
    result["model_load_seconds"] = round(time.time() - started, 3)

    one_token_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=1,
        seed=seed,
    )
    one_token_started = time.time()
    one_token_output = llm.generate(
        ["<|im_start|>user\\nSay hello.<|im_end|>\\n<|im_start|>assistant\\n"],
        one_token_params,
    )[0].outputs[0]
    result["one_token_probe"] = {
        "elapsed_seconds": round(time.time() - one_token_started, 3),
        "text": one_token_output.text,
        "token_ids": list(one_token_output.token_ids),
    }

    long_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=long_max_tokens,
        seed=seed,
    )
    long_prompt = (
        "<|im_start|>system\\n"
        f"{system_prompt}\\n"
        "<|im_end|>\\n"
        "<|im_start|>user\\n"
        "Write a compact Python function, tests, and a short explanation for "
        "parsing a simple key=value configuration format.\\n"
        "<|im_end|>\\n"
        "<|im_start|>assistant\\n"
    )
    long_started = time.time()
    long_output = llm.generate([long_prompt], long_params)[0].outputs[0]
    elapsed = time.time() - long_started
    token_count = len(long_output.token_ids)
    result["long_probe"] = {
        "max_tokens": long_max_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "generated_tokens": token_count,
        "tokens_per_second": (
            round(token_count / elapsed, 3) if elapsed > 0 else None
        ),
        "finish_reason": long_output.finish_reason,
        "text_prefix": long_output.text[:400],
    }
    result["ready"] = True
except Exception as exc:
    result["ready"] = False
    result["error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(result))
sys.stdout.flush()
os._exit(0)
"""
    args = [
        sys.executable,
        "-c",
        code,
        str(preflight.model_path),
        str(float(benchmark.generation["temperature"])),
        str(float(benchmark.generation["top_p"])),
        str(int(benchmark.generation["seed"])),
        str(long_max_tokens),
        benchmark.system,
    ]
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=1800,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "enabled": True,
            "subprocess": True,
            "ready": False,
            "error": f"TimeoutExpired: generation probe exceeded {exc.timeout}s",
        }

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {
            "enabled": True,
            "subprocess": True,
            "ready": False,
            "stdout": completed.stdout.strip(),
        }
    result["returncode"] = completed.returncode
    stderr = completed.stderr.strip()
    if stderr:
        result["stderr"] = stderr
    if completed.returncode != 0 and "error" not in result:
        result["ready"] = False
        result["error"] = f"generation subprocess exited {completed.returncode}"
    return result


def _probe_vllm_sts_default_runtime(
    model: AudexModel,
    *,
    audio_fixture: Path | None,
    play_audio: bool = False,
    speech_max_tokens: int | None = None,
    native_sampling_debug: bool = False,
) -> dict[str, Any]:
    if speech_max_tokens is None:
        speech_max_tokens = _sts_smoke_speech_max_tokens(play_audio)
    code = """
import json
import os
import sys
import time
import traceback
from pathlib import Path

repo_id = sys.argv[1]
fixture_arg = sys.argv[2]
play_audio = sys.argv[3] == "1"
speech_max_tokens = int(sys.argv[4]) if sys.argv[4] else None

result = {
    "enabled": True,
    "subprocess": True,
    "repo_id": repo_id,
    "audio_fixture": fixture_arg if fixture_arg else None,
    "play_audio": play_audio,
    "speech_max_tokens": speech_max_tokens,
}
try:
    from audex_mac.audio_runtime import preflight_audio_runtime
    from audex_mac.models import SUPPORTED_MODELS
    from audex_mac.speech_output import RUNS_DIR, write_pcm16_wav
    from audex_mac.vllm_commands import run_vllm_fixture_turn

    model = next(candidate for candidate in SUPPORTED_MODELS if candidate.repo_id == repo_id)
    preflight = preflight_audio_runtime(model)
    if not preflight.ready or preflight.model_path is None or preflight.decoder_path is None:
        result.update(
            {
                "ready": False,
                "error": "audio runtime preflight is not ready",
                "missing_items": list(preflight.missing_items),
            }
        )
    else:
        input_wav = Path(fixture_arg) if fixture_arg else RUNS_DIR / "vllm-sts-diagnostic-silence.wav"
        if not fixture_arg:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            write_pcm16_wav(input_wav, [0.0] * 16000, sample_rate=16000)

        started = time.time()
        turn = run_vllm_fixture_turn(
            full_model_path=preflight.model_path,
            decoder_path=preflight.decoder_path,
            input_wav_path=input_wav,
            selected_model_repo=repo_id,
            output_dir=RUNS_DIR,
            play=play_audio,
            speech_max_tokens=speech_max_tokens,
        )
        elapsed = round(time.time() - started, 3)
        turn_log = json.loads(Path(turn.run_log_path).read_text(encoding="utf-8"))
        speech_log_path = Path(turn_log["speech_output_run_log_path"])
        speech_log = json.loads(speech_log_path.read_text(encoding="utf-8"))
        response_word_count = len(turn.response_text.split())
        result.update(
            {
                "ready": True,
                "elapsed_seconds": elapsed,
                "model_path": str(preflight.model_path),
                "decoder_path": str(preflight.decoder_path),
                "input_wav_path": str(input_wav),
                "output_wav_path": str(turn.output_wav_path),
                "run_log_path": str(turn.run_log_path),
                "speech_output_run_log_path": str(speech_log_path),
                "transcript_prefix": turn.transcript[:300],
                "response_prefix": turn.response_text[:300],
                "response_word_count": response_word_count,
                "min_response_words": 8,
                "valid_response_length": response_word_count >= 8,
                "engine_class": turn_log.get("vllm", {}).get("engine_class"),
                "timings": turn_log.get("timings", {}),
                "speech_streaming": {
                    "streaming": speech_log.get("streaming"),
                    "vllm_token_streaming": speech_log.get("vllm_token_streaming"),
                    "decoder_streaming": speech_log.get("decoder_streaming"),
                    "playback_transport": speech_log.get("playback_transport"),
                    "first_audio_ready_seconds": speech_log.get("first_audio_ready_seconds"),
                    "first_playback_started_seconds": speech_log.get("first_playback_started_seconds"),
                    "generated_token_count": (
                        speech_log.get("generated_token_id_count")
                        or len(speech_log.get("generated_token_ids", []))
                    ),
                    "generated_codec_frame_count": (
                        speech_log.get("generated_codec_frame_count")
                        or len(speech_log.get("generated_codec_frames", []))
                    ),
                    "reached_end_token": speech_log.get("reached_end_token"),
                    "hit_max_tokens": speech_log.get("hit_max_tokens"),
                    "chunk_count": (
                        speech_log.get("decoded_chunk_count")
                        or len(speech_log.get("chunk_wav_paths", []))
                    ),
                    "tts_interleaved_initial_ready_batched": speech_log.get("tts_interleaved_initial_ready_batched"),
                    "tts_interleaved_tail_batched": speech_log.get("tts_interleaved_tail_batched"),
                    "tts_interleaved_all_ready_batched": speech_log.get("tts_interleaved_all_ready_batched"),
                    "playback_diagnostics": speech_log.get("playback_diagnostics"),
                    "stream_event_count": speech_log.get("stream_event_count"),
                    "first_token_event_seconds": speech_log.get("first_token_event_seconds"),
                    "last_token_event_seconds": speech_log.get("last_token_event_seconds"),
                    "first_codec_frame_seconds": speech_log.get("first_codec_frame_seconds"),
                    "last_codec_frame_seconds": speech_log.get("last_codec_frame_seconds"),
                    "stream_finished_seconds": speech_log.get("stream_finished_seconds"),
                    "playback_close_seconds": speech_log.get("playback_close_seconds"),
                    "tts_observed_segments": speech_log.get("tts_observed_segments"),
                    "tts_target_segments": speech_log.get("tts_target_segments"),
                    "tts_segment_codec_frame_counts": speech_log.get("tts_segment_codec_frame_counts"),
                },
            }
        )
except Exception as exc:
    result.update(
        {
            "ready": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    )

print(json.dumps(result))
sys.stdout.flush()
os._exit(0)
"""
    args = [
        sys.executable,
        "-c",
        code,
        model.repo_id,
        str(audio_fixture) if audio_fixture is not None else "",
        "1" if play_audio else "0",
        str(speech_max_tokens) if speech_max_tokens is not None else "",
    ]
    try:
        timeout_seconds = _sts_smoke_timeout_seconds()
        env = os.environ.copy()
        if native_sampling_debug:
            env["AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"] = "1"
        else:
            env.pop("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", None)
        env.pop("AUDEX_VLLM_ENABLE_CFG_WIRING", None)
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _subprocess_timeout_text(exc.stdout or "")
        result = {
            "enabled": True,
            "subprocess": True,
            "ready": False,
            "error": f"TimeoutExpired: STS smoke probe exceeded {exc.timeout}s",
        }
        parsed = _parse_json_from_subprocess_stdout(stdout)
        if parsed is not None:
            result.update(parsed)
            result["timed_out_after_seconds"] = exc.timeout
            if parsed.get("ready") is True:
                result.pop("error", None)
                result["subprocess_timeout_after_result"] = True
            else:
                result["ready"] = False
                result.setdefault(
                    "error",
                    f"TimeoutExpired: STS smoke probe exceeded {exc.timeout}s",
                )
            progress_stdout = _subprocess_progress_text(stdout)
            if progress_stdout:
                result["stdout"] = progress_stdout
        elif stdout:
            result["stdout"] = stdout
        if exc.stderr:
            result["stderr"] = _subprocess_timeout_text(exc.stderr)
        _attach_vllm_metal_timing(result)
        result["native_sampling_debug"] = native_sampling_debug
        return result

    result = _parse_json_from_subprocess_stdout(completed.stdout)
    if result is None:
        result = {
            "enabled": True,
            "subprocess": True,
            "ready": False,
            "stdout": completed.stdout.strip(),
        }
    else:
        progress_stdout = _subprocess_progress_text(completed.stdout)
        if progress_stdout:
            result["stdout"] = progress_stdout
    result["native_sampling_debug"] = native_sampling_debug
    result["returncode"] = completed.returncode
    stderr = completed.stderr.strip()
    if stderr:
        result["stderr"] = stderr
    _attach_vllm_metal_timing(result)
    if completed.returncode != 0 and "error" not in result:
        result["ready"] = False
        result["error"] = f"STS smoke subprocess exited {completed.returncode}"
    return result


def _probe_vllm_tts_batch_runtime(
    model: AudexModel,
    *,
    batch_size: int,
    max_tokens: int,
    text: str,
    use_cfg: bool = True,
    native_sampling_debug: bool = False,
) -> dict[str, Any]:
    code = """
import asyncio
import json
import os
import sys
import time
import traceback

repo_id = sys.argv[1]
batch_size = int(sys.argv[2])
max_tokens = int(sys.argv[3])
text = sys.argv[4]
use_cfg = sys.argv[5] == "1"

result = {
    "enabled": True,
    "subprocess": True,
    "repo_id": repo_id,
    "batch_size": batch_size,
    "max_tokens": max_tokens,
    "text": text,
    "cfg_enabled": use_cfg,
    "codec_window": not use_cfg,
    "output_kind": "CUMULATIVE" if use_cfg else "DELTA",
}

async def run_probe():
    from audex_mac.audio_runtime import preflight_audio_runtime
    from audex_mac.models import SUPPORTED_MODELS
    from audex_mac.vllm_runtime import AudexAsyncVllmRuntime, extract_tts_codec_frames

    model = next(candidate for candidate in SUPPORTED_MODELS if candidate.repo_id == repo_id)
    preflight = preflight_audio_runtime(model)
    if not preflight.ready or preflight.model_path is None:
        result.update(
            {
                "ready": False,
                "error": "audio runtime preflight is not ready",
                "missing_items": list(preflight.missing_items),
            }
        )
        return

    runtime = AudexAsyncVllmRuntime.from_model_path(preflight.model_path)
    try:
        started = time.time()
        cond_by_request = {}
        request_count = 0
        if use_cfg:
            requests = []
            for index in range(batch_size):
                pair_id = f"batch-{index}"
                requests.extend(
                    runtime.build_tts_cfg_pair(
                        text,
                        pair_id=pair_id,
                        max_tokens=max_tokens,
                    )
                )
            request_count = len(requests)
            async for delta in runtime.stream_many(tuple(requests)):
                if delta.request_debug_name != "tts-cond":
                    continue
                codec = extract_tts_codec_frames(delta.token_ids, runtime.token_map)
                cond_by_request[delta.request_id] = {
                    "generated_token_count": len(delta.token_ids),
                    "generated_codec_frame_count": len(codec.generated_codec_frames),
                    "reached_end_token": codec.reached_end_token,
                    "finished": delta.finished,
                    "elapsed_seconds": delta.elapsed_seconds,
                }
        else:
            request_count = batch_size

            async def run_no_cfg_request(index):
                generated_token_count = 0
                generated_codec_frame_count = 0
                reached_end_token = False
                finished = False
                elapsed_seconds = 0.0
                async for event in runtime.stream_tts_codec_frames(
                    text,
                    max_tokens=max_tokens,
                ):
                    generated_codec_frame_count += len(event.new_codec_frames)
                    reached_end_token = event.reached_end_token
                    finished = event.finished
                    elapsed_seconds = event.elapsed_seconds
                    if event.generated_token_ids:
                        generated_token_count = len(event.generated_token_ids)
                return (
                    f"no-cfg-{index}",
                    {
                        "generated_token_count": generated_token_count,
                        "generated_codec_frame_count": generated_codec_frame_count,
                        "reached_end_token": reached_end_token,
                        "finished": finished,
                        "elapsed_seconds": elapsed_seconds,
                    },
                )

            cond_by_request.update(
                await asyncio.gather(
                    *(run_no_cfg_request(index) for index in range(batch_size))
                )
            )

        elapsed = round(time.time() - started, 3)
        total_codec_frames = sum(
            item["generated_codec_frame_count"] for item in cond_by_request.values()
        )
        per_request_codec_frames = [
            item["generated_codec_frame_count"] for item in cond_by_request.values()
        ]
        hit_max_token_count = sum(
            1
            for item in cond_by_request.values()
            if item["generated_token_count"] >= max_tokens and not item["reached_end_token"]
        )
        reached_end_count = sum(
            1 for item in cond_by_request.values() if item["reached_end_token"]
        )
        result.update(
            {
                "ready": True,
                "elapsed_seconds": elapsed,
                "engine_class": runtime.stats.engine_class,
                "model_path": str(preflight.model_path),
                "request_count": request_count,
                "conditional_request_count": len(cond_by_request),
                "total_codec_frame_count": total_codec_frames,
                "codec_frames_per_second": (
                    round(total_codec_frames / elapsed, 3) if elapsed > 0 else None
                ),
                "min_codec_frames_per_request": (
                    min(per_request_codec_frames) if per_request_codec_frames else 0
                ),
                "max_codec_frames_per_request": (
                    max(per_request_codec_frames) if per_request_codec_frames else 0
                ),
                "hit_max_token_count": hit_max_token_count,
                "reached_end_count": reached_end_count,
                "per_request": cond_by_request,
            }
        )
    finally:
        shutdown = getattr(runtime.engine, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown(timeout=5.0)
            except TypeError:
                shutdown()

try:
    asyncio.run(run_probe())
except Exception as exc:
    result.update(
        {
            "ready": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    )

print(json.dumps(result))
sys.stdout.flush()
os._exit(0)
"""
    args = [
        sys.executable,
        "-c",
        code,
        model.repo_id,
        str(batch_size),
        str(max_tokens),
        text,
        "1" if use_cfg else "0",
    ]
    try:
        timeout_seconds = _sts_smoke_timeout_seconds()
        env = os.environ.copy()
        if native_sampling_debug:
            env["AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"] = "1"
        else:
            env.pop("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", None)
        if use_cfg:
            env["AUDEX_VLLM_ENABLE_CFG_WIRING"] = "1"
        else:
            env.pop("AUDEX_VLLM_ENABLE_CFG_WIRING", None)
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _subprocess_timeout_text(exc.stdout or "")
        result = {
            "enabled": True,
            "subprocess": True,
            "ready": False,
            "error": f"TimeoutExpired: TTS batch probe exceeded {exc.timeout}s",
        }
        parsed = _parse_json_from_subprocess_stdout(stdout)
        if parsed is not None:
            result.update(parsed)
            result["timed_out_after_seconds"] = exc.timeout
        progress_stdout = _subprocess_progress_text(stdout)
        if progress_stdout:
            result["stdout"] = progress_stdout
        if exc.stderr:
            result["stderr"] = _subprocess_timeout_text(exc.stderr)
            _attach_vllm_metal_timing(result)
        result["native_sampling_debug"] = native_sampling_debug
        return result

    result = _parse_json_from_subprocess_stdout(completed.stdout)
    if result is None:
        result = {
            "enabled": True,
            "subprocess": True,
            "ready": False,
            "stdout": completed.stdout.strip(),
        }
    else:
        progress_stdout = _subprocess_progress_text(completed.stdout)
        if progress_stdout:
            result["stdout"] = progress_stdout
    result["native_sampling_debug"] = native_sampling_debug
    result["returncode"] = completed.returncode
    stderr = completed.stderr.strip()
    if stderr:
        result["stderr"] = stderr
        _attach_vllm_metal_timing(result)
    if completed.returncode != 0 and "error" not in result:
        result["ready"] = False
        result["error"] = f"TTS batch subprocess exited {completed.returncode}"
    return result


def _sts_smoke_timeout_seconds() -> int:
    raw_value = os.environ.get("AUDEX_STS_SMOKE_TIMEOUT_SECONDS", "600")
    try:
        timeout = int(raw_value)
    except ValueError:
        return 600
    return max(30, timeout)


def _sts_smoke_speech_max_tokens(play_audio: bool) -> int | None:
    raw_value = os.environ.get("AUDEX_STS_SMOKE_SPEECH_MAX_TOKENS")
    if raw_value:
        try:
            return max(1, int(raw_value))
        except ValueError:
            return DEFAULT_STS_SMOKE_SPEECH_MAX_TOKENS
    if not play_audio:
        return None
    return DEFAULT_STS_SMOKE_SPEECH_MAX_TOKENS


def _subprocess_timeout_text(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return value.strip()


def _parse_json_from_subprocess_stdout(stdout: str) -> dict[str, Any] | None:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _subprocess_progress_text(stdout: str) -> str:
    lines = stdout.splitlines()
    while lines and lines[-1].strip().startswith("{"):
        try:
            json.loads(lines[-1].strip())
            lines.pop()
            break
        except json.JSONDecodeError:
            break
    return "\n".join(line for line in lines if line.strip()).strip()


def _attach_vllm_metal_timing(result: dict[str, Any]) -> None:
    parsed = _parse_vllm_metal_timing(str(result.get("stderr") or ""))
    if parsed:
        result["vllm_metal_timing"] = parsed
    assessment = _assess_sts_timing(result)
    if assessment:
        result["sts_timing_assessment"] = assessment


def _parse_vllm_metal_timing(stderr: str) -> dict[str, Any]:
    checkpoints: list[dict[str, Any]] = []
    non_paged_checkpoints: list[dict[str, Any]] = []
    native_used_counts: list[int] = []
    native_rejection_reasons: list[str] = []

    for line in stderr.splitlines():
        timing_match = PAGED_TIMING_RE.search(line)
        if timing_match:
            checkpoint = {
                "count": int(timing_match.group("count")),
                "avg_ms": float(timing_match.group("avg_ms")),
                "last_ms": float(timing_match.group("last_ms")),
                "decode_reqs": int(timing_match.group("decode_reqs")),
                "prefill_reqs": int(timing_match.group("prefill_reqs")),
                "decode_tokens": int(timing_match.group("decode_tokens")),
                "native_sample_ms": float(timing_match.group("native_sample_ms")),
            }
            if timing_match.group("native_sampled_rows") is not None:
                checkpoint["native_sampled_rows"] = int(
                    timing_match.group("native_sampled_rows")
                )
            if timing_match.group("native_output_rows") is not None:
                checkpoint["native_output_rows"] = int(
                    timing_match.group("native_output_rows")
                )
            if timing_match.group("skipped_logits_eval") is not None:
                checkpoint["skipped_logits_eval"] = int(
                    timing_match.group("skipped_logits_eval")
                )
            native_detail_ms = timing_match.group("native_detail_ms")
            if native_detail_ms:
                checkpoint["native_detail_ms"] = _parse_mx_eval_timing(native_detail_ms)
            mx_eval_ms = timing_match.group("mx_eval_ms")
            if mx_eval_ms:
                checkpoint["mx_eval_ms"] = _parse_mx_eval_timing(mx_eval_ms)
            mx_eval_shapes = timing_match.group("mx_eval_shapes")
            if mx_eval_shapes:
                checkpoint["mx_eval_shapes"] = _parse_mx_eval_shapes(mx_eval_shapes)
            checkpoints.append(checkpoint)
            continue

        non_paged_match = NON_PAGED_TIMING_RE.search(line)
        if non_paged_match:
            checkpoint = {
                "count": int(non_paged_match.group("count")),
                "avg_ms": float(non_paged_match.group("avg_ms")),
                "last_ms": float(non_paged_match.group("last_ms")),
                "decode_reqs": int(non_paged_match.group("decode_reqs")),
                "cached_reqs": int(non_paged_match.group("cached_reqs")),
                "batched": non_paged_match.group("batched") == "1",
                "native_sample_ms": float(non_paged_match.group("native_sample_ms")),
            }
            if non_paged_match.group("native_sampled_rows") is not None:
                checkpoint["native_sampled_rows"] = int(
                    non_paged_match.group("native_sampled_rows")
                )
            if non_paged_match.group("native_output_rows") is not None:
                checkpoint["native_output_rows"] = int(
                    non_paged_match.group("native_output_rows")
                )
            for field_name in (
                "cfg_cond_reqs",
                "cfg_uncond_reqs",
                "cfg_complete_pairs",
                "tts_window_decode_count",
                "tts_window_weight_cache_hits",
                "tts_window_weight_cache_misses",
                "nonpaged_persistent_cache_hits",
                "nonpaged_persistent_cache_misses",
                "nonpaged_persistent_cache_flushes",
            ):
                if non_paged_match.group(field_name) is not None:
                    checkpoint[field_name] = int(non_paged_match.group(field_name))
            native_detail_ms = non_paged_match.group("native_detail_ms")
            if native_detail_ms:
                checkpoint["native_detail_ms"] = _parse_mx_eval_timing(native_detail_ms)
            non_paged_checkpoints.append(checkpoint)
            continue

        used_match = NATIVE_SAMPLING_USED_RE.search(line)
        if used_match:
            native_used_counts.append(int(used_match.group("count")))
            continue

        skipped_match = NATIVE_SAMPLING_SKIPPED_RE.search(line)
        if skipped_match:
            native_rejection_reasons.append(skipped_match.group("reason").strip())

    result: dict[str, Any] = {}
    if checkpoints:
        result["paged_sample_checkpoints"] = checkpoints
        result["latest_paged_sample"] = checkpoints[-1]
    if non_paged_checkpoints:
        result["non_paged_decode_checkpoints"] = non_paged_checkpoints
        result["latest_non_paged_decode"] = non_paged_checkpoints[-1]
    if native_used_counts:
        result["native_sampling_fast_path_counts"] = native_used_counts
        result["latest_native_sampling_fast_path_count"] = native_used_counts[-1]
    if native_rejection_reasons:
        result["native_sampling_rejection_reasons"] = native_rejection_reasons
    return result


def _parse_mx_eval_timing(value: str) -> dict[str, dict[str, float | int]]:
    if value == "none":
        return {}
    parsed: dict[str, dict[str, float | int]] = {}
    for item in value.split(","):
        if ":" not in item or "/" not in item:
            continue
        category, rest = item.split(":", 1)
        milliseconds, count = rest.split("/", 1)
        try:
            parsed[category] = {
                "milliseconds": float(milliseconds),
                "count": int(count),
            }
        except ValueError:
            continue
    return parsed


def _parse_mx_eval_shapes(value: str) -> dict[str, list[dict[str, Any]]]:
    if value == "none":
        return {}
    parsed: dict[str, list[dict[str, Any]]] = {}
    for item in value.split(","):
        if ":" not in item:
            continue
        category, rest = item.split(":", 1)
        shapes: list[dict[str, Any]] = []
        for shape_item in rest.split("+"):
            if "x" not in shape_item:
                continue
            shape_text, count_text = shape_item.rsplit("x", 1)
            try:
                count = int(count_text)
            except ValueError:
                continue
            if shape_text == "scalar":
                shape: list[int] = []
            else:
                try:
                    shape = [int(dim) for dim in shape_text.split("x") if dim]
                except ValueError:
                    continue
            shapes.append({"shape": shape, "count": count})
        if shapes:
            parsed[category] = shapes
    return parsed


def _assess_sts_timing(sts_probe: dict[str, Any]) -> dict[str, Any]:
    streaming = sts_probe.get("speech_streaming", {})
    if not isinstance(streaming, dict):
        return {}

    assessment: dict[str, Any] = {}
    codec_frame_count = int(streaming.get("generated_codec_frame_count") or 0)
    last_codec_frame_seconds = _optional_float(
        streaming.get("last_codec_frame_seconds")
    )
    if (
        codec_frame_count > 0
        and last_codec_frame_seconds
        and last_codec_frame_seconds > 0
    ):
        frames_per_second = codec_frame_count / last_codec_frame_seconds
        assessment["codec_frames_per_second"] = round(frames_per_second, 3)
        assessment["audio_realtime_ratio"] = round(frames_per_second / 50.0, 3)
        assessment["below_realtime"] = frames_per_second < 50.0

    segment_counts = _segment_codec_frame_counts(streaming)
    if len(segment_counts) >= 2:
        segment_values = tuple(segment_counts.values())
        min_frames = min(segment_values)
        max_frames = max(segment_values)
        total_frames = sum(segment_values)
        mean_frames = total_frames / len(segment_values)
        tail_frames = segment_values[-1]
        assessment["tts_segment_count"] = len(segment_values)
        assessment["tts_segment_codec_frame_min"] = min_frames
        assessment["tts_segment_codec_frame_max"] = max_frames
        assessment["tts_segment_codec_frame_mean"] = round(mean_frames, 3)
        assessment["tts_segment_codec_frame_max_to_min_ratio"] = (
            round(max_frames / min_frames, 3) if min_frames > 0 else None
        )
        assessment["tts_tail_codec_frames"] = tail_frames
        assessment["tts_tail_to_mean_ratio"] = (
            round(tail_frames / mean_frames, 3) if mean_frames > 0 else None
        )
        assessment["tts_tail_underfilled"] = mean_frames > 0 and tail_frames < (
            mean_frames * 0.5
        )

    playback = streaming.get("playback_diagnostics")
    if isinstance(playback, dict):
        underflows = int(playback.get("device_underflow_count") or 0)
        underruns = int(playback.get("queue_underrun_count") or 0)
        overruns = int(playback.get("queue_overrun_count") or 0)
        assessment["playback_glitch_count"] = underflows + underruns + overruns

    timing = sts_probe.get("vllm_metal_timing", {})
    if isinstance(timing, dict):
        latest_paged = timing.get("latest_paged_sample", {})
        if isinstance(latest_paged, dict):
            native_ms = _optional_float(latest_paged.get("native_sample_ms")) or 0.0
            paged_count = _optional_float(latest_paged.get("count")) or 0.0
            paged_avg_ms = _optional_float(latest_paged.get("avg_ms"))
            mx_eval = latest_paged.get("mx_eval_ms", {})
            mx_eval_shapes = latest_paged.get("mx_eval_shapes", {})
            eval_ms_by_category = _mx_eval_milliseconds_by_category(mx_eval)
            eval_ms_per_step = _mx_eval_milliseconds_per_step_by_category(mx_eval)
            native_detail = latest_paged.get("native_detail_ms", {})
            native_detail_ms_by_category = _mx_eval_milliseconds_by_category(
                native_detail
            )
            native_detail_ms_per_step = _mx_eval_milliseconds_per_step_by_category(
                native_detail
            )
            if paged_avg_ms is not None:
                assessment["paged_sample_avg_ms"] = paged_avg_ms
            skipped_logits_eval = _optional_float(
                latest_paged.get("skipped_logits_eval")
            )
            if skipped_logits_eval is not None:
                assessment["skipped_logits_eval"] = int(skipped_logits_eval)
            if native_detail_ms_by_category:
                assessment["native_detail_ms_by_category"] = (
                    native_detail_ms_by_category
                )
                if native_detail_ms_per_step:
                    assessment["native_detail_ms_per_step_by_category"] = (
                        native_detail_ms_per_step
                    )
                    dominant_native_detail_category = max(
                        native_detail_ms_per_step,
                        key=lambda category: native_detail_ms_per_step[category],
                    )
                    assessment["dominant_native_detail_category"] = (
                        dominant_native_detail_category
                    )
                    assessment["dominant_native_detail_ms_per_step"] = (
                        native_detail_ms_per_step[dominant_native_detail_category]
                    )
            if eval_ms_by_category:
                assessment["mx_eval_ms_by_category"] = eval_ms_by_category
                if isinstance(mx_eval_shapes, dict) and mx_eval_shapes:
                    assessment["mx_eval_shapes_by_category"] = mx_eval_shapes
                if eval_ms_per_step:
                    assessment["mx_eval_ms_per_step_by_category"] = eval_ms_per_step
                    dominant_per_step_category = max(
                        eval_ms_per_step,
                        key=lambda category: eval_ms_per_step[category],
                    )
                    assessment["dominant_mx_eval_per_step_category"] = (
                        dominant_per_step_category
                    )
                    assessment["dominant_mx_eval_ms_per_step"] = eval_ms_per_step[
                        dominant_per_step_category
                    ]
                dominant_category = max(
                    eval_ms_by_category,
                    key=lambda category: eval_ms_by_category[category],
                )
                assessment["dominant_mx_eval_category"] = dominant_category
                assessment["dominant_mx_eval_ms"] = eval_ms_by_category[
                    dominant_category
                ]
            if native_ms:
                assessment["native_sample_ms"] = native_ms
                if paged_count > 0:
                    assessment["native_sample_ms_per_step"] = round(
                        native_ms / paged_count,
                        3,
                    )
                sampled_rows = _optional_float(latest_paged.get("native_sampled_rows"))
                output_rows = _optional_float(latest_paged.get("native_output_rows"))
                if sampled_rows is not None:
                    assessment["native_sampled_rows"] = int(sampled_rows)
                    if sampled_rows > 0:
                        assessment["native_sample_ms_per_sampled_row"] = round(
                            native_ms / sampled_rows,
                            3,
                        )
                if output_rows is not None:
                    assessment["native_output_rows"] = int(output_rows)
                if sampled_rows and output_rows and output_rows > 0:
                    assessment["native_sampling_row_ratio"] = round(
                        sampled_rows / output_rows,
                        3,
                    )
                logits_ms = eval_ms_per_step.get(
                    "logits",
                    eval_ms_by_category.get("logits", 0.0),
                )
                sample_tokens_ms = eval_ms_per_step.get(
                    "sample_tokens",
                    eval_ms_by_category.get("sample_tokens", 0.0),
                )
                per_native_sample_ms = assessment.get(
                    "native_sample_ms_per_sampled_row",
                    assessment.get("native_sample_ms_per_step", native_ms),
                )
                if logits_ms > per_native_sample_ms and logits_ms > sample_tokens_ms:
                    assessment["likely_bottleneck"] = "logits_eval"
                elif (
                    sample_tokens_ms > logits_ms
                    and sample_tokens_ms >= 0.9 * per_native_sample_ms
                ):
                    assessment["likely_bottleneck"] = (
                        "pending_graph_eval_during_sampling"
                    )
                elif (
                    per_native_sample_ms > logits_ms
                    and per_native_sample_ms > sample_tokens_ms
                ):
                    assessment["likely_bottleneck"] = "native_sampling"

        latest_non_paged = timing.get("latest_non_paged_decode", {})
        if isinstance(latest_non_paged, dict):
            nonpaged_avg_ms = _optional_float(latest_non_paged.get("avg_ms"))
            if nonpaged_avg_ms is not None:
                assessment["nonpaged_decode_avg_ms"] = nonpaged_avg_ms
            nonpaged_count = _optional_float(latest_non_paged.get("count")) or 0.0
            nonpaged_native_ms = (
                _optional_float(latest_non_paged.get("native_sample_ms")) or 0.0
            )
            if nonpaged_native_ms:
                assessment["nonpaged_native_sample_ms"] = nonpaged_native_ms
                if nonpaged_count > 0:
                    assessment["nonpaged_native_sample_ms_per_step"] = round(
                        nonpaged_native_ms / nonpaged_count,
                        3,
                    )

            for field_name in (
                "cfg_cond_reqs",
                "cfg_uncond_reqs",
                "cfg_complete_pairs",
                "tts_window_decode_count",
                "tts_window_weight_cache_hits",
                "tts_window_weight_cache_misses",
                "nonpaged_persistent_cache_hits",
                "nonpaged_persistent_cache_misses",
                "nonpaged_persistent_cache_flushes",
            ):
                field_value = _optional_float(latest_non_paged.get(field_name))
                if field_value is not None:
                    assessment[field_name] = int(field_value)

            cache_hits = _optional_float(
                latest_non_paged.get("tts_window_weight_cache_hits")
            )
            cache_misses = _optional_float(
                latest_non_paged.get("tts_window_weight_cache_misses")
            )
            if cache_hits is not None and cache_misses is not None:
                cache_total = cache_hits + cache_misses
                if cache_total > 0:
                    assessment["tts_window_weight_cache_hit_rate"] = round(
                        cache_hits / cache_total,
                        3,
                    )
            persistent_hits = _optional_float(
                latest_non_paged.get("nonpaged_persistent_cache_hits")
            )
            persistent_misses = _optional_float(
                latest_non_paged.get("nonpaged_persistent_cache_misses")
            )
            if persistent_hits is not None and persistent_misses is not None:
                persistent_total = persistent_hits + persistent_misses
                if persistent_total > 0:
                    assessment["nonpaged_persistent_cache_hit_rate"] = round(
                        persistent_hits / persistent_total,
                        3,
                    )

            nonpaged_native_detail = latest_non_paged.get("native_detail_ms", {})
            nonpaged_detail_ms_by_category = _mx_eval_milliseconds_by_category(
                nonpaged_native_detail
            )
            nonpaged_detail_ms_per_step = _mx_eval_milliseconds_per_step_by_category(
                nonpaged_native_detail
            )
            if nonpaged_detail_ms_by_category:
                assessment["nonpaged_native_detail_ms_by_category"] = (
                    nonpaged_detail_ms_by_category
                )
            if nonpaged_detail_ms_per_step:
                assessment["nonpaged_native_detail_ms_per_step_by_category"] = (
                    nonpaged_detail_ms_per_step
                )
                dominant_nonpaged_detail_category = max(
                    nonpaged_detail_ms_per_step,
                    key=lambda category: nonpaged_detail_ms_per_step[category],
                )
                assessment["dominant_nonpaged_native_detail_category"] = (
                    dominant_nonpaged_detail_category
                )
                assessment["dominant_nonpaged_native_detail_ms_per_step"] = (
                    nonpaged_detail_ms_per_step[dominant_nonpaged_detail_category]
                )
                if dominant_nonpaged_detail_category == "sample_eval":
                    assessment["likely_bottleneck"] = (
                        "pending_graph_eval_during_sampling"
                    )
                elif dominant_nonpaged_detail_category in {
                    "tts_window_sample",
                    "tts_window_batch_sample",
                }:
                    assessment["likely_bottleneck"] = (
                        "pending_graph_eval_during_tts_window_sampling"
                    )
                elif dominant_nonpaged_detail_category in {
                    "tts_window_forward_eval",
                    "tts_window_batch_forward_eval",
                }:
                    assessment["likely_bottleneck"] = (
                        "model_forward_eval_during_tts_window_decode"
                    )
                elif dominant_nonpaged_detail_category in {
                    "tts_window_project_eval",
                    "tts_window_batch_project_eval",
                }:
                    assessment["likely_bottleneck"] = (
                        "projection_eval_during_tts_window_decode"
                    )
                elif dominant_nonpaged_detail_category in {
                    "nonpaged_kv_cache_merge",
                    "nonpaged_kv_cache_extract",
                }:
                    assessment["likely_bottleneck"] = "nonpaged_kv_cache_copy"
                elif dominant_nonpaged_detail_category == (
                    "nonpaged_decode_logits_async_submit"
                ):
                    assessment["likely_bottleneck"] = "nonpaged_async_graph_submit"

    return assessment


def _segment_codec_frame_counts(streaming: dict[str, Any]) -> dict[int, int]:
    raw_counts = streaming.get("tts_segment_codec_frame_counts")
    if not isinstance(raw_counts, dict):
        return {}
    parsed: dict[int, int] = {}
    for raw_index, raw_count in raw_counts.items():
        try:
            index = int(raw_index)
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count >= 0:
            parsed[index] = count
    return dict(sorted(parsed.items()))


def _mx_eval_milliseconds_by_category(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for category, timing in value.items():
        if not isinstance(timing, dict):
            continue
        milliseconds = _optional_float(timing.get("milliseconds"))
        if milliseconds is not None:
            result[str(category)] = milliseconds
    return result


def _mx_eval_milliseconds_per_step_by_category(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for category, timing in value.items():
        if not isinstance(timing, dict):
            continue
        milliseconds = _optional_float(timing.get("milliseconds"))
        count = _optional_float(timing.get("count"))
        if milliseconds is not None and count and count > 0:
            result[str(category)] = round(milliseconds / count, 3)
    return result


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _adapter_mentions_audex(adapter_cls: type[Any]) -> bool:
    try:
        source = inspect.getsource(adapter_cls)
    except OSError:
        return False
    return "audex" in source.lower()


def _default_vllm_metal_source_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[1] / ".audex" / "vendor" / "vllm-metal"
    if candidate.is_dir():
        return candidate
    return None


def _scan_vllm_metal_sources(source_root: Path | None) -> list[SourceMatch]:
    if source_root is None or not source_root.is_dir():
        return []
    roots = [source_root / "vllm_metal"]
    matches: list[SourceMatch] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                for pattern in SOURCE_SCAN_PATTERNS:
                    if pattern in line:
                        matches.append(
                            SourceMatch(
                                file=str(path.relative_to(source_root)),
                                line=line_number,
                                pattern=pattern,
                                text=line.strip(),
                            )
                        )
                        break
            if len(matches) >= 200:
                return matches
    return matches


def _interpret_expected_cpu_facade() -> dict[str, Any]:
    return {
        "vllm_device_type_cpu_can_be_expected": True,
        "reason": (
            "vllm-metal advertises a PyTorch CPU device facade for vLLM "
            "compatibility while using MLX for execution when "
            "VLLM_METAL_USE_MLX=1 and VLLM_MLX_DEVICE=gpu."
        ),
        "cpu_fallback_indicators": [
            "VLLM_MLX_DEVICE is not gpu",
            "mlx.core.default_device() is not Device(gpu, 0)",
            "probe or model arrays report a CPU device",
            "vllm_metal.config.use_mlx is false",
            "vllm_metal.config.use_paged_attention is true for the fast default path",
        ],
        "required_env": dict(REQUIRED_METAL_ENV),
    }


def _diagnostic_verdict(
    report: dict[str, Any],
    *,
    require_generation: bool = False,
    require_sts: bool = False,
    require_tts_batch: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    parent_policy = report["parent_process"]["metal_policy"]
    if not parent_policy["ready"]:
        failures.append("parent Metal/MLX policy is not ready")

    vllm_metal = report["vllm_metal"]
    config_error = vllm_metal.get("platform", {}).get("error") or vllm_metal.get(
        "current_platform_error"
    )
    if config_error:
        failures.append(f"vLLM Metal config error: {config_error}")
    config = vllm_metal.get("config", {})
    if config:
        if config.get("use_mlx") is not True:
            failures.append("vLLM Metal config use_mlx is not true")
        if config.get("mlx_device") != "gpu":
            failures.append("vLLM Metal config mlx_device is not gpu")
        if config.get("use_paged_attention") is not False:
            failures.append(
                "vLLM Metal config use_paged_attention is not false for the fast default path"
            )

    spawn_probe = report["spawn_probe"]
    if spawn_probe.get("returncode") != 0:
        failures.append("spawn probe exited nonzero")
    if "mlx_error" in spawn_probe:
        failures.append(f"spawn probe MLX error: {spawn_probe['mlx_error']}")
    spawn_mlx = spawn_probe.get("mlx", {})
    if spawn_mlx and spawn_mlx.get("default_device") != "Device(gpu, 0)":
        failures.append(
            f"spawn probe default_device is {spawn_mlx.get('default_device')!r}"
        )

    platform_probe = report["platform_resolution_probe"]
    expected_platform = "vllm_metal.platform.MetalPlatform"
    if platform_probe.get("direct_vllm_metal_register") != expected_platform:
        failures.append("direct vLLM Metal registration did not return MetalPlatform")
    current_platform = platform_probe.get("current_platform", {})
    repaired_platform = platform_probe.get("current_platform_after_audex_patches", {})
    effective_platform = repaired_platform or current_platform
    if effective_platform.get("class") != expected_platform:
        failures.append(
            "vLLM effective current_platform is "
            f"{effective_platform.get('class')!r}, not MetalPlatform"
        )

    patch_report = report["audex_patches"]
    if patch_report.get("returncode") not in (0, None):
        failures.append(
            f"Audex runtime patch probe exited {patch_report.get('returncode')}"
        )
    if "error" in patch_report:
        failures.append(f"Audex runtime patch probe error: {patch_report['error']}")
    for name, error in patch_report.get("errors", {}).items():
        failures.append(f"Audex runtime patch {name} error: {error}")
    for name, installed in patch_report.items():
        if name in {"returncode", "stdout", "stderr", "error", "errors"}:
            continue
        if installed is not True:
            failures.append(f"Audex runtime patch {name} is not installed")

    adapter = report["model_adapter"]
    if adapter.get("audex_patch_installed") is not True:
        failures.append("vLLM Metal Audex adapter patch is not installed")
    if adapter.get("audex_adapter_selected") is not True:
        failures.append("vLLM Metal DefaultModelAdapter does not select Audex")

    processor = report.get("audex_processor", {})
    if processor.get("ready") is not True:
        failures.append(
            "Audex vLLM processor probe is not ready: " f"{processor.get('error')}"
        )
    processor_output = processor.get("output", {})
    if processor_output and processor_output.get("placeholder_length") != 3:
        failures.append("Audex vLLM processor placeholder length probe failed")

    cfg = report.get("audex_cfg", {})
    if cfg.get("enabled") and cfg.get("ready") is not True:
        failures.append(f"Audex vLLM CFG probe is not ready: {cfg.get('error')}")
    if cfg.get("ready"):
        processors = cfg.get("logits_processors", [])
        if not any(name.endswith(".CFGLogitsProcessor") for name in processors):
            failures.append("Audex vLLM CFG processor is not installed")
        if (
            "audex_mac.patches.vllm_metal_cfg.AudexMetalCFGTokenSyncInstaller"
            not in processors
        ):
            failures.append(
                "Audex vLLM Metal CFG token sync installer is not installed"
            )
        if cfg.get("enable_prefix_caching") is not False:
            failures.append("Audex vLLM CFG did not disable prefix caching")
        metal_patch = cfg.get("vllm_metal_patch", {})
        if metal_patch.get("ready") is not True:
            patch_error = str(metal_patch.get("error"))
            if not _is_same_missing_metal_failure(report, patch_error):
                failures.append(
                    "Audex vLLM Metal CFG sampler patch is not ready: "
                    f"{metal_patch.get('error')}"
                )

    if require_generation and report["generation_probe"].get("ready") is not True:
        failures.append(
            "vLLM generation probe is not ready: "
            f"{report['generation_probe'].get('error')}"
        )

    sts_probe = report.get("sts_probe", {})
    if require_sts and sts_probe.get("ready") is not True:
        failures.append(
            "vLLM default STS smoke probe is not ready: " f"{sts_probe.get('error')}"
        )
    if sts_probe.get("ready"):
        failures.extend(_sts_smoke_evidence_failures(sts_probe))

    tts_batch_probe = report.get("tts_batch_probe", {})
    if require_tts_batch and tts_batch_probe.get("ready") is not True:
        failures.append(
            "vLLM TTS batch probe is not ready: " f"{tts_batch_probe.get('error')}"
        )
    if (
        tts_batch_probe.get("ready")
        and _optional_float(tts_batch_probe.get("codec_frames_per_second")) is None
    ):
        failures.append("vLLM TTS batch probe did not measure codec throughput")

    return {
        "ready": not failures,
        "failures": failures,
    }


def _sts_smoke_evidence_failures(sts_probe: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    engine_class = str(sts_probe.get("engine_class") or "")
    if "Async" not in engine_class:
        failures.append("vLLM default STS smoke did not use an async vLLM engine class")

    streaming = sts_probe.get("speech_streaming", {})
    if streaming.get("vllm_token_streaming") is not True:
        failures.append("vLLM default STS smoke did not stream vLLM tokens")
    if streaming.get("decoder_streaming") is not True:
        failures.append("vLLM default STS smoke did not use streaming decoder")
    if streaming.get("first_audio_ready_seconds") is None:
        failures.append("vLLM default STS smoke did not record first-audio timing")
    if int(streaming.get("generated_token_count") or 0) <= 0:
        failures.append("vLLM default STS smoke generated no speech tokens")
    if int(streaming.get("generated_codec_frame_count") or 0) <= 0:
        failures.append("vLLM default STS smoke generated no codec frames")
    if int(streaming.get("chunk_count") or 0) <= 0:
        failures.append("vLLM default STS smoke wrote no decoder chunks")
    response_length_failure = _sts_response_length_failure(sts_probe)
    if response_length_failure:
        failures.append(response_length_failure)
    row_ratio_failure = _native_sampling_row_ratio_failure(sts_probe)
    if row_ratio_failure:
        failures.append(row_ratio_failure)
    realtime_failure = _sts_realtime_failure(sts_probe)
    if realtime_failure:
        failures.append(realtime_failure)
    segment_balance_failure = _sts_segment_balance_failure(sts_probe)
    if segment_balance_failure:
        failures.append(segment_balance_failure)
    if sts_probe.get("play_audio"):
        if streaming.get("playback_transport") != "sounddevice_raw_output_stream":
            failures.append(
                "vLLM default STS smoke did not use continuous PCM playback"
            )
        if streaming.get("first_playback_started_seconds") is None:
            failures.append(
                "vLLM default STS smoke did not record first-playback timing"
            )
        playback_diagnostics = streaming.get("playback_diagnostics")
        if not isinstance(playback_diagnostics, dict):
            failures.append(
                "vLLM default STS smoke did not record playback diagnostics"
            )
        else:
            for key in (
                "device_underflow_count",
                "queue_underrun_count",
                "queue_overrun_count",
                "chunks_written",
            ):
                if key not in playback_diagnostics:
                    failures.append(
                        "vLLM default STS smoke playback diagnostics missing " f"{key}"
                    )
    return failures


def _sts_response_length_failure(sts_probe: dict[str, Any]) -> str | None:
    if (
        "valid_response_length" in sts_probe
        and sts_probe.get("valid_response_length") is True
    ):
        return None
    min_words = int(
        sts_probe.get("min_response_words") or DEFAULT_STS_SMOKE_MIN_RESPONSE_WORDS
    )
    word_count = sts_probe.get("response_word_count")
    if word_count is None and "response_prefix" in sts_probe:
        word_count = len(str(sts_probe.get("response_prefix") or "").split())
    if word_count is None:
        return None
    try:
        parsed_word_count = int(word_count)
    except (TypeError, ValueError):
        return "vLLM default STS smoke response length is not measurable"
    if parsed_word_count >= min_words:
        return None
    return (
        "vLLM default STS smoke response is too short to trust timing evidence: "
        f"{parsed_word_count} words < {min_words}."
    )


def _sts_segment_balance_failure(sts_probe: dict[str, Any]) -> str | None:
    assessment = sts_probe.get("sts_timing_assessment")
    if not isinstance(assessment, dict):
        return None
    if assessment.get("tts_tail_underfilled") is not True:
        return None
    segment_count = int(assessment.get("tts_segment_count") or 0)
    if segment_count < 4:
        return None
    tail_frames = assessment.get("tts_tail_codec_frames")
    mean_ratio = assessment.get("tts_tail_to_mean_ratio")
    return (
        "vLLM default STS smoke CFG chunk planner left an underfilled final "
        f"segment: {tail_frames} codec frames, tail/mean ratio {mean_ratio}, "
        f"{segment_count} segments."
    )


def _native_sampling_row_ratio_failure(sts_probe: dict[str, Any]) -> str | None:
    assessment = sts_probe.get("sts_timing_assessment")
    if not isinstance(assessment, dict):
        return None
    row_ratio = _optional_float(assessment.get("native_sampling_row_ratio"))
    if row_ratio is None:
        return None
    if row_ratio <= 0.75:
        return None
    return (
        "vLLM default STS smoke native CFG sampling row ratio is too high: "
        f"{row_ratio}. Expected about 0.5 for paired CFG sampling."
    )


def _sts_realtime_failure(sts_probe: dict[str, Any]) -> str | None:
    assessment = sts_probe.get("sts_timing_assessment")
    if not isinstance(assessment, dict):
        return "vLLM default STS smoke did not measure speech-token throughput"
    codec_fps = assessment.get("codec_frames_per_second")
    realtime_ratio = assessment.get("audio_realtime_ratio")
    if _optional_float(codec_fps) is None:
        return "vLLM default STS smoke did not measure speech-token throughput"
    if assessment.get("below_realtime") is not True:
        return None
    return (
        "vLLM default STS smoke speech-token throughput is below realtime: "
        f"codec_fps={codec_fps} realtime_ratio={realtime_ratio}."
    )


def _is_same_missing_metal_failure(report: dict[str, Any], error: str) -> bool:
    if "No Metal device available" not in error:
        return False
    spawn_error = str(report.get("spawn_probe", {}).get("mlx_error"))
    return "No Metal device available" in spawn_error
