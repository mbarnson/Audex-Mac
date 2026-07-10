from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from audex_mac.patches import vllm_metal_cfg

pytestmark = pytest.mark.fast


def _load_bench_module():
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "bench_vllm_metal_sampler.py"
    )
    spec = importlib.util.spec_from_file_location("bench_vllm_metal_sampler", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sampler_benchmark_builds_nvidia_cfg_pair_shape() -> None:
    bench = _load_bench_module()
    allowed_window = (131077, 196612, 131076)

    sampling_params = bench._sampling_params_for_args(
        SimpleNamespace(
            batch_size=1,
            cfg_pairs=2,
            temperature=1.0,
            top_k=80,
            cfg_scale=3.0,
        ),
        allowed_window,
    )
    plan = vllm_metal_cfg._build_cfg_pair_sampling_plan(sampling_params)

    assert len(sampling_params) == 4
    assert plan.sample_row_indices == (0, 2)
    assert plan.output_slots == ((0, 1), (2, 3))
    assert plan.temperatures == (1.0, 1.0)
    assert plan.top_ks == (80, 80)
    assert plan.allowed_token_windows == (allowed_window, allowed_window)
    assert sampling_params[0].extra_args["cfg_scale"] == 3.0
    assert sampling_params[0].extra_args["cfg_role"] == "cond"
    assert sampling_params[1].extra_args["cfg_role"] == "uncond"


def test_sampler_benchmark_keeps_no_cfg_shape_by_default() -> None:
    bench = _load_bench_module()
    allowed_window = (131077, 196612, 131076)

    sampling_params = bench._sampling_params_for_args(
        SimpleNamespace(
            batch_size=3,
            cfg_pairs=0,
            temperature=0.8,
            top_k=0,
            cfg_scale=3.0,
        ),
        allowed_window,
    )
    plan = vllm_metal_cfg._build_cfg_pair_sampling_plan(sampling_params)

    assert len(sampling_params) == 3
    assert plan.sample_row_indices == (0, 1, 2)
    assert plan.output_slots == ((0,), (1,), (2,))
    assert plan.temperatures == (0.8, 0.8, 0.8)
    assert plan.top_ks == (0, 0, 0)
