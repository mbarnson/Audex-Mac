from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from audex_mac.patches import vllm_metal_cfg
from audex_mac.vllm_cfg import (
    AudexMetalCFGTokenSyncInstaller,
    AudexVllmCfgConfig,
    configure_audex_vllm_cfg,
    find_audex_audiogen_scripts,
    require_audex_vllm_cfg_ready,
)

pytestmark = pytest.mark.fast


def write_fake_nvidia_cfg_scripts(snapshot: Path) -> Path:
    script_dir = snapshot / "inference_scripts_vllm" / "audiogen_scripts"
    script_dir.mkdir(parents=True)
    (script_dir / "vllm_cfg_patch.py").write_text(
        "\n".join(
            [
                "APPLIED = False",
                "def apply_cfg_patches():",
                "    global APPLIED",
                "    APPLIED = True",
            ]
        ),
        encoding="utf-8",
    )
    (script_dir / "cfg_logits_processor.py").write_text(
        "class CFGLogitsProcessor:\n    pass\n",
        encoding="utf-8",
    )
    return script_dir


def test_find_audex_audiogen_scripts_from_checkpoint_dir(tmp_path: Path) -> None:
    script_dir = write_fake_nvidia_cfg_scripts(tmp_path)
    model_path = tmp_path / "checkpoint_folder_full"
    model_path.mkdir()

    assert find_audex_audiogen_scripts(model_path) == script_dir


def test_vllm_metal_cfg_counts_complete_decode_pairs() -> None:
    decode_reqs = (
        (
            "cond-0",
            SimpleNamespace(
                sampling_params=SimpleNamespace(
                    extra_args={"cfg_role": "cond", "cfg_pair_id": "pair-0"}
                )
            ),
        ),
        (
            "uncond-0",
            SimpleNamespace(
                sampling_params=SimpleNamespace(
                    extra_args={"cfg_role": "uncond", "cfg_pair_id": "pair-0"}
                )
            ),
        ),
        (
            "cond-1",
            SimpleNamespace(
                sampling_params=SimpleNamespace(
                    extra_args={"cfg_role": "cond", "cfg_pair_id": "pair-1"}
                )
            ),
        ),
        ("plain", SimpleNamespace(sampling_params=SimpleNamespace(extra_args={}))),
    )

    assert vllm_metal_cfg._cfg_counts_for_decode_reqs(decode_reqs) == {
        "requests": 4,
        "cond": 2,
        "uncond": 1,
        "complete_pairs": 1,
    }


def test_vllm_metal_cfg_capacity_blocks_per_request_rounds_up() -> None:
    assert vllm_metal_cfg._max_length_blocks_per_request(5120, 1072) == 5


def test_vllm_metal_cfg_capacity_blocks_per_request_rejects_invalid_values() -> None:
    assert vllm_metal_cfg._max_length_blocks_per_request(0, 1072) is None
    assert vllm_metal_cfg._max_length_blocks_per_request(5120, 0) is None
    assert vllm_metal_cfg._max_length_blocks_per_request(None, 1072) is None


def test_configure_audex_vllm_cfg_uses_mac_friendly_engine_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_dir = write_fake_nvidia_cfg_scripts(tmp_path)
    model_path = tmp_path / "checkpoint_folder_full"
    model_path.mkdir()
    monkeypatch.delitem(sys.modules, "vllm_cfg_patch", raising=False)
    monkeypatch.delitem(sys.modules, "cfg_logits_processor", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/already/here")
    engine_kwargs: dict[str, object] = {"enable_prefix_caching": True}

    config = configure_audex_vllm_cfg(engine_kwargs, model_path)

    import vllm_cfg_patch

    assert config.ready is True
    assert config.script_dir == script_dir
    assert vllm_cfg_patch.APPLIED is True
    assert sys.path[0] == str(script_dir)
    assert str(script_dir) in sys.path
    assert str(script_dir) in sys.path[0]
    assert engine_kwargs["logits_processors"][0].__name__ == "CFGLogitsProcessor"
    assert engine_kwargs["logits_processors"][1] is AudexMetalCFGTokenSyncInstaller
    assert engine_kwargs["enable_prefix_caching"] is False
    assert engine_kwargs["max_model_len"] == 262_144
    assert engine_kwargs["max_num_batched_tokens"] == 262_144
    assert engine_kwargs["max_num_seqs"] == 16


def test_configure_audex_vllm_cfg_preserves_checkpoint_context_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_fake_nvidia_cfg_scripts(tmp_path)
    model_path = tmp_path / "checkpoint_folder_full"
    model_path.mkdir()
    monkeypatch.delitem(sys.modules, "vllm_cfg_patch", raising=False)
    monkeypatch.delitem(sys.modules, "cfg_logits_processor", raising=False)
    engine_kwargs: dict[str, object] = {
        "enable_prefix_caching": True,
        "max_model_len": 131_072,
    }

    config = configure_audex_vllm_cfg(engine_kwargs, model_path)

    assert config.ready is True
    assert config.max_model_len == 131_072
    assert config.max_num_batched_tokens == 131_072
    assert engine_kwargs["max_model_len"] == 131_072


def test_configure_audex_vllm_cfg_installs_worker_patch_for_no_cfg(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "checkpoint_folder_full"
    model_path.mkdir()
    engine_kwargs: dict[str, object] = {"enable_prefix_caching": True}

    config = configure_audex_vllm_cfg(engine_kwargs, model_path, cfg_scale=0.0)

    assert config.enabled is False
    assert config.ready is False
    assert engine_kwargs["enable_prefix_caching"] is True
    assert engine_kwargs["logits_processors"] == [AudexMetalCFGTokenSyncInstaller]
    assert "max_model_len" not in engine_kwargs
    assert "max_num_batched_tokens" not in engine_kwargs
    assert "max_num_seqs" not in engine_kwargs


def test_configure_audex_vllm_cfg_allows_diagnostic_max_num_seqs_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_dir = write_fake_nvidia_cfg_scripts(tmp_path)
    model_path = tmp_path / "checkpoint_folder_full"
    model_path.mkdir()
    monkeypatch.delitem(sys.modules, "vllm_cfg_patch", raising=False)
    monkeypatch.delitem(sys.modules, "cfg_logits_processor", raising=False)
    monkeypatch.setenv("AUDEX_VLLM_CFG_MAX_NUM_SEQS", "2")
    engine_kwargs: dict[str, object] = {}

    config = configure_audex_vllm_cfg(engine_kwargs, model_path)

    assert config.ready is True
    assert config.script_dir == script_dir
    assert config.max_num_seqs == 2
    assert engine_kwargs["max_num_seqs"] == 2
    assert engine_kwargs["max_num_batched_tokens"] == 262_144


def test_configure_audex_vllm_cfg_allows_diagnostic_batched_token_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_dir = write_fake_nvidia_cfg_scripts(tmp_path)
    model_path = tmp_path / "checkpoint_folder_full"
    model_path.mkdir()
    monkeypatch.delitem(sys.modules, "vllm_cfg_patch", raising=False)
    monkeypatch.delitem(sys.modules, "cfg_logits_processor", raising=False)
    monkeypatch.setenv("AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS", "524288")
    engine_kwargs: dict[str, object] = {}

    config = configure_audex_vllm_cfg(engine_kwargs, model_path)

    assert config.ready is True
    assert config.script_dir == script_dir
    assert config.max_num_batched_tokens == 524_288
    assert engine_kwargs["max_num_batched_tokens"] == 524_288


def test_configure_audex_vllm_cfg_allows_diagnostic_max_model_len_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_dir = write_fake_nvidia_cfg_scripts(tmp_path)
    model_path = tmp_path / "checkpoint_folder_full"
    model_path.mkdir()
    monkeypatch.delitem(sys.modules, "vllm_cfg_patch", raising=False)
    monkeypatch.delitem(sys.modules, "cfg_logits_processor", raising=False)
    monkeypatch.setenv("AUDEX_VLLM_CFG_MAX_MODEL_LEN", "2048")
    engine_kwargs: dict[str, object] = {}

    config = configure_audex_vllm_cfg(engine_kwargs, model_path)

    assert config.ready is True
    assert config.script_dir == script_dir
    assert config.max_model_len == 2048
    assert config.max_num_batched_tokens == 8192
    assert engine_kwargs["max_model_len"] == 2048


def test_configure_audex_vllm_cfg_allows_scheduler_reserve_diagnostic_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_dir = write_fake_nvidia_cfg_scripts(tmp_path)
    model_path = tmp_path / "checkpoint_folder_full"
    model_path.mkdir()
    monkeypatch.delitem(sys.modules, "vllm_cfg_patch", raising=False)
    monkeypatch.delitem(sys.modules, "cfg_logits_processor", raising=False)
    monkeypatch.setenv("AUDEX_VLLM_CFG_SCHEDULER_RESERVE_FULL_ISL", "0")
    engine_kwargs: dict[str, object] = {}

    config = configure_audex_vllm_cfg(engine_kwargs, model_path)

    assert config.ready is True
    assert config.script_dir == script_dir
    assert config.scheduler_reserve_full_isl is False
    assert engine_kwargs["scheduler_reserve_full_isl"] is False


def test_configure_audex_vllm_cfg_rejects_invalid_scheduler_reserve_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_fake_nvidia_cfg_scripts(tmp_path)
    model_path = tmp_path / "checkpoint_folder_full"
    model_path.mkdir()
    monkeypatch.delitem(sys.modules, "vllm_cfg_patch", raising=False)
    monkeypatch.delitem(sys.modules, "cfg_logits_processor", raising=False)
    monkeypatch.setenv("AUDEX_VLLM_CFG_SCHEDULER_RESERVE_FULL_ISL", "maybe")

    with pytest.raises(ValueError, match="AUDEX_VLLM_CFG_SCHEDULER_RESERVE_FULL_ISL"):
        configure_audex_vllm_cfg({}, model_path)


def test_configure_audex_vllm_cfg_clamps_low_batched_token_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_fake_nvidia_cfg_scripts(tmp_path)
    model_path = tmp_path / "checkpoint_folder_full"
    model_path.mkdir()
    monkeypatch.delitem(sys.modules, "vllm_cfg_patch", raising=False)
    monkeypatch.delitem(sys.modules, "cfg_logits_processor", raising=False)
    monkeypatch.setenv("AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS", "128")
    engine_kwargs: dict[str, object] = {}

    config = configure_audex_vllm_cfg(engine_kwargs, model_path)

    assert config.ready is True
    assert config.max_model_len == 262_144
    assert config.max_num_batched_tokens == 262_144
    assert engine_kwargs["max_num_batched_tokens"] == 262_144


def test_required_audex_vllm_cfg_fails_loudly_when_missing() -> None:
    config = AudexVllmCfgConfig(
        enabled=True,
        cfg_scale=2.0,
        script_dir=None,
        logits_processors=(),
        max_model_len=None,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        scheduler_reserve_full_isl=None,
        error="missing inference_scripts_vllm/audiogen_scripts",
    )

    with pytest.raises(RuntimeError, match="Audex vLLM CFG is not ready"):
        require_audex_vllm_cfg_ready(config)


def test_sync_cfg_token_ids_copies_conditional_token_to_unconditional() -> None:
    token_ids = [111, 222, 333]
    sampling_params = [
        SimpleNamespace(
            extra_args={
                "cfg_role": "cond",
                "cfg_pair_id": "pair-1",
            }
        ),
        SimpleNamespace(
            extra_args={
                "cfg_role": "uncond",
                "cfg_pair_id": "pair-1",
            }
        ),
        SimpleNamespace(extra_args=None),
    ]

    synced = vllm_metal_cfg.sync_cfg_token_ids(token_ids, sampling_params)

    assert synced == 1
    assert token_ids == [111, 111, 333]


def test_vllm_metal_cfg_patch_wraps_sampler_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    def fake_sample_from_logits(logits_2d, batch, sampler, device):
        return FakeSamplingResult([10, 20])

    fake_sampling_batch = types.ModuleType("vllm_metal.v1.sampling_batch")
    fake_sampling_batch._SamplingResult = FakeSamplingResult
    fake_sampling_batch.sample_from_logits = fake_sample_from_logits
    fake_sampling_batch.sample_prefill_tokens = lambda *args, **kwargs: None
    fake_model_runner = types.ModuleType("vllm_metal.v1.model_runner")
    fake_model_runner.sample_from_logits = fake_sample_from_logits
    fake_model_runner.sample_prefill_tokens = fake_sampling_batch.sample_prefill_tokens

    class FakeMetalModelRunner:
        def _sample_paged_batch(self, grammar_output=None):
            return "sampled"

        def _run_non_paged_decode_batch(self, batch):
            return "decoded"

    fake_model_runner.MetalModelRunner = FakeMetalModelRunner
    monkeypatch.setitem(sys.modules, "vllm_metal", types.ModuleType("vllm_metal"))
    monkeypatch.setitem(sys.modules, "vllm_metal.v1", types.ModuleType("vllm_metal.v1"))
    monkeypatch.setitem(
        sys.modules,
        "vllm_metal.v1.sampling_batch",
        fake_sampling_batch,
    )
    monkeypatch.setitem(sys.modules, "vllm_metal.v1.model_runner", fake_model_runner)
    fake_parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    fake_parallel_state.cleanup_dist_env_and_memory = lambda: None
    fake_engine_core = types.ModuleType("vllm.v1.engine.core")
    fake_engine_core.cleanup_dist_env_and_memory = lambda: None
    fake_scheduler_module = types.ModuleType("vllm.v1.core.sched.scheduler")

    class FakeScheduler:
        def __init__(self) -> None:
            self.waiting = []
            self.running = []
            self.requests = {}
            self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
            self.max_num_scheduled_tokens = 2

        def add_request(self, request):
            self.requests[request.request_id] = request

        def finish_requests(self, request_ids, finished_status):
            ids = {request_ids} if isinstance(request_ids, str) else set(request_ids)
            for request_id in ids:
                self.requests.pop(request_id, None)

        def schedule(self):
            return SimpleNamespace(
                num_scheduled_tokens={}, total_num_scheduled_tokens=0
            )

    fake_scheduler_module.Scheduler = FakeScheduler
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        types.ModuleType("vllm.distributed"),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.parallel_state", fake_parallel_state
    )
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.core", types.ModuleType("vllm.v1.core"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.core.sched",
        types.ModuleType("vllm.v1.core.sched"),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.v1.core.sched.scheduler", fake_scheduler_module
    )
    monkeypatch.setitem(
        sys.modules, "vllm.v1.engine", types.ModuleType("vllm.v1.engine")
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", fake_engine_core)

    report = vllm_metal_cfg.apply_vllm_metal_cfg_patches()
    batch = SimpleNamespace(
        sampling_params_list=[
            SimpleNamespace(extra_args={"cfg_role": "cond", "cfg_pair_id": "pair"}),
            SimpleNamespace(extra_args={"cfg_role": "uncond", "cfg_pair_id": "pair"}),
        ]
    )

    result = fake_sampling_batch.sample_from_logits(None, batch, None, None)

    assert report.ready is True
    assert result.token_ids == [10, 10]
    assert getattr(
        fake_scheduler_module.Scheduler.schedule,
        vllm_metal_cfg.SCHEDULER_PATCH_SENTINEL,
        False,
    )
    assert (
        fake_model_runner.sample_from_logits is fake_sampling_batch.sample_from_logits
    )
    assert getattr(
        fake_model_runner.MetalModelRunner._sample_paged_batch,
        vllm_metal_cfg.TIMING_PATCH_SENTINEL,
        False,
    )
    assert getattr(
        fake_model_runner.MetalModelRunner._run_non_paged_decode_batch,
        vllm_metal_cfg.TIMING_PATCH_SENTINEL,
        False,
    )
    assert getattr(
        fake_parallel_state.cleanup_dist_env_and_memory,
        vllm_metal_cfg.MPS_CLEANUP_PATCH_SENTINEL,
        False,
    )
    assert getattr(
        fake_engine_core.cleanup_dist_env_and_memory,
        vllm_metal_cfg.MPS_CLEANUP_PATCH_SENTINEL,
        False,
    )


def test_vllm_metal_cfg_nonpaged_tts_window_decode_uses_backbone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")
    vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY.clear()
    vllm_metal_cfg.NATIVE_DETAIL_COUNT_BY_CATEGORY.clear()

    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeSamplingBatch:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("TTS window fast path should not build SamplingBatch")

    class FakeHiddenStates:
        def __init__(self) -> None:
            self.slices: list[object] = []

        def __getitem__(self, index):
            self.slices.append(index)
            return "last-hidden-state"

    class FakeBackbone:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []
            self.hidden_states = FakeHiddenStates()

        def __call__(self, input_ids, *, cache):
            self.calls.append((input_ids, cache))
            return self.hidden_states

    class FakeModel:
        def __init__(self) -> None:
            self.model = FakeBackbone()

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("full logits model call should not be used")

    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.int32 = object()
    fake_mx.array = lambda value, dtype=None: ("array", value, dtype)
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    fake_runner = SimpleNamespace(
        _forward_model=None,
        _vocab_size=205312,
        device="mps",
        _logitsprocs="processors",
    )

    def fake_project(runner, hidden_states, allowed_window, mx):
        assert runner is fake_runner
        assert hidden_states == "last-hidden-state"
        assert allowed_window == (131077, 196612, 131076)
        assert mx is fake_mx
        return "window-logits"

    def fake_sample(logits, sampling_params_list, logitsprocs, sampling_result_cls, mx):
        assert logits == "window-logits"
        assert len(sampling_params_list) == 1
        assert logitsprocs == "processors"
        assert sampling_result_cls is FakeSamplingResult
        assert mx is fake_mx
        return FakeSamplingResult([131077])

    monkeypatch.setattr(vllm_metal_cfg, "_project_tts_window_logits", fake_project)
    monkeypatch.setattr(
        vllm_metal_cfg,
        "_sample_tts_window_logits_from_params_if_supported",
        fake_sample,
    )

    fake_model = FakeModel()
    fake_runner._forward_model = fake_model
    state = SimpleNamespace(
        token_ids=[131075],
        prompt_len=1,
        sampling_params=SimpleNamespace(
            extra_args={
                "audex_tts_codec_min_id": 131077,
                "audex_tts_codec_max_id": 196612,
                "audex_tts_speechgen_end_id": 131076,
            }
        ),
        generator=None,
        cache="cache",
        generated_tokens=0,
    )
    fake_model_runner = SimpleNamespace(
        SamplingBatch=FakeSamplingBatch,
        _SamplingResult=FakeSamplingResult,
    )

    result = vllm_metal_cfg._try_sequential_tts_window_decode(
        fake_runner,
        ("req-1", state),
        fake_model_runner,
    )

    assert result == FakeSamplingResult([131077])
    assert state.token_ids == [131075, 131077]
    assert state.generated_tokens == 1
    assert fake_model.model.calls == [(("array", [[131075]], fake_mx.int32), "cache")]
    assert fake_model.model.hidden_states.slices == [
        (slice(None), -1, slice(None)),
    ]
    assert "tts_window_forward" in vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY
    assert "tts_window_project" in vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY
    assert "tts_window_sample" in vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY


def test_vllm_metal_cfg_nonpaged_tts_window_decode_rejects_cfg() -> None:
    class FakeModel:
        @property
        def model(self):
            raise AssertionError("CFG request should be rejected before model access")

    state = SimpleNamespace(
        token_ids=[131075],
        prompt_len=1,
        sampling_params=SimpleNamespace(
            extra_args={
                "cfg_role": "cond",
                "cfg_pair_id": "pair-1",
                "audex_tts_codec_min_id": 131077,
                "audex_tts_codec_max_id": 196612,
                "audex_tts_speechgen_end_id": 131076,
            }
        ),
        generator=None,
        cache="cache",
        generated_tokens=0,
    )
    fake_runner = SimpleNamespace(_forward_model=FakeModel())

    result = vllm_metal_cfg._try_sequential_tts_window_decode(
        fake_runner,
        ("req-1", state),
        SimpleNamespace(),
    )

    assert result is None
    assert state.token_ids == [131075]
    assert state.generated_tokens == 0


def test_seeded_compact_sampling_derives_a_per_step_mlx_key() -> None:
    key_calls: list[int] = []
    fake_mx = SimpleNamespace(
        random=SimpleNamespace(key=lambda seed: key_calls.append(seed) or ("key", seed))
    )
    sampling_params = SimpleNamespace(
        seed=17,
        temperature=0.8,
        top_k=0,
        extra_args={},
    )
    state = SimpleNamespace(generated_tokens=9)

    keys = vllm_metal_cfg._seeded_random_keys_for_states(
        [sampling_params],
        [state],
        fake_mx,
    )

    expected_seed = (17 ^ (10 * 0x9E3779B9)) & 0xFFFFFFFF
    assert keys == (("key", expected_seed),)
    assert key_calls == [expected_seed]


def test_required_sequential_tts_window_decode_fails_instead_of_falling_back() -> None:
    class FakeRunner:
        def _sequential_decode(self, decode_reqs):
            return "fallback"

    original = FakeRunner._sequential_decode
    vllm_metal_cfg._patch_model_runner_sequential_tts_window_decode(
        FakeRunner,
        original,
        SimpleNamespace(),
    )
    state = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={"audex_tts_require_compact_window_decode": True}
        )
    )

    with pytest.raises(RuntimeError, match="required compact TTS-window decode"):
        FakeRunner()._sequential_decode([("required", state)])


def test_required_batched_tts_window_decode_fails_instead_of_falling_back() -> None:
    class FakeRunner:
        def _batched_decode(self, decode_reqs):
            return "fallback"

    original = FakeRunner._batched_decode
    vllm_metal_cfg._patch_model_runner_batched_tts_window_decode(
        FakeRunner,
        original,
        SimpleNamespace(),
    )
    required = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={"audex_tts_require_compact_window_decode": True}
        )
    )
    ordinary = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={}))

    with pytest.raises(RuntimeError, match="required compact TTS-window decode"):
        FakeRunner()._batched_decode([("required", required), ("ordinary", ordinary)])


def test_vllm_metal_cfg_batched_tts_window_decode_updates_all_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_NONPAGED_PERSISTENT_BATCH_CACHE", "0")

    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeSamplingBatch:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("TTS window fast path should not build SamplingBatch")

    class FakeArray:
        def __init__(self, value) -> None:
            self.value = value
            self.indices: list[object] = []

        def __getitem__(self, index):
            self.indices.append(index)
            return ("batched-input", self.value, index)

    class FakeHiddenStates:
        def __init__(self) -> None:
            self.slices: list[object] = []

        def __getitem__(self, index):
            self.slices.append(index)
            return "batched-last-hidden-state"

    class FakeBackbone:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []
            self.hidden_states = FakeHiddenStates()

        def __call__(self, input_ids, *, cache):
            self.calls.append((input_ids, cache))
            return self.hidden_states

    class FakeModel:
        def __init__(self) -> None:
            self.model = FakeBackbone()

    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.int32 = object()
    fake_mx.array = lambda value, dtype=None: FakeArray((value, dtype))
    eval_calls: list[tuple[object, ...]] = []
    fake_mx.eval = lambda *values: eval_calls.append(values)
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")
    monkeypatch.setenv("AUDEX_VLLM_DEBUG_SYNC_TTS_WINDOW_STAGES", "1")
    vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY.clear()
    vllm_metal_cfg.NATIVE_DETAIL_COUNT_BY_CATEGORY.clear()

    fake_runner = SimpleNamespace(
        _forward_model=FakeModel(),
        _vocab_size=205312,
        device="mps",
        _logitsprocs="processors",
    )
    sampling_params = SimpleNamespace(
        extra_args={
            "audex_tts_codec_min_id": 131077,
            "audex_tts_codec_max_id": 196612,
            "audex_tts_speechgen_end_id": 131076,
        }
    )
    state_a = SimpleNamespace(
        token_ids=[131075],
        prompt_len=1,
        sampling_params=sampling_params,
        generator=None,
        cache="cache-a",
        generated_tokens=0,
    )
    state_b = SimpleNamespace(
        token_ids=[131075],
        prompt_len=1,
        sampling_params=sampling_params,
        generator=None,
        cache="cache-b",
        generated_tokens=0,
    )
    merged_cache = ("merged", ("cache-a", "cache-b"))

    def fake_project(runner, hidden_states, allowed_window, mx):
        assert runner is fake_runner
        assert hidden_states == "batched-last-hidden-state"
        assert allowed_window == (131077, 196612, 131076)
        assert mx is fake_mx
        return "batched-window-logits"

    def fake_sample(logits, sampling_params_list, logitsprocs, sampling_result_cls, mx):
        assert logits == "batched-window-logits"
        assert len(sampling_params_list) == 2
        assert logitsprocs == "processors"
        assert sampling_result_cls is FakeSamplingResult
        assert mx is fake_mx
        return FakeSamplingResult([131077, 131078])

    fake_model_runner = SimpleNamespace(
        SamplingBatch=FakeSamplingBatch,
        _SamplingResult=FakeSamplingResult,
        _merge_kv_caches=lambda caches: ("merged", tuple(caches)),
        _extract_kv_cache=lambda cache, index: ("extracted", cache, index),
    )
    monkeypatch.setattr(vllm_metal_cfg, "_project_tts_window_logits", fake_project)
    monkeypatch.setattr(
        vllm_metal_cfg,
        "_sample_tts_window_logits_from_params_if_supported",
        fake_sample,
    )

    result = vllm_metal_cfg._try_batched_tts_window_decode(
        fake_runner,
        [("req-a", state_a), ("req-b", state_b)],
        fake_model_runner,
    )

    assert result == FakeSamplingResult([131077, 131078])
    assert fake_runner._forward_model.model.calls == [
        (
            ("batched-input", ([131075, 131075], fake_mx.int32), (slice(None), None)),
            merged_cache,
        )
    ]
    assert eval_calls == [
        (fake_runner._forward_model.model.hidden_states,),
        ("batched-window-logits",),
    ]
    assert state_a.token_ids == [131075, 131077]
    assert state_b.token_ids == [131075, 131078]
    assert state_a.cache == ("extracted", merged_cache, 0)
    assert state_b.cache == ("extracted", merged_cache, 1)
    assert state_a.generated_tokens == 1
    assert state_b.generated_tokens == 1
    assert (
        "tts_window_batch_forward" in vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY
    )
    assert (
        "tts_window_batch_project" in vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY
    )
    assert (
        "tts_window_batch_forward_eval"
        in vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY
    )
    assert (
        "tts_window_batch_project_eval"
        in vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY
    )
    assert "tts_window_batch_sample" in vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY


def test_vllm_metal_cfg_batched_tts_window_decode_supports_cfg_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_CFG_TTS_WINDOW_DECODE", "1")
    monkeypatch.setenv("AUDEX_VLLM_NONPAGED_PERSISTENT_BATCH_CACHE", "0")

    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeArray:
        def __init__(self, value) -> None:
            self.value = value

        def __getitem__(self, index):
            return ("batched-input", self.value, index)

    class FakeHiddenStates:
        def __getitem__(self, index):
            return "batched-last-hidden-state"

    class FakeBackbone:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def __call__(self, input_ids, *, cache):
            self.calls.append((input_ids, cache))
            return FakeHiddenStates()

    class FakeModel:
        def __init__(self) -> None:
            self.model = FakeBackbone()

    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.int32 = object()
    fake_mx.array = lambda value, dtype=None: FakeArray((value, dtype))
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    fake_runner = SimpleNamespace(
        _forward_model=FakeModel(),
        _vocab_size=205312,
        device="mps",
        _logitsprocs="processors",
    )
    cond_params = SimpleNamespace(
        extra_args={
            "cfg_role": "cond",
            "cfg_pair_id": "pair-1",
            "audex_tts_codec_min_id": 131077,
            "audex_tts_codec_max_id": 196612,
            "audex_tts_speechgen_end_id": 131076,
        }
    )
    uncond_params = SimpleNamespace(
        extra_args={
            "cfg_role": "uncond",
            "cfg_pair_id": "pair-1",
            "audex_tts_codec_min_id": 131077,
            "audex_tts_codec_max_id": 196612,
            "audex_tts_speechgen_end_id": 131076,
        }
    )
    cond_state = SimpleNamespace(
        token_ids=[131075],
        prompt_len=1,
        sampling_params=cond_params,
        generator=None,
        cache="cond-cache",
        generated_tokens=0,
    )
    uncond_state = SimpleNamespace(
        token_ids=[131075],
        prompt_len=1,
        sampling_params=uncond_params,
        generator=None,
        cache="uncond-cache",
        generated_tokens=0,
    )
    merged_cache = ("merged", ("cond-cache", "uncond-cache"))

    def fake_project(runner, hidden_states, allowed_window, mx):
        assert runner is fake_runner
        assert hidden_states == "batched-last-hidden-state"
        assert allowed_window == (131077, 196612, 131076)
        assert mx is fake_mx
        return "cfg-window-logits"

    def fake_sample(logits, sampling_params_list, logitsprocs, sampling_result_cls, mx):
        assert logits == "cfg-window-logits"
        assert sampling_params_list == [cond_params, uncond_params]
        assert logitsprocs == "processors"
        assert sampling_result_cls is FakeSamplingResult
        assert mx is fake_mx
        return FakeSamplingResult([131088, 131088])

    fake_model_runner = SimpleNamespace(
        _SamplingResult=FakeSamplingResult,
        _merge_kv_caches=lambda caches: ("merged", tuple(caches)),
        _extract_kv_cache=lambda cache, index: ("extracted", cache, index),
    )
    monkeypatch.setattr(vllm_metal_cfg, "_project_tts_window_logits", fake_project)
    monkeypatch.setattr(
        vllm_metal_cfg,
        "_sample_tts_window_logits_from_params_if_supported",
        fake_sample,
    )

    result = vllm_metal_cfg._try_batched_tts_window_decode(
        fake_runner,
        [("cond", cond_state), ("uncond", uncond_state)],
        fake_model_runner,
    )

    assert result == FakeSamplingResult([131088, 131088])
    assert cond_state.token_ids == [131075, 131088]
    assert uncond_state.token_ids == [131075, 131088]
    assert cond_state.cache == ("extracted", merged_cache, 0)
    assert uncond_state.cache == ("extracted", merged_cache, 1)
    assert cond_state.generated_tokens == 1
    assert uncond_state.generated_tokens == 1


def test_vllm_metal_cfg_batched_tts_window_decode_reuses_persistent_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeArray:
        def __init__(self, value) -> None:
            self.value = value

        def __getitem__(self, index):
            return ("batched-input", self.value, index)

    class FakeHiddenStates:
        def __getitem__(self, index):
            return "batched-last-hidden-state"

    class FakeBackbone:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def __call__(self, input_ids, *, cache):
            self.calls.append((input_ids, cache))
            return FakeHiddenStates()

    class FakeModel:
        def __init__(self) -> None:
            self.model = FakeBackbone()

    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.int32 = object()
    fake_mx.array = lambda value, dtype=None: FakeArray((value, dtype))
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    sampling_params = SimpleNamespace(
        extra_args={
            "audex_tts_codec_min_id": 131077,
            "audex_tts_codec_max_id": 196612,
            "audex_tts_speechgen_end_id": 131076,
        }
    )
    state_a = SimpleNamespace(
        token_ids=[131075],
        prompt_len=1,
        sampling_params=sampling_params,
        generator=None,
        cache="cache-a",
        generated_tokens=0,
    )
    state_b = SimpleNamespace(
        token_ids=[131075],
        prompt_len=1,
        sampling_params=sampling_params,
        generator=None,
        cache="cache-b",
        generated_tokens=0,
    )
    fake_runner = SimpleNamespace(
        _forward_model=FakeModel(),
        _vocab_size=205312,
        device="mps",
        _logitsprocs="processors",
    )
    merge_calls: list[tuple[object, ...]] = []
    extract_calls: list[tuple[object, int]] = []

    def merge_kv_caches(caches):
        merge_calls.append(tuple(caches))
        return ("merged", len(merge_calls), tuple(caches))

    def extract_kv_cache(cache, index):
        extract_calls.append((cache, index))
        return ("extracted", cache, index)

    fake_model_runner = SimpleNamespace(
        _SamplingResult=FakeSamplingResult,
        _merge_kv_caches=merge_kv_caches,
        _extract_kv_cache=extract_kv_cache,
    )
    sample_results = iter(
        [
            FakeSamplingResult([131077, 131078]),
            FakeSamplingResult([131079, 131080]),
        ]
    )
    monkeypatch.setattr(
        vllm_metal_cfg,
        "_project_tts_window_logits",
        lambda *_args: "window-logits",
    )
    monkeypatch.setattr(
        vllm_metal_cfg,
        "_sample_tts_window_logits_from_params_if_supported",
        lambda *_args: next(sample_results),
    )

    first = vllm_metal_cfg._try_batched_tts_window_decode(
        fake_runner,
        [("req-a", state_a), ("req-b", state_b)],
        fake_model_runner,
    )
    second = vllm_metal_cfg._try_batched_tts_window_decode(
        fake_runner,
        [("req-a", state_a), ("req-b", state_b)],
        fake_model_runner,
    )

    assert first == FakeSamplingResult([131077, 131078])
    assert second == FakeSamplingResult([131079, 131080])
    assert merge_calls == [("cache-a", "cache-b")]
    assert extract_calls == []
    assert state_a.cache == "cache-a"
    assert state_b.cache == "cache-b"
    assert state_a.token_ids == [131075, 131077, 131079]
    assert state_b.token_ids == [131075, 131078, 131080]
    assert state_a.generated_tokens == 2
    assert state_b.generated_tokens == 2
    assert [call[1] for call in fake_runner._forward_model.model.calls] == [
        ("merged", 1, ("cache-a", "cache-b")),
        ("merged", 1, ("cache-a", "cache-b")),
    ]


def test_vllm_metal_cfg_batched_tts_window_decode_rejects_cfg_by_default() -> None:
    class FakeModel:
        @property
        def model(self):
            raise AssertionError("CFG window decode should be rejected before model")

    cond_params = SimpleNamespace(
        extra_args={
            "cfg_role": "cond",
            "cfg_pair_id": "pair-1",
            "audex_tts_codec_min_id": 131077,
            "audex_tts_codec_max_id": 196612,
            "audex_tts_speechgen_end_id": 131076,
        }
    )
    uncond_params = SimpleNamespace(
        extra_args={
            "cfg_role": "uncond",
            "cfg_pair_id": "pair-1",
            "audex_tts_codec_min_id": 131077,
            "audex_tts_codec_max_id": 196612,
            "audex_tts_speechgen_end_id": 131076,
        }
    )
    cond_state = SimpleNamespace(
        token_ids=[131075],
        sampling_params=cond_params,
        generator=None,
    )
    uncond_state = SimpleNamespace(
        token_ids=[131075],
        sampling_params=uncond_params,
        generator=None,
    )
    fake_runner = SimpleNamespace(_forward_model=FakeModel())

    result = vllm_metal_cfg._try_batched_tts_window_decode(
        fake_runner,
        [("cond", cond_state), ("uncond", uncond_state)],
        SimpleNamespace(),
    )

    assert result is None


def test_vllm_metal_cfg_batched_decode_submits_logits_with_async_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeArray:
        def __init__(self, value) -> None:
            self.value = value

        def __getitem__(self, index):
            return ("batched-input", self.value, index)

    class FakeLogits:
        def __getitem__(self, index):
            return ("next-token-logits", index)

    class FakeForwardModel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def __call__(self, input_ids, *, cache):
            self.calls.append((input_ids, cache))
            return "model-output"

    class FakeSamplingBatch:
        def __init__(
            self,
            sampling_params_list,
            prompt_token_ids_list,
            output_tokens_list,
            *,
            vocab_size,
            device,
            logitsprocs,
            generators,
        ) -> None:
            self.sampling_params_list = sampling_params_list
            self.prompt_token_ids_list = prompt_token_ids_list
            self.output_tokens_list = output_tokens_list
            self.vocab_size = vocab_size
            self.device = device
            self.logitsprocs = logitsprocs
            self.generators = generators

    async_calls: list[tuple[object, ...]] = []
    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.int32 = object()
    fake_mx.array = lambda value, dtype=None: FakeArray((value, dtype))
    fake_mx.async_eval = lambda *args: async_calls.append(args)
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")
    monkeypatch.setenv("AUDEX_VLLM_NONPAGED_PERSISTENT_BATCH_CACHE", "0")
    vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY.clear()
    vllm_metal_cfg.NATIVE_DETAIL_COUNT_BY_CATEGORY.clear()

    params_a = SimpleNamespace(extra_args=None)
    params_b = SimpleNamespace(extra_args=None)
    state_a = SimpleNamespace(
        token_ids=[10, 11],
        prompt_len=1,
        sampling_params=params_a,
        generator=None,
        cache="cache-a",
        generated_tokens=0,
    )
    state_b = SimpleNamespace(
        token_ids=[20, 21, 22],
        prompt_len=2,
        sampling_params=params_b,
        generator="gen-b",
        cache="cache-b",
        generated_tokens=3,
    )
    merged_cache = ("merged", ("cache-a", "cache-b"))
    forward_model = FakeForwardModel()
    fake_runner = SimpleNamespace(
        _forward_model=forward_model,
        _extract_logits=lambda output: FakeLogits(),
        _vocab_size=205312,
        device="mps",
        _logitsprocs="processors",
        _sampler="sampler",
    )

    captured_batches: list[FakeSamplingBatch] = []

    def fake_sample_from_logits(logits, batch, sampler, device):
        assert logits == ("next-token-logits", (slice(None), -1, slice(None)))
        assert sampler == "sampler"
        assert device == "mps"
        captured_batches.append(batch)
        return FakeSamplingResult([101, 202], logprobs="logprobs")

    fake_model_runner = SimpleNamespace(
        SamplingBatch=FakeSamplingBatch,
        _SamplingResult=FakeSamplingResult,
        sample_from_logits=fake_sample_from_logits,
        _merge_kv_caches=lambda caches: ("merged", tuple(caches)),
        _extract_kv_cache=lambda cache, index: ("extracted", cache, index),
    )

    result = vllm_metal_cfg._try_batched_decode_with_async_eval(
        fake_runner,
        [("req-a", state_a), ("req-b", state_b)],
        fake_model_runner,
    )

    assert result == FakeSamplingResult([101, 202], "logprobs")
    assert async_calls == [(("next-token-logits", (slice(None), -1, slice(None))),)]
    assert forward_model.calls == [
        (
            ("batched-input", ([11, 22], fake_mx.int32), (slice(None), None)),
            merged_cache,
        )
    ]
    assert captured_batches[0].sampling_params_list == [params_a, params_b]
    assert captured_batches[0].prompt_token_ids_list == [[10], [20, 21]]
    assert captured_batches[0].output_tokens_list == [[11], [22]]
    assert captured_batches[0].generators == {1: "gen-b"}
    assert state_a.cache == ("extracted", merged_cache, 0)
    assert state_b.cache == ("extracted", merged_cache, 1)
    assert state_a.token_ids == [10, 11, 101]
    assert state_b.token_ids == [20, 21, 22, 202]
    assert state_a.generated_tokens == 1
    assert state_b.generated_tokens == 4
    assert (
        "nonpaged_decode_logits_async_submit"
        in vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY
    )


def test_vllm_metal_cfg_batched_decode_can_skip_full_logits_async_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeArray:
        def __init__(self, value) -> None:
            self.value = value

        def __getitem__(self, index):
            return ("batched-input", self.value, index)

    class FakeLogits:
        def __getitem__(self, index):
            return ("next-token-logits", index)

    async_calls: list[tuple[object, ...]] = []
    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.int32 = object()
    fake_mx.array = lambda value, dtype=None: FakeArray((value, dtype))
    fake_mx.async_eval = lambda *args: async_calls.append(args)
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setenv("AUDEX_VLLM_NONPAGED_ASYNC_EVAL_TARGET", "sample_logits")

    state = SimpleNamespace(
        token_ids=[10],
        prompt_len=1,
        sampling_params=SimpleNamespace(extra_args=None),
        generator=None,
        cache="cache",
        generated_tokens=0,
    )
    fake_runner = SimpleNamespace(
        _forward_model=lambda _input_ids, *, cache: ("model-output", cache),
        _extract_logits=lambda _output: FakeLogits(),
        _vocab_size=205312,
        device="mps",
        _logitsprocs=(),
        _sampler="sampler",
    )
    fake_model_runner = SimpleNamespace(
        SamplingBatch=lambda *args, **kwargs: SimpleNamespace(
            args=args,
            kwargs=kwargs,
        ),
        _SamplingResult=FakeSamplingResult,
        sample_from_logits=lambda *_args: FakeSamplingResult([101]),
        _merge_kv_caches=lambda caches: ("merged", tuple(caches)),
        _extract_kv_cache=lambda cache, index: ("extracted", cache, index),
    )

    result = vllm_metal_cfg._try_batched_decode_with_async_eval(
        fake_runner,
        [("req", state)],
        fake_model_runner,
    )

    assert result == FakeSamplingResult([101])
    assert async_calls == []


def test_vllm_metal_cfg_batched_decode_reuses_persistent_batch_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeArray:
        def __init__(self, value) -> None:
            self.value = value

        def __getitem__(self, index):
            return ("batched-input", self.value, index)

    class FakeLogits:
        def __getitem__(self, index):
            return ("next-token-logits", index)

    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.int32 = object()
    fake_mx.array = lambda value, dtype=None: FakeArray((value, dtype))
    fake_mx.async_eval = lambda *_args: None
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    state_a = SimpleNamespace(
        token_ids=[10],
        prompt_len=1,
        sampling_params=SimpleNamespace(extra_args=None),
        generator=None,
        cache="cache-a",
        generated_tokens=0,
    )
    state_b = SimpleNamespace(
        token_ids=[20],
        prompt_len=1,
        sampling_params=SimpleNamespace(extra_args=None),
        generator=None,
        cache="cache-b",
        generated_tokens=0,
    )
    forward_calls: list[tuple[object, object]] = []
    fake_runner = SimpleNamespace(
        _forward_model=lambda input_ids, *, cache: forward_calls.append(
            (input_ids, cache)
        )
        or "model-output",
        _extract_logits=lambda _output: FakeLogits(),
        _vocab_size=205312,
        device="mps",
        _logitsprocs=(),
        _sampler="sampler",
    )
    sample_results = iter(
        [
            FakeSamplingResult([101, 202]),
            FakeSamplingResult([102, 203]),
        ]
    )
    merge_calls: list[tuple[object, ...]] = []
    extract_calls: list[tuple[object, int]] = []

    def merge_kv_caches(caches):
        merge_calls.append(tuple(caches))
        return ("merged-cache", len(merge_calls))

    def extract_kv_cache(cache, index):
        extract_calls.append((cache, index))
        return ("extracted", cache, index)

    fake_model_runner = SimpleNamespace(
        SamplingBatch=lambda *args, **kwargs: SimpleNamespace(
            args=args,
            kwargs=kwargs,
        ),
        _SamplingResult=FakeSamplingResult,
        sample_from_logits=lambda *_args: next(sample_results),
        _merge_kv_caches=merge_kv_caches,
        _extract_kv_cache=extract_kv_cache,
    )

    first = vllm_metal_cfg._try_batched_decode_with_async_eval(
        fake_runner,
        [("req-a", state_a), ("req-b", state_b)],
        fake_model_runner,
    )
    second = vllm_metal_cfg._try_batched_decode_with_async_eval(
        fake_runner,
        [("req-a", state_a), ("req-b", state_b)],
        fake_model_runner,
    )

    assert first == FakeSamplingResult([101, 202])
    assert second == FakeSamplingResult([102, 203])
    assert merge_calls == [("cache-a", "cache-b")]
    assert extract_calls == []
    assert forward_calls[0][1] == ("merged-cache", 1)
    assert forward_calls[1][1] == ("merged-cache", 1)
    assert state_a.cache == "cache-a"
    assert state_b.cache == "cache-b"
    assert state_a.token_ids == [10, 101, 102]
    assert state_b.token_ids == [20, 202, 203]
    assert state_a.generated_tokens == 2
    assert state_b.generated_tokens == 2

    vllm_metal_cfg._flush_persistent_nonpaged_batch_cache(
        fake_runner,
        fake_model_runner,
        reason="test",
    )

    assert extract_calls == [
        (("merged-cache", 1), 0),
        (("merged-cache", 1), 1),
    ]
    assert state_a.cache == ("extracted", ("merged-cache", 1), 0)
    assert state_b.cache == ("extracted", ("merged-cache", 1), 1)
    assert not hasattr(fake_runner, "_audex_nonpaged_batch_cache")


def test_vllm_metal_cfg_batched_decode_async_eval_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_NONPAGED_ASYNC_EVAL", "0")
    fake_runner = SimpleNamespace(
        _forward_model=lambda *_args, **_kwargs: pytest.fail("should not run")
    )

    result = vllm_metal_cfg._try_batched_decode_with_async_eval(
        fake_runner,
        [],
        SimpleNamespace(),
    )

    assert result is None


def test_vllm_metal_cfg_caches_tts_window_head() -> None:
    calls: list[tuple[str, object]] = []

    class FakeTensor:
        def __init__(self, name: str) -> None:
            self.name = name

        def astype(self, _dtype):
            calls.append(("astype", self.name))
            return ("astype", self.name)

    class FakeWeight:
        def __getitem__(self, index):
            calls.append(("weight_slice", index))
            return FakeTensor(f"weight:{index}")

    class FakeBias:
        def __getitem__(self, index):
            calls.append(("bias_slice", index))
            return FakeTensor(f"bias:{index}")

    fake_mx = SimpleNamespace(
        float32=object(),
        concatenate=lambda values, axis: calls.append(("concatenate", (values, axis)))
        or FakeTensor(f"concat:{len(calls)}"),
        transpose=lambda value: calls.append(("transpose", value.name))
        or FakeTensor(f"transpose:{value.name}"),
    )
    runner = SimpleNamespace()
    lm_head = SimpleNamespace(weight=FakeWeight(), bias=FakeBias())
    window = (131077, 196612, 131076)

    first = vllm_metal_cfg._cached_tts_window_head(
        runner,
        lm_head,
        window,
        fake_mx,
    )
    calls_after_first = list(calls)
    second = vllm_metal_cfg._cached_tts_window_head(
        runner,
        lm_head,
        window,
        fake_mx,
    )

    assert second == first
    assert calls == calls_after_first
    assert [call[0] for call in calls] == [
        "weight_slice",
        "weight_slice",
        "concatenate",
        "transpose",
        "astype",
        "bias_slice",
        "bias_slice",
        "concatenate",
        "astype",
    ]


def test_vllm_metal_cfg_scheduler_keeps_cfg_pairs_aligned() -> None:
    cond = SimpleNamespace(
        request_id="cond",
        sampling_params=SimpleNamespace(
            extra_args={"cfg_role": "cond", "cfg_pair_id": "pair"}
        ),
        num_computed_tokens=7,
    )
    uncond = SimpleNamespace(
        request_id="uncond",
        sampling_params=SimpleNamespace(
            extra_args={"cfg_role": "uncond", "cfg_pair_id": "pair"}
        ),
        num_computed_tokens=5,
    )
    other = SimpleNamespace(
        request_id="other",
        sampling_params=SimpleNamespace(extra_args=None),
        num_computed_tokens=1,
    )
    scheduler = SimpleNamespace(
        waiting=[cond, other, uncond],
        running=[cond, uncond],
        requests={"cond": cond, "uncond": uncond, "other": other},
        _audex_cfg_pairs={"pair": {"cond": "cond", "uncond": "uncond"}},
        _audex_cfg_req_to_pair={"cond": "pair", "uncond": "pair"},
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"cond": 3, "uncond": 3},
        total_num_scheduled_tokens=6,
    )

    vllm_metal_cfg._reorder_waiting_for_cfg(scheduler)
    vllm_metal_cfg._equalize_cfg_pair_progress(scheduler, scheduler_output)

    assert [request.request_id for request in scheduler.waiting] == [
        "cond",
        "uncond",
        "other",
    ]
    assert cond.num_computed_tokens == 5
    assert uncond.num_computed_tokens == 5
    assert scheduler_output.num_scheduled_tokens == {"cond": 1, "uncond": 3}
    assert scheduler_output.total_num_scheduled_tokens == 4


def test_vllm_metal_scheduler_gives_tts_exclusive_steps_and_resumes_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUDEX_VLLM_SPEECH_FIRST_SCHEDULING", raising=False)
    text = SimpleNamespace(
        request_id="text",
        sampling_params=SimpleNamespace(extra_args={}),
    )
    tts = SimpleNamespace(
        request_id="tts",
        sampling_params=SimpleNamespace(
            extra_args={"audex_tts_speechgen_end_id": 205_311}
        ),
    )
    scheduler = SimpleNamespace(
        running=[text],
        waiting=[tts],
        skipped_waiting=[],
        requests={"text": text, "tts": tts},
    )
    scheduled_batches: list[tuple[str, ...]] = []

    def schedule_once():
        if tts in scheduler.waiting:
            scheduler.waiting.remove(tts)
            scheduler.running.append(tts)
        batch = tuple(request.request_id for request in scheduler.running)
        scheduled_batches.append(batch)
        return SimpleNamespace(
            num_scheduled_tokens={request_id: 1 for request_id in batch}
        )

    first = vllm_metal_cfg._schedule_with_tts_priority(scheduler, schedule_once)
    second = vllm_metal_cfg._schedule_with_tts_priority(scheduler, schedule_once)

    assert first.num_scheduled_tokens == {"tts": 1}
    assert second.num_scheduled_tokens == {"tts": 1}
    assert scheduler.running == [tts, text]

    scheduler.running.remove(tts)
    scheduler.requests.pop("tts")
    third = vllm_metal_cfg._schedule_with_tts_priority(scheduler, schedule_once)

    assert third.num_scheduled_tokens == {"text": 1}
    assert scheduled_batches == [("tts",), ("tts",), ("text",)]


def test_vllm_metal_scheduler_can_disable_speech_first_scheduling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDEX_VLLM_SPEECH_FIRST_SCHEDULING", "0")
    text = SimpleNamespace(
        request_id="text",
        sampling_params=SimpleNamespace(extra_args={}),
    )
    tts = SimpleNamespace(
        request_id="tts",
        sampling_params=SimpleNamespace(
            extra_args={"audex_tts_speechgen_end_id": 205_311}
        ),
    )
    scheduler = SimpleNamespace(
        running=[text, tts],
        waiting=[],
        skipped_waiting=[],
        requests={"text": text, "tts": tts},
    )

    output = vllm_metal_cfg._schedule_with_tts_priority(
        scheduler,
        lambda: SimpleNamespace(
            num_scheduled_tokens={
                request.request_id: 1 for request in scheduler.running
            }
        ),
    )

    assert output.num_scheduled_tokens == {"text": 1, "tts": 1}
    assert scheduler.running == [text, tts]


def test_vllm_metal_scheduler_keeps_cfg_tts_pair_while_holding_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUDEX_VLLM_SPEECH_FIRST_SCHEDULING", raising=False)
    text = SimpleNamespace(
        request_id="text",
        sampling_params=SimpleNamespace(extra_args={}),
    )
    cond = SimpleNamespace(
        request_id="cond",
        sampling_params=SimpleNamespace(
            extra_args={"cfg_role": "cond", "cfg_pair_id": "pair"}
        ),
    )
    uncond = SimpleNamespace(
        request_id="uncond",
        sampling_params=SimpleNamespace(
            extra_args={"cfg_role": "uncond", "cfg_pair_id": "pair"}
        ),
    )
    scheduler = SimpleNamespace(
        running=[text, cond, uncond],
        waiting=[],
        skipped_waiting=[],
        requests={"text": text, "cond": cond, "uncond": uncond},
    )

    output = vllm_metal_cfg._schedule_with_tts_priority(
        scheduler,
        lambda: SimpleNamespace(
            num_scheduled_tokens={
                request.request_id: 1 for request in scheduler.running
            }
        ),
    )

    assert output.num_scheduled_tokens == {"cond": 1, "uncond": 1}
    assert scheduler.running == [cond, uncond, text]


def test_vllm_metal_cfg_paged_timing_wrapper_records_runner_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRunner:
        def __init__(self) -> None:
            self._execute_model_state = SimpleNamespace(
                decode_reqs=(("a", object()), ("b", object())),
                prefill_reqs=(object(),),
                num_decode_tokens=2,
            )

        def _sample_paged_batch(self, grammar_output=None):
            return "sampled"

    fake_model_runner = SimpleNamespace(MetalModelRunner=FakeRunner)
    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")

    vllm_metal_cfg._patch_model_runner_paged_timing(fake_model_runner)
    runner = FakeRunner()

    assert runner._sample_paged_batch() == "sampled"
    assert runner._audex_mac_paged_sample_count == 1
    assert runner._audex_mac_paged_sample_seconds >= 0.0


def test_vllm_metal_cfg_non_paged_timing_wrapper_records_runner_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRunner:
        def _run_non_paged_decode_batch(self, batch):
            return "decoded"

    fake_model_runner = SimpleNamespace(MetalModelRunner=FakeRunner)
    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")

    vllm_metal_cfg._patch_model_runner_non_paged_timing(fake_model_runner)
    runner = FakeRunner()
    batch = SimpleNamespace(
        valid_decode_reqs=(("a", object()),),
        scheduled_cached_req_ids=("a",),
    )

    assert runner._run_non_paged_decode_batch(batch) == "decoded"
    assert runner._audex_mac_non_paged_decode_count == 1
    assert runner._audex_mac_non_paged_decode_seconds >= 0.0


def test_vllm_metal_cfg_suppresses_only_mps_cleanup_allocator_assert() -> None:
    fake_module = types.ModuleType("fake_cleanup_module")
    calls = {"count": 0}

    def failing_cleanup():
        calls["count"] += 1
        raise RuntimeError(
            "device_allocator INTERNAL ASSERT FAILED: "
            "Allocator for mps is not a DeviceAllocator"
        )

    fake_module.cleanup_dist_env_and_memory = failing_cleanup

    assert vllm_metal_cfg._patch_cleanup_symbol(fake_module) is True
    fake_module.cleanup_dist_env_and_memory()
    assert calls["count"] == 1

    def unrelated_cleanup():
        raise RuntimeError("different cleanup failure")

    other_module = types.ModuleType("other_cleanup_module")
    other_module.cleanup_dist_env_and_memory = unrelated_cleanup
    vllm_metal_cfg._patch_cleanup_symbol(other_module)

    with pytest.raises(RuntimeError, match="different cleanup failure"):
        other_module.cleanup_dist_env_and_memory()


def test_vllm_metal_cfg_paged_timing_records_mx_eval_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeArray:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    fake_mx = SimpleNamespace(eval=lambda *args, **_kwargs: calls.append(args))

    class FakeRunner:
        def __init__(self) -> None:
            self._execute_model_state = SimpleNamespace(
                decode_reqs=(("a", object()),),
                prefill_reqs=(),
                num_decode_tokens=1,
            )

        def _sample_paged_batch(self, grammar_output=None):
            fake_mx.eval(FakeArray((1, 2, 205312)))
            fake_mx.eval(FakeArray((2,)))
            return "sampled"

    fake_model_runner = SimpleNamespace(MetalModelRunner=FakeRunner, mx=fake_mx)
    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")
    vllm_metal_cfg.MX_EVAL_SECONDS_BY_CATEGORY.clear()
    vllm_metal_cfg.MX_EVAL_COUNT_BY_CATEGORY.clear()

    vllm_metal_cfg._patch_mx_eval_timing(fake_model_runner)
    vllm_metal_cfg._patch_model_runner_paged_timing(fake_model_runner)
    runner = FakeRunner()

    assert runner._sample_paged_batch() == "sampled"
    assert len(calls) == 2
    assert vllm_metal_cfg.MX_EVAL_COUNT_BY_CATEGORY["logits"] == 1
    assert vllm_metal_cfg.MX_EVAL_COUNT_BY_CATEGORY["sample_tokens"] == 1
    assert "logits:" in vllm_metal_cfg._mx_eval_timing_summary()
    assert "sample_tokens:" in vllm_metal_cfg._mx_eval_timing_summary()


def test_vllm_metal_cfg_paged_timing_skips_cfg_decode_logits_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeArray:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    fake_mx = SimpleNamespace(eval=lambda *args, **_kwargs: calls.append(args))

    class FakeRunner:
        def __init__(self) -> None:
            self._execute_model_state = SimpleNamespace(
                decode_reqs=(
                    (
                        "cond",
                        SimpleNamespace(
                            sampling_params=SimpleNamespace(
                                extra_args={
                                    "cfg_role": "cond",
                                    "cfg_pair_id": "pair-1",
                                }
                            )
                        ),
                    ),
                    (
                        "uncond",
                        SimpleNamespace(
                            sampling_params=SimpleNamespace(
                                extra_args={
                                    "cfg_role": "uncond",
                                    "cfg_pair_id": "pair-1",
                                }
                            )
                        ),
                    ),
                ),
                prefill_reqs=(),
                target_hidden_states=None,
                pooling_hidden_states=None,
                num_decode_tokens=2,
            )

        def _sample_paged_batch(self, grammar_output=None):
            fake_mx.eval(FakeArray((1, 2, 205312)))
            fake_mx.eval(FakeArray((2,)))
            return "sampled"

    fake_model_runner = SimpleNamespace(MetalModelRunner=FakeRunner, mx=fake_mx)
    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")
    monkeypatch.setenv("AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL", "1")
    vllm_metal_cfg.MX_EVAL_SECONDS_BY_CATEGORY.clear()
    vllm_metal_cfg.MX_EVAL_COUNT_BY_CATEGORY.clear()
    vllm_metal_cfg.MX_EVAL_SHAPE_COUNT_BY_CATEGORY.clear()
    vllm_metal_cfg.PAGED_LOGITS_EVAL_SKIP_COUNT = 0
    vllm_metal_cfg.SKIP_NEXT_PAGED_LOGITS_EVAL = False

    vllm_metal_cfg._patch_mx_eval_timing(fake_model_runner)
    vllm_metal_cfg._patch_model_runner_paged_timing(fake_model_runner)
    runner = FakeRunner()

    assert runner._sample_paged_batch() == "sampled"
    assert len(calls) == 1
    assert calls[0][0].shape == (2,)
    assert vllm_metal_cfg.PAGED_LOGITS_EVAL_SKIP_COUNT == 1
    assert "logits" not in vllm_metal_cfg.MX_EVAL_COUNT_BY_CATEGORY
    assert vllm_metal_cfg.MX_EVAL_COUNT_BY_CATEGORY["sample_tokens"] == 1


def test_vllm_metal_cfg_paged_timing_skips_logits_eval_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeArray:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    fake_mx = SimpleNamespace(eval=lambda *args, **_kwargs: calls.append(args))

    class FakeRunner:
        def __init__(self) -> None:
            self._execute_model_state = SimpleNamespace(
                decode_reqs=(
                    (
                        "cond",
                        SimpleNamespace(
                            sampling_params=SimpleNamespace(
                                extra_args={
                                    "cfg_role": "cond",
                                    "cfg_pair_id": "pair-1",
                                }
                            )
                        ),
                    ),
                    (
                        "uncond",
                        SimpleNamespace(
                            sampling_params=SimpleNamespace(
                                extra_args={
                                    "cfg_role": "uncond",
                                    "cfg_pair_id": "pair-1",
                                }
                            )
                        ),
                    ),
                ),
                prefill_reqs=(),
                target_hidden_states=None,
                pooling_hidden_states=None,
                num_decode_tokens=2,
            )

        def _sample_paged_batch(self, grammar_output=None):
            fake_mx.eval(FakeArray((1, 2, 205312)))
            fake_mx.eval(FakeArray((2,)))
            return "sampled"

    fake_model_runner = SimpleNamespace(MetalModelRunner=FakeRunner, mx=fake_mx)
    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")
    monkeypatch.delenv("AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL", raising=False)
    vllm_metal_cfg.MX_EVAL_SECONDS_BY_CATEGORY.clear()
    vllm_metal_cfg.MX_EVAL_COUNT_BY_CATEGORY.clear()
    vllm_metal_cfg.MX_EVAL_SHAPE_COUNT_BY_CATEGORY.clear()
    vllm_metal_cfg.PAGED_LOGITS_EVAL_SKIP_COUNT = 0
    vllm_metal_cfg.SKIP_NEXT_PAGED_LOGITS_EVAL = False

    vllm_metal_cfg._patch_mx_eval_timing(fake_model_runner)
    vllm_metal_cfg._patch_model_runner_paged_timing(fake_model_runner)
    runner = FakeRunner()

    assert runner._sample_paged_batch() == "sampled"
    assert len(calls) == 1
    assert calls[0][0].shape == (2,)
    assert vllm_metal_cfg.PAGED_LOGITS_EVAL_SKIP_COUNT == 1
    assert "logits" not in vllm_metal_cfg.MX_EVAL_COUNT_BY_CATEGORY
    assert vllm_metal_cfg.MX_EVAL_COUNT_BY_CATEGORY["sample_tokens"] == 1


def test_vllm_metal_cfg_paged_timing_can_force_logits_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeArray:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    fake_mx = SimpleNamespace(eval=lambda *args, **_kwargs: calls.append(args))

    class FakeRunner:
        def __init__(self) -> None:
            self._execute_model_state = SimpleNamespace(
                decode_reqs=(
                    (
                        "cond",
                        SimpleNamespace(
                            sampling_params=SimpleNamespace(
                                extra_args={
                                    "cfg_role": "cond",
                                    "cfg_pair_id": "pair-1",
                                }
                            )
                        ),
                    ),
                    (
                        "uncond",
                        SimpleNamespace(
                            sampling_params=SimpleNamespace(
                                extra_args={
                                    "cfg_role": "uncond",
                                    "cfg_pair_id": "pair-1",
                                }
                            )
                        ),
                    ),
                ),
                prefill_reqs=(),
                target_hidden_states=None,
                pooling_hidden_states=None,
                num_decode_tokens=2,
            )

        def _sample_paged_batch(self, grammar_output=None):
            fake_mx.eval(FakeArray((1, 2, 205312)))
            fake_mx.eval(FakeArray((2,)))
            return "sampled"

    fake_model_runner = SimpleNamespace(MetalModelRunner=FakeRunner, mx=fake_mx)
    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")
    monkeypatch.setenv("AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL", "0")
    vllm_metal_cfg.MX_EVAL_SECONDS_BY_CATEGORY.clear()
    vllm_metal_cfg.MX_EVAL_COUNT_BY_CATEGORY.clear()
    vllm_metal_cfg.MX_EVAL_SHAPE_COUNT_BY_CATEGORY.clear()
    vllm_metal_cfg.PAGED_LOGITS_EVAL_SKIP_COUNT = 0
    vllm_metal_cfg.SKIP_NEXT_PAGED_LOGITS_EVAL = False

    vllm_metal_cfg._patch_mx_eval_timing(fake_model_runner)
    vllm_metal_cfg._patch_model_runner_paged_timing(fake_model_runner)
    runner = FakeRunner()

    assert runner._sample_paged_batch() == "sampled"
    assert len(calls) == 2
    assert vllm_metal_cfg.PAGED_LOGITS_EVAL_SKIP_COUNT == 0
    assert vllm_metal_cfg.MX_EVAL_COUNT_BY_CATEGORY["logits"] == 1
    assert vllm_metal_cfg.MX_EVAL_COUNT_BY_CATEGORY["sample_tokens"] == 1


def test_vllm_metal_cfg_sampler_uses_native_mlx_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeRow:
        def __init__(self, name: str) -> None:
            self.name = name
            self.cast = False

        def astype(self, _dtype):
            self.cast = True
            return self

        def __add__(self, other):
            return FakeRow(f"({self.name}+{other.name})")

        def __sub__(self, other):
            return FakeRow(f"({self.name}-{other.name})")

        def __mul__(self, other):
            return FakeRow(f"({self.name}*{other})")

        def __rmul__(self, other):
            return FakeRow(f"({other}*{self.name})")

    class FakeLogits:
        def __init__(self) -> None:
            self.selected_rows: list[int] = []
            self.full_cast = False

        def astype(self, _dtype):
            self.full_cast = True
            return self

        def __getitem__(self, index):
            self.selected_rows.append(index)
            return FakeRow(f"row-{index}")

        def __mul__(self, other):
            return self

        def __truediv__(self, _other):
            return self

    class FakeTemperatures:
        def __getitem__(self, _index):
            return self

    class FakeTokens:
        def __init__(self, token_ids: list[int]) -> None:
            self.token_ids = token_ids

        def tolist(self) -> list[int]:
            return list(self.token_ids)

    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.float32 = object()
    fake_mx.array = lambda *_args, **_kwargs: FakeTemperatures()
    eval_calls: list[tuple[object, ...]] = []
    fake_mx.eval = lambda *args: eval_calls.append(args)
    stacked_rows: list[list[object]] = []

    def fake_stack(rows):
        stacked_rows.append(list(rows))
        return FakeLogits()

    fake_mx.stack = fake_stack
    fake_mx.random = SimpleNamespace(categorical=lambda _logits: FakeTokens([10]))
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    vllm_metal_cfg.NATIVE_SAMPLED_ROWS = 0
    vllm_metal_cfg.NATIVE_OUTPUT_ROWS = 0

    fake_sampling_batch = SimpleNamespace(_SamplingResult=FakeSamplingResult)
    batch = SimpleNamespace(
        needs_logprobs=False,
        generators={},
        no_penalties=True,
        no_top_p=True,
        no_top_k=True,
        all_greedy=False,
        all_random=True,
        logitsprocs=SimpleNamespace(all=()),
        sampling_params_list=[
            SimpleNamespace(
                temperature=0.8,
                top_k=0,
                allowed_token_ids=None,
                bad_words_token_ids=None,
                extra_args={
                    "cfg_role": "cond",
                    "cfg_pair_id": "pair",
                    "cfg_scale": 2.0,
                },
            ),
            SimpleNamespace(
                temperature=0.8,
                top_k=0,
                allowed_token_ids=None,
                bad_words_token_ids=None,
                extra_args={"cfg_role": "uncond", "cfg_pair_id": "pair"},
            ),
        ],
    )

    logits = FakeLogits()
    result = vllm_metal_cfg._sample_native_mlx_if_supported(
        logits,
        batch,
        fake_sampling_batch,
    )

    assert result == FakeSamplingResult([10, 10])
    assert logits.full_cast is False
    assert logits.selected_rows == [0, 1]
    assert len(eval_calls) == 1
    assert isinstance(eval_calls[0][0], FakeTokens)
    assert len(stacked_rows) == 1
    assert len(stacked_rows[0]) == 1
    assert stacked_rows[0][0].name == "(row-1+(2.0*(row-0-row-1)))"
    assert vllm_metal_cfg.NATIVE_SAMPLED_ROWS == 1
    assert vllm_metal_cfg.NATIVE_OUTPUT_ROWS == 2


def test_vllm_metal_cfg_sampler_uses_native_mlx_for_unpaired_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeRow:
        def __init__(self, name: str) -> None:
            self.name = name

        def astype(self, _dtype):
            return self

    class FakeLogits:
        shape = (1, 64)

        def __init__(self) -> None:
            self.selected_rows: list[int] = []

        def __getitem__(self, index):
            self.selected_rows.append(index)
            return FakeRow(f"row-{index}")

        def __mul__(self, _other):
            return self

    class FakeTokens:
        def tolist(self) -> list[int]:
            return [42]

    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.float32 = object()
    fake_mx.eval = lambda *_args: None
    fake_mx.stack = lambda _rows: FakeLogits()
    fake_mx.random = SimpleNamespace(categorical=lambda _logits: FakeTokens())
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.delenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", raising=False)
    vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY.clear()
    vllm_metal_cfg.NATIVE_DETAIL_COUNT_BY_CATEGORY.clear()

    logits = FakeLogits()
    result = vllm_metal_cfg._sample_native_mlx_if_supported(
        logits,
        SimpleNamespace(
            needs_logprobs=False,
            generators={},
            no_penalties=True,
            no_top_p=True,
            no_top_k=True,
            all_greedy=False,
            all_random=True,
            logitsprocs=SimpleNamespace(all=()),
            sampling_params_list=[
                SimpleNamespace(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    allowed_token_ids=None,
                    bad_words_token_ids=None,
                    extra_args=None,
                )
            ],
        ),
        SimpleNamespace(_SamplingResult=FakeSamplingResult),
    )

    assert result == FakeSamplingResult([42])
    assert logits.selected_rows == [0]
    assert vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY == {}
    assert vllm_metal_cfg.NATIVE_DETAIL_COUNT_BY_CATEGORY == {}


def test_vllm_metal_cfg_sampler_rejects_top_p_for_fallback_correctness() -> None:
    result = vllm_metal_cfg._sample_native_mlx_if_supported(
        object(),
        SimpleNamespace(
            needs_logprobs=False,
            generators={},
            no_penalties=True,
            no_top_p=False,
            no_top_k=True,
            all_greedy=False,
            all_random=True,
            logitsprocs=SimpleNamespace(all=()),
            sampling_params_list=[
                SimpleNamespace(
                    temperature=1.0,
                    top_p=0.95,
                    top_k=0,
                    allowed_token_ids=None,
                    bad_words_token_ids=None,
                    extra_args=None,
                )
            ],
        ),
        SimpleNamespace(),
    )

    assert result is None


def test_vllm_metal_cfg_merges_text_mode_disallowed_token_ranges() -> None:
    sampling_params = SimpleNamespace(
        extra_args={
            "audex_disallow_token_ranges": [[3, 5], [4, 8], [99, 120]],
            "audex_disallow_token_ids": [0, 9, 50, -1, "bad"],
        }
    )

    ranges = vllm_metal_cfg._disallowed_token_ranges_from_sampling_params(
        sampling_params,
        vocab_size=100,
    )

    assert ranges == ((0, 0), (3, 9), (50, 50), (99, 99))


def test_vllm_metal_cfg_masks_disallowed_logits_with_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeRow:
        shape = (8,)
        dtype = "float32"

        def __getitem__(self, index):
            calls.append(("row_slice", index))
            return f"slice-{index.start}-{index.stop}"

    class FakeLogits:
        shape = (1, 8)

        def __getitem__(self, index):
            calls.append(("row", index))
            return FakeRow()

    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.float32 = object()
    fake_mx.full = (
        lambda shape, value, dtype: calls.append(("full", (shape, value, dtype)))
        or f"mask-{shape[0]}"
    )
    fake_mx.concatenate = (
        lambda parts, axis: calls.append(("concat", (tuple(parts), axis)))
        or "masked-row"
    )
    fake_mx.stack = lambda rows: calls.append(("stack", tuple(rows))) or "masked"
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    masked = vllm_metal_cfg._apply_disallowed_token_mask_mlx(
        FakeLogits(),
        [
            SimpleNamespace(
                extra_args={
                    "audex_disallow_token_ranges": [[2, 3]],
                    "audex_disallow_token_ids": [6],
                }
            )
        ],
    )

    assert masked == "masked"
    assert ("full", ((2,), -1.0e30, "float32")) in calls
    assert ("full", ((1,), -1.0e30, "float32")) in calls
    assert ("stack", ("masked-row",)) in calls


def test_vllm_metal_cfg_sampler_applies_uniform_top_k_natively() -> None:
    calls: list[tuple[str, object]] = []

    class FakeArray:
        shape = (1, 100)

        def __mul__(self, other):
            calls.append(("scale", other))
            return self

    class FakePartition:
        def __getitem__(self, index):
            calls.append(("top_indices_slice", index))
            return "top-indices"

    class FakeLocalTokens:
        def __getitem__(self, index):
            calls.append(("local_tokens_slice", index))
            return "local-token-column"

    class FakeMappedTokens:
        def __getitem__(self, index):
            calls.append(("mapped_tokens_slice", index))
            return "mapped-token-vector"

    fake_mx = SimpleNamespace(
        argpartition=lambda logits, kth, axis: calls.append(
            ("argpartition", (logits, kth, axis))
        )
        or FakePartition(),
        take_along_axis=lambda values, indices, axis: calls.append(
            ("take_along_axis", (values, indices, axis))
        )
        or ("top-logits" if indices == "top-indices" else FakeMappedTokens()),
        random=SimpleNamespace(
            categorical=lambda logits: calls.append(("categorical", logits))
            or FakeLocalTokens()
        ),
    )
    plan = vllm_metal_cfg.NativeSamplingPlan(
        sample_row_indices=(0,),
        output_slots=((0, 1),),
        temperatures=(0.1,),
        top_ks=(80,),
        allowed_token_windows=(None,),
    )

    token_vector = vllm_metal_cfg._sample_random_tokens_mlx(
        FakeArray(),
        plan,
        fake_mx,
    )

    assert token_vector == "mapped-token-vector"
    argpartition_calls = [call for call in calls if call[0] == "argpartition"]
    assert len(argpartition_calls) == 1
    assert argpartition_calls[0][1][1:] == (20, -1)
    assert ("categorical", "top-logits") in calls


def test_vllm_metal_cfg_sampler_restricts_tts_to_codec_window() -> None:
    calls: list[tuple[str, object]] = []

    class FakeArray:
        shape = (1, 205312)

        def __getitem__(self, index):
            calls.append(("logits_slice", index))
            return self

        def __mul__(self, other):
            calls.append(("scale", other))
            return self

    class FakeLocalToken:
        shape = (1,)

        def __add__(self, other):
            calls.append(("local_add", other))
            return "codec-token"

        def __eq__(self, other):
            calls.append(("local_eq", other))
            return "is-end-token"

    fake_mx = SimpleNamespace(
        int32=object(),
        concatenate=lambda values, axis: calls.append(("concatenate", (values, axis)))
        or FakeArray(),
        full=lambda shape, value, dtype: calls.append(("full", (shape, value)))
        or "end-token",
        where=lambda condition, x, y: calls.append(("where", (condition, x, y)))
        or "mapped-token",
        random=SimpleNamespace(
            categorical=lambda logits: calls.append(("categorical", logits))
            or FakeLocalToken()
        ),
    )
    plan = vllm_metal_cfg.NativeSamplingPlan(
        sample_row_indices=(0,),
        output_slots=((0, 1),),
        temperatures=(0.8,),
        top_ks=(0,),
        allowed_token_windows=((131077, 196612, 131076),),
    )

    token = vllm_metal_cfg._sample_random_tokens_mlx(FakeArray(), plan, fake_mx)

    assert token == "mapped-token"
    assert ("local_add", 131076) in calls
    assert ("full", ((1,), 131076)) in calls
    assert ("local_eq", 0) in calls
    assert ("where", ("is-end-token", "end-token", "codec-token")) in calls
    assert any(call[0] == "categorical" for call in calls)


def test_vllm_metal_cfg_tts_window_sampler_uses_params_without_sampling_batch() -> None:
    calls: list[tuple[str, object]] = []

    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeLogits:
        shape = (1, 65537)

        def __getitem__(self, index):
            calls.append(("logits_getitem", index))
            return self

        def astype(self, _dtype):
            calls.append(("astype", _dtype))
            return self

        def __mul__(self, other):
            calls.append(("scale", other))
            return self

    class FakeSampled:
        shape = (1,)

        def __add__(self, other):
            calls.append(("sampled_add", other))
            return "codec-token"

        def __eq__(self, other):
            calls.append(("sampled_eq", other))
            return "is-end-token"

    class FakeTokenIds:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def tolist(self):
            return self.values

    fake_mx = SimpleNamespace(
        float32="float32",
        int32=object(),
        eval=lambda tokens: calls.append(("eval", tokens)),
        full=lambda shape, value, dtype: calls.append(("full", (shape, value, dtype)))
        or "end-token",
        stack=lambda rows: calls.append(("stack", tuple(rows))) or rows[0],
        where=lambda condition, x, y: calls.append(("where", (condition, x, y)))
        or FakeTokenIds([131077]),
        random=SimpleNamespace(
            categorical=lambda logits: calls.append(("categorical", logits))
            or FakeSampled()
        ),
    )
    sampling_params = SimpleNamespace(
        temperature=0.8,
        top_p=1.0,
        top_k=0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        logprobs=None,
        allowed_token_ids=None,
        bad_words_token_ids=None,
        extra_args={
            "audex_tts_codec_min_id": 131077,
            "audex_tts_codec_max_id": 196612,
            "audex_tts_speechgen_end_id": 131076,
        },
    )

    result = vllm_metal_cfg._sample_tts_window_logits_from_params_if_supported(
        FakeLogits(),
        [sampling_params],
        SimpleNamespace(all=()),
        FakeSamplingResult,
        fake_mx,
    )

    assert result == FakeSamplingResult([131077])
    assert any(call[0] == "categorical" for call in calls)
    assert any(call[0] == "eval" for call in calls)


def test_vllm_metal_cfg_tts_window_sampler_expands_cfg_pair_tokens() -> None:
    calls: list[tuple[str, object]] = []

    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeRow:
        def __init__(self, name: str) -> None:
            self.name = name

        def astype(self, _dtype):
            calls.append(("astype", self.name))
            return self

        def __sub__(self, other):
            calls.append(("sub", (self.name, other.name)))
            return FakeRow(f"{self.name}-{other.name}")

        def __mul__(self, other):
            calls.append(("mul", (self.name, other)))
            return FakeRow(f"{self.name}*{other}")

        def __rmul__(self, other):
            calls.append(("rmul", (other, self.name)))
            return FakeRow(f"{other}*{self.name}")

        def __add__(self, other):
            calls.append(("add", (self.name, other.name)))
            return FakeRow(f"{self.name}+{other.name}")

    class FakeLogits:
        shape = (2, 65537)

        def __getitem__(self, index):
            calls.append(("logits_getitem", index))
            return FakeRow(f"row-{index}")

    class FakeSampled:
        shape = (1,)

        def __add__(self, other):
            calls.append(("sampled_add", other))
            return "codec-token"

        def __eq__(self, other):
            calls.append(("sampled_eq", other))
            return "is-end-token"

    class FakeTokenIds:
        def tolist(self):
            return [131088]

    fake_mx = SimpleNamespace(
        float32="float32",
        int32=object(),
        eval=lambda tokens: calls.append(("eval", tokens)),
        full=lambda shape, value, dtype: calls.append(("full", (shape, value, dtype)))
        or "end-token",
        stack=lambda rows: calls.append(("stack", tuple(row.name for row in rows)))
        or "sample-logits",
        where=lambda condition, x, y: calls.append(("where", (condition, x, y)))
        or FakeTokenIds(),
        random=SimpleNamespace(
            categorical=lambda logits: calls.append(("categorical", logits))
            or FakeSampled()
        ),
    )
    cond_params = SimpleNamespace(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        logprobs=None,
        allowed_token_ids=None,
        bad_words_token_ids=None,
        extra_args={
            "cfg_role": "cond",
            "cfg_pair_id": "pair-1",
            "cfg_scale": 3.0,
            "audex_tts_codec_min_id": 131077,
            "audex_tts_codec_max_id": 196612,
            "audex_tts_speechgen_end_id": 131076,
        },
    )
    uncond_params = SimpleNamespace(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        logprobs=None,
        allowed_token_ids=None,
        bad_words_token_ids=None,
        extra_args={
            "cfg_role": "uncond",
            "cfg_pair_id": "pair-1",
            "cfg_scale": 3.0,
            "audex_tts_codec_min_id": 131077,
            "audex_tts_codec_max_id": 196612,
            "audex_tts_speechgen_end_id": 131076,
        },
    )

    result = vllm_metal_cfg._sample_tts_window_logits_from_params_if_supported(
        FakeLogits(),
        [cond_params, uncond_params],
        SimpleNamespace(all=()),
        FakeSamplingResult,
        fake_mx,
    )

    assert result == FakeSamplingResult([131088, 131088])
    assert ("sub", ("row-0", "row-1")) in calls
    assert any(call[0] == "stack" for call in calls)
    assert ("categorical", "sample-logits") in calls


def test_vllm_metal_cfg_tts_window_sampler_preserves_zero_temperature_greedy() -> None:
    calls: list[tuple[str, object]] = []

    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeLogits:
        shape = (1, 65537)

        def __getitem__(self, index):
            calls.append(("logits_getitem", index))
            return self

        def astype(self, _dtype):
            calls.append(("astype", _dtype))
            return self

    class FakeSampled:
        shape = (1,)

        def __add__(self, other):
            calls.append(("sampled_add", other))
            return "codec-token"

        def __eq__(self, other):
            calls.append(("sampled_eq", other))
            return "is-end-token"

    class FakeTokenIds:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def tolist(self):
            return self.values

    fake_mx = SimpleNamespace(
        float32="float32",
        int32=object(),
        argmax=lambda logits, axis: calls.append(("argmax", (logits, axis)))
        or FakeSampled(),
        eval=lambda tokens: calls.append(("eval", tokens)),
        full=lambda shape, value, dtype: calls.append(("full", (shape, value, dtype)))
        or "end-token",
        stack=lambda rows: calls.append(("stack", tuple(rows))) or rows[0],
        where=lambda condition, x, y: calls.append(("where", (condition, x, y)))
        or FakeTokenIds([131077]),
        random=SimpleNamespace(
            categorical=lambda _logits: pytest.fail("greedy path must not sample")
        ),
    )
    sampling_params = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        logprobs=None,
        allowed_token_ids=None,
        bad_words_token_ids=None,
        extra_args={
            "audex_tts_codec_min_id": 131077,
            "audex_tts_codec_max_id": 196612,
            "audex_tts_speechgen_end_id": 131076,
        },
    )

    result = vllm_metal_cfg._sample_tts_window_logits_from_params_if_supported(
        FakeLogits(),
        [sampling_params],
        SimpleNamespace(all=()),
        FakeSamplingResult,
        fake_mx,
    )

    assert result == FakeSamplingResult([131077])
    assert any(call[0] == "argmax" for call in calls)


def test_vllm_metal_cfg_can_skip_logits_eval_for_no_cfg_tts_window() -> None:
    state = SimpleNamespace(
        target_hidden_states=None,
        pooling_hidden_states=None,
        prefill_reqs=[],
        decode_reqs=[
            (
                "tts-1",
                SimpleNamespace(
                    sampling_params=SimpleNamespace(
                        extra_args={
                            "audex_tts_codec_min_id": 131077,
                            "audex_tts_codec_max_id": 196612,
                            "audex_tts_speechgen_end_id": 131076,
                        }
                    )
                ),
            )
        ],
    )

    assert vllm_metal_cfg._can_skip_paged_logits_eval(state, None) is True


def test_vllm_metal_cfg_builds_tts_window_rows_before_float32_cast() -> None:
    calls: list[tuple[str, object]] = []

    class FakeSlice:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeWindowRow:
        def astype(self, _dtype):
            calls.append(("astype-window", None))
            return self

    class FakeRow:
        def __getitem__(self, index):
            calls.append(("slice", index))
            return FakeSlice(str(index))

        def astype(self, _dtype):
            calls.append(("astype-full-row", None))
            return self

    class FakeLogits:
        def __getitem__(self, index):
            calls.append(("row", index))
            return FakeRow()

    fake_mx = SimpleNamespace(
        float32=object(),
        concatenate=lambda parts, axis=-1: calls.append(
            ("concatenate", tuple(part.name for part in parts), axis)
        )
        or FakeWindowRow(),
        stack=lambda rows: tuple(rows),
    )
    plan = vllm_metal_cfg.NativeSamplingPlan(
        sample_row_indices=(0,),
        output_slots=((0,),),
        temperatures=(0.8,),
        top_ks=(0,),
        allowed_token_windows=((131077, 196612, 131076),),
    )

    rows = vllm_metal_cfg._build_native_sample_logits(
        FakeLogits(),
        [SimpleNamespace(extra_args=None)],
        plan,
        fake_mx,
        allowed_window=(131077, 196612, 131076),
    )

    assert rows
    assert ("row", 0) in calls
    assert ("slice", slice(131076, 131077, None)) in calls
    assert ("slice", slice(131077, 196613, None)) in calls
    assert ("astype-window", None) in calls
    assert ("astype-full-row", None) not in calls


def test_vllm_metal_cfg_can_skip_logits_eval_for_no_cfg_tts_hint() -> None:
    state = SimpleNamespace(
        target_hidden_states=None,
        pooling_hidden_states=None,
        prefill_reqs=[],
        decode_reqs=[
            (
                "tts-1",
                SimpleNamespace(
                    sampling_params=SimpleNamespace(
                        extra_args={"audex_tts_skip_paged_logits_eval": True}
                    )
                ),
            )
        ],
    )

    assert vllm_metal_cfg._can_skip_paged_logits_eval(state, None) is True


def test_vllm_metal_cfg_does_not_skip_logits_eval_for_plain_decode() -> None:
    state = SimpleNamespace(
        target_hidden_states=None,
        pooling_hidden_states=None,
        prefill_reqs=[],
        decode_reqs=[
            (
                "text-1",
                SimpleNamespace(sampling_params=SimpleNamespace(extra_args=None)),
            )
        ],
    )

    assert vllm_metal_cfg._can_skip_paged_logits_eval(state, None) is False


def test_vllm_metal_cfg_sampler_can_materialize_decode_logits_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeRow:
        def __init__(self, name: str) -> None:
            self.name = name

        def astype(self, _dtype):
            return self

        def __add__(self, other):
            return FakeRow(f"({self.name}+{other.name})")

        def __sub__(self, other):
            return FakeRow(f"({self.name}-{other.name})")

        def __mul__(self, other):
            return FakeRow(f"({self.name}*{other})")

        def __rmul__(self, other):
            return FakeRow(f"({other}*{self.name})")

    class FakeLogits:
        def __init__(self, name: str) -> None:
            self.name = name
            self.shape = (2, 205312)

        def __getitem__(self, index):
            return FakeRow(f"row-{index}")

        def __mul__(self, _other):
            return self

    class FakeTokens:
        def tolist(self) -> list[int]:
            return [10]

    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.float32 = object()
    eval_calls: list[tuple[object, ...]] = []
    fake_mx.eval = lambda *args: eval_calls.append(args)
    fake_mx.stack = lambda _rows: FakeLogits("stacked")
    fake_mx.random = SimpleNamespace(categorical=lambda _logits: FakeTokens())
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setenv("AUDEX_VLLM_MATERIALIZE_DECODE_LOGITS", "1")
    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")
    vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY.clear()
    vllm_metal_cfg.NATIVE_DETAIL_COUNT_BY_CATEGORY.clear()

    logits = FakeLogits("decode")
    result = vllm_metal_cfg._sample_native_mlx_if_supported(
        logits,
        SimpleNamespace(
            needs_logprobs=False,
            generators={},
            no_penalties=True,
            no_top_p=True,
            no_top_k=True,
            all_greedy=False,
            all_random=True,
            logitsprocs=SimpleNamespace(all=()),
            sampling_params_list=[
                SimpleNamespace(
                    temperature=0.8,
                    top_k=0,
                    allowed_token_ids=None,
                    bad_words_token_ids=None,
                    extra_args={
                        "cfg_role": "cond",
                        "cfg_pair_id": "pair",
                        "cfg_scale": 2.0,
                    },
                ),
                SimpleNamespace(
                    temperature=0.8,
                    top_k=0,
                    allowed_token_ids=None,
                    bad_words_token_ids=None,
                    extra_args={"cfg_role": "uncond", "cfg_pair_id": "pair"},
                ),
            ],
        ),
        SimpleNamespace(_SamplingResult=FakeSamplingResult),
    )

    assert result == FakeSamplingResult([10, 10])
    assert eval_calls[0] == (logits,)
    assert isinstance(eval_calls[1][0], FakeTokens)
    assert "materialize_decode_logits" in (
        vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY
    )


def test_vllm_metal_cfg_sampler_can_async_eval_sample_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FakeSamplingResult:
        token_ids: list[int]
        logprobs: object | None = None

    class FakeRow:
        def __init__(self, name: str) -> None:
            self.name = name

        def astype(self, _dtype):
            return self

    class FakeLogits:
        shape = (1, 205312)

        def __getitem__(self, index):
            return FakeRow(f"row-{index}")

    class FakeSampleLogits:
        shape = (1, 205312)

    class FakeTokens:
        def tolist(self) -> list[int]:
            return [42]

    sample_logits = FakeSampleLogits()
    async_calls: list[tuple[object, ...]] = []
    fake_mlx_package = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.float32 = object()
    fake_mx.eval = lambda *_args: None
    fake_mx.async_eval = lambda *args: async_calls.append(args)
    fake_mx.stack = lambda _rows: sample_logits
    fake_mx.random = SimpleNamespace(categorical=lambda _logits: FakeTokens())
    fake_mlx_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setenv("AUDEX_VLLM_NONPAGED_ASYNC_EVAL_TARGET", "sample_logits")
    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")
    vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY.clear()
    vllm_metal_cfg.NATIVE_DETAIL_COUNT_BY_CATEGORY.clear()

    result = vllm_metal_cfg._sample_native_mlx_if_supported(
        FakeLogits(),
        SimpleNamespace(
            needs_logprobs=False,
            generators={},
            no_penalties=True,
            no_top_p=True,
            no_top_k=True,
            all_greedy=False,
            all_random=True,
            logitsprocs=SimpleNamespace(all=()),
            sampling_params_list=[
                SimpleNamespace(
                    temperature=1.0,
                    top_k=0,
                    allowed_token_ids=None,
                    bad_words_token_ids=None,
                    extra_args=None,
                )
            ],
        ),
        SimpleNamespace(_SamplingResult=FakeSamplingResult),
    )

    assert result == FakeSamplingResult([42])
    assert async_calls == [(sample_logits,)]
    assert (
        "native_sample_logits_async_submit"
        in vllm_metal_cfg.NATIVE_DETAIL_SECONDS_BY_CATEGORY
    )


def test_vllm_metal_cfg_sampling_plan_keeps_unpaired_rows() -> None:
    plan = vllm_metal_cfg._build_cfg_pair_sampling_plan(
        [
            SimpleNamespace(
                temperature=0.8,
                top_k=80,
                extra_args={"cfg_role": "cond", "cfg_pair_id": "pair"},
            ),
            SimpleNamespace(
                temperature=0.8,
                top_k=80,
                extra_args={"cfg_role": "uncond", "cfg_pair_id": "pair"},
            ),
            SimpleNamespace(temperature=1.0, top_k=0, extra_args=None),
        ]
    )

    assert plan.sample_row_indices == (0, 2)
    assert plan.output_slots == ((0, 1), (2,))
    assert plan.temperatures == (0.8, 1.0)
    assert plan.top_ks == (80, 0)
    assert vllm_metal_cfg._expand_sampled_token_ids([10, 30], plan.output_slots) == [
        10,
        10,
        30,
    ]


def test_vllm_metal_cfg_sampler_rejects_token_constraints() -> None:
    batch = SimpleNamespace(
        needs_logprobs=False,
        generators={},
        no_penalties=True,
        no_top_p=True,
        no_top_k=True,
        all_greedy=False,
        all_random=True,
        logitsprocs=SimpleNamespace(all=()),
        sampling_params_list=[
            SimpleNamespace(
                allowed_token_ids=[101, 102],
                bad_words_token_ids=None,
                extra_args={"cfg_role": "cond", "cfg_pair_id": "pair"},
            ),
            SimpleNamespace(
                allowed_token_ids=[101, 102],
                bad_words_token_ids=None,
                extra_args={"cfg_role": "uncond", "cfg_pair_id": "pair"},
            ),
        ],
    )

    result = vllm_metal_cfg._sample_native_mlx_if_supported(
        object(),
        batch,
        SimpleNamespace(),
    )

    assert result is None


def test_vllm_metal_cfg_native_sampler_allows_only_inert_builtin_processors() -> None:
    class MinPLogitsProcessor:
        pass

    class MinTokensLogitsProcessor:
        pass

    class LogitBiasLogitsProcessor:
        pass

    MinPLogitsProcessor.__module__ = "vllm.v1.sample.logits_processor.builtin"
    MinTokensLogitsProcessor.__module__ = "vllm.v1.sample.logits_processor.builtin"
    LogitBiasLogitsProcessor.__module__ = "vllm.v1.sample.logits_processor.builtin"

    batch = SimpleNamespace(
        logitsprocs=SimpleNamespace(
            all=(
                MinPLogitsProcessor(),
                MinTokensLogitsProcessor(),
                LogitBiasLogitsProcessor(),
            )
        ),
        sampling_params_list=[
            SimpleNamespace(min_p=0.0, min_tokens=0, logit_bias=None),
            SimpleNamespace(min_p=0.0, min_tokens=0, logit_bias={}),
        ],
    )

    assert vllm_metal_cfg._unsupported_native_logits_processors(batch) == ()

    batch.sampling_params_list[0].min_tokens = 1
    assert vllm_metal_cfg._unsupported_native_logits_processors(batch) == (
        "vllm.v1.sample.logits_processor.builtin.MinTokensLogitsProcessor",
    )
