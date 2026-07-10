from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from audex_mac import vllm_diagnostics
from audex_mac.vllm_diagnostics import (
    _assess_sts_timing,
    _diagnostic_verdict,
    _interpret_expected_cpu_facade,
    _parse_json_from_subprocess_stdout,
    _parse_vllm_metal_timing,
    _probe_apple_silicon_topology,
    _probe_audex_cfg_subprocess,
    _probe_audex_processor_subprocess,
    _probe_vllm_generation,
    _probe_vllm_sts_default_runtime,
    _probe_vllm_tts_batch_runtime,
    _scan_vllm_metal_sources,
    _selected_env,
    _speech_runtime_report,
    _sts_smoke_evidence_failures,
    _sts_smoke_speech_max_tokens,
    _sts_smoke_timeout_seconds,
    _subprocess_progress_text,
    run_vllm_metal_diagnostics,
)
from audex_mac.vllm_streaming import VllmStreamingSupport

pytestmark = pytest.mark.fast


def test_selected_env_records_required_metal_variables() -> None:
    env = {
        "VLLM_METAL_USE_MLX": "1",
        "VLLM_MLX_DEVICE": "gpu",
        "VLLM_METAL_USE_PAGED_ATTENTION": "0",
        "AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL": "1",
        "AUDEX_VLLM_CFG_MAX_NUM_SEQS": "2",
        "AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS": "12288",
        "AUDEX_VLLM_MATERIALIZE_DECODE_LOGITS": "1",
        "UNRELATED": "ignored",
    }

    selected = _selected_env(env)

    assert selected["VLLM_METAL_USE_MLX"] == "1"
    assert selected["VLLM_MLX_DEVICE"] == "gpu"
    assert selected["VLLM_METAL_USE_PAGED_ATTENTION"] == "0"
    assert selected["AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL"] == "1"
    assert selected["AUDEX_VLLM_CFG_MAX_NUM_SEQS"] == "2"
    assert selected["AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS"] == "12288"
    assert selected["AUDEX_VLLM_MATERIALIZE_DECODE_LOGITS"] == "1"
    assert "UNRELATED" not in selected


def test_probe_apple_silicon_topology_records_sysctl_perflevels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "hw.optional.arm64": "1",
        "hw.memsize": "137438953472",
        "hw.pagesize": "16384",
        "hw.cachelinesize": "128",
        "hw.physicalcpu_max": "16",
        "hw.logicalcpu_max": "16",
        "hw.nperflevels": "2",
        "sysctl.proc_translated": "0",
        "hw.perflevel0.physicalcpu_max": "12",
        "hw.perflevel0.logicalcpu_max": "12",
        "hw.perflevel0.l1dcachesize": "131072",
        "hw.perflevel0.l1icachesize": "196608",
        "hw.perflevel0.l2cachesize": "50331648",
        "hw.perflevel0.cpusperl2": "6",
        "hw.perflevel1.physicalcpu_max": "4",
        "hw.perflevel1.logicalcpu_max": "4",
        "hw.perflevel1.l1dcachesize": "65536",
        "hw.perflevel1.l1icachesize": "131072",
        "hw.perflevel1.l2cachesize": "4194304",
        "hw.perflevel1.cpusperl2": "4",
    }

    def fake_run(command, **_kwargs):
        name = command[-1]
        if name not in values:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr=f"sysctl: unknown oid '{name}'",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{values[name]}\n",
            stderr="",
        )

    monkeypatch.setattr(vllm_diagnostics.sys, "platform", "darwin")
    monkeypatch.setattr(vllm_diagnostics.subprocess, "run", fake_run)

    topology = _probe_apple_silicon_topology()

    assert topology["enabled"] is True
    assert topology["sysctl"]["hw.pagesize"] == 16384
    assert topology["sysctl"]["hw.cachelinesize"] == 128
    assert topology["sysctl"]["sysctl.proc_translated"] == 0
    assert topology["perflevels"][0]["physicalcpu_max"] == 12
    assert topology["perflevels"][1]["l2cachesize"] == 4194304
    assert "cpusperl3" in topology["perflevels"][0]["errors"]


def test_sts_smoke_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDEX_STS_SMOKE_TIMEOUT_SECONDS", "5")
    assert _sts_smoke_timeout_seconds() == 30

    monkeypatch.setenv("AUDEX_STS_SMOKE_TIMEOUT_SECONDS", "120")
    assert _sts_smoke_timeout_seconds() == 120

    monkeypatch.setenv("AUDEX_STS_SMOKE_TIMEOUT_SECONDS", "nope")
    assert _sts_smoke_timeout_seconds() == 600


def test_sts_playback_smoke_uses_bounded_speech_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUDEX_STS_SMOKE_SPEECH_MAX_TOKENS", raising=False)

    assert _sts_smoke_speech_max_tokens(play_audio=False) is None
    assert _sts_smoke_speech_max_tokens(play_audio=True) == 256

    monkeypatch.setenv("AUDEX_STS_SMOKE_SPEECH_MAX_TOKENS", "96")
    assert _sts_smoke_speech_max_tokens(play_audio=False) == 96
    assert _sts_smoke_speech_max_tokens(play_audio=True) == 96

    monkeypatch.setenv("AUDEX_STS_SMOKE_SPEECH_MAX_TOKENS", "nope")
    assert _sts_smoke_speech_max_tokens(play_audio=False) == 256
    assert _sts_smoke_speech_max_tokens(play_audio=True) == 256


def test_diagnostics_uses_explicit_sts_speech_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: dict[str, object] = {}
    text_path = tmp_path / "text"
    speech_path = tmp_path / "checkpoint_folder_full"

    monkeypatch.setattr(
        vllm_diagnostics,
        "enforce_metal_env",
        lambda: SimpleNamespace(env={}, ready=True),
    )
    monkeypatch.setattr(
        vllm_diagnostics,
        "preflight_text_runtime",
        lambda _model, apply_patches=False: SimpleNamespace(
            model_path=text_path,
            ready=True,
            missing_items=(),
            dependency_checks=(),
        ),
    )
    monkeypatch.setattr(
        vllm_diagnostics,
        "preflight_audio_runtime",
        lambda _model: SimpleNamespace(model_path=speech_path),
    )
    monkeypatch.setattr(
        vllm_diagnostics,
        "_default_vllm_metal_source_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        vllm_diagnostics,
        "inspect_vllm_streaming_support",
        lambda: VllmStreamingSupport(
            sync_llm_generate_streams=False,
            async_engine_available=True,
            async_generate_is_asyncgen=True,
            request_output_kind_available=True,
            cumulative_output_kind_available=True,
            final_only_output_kind_available=True,
        ),
    )
    monkeypatch.setattr(vllm_diagnostics, "_probe_spawned_worker", lambda: {})
    monkeypatch.setattr(
        vllm_diagnostics,
        "_probe_platform_resolution_subprocess",
        lambda: {},
    )
    monkeypatch.setattr(vllm_diagnostics, "_probe_vllm_metal_modules", lambda: {})
    monkeypatch.setattr(vllm_diagnostics, "_probe_audex_patches_subprocess", lambda: {})
    monkeypatch.setattr(
        vllm_diagnostics, "_probe_audex_processor_subprocess", lambda: {}
    )
    monkeypatch.setattr(
        vllm_diagnostics,
        "_probe_audex_cfg_subprocess",
        lambda _model_path: {},
    )
    monkeypatch.setattr(
        vllm_diagnostics,
        "_speech_runtime_report",
        lambda _preflight: {"enabled": True, "ready": True},
    )
    monkeypatch.setattr(
        vllm_diagnostics,
        "_probe_model_adapter",
        lambda _model, _model_path: {},
    )
    monkeypatch.setattr(vllm_diagnostics, "_scan_vllm_metal_sources", lambda _root: [])
    monkeypatch.setattr(vllm_diagnostics, "_interpret_expected_cpu_facade", lambda: {})
    monkeypatch.setattr(
        vllm_diagnostics,
        "_diagnostic_verdict",
        lambda _report, **_kwargs: {"ready": True, "failures": []},
    )

    def fake_sts_probe(
        _model,
        *,
        audio_fixture,
        play_audio,
        speech_max_tokens,
        native_sampling_debug,
    ):
        called["audio_fixture"] = audio_fixture
        called["play_audio"] = play_audio
        called["speech_max_tokens"] = speech_max_tokens
        called["native_sampling_debug"] = native_sampling_debug
        return {"enabled": True, "ready": True, "speech_max_tokens": speech_max_tokens}

    monkeypatch.setattr(
        vllm_diagnostics,
        "_probe_vllm_sts_default_runtime",
        fake_sts_probe,
    )

    result = run_vllm_metal_diagnostics(
        SimpleNamespace(repo_id="nvidia/Nemotron-Labs-Audex-2B"),
        run_sts_smoke=True,
        sts_play_audio=False,
        sts_speech_max_tokens=192,
        output_dir=tmp_path,
    )

    assert result.report["sts_probe"]["speech_max_tokens"] == 192
    assert called == {
        "audio_fixture": None,
        "play_audio": False,
        "speech_max_tokens": 192,
        "native_sampling_debug": False,
    }


def test_subprocess_json_parser_accepts_progress_before_final_json() -> None:
    parsed = _parse_json_from_subprocess_stdout(
        "Audex STS: transcribing projected input speech with vLLM Metal...\n"
        '{"enabled": true, "ready": false, "error": "EngineGenerateError"}\n'
    )

    assert parsed == {
        "enabled": True,
        "ready": False,
        "error": "EngineGenerateError",
    }


def test_subprocess_progress_text_keeps_non_json_lines() -> None:
    progress = _subprocess_progress_text(
        "Audex STS: transcribing projected input speech with vLLM Metal...\n"
        "Audex STS: transcript: hello\n"
        '{"enabled": true, "ready": false, "error": "EngineDeadError"}\n'
    )

    assert progress == (
        "Audex STS: transcribing projected input speech with vLLM Metal...\n"
        "Audex STS: transcript: hello"
    )


def test_parse_vllm_metal_timing_extracts_paged_sample_and_eval_split() -> None:
    parsed = _parse_vllm_metal_timing(
        "\n".join(
            [
                "INFO unrelated",
                "Audex vLLM Metal: native MLX sampling fast path used 50 time(s)",
                "Audex vLLM Metal: paged sample timing "
                "count=200 avg_ms=182.9 last_ms=19.7 "
                "decode_reqs=2 prefill_reqs=0 decode_tokens=2 "
                "native_sample_ms=10048.3 "
                "native_sampled_rows=200 native_output_rows=400 "
                "skipped_logits_eval=137 "
                "native_detail_ms=build_sample_logits:80.0/200,sample_eval:9900.0/200,tolist:2.0/200 "
                "mx_eval_ms=logits:27221.5/200,sample_tokens:99.8/200 "
                "mx_eval_shapes=logits:1x2x205312x200,sample_tokens:1x200",
                "Audex vLLM Metal: nonpaged decode timing "
                "count=100 avg_ms=33.3 last_ms=31.2 "
                "decode_reqs=1 cached_reqs=1 batched=0 "
                "native_sample_ms=850.0 "
                "cfg_cond_reqs=2 cfg_uncond_reqs=2 cfg_complete_pairs=2 "
                "native_sampled_rows=100 native_output_rows=100 "
                "tts_window_decode_count=197 "
                "tts_window_weight_cache_hits=98 "
                "tts_window_weight_cache_misses=1 "
                "nonpaged_persistent_cache_hits=47 "
                "nonpaged_persistent_cache_misses=1 "
                "nonpaged_persistent_cache_flushes=0 "
                "native_detail_ms=tts_window_batch_forward:456.3/100,"
                "nonpaged_kv_cache_merge:111.0/100,"
                "nonpaged_kv_cache_extract:222.0/200,"
                "tts_window_batch_forward_eval:7278.4/101,"
                "tts_window_batch_project:3.9/100,"
                "tts_window_batch_project_eval:1223.0/101,"
                "tts_window_batch_sample:27304.5/100",
            ]
        )
    )

    assert parsed["latest_native_sampling_fast_path_count"] == 50
    latest = parsed["latest_paged_sample"]
    assert latest["count"] == 200
    assert latest["avg_ms"] == 182.9
    assert latest["decode_reqs"] == 2
    assert latest["native_sample_ms"] == 10048.3
    assert latest["native_sampled_rows"] == 200
    assert latest["native_output_rows"] == 400
    assert latest["skipped_logits_eval"] == 137
    assert latest["mx_eval_ms"]["logits"] == {
        "milliseconds": 27221.5,
        "count": 200,
    }
    assert latest["mx_eval_ms"]["sample_tokens"] == {
        "milliseconds": 99.8,
        "count": 200,
    }
    assert latest["native_detail_ms"]["sample_eval"] == {
        "milliseconds": 9900.0,
        "count": 200,
    }
    assert latest["mx_eval_shapes"] == {
        "logits": [{"shape": [1, 2, 205312], "count": 200}],
        "sample_tokens": [{"shape": [1], "count": 200}],
    }
    latest_non_paged = parsed["latest_non_paged_decode"]
    assert latest_non_paged["count"] == 100
    assert latest_non_paged["avg_ms"] == 33.3
    assert latest_non_paged["decode_reqs"] == 1
    assert latest_non_paged["cached_reqs"] == 1
    assert latest_non_paged["batched"] is False
    assert latest_non_paged["native_sample_ms"] == 850.0
    assert latest_non_paged["cfg_cond_reqs"] == 2
    assert latest_non_paged["cfg_uncond_reqs"] == 2
    assert latest_non_paged["cfg_complete_pairs"] == 2
    assert latest_non_paged["tts_window_decode_count"] == 197
    assert latest_non_paged["tts_window_weight_cache_hits"] == 98
    assert latest_non_paged["tts_window_weight_cache_misses"] == 1
    assert latest_non_paged["nonpaged_persistent_cache_hits"] == 47
    assert latest_non_paged["nonpaged_persistent_cache_misses"] == 1
    assert latest_non_paged["nonpaged_persistent_cache_flushes"] == 0
    assert latest_non_paged["native_detail_ms"]["tts_window_batch_sample"] == {
        "milliseconds": 27304.5,
        "count": 100,
    }
    assert latest_non_paged["native_detail_ms"]["tts_window_batch_forward_eval"] == {
        "milliseconds": 7278.4,
        "count": 101,
    }
    assert latest_non_paged["native_detail_ms"]["nonpaged_kv_cache_merge"] == {
        "milliseconds": 111.0,
        "count": 100,
    }
    assert latest_non_paged["native_detail_ms"]["nonpaged_kv_cache_extract"] == {
        "milliseconds": 222.0,
        "count": 200,
    }
    assert latest_non_paged["native_detail_ms"]["tts_window_batch_project_eval"] == {
        "milliseconds": 1223.0,
        "count": 101,
    }


def test_parse_vllm_metal_timing_extracts_native_rejection_reasons() -> None:
    parsed = _parse_vllm_metal_timing(
        "Audex vLLM Metal: native MLX sampling fast path skipped: "
        "top-k filtering requested"
    )

    assert parsed == {
        "native_sampling_rejection_reasons": ["top-k filtering requested"]
    }


def test_assess_sts_timing_reports_realtime_ratio_and_logits_bottleneck() -> None:
    assessment = _assess_sts_timing(
        {
            "speech_streaming": {
                "generated_codec_frame_count": 256,
                "last_codec_frame_seconds": 78.107,
                "playback_diagnostics": {
                    "device_underflow_count": 43,
                    "queue_underrun_count": 43,
                    "queue_overrun_count": 0,
                },
            },
            "vllm_metal_timing": {
                "latest_paged_sample": {
                    "count": 200,
                    "avg_ms": 182.9,
                    "native_sample_ms": 10048.3,
                    "native_sampled_rows": 200,
                    "native_output_rows": 400,
                    "skipped_logits_eval": 137,
                    "native_detail_ms": {
                        "build_sample_logits": {"milliseconds": 80.0, "count": 200},
                        "sample_eval": {"milliseconds": 9900.0, "count": 200},
                        "tolist": {"milliseconds": 2.0, "count": 200},
                    },
                    "mx_eval_ms": {
                        "logits": {"milliseconds": 27221.5, "count": 200},
                        "sample_tokens": {"milliseconds": 99.8, "count": 200},
                    },
                    "mx_eval_shapes": {
                        "logits": [{"shape": [1, 2, 205312], "count": 200}],
                        "sample_tokens": [{"shape": [1], "count": 200}],
                    },
                }
            },
        }
    )

    assert assessment["codec_frames_per_second"] == 3.278
    assert assessment["audio_realtime_ratio"] == 0.066
    assert assessment["below_realtime"] is True
    assert assessment["playback_glitch_count"] == 86
    assert assessment["paged_sample_avg_ms"] == 182.9
    assert assessment["skipped_logits_eval"] == 137
    assert assessment["native_sample_ms_per_step"] == 50.241
    assert assessment["native_sampled_rows"] == 200
    assert assessment["native_sample_ms_per_sampled_row"] == 50.241
    assert assessment["native_output_rows"] == 400
    assert assessment["native_sampling_row_ratio"] == 0.5
    assert assessment["dominant_mx_eval_category"] == "logits"
    assert assessment["dominant_mx_eval_per_step_category"] == "logits"
    assert assessment["dominant_mx_eval_ms_per_step"] == 136.107
    assert assessment["dominant_native_detail_category"] == "sample_eval"
    assert assessment["dominant_native_detail_ms_per_step"] == 49.5
    assert assessment["mx_eval_ms_per_step_by_category"] == {
        "logits": 136.107,
        "sample_tokens": 0.499,
    }
    assert assessment["mx_eval_shapes_by_category"] == {
        "logits": [{"shape": [1, 2, 205312], "count": 200}],
        "sample_tokens": [{"shape": [1], "count": 200}],
    }
    assert assessment["likely_bottleneck"] == "logits_eval"


def test_assess_sts_timing_reports_native_sampling_bottleneck() -> None:
    assessment = _assess_sts_timing(
        {
            "speech_streaming": {
                "generated_codec_frame_count": 100,
                "last_codec_frame_seconds": 1.0,
            },
            "vllm_metal_timing": {
                "latest_paged_sample": {
                    "native_sample_ms": 1200.0,
                    "mx_eval_ms": {
                        "logits": {"milliseconds": 400.0, "count": 100},
                        "sample_tokens": {"milliseconds": 100.0, "count": 100},
                    },
                }
            },
        }
    )

    assert assessment["below_realtime"] is False
    assert assessment["likely_bottleneck"] == "native_sampling"


def test_assess_sts_timing_reports_sample_token_eval_per_step() -> None:
    assessment = _assess_sts_timing(
        {
            "speech_streaming": {
                "generated_codec_frame_count": 240,
                "last_codec_frame_seconds": 52.01,
            },
            "vllm_metal_timing": {
                "latest_paged_sample": {
                    "count": 200,
                    "avg_ms": 166.8,
                    "native_sample_ms": 11597.0,
                    "native_sampled_rows": 137,
                    "native_output_rows": 274,
                    "mx_eval_ms": {
                        "logits": {"milliseconds": 21500.1, "count": 328},
                        "sample_tokens": {"milliseconds": 11582.8, "count": 137},
                    },
                }
            },
        }
    )

    assert assessment["codec_frames_per_second"] == 4.614
    assert assessment["native_sample_ms_per_sampled_row"] == 84.65
    assert assessment["mx_eval_ms_per_step_by_category"] == {
        "logits": 65.549,
        "sample_tokens": 84.546,
    }
    assert assessment["dominant_mx_eval_per_step_category"] == "sample_tokens"
    assert assessment["likely_bottleneck"] == "pending_graph_eval_during_sampling"


def test_assess_sts_timing_reports_nonpaged_tts_window_sampling_bottleneck() -> None:
    assessment = _assess_sts_timing(
        {
            "speech_streaming": {
                "generated_codec_frame_count": 512,
                "last_codec_frame_seconds": 109.209,
            },
            "vllm_metal_timing": {
                "latest_non_paged_decode": {
                    "count": 397,
                    "avg_ms": 77.3,
                    "batched": True,
                    "native_sample_ms": 28102.0,
                    "tts_window_decode_count": 1438,
                    "tts_window_weight_cache_hits": 396,
                    "tts_window_weight_cache_misses": 1,
                    "nonpaged_persistent_cache_hits": 47,
                    "nonpaged_persistent_cache_misses": 1,
                    "nonpaged_persistent_cache_flushes": 0,
                    "native_detail_ms": {
                        "tts_window_batch_forward": {
                            "milliseconds": 456.3,
                            "count": 397,
                        },
                        "tts_window_batch_project": {
                            "milliseconds": 3.9,
                            "count": 397,
                        },
                        "tts_window_batch_sample": {
                            "milliseconds": 27304.5,
                            "count": 397,
                        },
                    },
                }
            },
        }
    )

    assert assessment["codec_frames_per_second"] == 4.688
    assert assessment["nonpaged_decode_avg_ms"] == 77.3
    assert assessment["nonpaged_native_sample_ms_per_step"] == 70.786
    assert assessment["tts_window_decode_count"] == 1438
    assert assessment["tts_window_weight_cache_hit_rate"] == 0.997
    assert assessment["nonpaged_persistent_cache_hit_rate"] == 0.979
    assert assessment["nonpaged_persistent_cache_flushes"] == 0
    assert assessment["dominant_nonpaged_native_detail_category"] == (
        "tts_window_batch_sample"
    )
    assert assessment["dominant_nonpaged_native_detail_ms_per_step"] == 68.777
    assert assessment["likely_bottleneck"] == (
        "pending_graph_eval_during_tts_window_sampling"
    )


def test_assess_sts_timing_reports_synced_tts_window_forward_eval_bottleneck() -> None:
    assessment = _assess_sts_timing(
        {
            "speech_streaming": {
                "generated_codec_frame_count": 724,
                "last_codec_frame_seconds": 94.85,
            },
            "vllm_metal_timing": {
                "latest_non_paged_decode": {
                    "count": 200,
                    "avg_ms": 61.5,
                    "batched": True,
                    "native_sample_ms": 1897.7,
                    "tts_window_decode_count": 398,
                    "tts_window_weight_cache_hits": 100,
                    "tts_window_weight_cache_misses": 1,
                    "native_detail_ms": {
                        "tts_window_batch_forward": {
                            "milliseconds": 113.8,
                            "count": 101,
                        },
                        "tts_window_batch_forward_eval": {
                            "milliseconds": 7278.4,
                            "count": 101,
                        },
                        "tts_window_batch_project": {
                            "milliseconds": 3.6,
                            "count": 101,
                        },
                        "tts_window_batch_project_eval": {
                            "milliseconds": 1223.0,
                            "count": 101,
                        },
                        "tts_window_batch_sample": {
                            "milliseconds": 1347.8,
                            "count": 101,
                        },
                    },
                }
            },
        }
    )

    assert assessment["codec_frames_per_second"] == 7.633
    assert assessment["dominant_nonpaged_native_detail_category"] == (
        "tts_window_batch_forward_eval"
    )
    assert assessment["dominant_nonpaged_native_detail_ms_per_step"] == 72.063
    assert (
        assessment["nonpaged_native_detail_ms_per_step_by_category"][
            "tts_window_batch_sample"
        ]
        == 13.345
    )
    assert assessment["likely_bottleneck"] == (
        "model_forward_eval_during_tts_window_decode"
    )


def test_assess_sts_timing_reports_nonpaged_kv_cache_copy_bottleneck() -> None:
    assessment = _assess_sts_timing(
        {
            "speech_streaming": {
                "generated_codec_frame_count": 256,
                "last_codec_frame_seconds": 20.0,
            },
            "vllm_metal_timing": {
                "latest_non_paged_decode": {
                    "count": 100,
                    "avg_ms": 40.0,
                    "batched": True,
                    "native_sample_ms": 300.0,
                    "native_detail_ms": {
                        "nonpaged_kv_cache_merge": {
                            "milliseconds": 5200.0,
                            "count": 100,
                        },
                        "nonpaged_kv_cache_extract": {
                            "milliseconds": 1400.0,
                            "count": 200,
                        },
                        "tts_window_batch_sample": {
                            "milliseconds": 800.0,
                            "count": 100,
                        },
                    },
                }
            },
        }
    )

    assert assessment["dominant_nonpaged_native_detail_category"] == (
        "nonpaged_kv_cache_merge"
    )
    assert assessment["dominant_nonpaged_native_detail_ms_per_step"] == 52.0
    assert assessment["likely_bottleneck"] == "nonpaged_kv_cache_copy"


def test_assess_sts_timing_reports_nonpaged_sampling_eval_bottleneck() -> None:
    assessment = _assess_sts_timing(
        {
            "speech_streaming": {
                "generated_codec_frame_count": 128,
                "last_codec_frame_seconds": 7.661,
            },
            "vllm_metal_timing": {
                "latest_non_paged_decode": {
                    "count": 50,
                    "avg_ms": 114.4,
                    "batched": True,
                    "native_sample_ms": 2643.1,
                    "native_detail_ms": {
                        "nonpaged_kv_cache_extract": {
                            "milliseconds": 3666.6,
                            "count": 192,
                        },
                        "nonpaged_kv_cache_merge": {
                            "milliseconds": 15.5,
                            "count": 48,
                        },
                        "sample_eval": {
                            "milliseconds": 2638.6,
                            "count": 52,
                        },
                    },
                }
            },
        }
    )

    assert assessment["dominant_nonpaged_native_detail_category"] == "sample_eval"
    assert assessment["dominant_nonpaged_native_detail_ms_per_step"] == 50.742
    assert assessment["likely_bottleneck"] == "pending_graph_eval_during_sampling"


def test_assess_sts_timing_reports_nonpaged_async_submit_bottleneck() -> None:
    assessment = _assess_sts_timing(
        {
            "speech_streaming": {
                "generated_codec_frame_count": 128,
                "last_codec_frame_seconds": 6.695,
            },
            "vllm_metal_timing": {
                "latest_non_paged_decode": {
                    "count": 50,
                    "avg_ms": 95.2,
                    "batched": True,
                    "native_sample_ms": 1123.5,
                    "native_detail_ms": {
                        "nonpaged_decode_logits_async_submit": {
                            "milliseconds": 1492.7,
                            "count": 48,
                        },
                        "nonpaged_kv_cache_extract": {
                            "milliseconds": 2726.9,
                            "count": 192,
                        },
                        "sample_eval": {
                            "milliseconds": 1117.7,
                            "count": 52,
                        },
                    },
                }
            },
        }
    )

    assert assessment["dominant_nonpaged_native_detail_category"] == (
        "nonpaged_decode_logits_async_submit"
    )
    assert assessment["dominant_nonpaged_native_detail_ms_per_step"] == 31.098
    assert assessment["likely_bottleneck"] == "nonpaged_async_graph_submit"


def test_cpu_facade_interpretation_distinguishes_vllm_from_mlx() -> None:
    interpretation = _interpret_expected_cpu_facade()

    assert interpretation["vllm_device_type_cpu_can_be_expected"] is True
    assert "VLLM_MLX_DEVICE is not gpu" in interpretation["cpu_fallback_indicators"]
    assert interpretation["required_env"]["VLLM_MLX_DEVICE"] == "gpu"


def test_source_scan_captures_vllm_metal_device_paths(tmp_path: Path) -> None:
    source = tmp_path / "vllm_metal" / "platform.py"
    source.parent.mkdir()
    source.write_text(
        "\n".join(
            [
                'device_type: str = "cpu"',
                "device_type = mx.DeviceType.cpu",
                "if config.use_paged_attention:",
            ]
        ),
        encoding="utf-8",
    )

    matches = _scan_vllm_metal_sources(tmp_path)

    assert [match.pattern for match in matches] == [
        'device_type: str = "cpu"',
        "DeviceType.cpu",
        "use_paged_attention",
    ]


def test_speech_runtime_report_records_full_checkpoint_paths(tmp_path: Path) -> None:
    preflight = SimpleNamespace(
        ready=True,
        model_path=tmp_path / "checkpoint_folder_full",
        decoder_path=tmp_path / "audex_causal_speech_decoder",
        missing_items=(),
    )

    report = _speech_runtime_report(preflight)

    assert report == {
        "enabled": True,
        "ready": True,
        "model_path": str(tmp_path / "checkpoint_folder_full"),
        "decoder_path": str(tmp_path / "audex_causal_speech_decoder"),
        "missing_items": [],
    }


def test_generation_probe_returns_missing_preflight_without_subprocess() -> None:
    preflight = SimpleNamespace(
        ready=False,
        model_path=None,
        missing_items=("missing model",),
    )

    result = _probe_vllm_generation(preflight, max_tokens=None)

    assert result == {
        "enabled": True,
        "ready": False,
        "error": "text runtime preflight is not ready",
        "missing_items": ["missing model"],
    }


def test_generation_probe_runs_vllm_in_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = SimpleNamespace(
        system="You are concise.",
        generation={
            "temperature": 0.6,
            "top_p": 0.95,
            "seed": 7,
            "max_tokens": 4096,
        },
    )
    preflight = SimpleNamespace(
        ready=True,
        model_path=tmp_path / "checkpoint_folder_textonly",
        benchmark=benchmark,
    )
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        assert kwargs["timeout"] == 1800
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"enabled": true, "subprocess": true, "ready": true}',
            stderr="",
        )

    monkeypatch.setattr(vllm_diagnostics.subprocess, "run", fake_run)

    result = _probe_vllm_generation(preflight, max_tokens=12)

    assert result["ready"] is True
    assert result["subprocess"] is True
    assert result["returncode"] == 0
    assert calls[0][0:3] == [vllm_diagnostics.sys.executable, "-c", calls[0][2]]
    assert calls[0][-6:] == [
        str(preflight.model_path),
        "0.6",
        "0.95",
        "7",
        "12",
        "You are concise.",
    ]


def test_sts_default_runtime_probe_runs_fixture_turn_in_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []
    fixture = tmp_path / "input.wav"

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs["env"]))
        assert kwargs["timeout"] == 600
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=(
                '{"enabled": true, "subprocess": true, "ready": true, '
                '"engine_class": "vllm.AsyncLLMEngine", '
                '"speech_streaming": {"vllm_token_streaming": true, '
                '"decoder_streaming": true, '
                '"first_audio_ready_seconds": 0.5, '
                '"generated_token_count": 4, '
                '"generated_codec_frame_count": 3, '
                '"chunk_count": 1}}'
            ),
            stderr=(
                "Audex vLLM Metal: native MLX sampling fast path used 50 time(s)\n"
                "Audex vLLM Metal: paged sample timing "
                "count=50 avg_ms=98.5 last_ms=14.2 "
                "decode_reqs=1 prefill_reqs=0 decode_tokens=1 "
                "native_sample_ms=123.4 "
                "mx_eval_ms=logits:456.7/50,sample_tokens:12.3/50"
            ),
        )

    monkeypatch.setattr(vllm_diagnostics.subprocess, "run", fake_run)

    result = _probe_vllm_sts_default_runtime(
        SimpleNamespace(repo_id="nvidia/Nemotron-Labs-Audex-2B"),
        audio_fixture=fixture,
        play_audio=True,
        native_sampling_debug=True,
    )

    assert result["ready"] is True
    assert result["returncode"] == 0
    assert result["engine_class"] == "vllm.AsyncLLMEngine"
    assert result["vllm_metal_timing"]["latest_paged_sample"]["count"] == 50
    assert (
        result["vllm_metal_timing"]["latest_paged_sample"]["mx_eval_ms"]["logits"][
            "milliseconds"
        ]
        == 456.7
    )
    assert result["native_sampling_debug"] is True
    assert calls[0][1]["AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"] == "1"
    args = calls[0][0]
    assert args[0:3] == [vllm_diagnostics.sys.executable, "-c", args[2]]
    assert args[-4:] == [
        "nvidia/Nemotron-Labs-Audex-2B",
        str(fixture),
        "1",
        "256",
    ]


def test_sts_default_runtime_probe_preserves_success_json_on_pipe_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(args, **kwargs):
        assert "AUDEX_VLLM_NATIVE_SAMPLING_DEBUG" not in kwargs["env"]
        raise subprocess.TimeoutExpired(
            cmd=args,
            timeout=300,
            output=(
                "Audex STS: transcript: oh\n"
                '{"enabled": true, "subprocess": true, "ready": true, '
                '"elapsed_seconds": 28.786, '
                '"speech_streaming": {"vllm_token_streaming": true, '
                '"decoder_streaming": true, '
                '"first_audio_ready_seconds": 1.344, '
                '"generated_token_count": 57, '
                '"generated_codec_frame_count": 56, '
                '"last_codec_frame_seconds": 1.344, '
                '"chunk_count": 12, '
                '"tts_segment_codec_frame_counts": '
                '{"0": 20, "1": 24, "2": 12}}}'
            ),
        )

    monkeypatch.setattr(vllm_diagnostics.subprocess, "run", fake_run)

    result = _probe_vllm_sts_default_runtime(
        SimpleNamespace(repo_id="nvidia/Nemotron-Labs-Audex-2B"),
        audio_fixture=None,
        play_audio=False,
    )

    assert result["ready"] is True
    assert result["elapsed_seconds"] == 28.786
    assert result["sts_timing_assessment"]["codec_frames_per_second"] == 41.667
    assert result["sts_timing_assessment"]["below_realtime"] is True
    assert result["sts_timing_assessment"]["tts_segment_count"] == 3
    assert result["sts_timing_assessment"]["tts_segment_codec_frame_min"] == 12
    assert result["sts_timing_assessment"]["tts_segment_codec_frame_max"] == 24
    assert result["sts_timing_assessment"]["tts_tail_to_mean_ratio"] == 0.643
    assert result["timed_out_after_seconds"] == 300
    assert result["subprocess_timeout_after_result"] is True
    assert result["native_sampling_debug"] is False
    assert "error" not in result
    assert result["stdout"] == "Audex STS: transcript: oh"


def test_sts_default_runtime_probe_can_bound_silent_speech_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs["env"]))
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=(
                '{"enabled": true, "subprocess": true, "ready": true, '
                '"engine_class": "vllm.AsyncLLMEngine", '
                '"speech_max_tokens": 128, '
                '"speech_streaming": {"vllm_token_streaming": true, '
                '"decoder_streaming": true, '
                '"first_audio_ready_seconds": 0.5, '
                '"generated_token_count": 4, '
                '"generated_codec_frame_count": 3, '
                '"last_codec_frame_seconds": 0.5, '
                '"chunk_count": 1}}'
            ),
            stderr="",
        )

    monkeypatch.setenv("AUDEX_STS_SMOKE_SPEECH_MAX_TOKENS", "128")
    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")
    monkeypatch.setattr(vllm_diagnostics.subprocess, "run", fake_run)

    result = _probe_vllm_sts_default_runtime(
        SimpleNamespace(repo_id="nvidia/Nemotron-Labs-Audex-2B"),
        audio_fixture=None,
        play_audio=False,
    )

    assert result["ready"] is True
    assert result["speech_max_tokens"] == 128
    assert result["sts_timing_assessment"]["codec_frames_per_second"] == 6.0
    assert result["native_sampling_debug"] is False
    assert "AUDEX_VLLM_NATIVE_SAMPLING_DEBUG" not in calls[0][1]
    assert calls[0][0][-2:] == ["0", "128"]


def test_assess_sts_timing_reports_cfg_segment_tail_imbalance() -> None:
    assessment = _assess_sts_timing(
        {
            "speech_streaming": {
                "generated_codec_frame_count": 1300,
                "last_codec_frame_seconds": 56.066,
                "tts_segment_codec_frame_counts": {
                    "0": 165,
                    "1": 252,
                    "2": 181,
                    "3": 169,
                    "4": 205,
                    "5": 136,
                    "6": 128,
                    "7": 64,
                },
            }
        }
    )

    assert assessment["codec_frames_per_second"] == 23.187
    assert assessment["tts_segment_count"] == 8
    assert assessment["tts_segment_codec_frame_min"] == 64
    assert assessment["tts_segment_codec_frame_max"] == 252
    assert assessment["tts_segment_codec_frame_mean"] == 162.5
    assert assessment["tts_segment_codec_frame_max_to_min_ratio"] == 3.938
    assert assessment["tts_tail_codec_frames"] == 64
    assert assessment["tts_tail_to_mean_ratio"] == 0.394
    assert assessment["tts_tail_underfilled"] is True


def test_tts_batch_runtime_probe_runs_parallel_cfg_pairs_in_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs["env"]))
        assert kwargs["timeout"] == 600
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=(
                '{"enabled": true, "subprocess": true, "ready": true, '
                '"batch_size": 4, "cfg_enabled": true, "request_count": 8, '
                '"total_codec_frame_count": 512, '
                '"min_codec_frames_per_request": 128, '
                '"max_codec_frames_per_request": 128, '
                '"hit_max_token_count": 4, '
                '"reached_end_count": 0, '
                '"codec_frames_per_second": 32.0}'
            ),
            stderr=(
                "Audex vLLM Metal: paged sample timing "
                "count=50 avg_ms=40.0 last_ms=25.0 "
                "decode_reqs=8 prefill_reqs=0 decode_tokens=8 "
                "native_sample_ms=1000.0 native_sampled_rows=200 "
                "native_output_rows=400 native_detail_ms=sample_eval:900.0/50 "
                "mx_eval_ms=logits:2000.0/50"
            ),
        )

    monkeypatch.setattr(vllm_diagnostics.subprocess, "run", fake_run)

    result = _probe_vllm_tts_batch_runtime(
        SimpleNamespace(repo_id="nvidia/Nemotron-Labs-Audex-2B"),
        batch_size=4,
        max_tokens=128,
        text="hello",
        use_cfg=True,
        native_sampling_debug=True,
    )

    assert result["ready"] is True
    assert result["cfg_enabled"] is True
    assert result["codec_frames_per_second"] == 32.0
    assert result["min_codec_frames_per_request"] == 128
    assert result["hit_max_token_count"] == 4
    assert result["native_sampling_debug"] is True
    assert result["vllm_metal_timing"]["latest_paged_sample"]["decode_reqs"] == 8
    assert (
        result["vllm_metal_timing"]["latest_paged_sample"]["native_detail_ms"][
            "sample_eval"
        ]["milliseconds"]
        == 900.0
    )
    assert calls[0][1]["AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"] == "1"
    assert calls[0][1]["AUDEX_VLLM_ENABLE_CFG_WIRING"] == "1"
    args = calls[0][0]
    assert args[0:3] == [vllm_diagnostics.sys.executable, "-c", args[2]]
    compile(args[2], "<tts-batch-probe>", "exec")
    assert args[-5:] == [
        "nvidia/Nemotron-Labs-Audex-2B",
        "4",
        "128",
        "hello",
        "1",
    ]


def test_tts_batch_runtime_probe_clears_native_sampling_debug_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_env: dict[str, str] = {}

    def fake_run(args, **kwargs):
        captured_env.update(kwargs["env"])
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"enabled": true, "subprocess": true, "ready": true}',
            stderr="",
        )

    monkeypatch.setenv("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG", "1")
    monkeypatch.setattr(vllm_diagnostics.subprocess, "run", fake_run)

    result = _probe_vllm_tts_batch_runtime(
        SimpleNamespace(repo_id="nvidia/Nemotron-Labs-Audex-2B"),
        batch_size=2,
        max_tokens=64,
        text="hello",
        use_cfg=False,
    )

    assert result["ready"] is True
    assert result["native_sampling_debug"] is False
    assert "AUDEX_VLLM_NATIVE_SAMPLING_DEBUG" not in captured_env
    assert "AUDEX_VLLM_ENABLE_CFG_WIRING" not in captured_env


def test_audex_processor_probe_runs_in_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        assert kwargs["timeout"] == 25
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=(
                '{"ready": true, "output": {'
                '"type": "multimodal", '
                '"placeholder_offset": 1, '
                '"placeholder_length": 3'
                "}}"
            ),
            stderr="",
        )

    monkeypatch.setattr(vllm_diagnostics.subprocess, "run", fake_run)

    result = _probe_audex_processor_subprocess()

    assert result["ready"] is True
    assert result["returncode"] == 0
    assert result["output"]["placeholder_length"] == 3
    assert calls[0][0:2] == [vllm_diagnostics.sys.executable, "-c"]


def _ready_processor_probe() -> dict:
    return {
        "ready": True,
        "output": {
            "placeholder_offset": 1,
            "placeholder_length": 3,
            "placeholder_embeds": 3,
        },
    }


def _ready_cfg_probe() -> dict:
    return {
        "enabled": True,
        "ready": True,
        "logits_processors": [
            "cfg_logits_processor.CFGLogitsProcessor",
            ("audex_mac.patches.vllm_metal_cfg." "AudexMetalCFGTokenSyncInstaller"),
        ],
        "enable_prefix_caching": False,
        "vllm_metal_patch": {
            "ready": True,
            "sample_from_logits": True,
            "sample_prefill_tokens": True,
            "model_runner_symbols": True,
            "error": None,
        },
    }


def test_audex_cfg_probe_runs_in_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        assert kwargs["timeout"] == 25
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=(
                '{"enabled": true, "ready": true, '
                '"logits_processors": ["cfg_logits_processor.CFGLogitsProcessor"], '
                '"enable_prefix_caching": false, '
                '"vllm_metal_patch": {"ready": true}}'
            ),
            stderr="",
        )

    monkeypatch.setattr(vllm_diagnostics.subprocess, "run", fake_run)

    result = _probe_audex_cfg_subprocess(tmp_path / "checkpoint_folder_full")

    assert result["ready"] is True
    assert result["returncode"] == 0
    assert calls[0][0:2] == [vllm_diagnostics.sys.executable, "-c"]


def test_diagnostic_verdict_fails_on_cpu_platform_despite_cpu_facade() -> None:
    report = {
        "parent_process": {"metal_policy": {"ready": True}},
        "spawn_probe": {
            "returncode": 0,
            "mlx": {
                "default_device": "Device(gpu, 0)",
                "probe_array_device": "Device(gpu, 0)",
            },
        },
        "platform_resolution_probe": {
            "direct_vllm_metal_register": "vllm_metal.platform.MetalPlatform",
            "current_platform": {
                "class": "vllm.platforms.cpu.CpuPlatform",
            },
        },
        "vllm_metal": {
            "config": {
                "use_mlx": True,
                "mlx_device": "gpu",
                "use_paged_attention": False,
            },
        },
        "audex_patches": {
            "transformers_local_dynamic_modules": True,
            "mlx_lm_nemotron_dense": True,
            "mlx_lm_nemotron_h_audex": True,
            "vllm_metal_platform_repair": True,
            "vllm_nemotron_dense": True,
            "vllm_metal_audex_adapter": True,
        },
        "model_adapter": {
            "audex_patch_installed": True,
            "audex_adapter_selected": True,
        },
        "audex_processor": _ready_processor_probe(),
        "audex_cfg": _ready_cfg_probe(),
        "generation_probe": {"enabled": False},
    }

    verdict = _diagnostic_verdict(report)

    assert verdict["ready"] is False
    assert "vLLM effective current_platform" in verdict["failures"][0]


def test_diagnostic_verdict_accepts_repaired_metal_platform() -> None:
    report = {
        "parent_process": {"metal_policy": {"ready": True}},
        "spawn_probe": {
            "returncode": 0,
            "mlx": {
                "default_device": "Device(gpu, 0)",
                "probe_array_device": "Device(gpu, 0)",
            },
        },
        "platform_resolution_probe": {
            "direct_vllm_metal_register": "vllm_metal.platform.MetalPlatform",
            "current_platform": {
                "class": "vllm.platforms.cpu.CpuPlatform",
            },
            "current_platform_after_audex_patches": {
                "class": "vllm_metal.platform.MetalPlatform",
            },
        },
        "vllm_metal": {
            "config": {
                "use_mlx": True,
                "mlx_device": "gpu",
                "use_paged_attention": False,
            },
        },
        "audex_patches": {
            "transformers_local_dynamic_modules": True,
            "mlx_lm_nemotron_dense": True,
            "mlx_lm_nemotron_h_audex": True,
            "vllm_metal_platform_repair": True,
            "vllm_nemotron_dense": True,
            "vllm_metal_audex_adapter": True,
        },
        "model_adapter": {
            "audex_patch_installed": True,
            "audex_adapter_selected": True,
        },
        "audex_processor": _ready_processor_probe(),
        "audex_cfg": _ready_cfg_probe(),
        "generation_probe": {"enabled": False},
    }

    verdict = _diagnostic_verdict(report)

    assert verdict == {"ready": True, "failures": []}


def test_diagnostic_verdict_reports_vllm_metal_config_error() -> None:
    report = {
        "parent_process": {"metal_policy": {"ready": True}},
        "spawn_probe": {
            "returncode": 0,
            "mlx": {
                "default_device": "Device(gpu, 0)",
                "probe_array_device": "Device(gpu, 0)",
            },
        },
        "platform_resolution_probe": {
            "direct_vllm_metal_register": "vllm_metal.platform.MetalPlatform",
            "current_platform_after_audex_patches": {
                "class": "vllm_metal.platform.MetalPlatform",
            },
        },
        "vllm_metal": {
            "platform": {
                "error": (
                    "ValueError: VLLM_METAL_MEMORY_FRACTION=0.85 "
                    "requires VLLM_METAL_MEMORY_FRACTION=auto."
                ),
            },
            "config": {},
        },
        "audex_patches": {
            "transformers_local_dynamic_modules": True,
            "mlx_lm_nemotron_dense": True,
            "mlx_lm_nemotron_h_audex": True,
            "vllm_metal_platform_repair": True,
            "vllm_nemotron_dense": True,
            "vllm_metal_audex_adapter": True,
        },
        "model_adapter": {
            "audex_patch_installed": True,
            "audex_adapter_selected": True,
        },
        "audex_processor": _ready_processor_probe(),
        "audex_cfg": _ready_cfg_probe(),
        "generation_probe": {"enabled": False},
    }

    verdict = _diagnostic_verdict(report)

    assert verdict["ready"] is False
    assert verdict["failures"] == [
        (
            "vLLM Metal config error: ValueError: "
            "VLLM_METAL_MEMORY_FRACTION=0.85 requires "
            "VLLM_METAL_MEMORY_FRACTION=auto."
        )
    ]


def test_diagnostic_verdict_rejects_missing_cfg_sampler_patch() -> None:
    report = {
        "parent_process": {"metal_policy": {"ready": True}},
        "spawn_probe": {
            "returncode": 0,
            "mlx": {
                "default_device": "Device(gpu, 0)",
                "probe_array_device": "Device(gpu, 0)",
            },
        },
        "platform_resolution_probe": {
            "direct_vllm_metal_register": "vllm_metal.platform.MetalPlatform",
            "current_platform_after_audex_patches": {
                "class": "vllm_metal.platform.MetalPlatform",
            },
        },
        "vllm_metal": {
            "config": {
                "use_mlx": True,
                "mlx_device": "gpu",
                "use_paged_attention": False,
            },
        },
        "audex_patches": {
            "transformers_local_dynamic_modules": True,
            "mlx_lm_nemotron_dense": True,
            "mlx_lm_nemotron_h_audex": True,
            "vllm_metal_platform_repair": True,
            "vllm_nemotron_dense": True,
            "vllm_metal_audex_adapter": True,
        },
        "model_adapter": {
            "audex_patch_installed": True,
            "audex_adapter_selected": True,
        },
        "audex_processor": _ready_processor_probe(),
        "audex_cfg": {
            **_ready_cfg_probe(),
            "vllm_metal_patch": {"ready": False, "error": "missing vllm_metal"},
        },
        "generation_probe": {"enabled": False},
    }

    verdict = _diagnostic_verdict(report)

    assert verdict["ready"] is False
    assert "CFG sampler patch" in verdict["failures"][0]


def test_diagnostic_verdict_does_not_duplicate_missing_metal_cfg_patch_failure() -> (
    None
):
    report = {
        "parent_process": {"metal_policy": {"ready": True}},
        "spawn_probe": {
            "returncode": 0,
            "mlx_error": "RuntimeError: [metal::load_device] No Metal device available.",
            "mlx": {},
        },
        "platform_resolution_probe": {
            "direct_vllm_metal_register": "vllm_metal.platform.MetalPlatform",
            "current_platform_after_audex_patches": {
                "class": "vllm_metal.platform.MetalPlatform",
            },
        },
        "vllm_metal": {
            "config": {
                "use_mlx": True,
                "mlx_device": "gpu",
                "use_paged_attention": False,
            },
        },
        "audex_patches": {
            "transformers_local_dynamic_modules": True,
            "mlx_lm_nemotron_dense": True,
            "mlx_lm_nemotron_h_audex": True,
            "vllm_metal_platform_repair": True,
            "vllm_nemotron_dense": True,
            "vllm_metal_audex_adapter": True,
        },
        "model_adapter": {
            "audex_patch_installed": True,
            "audex_adapter_selected": True,
        },
        "audex_processor": _ready_processor_probe(),
        "audex_cfg": {
            **_ready_cfg_probe(),
            "vllm_metal_patch": {
                "ready": False,
                "error": (
                    "RuntimeError: [metal::load_device] " "No Metal device available."
                ),
            },
        },
        "generation_probe": {"enabled": False},
    }

    verdict = _diagnostic_verdict(report)

    assert verdict["ready"] is False
    assert verdict["failures"] == [
        (
            "spawn probe MLX error: RuntimeError: [metal::load_device] "
            "No Metal device available."
        )
    ]


def test_diagnostic_verdict_can_require_generation_probe() -> None:
    report = {
        "parent_process": {"metal_policy": {"ready": True}},
        "spawn_probe": {
            "returncode": 0,
            "mlx": {"default_device": "Device(gpu, 0)"},
        },
        "platform_resolution_probe": {
            "direct_vllm_metal_register": "vllm_metal.platform.MetalPlatform",
            "current_platform": {
                "class": "vllm_metal.platform.MetalPlatform",
            },
        },
        "vllm_metal": {
            "config": {
                "use_mlx": True,
                "mlx_device": "gpu",
                "use_paged_attention": False,
            },
        },
        "audex_patches": {
            "transformers_local_dynamic_modules": True,
            "mlx_lm_nemotron_dense": True,
            "mlx_lm_nemotron_h_audex": True,
            "vllm_metal_platform_repair": True,
            "vllm_nemotron_dense": True,
            "vllm_metal_audex_adapter": True,
        },
        "model_adapter": {
            "audex_patch_installed": True,
            "audex_adapter_selected": True,
        },
        "audex_processor": _ready_processor_probe(),
        "audex_cfg": _ready_cfg_probe(),
        "generation_probe": {
            "enabled": True,
            "ready": False,
            "error": "boom",
        },
    }

    verdict = _diagnostic_verdict(report, require_generation=True)

    assert verdict["ready"] is False
    assert "generation probe" in verdict["failures"][0]


def test_diagnostic_verdict_can_require_sts_probe() -> None:
    report = {
        "parent_process": {"metal_policy": {"ready": True}},
        "spawn_probe": {
            "returncode": 0,
            "mlx": {"default_device": "Device(gpu, 0)"},
        },
        "platform_resolution_probe": {
            "direct_vllm_metal_register": "vllm_metal.platform.MetalPlatform",
            "current_platform": {
                "class": "vllm_metal.platform.MetalPlatform",
            },
        },
        "vllm_metal": {
            "config": {
                "use_mlx": True,
                "mlx_device": "gpu",
                "use_paged_attention": False,
            },
        },
        "audex_patches": {
            "transformers_local_dynamic_modules": True,
            "mlx_lm_nemotron_dense": True,
            "mlx_lm_nemotron_h_audex": True,
            "vllm_metal_platform_repair": True,
            "vllm_nemotron_dense": True,
            "vllm_metal_audex_adapter": True,
        },
        "model_adapter": {
            "audex_patch_installed": True,
            "audex_adapter_selected": True,
        },
        "audex_processor": _ready_processor_probe(),
        "audex_cfg": _ready_cfg_probe(),
        "generation_probe": {"enabled": False},
        "sts_probe": {
            "enabled": True,
            "ready": False,
            "error": "boom",
        },
    }

    verdict = _diagnostic_verdict(report, require_sts=True)

    assert verdict["ready"] is False
    assert "STS smoke probe" in verdict["failures"][0]


def test_diagnostic_verdict_can_require_tts_batch_probe() -> None:
    report = {
        "parent_process": {"metal_policy": {"ready": True}},
        "spawn_probe": {
            "returncode": 0,
            "mlx": {"default_device": "Device(gpu, 0)"},
        },
        "platform_resolution_probe": {
            "direct_vllm_metal_register": "vllm_metal.platform.MetalPlatform",
            "current_platform": {
                "class": "vllm_metal.platform.MetalPlatform",
            },
        },
        "vllm_metal": {
            "config": {
                "use_mlx": True,
                "mlx_device": "gpu",
                "use_paged_attention": False,
            },
        },
        "audex_patches": {
            "transformers_local_dynamic_modules": True,
            "mlx_lm_nemotron_dense": True,
            "mlx_lm_nemotron_h_audex": True,
            "vllm_metal_platform_repair": True,
            "vllm_nemotron_dense": True,
            "vllm_metal_audex_adapter": True,
        },
        "model_adapter": {
            "audex_patch_installed": True,
            "audex_adapter_selected": True,
        },
        "audex_processor": _ready_processor_probe(),
        "audex_cfg": _ready_cfg_probe(),
        "generation_probe": {"enabled": False},
        "tts_batch_probe": {
            "enabled": True,
            "ready": False,
            "error": "boom",
        },
    }

    verdict = _diagnostic_verdict(report, require_tts_batch=True)

    assert verdict["ready"] is False
    assert "TTS batch probe" in verdict["failures"][0]


def test_sts_smoke_evidence_failures_accept_complete_async_streaming_report() -> None:
    failures = _sts_smoke_evidence_failures(
        {
            "engine_class": "vllm.engine.async_llm_engine.AsyncLLMEngine",
            "sts_timing_assessment": {
                "codec_frames_per_second": 52.0,
                "audio_realtime_ratio": 1.04,
                "below_realtime": False,
                "native_sampling_row_ratio": 0.5,
            },
            "speech_streaming": {
                "vllm_token_streaming": True,
                "decoder_streaming": True,
                "playback_transport": None,
                "first_audio_ready_seconds": 0.75,
                "first_playback_started_seconds": None,
                "generated_token_count": 12,
                "generated_codec_frame_count": 10,
                "chunk_count": 2,
            },
        }
    )

    assert failures == []


def test_sts_smoke_evidence_failures_reject_below_realtime_throughput() -> None:
    failures = _sts_smoke_evidence_failures(
        {
            "engine_class": "vllm.engine.async_llm_engine.AsyncLLMEngine",
            "sts_timing_assessment": {
                "codec_frames_per_second": 3.278,
                "audio_realtime_ratio": 0.066,
                "below_realtime": True,
                "native_sampling_row_ratio": 0.5,
            },
            "speech_streaming": {
                "vllm_token_streaming": True,
                "decoder_streaming": True,
                "playback_transport": None,
                "first_audio_ready_seconds": 0.75,
                "first_playback_started_seconds": None,
                "generated_token_count": 12,
                "generated_codec_frame_count": 10,
                "chunk_count": 2,
            },
        }
    )

    assert failures == [
        (
            "vLLM default STS smoke speech-token throughput is below realtime: "
            "codec_fps=3.278 realtime_ratio=0.066."
        )
    ]


def test_sts_smoke_evidence_failures_reject_duplicate_cfg_row_sampling() -> None:
    failures = _sts_smoke_evidence_failures(
        {
            "engine_class": "vllm.engine.async_llm_engine.AsyncLLMEngine",
            "sts_timing_assessment": {
                "codec_frames_per_second": 52.0,
                "audio_realtime_ratio": 1.04,
                "below_realtime": False,
                "native_sampling_row_ratio": 1.0,
            },
            "speech_streaming": {
                "vllm_token_streaming": True,
                "decoder_streaming": True,
                "playback_transport": None,
                "first_audio_ready_seconds": 0.75,
                "first_playback_started_seconds": None,
                "generated_token_count": 12,
                "generated_codec_frame_count": 10,
                "chunk_count": 2,
            },
        }
    )

    assert failures == [
        (
            "vLLM default STS smoke native CFG sampling row ratio is too high: "
            "1.0. Expected about 0.5 for paired CFG sampling."
        )
    ]


def test_sts_smoke_evidence_failures_reject_too_short_response() -> None:
    failures = _sts_smoke_evidence_failures(
        {
            "engine_class": "vllm.engine.async_llm_engine.AsyncLLMEngine",
            "response_prefix": "A context manager is an",
            "response_word_count": 5,
            "min_response_words": 8,
            "valid_response_length": False,
            "sts_timing_assessment": {
                "codec_frames_per_second": 52.0,
                "audio_realtime_ratio": 1.04,
                "below_realtime": False,
                "native_sampling_row_ratio": 0.5,
            },
            "speech_streaming": {
                "vllm_token_streaming": True,
                "decoder_streaming": True,
                "playback_transport": None,
                "first_audio_ready_seconds": 0.75,
                "first_playback_started_seconds": None,
                "generated_token_count": 12,
                "generated_codec_frame_count": 10,
                "chunk_count": 2,
            },
        }
    )

    assert failures == [
        (
            "vLLM default STS smoke response is too short to trust timing "
            "evidence: 5 words < 8."
        )
    ]


def test_sts_smoke_evidence_failures_reject_underfilled_tts_tail() -> None:
    failures = _sts_smoke_evidence_failures(
        {
            "engine_class": "vllm.engine.async_llm_engine.AsyncLLMEngine",
            "valid_response_length": True,
            "sts_timing_assessment": {
                "codec_frames_per_second": 52.0,
                "audio_realtime_ratio": 1.04,
                "below_realtime": False,
                "native_sampling_row_ratio": 0.5,
                "tts_segment_count": 8,
                "tts_tail_codec_frames": 64,
                "tts_tail_to_mean_ratio": 0.394,
                "tts_tail_underfilled": True,
            },
            "speech_streaming": {
                "vllm_token_streaming": True,
                "decoder_streaming": True,
                "playback_transport": None,
                "first_audio_ready_seconds": 0.75,
                "first_playback_started_seconds": None,
                "generated_token_count": 12,
                "generated_codec_frame_count": 10,
                "chunk_count": 2,
            },
        }
    )

    assert failures == [
        (
            "vLLM default STS smoke CFG chunk planner left an underfilled final "
            "segment: 64 codec frames, tail/mean ratio 0.394, 8 segments."
        )
    ]


def test_sts_smoke_evidence_failures_require_playback_when_requested() -> None:
    failures = _sts_smoke_evidence_failures(
        {
            "engine_class": "vllm.engine.async_llm_engine.AsyncLLMEngine",
            "play_audio": True,
            "sts_timing_assessment": {
                "codec_frames_per_second": 52.0,
                "audio_realtime_ratio": 1.04,
                "below_realtime": False,
            },
            "speech_streaming": {
                "vllm_token_streaming": True,
                "decoder_streaming": True,
                "playback_transport": None,
                "first_audio_ready_seconds": 0.75,
                "first_playback_started_seconds": None,
                "generated_token_count": 12,
                "generated_codec_frame_count": 10,
                "chunk_count": 2,
            },
        }
    )

    assert failures == [
        "vLLM default STS smoke did not use continuous PCM playback",
        "vLLM default STS smoke did not record first-playback timing",
        "vLLM default STS smoke did not record playback diagnostics",
    ]


def test_sts_smoke_evidence_failures_accept_playback_diagnostics() -> None:
    failures = _sts_smoke_evidence_failures(
        {
            "engine_class": "vllm.engine.async_llm_engine.AsyncLLMEngine",
            "play_audio": True,
            "sts_timing_assessment": {
                "codec_frames_per_second": 52.0,
                "audio_realtime_ratio": 1.04,
                "below_realtime": False,
            },
            "speech_streaming": {
                "vllm_token_streaming": True,
                "decoder_streaming": True,
                "playback_transport": "sounddevice_raw_output_stream",
                "first_audio_ready_seconds": 0.75,
                "first_playback_started_seconds": 0.9,
                "generated_token_count": 12,
                "generated_codec_frame_count": 10,
                "chunk_count": 2,
                "playback_diagnostics": {
                    "device_underflow_count": 0,
                    "queue_underrun_count": 0,
                    "queue_overrun_count": 0,
                    "chunks_written": 2,
                },
            },
        }
    )

    assert failures == []


def test_sts_smoke_evidence_failures_reject_missing_throughput_measurement() -> None:
    failures = _sts_smoke_evidence_failures(
        {
            "engine_class": "vllm.engine.async_llm_engine.AsyncLLMEngine",
            "speech_streaming": {
                "vllm_token_streaming": True,
                "decoder_streaming": True,
                "playback_transport": None,
                "first_audio_ready_seconds": 0.75,
                "first_playback_started_seconds": None,
                "generated_token_count": 12,
                "generated_codec_frame_count": 10,
                "chunk_count": 2,
            },
        }
    )

    assert failures == [
        "vLLM default STS smoke did not measure speech-token throughput"
    ]


def test_sts_smoke_evidence_failures_reject_sync_or_empty_report() -> None:
    failures = _sts_smoke_evidence_failures(
        {
            "engine_class": "vllm.entrypoints.llm.LLM",
            "speech_streaming": {
                "vllm_token_streaming": True,
                "decoder_streaming": True,
                "first_audio_ready_seconds": None,
                "generated_token_count": 0,
                "generated_codec_frame_count": 0,
                "chunk_count": 0,
            },
        }
    )

    assert failures == [
        "vLLM default STS smoke did not use an async vLLM engine class",
        "vLLM default STS smoke did not record first-audio timing",
        "vLLM default STS smoke generated no speech tokens",
        "vLLM default STS smoke generated no codec frames",
        "vLLM default STS smoke wrote no decoder chunks",
        "vLLM default STS smoke did not measure speech-token throughput",
    ]
